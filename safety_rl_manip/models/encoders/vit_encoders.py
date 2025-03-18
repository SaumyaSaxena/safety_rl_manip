"""
Encoders more suitable for ViT architectures.

- PatchEncoder: Just patchifies the image
- SmallStem: 3 conv layers, then patchifies the image (from xiao et al. 2021)
- ViTResnet: ResNetv2, followed by patchification (from google-research/vision_transformer)
"""

import functools as ft
from typing import Callable, Sequence, TypeVar

import torch
import torch.nn as nn
from torch.nn import functional as F

from safety_rl_manip.models.encoders.film_conditioning_layer import FilmConditioning

T = TypeVar("T")


def normalize_images(img, img_norm_type="default"):
    if img_norm_type == "default":
        # put pixels in [-1, 1]
        return img.to(torch.float32) / 127.5 - 1.0
    elif img_norm_type == "imagenet":
        # put pixels in [0,1]
        img = img.to(torch.float32) / 255
        assert img.shape[-1] % 3 == 0, "images should have rgb channels!"

        # define pixel-wise mean/std stats calculated from ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape((1, 1, 1, 3))
        std = torch.tensor([0.229, 0.224, 0.225]).reshape((1, 1, 1, 3))

        # tile mean and std (to account for stacked early_fusion images)
        num_tile = (1, 1, 1, int(img.shape[-1] / 3))
        mean_tile = torch.tile(mean, num_tile)
        std_tile = torch.tile(std, num_tile)

        # tile the mean/std, normalize image, and return
        return (img - mean_tile) / std_tile
    raise ValueError()


def weight_standardize(weight: torch.Tensor, eps: float):
    """Subtracts mean and divides by standard deviation.
    https://nn.labml.ai/normalization/weight_standardization/index.html
    """
    c_out, c_in, *kernel_shape = weight.shape
    weight = weight.view(c_out, -1)
    var, mean = torch.var_mean(weight, dim=1, keepdim=True)
    weight = (weight - mean) / (torch.sqrt(var + eps))
    return weight.view(c_out, c_in, *kernel_shape)


class StdConv(nn.Conv2d):
    """Convolution with weight standardization.
    https://nn.labml.ai/normalization/weight_standardization/conv2d.html
    """
    def __init__(self, in_channels, out_channels, kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        eps: float = 1e-5
    ):
        super(StdConv, self).__init__(in_channels, out_channels, kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return F.conv2d(x, weight_standardize(self.weight, self.eps), self.bias, self.stride, self.padding, self.dilation, self.groups)

class PatchEncoder(nn.Module):
    """Takes an image and breaks it up into patches of size (patch_size x patch_size),
    applying a fully connected network to each patch individually.

    The default "encoder" used by most ViTs in practice.
    """

    use_film: bool = False
    patch_size: int = 32
    num_features: int = 512
    img_norm_type: str = "default"

    def __call__(self, observations: torch.Tensor, train: bool = True, cond_var=None):
        expecting_cond_var = self.use_film
        received_cond_var = cond_var is not None
        assert (
            expecting_cond_var == received_cond_var
        ), "Only pass in cond var iff model expecting cond var"
        x = normalize_images(observations, self.img_norm_type)
        x = nn.Conv(
            features=self.num_features,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            name="embedding",
        )(x)
        if self.use_film:
            assert cond_var is not None, "Cond var is None, nothing to condition on"
            x = FilmConditioning()(x, cond_var)
        return x


class SmallStem(nn.Module):
    """Passes the image through a few light-weight convolutional layers,
    before patchifying the image. Empirically useful for many computer vision tasks.

    See Xiao et al: Early Convolutions Help Transformers See Better
    """

    def __init__(self,
        use_film: bool = False,
        patch_size: int = 32,
        kernel_sizes: tuple = (3, 3, 3, 3),
        strides: tuple = (2, 2, 2, 2),
        features: tuple = (32, 96, 192, 384),
        padding: tuple = (1, 1, 1, 1),
        num_features: int = 512,
        img_norm_type: str = "default",
        num_inp_channels: int = 3,
        trainable: bool = False,
        device: str = 'cuda',
    ):
        super().__init__()
        self.use_film = use_film
        self.patch_size = patch_size
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.features = features
        self.padding = padding
        self.num_features = num_features
        self.img_norm_type = img_norm_type
        self.device = device

        self.std_convs, self.group_norms = nn.ModuleList(), nn.ModuleList()
        _in_channels = num_inp_channels
        for n, (kernel_size, stride, feature, padding) in enumerate(
            zip(
                kernel_sizes,
                strides,
                features,
                padding,
            )
        ):  
            
            self.std_convs.append(
                StdConv(
                    in_channels=_in_channels,
                    out_channels=feature,
                    kernel_size=(kernel_size, kernel_size),
                    stride=(stride, stride),
                    padding=padding,
                )
            )
            self.group_norms.append(nn.GroupNorm(feature, feature))
            _in_channels = feature
            
        self.conv = nn.Conv2d(
            in_channels=features[-1],
            out_channels=num_features,
            kernel_size=(patch_size // 16, patch_size // 16),
            stride=(patch_size // 16, patch_size // 16),
            padding="valid",
        )

        if use_film:
            self.film_conditioning = FilmConditioning()

        for param in self.parameters():
            param.requires_grad_(trainable)
        
    def forward(self, observations: torch.Tensor, cond_var=None):
        if observations.shape[1] > 4: # permute to change to (b,c,h,w) format
            observations = observations.permute(0,3,1,2)
        expecting_cond_var = self.use_film
        received_cond_var = cond_var is not None
        assert (
            expecting_cond_var == received_cond_var
        ), "Only pass in cond var iff model expecting cond var"

        x = normalize_images(observations, self.img_norm_type)
        for n, (kernel_size, stride, feature, padding) in enumerate(
            zip(
                self.kernel_sizes,
                self.strides,
                self.features,
                self.padding,
            )
        ):
            x = self.std_convs[n](x)
            x = self.group_norms[n](x)
            x = F.relu(x)
        x = self.conv(x)
        if self.use_film:
            assert cond_var is not None, "Cond var is None, nothing to condition on"
            x = self.film_conditioning(x, cond_var)
        return x.permute(0,2,3,1) # permute to change back to (b,h,w,c) format.


class ResidualUnit(nn.Module):
    """Bottleneck ResNet block."""

    features: int
    strides: Sequence[int] = (1, 1)

    def __call__(self, x):
        needs_projection = x.shape[-1] != self.features * 4 or self.strides != (1, 1)

        residual = x
        if needs_projection:
            residual = StdConv(
                features=self.features * 4,
                kernel_size=(1, 1),
                strides=self.strides,
                use_bias=False,
                name="conv_proj",
            )(residual)
            residual = nn.GroupNorm(name="gn_proj")(residual)

        y = StdConv(
            features=self.features, kernel_size=(1, 1), use_bias=False, name="conv1"
        )(x)
        y = nn.GroupNorm(name="gn1")(y)
        y = nn.relu(y)
        y = StdConv(
            features=self.features,
            kernel_size=(3, 3),
            strides=self.strides,
            use_bias=False,
            name="conv2",
        )(y)
        y = nn.GroupNorm(name="gn2")(y)
        y = nn.relu(y)
        y = StdConv(
            features=self.features * 4, kernel_size=(1, 1), use_bias=False, name="conv3"
        )(y)

        y = nn.GroupNorm(name="gn3", scale_init=nn.initializers.zeros)(y)
        y = nn.relu(residual + y)
        return y


class ResNetStage(nn.Module):
    """A ResNet stage."""

    block_size: Sequence[int]
    nout: int
    first_stride: Sequence[int]

    def __call__(self, x):
        x = ResidualUnit(self.nout, strides=self.first_stride, name="unit1")(x)
        for i in range(1, self.block_size):
            x = ResidualUnit(self.nout, strides=(1, 1), name=f"unit{i + 1}")(x)
        return x


class ViTResnet(nn.Module):
    """Resnet-v2 architecture used in the original ViT paper for hybrid (Resnet+ViT) architectures

    Mostly copied from https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py

    There exist pre-trained parameters here: github.com/google-research/vision_transformer/
    """

    use_film: bool = False
    width: int = 1
    num_layers: tuple = tuple()
    img_norm_type: str = "default"

    def __call__(self, observations: torch.Tensor, train: bool = True, cond_var=None):
        expecting_cond_var = self.use_film
        received_cond_var = cond_var is not None
        assert (
            expecting_cond_var == received_cond_var
        ), "Only pass in cond var iff model expecting cond var"

        x = normalize_images(observations, self.img_norm_type)
        width = int(64 * self.width)
        x = StdConv(
            features=width,
            kernel_size=(7, 7),
            strides=(2, 2),
            use_bias=False,
            name="conv_root",
        )(x)
        x = nn.GroupNorm(name="gn_root")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        if self.num_layers:
            x = ResNetStage(
                block_size=self.num_layers[0],
                nout=width,
                first_stride=(1, 1),
                name="block1",
            )(x)
            for i, block_size in enumerate(self.num_layers[1:], 1):
                x = ResNetStage(
                    block_size=block_size,
                    nout=width * 2**i,
                    first_stride=(2, 2),
                    name=f"block{i + 1}",
                )(x)
                if self.use_film:
                    assert (
                        cond_var is not None
                    ), "Cond var is None, nothing to condition on"
                    x = FilmConditioning()(x, cond_var)
        else:
            if self.use_film:
                assert cond_var is not None, "Cond var is None, nothing to condition on"
                x = FilmConditioning()(x, cond_var)

        return x


class SmallStem16(SmallStem):
    def __init__(
            self,
            use_film: bool = False,
            patch_size: int = 16,
            kernel_sizes: tuple = (3, 3, 3, 3),
            strides: tuple = (2, 2, 2, 2),
            features: tuple = (32, 96, 192, 384),
            padding: tuple = (1, 1, 1, 1),
            num_features: int = 512,
            img_norm_type: str = "default",
            num_inp_channels: int = 3,
            trainable: bool = False,
            device: str = 'cuda,'
        ):
        super().__init__(use_film, patch_size, kernel_sizes, strides, features, padding, num_features, img_norm_type, num_inp_channels, trainable, device)


class SmallStem32(SmallStem):
    patch_size: int = 32


vit_encoder_configs = {
    "patchify-32-film": ft.partial(
        PatchEncoder,
        use_film=True,
        patch_size=32,
    ),
    "patchify-16-film": ft.partial(
        PatchEncoder,
        use_film=True,
        patch_size=16,
    ),
    "small-stem-8-film": ft.partial(
        SmallStem,
        use_film=True,
        patch_size=16,
        kernel_sizes=(3, 3, 3),
        strides=(2, 2, 2),
        features=(32, 96, 192),
        padding=(1, 1, 1),
    ),
    "small-stem-16": ft.partial(
        SmallStem,
        patch_size=16,
    ),
    "small-stem-16-film": ft.partial(
        SmallStem,
        use_film=True,
        patch_size=16,
    ),
    "small-stem-32-film": ft.partial(
        SmallStem,
        use_film=True,
        patch_size=32,
    ),
    "resnetv2-26-film": ft.partial(
        ViTResnet,
        use_film=True,
        num_layers=(2, 2, 2, 2),
    ),
    "resnetv2-50-film": ft.partial(
        ViTResnet,
        use_film=True,
        num_layers=(3, 4, 6, 3),
    ),
}
