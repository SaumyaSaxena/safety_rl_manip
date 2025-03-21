# adapted from https://github.com/google-research/robotics_transformer/blob/master/film_efficientnet/film_conditioning_layer.py
import torch
import torch.nn as nn


class FilmConditioning(nn.Module):

    def __call__(self, conv_filters: torch.Tensor, conditioning: torch.Tensor):
        """Applies FiLM conditioning to a convolutional feature map.

        Args:
            conv_filters: A tensor of shape [batch_size, height, width, channels].
            conditioning: A tensor of shape [batch_size, conditioning_size].

        Returns:
            A tensor of shape [batch_size, height, width, channels].
        """
        projected_cond_add = nn.Dense(
            features=conv_filters.shape[-1],
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(conditioning)
        projected_cond_mult = nn.Dense(
            features=conv_filters.shape[-1],
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(conditioning)

        projected_cond_add = projected_cond_add[:, None, None, :]
        projected_cond_mult = projected_cond_mult[:, None, None, :]

        return conv_filters * (1 + projected_cond_add) + projected_cond_mult
