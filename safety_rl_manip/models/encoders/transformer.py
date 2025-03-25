# adapted from https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py
from typing import Callable, Optional, Union, Any
from itertools import chain
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from safety_rl_manip.models.encoders.base import TokenGroup

def wt_init_(l, activation = "relu"):
    nn.init.orthogonal_(l.weight, gain=nn.init.calculate_gain(activation))
    if l.bias is not None:
        nn.init.constant_(l.bias, 0)
    return l

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class FF_MLP(nn.Module):
    def __init__(self, model_dim, dim_ff):
        super(FF_MLP, self).__init__()
        self.fc1 = wt_init_(nn.Linear(model_dim, dim_ff))
        self.fc2 = wt_init_(nn.Linear(dim_ff, model_dim))
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))
    

class AddPositionEmbs(nn.Module):
    """Adds learned positional embeddings to the inputs."""

    def __init__(self, window_size, dim):
        super().__init__()
        self.embed = nn.Embedding(window_size, dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embed.weight)

    def forward(self, inputs):
        """Applies the AddPositionEmbs module.

        Args:
          inputs: Inputs to the layer.

        Returns:
          Output tensor with shape `(bs, timesteps, in_dim)`.
        """
        # inputs.shape is (batch_size, seq_len, emb_dim).
        assert inputs.ndim == 3, (
            "Number of dimensions should be 3," " but it is: %d" % inputs.ndim
        )
        return inputs + self.embed


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""
    def __init__(self,
        mlp_dim: int,
        out_dim: int,
        dropout_rate: float = 0.1,
        device: str = 'cuda',
    ):
        super().__init__()

        self.dense_layer1 = nn.Linear(
            in_features=out_dim,
            out_features=mlp_dim
        ).to(device)
        self.dropout1 = nn.Dropout(dropout_rate).to(device)

        self.dense_layer2 = nn.Linear(
            in_features=mlp_dim,
            out_features=out_dim
        ).to(device)
        self.dropout2 = nn.Dropout(dropout_rate).to(device)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.dense_layer1.weight)
        if self.dense_layer1.bias is not None:
            nn.init.normal_(self.dense_layer1.bias)

        nn.init.xavier_uniform_(self.dense_layer2.weight)
        if self.dense_layer2.bias is not None:
            nn.init.normal_(self.dense_layer2.bias)
    

    def forward(self, inputs):
        """Applies Transformer MlpBlock module."""
        x = self.dense_layer1(inputs)
        x = F.gelu(x)
        x = self.dropout1(x)
        
        output = self.dense_layer2(x)
        output = self.dropout2(output)
        return output


class MAPHead(nn.Module):
    """Multihead Attention Pooling.

    From https://github.com/google-research/big_vision/blob/main/big_vision/models/vit.py
    """
    def __init__(self,
        mlp_dim: Optional[int] = None,  # Defaults to 4x input dim
        num_heads: int = 8,
        num_readouts: int = 1,
    ):
        super().__init__()
        self.num_readouts = num_readouts
        self.probe = nn.Embedding(1, num_readouts, dim) # TODO(saumya)
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=mlp_dim,
            num_heads=num_heads
        )
        self.layer_norm = nn.LayerNorm(mlp_dim)
        self.mlp_block = MlpBlock(mlp_dim=mlp_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.probe.weight)


    def forward(self, x: Union[torch.Tensor, TokenGroup], train=True):
        if isinstance(x, TokenGroup):
            x, mask = x.tokens, x.mask
        else:
            mask = None

        *batch_dims, l, d = x.shape
        x = x.reshape(-1, l, d)
        batch_size = x.shape[0]

        self.probe = torch.tile(self.probe, [batch_size, 1, 1])

        if mask is not None:
            mask = mask.reshape(-1, l)
            mask = torch.broadcast_to(
                mask[:, None, None, :], (batch_size, 1, self.num_readouts, l)
            )

        out = self.multihead_attention(self.probe, x, x, mask=mask)

        # TODO: dropout on head?
        y = self.layer_norm(out)

        out = out + self.mlp_block(y)
        out = out.reshape(*batch_dims, self.num_readouts, d)
        return out


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        attn_dropout=0.1,
        output_dropout=0.1,
        device='cuda'
    ):
        """
        Multi-head masked self-attention layer + projection (MLP layer).

        For normal self-attention (@num_heads = 1), every single input in the sequence is
        mapped to a key, query, and value embedding of size @embed_dim. For each input,
        its query vector is compared (using dot-product) with all other key vectors in the
        sequence, and softmax normalized to compute an attention over all members of the
        sequence. This is used to take a linear combination of corresponding value embeddings.

        The @num_heads argument is for multi-head attention, where the self-attention operation above
        is performed in parallel over equal size partitions of the @embed_dim, allowing for different
        portions of the embedding dimension to model different kinds of attention. The attention
        output for each head is concatenated together.

        Finally, we use a causal mask here to ensure that each output only depends on inputs that come
        before it.

        Args:
            embed_dim (int): dimension of embeddings to use for keys, queries, and values
                used in self-attention

            num_heads (int): number of attention heads - must divide @embed_dim evenly. Self-attention is
                computed over this many partitions of the embedding dimension separately.

            attn_dropout (float): dropout probability for attention outputs

            output_dropout (float): dropout probability for final outputs
        """
        super(CausalSelfAttention, self).__init__()

        assert (
            embed_dim % num_heads == 0
        ), "num_heads: {} does not divide embed_dim: {} exactly".format(num_heads, embed_dim)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.output_dropout = output_dropout
        self.device = device
        self.nets = nn.ModuleDict()

        # projection layers for key, query, value, across all attention heads
        self.nets["qkv"] = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)

        # dropout layers
        self.nets["attn_dropout"] = nn.Dropout(self.attn_dropout)
        self.nets["output_dropout"] = nn.Dropout(self.output_dropout)

        # output layer
        self.nets["output"] = nn.Linear(self.embed_dim, self.embed_dim)

        # causal mask (ensures attention is only over previous inputs) - just a lower triangular matrix of 1s
        # mask = torch.tril(torch.ones(context_length, context_length)).view(
        #     1, 1, context_length, context_length
        # )
        # self.register_buffer("mask", mask)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.nets["qkv"].weight)
        if self.nets["qkv"].bias is not None:
            nn.init.normal_(self.nets["qkv"].bias)

        nn.init.xavier_uniform_(self.nets["output"].weight)
        if self.nets["output"].bias is not None:
            nn.init.normal_(self.nets["output"].bias)

    def forward(self, x):
        """
        Forward pass through Self-Attention block.
        Input should be shape (B, T, D) where B is batch size, T is seq length (@self.context_length), and
        D is input dimension (@self.embed_dim).
        """

        # enforce shape consistency
        assert len(x.shape) == 3
        B, T, D = x.shape

        assert D == self.embed_dim
        NH = self.num_heads  # number of attention heads
        DH = D // NH  # embed dimension for each attention head

        # compute key, query, and value vectors for each member of sequence, and split across attention heads
        qkv = self.nets["qkv"](x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        k = k.view(B, T, NH, DH).transpose(1, 2)  # [B, NH, T, DH]
        q = q.view(B, T, NH, DH).transpose(1, 2)  # [B, NH, T, DH]
        v = v.view(B, T, NH, DH).transpose(1, 2)  # [B, NH, T, DH]

        # causal self-attention mechanism

        # batched matrix multiplication between queries and keys to get all pair-wise dot-products.
        # We broadcast across batch and attention heads and get pair-wise dot-products between all pairs of timesteps
        # [B, NH, T, DH] x [B, NH, DH, T] -> [B, NH, T, T]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # use mask to replace entries in dot products with negative inf to ensure they don't contribute to softmax,
        # then take softmax over last dimension to end up with attention score for each member of sequence.
        # Note the use of [:T, :T] -  this makes it so we can handle sequences less than @self.context_length in length.

        # mask = torch.ones(T, T)
        # mask[:, -2:] = 0
        # mask[-2:, -2:] = torch.eye(2)
        # mask = mask.view(1,1,T,T).to(self.device)
        # att = att.masked_fill(mask[..., :T, :T] == 0, float("-inf"))

        att = F.softmax(
            att, dim=-1
        )  # shape [B, NH, T, T], last dimension has score over all T for each sequence member

        # dropout on attention
        # att = self.nets["attn_dropout"](att)

        # take weighted sum of value vectors over whole sequence according to attention, with batched matrix multiplication
        # [B, NH, T, T] x [B, NH, T, DH] -> [B, NH, T, DH]
        y = att @ v
        # reshape [B, NH, T, DH] -> [B, T, NH, DH] -> [B, T, NH * DH] = [B, T, D]
        y = y.transpose(1, 2).contiguous().view(B, T, D)

        # pass through output layer + dropout
        y = self.nets["output"](y)
        y = self.nets["output_dropout"](y)

        return y

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # this module doesn't modify the size of the input, it goes from (B, T, D) -> (B, T, D)
        return list(input_shape)


class CrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        attn_dropout=0.1,
        output_dropout=0.1,
        device='cuda'
    ):
        super(CrossAttention, self).__init__()

        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.output_dropout = output_dropout
        self.device = device
        self.nets = nn.ModuleDict()

        # projection layers for key, query, value, across all attention heads
        self.nets["kv"] = nn.Linear(self.embed_dim, 2 * self.embed_dim, bias=False)
        self.nets["q"] = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # dropout layers
        self.nets["attn_dropout"] = nn.Dropout(self.attn_dropout)
        self.nets["output_dropout"] = nn.Dropout(self.output_dropout)

        # output layer
        self.nets["output"] = nn.Linear(self.embed_dim, self.embed_dim)

        # causal mask (ensures attention is only over previous inputs) - just a lower triangular matrix of 1s
        # mask = torch.tril(torch.ones(context_length, context_length)).view(
        #     1, 1, context_length, context_length
        # )
        # self.register_buffer("mask", mask)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.nets["kv"].weight)
        if self.nets["kv"].bias is not None:
            nn.init.normal_(self.nets["kv"].bias)
        
        nn.init.xavier_uniform_(self.nets["q"].weight)
        if self.nets["q"].bias is not None:
            nn.init.normal_(self.nets["q"].bias)

        nn.init.xavier_uniform_(self.nets["output"].weight)
        if self.nets["output"].bias is not None:
            nn.init.normal_(self.nets["output"].bias)

    def forward(self, x, cond):
        """
        Input should be shape (B, T, D) where B is batch size, T is seq length (@self.context_length), and
        D is input dimension (@self.embed_dim).
        """

        # enforce shape consistency
        assert len(x.shape) == 3
        B, T, D = x.shape

        assert D == self.embed_dim

        # compute key, query, and value vectors for each member of sequence, and split across attention heads
        kv = self.nets["kv"](cond)
        k, v = torch.chunk(kv, 2, dim=-1)
        q = self.nets["q"](x)

        # batched matrix multiplication between queries and keys to get all pair-wise dot-products.
        # We broadcast across batch and attention heads and get pair-wise dot-products between all pairs of timesteps
        # [B, T, D] x [B, D, T] -> [B, T, T]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # use mask to replace entries in dot products with negative inf to ensure they don't contribute to softmax,
        # then take softmax over last dimension to end up with attention score for each member of sequence.
        # Note the use of [:T, :T] -  this makes it so we can handle sequences less than @self.context_length in length.

        att = F.softmax(
            att, dim=-1
        )  # shape [B, T, T], last dimension has score over all T for each sequence member

        # dropout on attention
        # att = self.nets["attn_dropout"](att)

        # take weighted sum of value vectors over whole sequence according to attention, with batched matrix multiplication
        # [B, T, T] x [B, T, DH] -> [B, T, DH]
        y = att @ v

        # pass through output layer + dropout
        y = self.nets["output"](y)
        y = self.nets["output_dropout"](y)

        return y

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # this module doesn't modify the size of the input, it goes from (B, T, D) -> (B, T, D)
        return list(input_shape)
    
class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
      inputs: input data.
      mlp_dim: dimension of the mlp on top of attention block.
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout for attention heads.
      deterministic: bool, deterministic or not (to apply dropout).
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
    """
    def __init__(self,
        token_embedding_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        device: str = 'cuda'
    ):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.device = device

        self.layer_norm1 = nn.LayerNorm(token_embedding_size)

        self.use_rs_attn = True
        if self.use_rs_attn:
            self.multihead_attention = CausalSelfAttention(
                embed_dim=token_embedding_size,
                num_heads=num_heads,
                attn_dropout=attention_dropout_rate,
                output_dropout=dropout_rate,
                device = device
            )
        else:
            self.multihead_attention = nn.MultiheadAttention(
                embed_dim=token_embedding_size,
                num_heads=num_heads,
                dropout=attention_dropout_rate,
                batch_first=True
            )

        self.dropout = nn.Dropout(dropout_rate)

        self.layer_norm2 = nn.LayerNorm(token_embedding_size)
        self.mlp_block = MlpBlock(mlp_dim=mlp_dim, out_dim=token_embedding_size, dropout_rate=dropout_rate, device=device)

    def forward(self, inputs, cond, attention_mask):
        """Applies Encoder1DBlock module.

        Args:
          inputs: Inputs to the layer.
          deterministic: Dropout will not be applied when set to true.

        Returns:
          output after transformer encoder block.
        """

        # Attention block.
        assert inputs.ndim == 3, f"Expected (batch, seq, hidden) got {inputs.shape}"
        x = self.layer_norm1(inputs)

        if self.use_rs_attn:
            x = self.multihead_attention(x)
        else:
            x, attn_weights = self.multihead_attention(x, x, x, attn_mask=attention_mask)

        x = self.dropout(x)
        x = x + inputs

        # MLP block.
        y = self.layer_norm2(x)
        y = self.mlp_block(y)

        return x + y

class AdaLNBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self,
        token_embedding_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        device: str = 'cuda'
    ):
        super().__init__()
        self.use_rs_attn = True
        if self.use_rs_attn:
            self.multihead_attention = CausalSelfAttention(
                embed_dim=token_embedding_size,
                num_heads=num_heads,
                attn_dropout=attention_dropout_rate,
                output_dropout=dropout_rate,
                device = device
            )
        else:
            self.multihead_attention = nn.MultiheadAttention(
                embed_dim=token_embedding_size,
                num_heads=num_heads,
                dropout=attention_dropout_rate,
                batch_first=True
            )
        self.mlp = FF_MLP(token_embedding_size, mlp_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(token_embedding_size, 6 * token_embedding_size, bias=True)
        )
        self.layer_norm1 = nn.LayerNorm(token_embedding_size, elementwise_affine=False, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(token_embedding_size, elementwise_affine=False, eps=1e-6)

    def forward(self, x, cond, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=2)
        import ipdb; ipdb.set_trace()
        moduln = modulate(self.layer_norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.multihead_attention(moduln, moduln, moduln)
        x = x + gate_mlp * self.mlp(modulate(self.layer_norm2(x), shift_mlp, scale_mlp))
        return x

class CrossAttentionBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
      inputs: input data.
      mlp_dim: dimension of the mlp on top of attention block.
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout for attention heads.
      deterministic: bool, deterministic or not (to apply dropout).
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
    """
    def __init__(self,
        token_embedding_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        device: str = 'cuda'
    ):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.device = device

        self.layer_norm1 = nn.LayerNorm(token_embedding_size)

        self.multihead_self_attention = CausalSelfAttention(
            embed_dim=token_embedding_size,
            num_heads=num_heads,
            attn_dropout=attention_dropout_rate,
            output_dropout=dropout_rate,
            device = device
        )
        
        self.cross_attention = CrossAttention(
            embed_dim=token_embedding_size,
            attn_dropout=attention_dropout_rate,
            output_dropout=dropout_rate,
            device = device
        )
        self.dropout = nn.Dropout(dropout_rate)

        self.layer_norm2 = nn.LayerNorm(token_embedding_size)
        self.mlp_block = MlpBlock(mlp_dim=mlp_dim, out_dim=token_embedding_size, dropout_rate=dropout_rate, device=device)

    def forward(self, inputs, cond, attn_mask=None):
        """Applies Encoder1DBlock module.

        Args:
          inputs: Inputs to the layer.
          deterministic: Dropout will not be applied when set to true.

        Returns:
          output after transformer encoder block.
        """

        # Attention block.
        assert inputs.ndim == 3, f"Expected (batch, seq, hidden) got {inputs.shape}"
        x = self.layer_norm1(inputs)

        x = self.multihead_self_attention(x)

        x = self.cross_attention(x, cond)

        x = self.dropout(x)
        x = x + inputs

        # MLP block.
        y = self.layer_norm2(x)
        y = self.mlp_block(y)

        return x + y
    
class Transformer(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation.

    Attributes:
      num_layers: number of layers
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout rate in self attention.
    """
    def __init__(
        self,
        num_layers: int,
        mlp_dim: int,
        num_attention_heads: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        add_position_embedding: bool = False,
        window_size: int = 2,
        token_embedding_size: int = 384,
        attention_type: str = 'SA',
        device: str = 'cuda',
        **kwargs
    ):
        super().__init__()
        self.num_layers = num_layers
        self.add_position_embedding = add_position_embedding
        self.attention_type = attention_type
        self.device = device

        self.position_emb = AddPositionEmbs(window_size=window_size, dim=mlp_dim).to(self.device)
        self.pos_emb_dropout = nn.Dropout(dropout_rate).to(self.device)


        if attention_type == 'SA':
            attn_block = Encoder1DBlock
        elif attention_type == 'CA':
            attn_block = CrossAttentionBlock
        elif attention_type == 'AdaLN':
            attn_block = AdaLNBlock
        else:
            raise NotImplementedError('Attention type not implemented!')
        
        self.encoder_blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            self.encoder_blocks.append(
                attn_block(
                    token_embedding_size=token_embedding_size,
                    mlp_dim=mlp_dim,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    num_heads=num_attention_heads,
                    device=device,
                ).to(self.device)
            )
        
        self.layer_norm = nn.LayerNorm(token_embedding_size).to(self.device)

    def forward(self, x, cond=None, attention_mask=None):
        """Applies Transformer model on the inputs.

        Args:
          x: Inputs to the layer.
          train: Set to `True` when training.

        Returns:
          output of a transformer encoder.
        """
        assert x.ndim == 3  # (batch, len, emb)
        if self.add_position_embedding:
            x = self.position_emb(x)
            x = self.pos_emb_dropout(x)

        # Input Encoder
        for lyr in range(self.num_layers):
            x = self.encoder_blocks[lyr](x, cond, attention_mask)
        
        encoded = self.layer_norm(x)
        return encoded


def common_transformer_sizes(transformer_size: str) -> (int, dict):
    """
    Args:
        transformer_size (str): The size of the transformer. One of "dummy", "vanilla", "vit_s", "vit_b", "vit_l", "vit_h"

    Returns:
            token_embedding_size (int): The size of the token embeddings
            transformer_kwargs (dict): The kwargs to pass to the transformer

    """
    assert transformer_size in ["dummy", "vanilla", "vit_s", "vit_b", "vit_l", "vit_h"]
    default_params = {
        "attention_dropout_rate": 0.0,
        "add_position_embedding": False,
    }

    TRANSFORMER_SIZES = {
        "dummy": dict(
            num_layers=1,
            mlp_dim=256,
            num_attention_heads=2,
            dropout_rate=0.1,
        ),
        "vanilla": dict(
            num_layers=4,
            mlp_dim=1024,
            num_attention_heads=8,
            dropout_rate=0.1,
        ),
        "vit_s": dict(
            num_layers=12,
            mlp_dim=1536,
            num_attention_heads=6,
            dropout_rate=0.0,
        ),
        "vit_b": dict(
            num_layers=12,
            mlp_dim=3072,
            num_attention_heads=12,
            dropout_rate=0.0,
        ),
        "vit_l": dict(
            num_layers=24,
            mlp_dim=4096,
            num_attention_heads=16,
            dropout_rate=0.1,
        ),
        "vit_h": dict(
            num_layers=32,
            mlp_dim=5120,
            num_attention_heads=16,
            dropout_rate=0.1,
        ),
    }

    TOKEN_DIMS = {
        "dummy": 256,
        "vanilla": 256,
        "vit_s": 384,
        "vit_b": 768,
        "vit_l": 1024,
        "vit_h": 1280,
    }

    return TOKEN_DIMS[transformer_size], {
        **default_params,
        **TRANSFORMER_SIZES[transformer_size],
    }