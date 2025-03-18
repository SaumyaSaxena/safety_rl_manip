import torch
from typing import Sequence

class TokenGroup:
    """A group of tokens that have semantic meaning together (e.g. the tokens for a single observation)

    Attributes:
        tokens: jax.Array of shape (..., n_tokens, token_dim)
        mask: jax.Array of shape (..., n_tokens) indicating which tokens are valid (1) vs padding (0)
    """
    def __init__(self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ):
        self.tokens = tokens
        self.mask = mask

    @classmethod
    def create(
        cls, tokens: torch.Tensor, mask: torch.Tensor = None, **kwargs
    ):
        if mask is None:
            mask = torch.ones(tokens.shape[:-1])
        assert mask.ndim == tokens.ndim - 1
        return cls(tokens, mask, **kwargs)

    @classmethod
    def concatenate(cls, group_list: Sequence["TokenGroup"], axis=-2):
        data = torch.concatenate([t.tokens for t in group_list], axis=axis)
        mask = torch.concatenate([t.mask for t in group_list], axis=axis + 1)
        return cls(data, mask)