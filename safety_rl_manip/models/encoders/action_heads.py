from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any

from einops import rearrange

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from safety_rl_manip.models.encoders.base import TokenGroup
from safety_rl_manip.models.encoders.tokenizers import BinTokenizer
from safety_rl_manip.models.encoders.transformer import MAPHead
    

def masked_mean(x, mask):
    mask = torch.broadcast_to(mask, x.shape)
    return torch.mean((x * mask).type(torch.float)) / torch.clip(torch.mean(mask.type(torch.float)), min=1e-5)


def chunk_actions(actions, pred_horizon):
    """Chunk actions for predicting actions `pred_horizon` steps into the future.

    The resulting actions have shape (batch, actions.shape[-2] - (pred_horizon - 1), pred_horizon, action_dim)

    For example: chunk_actions([a_1, a_2, a_3, a_4, a_5], 3) ->
        [
            [a_1, a_2, a_3],
            [a_2, a_3, a_4],
            [a_3, a_4, a_5],
        ]

    """
    assert (
        actions.ndim == 3
    ), f"Expected actions to have shape (batch, window_size, action_dim), but got shape {actions.shape}"
    window_size = actions.shape[1]
    assert (
        window_size >= pred_horizon
    ), f"pred_horizon {pred_horizon} too large for window size {window_size}"
    chunk_window_size = window_size - (pred_horizon - 1)

    curr_step = torch.arange(chunk_window_size)
    action_offset = torch.arange(pred_horizon)
    chunk_indices = curr_step[:, None] + action_offset[None, :]
    return actions[:, chunk_indices]


def _check_action_window_size(actions, window_size, pred_horizon):
    assert (
        actions.shape[1] >= window_size + pred_horizon - 1
    ), f"""
        To predict actions for window_size {window_size} and future prediction horizon {pred_horizon},
        the ground-truth actions must have at least {window_size + pred_horizon - 1} timesteps, but got shape {actions.shape}.

        Did you make sure to set "future_action_window_size" correctly in the data config?
    """

class DiscreteActionHead(nn.Module):
    def __init__(self,
        readout_key: str,
        use_map: bool = False,
        pred_horizon: int = 1,
        action_dim: int = 7,
        max_action: float = 6.0,
        min_action: float = 0.0,
        prediction_type: str = 'action',
        action_repr: str = 'one-hot',
        loss_type: str = "mse",
        embedding_size: int = 384,
        device: str = 'cuda',
        **kwargs,
    ):
        super().__init__()
        self.readout_key = readout_key
        self.prediction_type = prediction_type
        self.action_repr = action_repr
        self.max_action = max_action
        self.min_action = min_action
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

        self.action_mlp = nn.Linear(
            in_features=embedding_size,
            out_features=action_dim*pred_horizon,
        )

    def forward(self, transformer_outputs: Dict[str, TokenGroup]):
        token_group = transformer_outputs[self.readout_key]

        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )
        # Mean pooling
        embeddings = token_group.tokens.mean(axis=-2)
        # Now, embeddings is (batch_size, window_size, embedding_size)

        pred_logits = self.action_mlp(embeddings)
        pred_logits = rearrange(
            pred_logits, "b w (p a) -> b w p a", p=self.pred_horizon, a=self.action_dim
        )
        return pred_logits
    
    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        actions,
        pad_mask,
        train: bool = True,
    ):
        pred_logits = self(transformer_outputs)

        batch_size, window_size = pad_mask.shape
        _check_action_window_size(actions, window_size, self.pred_horizon)
        actions_chunked = chunk_actions(actions, self.pred_horizon) # shape(bs, num_chunks, pred_horizon, action_dim)
        actions_chunked = actions_chunked[:, :window_size]
        
        labels = torch.argmax(actions_chunked, axis=-1)

        cross_ent_loss = torch.nn.CrossEntropyLoss(reduction='none')

        logits_flat = rearrange(
            pred_logits,
            "b w h a -> (b w h) a"
        )

        labels_flat = rearrange(
            labels,
            "b w h -> (b w h)"
        )
        loss = cross_ent_loss(logits_flat, labels_flat) 

        loss = rearrange(
            loss,
            "(b w h) -> b w h",
            b = labels.shape[0],
            w = labels.shape[1],
            h = labels.shape[2],
        )
        loss = masked_mean(loss, pad_mask[:,:,None])

        # compute accuracy between predicted actions and target actions
        pred_label = torch.argmax(pred_logits, axis=-1)
        accuracy = pred_label == labels
        accuracy = masked_mean(accuracy, pad_mask[:,:,None])

        loss = loss * self.action_dim
        return loss, {
            "cross_ent_loss": loss,
            "accuracy": accuracy,
        }

    def predict_action(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        train: bool = True,
        argmax: bool = False,
        sample_shape: tuple = (),
        temperature: float = 1.0,
        device = 'cuda',
    ) -> torch.Tensor:
        pred_logits = self(transformer_outputs)[:,-1,0] # last window and first time step
        action_tokens = torch.argmax(pred_logits, axis=-1, keepdim=True)
        return action_tokens

class ContinuousActionHead(nn.Module):
    """Predicts continuous actions (as opposed to discretized).

    Continuous actions are predicted by tanh squashing the model output to [-max_action, max_action], and then
    optimized using a standard regression loss.

    You may create an embedding by either mean-pooling across tokens (use_map=False) or using multi-head
    attention pooling (use_map=True). It is recommended to use MAP when decoding from the observation token
    stream.
    """
    def __init__(self,
        readout_key: str,
        use_map: bool = False,
        pred_horizon: int = 1,
        action_dim: int = 7,
        min_action: float = 6.0,
        max_action: float = 6.0,
        loss_type: str = "mse",
        embedding_size: int = 384,
        hidden_sizes: list = [256, 256],
        device: str = 'cuda',
    ):
        super().__init__()
        self.readout_key = readout_key
        self.use_map = use_map
        self.min_action = min_action
        self.max_action = max_action
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.loss_type = loss_type
        self.embedding_size = embedding_size

        if self.use_map:
            self.map_head = MAPHead()
        
        # self.mean_proj = nn.Linear(
        #     in_features=embedding_size,
        #     out_features=action_dim*pred_horizon,
        # )

        sizes = [embedding_size] + list(hidden_sizes) + [action_dim*pred_horizon]
        layers = []
        for j in range(len(sizes)-1):
            act = nn.SiLU if j < len(sizes)-2 else nn.Tanh
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        self.mean_proj = nn.Sequential(*layers)

    def forward(self, transformer_outputs: Dict[str, TokenGroup], train: bool = True):
        """
        Returns:
            mean: Predicted actions w/ shape (batch_size, window_size, pred_horizon, action_dim)
        """
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )
        if self.use_map:  # Multi-head attention pooling
            embeddings = self.map_head(token_group, train=train)[:, :, 0]
        else:  # mean pooling
            embeddings = token_group.tokens.mean(axis=-2)
        # Now, embeddings is (batch_size, window_size, embedding_size)

        mean = self.forward_emb(embeddings)
        
        return mean

    def forward_emb(self, inputs):
        """ 
            inputs: shape (batch_size, window_size, embedding_size)
            outputs: shape (batch_size, window_size, pred_horizon, value_dim)
        """
        actions = self.mean_proj(inputs)
        actions = rearrange(
            actions, "b w (p a) -> b w p a", p=self.pred_horizon, a=self.action_dim
        )
        actions = (actions + 1) / 2 * (self.max_action - self.min_action) + self.min_action
        return actions

    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        actions,
        pad_mask,
        train: bool = True,
    ):
        """Computes the loss for the action regression objective.

        Args:
            transformer_ouputs: must contain self.readout_key with shape (batch_size, window_size, num_tokens,
                embedding_size)
            actions: shape (batch_size, >= window_size + pred_horizon - 1, action_dim)
            pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep

        Returns:
            loss: float
            metrics: dict
        """
        # (batch, window_size, pred_horizon, action_dim)
        mean = self(transformer_outputs, train=train)

        window_size = mean.shape[1]
        _check_action_window_size(actions, window_size, self.pred_horizon)
        actions_chunked = chunk_actions(actions, self.pred_horizon)
        actions_chunked = actions_chunked[:, :window_size]

        loss, metrics = continuous_loss(
            mean, actions_chunked, pad_mask[:, :, None, None], loss_type=self.loss_type
        )
        # Sum over action dimension instead of averaging
        loss = loss * self.action_dim
        metrics["loss"] = metrics["loss"] * self.action_dim
        metrics["mse"] = metrics["mse"] * self.action_dim
        return loss, metrics

    def predict_action(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        train: bool = True,
        *args,
        sample_shape: tuple = (),
        **kwargs,
    ):
        """Convenience methods for predicting actions for the final timestep in the window."""
        # only get the last timestep in the window
        # (batch, pred_horizon, action_dim)
        mean = self(transformer_outputs, train=train)[:, -1]
        return torch.broadcast_to(mean, sample_shape + mean.shape)


LOG_STD_MAX = 2
LOG_STD_MIN = -20
from torch.distributions.normal import Normal

class SquashedGaussianActionHead(nn.Module):

    def __init__(self,
        readout_key: str,
        use_map: bool = False,
        pred_horizon: int = 1,
        action_dim: int = 7,
        min_action: float = 6.0,
        max_action: float = 6.0,
        loss_type: str = "mse",
        embedding_size: int = 384,
        hidden_sizes: list = [256, 256],
        device: str = 'cuda',
    ):
        super().__init__()
        self.readout_key = readout_key
        self.use_map = use_map
        self.min_action = min_action
        self.max_action = max_action
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.loss_type = loss_type
        self.embedding_size = embedding_size

        if self.use_map:
            self.map_head = MAPHead()
        
        # self.mean_proj = nn.Linear(
        #     in_features=embedding_size,
        #     out_features=action_dim*pred_horizon,
        # )

        sizes = [embedding_size] + list(hidden_sizes)
        layers = []
        for j in range(len(sizes)-1):
            layers += [nn.Linear(sizes[j], sizes[j+1]), nn.SiLU()]
        self.mean_proj = nn.Sequential(*layers)

        self.mu_layer = nn.Linear(hidden_sizes[-1], action_dim*pred_horizon)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], action_dim*pred_horizon)

    def forward(self, transformer_outputs: Dict[str, TokenGroup], deterministic: bool = False, with_logprob: bool = True):
        """
        Returns:
            mean: Predicted actions w/ shape (batch_size, window_size, pred_horizon, action_dim)
        """
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )
        if self.use_map:  # Multi-head attention pooling
            embeddings = self.map_head(token_group)[:, :, 0]
        else:  # mean pooling
            embeddings = token_group.tokens.mean(axis=-2)
        # Now, embeddings is (batch_size, window_size, embedding_size)

        pi_action, logp_pi = self.forward_emb(embeddings)
        
        return pi_action, logp_pi

    def forward_emb(self, embeddings, deterministic: bool = False, with_logprob: bool = True):
        """
        inputs: shape (batch_size, window_size, token_dim)
        Returns:
            mean: Predicted actions w/ shape (batch_size, window_size, pred_horizon, action_dim)
        """

        net_out = self.mean_proj(embeddings)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        mu = rearrange(
            mu, "b w (p a) -> b w p a", p=self.pred_horizon, a=self.action_dim
        )

        std = rearrange(
            std, "b w (p a) -> b w p a", p=self.pred_horizon, a=self.action_dim
        )

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=-1)
        else:
            logp_pi = None

        pi_action = (torch.tanh(pi_action) + 1) / 2 * (self.max_action - self.min_action) + self.min_action
        return pi_action, logp_pi

    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        actions,
        pad_mask,
        train: bool = True,
    ):
        """Computes the loss for the action regression objective.

        Args:
            transformer_ouputs: must contain self.readout_key with shape (batch_size, window_size, num_tokens,
                embedding_size)
            actions: shape (batch_size, >= window_size + pred_horizon - 1, action_dim)
            pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep

        Returns:
            loss: float
            metrics: dict
        """
        # (batch, window_size, pred_horizon, action_dim)
        mean = self(transformer_outputs, train=train)

        window_size = mean.shape[1]
        _check_action_window_size(actions, window_size, self.pred_horizon)
        actions_chunked = chunk_actions(actions, self.pred_horizon)
        actions_chunked = actions_chunked[:, :window_size]

        loss, metrics = continuous_loss(
            mean, actions_chunked, pad_mask[:, :, None, None], loss_type=self.loss_type
        )
        # Sum over action dimension instead of averaging
        loss = loss * self.action_dim
        metrics["loss"] = metrics["loss"] * self.action_dim
        metrics["mse"] = metrics["mse"] * self.action_dim
        return loss, metrics

    def predict_action(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        train: bool = True,
        *args,
        sample_shape: tuple = (),
        **kwargs,
    ):
        """Convenience methods for predicting actions for the final timestep in the window."""
        # only get the last timestep in the window
        # (batch, pred_horizon, action_dim)
        mean = self(transformer_outputs, train=train)[:, -1]
        return torch.broadcast_to(mean, sample_shape + mean.shape)
    

class ValueHead(nn.Module):
    """Predicts continuous values.

    Continuous actions are predicted by tanh squashing the model output to [-max_action, max_action], and then
    optimized using a standard regression loss.

    You may create an embedding by either mean-pooling across tokens (use_map=False) or using multi-head
    attention pooling (use_map=True). It is recommended to use MAP when decoding from the observation token
    stream.
    """
    def __init__(self,
        readout_key: str,
        use_map: bool = False,
        pred_horizon: int = 1,
        value_dim: int = 1,
        loss_type: str = "mse",
        embedding_size: int = 384,
        hidden_sizes: list = [256, 256],
        activation: nn.Module = nn.SiLU,
        output_activation: nn.Module = nn.Identity,
        device: str = 'cuda',
    ):
        super().__init__()
        self.readout_key = readout_key
        self.use_map = use_map
        self.pred_horizon = pred_horizon
        self.value_dim = value_dim
        self.loss_type = loss_type
        self.embedding_size = embedding_size

        if self.use_map:
            self.map_head = MAPHead()
        
        # self.mean_proj = nn.Sequential(
        #     nn.Linear(
        #         in_features=embedding_size,
        #         out_features=value_dim*pred_horizon,
        #     ),
        #     nn.ReLU(),
        # )

        sizes = [embedding_size] + list(hidden_sizes) + [value_dim*pred_horizon]
        layers = []
        for j in range(len(sizes)-1):
            act = activation if j < len(sizes)-2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        self.mean_proj = nn.Sequential(*layers)

    def forward(self, transformer_outputs: Dict[str, TokenGroup], actions_out: torch.Tensor, train: bool = True):
        """
        Returns:
            mean: Predicted values w/ shape (batch_size, window_size, pred_horizon, value_dim)
        """
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )

        if self.use_map:  # Multi-head attention pooling
            embeddings = self.map_head(token_group, train=train)[:, :, 0]
        else:  # mean pooling
            embeddings = token_group.tokens.mean(axis=-2)
        # Now, embeddings is (batch_size, window_size, embedding_size)

        inputs = torch.cat([embeddings, actions_out], dim=-1) 
        mean = self.forward_emb(inputs)
        
        return mean
    
    def forward_emb(self, inputs):
        """ 
            inputs: shape (batch_size, window_size, embedding_size)
            outputs: shape (batch_size, window_size, pred_horizon, value_dim)
        """
        values = self.mean_proj(inputs)
        values = rearrange(
            values, "b w (p a) -> b w p a", p=self.pred_horizon, a=self.value_dim
        )
        return values

    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        actions,
        pad_mask,
        train: bool = True,
    ):
        """Computes the loss for the action regression objective.

        Args:
            transformer_ouputs: must contain self.readout_key with shape (batch_size, window_size, num_tokens,
                embedding_size)
            actions: shape (batch_size, >= window_size + pred_horizon - 1, action_dim)
            pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep

        Returns:
            loss: float
            metrics: dict
        """
        # (batch, window_size, pred_horizon, action_dim)
        mean = self(transformer_outputs, train=train)

        window_size = mean.shape[1]
        _check_action_window_size(actions, window_size, self.pred_horizon)
        actions_chunked = chunk_actions(actions, self.pred_horizon)
        actions_chunked = actions_chunked[:, :window_size]

        loss, metrics = continuous_loss(
            mean, actions_chunked, pad_mask[:, :, None, None], loss_type=self.loss_type
        )
        # Sum over action dimension instead of averaging
        loss = loss * self.action_dim
        metrics["loss"] = metrics["loss"] * self.action_dim
        metrics["mse"] = metrics["mse"] * self.action_dim
        return loss, metrics

    def predict_value(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        actions_out: torch.Tensor,
        train: bool = True,
        *args,
        sample_shape: tuple = (),
        **kwargs,
    ):
        """Convenience methods for predicting actions for the final timestep in the window."""
        # only get the last timestep in the window
        # (batch, pred_horizon, action_dim)
        mean = self(transformer_outputs, train=train)[:, -1]
        return torch.broadcast_to(mean, sample_shape + mean.shape)
    

def continuous_loss(
    pred_value,
    noise,
    actions_flat,
    mask,
    loss_type: str = "mse",
    pred_type='noise',
):
    """
    Args:
        pred_value: shape (batch_dims...)
        ground_truth_value: continuous values w/ shape (batch_dims...)
        mask: broadcastable to ground_truth
    """
    if 'noise' in pred_type:
        if loss_type == "mse":
            loss = torch.square(pred_value - noise)
        elif loss_type == "l1":
            loss = torch.abs(pred_value - noise)
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")
    
    if 'action' in pred_type:
        if loss_type == "mse":
            loss = torch.square(pred_value - actions_flat)
        elif loss_type == "l1":
            loss = torch.abs(pred_value - actions_flat)
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")
    
    loss = masked_mean(loss, mask)
    return loss, {
        f"{loss_type}_loss": loss
    }


def discrete_loss(
    discrete_tokenizer: BinTokenizer,
    logits,
    ground_truth_value,
    mask,
):
    """
    Args:
        discrete_tokenizer: BinTokenizer to use on ground_truth_value
        logits: shape (batch_dims..., vocab_size)
        ground_truth_value: continuous values in w/ shape (batch_dims...)
        mask: broadcastable to ground_truth_value
    """
    labels = discrete_tokenizer(ground_truth_value)
    labels_one_hot = torch.nn.functional.one_hot(labels, logits.shape[-1])

    loss = -torch.sum(logits * labels_one_hot, axis=-1)
    loss = masked_mean(loss, mask)

    # compute accuracy between predicted actions and target actions
    pred_label = torch.argmax(logits, axis=-1)
    accuracy = pred_label == labels
    accuracy = masked_mean(accuracy, mask)

    # detokenize the predicted actions
    pred_value = discrete_tokenizer.decode(pred_label)
    mse = torch.square(pred_value - ground_truth_value)
    mse = masked_mean(mse, mask)
    return loss, {
        "loss": loss,
        "mse": mse,
        "accuracy": accuracy,
    }

def softmax_cross_entropy_loss(
    pred_logits,
    noise,
    actions,
    mask,
    pred_type='action',
    action_repr='bits',
):
    """
    Args:
        logits: shape (batch_dims..., vocab_size)
        ground_truth_value: continuous values in w/ shape (batch_dims...)
        mask: broadcastable to ground_truth_value
    """
    if 'action' in pred_type:
        if 'bits' in action_repr: # actions_flat is in bits
            labels = bits2int(actions > 0)
        else:
            labels = torch.argmax(actions, axis=-1)

        cross_ent_loss = torch.nn.CrossEntropyLoss(reduction='none')

        logits_channel_first = rearrange(
            pred_logits,
            "b w p a -> b a w p"
        )

        loss = cross_ent_loss(logits_channel_first, labels) 
        
        loss = masked_mean(loss, mask)

        # compute accuracy between predicted actions and target actions
        pred_label = torch.argmax(pred_logits, axis=-1)
        accuracy = pred_label == labels
        accuracy = masked_mean(accuracy, mask)

        return loss, {
            "cross_ent_loss": loss,
            "accuracy": accuracy,
        }
    else:
        raise NotImplementedError('Softmax cross entropy loss for noise prediction is not implemented.')

