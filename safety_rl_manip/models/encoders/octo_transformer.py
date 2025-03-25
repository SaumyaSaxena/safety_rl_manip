import torch
import torch.nn as nn
import logging
from typing import List, Dict
from typing import Optional, Sequence

from safety_rl_manip.models.encoders.base import TokenGroup
from safety_rl_manip.models.encoders.block_transformer import (
    AttentionRule,
    BlockTransformer,
    PrefixGroup,
    TimestepGroup,
)
from itertools import chain

class OctoTransformer(nn.Module):
    """
    This module forms the base of the Octo architecture.

    The core idea is to run a causal transformer on the following sequence,

        [task, observation 0, observation 1, observation 2, ...]

    The task is tokenized using a set of *task tokenizers* (for example, a tokenizer that processes the
    language instruction into tokens, or one that processes the goal images into tokens).

    The observation at each timestep is tokenized using a set of *observation tokenizers*
    (for example, a tokenizer that processes the primary image into tokens, or one that processes
    the wrist image into tokens).

    We introduce additional tokens ("readouts") that "read out" the information in the transformer for
    downstream action or value prediction. For example, we may have an "action" readout that provides
    embeddings that are useful for predicting actions, and a "value" readout with embeddings that are useful
    for predicting values.

    The transformer is a blockwise-causal transformer, where each timestep only attends to the same or
    previous timesteps.  The easiest way to understand how the model works is to run:

    ```
        >>> model(observations, tasks, pad_mask, verbose=True)
    ```

    Generally, the model runs the transformer on something like the following sequence:

    [
        <task language tokens>,
        <t=0 "image_primary" tokens>, <t=0 "image_wrist" tokens>, <t=0 readout_action tokens>, ...
        <t=1 "image_primary" tokens>, <t=1 "image_wrist" tokens>, <t=1 readout_action tokens>, ...
        <t=2 "image_primary" tokens>, <t=2 "image_wrist" tokens>, <t=2 readout_action tokens>, ...
        ...
    ]

    The observation tokens attend to the task prefix, and to all observation tokens in the same or previous
    timesteps. So, "image_wrist" can attend to "image_primary" and vice versa.

    Readouts provide a mechanism for "reading out" the information in the transformer. They are designed to
    only *read* from the sequence before it, without the ability to influence (i.e. write) the computation for
    any of the non-readout tokens. By design, different readouts (e.g. "action" vs "value") are completely
    independent of each other, meaning they can be run separately without affecting each other.

    Args:
        observations_tokenizers (Dict[str, nn.Module]): Dictionary of flax modules for tokenizing the observations.
            The output of each tokenizer is concatenated to form the observation tokens.
        task_tokenizers (Dict[str, nn.Module]): Dictionary of flax modules for tokenizing the task.
            The output of each tokenizer is concatenated to form the task token prefix.
        readouts (Dict[str, int]): Dictionary of {readout_name: n_tokens_for_readout}.
        transformer_kwargs (Dict): Dictionary of kwargs to forward to the Transformer.
        token_embedding_size (int): Dimension of the token embeddings
        max_horizon (int): The maximum number of timesteps that the transformer can be run with. Note that while the
            transformer can be run with any horizon <= max_horizon, the model will only generate sane outputs for
            horizon lengths smaller or equal to the pre-training horizon.
    """
    def __init__(self,
        observation_tokenizers_names: List[str] = [],
        observation_tokenizers: Dict[str, nn.Module] = {},
        task_tokenizers_names: List[str] = [],
        task_tokenizers: Dict[str, nn.Module] = {},
        readouts: Dict[str, int] = {},
        transformer_kwargs: Dict = {},
        token_embedding_size: int = 256,
        max_horizon: int = 1,
        max_tokens: int= 16,
        device: str = 'cuda',
    ):
        super().__init__()
        self.observation_tokenizers_names = observation_tokenizers_names
        self.observation_tokenizers = observation_tokenizers
        self.task_tokenizers_names = task_tokenizers_names
        self.task_tokenizers = task_tokenizers
        self.readouts = readouts
        self.transformer_kwargs = transformer_kwargs
        self.token_embedding_size = token_embedding_size
        self.max_horizon = max_horizon
        self.max_tokens = max_tokens
        self.device = device

        self.task_dense_layers = nn.ModuleList()
        self.task_pos_emb = nn.ModuleList()
        for tok in self.task_tokenizers:
            self.task_dense_layers.append(
                nn.Linear(
                    in_features=tok.hidden_dim,
                    out_features=token_embedding_size
                )
            )
            self.task_pos_emb.append(nn.Embedding(max_tokens, token_embedding_size))

        self.observation_dense_layers = nn.ModuleList()
        self.observation_pos_emb = nn.ModuleList()
        for tok in self.observation_tokenizers:
            self.observation_dense_layers.append(
                nn.Linear(
                    in_features=tok.num_features, 
                    out_features=token_embedding_size
                )
            )
            self.observation_pos_emb.append(
                nn.Embedding(max_horizon*tok.num_tokens, token_embedding_size)
            ) # TODO(saumya): tok.num_tokens is only correct only for image size 256*256

        self.readout_pos_emb = nn.ModuleList()
        for _, n_tokens in self.readouts.items():
            self.readout_pos_emb.append(nn.Embedding(max_horizon*n_tokens, token_embedding_size))

        self.block_transformer = BlockTransformer(transformer_kwargs, device = device)
        self.reset_parameters()

    def reset_parameters(self):
        for emb in chain(self.task_pos_emb, self.observation_pos_emb, self.readout_pos_emb):
            nn.init.normal_(emb.weight)

        for layer in chain(self.task_dense_layers, self.observation_dense_layers):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.normal_(layer.bias)
                
        
    def forward(
        self,
        observations: Dict,
        tasks: Dict = {},
        pad_mask: torch.Tensor = None,
        readouts: Optional[Sequence[str]] = None,
        train: bool = False,
        verbose: bool = False,
    ) -> Dict[str, TokenGroup]:
        """
        Args:
            observations: A dictionary containing observation data for a batch of trajectory windows.
                Each entry has shape (batch, horizon, *).
            tasks: A dictionary containing task data for the trajectory windows.
                Each entry has shape (batch, *).
            pad_mask: A boolean mask of shape (batch, horizon) where False indicates a padded timestep.
            readouts: A list of readouts to compute. If None, defaults to all readouts. Must be a subset of the readouts specified in the model config.
            train: Whether model is being trained.
            verbose: If True, prints out the transformer structure.

        Returns:
            transformer_outputs: A dictionary {token_group_name: token_group},
                which contain the transformer embeddings for all observation tokens, task tokens, and readout tokens.
                The special keys "task" and "obs" contain the concatenated embeddings for all task tokens and observation tokens, respectively.

        Note: Horizon can be anything <= max_horizon.
        """
        if readouts is None:
            readouts = list(self.readouts.keys())

        #
        # Check that all inputs are valid
        #

        assert set(readouts).issubset(
            set(self.readouts.keys())
        ), "readouts must be specified in the model config"

        # batch_size, horizon = jax.tree_util.tree_leaves(observations)[0].shape[:2]
        batch_size, horizon = observations[list(observations.keys())[0]].shape[:2]
        assert horizon <= self.max_horizon, "horizon must be <= max_horizon"

        #
        # Attention rules for the transformer
        #

        # Tasks attend to all other tasks, but not to observations or readouts
        task_attention_rules = {"task_*": AttentionRule.CAUSAL}

        # Observations attend to all tasks and all other observations tokens causally,
        # e.g. at same timestep or before, but do not attend to readouts

        observation_attention_rules = {
            "task_*": AttentionRule.CAUSAL,
            "obs_*": AttentionRule.CAUSAL,
        }

        #
        # Create inputs for the transformer
        #

        all_prefix_groups = []
        all_timestep_groups = []

        #
        # First, add the task tokens
        #
        for i, (name, tok) in enumerate(zip(self.task_tokenizers_names, self.task_tokenizers)):
            group_name = f"task_{name}"
            # Receive inputs from tokenizer and cast to embedding size
            tokenizer_output: TokenGroup = tok(observations, tasks, train=train)
            if tokenizer_output is None:
                logging.warning(f"Skipping task tokenizer: {group_name}")
                continue

            task_tokens = self.task_dense_layers[i](tokenizer_output.tokens)
            # task_tokens shape is (batch, n_tokens, token_embedding_size)

            # Add positional embedding
            embedding = self.task_pos_emb[i](torch.arange(self.max_tokens).to(self.device))
            task_tokens += torch.broadcast_to(embedding, task_tokens.shape)

            all_prefix_groups.append(
                PrefixGroup(
                    tokens=task_tokens,
                    mask=tokenizer_output.mask,
                    name=group_name,
                    attention_rules=task_attention_rules,
                )
            )

        #
        # Next, add the observation tokens
        #
        for i , (name, tok) in enumerate(zip(self.observation_tokenizers_names, self.observation_tokenizers)):
            group_name = f"obs_{name}"
            # Receive inputs from tokenizer and cast to embedding size
            tokenizer_output: TokenGroup = tok(observations, tasks)
            
            if tokenizer_output is None:
                # logging.warning(f"Skipping observation tokenizer: {group_name}")
                continue

            obs_tokens = self.observation_dense_layers[i](tokenizer_output.tokens)
            # obs_tokens shape is (batch, horizon, n_tokens, token_embedding_size)

            # # Add positional embedding # TODO: remove hardcoded
            # embedding = self.observation_pos_emb[i](torch.arange(tok.num_tokens*horizon).to(self.device)) # Use only the timesteps we receive as input
            # embedding = embedding.reshape(1, *obs_tokens.shape[1:])
            # obs_tokens += torch.broadcast_to(embedding, obs_tokens.shape)

            # Update mask to account for which timesteps are padding
            obs_pad_mask = torch.logical_and(pad_mask[:, :, None], tokenizer_output.mask)

            all_timestep_groups.append(
                TimestepGroup(
                    tokens=obs_tokens,
                    mask=obs_pad_mask,
                    name=group_name,
                    attention_rules=observation_attention_rules,
                )
            )
        #
        # Finally, add the readout tokens
        #
        for i, readout_name in enumerate(readouts):
            group_name = f"readout_{readout_name}"
            # Readouts do not correspond to any inputs, just positional embeddings
            n_tokens_for_readout = self.readouts[readout_name]
            readout_tokens = torch.zeros(
                (batch_size, horizon, n_tokens_for_readout, self.token_embedding_size),
                device=self.device
            )
            # Add positional embedding
            embedding = self.readout_pos_emb[i](torch.arange(n_tokens_for_readout*horizon).to(self.device)) # Use only the timesteps we receive as input
            embedding = embedding.reshape(1, horizon, n_tokens_for_readout, self.token_embedding_size)
            readout_tokens += torch.broadcast_to(embedding, readout_tokens.shape)

            readout_mask = torch.ones((batch_size, horizon, n_tokens_for_readout), device=self.device)
            readout_attention_rules = {
                "task_*": AttentionRule.CAUSAL,
                "obs_*": AttentionRule.CAUSAL,
                group_name: AttentionRule.CAUSAL,
            }  # Attend to tasks, all previous observations, and *only it's own own readout*

            all_timestep_groups.append(
                TimestepGroup(
                    tokens=readout_tokens,
                    mask=readout_mask,
                    name=group_name,
                    attention_rules=readout_attention_rules,
                )
            )

        # Run the transformer!
        assert (
            self.transformer_kwargs.get("add_position_embedding", False) is False
        ), "Already added positional embeddings to the tokens"


        prefix_outputs, timestep_outputs = self.block_transformer(
            all_prefix_groups,
            all_timestep_groups,
            train=train,
            verbose=verbose,
        )

        outputs = {}
        outputs.update(
            {
                group.name: TokenGroup(group.tokens, group.mask)
                for group in prefix_outputs
            }
        )
        outputs.update(
            {
                group.name: TokenGroup(group.tokens, group.mask)
                for group in timestep_outputs
            }
        )

        if len(prefix_outputs) > 0:
            outputs["task"] = TokenGroup.concatenate(
                [TokenGroup(group.tokens, group.mask) for group in prefix_outputs]
            )

        outputs["obs"] = TokenGroup.concatenate(
            [
                TokenGroup(group.tokens, group.mask)
                for group in timestep_outputs
                if group.name.startswith("obs_")
            ],
            axis=-2,
        )
        # with torch.cuda.device(self.device):
        #     torch.cuda.empty_cache()
        return outputs

    def _create_positional_embedding(self, name: str, tokens: torch.Tensor):
        if tokens.ndim == 3:  # for prefixes
            shape = (1, *tokens.shape[-2:])
        elif (
            tokens.ndim == 4
        ):  # for timesteps, create embedding for max_horizon, then truncate
            shape = (1, self.max_horizon, *tokens.shape[-2:])
        else:
            raise ValueError(f"Invalid tokens shape: {tokens.shape}")

        embedding = self.param(
            f"{name}_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            shape,
        )
        if tokens.ndim == 4:
            # Use only the timesteps we receive as input
            embedding = embedding[:, : tokens.shape[1]]
        return torch.broadcast_to(embedding, tokens.shape)