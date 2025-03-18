import re
from typing import Dict, Optional, Sequence, Dict
import torch
import torch.nn as nn
from safety_rl_manip.models.encoders.base import TokenGroup
from safety_rl_manip.models.encoders.vit_encoders import *

import torch.nn.functional as F
import logging

EPS = 1e-6

def normalize_low_dim_obs(obs, min_obs, max_obs):
    return  2 * (obs-min_obs)/(max_obs-min_obs) - 1

def regex_match(regex_keys, x):
    return any([re.match(r_key, x) for r_key in regex_keys])

def regex_filter(regex_keys, xs):
    return list(filter(lambda x: regex_match(regex_keys, x), xs))

def generate_proper_pad_mask(
    tokens: torch.Tensor,
    pad_mask_dict: Optional[Dict[str, torch.Tensor]],
    keys: Sequence[str],
) -> torch.Tensor:
    if pad_mask_dict is None:
        # logging.warning("No pad_mask_dict found. Nothing will be masked.")
        return torch.ones(tokens.shape[:-1])
    if not all([key in pad_mask_dict for key in keys]):
        logging.warning(
            f"pad_mask_dict missing keys {set(keys) - set(pad_mask_dict.keys())}."
            "Nothing will be masked."
        )
        return torch.ones(tokens.shape[:-1])

    pad_mask = torch.stack([pad_mask_dict[key] for key in keys], axis=-1)
    pad_mask = torch.any(pad_mask, axis=-1)
    pad_mask = torch.broadcast_to(pad_mask[..., None], tokens.shape[:-1])
    return pad_mask

class ImageTokenizer(nn.Module):
    """Image tokenizer that encodes image stack into tokens with optional FiLM conditioning.

    Args:
        encoder: Encoder class.
        use_token_learner (bool): Whether to use token learner. Defaults to False.
        num_tokens (int): Number of output tokens, only enforced when use_token_learner is True.
        obs_stack_keys (Sequence[str]): Which spatial observation inputs get stacked for encoder input. Supports regex.
        task_stack_keys (Sequence[str]): Which spatial task inputs get stacked for encoder input. Supports regex.
        task_film_keys (Sequence[str]): Which non-spatial task keys get passed into FiLM conditioning. Supports regex.
    """

    def __init__(self,
        encoder: str,
        env_observation_shapes: Dict = {},
        use_token_learner: bool = False,
        num_tokens: int = 256,
        conditioning_type: str = "none",
        obs_stack_keys: Sequence[str] = ("image_.*", "depth_.*"),
        task_stack_keys: Sequence[str] = tuple(),
        task_film_keys: Sequence[str] = tuple(),
        proper_pad_mask: bool = True,
        finetune_encoder: bool = True,
        device: str = 'cuda',
    ):
        super().__init__()
        self.use_token_learner = use_token_learner
        self.num_tokens = num_tokens
        self.conditioning_type = conditioning_type
        self.obs_stack_keys = obs_stack_keys
        self.task_stack_keys = task_stack_keys
        self.task_film_keys = task_film_keys
        self.proper_pad_mask = proper_pad_mask
        self.finetune_encoder = finetune_encoder
        self.device = device

        # self.encoder_def = ModuleSpec.instantiate(encoder, device)()
        num_inp_channels = env_observation_shapes[obs_stack_keys[0]][2] # assuming all obs_stack_keys have same num_channels
        self.encoder_def = eval(encoder.name)(**encoder.kwargs, num_inp_channels=num_inp_channels, device=device)
        self.num_features = self.encoder_def.num_features

        # if not self.finetune_encoder:
        #     for p in self.encoder_def.parameters():
        #         p.requires_grad_(False)

    def forward(
        self,
        observations,
        tasks=None,
    ):
        def extract_inputs(keys, inputs, check_spatial=False):
            extracted_outputs = []
            for key in keys:
                if check_spatial:
                    assert len(inputs[key].shape) >= 4
                extracted_outputs.append(inputs[key])
            return torch.concatenate(extracted_outputs, axis=-1)

        obs_stack_keys = regex_filter(self.obs_stack_keys, sorted(observations.keys()))
        if len(obs_stack_keys) == 0:
            logging.info(
                f"No image inputs matching {self.obs_stack_keys} were found."
                "Skipping tokenizer entirely."
            )
            assert self.proper_pad_mask, "Cannot skip unless using proper_pad_mask."
            return None

        # stack all spatial observation and task inputs
        enc_inputs = extract_inputs(obs_stack_keys, observations, check_spatial=True)
        if tasks and self.task_stack_keys:
            needed_task_keys = regex_filter(self.task_stack_keys, observations.keys())
            # if any task inputs are missing, replace with zero padding (TODO: be more flexible)
            for k in needed_task_keys:
                if k not in tasks:
                    tasks['k'] = torch.zeros_like(observations[k][:, 0]).to(self.device)
            task_stack_keys = regex_filter(self.task_stack_keys, sorted(tasks.keys()))
            if len(task_stack_keys) == 0:
                raise ValueError(
                    f"No task inputs matching {self.task_stack_keys} were found."
                )
            task_inputs = extract_inputs(task_stack_keys, tasks, check_spatial=True)
            task_inputs = torch.stack([task_inputs]*enc_inputs.shape[1], dim=1)
            enc_inputs = torch.concatenate([enc_inputs, task_inputs], axis=-1)
        b, t, h, w, c = enc_inputs.shape
        enc_inputs = torch.reshape(enc_inputs, (b * t, h, w, c))

        # extract non-spatial FiLM inputs
        encoder_input_kwargs = {}
        if self.task_film_keys:
            film_inputs = extract_inputs(self.task_film_keys, tasks)
            film_inputs = film_inputs[:, None].repeat(t, axis=1)
            encoder_input_kwargs.update(
                {"cond_var": torch.reshape(film_inputs, (b * t, -1))}
            )
        
        # run visual encoder
        image_tokens = self.encoder_def(enc_inputs, **encoder_input_kwargs) # TODO(saumya): permute to change to (b,c,h,w) format. Check!
        image_tokens = torch.reshape(image_tokens, (b, t, -1, image_tokens.shape[-1]))

        if self.proper_pad_mask:
            pad_mask = generate_proper_pad_mask(
                image_tokens,
                observations.get("pad_mask_dict", None),
                obs_stack_keys,
            ).to(self.device)
        else:
            pad_mask = torch.ones(image_tokens.shape[:-1]).to(self.device)
        return TokenGroup(image_tokens, pad_mask)


class LanguageTokenizer(nn.Module):
    """
    Language tokenizer that embeds text input IDs into continuous language embeddings. Supports pre-trained HF models.

     Args:
         num_tokens (int): Number of output tokens (not enforced).
         encoder (str, optional): Optional HuggingFace AutoModel name for encoding input IDs.
         finetune_encoder (bool, optional): Optional finetune last layers of the language model.
    """

    def __init__(self,
        encoder: str = None,
        finetune_encoder: bool = False,
        proper_pad_mask: bool = True,
        device: str = 'cuda',
    ):
        super().__init__()
        self.encoder = encoder
        self.finetune_encoder = finetune_encoder
        self.proper_pad_mask = proper_pad_mask
        self.device = device

        
        if encoder is not None:
            if "t5" in self.encoder:
                from transformers import AutoConfig, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
                config = AutoConfig.from_pretrained(self.encoder)
                self.hidden_dim = config.d_model
                self.hf_model = T5ForConditionalGeneration.from_pretrained(self.encoder).to(self.device)
                # self.hf_model = AutoModelForSeq2SeqLM.from_pretrained(self.encoder).to(self.device)
            elif "bert" in self.encoder:
                self.hidden_dim = 768
            else:
                raise NotImplementedError(f"Tokenizer {self.encoder} not implemented")

            if not self.finetune_encoder:
                if "t5" in self.encoder:
                    for p in self.hf_model.parameters():
                        p.requires_grad_(False)
    
    def forward(
        self,
        observations,
        tasks=None,
        train: bool = True,
    ):
        if "t5" in self.encoder:
            if "language_instruction" not in tasks:
                logging.warning("No language inputs found. Skipping tokenizer entirely.")
                assert self.proper_pad_mask, "Cannot skip unless using proper pad mask."
                return None
            if not isinstance(tasks["language_instruction"], torch.Tensor):
                assert (
                    self.encoder is not None
                ), "Received language tokens but no encoder specified."
                bs = tasks["language_instruction"]['input_ids'].shape[0]
                decoder_input_ids = torch.tensor([[0]]*bs).to(self.device)

                if not self.finetune_encoder:
                    with torch.no_grad():
                        tokens = self.hf_model(
                            input_ids=tasks["language_instruction"]['input_ids'].type(torch.long),
                            attention_mask=tasks["language_instruction"]['attention_mask'], 
                            decoder_input_ids=decoder_input_ids).encoder_last_hidden_state
                else:
                    tokens = self.hf_model(
                        input_ids=tasks["language_instruction"]['input_ids'].type(torch.long),
                        attention_mask=tasks["language_instruction"]['attention_mask'], 
                        decoder_input_ids=decoder_input_ids).encoder_last_hidden_state
            else:
                # add a # tokens dimension to language
                if tasks["language_instruction"].ndim == 2:
                    tokens = tasks["language_instruction"][:, None, :]
                else:
                    tokens = tasks["language_instruction"]

        if "bert" in self.encoder:
            tokens = observations['bert_tokens']

        # TODO: incorporate padding info from language tokens here too
        if self.proper_pad_mask:
            pad_mask = generate_proper_pad_mask(
                tokens,
                tasks.get("pad_mask_dict", None),
                ("language_instruction",),
            )
        else:
            pad_mask = torch.ones(tokens.shape[:-1], device=self.device)

        return TokenGroup(tokens, pad_mask)

class BinTokenizer(nn.Module):
    """
    Tokenizes continuous inputs via dimension-wise binning in given range.

    Args:
        n_bins (int): Number of discrete bins per dimension.
        bin_type (str): Type of binning. ['uniform', 'normal' = Gaussian]
        low (float): Lower bound for bin range.
        high (float): Upper bound for bin range.
    """
    def __init__(self,
        n_bins: int,
        bin_type: str = "uniform",
        low: float = 0,
        high: float = 1,
        device: str = 'cuda',
    ):
        super().__init__()
        self.n_bins = n_bins
        self.bin_type = bin_type
        self.low = low
        self.high = high
        self.device = device

        if self.bin_type == "uniform":
            self.thresholds = torch.linspace(self.low, self.high, self.n_bins + 1).to(self.device)
        elif self.bin_type == "normal":
            normal = torch.distributions.Normal(0,1)
            self.thresholds = normal.icdf(torch.linspace(EPS, 1 - EPS, self.n_bins + 1).to(self.device)) #TODO(saumya): check
        else:
            raise ValueError(
                f"Binning type {self.bin_type} not supported in BinTokenizer."
            )

    def forward(self, inputs):
        if self.bin_type == "uniform":
            inputs = torch.clip(inputs, self.low + EPS, self.high - EPS)
        inputs = inputs[..., None]
        token_one_hot = (inputs < self.thresholds[1:]) & (
            inputs >= self.thresholds[:-1]
        ).type(torch.uint8)

        output_tokens = torch.argmax(token_one_hot, axis=-1)
        return output_tokens

    def decode(self, inputs):
        one_hot = F.one_hot(inputs, self.n_bins)
        bin_avgs = (self.thresholds[1:] + self.thresholds[:-1]) / 2
        outputs = torch.sum(one_hot * bin_avgs, axis=-1)
        return outputs


class LowdimObsTokenizer(BinTokenizer):
    """
    Tokenizer for non-spatial observations. Optionally discretizes into bins per dimension (see BinTokenizer).

    Args:
        obs_stack_keys (Sequence[str]): List of non-spatial keys to concatenate & tokenize. Supports regex.
        discretize (bool): If True, discretizes inputs per dimension, see BinTokenizer.
    """
    def __init__(self,
        n_bins: int,
        env_observation_shapes: Dict = {},
        bin_type: str = "uniform",
        low: float = 0,
        high: float = 1,
        obs_stack_keys: Sequence[str] = tuple(),
        discretize: bool = False,
        proper_pad_mask: bool = True,
        device: str = 'cuda',
    ):
        super().__init__(n_bins, bin_type=bin_type, low=low, high=high, device=device)
        
        self.device = device
        self.obs_stack_keys = obs_stack_keys
        self.discretize = discretize
        self.proper_pad_mask = proper_pad_mask
        self.env_observation_shapes = env_observation_shapes

    def forward(self, observations):
        assert self.obs_stack_keys, "Need to specify observation keys to tokenize."
        if len(regex_filter(self.obs_stack_keys, sorted(observations.keys()))) == 0:
            logging.warning(
                f"No observation inputs matching {self.obs_stack_keys} were found."
                "Skipping tokenizer entirely."
            )
            assert self.proper_pad_mask, "Cannot skip unless using proper pad mask."
            return None

        tokenizer_inputs = []
        for o_key in self.obs_stack_keys:
            for key in filter(re.compile(o_key).match, sorted(observations.keys())):
                assert (
                    len(observations[key].shape) == 3 # supports (batch, horizon, feature_dim)
                ), f"Only supports non-spatial inputs but {key} has shape {observations[key].shape}."
                tokenizer_inputs.append(observations[key])
        tokenizer_inputs = torch.concatenate(tokenizer_inputs, axis=-1)
        if self.discretize:
            tokenized_inputs = super().__call__(tokenizer_inputs)
            tokens = F.one_hot(tokenized_inputs, self.n_bins)
        else:
            tokens = tokenizer_inputs[..., None]
        mask = torch.ones(tokens.shape[:-1])
        return TokenGroup(tokens, mask)

class IdentityLowdimObsTokenizer(nn.Module):
    def __init__(self,
        env_observation_shapes: Dict = {},
        min_obs: int = -1e-6,
        max_obs: int = 1e6,
        obs_stack_keys: Sequence[str] = tuple(),
        proper_pad_mask: bool = True,
        device: str = 'cuda',
    ):
        super().__init__()
        self.device = device
        self.obs_stack_keys = obs_stack_keys
        self.proper_pad_mask = proper_pad_mask
        self.env_observation_shapes = env_observation_shapes
        self.min_obs = min_obs
        self.max_obs = max_obs
        
        if len(env_observation_shapes[obs_stack_keys[0]]) == 2: # 2D input
            self.num_tokens = env_observation_shapes[obs_stack_keys[0]][0]
            self.num_features = env_observation_shapes[obs_stack_keys[0]][1]
        elif len(env_observation_shapes[obs_stack_keys[0]]) == 1: 
            self.num_tokens = 1
            self.num_features = env_observation_shapes[obs_stack_keys[0]][0]
        else:
            raise ValueError("Invalid observation shape for IdentityLowdimObsTokenizer")

        self.identity = nn.Identity()
    
    def forward(self, observations, tasks=None):
        assert self.obs_stack_keys, "Need to specify observation keys to tokenize."
        if len(regex_filter(self.obs_stack_keys, sorted(observations.keys()))) == 0:
            logging.warning(
                f"No observation inputs matching {self.obs_stack_keys} were found."
                "Skipping tokenizer entirely."
            )
            assert self.proper_pad_mask, "Cannot skip unless using proper pad mask."
            return None

        tokenizer_inputs = []
        for o_key in self.obs_stack_keys:
            for key in filter(re.compile(o_key).match, sorted(observations.keys())):
                assert (
                    len(observations[key].shape) <= 4 # supports (batch, horizon, feature_dim) or (batch, horizon, num_features, feature_dim)
                ), f"Only supports non-spatial inputs but {key} has shape {observations[key].shape}."
                tokenizer_inputs.append(observations[key])
        tokenizer_inputs = torch.concatenate(tokenizer_inputs, axis=-1)
        
        if len(self.env_observation_shapes[self.obs_stack_keys[0]]) == 1: 
            tokenizer_inputs = tokenizer_inputs.unsqueeze(2)

        # tokenizer_inputs = normalize_low_dim_obs(tokenizer_inputs, self.min_obs, self.max_obs)

        mask = torch.ones(tokenizer_inputs.shape[:-1]).to(self.device)
        return TokenGroup(tokenizer_inputs, mask)