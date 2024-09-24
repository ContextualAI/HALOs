# Copyright (c) 2023 Contextual AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Contains the classes necessary for doing PPO (offline, one-step) with language model.
This code is largely from the TRL library, with some modifications to ensure stability.
"""
import json
import os
import pickle
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from huggingface_hub import hf_hub_download
from transformers import PreTrainedModel, AutoModelForCausalLM
from accelerate.utils import gather_object
from tqdm import tqdm
from typing import Dict, Any, Tuple


class PreTrainedModelWrapper(nn.Module):
    r"""
    A wrapper class around a (`transformers.PreTrainedModel`) to be compatible with the
    (`~transformers.PreTrained`) class in order to keep some attributes and methods of the
    (`~transformers.PreTrainedModel`) class.

    Attributes:
        pretrained_model: (`transformers.PreTrainedModel`)
            The model to be wrapped.
        parent_class: (`transformers.PreTrainedModel`)
            The parent class of the model to be wrapped.
        supported_args: (`list`)
            The list of arguments that are supported by the wrapper class.
    """
    transformers_parent_class = None
    supported_args = None
    supported_modules = ("v_head",)

    def __init__(self, pretrained_model=None, **kwargs):
        super().__init__()
        self.pretrained_model = pretrained_model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Instantiates a new model from a pretrained model from `transformers`. The
        pretrained model is loaded using the `from_pretrained` method of the
        `transformers.PreTrainedModel` class. The arguments that are specific to the
        `transformers.PreTrainedModel` class are passed along this method and filtered
        out from the `kwargs` argument.


        Args:
            pretrained_model_name_or_path (`str` or `transformers.PreTrainedModel`):
                The path to the pretrained model or its name.
            *model_args (`list`, *optional*)):
                Additional positional arguments passed along to the underlying model's
                `from_pretrained` method.
            **kwargs (`dict`, *optional*):
                Additional keyword arguments passed along to the underlying model's
                `from_pretrained` method. 
        """
        if kwargs is not None:
            model_kwargs, pretrained_kwargs = cls._split_kwargs(kwargs)
        else:
            model_kwargs, pretrained_kwargs = {}, {}

        # First, load the pre-trained model using the parent-class
        # either `AutoModelForCausalLM` or `AutoModelForSeq2SeqLM`
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model = cls.transformers_parent_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, **pretrained_kwargs
            )
        elif isinstance(pretrained_model_name_or_path, PreTrainedModel):
            pretrained_model = pretrained_model_name_or_path
        else:
            raise ValueError(
                "pretrained_model_name_or_path should be a string or a PreTrainedModel, "
                f"but is {type(pretrained_model_name_or_path)}"
            )
        # Then, create the full model by instantiating the wrapper class
        model = cls(pretrained_model, *model_args, **model_kwargs)

        # if resume_training, load the state_dict again - this is ok since the
        # state_dict is removed from the model after loading it.
        if isinstance(pretrained_model_name_or_path, str):
            filename = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
            sharded_index_filename = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin.index.json")
            is_shared = False

            if not os.path.exists(filename):
                try:
                    filename = hf_hub_download(pretrained_model_name_or_path, "pytorch_model.bin")
                # sharded
                except:  # noqa
                    if os.path.exists(sharded_index_filename):
                        index_file_name = sharded_index_filename
                    else:
                        index_file_name = hf_hub_download(
                            pretrained_model_name_or_path, "pytorch_model.bin.index.json"
                        )
                    # load json
                    with open(index_file_name, "r") as f:
                        index = json.load(f)
                    # check filename with `v_head` or any known extra module:
                    files_to_download = set()
                    for k, v in index["weight_map"].items():
                        if any([module in k for module in cls.supported_modules]):
                            files_to_download.add(v)
                    is_shared = True

            if is_shared:
                # download each file and add it to the state_dict
                state_dict = {}
                for shard_file in files_to_download:
                    filename = hf_hub_download(pretrained_model_name_or_path, shard_file)
                    state_dict.update(torch.load(filename, map_location="cpu"))
            else:
                state_dict = torch.load(filename, map_location="cpu")

        else:
            state_dict = pretrained_model_name_or_path.state_dict()

        model.post_init(state_dict=state_dict)

        return model

    @classmethod
    def _split_kwargs(cls, kwargs):
        """
        Separate the kwargs from the arguments that we support inside
        `supported_args` and the ones that we don't.
        """
        supported_kwargs = {}
        unsupported_kwargs = {}

        for key, value in kwargs.items():
            if key in cls.supported_args:
                supported_kwargs[key] = value
            else:
                unsupported_kwargs[key] = value

        return supported_kwargs, unsupported_kwargs

    def push_to_hub(self, *args, **kwargs):
        r"""
        Push the pretrained model to the hub. This method is a wrapper around
        `transformers.PreTrainedModel.push_to_hub`. Please refer to the documentation
        of `transformers.PreTrainedModel.push_to_hub` for more information.

        Args:
            *args (`list`, *optional*):
                Positional arguments passed along to the underlying model's
                `push_to_hub` method.
            **kwargs (`dict`, *optional*):
                Keyword arguments passed along to the underlying model's
                `push_to_hub` method.
        """
        raise NotImplementedError

    def save_pretrained(self, *args, **kwargs):
        r"""
        Save the pretrained model to a directory. This method is a wrapper around
        `transformers.PreTrainedModel.save_pretrained`. Please refer to the documentation
        of `transformers.PreTrainedModel.save_pretrained` for more information.

        Args:
            *args (`list`, *optional*):
                Positional arguments passed along to the underlying model's
                `save_pretrained` method.
            **kwargs (`dict`, *optional*):
                Keyword arguments passed along to the underlying model's
                `save_pretrained` method.
        """
        state_dict = kwargs.pop("state_dict", None)
        if state_dict is None:
            state_dict = self.state_dict()
            kwargs["state_dict"] = state_dict

        return self.pretrained_model.save_pretrained(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        r"""
        Return the state_dict of the pretrained model.
        """
        raise NotImplementedError

    def post_init(self, *args, **kwargs):
        r"""
        Post initialization method. This method is called after the model is
        instantiated and loaded from a checkpoint. It can be used to perform
        additional operations such as loading the state_dict.
        """
        raise NotImplementedError


class ValueHead(nn.Module):
    r"""
    The ValueHead class implements a head for autoregressive that returns a scalar for each output token.
    The weights of the value head need to be in FP32.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob

        # some models such as OPT have a projection layer before the word embeddings - e.g. OPT-350m
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        else:
            hidden_size = config.hidden_size

        self.summary = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)    
        )
        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        # detach so that loss isn't backproped through LM
        # upcast since fp32 is important for good value predictions
        hidden_states = hidden_states.detach().to(torch.float32)
        output = self.summary(hidden_states)
        return output


class AutoModelForCausalLMWithValueHead(PreTrainedModelWrapper):
    r"""
    An autoregressive model with a value head in addition to the language model head.

    Class attributes:
        - **transformers_parent_class** (`transformers.PreTrainedModel`) -- The parent class of the wrapped model. This
            should be set to `transformers.AutoModelForCausalLM` for this class.
        - **lm_head_namings** (`tuple`) -- A tuple of strings that are used to identify the language model head of the
            wrapped model. This is set to `("lm_head", "embed_out")` for this class but can be changed for other models
            in the future
        - **supported_args** (`tuple`) -- A tuple of strings that are used to identify the arguments that are supported
            by the `ValueHead` class. Currently, the supported args are:
            - **summary_dropout_prob** (`float`, `optional`, defaults to `None`) -- The dropout probability for the
                `ValueHead` class.
            - **v_head_initializer_range** (`float`, `optional`, defaults to `0.2`) -- The initializer range for the
                `ValueHead` if a specific initialization strategy is selected.
            - **v_head_init_strategy** (`str`, `optional`, defaults to `None`) -- The initialization strategy for the
                `ValueHead`. Currently, the supported strategies are:
                - **`None`** -- Initializes the weights of the `ValueHead` with a random distribution. This is the default
                    strategy.
                - **"normal"** -- Initializes the weights of the `ValueHead` with a normal distribution.

    """
    transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]
    supported_args = (
        "summary_dropout_prob",
        "v_head_initializer_range",
        "v_head_init_strategy",
    )

    def __init__(self, pretrained_model, *args, **kwargs):
        r"""
        Initializes the model.

        Args:
            pretrained_model (`transformers.PreTrainedModel`):
                The model to wrap. It should be a causal language model such as GPT2.
                or any model mapped inside the `AutoModelForCausalLM` class.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `ValueHead` class.
        """
        super().__init__(pretrained_model)
        v_head_kwargs, other_kwargs = self._split_kwargs(kwargs)
        
        if not any(hasattr(self.pretrained_model, attribute) for attribute in self.lm_head_namings):
            raise ValueError("The model does not have a language model head, please use a model that has one.")

        self.v_head = ValueHead(self.pretrained_model.config, **v_head_kwargs)
        self._init_weights(**v_head_kwargs)

    def _init_weights(self, **kwargs):
        r"""
        Initializes the weights of the value head. The default initialization strategy is random.
        Users can pass a different initialization strategy by passing the `v_head_init_strategy` argument
        when calling `.from_pretrained`. Supported strategies are:
        - `normal`: initializes the weights with a normal distribution.

        Args:
            **kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `ValueHead` class. These arguments
                can contain the `v_head_init_strategy` argument as well as the `v_head_initializer_range`
                argument.
        """
        initializer_range = kwargs.pop("v_head_initializer_range", 0.2)
        # random init by default
        init_strategy = kwargs.pop("v_head_init_strategy", None)
        if init_strategy is None:
            # do nothing
            pass
        elif init_strategy == "normal":
            def weights_init(m):
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(mean=0.0, std=initializer_range)
                    m.bias.data.zero_()

            self.summary.apply(weights_init)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        r"""
        Applies a forward pass to the wrapped model and returns the logits of the value head.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        """
        kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples
        kwargs["past_key_values"] = past_key_values

        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss

        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()

        value = self.v_head(last_hidden_state).squeeze(-1)

        return (lm_logits, loss, value)

    def generate(self, *args, **kwargs):
        r"""
        A simple wrapper around the `generate` method of the wrapped model.
        Please refer to the [`generate`](https://huggingface.co/docs/transformers/internal/generation_utils)
        method of the wrapped model for more information about the supported arguments.

        Args:
            *args (`list`, *optional*):
                Positional arguments passed to the `generate` method of the wrapped model.
            **kwargs (`dict`, *optional*):
                Keyword arguments passed to the `generate` method of the wrapped model.
        """
        return self.pretrained_model.generate(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        r"""
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
       
        v_head_state_dict = self.v_head.state_dict(*args, **kwargs)
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f"v_head.{k}"] = v
        return pretrained_model_state_dict

    def push_to_hub(self, *args, **kwargs):
        setattr(self.pretrained_model, "v_head", self.v_head)
        return self.pretrained_model.push_to_hub(*args, **kwargs)

    def post_init(self, state_dict):
        r"""
        We add the state dictionary of the value head to the state dictionary of the wrapped model
        by prepending the key with `v_head.`. This function removes the `v_head.` prefix from the
        keys of the value head state dictionary.
        """
        for k in list(state_dict.keys()):
            if "v_head." in k:
                state_dict[k.replace("v_head.", "")] = state_dict.pop(k)
        self.v_head.load_state_dict(state_dict, strict=False)
        del state_dict

    def resize_token_embeddings(self, vocab_size):
        """
        Resize the vocabulary size of the language model.
        """
        self.pretrained_model.resize_token_embeddings(vocab_size)

    @classmethod
    def from_pretrained(cls, load_from, *args, **kwargs):
        pretrained_model = AutoModelForCausalLM.from_pretrained(load_from, *args, **kwargs)
        model_with_value_head = cls(pretrained_model)

        if os.path.exists(os.path.join(load_from, "v_head.pt")):
            state_dict = torch.load(os.path.join(load_from, "v_head.pt"), map_location='cpu')
            model_with_value_head.v_head.load_state_dict(state_dict['state'])

        return model_with_value_head


class ReferenceModelWrapper(nn.Module):
    """
    A wrapper around the reference model that precomputes the logprobs and saves them in a local dict,
    after which the reference model and accelerator are deleted to save GPU memory.

    Note that the wrapper returns the logprobs of the sequence, not the logits (like the underlying
    model would).
    """
    def __init__(self, reference_accelerator, reference_model, tokenizer, config, iterators):
        """
        Args:
            - reference_accelerator: accelerator that should be used for caching (different from main accelerator)
            - reference_model: reference model
            - tokenizer: instance of AutoTokenizer
            - config: Hydra config
            - iterators: list of iterators, each instantiated by calling iter on a dataloader.DataLoader
        """
        super().__init__()
        self.reference_accelerator = reference_accelerator
        self.num_processes = reference_accelerator.num_processes 
        self.reference_model = reference_model
        self.reference_dtype = getattr(torch, config.model.reference_dtype)
        self.tokenizer = tokenizer
        self.iterators = iterators
        self.config = config

        self.reference_model, self.tokenizer, self.iterators = reference_accelerator.prepare(
            self.reference_model,
            self.tokenizer,
            self.iterators
        )

        self.logprobs = {}

        if config.load_reference_logprobs:
            self.logprobs = pickle.load(open(config.load_reference_logprobs, 'rb'))
        else:
            self._precompute_log_probs()

        self._free_memory() # delete the reference model and the accelerator to free up memory

    def _remove_padding(self, token_ids):
        return [ t for t in token_ids if t not in [ self.tokenizer.bos_token_id, self.tokenizer.pad_token_id, self.tokenizer.eos_token_id ]]

    def _precompute_log_probs(self):
        """
        Calculate the log probabilities of every input-output sequence in every iterator in self.iterators.
        Save these in self.logprobs as the values, where the key is the sequence of token ids (as a tuple, not a list).
        These are then saved to 'cached_ref_logprobs.json' in the run directory.
        """
        self.reference_model.eval()
        logprobs = []
        example_counter = 0
        
        pbar = tqdm(disable=not self.reference_accelerator.is_local_main_process, dynamic_ncols=True)
        pbar.set_description(f"Caching logprobs for reference model")

        with torch.no_grad():
            for data_iterator in self.iterators:
                for batch in data_iterator:
                    batch = {k: v.to(self.reference_accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    
                    # should be 'target', 'KL' for KTO and just 'target' for everything els
                    prefixes = [ k[:k.index('_')] for k in batch if k.endswith('_combined_input_ids') ]

                    for prefix in prefixes:
                        logits = self.reference_model(
                            batch[f'{prefix}_combined_input_ids'], 
                            attention_mask=batch[f'{prefix}_combined_attention_mask']
                        ).logits.to(self.reference_dtype)

                        batch_logprobs = self._compute_log_probs(logits, batch[f'{prefix}_labels']).tolist()

                        for k,v in zip(batch[f'{prefix}_combined_input_ids'].tolist(), batch_logprobs):
                            logprobs.append((tuple(self._remove_padding(k)), v))
                    
                    example_counter += self.config.model.batch_size
                    pbar.update(self.config.model.batch_size)
                    pbar.set_postfix(examples=example_counter)

        self.reference_accelerator.wait_for_everyone()
        pbar.close()

        # Gather dictionaries from all processes
        gathered_logprobs = gather_object(logprobs)
        self.logprobs = dict(gathered_logprobs)

        if self.reference_accelerator.is_main_process:
            with open(os.path.join(self.config.local_run_dir, 'cached_reference_logprobs.pkl'), 'wb') as f:
                pickle.dump(self.logprobs, f) 

    def _compute_log_probs(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the token-level log probabilities of the given labels under the given logits."""
        # ignoring vocab size, batch size x length should be equal
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != -100)

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == -100] = 0

        distribution_logps = logits.float().log_softmax(-1)
        per_token_logps = torch.gather(distribution_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return per_token_logps * loss_mask
    
    def _free_memory(self):
        del self.reference_accelerator
        del self.reference_model
        torch.cuda.empty_cache()

    def forward(self, input_ids: Dict[str, Any], *args, **kwargs) -> torch.Tensor:
        """
        Return the cached log probabilities for the given input ids.
        """
        batch_logprobs = [ torch.Tensor(self.logprobs[tuple(self._remove_padding(k))]) for k in input_ids.tolist() ]
        batch_logprobs = pad_sequence(batch_logprobs, batch_first=True, padding_value=0)
        return batch_logprobs

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def eval(self):
        pass # pass through, allows wrapper to be treated like a neural network