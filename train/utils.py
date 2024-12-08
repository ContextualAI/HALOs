# Copyright (c) 2023 Contextual AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from datetime import datetime
import torch
import torch.distributed as dist
import os
import json
from typing import Dict, Union, Type, List, TextIO
from collections.abc import Mapping

import huggingface_hub
from huggingface_hub import HfApi, HfFolder
from huggingface_hub.utils import LocalTokenNotFoundError


class StreamingJSONWriter:
    """Writes JSON arrays to a file in a streaming fashion."""
    def __init__(self, file: TextIO):
        self.file = file
        self.is_first = True
        self.file.write('[\n')
    
    def write_item(self, item: Dict):
        """Write a single item to the JSON array."""
        if not self.is_first:
            self.file.write(',\n')
        json.dump(item, self.file, indent=2)
        self.is_first = False
        # Flush after each write to ensure immediate disk writing
        self.file.flush()
    
    def close(self):
        """Close the JSON array and the file."""
        self.file.write('\n]')
        self.file.flush()


def get_base_model_state_dict_from_peft(peft_state_dict, lora_alpha, lora_r):
    """
    Return the state dict for the base model given the state dict for a lora-wrapped 
    AutoModelForCausalLM, merging the lora weights as needed.

    This helper is needed because automated weight merging does not work with FSDP.
    """
    state_dict = {}

    for name in peft_state_dict.keys():
        if 'lora_A' in name:
            base_param_name = name.replace('lora_A.default', 'base_layer')
            
            lora_a = peft_state_dict[name]
            lora_b = peft_state_dict[name.replace('lora_A', 'lora_B')]
            scaling = lora_alpha / lora_r

            new_name = name.replace('lora_A.default.', '').replace('base_model.model.', '')
            state_dict[new_name] = peft_state_dict[base_param_name] + (lora_b @ lora_a) * scaling
        elif 'lora_B' in name or 'base_layer' in name:
            continue
        else:
            new_name = name.replace('base_model.model.', '')
            state_dict[new_name] = peft_state_dict[name]

    return state_dict


def set_offline_if_needed():
    try:
        token = HfFolder.get_token()
        api = HfApi()
        api.whoami(token)

        os.environ['HF_DATASETS_OFFLINE'] = '0'
        os.environ['HF_HUB_OFFLINE'] = '0'
    except huggingface_hub.errors.OfflineModeIsEnabled:
        print("No valid token found. Falling back to offline mode.")
        os.environ['HF_DATASETS_OFFLINE'] = '1' 
        os.environ['HF_HUB_OFFLINE'] = '1'


def rank0_print(*args, **kwargs):
    """Print, but only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def on_rank0():
    return (not dist.is_initialized()) or (dist.get_rank() == 0)


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)


def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    return variance


def rowwise_product(mat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Calculate the row-wise product over all the elements that have not been masked out.

    Args:
        mat: tensor of shape (batch_size, sequence length)
        mask: tensor of shape (batch_size, sequence length) 

    Returns:
        Matrix of batch size. 
    """
    mat = mat.clone()
    indices = (mask == 0).long().nonzero()
    mat[indices[:,0], indices[:,1]] = 1
    return mat.prod(dim=1)


def entropy_from_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Calculate entropy from logits.
    
    Args:
        logits: tensor of shape (batch_size, sequence length, vocab)
        mask: tensor of shape (batch_size, sequence length)
    
    Returns:
        The average tokenwise entropy across all non-masked tokens (of shape (1,)).
    """
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = masked_mean(torch.logsumexp(logits, axis=-1) - torch.sum(pd * logits, axis=-1), mask)
    return entropy


def formatted_dict(d: Dict) -> Dict:
    """Format a dictionary for printing."""
    return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}
    

def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def delete_dict(d: Dict):
    """Delete all items inside the dict."""
    for k in list(d.keys()):
        del d[k]


def delete_dicts(*dicts: Dict):
    """Delete all items inside the given dictionaries."""
    for d in dicts:
        for k in list(d.keys()):
            del d[k]


def print_gpu_memory(rank: int = None, message: str = ''):
    """Print the amount of GPU memory currently allocated for each GPU."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            device = torch.device(f'cuda:{i}')
            allocated_bytes = torch.cuda.memory_allocated(device)
            if allocated_bytes == 0:
                continue
            print('*' * 40)
            print(f'[{message} rank {rank} ] GPU {i}: {allocated_bytes / 1024**2:.2f} MB')
        print('*' * 40)
