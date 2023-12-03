import os
import getpass
from datetime import datetime
import torch
import random
import numpy as np
import torch.distributed as dist
import inspect
import importlib.util
import socket
import os
from typing import Dict, Union, Type, List
from collections.abc import Mapping


def deepcopy_fsdp_models(src, tgt):
    """Given two models src and tgt, copy every parameter from the src to the tgt model."""
    with torch.no_grad():
        src_params = { k: v for k,v in src.named_parameters() }
        tgt_params = { k: v for k,v in tgt.named_parameters() }

        for k in tgt_params:
            if k in src_params:
                tgt_params[k].data.copy_(src_params[k].data.detach())
            else:
                rank0_print(f"{k} not found")


def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0)) # bind to all interfaces and use an OS provided port
        return s.getsockname()[1] # return only the port number


def get_remote_file(remote_path, local_path=None):
    hostname, path = remote_path.split(':')
    local_hostname = socket.gethostname()
    if hostname == local_hostname or hostname == local_hostname[:local_hostname.find('.')]:
        return path
    
    if local_path is None:
        local_path = path
    # local_path = local_path.replace('/scr-ssd', '/scr')    
    if os.path.exists(local_path):
        return local_path
    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)

    print(f'Copying {hostname}:{path} to {local_path}')
    os.system(f'scp {remote_path} {local_path}')
    return local_path


def rank0_print(*args, **kwargs):
    """Print, but only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def on_rank0():
    return (not dist.is_initialized()) or (dist.get_rank() == 0)


def slice_and_move_batch_for_device(batch: Dict, rank: int, world_size: int, device: str) -> Dict:
    """Slice a batch into chunks, and move each chunk to the specified device."""
    chunk_size = len(list(batch.values())[0]) // world_size
    start = chunk_size * rank
    end = chunk_size * (rank + 1)
    sliced = {k: v[start:end] for k, v in batch.items()}
    on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sliced.items()}
    return on_device


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)


def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False, token_level: bool = False):
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
        token_level: If true, return the token-level log probabilities (do not aggregate across tokens)

    Returns:
        The relevant log probabilities. Of shape (batch_size,) by default and shape (batch size, sequence length) if token_level.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0
    distribution_logps = logits.log_softmax(-1)

    per_token_logps = torch.gather(distribution_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if token_level: 
        return (per_token_logps * loss_mask)
    elif average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


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


def flatten_dict(nested, sep="/"):
    """Flatten dictionary and concatenate nested keys with separator."""

    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, Mapping):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v

    flat = {}
    rec(nested, "", flat)
    return flat


def all_gather_if_needed(values: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Gather and stack/cat values from all processes, if there are multiple processes."""
    if world_size == 1:
        return values

    device = torch.device('cuda', rank)
    all_values = [torch.empty_like(values).to(device) for _ in range(world_size)]
    dist.all_gather(all_values, values)
    cat_function = torch.cat if values.dim() > 0 else torch.stack
    return cat_function(all_values, dim=0)


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


def get_block_class_from_model(model: torch.nn.Module, block_class_name: str) -> torch.nn.Module:
    """Get the class of a block from a model, using the block's class name."""
    for module in model.modules():
        if module.__class__.__name__ == block_class_name:
            return module.__class__
    raise ValueError(f"Could not find block class {block_class_name} in model {model}")


def get_block_class_from_model_class_and_block_name(model_class: Type, block_class_name: str) -> Type:
    filepath = inspect.getfile(model_class)
    assert filepath.endswith('.py'), f"Expected a .py file, got {filepath}"
    assert os.path.exists(filepath), f"File {filepath} does not exist"
    assert "transformers" in filepath, f"Expected a transformers model, got {filepath}"

    module_name = filepath[filepath.find('transformers'):].replace('/', '.')[:-3]
    print(f"Searching in file {filepath}, module {module_name} for class {block_class_name}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the class dynamically
    class_ = getattr(module, block_class_name)
    print(f"Found class {class_} in module {module_name}")
    return class_


def init_distributed(rank: int, world_size: int, master_addr: str = 'localhost', port: int = 12355, backend: str = 'nccl'):
    print(rank, 'initializing distributed')
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
