# Copyright (c) 2023 Contextual AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Main script for training.

Sample use is:

python train.py loss=ppo model=llama30b datasets=[shp,hh,oasst] exp_name=archangel_sft+ppo_llama30b mode=train \
     ++cache_dir=/data/models/archangel ++model.load_from=archangel_sft_llama30b/LATEST/policy.pt

where
- loss should have a file under config/loss that specifies the trainer in trainers.py and dataloader in dataloader.py
- model should have a file under config/model
- datasets is a list of datasets, each of which has a get_{name} function in dataloader.py
- exp_name is the experiment name (on WANDB); model will be saved to the cache_dir/exp_name
- model.load_from should be used for aligning a model that has already been finetuned

Remember to allocate enough RAM before running this (you need aroundd 800 GB for Llama-13B).
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
from utils import disable_dropout
import os
import hydra
from omegaconf import OmegaConf, DictConfig
import trainers
import wandb
import json
from typing import Optional, Set
import resource
from models import AutoModelForCausalLMWithValueHead
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import numpy as np
import random
import dataloader
import gc
from utils import delete_dict
from accelerate import Accelerator, DistributedDataParallelKwargs


def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and starts training."""
    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")
    
    set_seed(config.seed)
    
    # Initialize Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        project_dir=config.local_run_dir,
        gradient_accumulation_steps=config.model.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs]
    )

    if accelerator.state.fsdp_plugin is not None:
        accelerator.state.fsdp_plugin.transformer_layer_cls_to_wrap = config.model.block_name

    if config.eval_every % config.model.batch_size != 0:
        accelerator.print('WARNING: eval_every must be divisible by batch_size')
        accelerator.print('Setting eval_every to', config.eval_every - config.eval_every % config.model.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.model.batch_size

    if config.saved_policy is None:
        config.saved_policy = f"{config.local_run_dir}/LATEST/policy.pt"

    accelerator.print(OmegaConf.to_yaml(config))

    if accelerator.is_main_process:
        os.makedirs(config.local_run_dir, exist_ok=True)
        accelerator.print("Making experiment directory", config.local_run_dir)

        os.environ['WANDB_CACHE_DIR'] = config.cache_dir
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=config.cache_dir,
            name=config.exp_name,
        )
    
        config_path = os.path.join(config.local_run_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config, f)

        accelerator.print('=' * 80)
        accelerator.print(f'Writing to {config.local_run_dir}')
        accelerator.print('=' * 80)

    # Prepare tokenizer
    tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
    accelerator.print(f'Loading tokenizer {tokenizer_name_or_path}')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    control_tokens = list(config.loss.get("control_tokens", {}).values())
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": control_tokens})

    # Building policy
    policy_cls = globals()[config.loss.policy_hf_model_class]
    policy_kwargs = {'torch_dtype': getattr(torch, config.model.policy_dtype)}
    policy_path = config.model.load_from or config.model.name_or_path
    accelerator.print(f'Loading policy from {policy_path}')
    policy = policy_cls.from_pretrained(policy_path, **policy_kwargs)

    if num_added:
        policy.resize_token_embeddings(len(tokenizer))

    # Building reference
    if config.loss.use_reference_model:
        reference_cls = globals()[config.loss.reference_hf_model_class]
        reference_kwargs = {'torch_dtype': getattr(torch, config.model.reference_dtype)}
        reference_path = config.model.load_from or config.model.name_or_path
        accelerator.print(f'Loading reference model from {reference_path}')
        reference_model = reference_cls.from_pretrained(reference_path, **reference_kwargs)

        if num_added:
            reference_model.resize_token_embeddings(len(tokenizer))
    else:
        reference_model = None

    # Create data loaders
    accelerator.print(f'Loading data')
    data_loader_class = getattr(dataloader, config.loss.dataloader)
    data_iterator_kwargs = dict(
        max_length=config.model.max_length,
        max_prompt_length=config.model.max_prompt_length,
        human_prefix=config.human_prefix,
        human_suffix=config.human_suffix,
        assistant_prefix=config.assistant_prefix,
        assistant_suffix=config.assistant_suffix,
        seed=config.seed,
        frac_unique_desirable=config.frac_unique_desirable,
        frac_unique_undesirable=config.frac_unique_undesirable,
        control_tokens=config.loss.get("control_tokens", {}),
    )
    train_iterator = data_loader_class(
        config.datasets, 
        tokenizer,
        split='train',
        batch_size=config.model.batch_size,
        n_epochs=config.n_epochs,
        n_examples=config.n_examples,
        **data_iterator_kwargs
    )
    eval_iterator = data_loader_class(
        config.datasets, 
        tokenizer,
        split='test',
        batch_size=config.model.eval_batch_size,
        n_examples=config.n_eval_examples, 
        n_epochs=(1 if config.n_eval_examples is None else None),
        **data_iterator_kwargs
    )
    
    # Initialize trainer
    TrainerClass = getattr(trainers, config.loss.trainer)
    trainer = TrainerClass(
        tokenizer, 
        config, 
        train_iterator, 
        eval_iterator,
        accelerator, 
        policy, 
        reference_model=reference_model
    )

    trainer.train()
    trainer.save(os.path.join(config.local_run_dir, 'FINAL'))

@hydra.main(version_base=None, config_path="config", config_name="config")
def hydra_main(config: DictConfig):
    main(config)

if __name__ == '__main__':
    hydra_main()