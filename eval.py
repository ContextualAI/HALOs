# Copyright (c) 2023 Contextual AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Main script for running evals. This will run an eval according to the specified config, which should be a YAML file generated during training.

For sampling, do something like:
    python eval.py -c $MODEL_PATH/config.yaml -m sample -n 512 -b 32    

For pure evaluation, do something like:
    python eval.py -c $MODEL_PATH/config.yaml   
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from utils import disable_dropout, init_distributed, get_open_port, rank0_print
import os
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
import json
import socket
from typing import Optional, Set
from trainers import BasicTrainer
import dataloader
import numpy as np
import random

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', help="saved config file", type=str)
parser.add_argument('--mode', '-m', help="either sample or eval", type=str, default='sample')
parser.add_argument('--use_reference_model', '-u', help="load reference model", type=bool, default=True)
parser.add_argument('--load_from_scratch', '-s', help="don't load finetuned reference model, just use the default one from Huggingface (for SFT+KTO, SFT+DPO)", type=bool, default=True)
parser.add_argument('--num_samples', '-n', help="number of samples", type=int, default=None, required=False)
parser.add_argument('--eval_batch_size', '-b', help="eval batch size", type=int, default=None, required=False)

if __name__ == "__main__":
    """Main entry point for evaluating. Validates config, loads model(s), and kicks off worker process(es)."""
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(config))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    print('=' * 80)
    print(f'Writing to {socket.gethostname()}:{config.local_run_dir}')
    print('=' * 80)

    # purely inference, so put as much as possible onto the first gpu
    model_kwargs = {'device_map': "balanced_low_0"} 

    if config.saved_policy is None:
        config.saved_policy = f"{config.exp_name}/LATEST/policy.pt"

    print('building policy')
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, low_cpu_mem_usage=True, use_flash_attention_2=config.model.use_flash_attention, torch_dtype=policy_dtype, **model_kwargs)
    disable_dropout(policy)
    state_dict = torch.load(os.path.join(config.cache_dir, config.saved_policy), map_location='cpu')
    step, metrics = state_dict['step_idx'], state_dict['metrics']
    print(f'loading pre-trained weights for policy at step {step} from {config.saved_policy} with metrics {json.dumps(metrics, indent=2)}')
    policy.load_state_dict(state_dict['state'])
    
    if config.loss.use_reference_model or args.use_reference_model:
        print('building reference model')
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, low_cpu_mem_usage=True, use_flash_attention_2=config.model.use_flash_attention, torch_dtype=reference_model_dtype, **model_kwargs)
        disable_dropout(reference_model)

        if (not args.load_from_scratch) and (config.model.load_from is not None):
            state_dict = torch.load(os.path.join(config.cache_dir, config.model.load_from), map_location='cpu')
            step, metrics = state_dict['step_idx'], state_dict['metrics']
            print(f'loading pre-trained weights for reference at step {step} from {config.model.load_from} with metrics {json.dumps(metrics, indent=2)}')
            reference_model.load_state_dict(state_dict['state'])
    else:
        reference_model = None
        
    tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
    print(f'Loading tokenizer {tokenizer_name_or_path}')
    tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data_loader_class = getattr(dataloader, config.loss.dataloader)
    data_iterator_kwargs = dict(
        max_length=config.model.max_length,
        max_prompt_length=config.model.max_prompt_length,
        seed=config.seed,
    )

    if args.mode == 'sample':
        print(f'Loading dataloader')
        if args.num_samples is not None:
            config.n_samples = args.num_samples

        if args.eval_batch_size is not None:
            config.model.eval_batch_size = args.eval_batch_size

        # use the SFT dataloader because we don't want to repeat prompts
        # and bc data ordering is different in paired vs unpaired data loaders
        eval_iterator = dataloader.SFTDataLoader(
            config.datasets, 
            tokenizer,
            split='test',
            batch_size=config.model.eval_batch_size,
            n_examples=config.n_samples,
            max_prompt_count=1,
            **data_iterator_kwargs
        )

        trainer = BasicTrainer(tokenizer, config, None, eval_iterator, policy, reference_model=reference_model)
        samples = trainer.sample() 
        json.dump(samples, open(f'samples/{config.exp_name}.json', 'w'), indent=2)
    elif args.mode == 'eval':
        print(f'Loading dataloader')
        eval_iterator = data_loader_class(
            config.datasets, 
            tokenizer,
            split='test',
            batch_size=config.model.eval_batch_size,
            n_examples=config.n_eval_examples,
            n_epochs=(1 if config.n_eval_examples is None else None),
            **data_iterator_kwargs
        )

        trainer = BasicTrainer(tokenizer, config, None, eval_iterator, policy, reference_model=reference_model)
        results = trainer.eval() 
        rank0_print(results)
    else:
        raise Exception("mode is neither sample nor eval")