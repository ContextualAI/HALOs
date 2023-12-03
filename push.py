"""
Script to push model to the hugging face hub in the loadable format.

Typical use:

    python push.py -c $MODEL_PATH/config.yaml

where config.yaml is generated during training.
"""
import transformers
import torch
import hydra
from omegaconf import OmegaConf, DictConfig
from typing import Optional, Set
import json, os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', help="saved config file", type=str)


if __name__ == "__main__":
    """Main entry point for evaluating. Validates config, loads model(s), and kicks off worker process(es)."""
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(config))
    exp_name = config.exp_name
    if '+' in exp_name: exp_name = config.exp_name.replace('+', '-')
    repo = f'ContextualAI/{exp_name}'

    tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
    print(f'Loading tokenizer {tokenizer_name_or_path}')
    tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f'Pushing tokenizer to {repo}')
    tokenizer.push_to_hub(repo, use_temp_dir=True, private=True)
    
    print('building policy')
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = transformers.AutoModelForCausalLM.from_pretrained(config.model.name_or_path, low_cpu_mem_usage=True, torch_dtype=policy_dtype)
    
    state_dict = torch.load(os.path.join(config.cache_dir, config.saved_policy), map_location='cpu')
    step, metrics = state_dict['step_idx'], state_dict['metrics']
    print(f'loading pre-trained weights at step {step} from {config.saved_policy} with metrics {json.dumps(metrics, indent=2)}')
    policy.load_state_dict(state_dict['state'])
    print(f'Pushing model to {repo}')
    policy.push_to_hub(repo, use_temp_dir=True, private=True)

    # check that the model can be loaded without problems 
    try:
        print('loading model from hub')
        tokenizer = transformers.AutoTokenizer.from_pretrained(repo)
        policy = transformers.AutoModelForCausalLM.from_pretrained(repo, low_cpu_mem_usage=True, torch_dtype=policy_dtype)
        print('model loaded successfully')
    except:
        print(f'model failed to load from hub {repo}')
