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
from jinja2 import Template, Environment, FileSystemLoader
from io import BytesIO
from huggingface_hub import HfApi

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

    env = Environment(loader=FileSystemLoader("assets/"))
    template = env.get_template("model_readme.jinja")
    output = template.render(model=config.model.name_or_path, loss=config.loss.name.upper(), thumbnail="https://gist.github.com/assets/29318529/fe2d8391-dbd1-4b7e-9dc4-7cb97e55bc06")
    print(f'Pushing model card to {repo}')
    with open('assets/temp.md', 'w') as f:
        f.write(output)
    api = HfApi()
    api.upload_file(
        path_or_fileobj='assets/temp.md',
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="model",
    )
    os.remove('assets/temp.md')

    tokenizer_name_or_path = config.local_run_dir
    print(f'Loading tokenizer at {tokenizer_name_or_path}')
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f'Pushing tokenizer to {repo}')
    tokenizer.push_to_hub(repo, use_temp_dir=True, private=True)
    
    print('building policy')
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = transformers.AutoModelForCausalLM.from_pretrained(config.model.name_or_path, low_cpu_mem_usage=True, torch_dtype=policy_dtype)
    # note that models were only resized for csft before saving
    # important because number of tokens in pretrained tokenizer is different from model.config.vocab_size, 
    # so resizing at eval will throw an error if not resized before training
    if config.loss.name == 'csft':
        policy.resize_token_embeddings(len(tokenizer)) # model being loaded should already be trained with additional tokens for this to be valid

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
