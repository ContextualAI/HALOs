"""
A script for creating feedback datasets from data that was sampled with train.sample.
Supports both reward model-based and API-based labeling.

Sample usage with Accelerate for reward model:
accelerate launch --config_file accelerate_config/fsdp_4gpu.yaml --main_process_port 29500 \ 
    -m train.label --reward_model_path models/llama3-8B-bt/FINAL outputs.json reward_data.json --feedback_type pairwise

Sample usage for API labeling (accelerate not needed):
python -m train.label --api_type openai --api_key YOUR_KEY --api_model gpt-4 \
    --label_prompt "Rate this response's quality from 0 to 1:" \
    outputs.json reward_data.json --feedback_type binary

Sample usage for pairwise labeling of two sample files:
python -m train.label --second_samples_path baseline_samples.json --api_type openai --api_key YOUR_KEY \
    outputs.json reward_data.json --feedback_type pairwise
"""

import argparse
import json
import os
import torch
import random
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from typing import List, Dict, Optional, Union
import re
from .utils import StreamingJSONWriter, get_api_completion
from .dataloader import SFTDataLoader
from collections import defaultdict
import openai
import asyncio
import torch.distributed as dist


def process_batch_with_reward_model(
    batch: Dict,
    reward_model: AutoModelForSequenceClassification,
    accelerator: Accelerator
) -> List[Dict]:
    """Process a batch through the reward model using the already tokenized sequences."""
    
    reward_model.eval()
    with torch.no_grad():
        outputs = reward_model(
            input_ids=batch['target_combined_input_ids'],
            attention_mask=batch['target_combined_attention_mask']
        )
        reward_scores = outputs.logits[:, 1]
    
    processed_samples = []
    for i in range(len(batch['prompt'])):
        sample = {
            'prompt': batch['prompt'][i],
            'instruction': batch['original_prompt'][i] if 'original_prompt' in batch else batch['prompt'][i][0]['content'],
            'output': batch['target'][i], 
            'reward': reward_scores[i].item(),
            'prompt_id': batch['prompt_id'][i],
        }
        processed_samples.append(sample)

    # Gather samples from all processes
    gathered_samples = [None] * accelerator.num_processes
    dist.all_gather_object(gathered_samples, processed_samples)
        
    # Return gathered samples from all processes, not just main process
    # This ensures samples from all processes are included
    if accelerator.is_main_process:
        return [item for sublist in gathered_samples for item in sublist]
    
    return []


async def process_samples_with_api(
    samples: Dict,
    client: openai.AsyncOpenAI,
    system_prompt: str,
    label_prompt: str,
    model: str,
    batch_size: int = 10
) -> List[Dict]:
    """Process all samples through the API."""
    scores = []
    processed_samples = []
    total_samples = len(samples)
    
    # Create progress bar for overall progress
    pbar = tqdm(total=total_samples, desc="Processing samples through API")

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        tasks = []

        for sample in batch:
            prompt = f"{label_prompt}\n\nPrompt: {sample['prompt'][0]['content']}\nResponse: {sample['output']}"
            tasks.append(get_api_completion(client, system_prompt, prompt, model))

        batch_scores = await asyncio.gather(*tasks)
        scores.extend(batch_scores)

        # Update progress bar
        pbar.update(len(batch))
        pbar.set_postfix({'Batch': f'{i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}',
                         'Samples': f'{min(i + batch_size, total_samples)}/{total_samples}'})
    
    pbar.close()
    
    for sample, score in zip(samples, scores):
        try:
            score = float(score)
        except ValueError:
            print(f"Warning: Could not parse API response as float. skipping prompt f{sample['prompt_id']}")
            continue
                
        processed_sample = sample.copy()
        processed_sample['reward'] = score
        # since a dataloader isn't used, the output has to be explicitly formatted
        processed_sample['output'] = [{ "role" : "assistant", "content" : processed_sample['output']}]
        processed_samples.append(processed_sample)
            
    return processed_samples


def convert_to_binary_feedback(samples: List[Dict], threshold: float=0.5) -> List[Dict]:
    """Convert samples to binary feedback format."""
    feedback = []
    for sample in samples:
        feedback_item = {
            'prompt_id': sample['prompt_id'],
            'prompt': sample['prompt'],
            'output': sample['output'],
            'label': 1 if sample['reward'] > threshold else 0,
            'reward': sample['reward'],
            'type': 'binary_feedback',
        }
        feedback.append(feedback_item)
    return feedback


def convert_to_pairwise_feedback(samples: List[Dict], seed: int, threshold: float=0, mode: str='train') -> List[Dict]:
    """Convert samples to pairwise feedback format."""
    random.seed(seed)
    
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample['prompt_id']].append(sample)
    print(len(grouped))
    
    feedback = []
    for prompt_id, group in grouped.items():
        if len(group) < 2 and mode == "train":
            continue
        
        group.sort(key=lambda x: x['reward'], reverse=True)
        
        for i in range(len(group) - 1):
            higher_reward, lower_reward = group[i], group[i + 1]
            
            if mode == "train":
                if higher_reward['reward'] == lower_reward['reward']:
                    continue

                if threshold and abs(higher_reward['reward'] - lower_reward['reward']) < threshold:
                    continue
            
            if random.random() < 0.5:
                sample_A, sample_B = higher_reward, lower_reward
                label = 1
            else:
                sample_A, sample_B = lower_reward, higher_reward
                label = 0
                
            feedback_item = {
                'prompt_id': prompt_id,
                'prompt': sample_A['prompt'],
                'output_A': sample_A['output'],
                'output_B': sample_B['output'],
                'label': label,
                'reward_A': sample_A['reward'],
                'reward_B': sample_B['reward'],
                'reward_difference': abs(sample_A['reward'] - sample_B['reward']),
                'type': 'pairwise_feedback',
            }
            feedback.append(feedback_item)
    
    return feedback


async def main(args):
    accelerator = Accelerator()
    # Load samples
    with open(args.samples_path, 'r') as f:
        samples = json.load(f)

    # If second samples file provided, merge them for pairwise comparison
    if args.second_samples_path:
        with open(args.second_samples_path, 'r') as f:
            second_samples = json.load(f)
        for sample in second_samples:
            sample['sample_id'] += 1
        samples.extend(second_samples)
    
    if args.api_type:
        # API-based labeling path
        print(f"Processing {len(samples)} samples using {args.api_type} API")
        
        if args.api_type == "openai":
            if not args.api_key:
                raise ValueError("API key must be provided when using API labeling")
            client = openai.AsyncOpenAI(api_key=args.api_key)
        else:
            raise ValueError(f"Unsupported API type: {args.api_type}")
            
        processed_samples = await process_samples_with_api(
            samples, client, args.system_prompt, args.label_prompt,
            args.api_model, args.batch_size
        )
    else:
        if accelerator.is_main_process:
            print(f"Loading reward model from {args.reward_model_path}")

        reward_model = AutoModelForSequenceClassification.from_pretrained(
            args.reward_model_path,
            local_files_only=True,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.reward_model_path,
            local_files_only=True,
            trust_remote_code=True
        )
        
        # Initialize the dataloader for reward model processing
        if args.second_samples_path:
            names = [args.samples_path, args.second_samples_path]
        else:
            names = [args.samples_path]
        
        dataloader = SFTDataLoader(
            dataset_names=names,
            tokenizer=tokenizer,
            process_index=accelerator.process_index,
            num_processes=accelerator.num_processes,
            split='train',
            microbatch_size=(args.batch_size // accelerator.num_processes),
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,
            n_epochs=1,
            seed=args.seed,
        )
        dataloader, reward_model = accelerator.prepare(dataloader, reward_model)
        
        # Initialize distributed setup if not already done
        if accelerator.num_processes > 1 and not dist.is_initialized():
            accelerator.wait_for_everyone()
            
            if accelerator.is_main_process:
                dist.init_process_group(backend='nccl')
        
        processed_samples = []
        for batch in tqdm(dataloader, disable=not accelerator.is_main_process):
            batch = {k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch_samples = process_batch_with_reward_model(batch, reward_model, accelerator)
            
            if accelerator.is_main_process:
                processed_samples.extend(batch_samples)
        
        # Wait for all processes to complete
        accelerator.wait_for_everyone()

    # Set up output writer
    if accelerator.is_main_process:
        accelerator.print(f"Writing feedback to {args.output_path}")
        with open(args.output_path, 'w') as f:
            writer = StreamingJSONWriter(f)
        
            # Process and write feedback
            if args.feedback_type == 'binary':
                feedback = convert_to_binary_feedback(processed_samples, threshold=args.threshold)
            elif args.feedback_type == 'pairwise' and args.second_samples_path is None:
                feedback = convert_to_pairwise_feedback(processed_samples, args.seed, threshold=args.threshold)
            elif args.feedback_type == 'pairwise' and args.second_samples_path:
                feedback = convert_to_pairwise_feedback(processed_samples, args.seed, threshold=args.threshold, mode='eval')
            else:
                feedback = processed_samples
                for x in feedback:
                    x['type'] = 'scalar_feedback'
                
            # Add split information and write
            for item in feedback:
                item['split'] = args.split
                writer.write_item(item)
            
            writer.close()
    
    # Clean up distributed training resources if using reward model
    accelerator.end_training()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label samples using either a reward model or API")
    
    # Input/output arguments
    parser.add_argument("samples_path", type=str, help="Path to the JSON file containing samples")
    parser.add_argument("output_path", type=str, help="Path to save the feedback file")
    parser.add_argument("--second_samples_path", type=str, default=None, help="Optional second samples file for pairwise comparison")
    
    # Labeling method arguments
    labeling_group = parser.add_mutually_exclusive_group(required=True)
    labeling_group.add_argument("--reward_model_path", type=str, help="Path to the reward model")
    labeling_group.add_argument("--api_type", type=str, choices=['openai'], help="Type of API to use for labeling")
    
    # API-specific arguments
    parser.add_argument("--api_key", type=str, help="API key for the chosen API service")
    parser.add_argument("--api_model", type=str, default="gpt-4", help="Model to use for API labeling")
    parser.add_argument("--system_prompt", type=str, 
                      default="You are a helpful assistant that rates the quality of responses. Provide only a number between 0 and 1.",
                      help="System prompt for API labeling")
    parser.add_argument("--label_prompt", type=str, 
                     default="Rate this response's quality from 0 to 1:",
                     help="Prompt template for API labeling")
    
    # Processing arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
    parser.add_argument("--max_length", type=int, default=1024, 
                        help="Maximum sequence length for input")
    parser.add_argument("--max_prompt_length", type=int, default=512, 
                        help="Maximum prompt length for input")
    parser.add_argument("--feedback_type", type=str, choices=['binary', 'pairwise', None], default=None,
                        help="Type of feedback to generate")
    parser.add_argument("--threshold", type=float, default=0,
                        help="Reward threshold for feedback")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--split", type=str, default='train', help="Split of data (train by default)")

    args = parser.parse_args()
    
    if args.api_type and not args.api_key:
        parser.error("--api_key is required when using --api_type")

    asyncio.run(main(args))