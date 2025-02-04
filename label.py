"""
A script for creating feedback datasets from data that was sampled with train.sample.
Supports both reward model-based and API-based labeling.

Sample usage with Accelerate for reward model:
accelerate launch --config_file accelerate_config/fsdp_4gpu.yaml --main_process_port 29500 \ 
    label.py --reward_model_path models/llama3-8B-bt/FINAL outputs.json reward_data.json --feedback_type pairwise

Sample usage for API labeling:
accelerate launch label.py --api_type openai --api_key YOUR_KEY --api_model gpt-4 \
    --label_prompt "Rate this response's quality from 0 to 1:" \
    outputs.json reward_data.json --feedback_type binary
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
from train.utils import StreamingJSONWriter
from train.dataloader import SFTDataLoader
from collections import defaultdict
import openai
import time
import asyncio
import aiohttp


async def get_api_completion(
    client: openai.AsyncOpenAI,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.0,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> float:
    """Get completion from API with retry logic."""
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            # Extract float from response
            try:
                return float(response.choices[0].message.content.strip())
            except ValueError:
                print(f"Warning: Could not parse API response as float: {response.choices[0].message.content}")
                return 0.0
                
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error after {max_retries} attempts: {e}")
                return 0.0
            await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff

async def batch_api_scoring(
    samples: List[Dict],
    client: openai.AsyncOpenAI,
    system_prompt: str,
    label_prompt: str,
    model: str,
    batch_size: int = 10
) -> List[float]:
    """Process a batch of samples through the API concurrently."""
    scores = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        tasks = []
        for sample in batch:
            prompt = f"{label_prompt}\n\nPrompt: {sample['instruction']}\nResponse: {sample['output'][0]['content']}"
            tasks.append(get_api_completion(client, system_prompt, prompt, model))
        batch_scores = await asyncio.gather(*tasks)
        scores.extend(batch_scores)
    return scores

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
    
    gathered_samples = [None] * accelerator.num_processes
    torch.distributed.all_gather_object(gathered_samples, processed_samples)
    
    if accelerator.is_main_process:
        return [item for sublist in gathered_samples for item in sublist]
    return []

async def process_batch_with_api(
    batch: Dict,
    client: openai.AsyncOpenAI,
    system_prompt: str,
    label_prompt: str,
    model: str,
    accelerator: Accelerator
) -> List[Dict]:
    """Process a batch through the API."""
    processed_samples = []
    if accelerator.is_main_process:
        samples = []
        for i in range(len(batch['prompt'])):
            sample = {
                'prompt': batch['prompt'][i],
                'instruction': batch['original_prompt'][i] if 'original_prompt' in batch else batch['prompt'][i][0]['content'],
                'output': batch['target'][i],
                'prompt_id': batch['prompt_id'][i],
            }
            samples.append(sample)
            
        scores = await batch_api_scoring(samples, client, system_prompt, label_prompt, model)
        for sample, score in zip(samples, scores):
            sample['reward'] = score
            processed_samples.append(sample)
            
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

def convert_to_pairwise_feedback(samples: List[Dict], seed: int, threshold: float=0) -> List[Dict]:
    """Convert samples to pairwise feedback format."""
    random.seed(seed)
    
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample['prompt_id']].append(sample)
    
    feedback = []
    for prompt_id, group in grouped.items():
        if len(group) < 2:
            continue
        
        group.sort(key=lambda x: x['reward'], reverse=True)
        
        for i in range(len(group) - 1):
            higher_reward, lower_reward = group[i], group[i + 1]
            
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
    
    # Set up API client if using API labeling
    client = None
    if args.api_type:
        tokenizer = AutoTokenizer.from_pretrained("gpt2") # Use a basic tokenizer for API-based labeling

        if args.api_type == "openai":
            if not args.api_key:
                raise ValueError("API key must be provided when using API labeling")
            client = openai.AsyncOpenAI(api_key=args.api_key)
        else:
            raise ValueError(f"Unsupported API type: {args.api_type}")

    # Set up reward model if using model-based labeling
    reward_model = None
    tokenizer = None
    if args.reward_model_path:
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
        reward_model = accelerator.prepare(reward_model)
    
    # Initialize the dataloader
    dataloader = SFTDataLoader(
        dataset_names=[args.samples_path],
        tokenizer=tokenizer,
        process_index=accelerator.process_index,
        num_processes=accelerator.num_processes,
        split='train',
        microbatch_size=(args.batch_size // accelerator.num_processes),
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        n_epochs=1,
        seed=args.seed
    )
    
    # Process samples
    all_processed_samples = []
    for batch in tqdm(dataloader, disable=not accelerator.is_main_process):
        if args.reward_model_path:
            processed_batch = process_batch_with_reward_model(batch, reward_model, accelerator)
        else:
            processed_batch = await process_batch_with_api(
                batch, client, args.system_prompt, args.label_prompt, 
                args.api_model, accelerator
            )

        if accelerator.is_main_process:
            all_processed_samples.extend(processed_batch)

    if accelerator.is_main_process:
        accelerator.print(f"{len(all_processed_samples)} samples gathered")

        # Set up output writer
        print(f"Writing feedback to {args.output_path}")
        output_file = open(args.output_path, 'w')
        writer = StreamingJSONWriter(output_file)
        
        # Process and write feedback
        if args.feedback_type == 'binary':
            feedback = convert_to_binary_feedback(all_processed_samples, threshold=args.threshold)
        elif args.feedback_type == 'pairwise':
            feedback = convert_to_pairwise_feedback(all_processed_samples, args.seed, threshold=args.threshold)
        else:
            feedback = all_processed_samples
            for x in feedback:
                x['type'] = 'scalar_feedback'
            
        # Add split information and write
        for item in feedback:
            item['split'] = args.split
            writer.write_item(item)
            
        writer.close()
        output_file.close()

    accelerator.end_training()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label samples using either a reward model or API")
    
    # Input/output arguments
    parser.add_argument("samples_path", type=str, help="Path to the JSON file containing samples")
    parser.add_argument("output_path", type=str, help="Path to save the feedback file")
    
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