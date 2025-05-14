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
import numpy as np
import torch
import random
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm, trange, trange
from typing import List, Dict, Optional, Union
import re
import os
import os
from .utils import StreamingJSONWriter, get_api_completion
from .dataloader import SFTDataLoader
from collections import defaultdict
import openai
import asyncio
import torch.distributed as dist


def process_batch_with_reward_model(
    samples: List,
    reward_model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    tokenizer: AutoTokenizer,
    accelerator: Accelerator
) -> List[Dict]:
    """Process a batch through the reward model using the already tokenized sequences."""
    processed_samples = []
    chop = lambda txt: re.sub(r'([.!?])[^.!?]*\Z', r'\1', txt.strip())

    for i in trange(len(samples), disable=(not accelerator.is_main_process)):
        if i % accelerator.num_processes == accelerator.process_index:
            input_ids = tokenizer.apply_chat_template(
                samples[i]["prompt"] + [{"role":"assistant", "content":chop(samples[i]["output"])}],
                return_tensors="pt",
                max_length=2048,
            ).to(accelerator.device)

            with torch.no_grad():
                outputs = reward_model(
                    input_ids=input_ids,
                )
                if args.reward_model_path == "RLHFlow/ArmoRM-Llama3-8B-v0.1":
                    reward_scores = outputs.score
                else:
                    reward_scores = outputs.logits[:, 1]
    
            processed_sample = samples[i].copy()
            processed_sample['reward'] = reward_scores[0].item()
            # since a dataloader isn't used, the output has to be explicitly formatted
            processed_sample['output'] = [{ "role" : "assistant", "content" : processed_sample['output'] }]
            processed_samples.append(processed_sample)
    
    return processed_samples
    processed_samples = []
    chop = lambda txt: re.sub(r'([.!?])[^.!?]*\Z', r'\1', txt.strip())

    for i in trange(len(samples), disable=(not accelerator.is_main_process)):
        if i % accelerator.num_processes == accelerator.process_index:
            input_ids = tokenizer.apply_chat_template(
                samples[i]["prompt"] + [{"role":"assistant", "content":chop(samples[i]["output"])}],
                return_tensors="pt",
                max_length=2048,
            ).to(accelerator.device)

            with torch.no_grad():
                outputs = reward_model(
                    input_ids=input_ids,
                )
                if args.reward_model_path == "RLHFlow/ArmoRM-Llama3-8B-v0.1":
                    reward_scores = outputs.score
                else:
                    reward_scores = outputs.logits[:, 1]
    
            processed_sample = samples[i].copy()
            processed_sample['reward'] = reward_scores[0].item()
            # since a dataloader isn't used, the output has to be explicitly formatted
            processed_sample['output'] = [{ "role" : "assistant", "content" : processed_sample['output'] }]
            processed_samples.append(processed_sample)
    
    return processed_samples


async def process_samples_with_api(
    samples: List,
    samples: List,
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
            prompt = f"INSTRUCTION: {sample['prompt'][0]['content']}\n\nRESPONSE: {sample['output']}\n\n{label_prompt}"
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
            # Try to extract 'Final Score: X' from the response
            match = re.search(r"Final Score:\s*([0-9]+(?:\.[0-9]+)?)", str(score), re.IGNORECASE)
            if match:
                score = float(match.group(1))
            else:
                # Fallback: extract first number
                score = float(re.search(r'\d+(?:\.\d+)?', str(score)).group())
        except Exception:
            print(f"Warning: Could not parse API response {score} as float. skipping prompt {sample['prompt_id']}")
            continue
                
        processed_sample = sample.copy()
        processed_sample['reward'] = score
        # since a dataloader isn't used, the output has to be explicitly formatted
        processed_sample['output'] = [{ "role" : "assistant", "content" : processed_sample['output'] }]
        processed_samples.append(processed_sample)
            
    return processed_samples


def convert_to_binary_feedback(samples: List[Dict], threshold=0) -> List[Dict]:
    """Convert samples to binary feedback format."""
    feedback = []

    if threshold == 'mean':
        rewards = [ sample['reward'] for sample in samples ]
        threshold = np.mean(rewards)
    elif threshold == 'median':
        rewards = [ sample['reward'] for sample in samples ]
        threshold = np.median(rewards)
    else:
        threshold = int(threshold)

    for sample in samples:
        feedback_item = {
            'prompt_id': sample['prompt_id'],
            'prompt': sample['prompt'],
            'output': sample['output'],
            'label': 1 if sample['reward'] >= threshold else 0,
            'reward': sample['reward'],
            'type': 'binary_feedback',
        }
        feedback.append(feedback_item)
    return feedback


def convert_to_pairwise_feedback(samples: List[Dict], seed: int, threshold=0) -> List[Dict]:
    """Convert samples to pairwise feedback format."""
    random.seed(seed)
    
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample['prompt_id']].append(sample)
    
    feedback = []
    for prompt_id, group in grouped.items():
        if len(group) < 2:
            continue
        
        random.shuffle(group)
        
        for i in range(0, len(group) - 1, 2):
            sample_A, sample_B = group[i], group[i + 1]

            if abs(float(float(sample_A['reward'])) - float(float(sample_B['reward']))) <= float(float(threshold)):
                continue

            label = int(sample_A['reward'] > sample_B['reward'])
                
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

        print(f"Labelled {len(processed_samples)} samples using {args.api_type} API")
    else:
        if accelerator.is_main_process:
            accelerator.accelerator.print(f"Loading reward model from {args.reward_model_path}")


        # trust_remote_code is necessary for Armo RM to be downloaded correctly
        reward_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
        reward_model, tokenizer, samples = accelerator.prepare(reward_model, tokenizer, samples)
        
        # Initialize distributed setup if not already done
        if accelerator.num_processes > 1 and not dist.is_initialized():
            accelerator.wait_for_everyone()
            samples = samples[:((len(samples) // accelerator.num_processes) * accelerator.num_processes)]
            samples = samples[:((len(samples) // accelerator.num_processes) * accelerator.num_processes)]
            
            if accelerator.is_main_process:
                dist.init_process_group(backend='nccl')
        
        processed_samples = process_batch_with_reward_model(samples, reward_model, tokenizer, accelerator)
        json.dump(processed_samples, open(f'temp_{accelerator.process_index}.json', 'w'))
        accelerator.wait_for_everyone()
        processed_samples = process_batch_with_reward_model(samples, reward_model, tokenizer, accelerator)
        json.dump(processed_samples, open(f'temp_{accelerator.process_index}.json', 'w'))
        accelerator.wait_for_everyone()
        processed_samples = []

        if accelerator.is_main_process:
            for i in range(accelerator.num_processes):
                processed_samples.extend(json.load(open(f'temp_{i}.json')))
                os.remove(f'temp_{i}.json')

            accelerator.print(f"Labelled {len(processed_samples)} samples using {args.reward_model_path}")

        if accelerator.is_main_process:
            for i in range(accelerator.num_processes):
                processed_samples.extend(json.load(open(f'temp_{i}.json')))
                os.remove(f'temp_{i}.json')

            accelerator.print(f"Labelled {len(processed_samples)} samples using {args.reward_model_path}")

    # Set up output writer
    if accelerator.is_main_process:
        accelerator.print(f"Writing feedback to {args.output_path}")
        with open(args.output_path, 'w') as f:
            writer = StreamingJSONWriter(f)
        
            # Process and write feedback
            if args.feedback_type == 'binary':
                feedback = convert_to_binary_feedback(processed_samples, threshold=args.threshold)
            elif args.feedback_type == 'pairwise':
                feedback = convert_to_pairwise_feedback(processed_samples, args.seed, threshold=args.threshold)
            else:
                feedback = processed_samples
                for x in feedback:
                    x['type'] = 'scalar_feedback'
            
            # Include split if specified
            if args.split:
                for x in feedback:
                    item['split'] = args.split

            # Add split information and write
            for item in feedback:
                writer.write_item(item)
            
            writer.close()
    
    # Clean up distributed training resources if using reward model
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
    parser.add_argument("--api_model", type=str, default="gpt-4.1-mini", help="Model to use for API labeling")
    parser.add_argument("--system_prompt", type=str, default="""You are an expert evaluator for language model responses. Your task is to score the quality of responses to user queries on a scale from 1-10.
                        \n For the given RESPONSE, consider these key factors:
                        \n 1. Helpfulness: Does the RESPONSE directly address the INSTRUCTION with useful information?
                        \n 2. Accuracy: Is the information provided factually correct?
                        \n 3. Conciseness: Does the RESPONSE avoid unnecessary verbosity while remaining complete?
                        \n 4. Natural language: Does the RESPONSE use natural, human-like language without formulaic patterns?
                        \n 5. Instruction following: Does the RESPONSE precisely follow the INSTRUCTION ?
                        \n 6. Reasoning quality: Does the RESPONSE demonstrate clear, logical reasoning when needed?
                        \n 7. Creativity: For creative tasks, does the RESPONSE show originality and inventiveness?
                        \n IMPORTANT SCORING GUIDELINES:
                        \n - Responses that are both concise AND directly address the INSTRUCTION should receive the highest scores
                        \n - Responses with unnecessary preambles like "I'd be happy to help" should be penalized
                        \n - Responses that over-explain simple concepts should be penalized
                        \n - Responses that acknowledge limitations when appropriate should be rewarded
                        \n - Responses that provide step-by-step reasoning for complex problems should be rewarded
                        \n - Responses that use natural, conversational language should be rewarded over formulaic responses
                        \n Output format: Provide a single overall score from 1-10.""", help="System prompt for API labeling")
    parser.add_argument("--label_prompt", type=str, default="Provide only a RATING from 0 to 10 based on how well the RESPONSE satisfied the INSTRUCTION according to the rubric given earlier : ", help="Prompt template for API labeling")
    # Processing arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length for input")
    parser.add_argument("--max_prompt_length", type=int, default=1024, help="Maximum prompt length for input")
    parser.add_argument("--feedback_type", type=str, choices=['binary', 'pairwise', None], default=None, help="Type of feedback to generate")
    parser.add_argument("--threshold", type=str, default="median", help="How the reward threshold is calculated; this can also be a number (e.g., 0.5)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--split", type=str, default=None, help="Split of data")

    args = parser.parse_args()
    
    if args.api_type and not args.api_key:
        parser.error("--api_key is required when using --api_type")

    asyncio.run(main(args))