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
        if args.reward_model_path == "RLHFlow/ArmoRM-Llama3-8B-v0.1":
            reward_scores = outputs.score.cpu().float()
        else:
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
        
    # Return gathered samples from all processes, not just main process
    # This ensures samples from all processes are included
    dist.all_gather_object(gathered_samples, processed_samples)
    
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
            prompt = f"INSTRUCTION: {sample['prompt'][0]['content']}\n\nRESPONSE: {sample['output']}\n\n{label_prompt}: "
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


def convert_to_pairwise_feedback(samples: List[Dict], seed: int) -> List[Dict]:
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

        # Initialize the dataloader for reward model processing
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
        dataloader, reward_model = accelerator.prepare(dataloader, reward_model)
        
        # Initialize distributed setup if not already done
        if accelerator.num_processes > 1 and not dist.is_initialized():
            accelerator.wait_for_everyone()
            
            if accelerator.is_main_process:
                dist.init_process_group(backend='nccl')
        
        processed_samples = []
        for batch in tqdm(dataloader, disable=not accelerator.is_main_process):
            batch_samples = process_batch_with_reward_model(batch, reward_model, accelerator)
            if accelerator.is_main_process:
                processed_samples.extend(batch_samples)
        
        # Wait for all processes to complete
        accelerator.wait_for_everyone()

        print(f"Labelled {len(processed_samples)} samples using {args.reward_model_path}")

    # Set up output writer
    if accelerator.is_main_process:
        accelerator.print(f"Writing feedback to {args.output_path}")
        with open(args.output_path, 'w') as f:
            writer = StreamingJSONWriter(f)
        
            # Process and write feedback
            if args.feedback_type == 'binary':
                feedback = convert_to_binary_feedback(processed_samples, threshold=args.threshold)
            elif args.feedback_type == 'pairwise':
                feedback = convert_to_pairwise_feedback(processed_samples, args.seed)
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
    parser.add_argument("--system_prompt", type=str, default="You are an expert evaluator with deep experience in assessing AI responses. Your role is to provide consistent, fair, and detailed ratings of responses based on their alignment with given instructions. You must maintain objectivity, avoid personal bias, and focus on concrete aspects like accuracy, relevance, and safety. Your evaluations should be reproducible and based on clear criteria.", help="System prompt for API labeling")
    parser.add_argument("--label_prompt", type=str, default="""
                        # Instructions for Rating the Response
                        
                        # You are an expert evaluator. Please rate the quality of the RESPONSE to the INSTRUCTION according to the following criteria:

                        ## Evaluation Criteria (Score Impact)
                        1. Accuracy (±3 points):
                        - No factual errors or hallucinations (+3)
                        - Minor inaccuracies (-1 each)
                        - Major factual errors (-3)

                        2. Completeness (±3 points):
                        - Fully addresses all aspects of the instruction (+3)
                        - Partial completion (-1 to -2)
                        - Missing critical components (-3)

                        3. Clarity & Structure (±2 points):
                        - Well-organized and easy to understand (+2)
                        - Clear but could be better structured (+1)
                        - Confusing or poorly structured (-1 to -2)

                        4. Helpfulness (±2 points):
                        - Goes above and beyond to be helpful (+2)
                        - Adequately helpful (+1)
                        - Minimal or unhelpful (-1 to -2)

                        ## Safety & Guidelines
                        - Automatic 0 score for:
                        * Harmful or unsafe content
                        * Explicit violations of guidelines
                        * Malicious or dangerous suggestions

                        ## Workflow
                        1. Evaluate the response against each criterion
                        2. Provide brief reasoning (2-3 sentences) highlighting key strengths and weaknesses
                        3. Sum up the points to determine final score
                        4. Present the final rating as a single number from 0 (worst) to 10 (best)

                        Format your response as:
                        Reasoning: [Your explanation]

                        Final Score: X

                        ---

                        """, help="Prompt template for API labeling")
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