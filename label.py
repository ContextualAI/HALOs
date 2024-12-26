"""
A script for creating feedback datasets from data that was sampled with train.sample.
Sample usage with Accelerate would be

accelerate launch --config_file accelerate_config/fsdp_4gpu.yaml --main_process_port 29500 \ 
    label.py models/llama3-8B-bt/FINAL outputs.json reward_data.json --feedback_type pairwise

where models/llama3-8B-bt/FINAL is the reward model that was trained using a BradleyTerryTrainer,
outputs.json consists of samples produced by train.sample, and the type of feedback generated is
pairwise.
"""

import argparse
import json
import os
import torch
import random
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from typing import List, Dict
import re
from train.utils import StreamingJSONWriter
from train.dataloader import SFTDataLoader
from collections import defaultdict

def process_batch(batch: Dict, 
                 reward_model: AutoModelForSequenceClassification,
                 accelerator: Accelerator) -> List[Dict]:
    """
    Process a batch through the reward model using the already tokenized sequences.
    Add the reward as a field to each item.
    """
    reward_model.eval()
    with torch.no_grad():
        outputs = reward_model(
            input_ids=batch['target_combined_input_ids'],
            attention_mask=batch['target_combined_attention_mask']
        )
        # Use the positive class logit as the reward score
        reward_scores = outputs.logits[:, 1]
    
    all_reward_scores = accelerator.gather(reward_scores)
    
    # Create list of dicts with all necessary information
    processed_samples = []
    for i in range(len(batch['prompt'])):
        sample = {
            'prompt_key': ' '.join([ x['content'] for x in batch['prompt'][i] ]), # flattened prompt
            'prompt': batch['prompt'][i],
            'instruction': batch['original_prompt'][i] if 'original_prompt' in batch else batch['prompt'][i][0]['content'],
            'output': batch['target'][i], 
            'reward': all_reward_scores[i].item()
        }
        processed_samples.append(sample)
    
    return processed_samples

def convert_batch_to_binary_feedback(samples: List[Dict], threshold: float=0.5) -> List[Dict]:
    """
    Convert samples to binary feedback format. A sample is considered desirable if its
    reward crosses the threshold (label = 1) and undesirable otherwise (label = 0).
    """
    feedback = []
    for sample in samples:
        feedback_item = {
            'prompt_key': sample['prompt_key'],
            'prompt': sample['prompt'],
            'output': sample['output'],
            'label': 1 if sample['reward'] > threshold else 0,
            'reward': sample['reward'],
            'type': 'binary_feedback',
        }
        feedback.append(feedback_item)
    return feedback

def convert_to_pairwise_feedback(samples: List[Dict], seed: int, threshold: float=0) -> List[Dict]:
    """
    Convert samples to pairwise feedback format.
    """
    random.seed(seed)
    
    # Group samples by prompt_id
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample['prompt_key']].append(sample)
    
    feedback = []
    for prompt_key, group in grouped.items():
        if len(group) < 2:
            continue
        
        # Sort by reward
        group.sort(key=lambda x: x['reward'], reverse=True)
        
        # Create pairs
        for i in range(len(group) - 1):
            higher_reward, lower_reward = group[i], group[i + 1]
            
            if higher_reward['reward'] == lower_reward['reward']:
                continue
                
            if threshold and abs(higher_reward['reward'] - lower_reward['reward']) < threshold:
                continue
            
            # Randomly decide order
            if random.random() < 0.5:
                sample_A, sample_B = higher_reward, lower_reward
                label = 1
            else:
                sample_A, sample_B = lower_reward, higher_reward
                label = 0
                
            feedback_item = {
                'prompt_key': prompt_key,
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

def main(args):
    accelerator = Accelerator()
    
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

    # Initialize the dataloader with samples file
    dataloader = SFTDataLoader(
        dataset_names=[args.samples_path],
        tokenizer=tokenizer,
        split='train',  # We'll handle train/test split later
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        n_epochs=1,
        seed=args.seed
    )
    
    # Process all samples first to get rewards
    all_processed_samples = []
    for batch in tqdm(dataloader, disable=not accelerator.is_main_process):
        processed_batch = process_batch(
            batch,
            reward_model,
            accelerator
        )
        all_processed_samples.extend(processed_batch)
    
    # Split into train and test based on prompt_ids
    prompt_keys = list(set(sample['prompt_key'] for sample in all_processed_samples))
    random.Random(args.seed).shuffle(prompt_keys)
    test_prompt_keys = set(prompt_keys[:int(args.fraction_test * len(prompt_keys))])
    
    # Set up output writer
    if accelerator.is_main_process:
        print(f"Writing feedback to {args.output_path}")
        output_file = open(args.output_path, 'w')
        writer = StreamingJSONWriter(output_file)
        
        # Process and write feedback
        if args.feedback_type == 'binary':
            feedback = convert_batch_to_binary_feedback(all_processed_samples, threshold=args.threshold)
        elif args.feedback_type == 'pairwise':
            feedback = convert_to_pairwise_feedback(all_processed_samples, args.seed, threshold=args.threshold)
        else:
            feedback = all_processed_samples
            
        # Add split information and write
        for item in feedback:
            item['split'] = 'test' if item['prompt_key'] in test_prompt_keys else 'train'
            item.pop('prompt_key')
            writer.write_item(item)
            
        writer.close()
        output_file.close()

    accelerator.end_training()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assign rewards to samples using a reward model")
    parser.add_argument("reward_model_path", type=str, help="Path to the reward model")
    parser.add_argument("samples_path", type=str, help="Path to the JSON file containing samples")
    parser.add_argument("output_path", type=str, help="Path to save the feedback file")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
    parser.add_argument("--max_length", type=int, default=1024, 
                        help="Maximum sequence length for reward model input")
    parser.add_argument("--max_prompt_length", type=int, default=800, 
                        help="Maximum prompt length for reward model input")
    parser.add_argument("--feedback_type", type=str, choices=['binary', 'pairwise', None], default=None,
                        help="Type of feedback to generate (either binary, pairwise, or just annotate with rewards)")
    parser.add_argument("--threshold", type=float, default=0,
                        help="Reward threshold for feedback (absolute threshold for binary feedback and minimum reward difference for pairwise)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--fraction_test", type=float, default=0.1,
                        help="Fraction of prompts to use for test set (default: 0.1)")

    args = parser.parse_args()
    main(args)