"""
A script for create feedback datasets from data that was sampled with train.sample.
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
from typing import List, Dict, Iterator
import copy
from train.utils import StreamingJSONWriter
from collections import defaultdict

def create_batches_respecting_prompts(samples: List[Dict], batch_size: int) -> Iterator[List[Dict]]:
    """
    Create batches that keep samples with the same prompt_id together while respecting max batch size.
    This is needed for creating pairwise feedback.
    """
    # Group samples by prompt_id
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample.get('prompt_id', None)].append(sample)
    
    current_batch = []
    for prompt_id, group in grouped.items():
        # If adding this group would exceed batch_size, yield current batch
        if len(current_batch) + len(group) > batch_size and current_batch:
            yield current_batch
            current_batch = []
        
        # If group itself exceeds batch_size, split it into smaller batches
        if len(group) > batch_size:
            # yield full batches of this group
            for i in range(0, len(group), batch_size):
                yield group[i:i + batch_size]
        else:
            current_batch.extend(group)

            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []
    
    # Yield any remaining samples
    if current_batch:
        yield current_batch

def create_simple_batches(samples: List[Dict], batch_size: int) -> Iterator[List[Dict]]:
    """Create batches of specified size without any grouping requirements."""
    for i in range(0, len(samples), batch_size):
        yield samples[i:i + batch_size]

def process_batch(batch: List[Dict], 
                 reward_model: AutoModelForSequenceClassification,
                 reward_tokenizer: AutoTokenizer, 
                 max_length: int,
                 accelerator: Accelerator) -> List[Dict]:
    """
    Process a batch of samples through the reward model. Add the reward as a field to each item.
    """
    sequences = [item['instruction'] + item['output'] for item in batch]

    reward_model.eval()
    with torch.no_grad():
        reward_inputs = reward_tokenizer(
            sequences,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=max_length
        ).to(accelerator.device)

        outputs = reward_model(**reward_inputs)
        # Use the positive class logit as the reward score
        reward_scores = outputs.logits[:, 1]
    
    all_reward_scores = accelerator.gather(reward_scores)
    
    updated_batch = copy.deepcopy(batch)
    for item, score in zip(updated_batch, all_reward_scores.tolist()):
        item['reward'] = score
    
    return updated_batch

def convert_batch_to_binary_feedback(batch: List[Dict], threshold: float=0.5) -> List[Dict]:
    """
    Convert a batch of samples to binary feedback format. A sample is considered desirable if its
    reward crosses the threshold (label = 1) and undesirable otherwise (label = 0). Returns a list
    of JSON objects with fields

    - instruction: clean prompt, without the the chat template 
    - output: clean output, without the chat template
    - prompt_id: unique integer for the prompt
    - sample_id: integer from 0 to k - 1 for one of the k samples produced per prompt_id
    - reward: reward assigned to instruction + output by reward model
    - label: 1 if the reward surpasses a given threshold and 0 otherwise
    """
    feedback = []
    for sample in batch:
        feedback_item = {
            'instruction': sample['instruction'],
            'prompt_id': sample['prompt_id'],
            'output': sample['output'],
            'sample_id': sample['sample_id'],
            'label': 1 if sample['reward'] > threshold else 0,
            'reward': sample['reward'],
            'type': 'binary_feedback',
        }
        feedback.append(feedback_item)
    return feedback

def convert_batch_to_pairwise_feedback(batch: List[Dict], seed: int, threshold: float=0) -> List[Dict]:
    """
    Convert a batch of samples to pairwise feedback format. Returns a list of JSON objects with fields

    - instruction: clean prompt, without the the chat template 
    - prompt_id: unique integer for the prompt
    - output_A: clean output of the first sample, without the chat template
    - output_B: clean output of the second sample, without the chat template
    - sample_id_A: sample ID for output A
    - sample_id_B: sample ID for output B
    - label: 1 if reward A if greater than reward B; 0 otherwise
    - reward_A: reward assigned by reward model to sample A
    - reward_B: reward assigned by reward model to sample B
    - reward_difference: absolute difference in rewards
    """
    random.seed(seed)
    
    # Group samples by prompt_id
    grouped = defaultdict(list)
    for sample in batch:
        if 'prompt_id' in sample:
            grouped[sample['prompt_id']].append(sample)
    
    feedback = []
    for prompt_id, group in grouped.items():
        if len(group) < 2:
            continue
        
        # Sort by reward
        group.sort(key=lambda x: x['reward'], reverse=True)
        
        # Create pairs
        for i in range(len(group) - 1):
            higher_reward, lower_reward = group[i], group[i + 1]

            if higher_reward['reward'] == lower_reward['reward']:
                continue

            if threshold and abs(sample_A['reward'] - sample_B['reward']) < threshold:
                continue
            
            # Randomly decide order
            if random.random() < 0.5:
                sample_A, sample_B = higher_reward, lower_reward
                label = 1
            else:
                sample_A, sample_B = lower_reward, higher_reward
                label = 0
                
            feedback_item = {
                'instruction': sample_A['instruction'],
                'prompt_id': prompt_id,
                'output_A': sample_A['output'],
                'output_B': sample_B['output'],
                'sample_id_A': sample_A['sample_id'],
                'sample_id_B': sample_B['sample_id'],
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
    reward_tokenizer = AutoTokenizer.from_pretrained(
        args.reward_model_path,
        local_files_only=True,
        trust_remote_code=True
    ) 

    reward_model = accelerator.prepare(reward_model)

    if accelerator.is_main_process:
        print(f"Loading samples from {args.samples_path}")

    with open(args.samples_path, 'r') as f:
        samples = json.load(f)
    
    # get train and test splits, with no prompts appearing in both
    prompt_ids = list(set([ item['prompt_id'] for item in samples ]))
    test_prompt_ids = prompt_ids[:int(args.fraction_test * len(prompt_ids))]
    train_prompt_ids = prompt_ids[int(args.fraction_test * len(prompt_ids)):]

    # Set up batching strategy based on feedback type
    if args.feedback_type == 'pairwise':
        batch_iterator = create_batches_respecting_prompts(samples, args.batch_size)
    else:
        batch_iterator = create_simple_batches(samples, args.batch_size)

    # Set up output writer
    if accelerator.is_main_process:
        print(f"Writing feedback for {len(samples)} samples to {args.output_path}")
        output_file = open(args.output_path, 'w')
        writer = StreamingJSONWriter(output_file)

    # Process batches
    for batch_idx, batch in enumerate(tqdm(batch_iterator, 
                                         desc="Processing batches",
                                         disable=not accelerator.is_main_process)):
        # Get rewards for batch
        processed_batch = process_batch(
            batch,
            reward_model,
            reward_tokenizer,
            args.max_length,
            accelerator
        )

        # Convert to appropriate feedback format
        if args.feedback_type == 'binary':
            feedback_batch = convert_batch_to_binary_feedback(processed_batch, threshold=args.threshold)
        elif args.feedback_type == 'pairwise':
            feedback_batch = convert_batch_to_pairwise_feedback(processed_batch, args.seed + batch_idx, threshold=args.threshold)
        else:
            feedback_batch = processed_batch

        # Write feedback (only on main process)
        if accelerator.is_main_process:
            for item in feedback_batch:
                if item['prompt_id'] in train_prompt_ids:
                    item['split'] = 'train'
                elif item['prompt_id'] in test_prompt_ids:
                    item['split'] = 'test'
                else:
                    continue 

                writer.write_item(item)

    # Close writer
    if accelerator.is_main_process:
        writer.close()
        output_file.close()

    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assign rewards to samples using a reward model")
    parser.add_argument("reward_model_path", type=str, help="Path to the reward model")
    parser.add_argument("samples_path", type=str, help="Path to the JSON file containing samples")
    parser.add_argument("output_path", type=str, help="Path to save the feedback file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--max_length", type=int, default=2048, 
                        help="Maximum sequence length for reward model input")
    parser.add_argument("--feedback_type", type=str, choices=['binary', 'pairwise', None], default=None,
                        help="Type of feedback to generate (either binary, pairwise, or just annotate with rewards)")
    parser.add_argument("--threshold", type=float, default=0,
                        help="Reward threshold for feedback (absolute threshold for binary feedback and minimum reward difference for pairwise)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for pairwise feedback generation")
    parser.add_argument("--fraction_test", type=float, default=0.1,
                        help="Fraction of prompts to use for test set (default: 0.1)")

    args = parser.parse_args()
    main(args)