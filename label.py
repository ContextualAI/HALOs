import argparse
import json
import os
import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from typing import List, Dict
import copy
from train.utils import StreamingJSONWriter


def process_batch(batch: List[Dict], 
                 reward_model: AutoModelForSequenceClassification,
                 reward_tokenizer: AutoTokenizer, 
                 max_length: int,
                 accelerator: Accelerator) -> List[Dict]:
    """Process a batch of samples through the reward model."""
    # Construct full sequences for reward model
    sequences = [ item['raw_input'] + item['raw_output'] for item in batch ]

    reward_model.eval()
    with torch.no_grad():
        # Encode with reward model tokenizer
        reward_inputs = reward_tokenizer(
            sequences,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=max_length
        ).to(accelerator.device)

        # Get reward model scores
        outputs = reward_model(**reward_inputs)
        # Use the positive class logit as the reward score
        reward_scores = outputs.logits[:, 1]
    
    all_reward_scores = accelerator.gather(reward_scores)
    
    # Update items with rewards
    updated_batch = copy.deepcopy(batch)
    for item, score in zip(updated_batch, all_reward_scores.tolist()):
        item['reward'] = score
    
    return updated_batch

def main(args):
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load reward model and tokenizer
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

    # Prepare model with accelerator
    reward_model = accelerator.prepare(reward_model)

    # Load samples
    if accelerator.is_main_process:
        print(f"Loading samples from {args.samples_path}")

    with open(args.samples_path, 'r') as f:
        samples = json.load(f)
    
    # Create a new file for writing (only on main process)
    if accelerator.is_main_process:
        output_path = args.samples_path + '.tmp'
        output_file = open(output_path, 'w')
        writer = StreamingJSONWriter(output_file)
    
    for i in tqdm(range(0, len(samples), args.batch_size), 
                  desc="Processing samples",
                  disable=not accelerator.is_main_process):
        batch = samples[i:i + args.batch_size]
        processed_batch = process_batch(
            batch, 
            reward_model, 
            reward_tokenizer,
            args.max_length,
            accelerator
        )
        
        # Write processed batch (only on main process)
        if accelerator.is_main_process:
            for item in processed_batch:
                writer.write_item(item)

    # Close the writer and replace the original file (only on main process)
    if accelerator.is_main_process:
        writer.close()
        output_file.close()
        os.replace(output_path, args.samples_path)
        print(f"Saved results to {args.samples_path}")

    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assign rewards to samples using a reward model")
    parser.add_argument("reward_model_path", type=str, help="Path to the reward model")
    parser.add_argument("samples_path", type=str, help="Path to the JSON file containing samples")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length for reward model input")
    
    args = parser.parse_args()
    main(args)