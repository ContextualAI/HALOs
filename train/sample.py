import argparse
import json
import sys
import re
import os
import inspect
from typing import List, Dict, TextIO
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from train.dataloader import SFTDataLoader
from train.utils import set_offline_if_needed
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)
import train.dataloader as dataloader_module


class StreamingJSONWriter:
    """Writes JSON arrays to a file in a streaming fashion."""
    def __init__(self, file: TextIO):
        self.file = file
        self.is_first = True
        self.file.write('[\n')
    
    def write_item(self, item: Dict):
        """Write a single item to the JSON array."""
        if not self.is_first:
            self.file.write(',\n')
        json.dump(item, self.file, indent=2)
        self.is_first = False
        # Flush after each write to ensure immediate disk writing
        self.file.flush()
    
    def close(self):
        """Close the JSON array and the file."""
        self.file.write('\n]')
        self.file.flush()


def get_available_datasets():
    """Get list of available datasets by finding all get_* functions in dataloader.py"""
    return [name[4:] for name, _ in inspect.getmembers(dataloader_module, inspect.isfunction) 
            if name.startswith('get_')]


def validate_datasets(datasets):
    """Validate that all requested datasets have corresponding get_* functions"""
    available_datasets = get_available_datasets()
    invalid_datasets = [d for d in datasets if d not in available_datasets]
    
    if invalid_datasets:
        available_str = "\n- ".join(available_datasets)
        raise ValueError(
            f"The following datasets are not available: {invalid_datasets}\n"
            f"Available datasets must have a corresponding get_* function in dataloader.py.\n"
            f"Currently available datasets are:\n- {available_str}"
        )


def main(args):
    validate_datasets(args.datasets)
    set_offline_if_needed()
    
    # Load the model and tokenizer
    print(f"Loading model and tokenizer from {args.model_path}")
    llm = LLM(model=args.model_path, tensor_parallel_size=args.gpu_count)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.chat_template = open('template.jinja').read()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=['<|im_end|>'],
        n=args.num_samples
    )

    prompt_idx = 0
    # Open the output file and create a streaming writer
    with open(args.output_file, 'w') as f:
        writer = StreamingJSONWriter(f)

        # Process each dataset
        for dataset in args.datasets:
            print(f"\nProcessing dataset: {dataset}")

            # Initialize the SFTDataLoader
            dataloader = SFTDataLoader(
                dataset_names=[dataset],
                tokenizer=tokenizer,
                split=args.split,
                max_prompt_length=args.max_prompt_length,
                n_epochs=1,
                seed=args.seed,
                batch_size=args.batch_size
            )

            # Process the dataset in batches
            for batch_idx, batch in enumerate(dataloader):
                prompts = batch['prompt_text']
                metadata = batch['original_prompt'] if 'original_prompt' in batch else batch['prompt_text']

                print(f"Generating responses for batch {batch_idx + 1} ({len(prompts)} prompts)...")
                responses = llm.generate(prompts, sampling_params)

                # Process and write each output
                for prompt, response in zip(metadata, responses):
                    for sample_idx, sample in enumerate(response.outputs):
                        output = {
                            "instruction": re.sub(r"<?\|(im_start|im_end)\|>?", "", prompt),
                            "output": re.sub(r"<?\|(im_start|im_end)\|>?", "", sample.text.strip()),
                            "generator": args.model_path,
                            "dataset": dataset,
                            "split": args.split,
                            "prompt_id": prompt_idx,
                            "sample_id": sample_idx
                        }
                        writer.write_item(output)

                    prompt_idx += 1

        writer.close()

    destroy_model_parallel()
    destroy_distributed_environment()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from a local model using vllm for AlpacaEval")
    parser.add_argument("model_path", type=str, help="Path to the local model folder or the Huggingface repo")
    parser.add_argument("--datasets", type=str, nargs="+", default=["alpacaeval"], help="List of datasets to sample from (space-separated)")
    parser.add_argument("--output_file", type=str, default="outputs.json", help="Path to save the output JSON file")
    parser.add_argument("--gpu_count", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum number of tokens to generate")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Maximum length of prompt (in tokens)")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing datasets")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (train/test)")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per input")
  
    args = parser.parse_args()
    main(args)