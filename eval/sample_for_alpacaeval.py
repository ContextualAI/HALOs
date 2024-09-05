import argparse
import json
import sys
import os
from typing import List, Dict
from vllm import LLM, SamplingParams

# Now we can import from the parent directory
from train.dataloader import get_alpacaeval

def main(args):
    # Load the model
    print(f"Loading model from {args.model_path}")
    llm = LLM(model=args.model_path, tensor_parallel_size=args.gpu_count)

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # Load prompts using the AlpacaEval dataloader
    data = get_alpacaeval(
        split='test',
        human_prefix='<|user|>',
        human_suffix='',
        assistant_prefix='<|assistant|>',
        assistant_suffix=''
    )

    prompts = [x for x in data]
    responses = llm.generate(prompts, sampling_params)
    outputs = [{
        "instruction": data[x].original_prompt,
        "output": y.outputs[0].text.strip(),
        "generator": args.model_path,
    } for x,y in zip(prompts, responses) ]

    with open(args.output_file, 'w') as f:
        json.dump(outputs, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from a local model using vllm for AlpacaEval")
    parser.add_argument("model_path", type=str, help="Path to the local model")
    parser.add_argument("--output_file", type=str, default="alpaca_eval_outputs.json",
                        help="Path to save the output JSON file (default: alpaca_eval_outputs.json)")
    parser.add_argument("--gpu_count", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate")
  
    args = parser.parse_args()
    main(args)