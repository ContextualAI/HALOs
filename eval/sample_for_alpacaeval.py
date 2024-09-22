import argparse
import json
import sys
import re
from typing import List, Dict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from train.dataloader import SFTDataLoader

def main(args):
    # Load the model and tokenizer
    print(f"Loading model and tokenizer from {args.model_path}")
    llm = LLM(model=args.model_path, tensor_parallel_size=args.gpu_count)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if not tokenizer.chat_template:
        tokenizer.chat_template = open('template.jinja').read()

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=['<|im_end|>']
    )

    # Initialize the SFTDataLoader
    dataloader = SFTDataLoader(
        dataset_names=['alpacaeval'],
        tokenizer=tokenizer,
        split='test',
        max_prompt_length=args.max_prompt_length,
        n_epochs=1,
        seed=args.seed
    )

    prompts, unformatted_prompts = [], []
    # Iterate through the dataloader to get the formatted prompts
    for batch in dataloader:
        prompts.extend(batch['prompt_text'])
        unformatted_prompts.extend(batch['original_prompt'])

    # Generate responses
    responses = llm.generate(prompts, sampling_params)

    # Process the outputs. im_start and im_end are special tokens used in alpacaeval preprocessing; they must be removed
    outputs = []
    for prompt, response in zip(unformatted_prompts, responses):
        output = {
            "instruction": re.sub(r"<\|(im_start|im_end)\|>", "", prompt),
            "output": re.sub(r"<\|(im_start|im_end)\|>", "", response.outputs[0].text.strip()),
            "generator": args.model_path,
        }
        outputs.append(output)

    if args.output_file == "":
        args.output_file = f"outputs/alpacaeval/{args.model_path.replace('/', '_')}.json"

    with open(args.output_file, 'w') as f:
        json.dump(outputs, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from a local model using vllm for AlpacaEval")
    parser.add_argument("model_path", type=str, help="Path to the local model")
    parser.add_argument("--output_file", type=str, default="", help="Path to save the output JSON file")
    parser.add_argument("--gpu_count", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum number of tokens to generate")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Maximum length of prompt (in tokens)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
  
    args = parser.parse_args()
    main(args)