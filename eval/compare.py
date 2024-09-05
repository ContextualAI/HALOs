"""
Eval for comparing any two LLMs (local or on HF) using LLM-as-a-judge.

    python -m eval.compare --model1 /nlp/scr/kawin/models/deepspeed_kto/FINAL --model2 EleutherAI/pythia-1.4b --datasets shp oasst --n_samples 10

Each dataset must have a corresponding get_* method in dataloader.py 
"""
import argparse
import json
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Optional

import torch
import transformers
from vllm import LLM, SamplingParams

import openai
from openai import OpenAI
import sys, os, signal, re
from dataclasses import dataclass

# Import necessary modules from your project
from train import dataloader
from train.utils import disable_dropout


@dataclass
class PromptTemplate:
    """
    Prompt generator for comparing the outputs of any number of models using GPT-4 as a judge.
    """
    models: List[str]  # list of models under consideration
    labels: str         # list of labels to assign to models (e.g., "12345")
    seed: int           # random seed 
    verbose: bool
    human_prefix: str="\n<|user|>\n"
    human_suffix: str=""
    assistant_prefix: str="\n<|assistant|>\n"   # Tulu format; modify as needed
    assistant_suffix: str=""

    def __post_init__(self):
        random.seed(self.seed)

    def shuffle(self):
        """
        Shuffle the list of models under consideration. This is to correct for any bias GPT-4 might have
        in picking the first/last response.
        """
        random.shuffle(self.models)

    def clean_response(self, response):
        """
        Truncate any hanging sentence (not ending in a ?!.) and return the cleaned response.
        """
        match = re.search(r'[.!?](?=[^.!?]*$)', response)
        if match:
            return response[:match.start() + 1]
        else:
            return response

    def format(self, history: str, samples: Dict[str, str]) -> str:
        """
        Format samples into a given prompt. Overwrite in subclass as needed.

        Args:
        - history: the input/instruction
        - samples: a dictionary mapping models to their generations

        Returns:
            A list of messages to send to the GPT-4 API. Each item is a dictionary containing the role of the speaker
            and the candidate text.
        """
        self.shuffle()
        history = history[:history.rfind(self.assistant_prefix)].strip()
        messages = [{"role": "user", "content": f"Here is a conversation between {self.human_prefix.strip()} and {self.assistant_prefix.strip()}.\n\n" + history }]
        prompt = [f'{self.assistant_prefix.strip()} can now choose from different responses.']

        for i,m in enumerate(self.models):
            response = self.clean_response(samples[m])
            prompt.append(f'Response {self.labels[i]}: {response}\nEND OF RESPONSE {self.labels[i]}')

        prompt.append("Between these two responses, which response is more helpful, harmless, and concise? Pick one.")

        messages.append({
            "role": "user",
            "content": "\n\n".join(prompt),
        })

        return messages

    def get_model_choice_from_response(self, response) -> str:
        """
        Given a response from the GPT-4 evaluator, identify and return the model it chose.

        Args:
        - response: response from calling GPT-4 API

        Returns:
            One of the models in self.models (or None if LLM judge's choice cannot be inferred).
        """
        completion = response.choices[0].message.content
        answer = re.search(r'response (.).*', completion, re.IGNORECASE)

        if self.verbose:
            print(completion)
        
        if answer is None:
            return None

        idx = self.labels.index(answer.group(1))
        return self.models[idx]


class APITimeoutException(Exception):
    pass


def get_preferred_model(client, history: str, samples: Dict[str, str], prompt_template: PromptTemplate, judge: str, rate_limit_size: int=1000) -> str:
    """
    Find the model whose generation is most preferred by the judge.

    Args:
    - client: OpenAI client
    - history: prompt used to condition generations
    - samples: generations for the given history, indexed by model name
    - prompt_template: instance of PromptTemplate
    - judge: one of the OpenAI chat models
    - rate_limit_size: maximum number of characters that can be in any message to avoid rate limit problem (tokens is ~ 1/3 of chars)

    Returns:
        The name of the more preferred model.
    """
    # Set up a timeout handler
    def timeout_handler(signum, frame):
        """Handler for when OpenAI call takes too long."""
        raise APITimeoutException("API call took too long")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)

    try:
        response = client.chat.completions.create( 
            model=judge,
            messages=prompt_template.format(history, samples),
            temperature=0,
            max_tokens=10,
            seed=prompt_template.seed,
        )

        signal.alarm(0)  # Cancel the alarm since the call completed within the timeout 
        return prompt_template.get_model_choice_from_response(response)
    except ValueError:
        print("The chosen response could not be determined.")
        pass
    except APITimeoutException:
        pass
    except openai.APIConnectionError as e:
        print("The server could not be reached.")
        print(e.__cause__)  # an underlying Exception, likely raised within httpx.
    except openai.RateLimitError as e:
        print("A 429 status code was received; we should back off a bit.")
        signal.alarm(0)
        time.sleep(5)
    except openai.APIStatusError as e:
        print("Another non-200-range status code was received")
        print(e.response)
    finally:
        signal.alarm(0) 
    
    return None


def sample_from_model(model: str, prompts: List[str], sampling_params: SamplingParams, gpu_count: int) -> List[str]:
    model = LLM(model=model, tensor_parallel_size=gpu_count)
    outputs = model.generate(prompts, sampling_params)

    del model
    torch.cuda.empty_cache()

    return [output.outputs[0].text for output in outputs]


def main():
    parser = argparse.ArgumentParser(description="Sample from two models and compare them using GPT-4.")
    parser.add_argument("--model1", required=True, help="Path or name of the first model")
    parser.add_argument("--model2", required=True, help="Path or name of the second model")
    parser.add_argument("--datasets", nargs='+', required=True, help="List of datasets to use")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--judge", default="gpt-4-0613", help="GPT model to use as judge")
    parser.add_argument("--output", default="comparison_results.json", help="Output file for results")
    parser.add_argument("--max_length", type=int, default=512, help="Max length for generated sequences")
    parser.add_argument("--human_prefix", default="\n<|user|>\n", help="Prefix for human turns")
    parser.add_argument("--human_suffix", default="", help="Suffix for human turns")
    parser.add_argument("--assistant_prefix", default="\n<|assistant|>\n", help="Prefix for assistant turns")
    parser.add_argument("--assistant_suffix", default="", help="Suffix for assistant turns")
    parser.add_argument("--gpu_count", type=int, default=1, help="Number of GPUs to use")
    args = parser.parse_args()

    # Load datasets
    all_examples = []
    for dataset_name in args.datasets:
        dataset_func = getattr(dataloader, f"get_{dataset_name}")
        dataset = dataset_func(
            split='test',
            human_prefix=args.human_prefix,
            human_suffix=args.human_suffix,
            assistant_prefix=args.assistant_prefix,
            assistant_suffix=args.assistant_suffix
        )
        all_examples.extend(dataset.data.values())

    # Randomly sample from all examples
    sampled_examples = random.sample(all_examples, min(args.n_samples, len(all_examples)))
    prompts = [example.prompt for example in sampled_examples]

    # Sample from both models
    sampling_params = SamplingParams(temperature=0.7, max_tokens=args.max_length)
    
    samples1 = sample_from_model(args.model1, prompts, sampling_params, args.gpu_count)
    samples2 = sample_from_model(args.model2, prompts, sampling_params, args.gpu_count)

    # Compare samples
    prompt_template = PromptTemplate(
        models=[args.model1, args.model2],
        labels="12",
        seed=42,
        verbose=True,
        human_prefix=args.human_prefix,
        human_suffix=args.human_suffix,
        assistant_prefix=args.assistant_prefix,
        assistant_suffix=args.assistant_suffix
    )

    wins = {args.model1: 0, args.model2: 0}
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    comparisons = []

    for prompt, sample1, sample2 in zip(prompts, samples1, samples2):
        samples = {args.model1: sample1, args.model2: sample2}
        choice = get_preferred_model(client, prompt, samples, prompt_template, args.judge)
        if choice:
            wins[choice] += 1
            samples['choice'] = choice
        else:
            samples['choice'] = ''

        comparisons.append(samples)
        time.sleep(0.5)  # To avoid rate limiting

    # Prepare results
    results = {
        "date": str(datetime.now()),
        "total_comparisons": len(prompts),
        "judge": args.judge,
        "model1": {
            "name": args.model1,
            "wins": wins[args.model1],
        },
        "model2": {
            "name": args.model2,
            "wins": wins[args.model2],
        },
        "datasets": args.datasets,
        "human_prefix": args.human_prefix,
        "human_suffix": args.human_suffix,
        "assistant_prefix": args.assistant_prefix,
        "assistant_suffix": args.assistant_suffix,
        "comparisons": comparisons,
    }

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()