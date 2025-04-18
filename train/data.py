# Copyright (c) 2023 Contextual AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Contains the functions for loading data.
Each function of the form get_{dataset_name} (e.g., get_shp, get_oasst, etc.) will return a dict of Example objects, indexed by the prompt for the text.

Each Example object will contain
- the prompt
- a prompt ID (a hash of the prompt unless otherwise specified)
- a list L of generations
- the index in L of the generation that should be the finetuning target
- a list S of the scores/rewards for the generations
- for preference feedback data: pairs of indices (i,j) in L, where generation i is preferable to generation j
- for binary feedback data: whether each generation is desirable/chosen or undesirable/rejected
- the dataset name
- the unformatted prompt
"""

import datasets
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import re
import random
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from .utils import rank0_print, on_rank0, delete_dict
import pandas as pd
import numpy as np
import hashlib


@dataclass
class Example:
    """
    Class for an example in a preference or SFT dataset. If you want each prompt to be uniquely associated with an Example instance, save it in a dict.
    """
    prompt: List = field(default_factory=list)                  # list of turns, each with two keys: "role" and "content"
    prompt_id: int = -1                                         # unique identifier for prompt (optional)
    generations: List = field(default_factory=list)             # list of list of turns (the output sequences to predict)
    sft_index: int = -1                                         # which response in self.generations should be generated for SFT
    scores: List[float] = field(default_factory=list)           # score for each generation
    pairs: List[Tuple[int, int]] = field(default_factory=list)  # for preference feedback data: indices in responses, where i > j in pair (i,j) is a preference
    desirable: List[bool] = field(default_factory=list)         # for binary feedback data: whether the generation at the corresponding index in self.generations is desirable 
    dataset_name: str = ''
    original_prompt: str = ''                                   # the unformatted prompt (needed to recover instruction for AlpacaEval)

    def __setattr__(self, name, value):
        """Set prompt ID automatically."""
        if name == 'prompt' and value is not None:
            content = ''

            for turn in value:
                if "role" not in turn:
                    raise ValueError("every turn in an example must have a 'role' field")
                
                if "content" not in turn:
                    raise ValueError("every turn in an example must have a 'content' field")
                
                content = content + turn['content']

            self.prompt_id = hashlib.sha256(content.encode()).hexdigest()
        
        super().__setattr__(name, value)

    def num_generations(self):
        return len(self.generations)
    
    def remove_extra_spaces(self):
        """
        Remove double spaces in the prompt and generations to standardize spacing.
        """
        def clean(text: str) -> str:
            return re.sub(r'[ \t]{2,}', ' ', text)

        # Clean the prompt
        for turn in self.prompt:
            turn['content'] = clean(turn['content'])

        # Clean the generations
        for x in self.generations:
            for turn in x:
                turn["content"] = clean(turn["content"])


class Dataset:
    """
    A collection of Example instances, indexed by prompt.
    """
    def __init__(self, name):
        self.name = name
        self.data = defaultdict(Example)

    def __setitem__(self, key, value):
        if not isinstance(value, Example):
            raise ValueError("value must be a Example")
        
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return iter(self.data)


def get_alpacaeval(split: str) -> Dataset:
    """
    Load the AlpacaEval dataset (for evaluation only) and convert it into a Dataset.

    Args:
        - split: must be 'test'; otherwise error will be thrown

    Returns:   
        A Dataset instance.
    """
    if split == 'test':
        split = 'eval'
    else:
        raise ValueError('alpacaeval is only for evaluation')

    rank0_print(f'Loading AlpacaEval dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('tatsu-lab/alpaca_eval', split=split)
    if on_rank0():
        dataset = tqdm.tqdm(dataset, desc='Processing AlpacaEval')

    data = Dataset('alpacaeval')

    for row in dataset:
        conversation = [{"role": "user", "content": row['instruction']}]
        data[row['instruction']].prompt = conversation
        data[row['instruction']].generations.append([{"role": "assistant", "content": row['output']}])
        data[row['instruction']].dataset_name = row['dataset']
        data[row['instruction']].original_prompt = row['instruction']
        data[row['instruction']].sft_index = 0

    return data


def get_sampled_data(samples_path: str, split: str) -> Dataset:
    """
    Load samples generated by train.sample and convert it into a Dataset.
    """
    rank0_print(f'Loading samples from {samples_path}...')

    # Load all sample data
    with open(samples_path, 'r') as f:
        sample_data = json.load(f)

    data = Dataset('samples')

    for sample in sample_data:
        if sample.get('split', split) != split:
            continue

        prompt_key = str(sample['prompt_id']) + ' ' + str(sample['sample_id'])
        data[prompt_key].prompt = sample['prompt']
        data[prompt_key].generations.append([{"role": "assistant", "content": sample['output']}])
        data[prompt_key].dataset_name = sample['dataset']
        data[prompt_key].sft_index = 0
        data[prompt_key].prompt_id = sample['prompt_id']

    return data    


def get_feedback(feedback_path: str, split: str) -> Dataset:
    """
    Load feedback data created by label.py and convert it into a Dataset.
    Supports both binary and pairwise feedback formats.

    Args:
        feedback_path: path to the JSON file containing feedback data
        split: only include objects whose 'split' value matches the given split

    Returns:
        A Dataset instance containing the feedback data.
    """
    rank0_print(f'Loading feedback dataset from {feedback_path}...')
    
    # Load all feedback data
    with open(feedback_path, 'r') as f:
        feedback_data = json.load(f)
    
    if not feedback_data:
        raise ValueError(f"No feedback data found in {feedback_path}")
    
    data = Dataset('feedback')

    # Group samples by flattened prompt to ensure we handle all samples for each prompt together
    grouped_samples = defaultdict(list)
    for item in feedback_data:
        grouped_samples[item['prompt_id']].append(item)
    
    for prompt_id, samples in grouped_samples.items():
        if not samples: continue

        feedback_type = samples[0].get('type', None)   
        if not feedback_type:
            raise ValueError("Feedback type not specified in data") 
        
        # split is same for all examples with the same prompt
        if samples[0].get('split', split) != split:
            continue
        
        example = Example()

        if feedback_type == 'binary_feedback':    
            # Add each sample's output and its desirability based on the label
            for sample in samples:
                example.prompt = sample['prompt']
                example.generations.append(sample['output'])
                example.desirable.append(bool(sample['label']))
                example.scores.append(sample['reward'])
                example.dataset_name = feedback_type
            
            # For binary feedback, use any desirable response as the SFT target
            # If no desirable responses, use the highest scoring one
            desirable_indices = [i for i, d in enumerate(example.desirable) if d]
            if desirable_indices:
                example.sft_index = desirable_indices[0]
            else:
                example.sft_index = max(range(len(example.scores)), key=lambda i: example.scores[i])
        elif feedback_type == 'pairwise_feedback':
            # Track unique outputs to avoid duplicates
            output_to_idx = {}
            
            for pair in samples:
                example.prompt = pair['prompt']
                example.dataset_name = feedback_type

                # Add outputs if not already added
                if pair['output_A'][0]['content'] not in output_to_idx:
                    output_to_idx[pair['output_A'][0]['content']] = len(example.generations)
                    example.generations.append(pair['output_A'])
                    example.scores.append(pair['reward_A'])
                    
                if pair['output_B'][0]['content'] not in output_to_idx:
                    output_to_idx[pair['output_B'][0]['content']] = len(example.generations)
                    example.generations.append(pair['output_B'])
                    example.scores.append(pair['reward_B'])
                
                # Add preference pair as a tuple of indices
                if pair['label'] == 1:
                    example.pairs.append((output_to_idx[pair['output_A'][0]['content']], output_to_idx[pair['output_B'][0]['content']]))
                else:
                    example.pairs.append((output_to_idx[pair['output_B'][0]['content']], output_to_idx[pair['output_A'][0]['content']]))
            
            # Use highest scoring response as SFT target
            example.sft_index = max(range(len(example.scores)), key=lambda i: example.scores[i])
        else:
            for sample in samples:
                example.prompt = sample['prompt']
                example.generations.append(sample['output'])
                example.scores.append(sample['reward'])
                example.dataset_name = feedback_type
        
        data[prompt_id] = example
    
    return data


def get_shp(split: str, seed: int=0) -> Dataset:
    """
    Load the Stanford Human Preferences dataset from Huggingface and convert it into to a Dataset.

    We filter preference pairs to only keep pairs where the score ratio is at least 2 (as in original SHP).
    For this dataset, the SFT text is the first response in SHP for a given prompt. 
    This is because the globally best response cannot be inferred from SHP, but all responses are a good option because they have a positive score.
    """
    MAX_PAIRS_PER_PROMPT = 5
    MIN_SCORE_RATIO = 2

    rank0_print(f'Loading SHP dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('stanfordnlp/SHP', split=split)
    if on_rank0():
        dataset = tqdm.tqdm(dataset, desc='Processing SHP')

    data = Dataset('shp')

    for row in dataset:
        conversation = [{"role": "user", "content": row['history']}]
        scores = [row['score_A'], row['score_B']]
        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])

        if score_ratio < MIN_SCORE_RATIO and split == 'train':
            continue

        i, j = data[row['history']].num_generations(), data[row['history']].num_generations() + 1
        data[row['history']].prompt = conversation
        data[row['history']].original_prompt = row['history']
        data[row['history']].generations.append([{"role": "assistant", "content": row['human_ref_A']}])
        data[row['history']].generations.append([{"role": "assistant", "content": row['human_ref_B']}])
        data[row['history']].pairs.append((i, j) if row['labels'] == 1 else (j, i))
        data[row['history']].scores.extend(scores)
        data[row['history']].sft_index = 0  # absolute best response cannot be inferred, so just pick the first
        data[row['history']].dataset_name = 'shp'
        data[row['history']].remove_extra_spaces()

    # prevent over-fitting
    if split == 'train':
        for prompt in data:
            data[prompt].pairs = random.Random(seed).sample(data[prompt].pairs, min(MAX_PAIRS_PER_PROMPT, len(data[prompt].pairs)))

    return data


def get_hh(split: str, only_helpful=False, only_harmless=False) -> Dataset:
    """
    Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it into to a Dataset.
    For this dataset, the SFT text is the preferred response.
    
    Args:
        - split: one of 'test', 'train'
        - only_helpful: only the helpfulness data
        - only_harmless: only the harmlessness data

    Returns:   
        A Dataset instance.
    """
    if only_helpful:
        dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split, data_dir="helpful-base")
        data = Dataset('Anthropic-HH-helpful')
    elif only_harmless:
        dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split, data_dir="harmless-base")
        data = Dataset('Anthropic-HH-harmless')
    else:
        rank0_print(f'Loading HH dataset ({split} split) from Huggingface...')
        dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split)
        data = Dataset('Anthropic-HH')
        
    if on_rank0():
        dataset = tqdm.tqdm(dataset, desc='Processing HH')

    def split_prompt_and_responses(ex):
        parts = re.split(r'\n\nHuman: |\n\nAssistant: ', ex['chosen'])
        conversation = []
        for i, part in enumerate(parts[1:]):  # Skip the first empty part
            role = "user" if i % 2 == 0 else "assistant"
            conversation.append({"role": role, "content": part.strip()})
        chosen_response = conversation.pop()['content']
        rejected_response = ex['rejected'].split('\n\nAssistant: ')[-1].strip()
        return conversation, chosen_response, rejected_response

    for row in dataset:
        conversation, chosen, rejected = split_prompt_and_responses(row)
        prompt_key = ' '.join([turn['content'] for turn in conversation])  # Use full conversation as key
        i, j = data[prompt_key].num_generations(), data[prompt_key].num_generations() + 1

        data[prompt_key].prompt = conversation
        data[prompt_key].generations.append([{"role": "assistant", "content": chosen}])
        data[prompt_key].generations.append([{"role": "assistant", "content": rejected}])
        data[prompt_key].pairs.append((i, j))
        data[prompt_key].sft_index = 0

        if only_helpful:
            data[prompt_key].dataset_name = 'hh_helpful'
        elif only_harmless:
            data[prompt_key].dataset_name = 'hh_harmless'
        else:
            data[prompt_key].dataset_name = 'hh'

        data[prompt_key].remove_extra_spaces()

    return data


def get_hh_helpful(split: str) -> Dataset:
    return get_hh(split, only_helpful=True)


def get_hh_harmless(split: str) -> Dataset:
    return get_hh(split, only_harmless=True)


def get_oasst(split: str) -> Dataset:
    """
    Load the Open Assistant dataset from Huggingface and convert it into to a Dataset.
    For this dataset, the SFT text is the preferred response.

    OASST is a dataset of ranked responses (not just pairwise), but since we are working with losses that expect paired preferences, 
    turn a ranking (a, b, c, d, e) into pairwise preferences ((a,b), (b,c), (c,d), (d,e)).
    
    Args:
        - split: one of 'test', 'train'

    Returns:   
        A Dataset instance.
    """
    rank0_print(f'Loading OASST dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('OpenAssistant/oasst1', split=('validation' if split == 'test' else 'train'))
    dataset = dataset.filter(lambda x: x['lang'] == 'en')

    message_indexed_df = pd.DataFrame(dataset).set_index('message_id')
    parent_indexed_df = pd.DataFrame(dataset).set_index('parent_id')

    def get_path_to_root(node: pd.Series):
        if node['parent_id'] is None:
            return [node]
        else:
            parent = message_indexed_df.loc[node['parent_id']]
            return [node] + get_path_to_root(parent)
    
    def build_conversation(path: List[pd.Series]):
        conversation = []
        for node in reversed(path):
            role = "user" if node['role'] == 'prompter' else "assistant"
            conversation.append({"role": role, "content": node['text']})
        return conversation

    data = Dataset('OASST')

    for row in (tqdm.tqdm(dataset, desc='Processing OASST') if on_rank0() else dataset):
        if row['rank'] == 0 or row['rank'] is None:
            continue

        try:
            sibling_df = parent_indexed_df.loc[row['parent_id']]
            next_best_sibling = sibling_df[sibling_df['rank'] == (row['rank'] - 1)].iloc[0]
            path_to_root = get_path_to_root(message_indexed_df.loc[next_best_sibling['message_id']])
        except KeyError:
            continue
        except IndexError:
            continue

        conversation = build_conversation(path_to_root[1:])  # Exclude the current message
        prompt_key = json.dumps(conversation)  # Use the conversation as the key

        data[prompt_key].prompt = conversation
        data[prompt_key].generations.append([{"role": "assistant", "content": next_best_sibling['text']}])
        data[prompt_key].generations.append([{"role": "assistant", "content": row['text']}])
        data[prompt_key].pairs.append((len(data[prompt_key].generations) - 2, len(data[prompt_key].generations) - 1))
        data[prompt_key].scores.extend([next_best_sibling['rank'], row['rank']])
        data[prompt_key].dataset_name = 'oasst'
        data[prompt_key].remove_extra_spaces()
    
    return data


def get_ultrabin(split: str) -> Dataset:
    """
    Load the Ultrafeedback (binarized) dataset from Huggingface and convert it into to a Dataset.
    For this dataset, the SFT text is the preferred response.

    Args:
        - split: one of 'test', 'train'

    Returns:   
        A Dataset instance.
    """
    if split == 'train':
        split = 'train_prefs'
    elif split == 'test':
        split = 'test_prefs'
    else:
        raise ValueError("Split must be either 'train' or 'test'")
    
    rank0_print(f'Loading Ultra Binarized dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('HuggingFaceH4/ultrafeedback_binarized', split=split)
    if on_rank0():
        dataset = tqdm.tqdm(dataset, desc='Processing Ultrachat Binarized')

    data = Dataset('ultrabin')

    for row in dataset:
        # Convert the prompt into the new format
        conversation = [{"role": "user", "content": row['prompt']}]

        # Get the chosen and rejected responses
        chosen_response = row['chosen'][-1]['content']
        rejected_response = row['rejected'][-1]['content']

        # Create a unique key for this example (using the prompt)
        key = row['prompt']

        # Update the dataset
        data[key].prompt = conversation
        data[key].generations.append([{"role": "assistant", "content": chosen_response}])
        data[key].generations.append([{"role": "assistant", "content": rejected_response}])
        i, j = data[key].num_generations() - 2, data[key].num_generations() - 1
        data[key].pairs.append((i, j))
        data[key].sft_index = 0
        data[key].dataset_name = data.name
        data[key].remove_extra_spaces()

    return data


def get_ultrafeedback_armorm(split: str) -> Dataset:
    rank0_print(f'Loading ultrafeedback_armorm dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('princeton-nlp/llama3-ultrafeedback-armorm', split=split)
    if on_rank0():
        dataset = tqdm.tqdm(dataset, desc='Processing ultrafeedback armorm')

    data = Dataset('ultrafeedback_armorm')

    for row in dataset:
        # Convert the prompt into the new format
        conversation = [{"role": "user", "content": row['prompt']}]

        # Create a unique key for this example (using the prompt)
        key = row['prompt']

        # Update the dataset
        data[key].prompt = conversation
        data[key].generations.append(row['chosen'][1:])
        data[key].generations.append(row['rejected'][1:])
        i, j = data[key].num_generations() - 2, data[key].num_generations() - 1
        data[key].pairs.append((i, j))
        data[key].sft_index = 0
        data[key].dataset_name = data.name
        data[key].remove_extra_spaces()

    return data


def get_ultrachat(split: str) -> Dataset:
    rank0_print(f'Loading ultrachat dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('HuggingFaceH4/ultrachat_200k', split=f'{split}_sft')
    if on_rank0():
        dataset = tqdm.tqdm(dataset, desc='Processing ultrachat')

    data = Dataset('ultrachat')

    for row in dataset:
        key = row["prompt"]
        data[key].prompt = [row["messages"][0]]
        data[key].generations.append(row["messages"][1:])
        data[key].sft_index = 0 
        data[key].dataset_name = data.name
        data[key].remove_extra_spaces()

    return data


def get_s1k_11(split: str = "train") -> Dataset:
    """
    Load the Ultrafeedback (binarized) dataset from Huggingface and convert it into to a Dataset.
    For this dataset, the SFT text is the preferred response.

    Args:
        - split: 'train'

    Returns:   
        A Dataset instance.
    """
    if split != 'train':
        split = 'train'
        print(f"Warning: s1K-1.1 only has a 'train' split but requested '{split}' split. Using 'train' split.")
    
    rank0_print(f'Loading s1K-1.1 (s1k_11) dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('simplescaling/s1K-1.1', split=split)
    if on_rank0():
        dataset = tqdm.tqdm(dataset, desc='Processing s1K-1.1')

    data = Dataset('s1k_11')

    for row in dataset:
        # Create a unique key for this example (using the question)
        key = row['question']
        data[key].prompt = [{"role": "user", "content": row['question']}]
        data[key].generations.append([{
            "role": "assistant", 
            "content": "<|im_start|>think\n" + row['deepseek_thinking_trajectory'] + "\n<|im_start|>answer\n" + row['deepseek_attempt']
        }])
        data[key].dataset_name = data.name
        data[key].sft_index = 0
    return data
