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
- a list L of generations
- the index in L of the generation that should be the finetuning target
- a list S of the scores for the generations
- pairs of indices (i,j) in L, where generation i is preferable to generation j
- whether to truncate the beginning or end if the maximum number of tokens is exceeded
"""

import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from utils import rank0_print, on_rank0
import pandas as pd


@dataclass
class Example:
    """
    Class for an example in a preference or SFT dataset. If you want each prompt to be uniquely associated with an Example instance, save it in a dict.
    """
    prompt: str = ''                                            # prompt for the generated texts
    generations: List[str] = field(default_factory=list)        # list of generations
    sft_index: int = -1                                         # which response in generations should be generated for SFT
    scores: List[float] = field(default_factory=list)           # score for each generation
    pairs: List[Tuple[int, int]] = field(default_factory=list)  # indices in responses, where i > j in pair (i,j) is a preference
    truncation_mode: str = 'keep_start'                         # if truncation needed, keep the beginning (keep_start) or end (keep_end)

    def num_generations(self):
        return len(self.generations)


def get_shp(split: str) -> Dict[str, Example]:
    """
    Load the Stanford Human Preferences dataset from Huggingface and convert it a dictionary of Examples. 

    We filter preference pairs to only keep pairs where the score ratio is at least 2 (as in original SHP).
    For this dataset, the SFT text is the first response in SHP for a given prompt. 
    This is because the globally best response cannot be inferred from SHP, but all responses are a good option because they have a positive score.

    As recommended in the SteamSHPs' (reward models) data cards:
        Maximum number of pairs per prompt is 5 (in the training data, to avoid overfitting).
        Minimum score ratio of preferred to dispreferred response is 2, t 

    Args:
        - split: one of 'test', 'train'

    Returns:   
        A dictionary mapping prompts to Examples.
    """
    MAX_PAIRS_PER_PROMPT = 5
    MIN_SCORE_RATIO = 2

    rank0_print(f'Loading SHP dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('stanfordnlp/SHP', split=split)
    if on_rank0():
        dataset = tqdm.tqdm(dataset, desc='Processing SHP')

    data = defaultdict(Example)

    for row in dataset:
        prompt = f"\n\nHuman: {row['history']} \n\nAssistant:"
        responses = [' ' + row['human_ref_A'], ' ' + row['human_ref_B']]
        scores = [row['score_A'], row['score_B']]
        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])

        if score_ratio < MIN_SCORE_RATIO and split == 'train':
            continue

        i,j = data[prompt].num_generations(), data[prompt].num_generations() + 1
        data[prompt].prompt = prompt
        data[prompt].generations.extend(responses)
        data[prompt].pairs.append((i, j) if row['labels'] == 1 else (j, i))
        data[prompt].scores.extend(scores)
        data[prompt].truncation_mode = 'keep_start'
        # absolute best response cannot be inferred, so just pick the first
        data[prompt].sft_index = 0

    # prevent over-fitting
    if split == 'train':
        for prompt in data:
            data[prompt].pairs = random.sample(data[prompt].pairs, min(MAX_PAIRS_PER_PROMPT, len(data[prompt].pairs)))

    return data


def get_hh(split: str, only_helpful = False, only_harmless = False) -> Dict[str, Example]:
    """
    Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it to a dictionary of Examples.
    For this dataset, the SFT text is the preferred response.
    
    Args:
        - split: one of 'test', 'train'
        - only_helpful: only the helpfulness data
        - only_harmless: only the harmlessness data

    Returns:   
        A dictionary mapping prompts to Examples.
    """
    if only_helpful:
        dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split, data_dir="helpful-base")
    elif only_harmless:
        dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split, data_dir="harmless-base")
    else:
        rank0_print(f'Loading HH dataset ({split} split) from Huggingface...')
        dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split)
        
    if on_rank0():
        dataset = tqdm.tqdm(dataset, desc='Processing HH')

    def split_prompt_and_responses(ex):
        search_term = '\n\nAssistant:'
        search_term_idx = ex['chosen'].rfind(search_term)
        prompt = ex['chosen'][:search_term_idx + len(search_term)]
        chosen_response = ex['chosen'][len(prompt):]
        rejected_response = ex['rejected'][len(prompt):]
        return prompt, chosen_response, rejected_response

    data = defaultdict(Example)

    for row in dataset:
        prompt, chosen, rejected = split_prompt_and_responses(row)
        responses = [chosen, rejected]
        i,j = data[prompt].num_generations(), data[prompt].num_generations() + 1

        data[prompt].prompt = prompt
        data[prompt].generations.extend(responses)
        data[prompt].pairs.append((i, j))
        data[prompt].sft_index = 0
        data[prompt].truncation_mode = 'keep_end'

    return data


def get_hh_helpful(split: str) -> Dict[str, Example]:
    rank0_print(f'Loading helpful HH dataset ({split} split) from Huggingface...')
    return get_hh(split, only_helpful=True)


def get_hh_harmless(split: str) -> Dict[str, Example]:
    rank0_print(f'Loading harmless HH dataset ({split} split) from Huggingface...')
    return get_hh(split, only_harmless=True)


def get_oasst(split: str) -> Dict[str, Example]:
    """
    Load the Open Assistant dataset from Huggingface and convert it to a dictionary of Examples.
    For this dataset, the SFT text is the preferred response.
    OASST is a dataset of ranked responses (not just pairwise), but since we are working with losses that expect paired preferences, 
    turn a ranking (a, b, c, d, e) into pairwise preferences ((a,b), (b,c), (c,d), (d,e)).
    
    Args:
        - split: one of 'test', 'train'

    Returns:   
        A dictionary mapping prompts to Examples.
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
    
    def turn_path_to_prompt(path: List[pd.Series]):
        prompt = []
        while path != []:
            node = path.pop() # earlier messages are at end of list
            role = 'Assistant' if node['role'] == 'assistant' else 'Human'
            prompt.append(f"\n\n{role}: {node['text']}")
        
        prompt.append('\n\nAssistant:')
        return " ".join(prompt)

    data = defaultdict(Example)

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

        prompt = turn_path_to_prompt(path_to_root[1:])
        responses = [' ' + next_best_sibling['text'], ' ' + row['text']]
        i,j = data[prompt].num_generations(), data[prompt].num_generations() + 1

        data[prompt].prompt = prompt
        data[prompt].generations.extend(responses)
        data[prompt].pairs.append((i,j))
        data[prompt].truncation_mode = 'keep_start'
        data[prompt].scores.extend([next_best_sibling['rank'], row['rank']])
    
    return data


class DataLoader:
    """
    The base data loader class, similar to the one from the DPO repo.
    Subclass this and overwrite the __iter__ method as needed, since the batcch elements will be different depending
    on whether you're doing SFT, aligning with a pairwise loss like DPO, or alignment with a unary loss like KTO. 
    """
    def __init__(self, 
                 dataset_names: List[str],      # e.g., ['shp', 'oasst']; should have  get_{name} method in this file
                 tokenizer,                     # Huggingface tokenizer object
                 split: str = 'train',
                 batch_size: int = 1,
                 max_length: int = 512,         # max length of prompt + response
                 max_prompt_length: int = 128,  # max length of prompt alone
                 max_prompt_count: int = None,
                 n_epochs: Optional[int] = None,
                 n_examples: Optional[int] = None,
                 seed:int = 0):
        
        torch.manual_seed(seed)
        random.seed(seed)

        self.tokenizer = tokenizer
        self.split = split
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.max_prompt_count = max_prompt_count

        assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
        self.n_epochs = n_epochs
        self.epoch_idx = 0
        self.n_examples = n_examples
        
        self.full_data = {}

        for name in dataset_names:
            data = globals()[f"get_{name}"](split)
            self.full_data.update(data)

    def collate(self, batch: Dict[str, List]) -> Dict:
        """
        Takes a list of examples (dicts, where values are lists of ints [tokens] or strings [the original texts]) and returns a batch of examples,
        PyTorch tensors padded to the maximum length. Strings are passed through.
        """
        if self.tokenizer.pad_token_id is None:
            raise Exception("tokenizer's pad_token_id is not specified")
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                    # flip prompt so that you are padding to the beginning
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = self.tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    def tokenize_batch_element(self, prompt: str, generation: str, truncation_mode: str, prefix: str='chosen') -> Dict:
        """
        Tokenize a single batch element.
        
        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation in case the prompt + generation is too long. 
        First we truncate the prompt; if we're still too long, we truncate the generation.
        
        We also create the labels for the generation, which are of length equal to the sum of the length of the prompt and the generation, with -100 for the prompt tokens.
        """
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        generation_tokens = self.tokenizer(generation, add_special_tokens=False)

        generation_tokens['input_ids'].append(self.tokenizer.eos_token_id)
        generation_tokens['attention_mask'].append(1)

        # if combined sequence is too long, first truncate prompt
        if (len(prompt_tokens['input_ids']) + len(generation_tokens['input_ids']) > self.max_length) and (len(prompt_tokens['input_ids']) > self.max_prompt_length):
            if truncation_mode == 'keep_start':
                prompt_tokens = {k: v[:self.max_prompt_length] for k, v in prompt_tokens.items()}
            elif truncation_mode == 'keep_end':
                prompt_tokens = {k: v[-self.max_prompt_length:] for k, v in prompt_tokens.items()}
            else:
                raise ValueError(f'Unknown truncation mode: {truncation_mode}')

        # then truncate generation if needed
        if (len(prompt_tokens['input_ids']) + len(generation_tokens['input_ids']) > self.max_length):
            generation_tokens = {k: v[:(self.max_length - len(prompt_tokens['input_ids']))] for k, v in generation_tokens.items()}

        # Create labels
        combined_sequence_tokens = {k: prompt_tokens[k] + generation_tokens[k] for k in generation_tokens}
        combined_sequence_tokens['labels'] = combined_sequence_tokens['input_ids'][:]  # contains both input and response (unpadded)
        combined_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

        batch = {}

        batch['prompt'] = prompt
        batch['generation'] = generation
        batch['combined'] = prompt + generation

        # this is to preserve the legacy use of the 'chosen'/'rejected' prefixes from the DPO repo
        for k, toks in {'combined': combined_sequence_tokens, 'prompt': prompt_tokens}.items():
            for type_key, tokens in toks.items():
                if type_key == 'token_type_ids':
                    continue

                if k == 'combined' and prefix != '':
                    batch[f'{prefix}_{type_key}'] = tokens  # e.g., key is 'chosen_input_ids'
                else:
                    batch[f'{k}_{type_key}'] = tokens  

        return batch

    def __iter__(self):
        """Create a flat version of the data and yield batches."""
        raise NotImplementedError
    

class SFTDataLoader(DataLoader):
    """
    Dataloader for supervised fine-tuning.
    """
    def __iter__(self):
        flat_data = []
        prompts = list(self.full_data.keys())
        random.shuffle(prompts) # otherwise, will be frontloaded with prompts in same domain

        for prompt in prompts:
            flat_data.append(self.full_data[prompt])

        epoch_idx = 0
        example_idx = 0
        done = False
        
        while True:
            if done: break
            random.shuffle(flat_data)

            batch = []

            for example in flat_data:
                batch_element = self.tokenize_batch_element(
                    example.prompt,
                    example.generations[example.sft_index],
                    example.truncation_mode
                )
                batch.append(batch_element)
                example_idx += 1

                if len(batch) == self.batch_size:
                    yield self.collate(batch)

                    if self.n_examples is not None and example_idx >= self.n_examples:
                        rank0_print(f'Finished generating {self.n_examples} examples on {self.split} split')
                        done = True
                        break

                    batch = []

            epoch_idx += 1
            if self.n_epochs is not None and epoch_idx >= self.n_epochs:
                done = True
                break


class UnpairedPreferenceDataLoader(DataLoader):
    """
    Dataloader for losses that do not require pairwise preferences (e.g., KTO).
    Since all the datasets have (or imply) pairwise preferences, this function assumes all preferred/dispreferred
    generations are from the desirable/undesirable conditional generations given x. 

    Each batch contains half (x, desired output y) and half (x, undesired output y), where no x should appear 
    twice because of shuffling. The desirable and undesirable examples are interleaved in the batch (e.g.,
    [desirable, undesirable, desirable, ...]).
    """
    def __iter__(self):
        flat_data = []
        prompts = list(self.full_data.keys()) 
        random.shuffle(prompts) # otherwise, will be frontloaded with prompts in same domain

        for prompt in prompts:
            example = self.full_data[prompt]

            if self.max_prompt_count:
                example.pairs = random.sample(example.pairs, min(self.max_prompt_count, len(example.pairs)))

            for i,j in example.pairs:
                flat_data.append((example, example.generations[i], 'chosen'))
                flat_data.append((example, example.generations[j], 'rejected'))

        epoch_idx = 0
        example_idx = 0
        done = False

        while True:
            if done: break
            random.shuffle(flat_data)   # so generations in the same preference are not in the same batch
            prev_example = None
            batch = []

            chosen_example_queue, rejected_example_queue = [], [] 
            quota = self.batch_size // 2

            for example, generation, status in flat_data:
                batch_element = self.tokenize_batch_element(example.prompt, generation, example.truncation_mode)
                batch_element['status'] = status 
                prev_example = example

                if status == 'chosen':
                    chosen_example_queue.append(batch_element)
                else:
                    rejected_example_queue.append(batch_element)

                # only flush queues when you can get an even number of chosen and rejected examples
                # weave together chosen and rejected examples one after the other to prevent per-device microbatch from being all chosen or all rejected
                if len(chosen_example_queue) >= quota and len(rejected_example_queue) >= quota:
                    while len(batch) < self.batch_size:
                        batch.append(chosen_example_queue.pop(0))
                        batch.append(rejected_example_queue.pop(0))
                    
                if len(batch) >= self.batch_size:
                    example_idx += len(batch)
                    yield self.collate(batch)
                    batch = []

                    if self.n_examples is not None and example_idx >= self.n_examples:
                        rank0_print(f'Finished generating {example_idx} examples on {self.split} split')
                        done = True
                        break

            epoch_idx += 1
            if self.n_epochs is not None and epoch_idx >= self.n_epochs:
                done = True
                break
        

class PairedPreferenceDataLoader(DataLoader):
    """
    Dataloader for losses that do require pairwise preferences (e.g., DPO).
    """
    def __iter__(self):
        flat_data = []
        prompts = list(self.full_data.keys())
        random.shuffle(prompts) # otherwise, will be frontloaded with prompts in same domain

        for prompt in prompts:
            example = self.full_data[prompt]

            if self.max_prompt_count:
                example.pairs = random.sample(example.pairs, min(self.max_prompt_count, len(example.pairs)))

            for pair in example.pairs:
                flat_data.append((example, pair))
         
        epoch_idx = 0
        example_idx = 0
        done = False

        while True:
            if done: break
            random.shuffle(flat_data)
            prev_example = None
            batch = []

            for example, (i,j) in flat_data:
                batch_element = {}
                batch_element.update(self.tokenize_batch_element(example.prompt, example.generations[i], example.truncation_mode, prefix='chosen'))
                batch_element.update(self.tokenize_batch_element(example.prompt, example.generations[j], example.truncation_mode, prefix='rejected'))
                batch.append(batch_element)
                example_idx += 1
                prev_example = example

                if len(batch) >= self.batch_size:
                    yield self.collate(batch)
                    if self.n_examples is not None and example_idx >= self.n_examples:
                        rank0_print(f'Finished {example_idx} examples on {self.split} split')
                        done = True
                        break

                    batch = []

            epoch_idx += 1
            if self.n_epochs is not None and epoch_idx >= self.n_epochs:
                done = True
                break