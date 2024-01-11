# Copyright (c) 2023 Contextual AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Contains the functions for loading data.
Each function of the form get_{dataset_name} (e.g., get_shp, get_oasst, etc.) will return a dict of Example objects, indexed by the prompt for the text.

Each Example object will contain
- the prompt (formatted with config.human_prefix, config.assistant_prefix)
- a list L of generations
- the index in L of the generation that should be the finetuning target
- a list S of the scores for the generations
- for binary feedback data: pairs of indices (i,j) in L, where generation i is preferable to generation j
- for unary feedback data: whether each generation is desirable/chosen or undesirable/rejected
- whether to truncate the beginning or end if the maximum number of tokens is exceeded
- the dataset name
- the unformatted prompt (needed for alpaca)
"""

import datasets
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import re
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from utils import rank0_print, on_rank0, delete_dict
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
    pairs: List[Tuple[int, int]] = field(default_factory=list)  # for binary feedback data:: indices in responses, where i > j in pair (i,j) is a preference
    desirable: List[bool] = field(default_factory=list)         # for unary feedback data: whether the generation at the corresponding index in self.generations is desirable 
    truncation_mode: str = 'keep_end'                           # if truncation needed, keep the beginning (keep_start) or end (keep_end) (only override default for SHP)
    dataset_name: str = ''
    original_prompt: str = ''                                   # the unformatted prompt (needed to recover instruction for AlpacaEval)

    def num_generations(self):
        return len(self.generations)
    
    def remove_extra_spaces(self):
        """
        Remove double spaces in certain datasets, like Anthropic HH, to standardize spacing.
        """
        clean = lambda x: re.sub(r'[ \t]{2,}', ' ', x)
        self.prompt = clean(self.prompt)
        self.generations = list(map(clean, self.generations))


class Dataset:
    """
    A collection of Example instances, indexed by prompt.
    """
    def __init__(self, name):
        self.name = name
        self.data = defaultdict(Example)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise KeyError("key must be a string")
        
        if not isinstance(value, Example):
            raise ValueError("value must be a Example")
        
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return iter(self.data)


def get_alpacaeval(split: str, human_prefix: str, human_suffix: str, assistant_prefix: str, assistant_suffix: str) -> Dataset:
    """
    Load the AlpacaEval dataset (for evaluation only) and convert it into to a Dataset.

    Args:
        - split: must be 'test'; otherwise error will be thrown
        - human_prefix: marks start of human turn ('<|user|>' is the recommended choice and is set in config.yaml)
        - human_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)
        - assistant_prefix: marks start of human turn ('<|assistant|>' is the recommended choice and is set in config.yaml)
        - assistant_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)

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
        prompt = human_prefix + row['instruction'] + human_suffix + assistant_prefix
        data[prompt].prompt = prompt
        data[prompt].generations.append(row['output'] + assistant_suffix)
        data[prompt].dataset_name = row['dataset']
        # keep original prompt so that it can be dumped into a JSON file before running the alpacaeval command
        data[prompt].original_prompt = row['instruction']

    return data


def get_shp(split: str, human_prefix: str, human_suffix: str, assistant_prefix: str, assistant_suffix: str) -> Dataset:
    """
    Load the Stanford Human Preferences dataset from Huggingface and convert it into to a Dataset.

    We filter preference pairs to only keep pairs where the score ratio is at least 2 (as in original SHP).
    For this dataset, the SFT text is the first response in SHP for a given prompt. 
    This is because the globally best response cannot be inferred from SHP, but all responses are a good option because they have a positive score.

    As recommended in the SteamSHPs' (reward models) data cards:
        Maximum number of pairs per prompt is 5 (in the training data, to avoid overfitting).
        Minimum score ratio of preferred to dispreferred response is 2

    Args:
        - split: one of 'test', 'train'
        - human_prefix: marks start of human turn ('<|user|>' is the recommended choice and is set in config.yaml)
        - human_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)
        - assistant_prefix: marks start of human turn ('<|assistant|>' is the recommended choice and is set in config.yaml)
        - assistant_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)

    Returns:   
        A Dataset instance.
    """
    MAX_PAIRS_PER_PROMPT = 5
    MIN_SCORE_RATIO = 2

    rank0_print(f'Loading SHP dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('stanfordnlp/SHP', split=split)
    if on_rank0():
        dataset = tqdm.tqdm(dataset, desc='Processing SHP')

    data = Dataset('shp')

    for row in dataset:
        prompt = human_prefix + row['history'] + human_suffix + assistant_prefix
        responses = [row['human_ref_A'] + assistant_suffix, row['human_ref_B'] + assistant_suffix]
        scores = [row['score_A'], row['score_B']]
        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])

        if score_ratio < MIN_SCORE_RATIO and split == 'train':
            continue

        i,j = data[prompt].num_generations(), data[prompt].num_generations() + 1
        data[prompt].prompt = prompt
        data[prompt].generations.extend(responses)
        data[prompt].pairs.append((i, j) if row['labels'] == 1 else (j, i))
        data[prompt].scores.extend(scores)
        data[prompt].truncation_mode = 'keep_start' # keep start for SHP because it's single-turn with long prompts
        data[prompt].sft_index = 0  # absolute best response cannot be inferred, so just pick the first
        data[prompt].dataset_name = 'shp'
        data[prompt].remove_extra_spaces()

    # prevent over-fitting
    if split == 'train':
        for prompt in data:
            data[prompt].pairs = random.sample(data[prompt].pairs, min(MAX_PAIRS_PER_PROMPT, len(data[prompt].pairs)))

    return data


def get_hh(split: str, human_prefix: str, human_suffix: str, assistant_prefix: str, assistant_suffix: str, only_helpful = False, only_harmless = False) -> Dataset:
    """
    Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it into to a Dataset.
    For this dataset, the SFT text is the preferred response.
    
    Args:
        - split: one of 'test', 'train'
        - human_prefix: marks start of human turn ('<|user|>' is the recommended choice and is set in config.yaml)
        - human_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)
        - assistant_prefix: marks start of human turn ('<|assistant|>' is the recommended choice and is set in config.yaml)
        - assistant_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)
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
        search_term = '\n\nAssistant: '
        search_term_idx = ex['chosen'].rfind(search_term)
        prompt = ex['chosen'][:search_term_idx + len(search_term)]
        chosen_response = ex['chosen'][len(prompt):]
        rejected_response = ex['rejected'][len(prompt):]
        return prompt, chosen_response, rejected_response

    for row in dataset:
        prompt, chosen, rejected = split_prompt_and_responses(row)
        # strip trailing spaces to avoid tokenization issues
        chunks = []
        # turn doesn't always start with \n\n so watch out
        for chunk in re.split(r'\s*(Human:|Assistant:)\s+', prompt): 
            if chunk.startswith('Human'):
                chunk = re.sub(r'\s*Human:\s*', human_prefix, chunk) + human_suffix
            elif chunk.startswith('Assistant'):
                chunk = re.sub(r'\s*Assistant:\s*', assistant_prefix, chunk) + assistant_suffix
            else:
                pass

            if chunk != '':
                chunks.append(chunk)

        prompt = ''.join(chunks)
        responses = [chosen + assistant_suffix, rejected + assistant_suffix]
        i,j = data[prompt].num_generations(), data[prompt].num_generations() + 1

        data[prompt].prompt = prompt
        data[prompt].generations.extend(responses)
        data[prompt].pairs.append((i, j))
        data[prompt].sft_index = 0

        if only_helpful:
            data[prompt].dataset_name = 'hh_helpful'
        elif only_harmless:
            data[prompt].dataset_name = 'hh_harmless'
        else:
            data[prompt].dataset_name = 'hh'

        data[prompt].remove_extra_spaces()

    return data


def get_hh_helpful(split: str, human_prefix: str, human_suffix: str, assistant_prefix: str, assistant_suffix: str) -> Dataset:
    rank0_print(f'Loading helpful HH dataset ({split} split) from Huggingface...')
    return get_hh(split, human_prefix, human_suffix, assistant_prefix, assistant_suffix, only_helpful=True)


def get_hh_harmless(split: str, human_prefix: str, human_suffix: str, assistant_prefix: str, assistant_suffix: str) -> Dataset:
    rank0_print(f'Loading harmless HH dataset ({split} split) from Huggingface...')
    return get_hh(split, human_prefix, human_suffix, assistant_prefix, assistant_suffix, only_harmless=True)


def get_oasst(split: str, human_prefix: str, human_suffix: str, assistant_prefix: str, assistant_suffix: str) -> Dataset:
    """
    Load the Open Assistant dataset from Huggingface and convert it into to a Dataset.
    For this dataset, the SFT text is the preferred response.

    OASST is a dataset of ranked responses (not just pairwise), but since we are working with losses that expect paired preferences, 
    turn a ranking (a, b, c, d, e) into pairwise preferences ((a,b), (b,c), (c,d), (d,e)).
    
    Args:
        - split: one of 'test', 'train'
        - human_prefix: marks start of human turn ('<|user|>' is the recommended choice and is set in config.yaml)
        - human_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)
        - assistant_prefix: marks start of human turn ('<|assistant|>' is the recommended choice and is set in config.yaml)
        - assistant_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)

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
    
    def turn_path_to_prompt(path: List[pd.Series]):
        prompt = []
        while path != []:
            node = path.pop() # earlier messages are at end of list
            prefix = assistant_prefix if node['role'] == 'assistant' else human_prefix
            suffix = assistant_suffix if node['role'] == 'assistant' else human_suffix
            prompt.append(prefix + node['text'] + suffix)
        
        prompt.append(assistant_prefix)
        return "".join(prompt)

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

        prompt = turn_path_to_prompt(path_to_root[1:])
        responses = [next_best_sibling['text'] + assistant_suffix, row['text'] + assistant_suffix]
        i,j = data[prompt].num_generations(), data[prompt].num_generations() + 1

        data[prompt].prompt = prompt
        data[prompt].generations.extend(responses)
        data[prompt].pairs.append((i,j))
        data[prompt].scores.extend([next_best_sibling['rank'], row['rank']])
        data[prompt].dataset_name = 'oasst'
        data[prompt].remove_extra_spaces()
    
    return data


def get_ultrabin(split: str, human_prefix: str, human_suffix: str, assistant_prefix: str, assistant_suffix: str) -> Dataset:
    """
    Load the Ultrafeedback (binarized) dataset from Huggingface and convert it into to a Dataset.
    For this dataset, the SFT text is the preferred response.

    Args:
        - split: one of 'test', 'train'
        - human_prefix: marks start of human turn ('<|user|>' is the recommended choice and is set in config.yaml)
        - human_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)
        - assistant_prefix: marks start of human turn ('<|assistant|>' is the recommended choice and is set in config.yaml)
        - assistant_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)

    Returns:   
        A Dataset instance.
    """
    if split == 'train':
        split = 'train_prefs'
    elif split == 'test':
        split = 'test_prefs'
    else:
        raise ValueError()
    
    rank0_print(f'Loading Ultra Binarized dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('HuggingFaceH4/ultrafeedback_binarized', split=split)
    if on_rank0():
        dataset = tqdm.tqdm(dataset, desc='Processing Ultrachat Binarized')

    data = Dataset('ultrabin')

    for row in dataset:
        prompt = human_prefix + row['prompt'] + human_suffix + assistant_prefix
        responses = [row['chosen'][-1]['content'] + assistant_suffix, row['rejected'][-1]['content'] + assistant_suffix]

        i, j = data[prompt].num_generations(), data[prompt].num_generations() + 1
        data[prompt].prompt = prompt
        data[prompt].generations.extend(responses)
        data[prompt].pairs.append((i, j))
        data[prompt].sft_index = 0
        data[prompt].dataset_name = data.name
        data[prompt].truncation_mode = 'keep_start'
        data[prompt].remove_extra_spaces()

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
                 human_prefix: str = '\n<|user|>\n',            # marks start of human's turn
                 human_suffix: str = '',                        # marks end of human's turn
                 assistant_prefix: str = '\n<|assistant|>\n',   # marks start of assistant's turn
                 assistant_suffix: str = '',                    # marks end of assistant's turn
                 seed:int = 0,
                 **kwargs):
        
        torch.manual_seed(seed)
        random.seed(seed)

        self.tokenizer = tokenizer
        self.split = split
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.max_prompt_count = max_prompt_count
        self.kwargs = kwargs

        assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
        self.n_epochs = n_epochs
        self.epoch_idx = 0
        self.n_examples = n_examples
        
        self.full_data = {}

        for name in dataset_names:
            dataset = globals()[f"get_{name}"](split, human_prefix, human_suffix, assistant_prefix, assistant_suffix)
            self.full_data.update(dataset.data)

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
                if 'prompt' in k:
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
                if 'prompt' in k:
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    def tokenize_batch_element(self, prompt: str, generation: str, truncation_mode: str, prefix: str='target') -> Dict:
        """
        Tokenize a single batch element and truncate if prompt + generation is too long. Batch element is turned into Pytorch 
        tensors in self.collate. Create the labels for the generation, which are of length equal to the sum of the length of 
        the prompt and the generation, with -100 for the prompt tokens.

        Args:
        - prompt: the input/instruction text
        - generation: output text
        - truncation_mode: one of 'keep_start'/'keep_end' (truncate end/beginning of combined text respectively)
        - prefix: the prefix corresponding to the generation (e.g., 'chosen', 'rejected', 'target')

        Returns:
            A dict of the tokenized prompt, tokenized generation, and the concatenation of the two on all relevant elements
            (e.g., tokens, attention mask, etc.). The generation elements will have keys starting with '{prefix}_' and the
            concatenated elements will have keys starting with '{prefix}_combined_'.
        """
        prompt_token_ids = self.tokenizer.encode(prompt)
        generation_token_ids = self.tokenizer.encode(generation)

        # clip EOS token at end of input
        if len(prompt_token_ids) > 0 and prompt_token_ids[-1] == self.tokenizer.eos_token_id:
            prompt_token_ids.pop()

        # clip BOS token at start of output
        if len(generation_token_ids) > 0 and generation_token_ids[0] == self.tokenizer.bos_token_id:
            generation_token_ids.pop(0)

        # clip EOS at end of output since it will be added later anyway
        if len(generation_token_ids) > 0 and generation_token_ids[-1] == self.tokenizer.eos_token_id:
            generation_token_ids.pop()

        # if combined sequence is too long, first truncate prompt
        if (len(prompt_token_ids) + len(generation_token_ids) > self.max_length) and (len(prompt_token_ids) > self.max_prompt_length):
            if truncation_mode == 'keep_start':
                prompt_token_ids = prompt_token_ids[:self.max_prompt_length]
            elif truncation_mode == 'keep_end':
                prompt_token_ids = prompt_token_ids[-self.max_prompt_length:]
            else:
                raise ValueError(f'Unknown truncation mode: {truncation_mode}')

        # then truncate generation if needed
        if (len(prompt_token_ids) + len(generation_token_ids) > self.max_length):
            generation_token_ids = generation_token_ids[:(self.max_length - len(prompt_token_ids))]

        # reconstitute the prompt and generation
        prompt = self.tokenizer.decode(prompt_token_ids, skip_special_tokens=True)
        generation = self.tokenizer.decode(generation_token_ids, skip_special_tokens=True) + ' ' + self.tokenizer.eos_token

        batch_element = { 'prompt_text' : prompt, f'{prefix}_text': generation }

        for k,v in self.tokenizer(prompt).items():
            batch_element[f'prompt_{k}'] = v

        for k,v in self.tokenizer(generation).items():
            batch_element[f'{prefix}_{k}'] = v

        # combine the prompt and generation belonging to the same example
        batch_element.update(self.combine_prompt_and_generation(batch_element, batch_element, prefix=prefix))
  
        return batch_element

    def combine_prompt_and_generation(self, prompt_dict: Dict, generation_dict: Dict, prefix: str='target') -> Dict:
        """
        Tokenize the concatenated prompt and generation. 
        
        Note that you cannot just concatenate the input ids, attention mask, etc. after the fact -- as done 
        in the DPO repo -- because of subtle differences. For example, the ID for 'Well' corresponds to no 
        space ('Well') when at the start of a text but a space ('\n Well) when succeeding a newline. Therefore
        we could not get the correct token ID for '\nWell' by first tokenizing '\n' then 'Well' then concatenating
        the resulting tokens together.

        The prefix for each concantenated element will be f'{prefix}_combined_'.

        Args:
        - prompt_dict: dict of the prompt text, tokens, attention mask, etc.
        - generation_dict: dict of the generation text, tokens, attention mask, etc.
        - prefix: str to prepend to the the keys of the tokenized (prompt + generation)

        Returns:
            A dict of the (prompt + generation) text, tokens, attention mask, etc, along with the labels for the
            joint sequence, where the prompt token labels have been set to -100.
        """
        combined_dict = { f'{prefix}_combined_text' : prompt_dict['prompt_text'] + generation_dict[f'{prefix}_text'] }

        for k,v in self.tokenizer(prompt_dict['prompt_text'] + generation_dict[f'{prefix}_text']).items():
            combined_dict[f'{prefix}_combined_{k}'] = v

        combined_dict[f'{prefix}_labels'] = combined_dict[f'{prefix}_combined_input_ids'][:]  # contains both input and response (unpadded)
        combined_dict[f'{prefix}_labels'][:len(prompt_dict['prompt_input_ids'])] = [-100] * len(prompt_dict['prompt_input_ids'])

        return combined_dict

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
                    # control token will be None for all losses other than csft
                    example.prompt + (self.kwargs.get('chosen_control_token') or ''),
                    example.generations[example.sft_index],
                    example.truncation_mode
                )
                batch.append(batch_element)

                if len(batch) == self.batch_size:
                    example_idx += len(batch)
                    yield self.collate(batch)
                    batch = []

                    if self.n_examples is not None and example_idx >= self.n_examples:
                        rank0_print(f'Finished generating {self.n_examples} examples on {self.split} split')
                        done = True
                        break

            epoch_idx += 1
            if self.n_epochs is not None and epoch_idx >= self.n_epochs:
                done = True
                break


class ConditionalSFTDataLoader(DataLoader):
    """
    Dataloader for token-conditioned SFT, in the style of Korbak et al.'s (2023) "Pretraining Models with Human
    Feedback."

    For training, each output is prepended with a control token denoting whether it's desirable or undesirable
    (<|good|> or <|bad|> respectively). For sampling, each input is postpended with the <good> token to ensure
    that only desirable outputs are generated.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.kwargs.get('chosen_control_token') is None:
            raise KeyError("control token for chosen outputs not specified")
        
        if self.kwargs.get('rejected_control_token') is None:
            raise KeyError("control token for rejected outputs not specified")

    def get_flat_data(self, prompts):
        """
        Return a flat list of examples given a list of prompts that index self.full_data.
        Prepend the examples with the appropriate control tokens.
        """
        flat_data = []

        for prompt in prompts:
            example = self.full_data[prompt]

            if self.max_prompt_count:
                example.pairs = random.sample(example.pairs, min(self.max_prompt_count, len(example.pairs)))

            for i,j in example.pairs:
                flat_data.append((example, example.generations[i], 'chosen'))
                flat_data.append((example, example.generations[j], 'rejected'))

        return flat_data
    
    def __iter__(self):
        prompts = list(self.full_data.keys()) 
        random.shuffle(prompts) # otherwise, will be frontloaded with prompts in same domain
        flat_data = self.get_flat_data(prompts)
      
        epoch_idx = 0
        example_idx = 0
        done = False
        
        while True:
            if done: break
            random.shuffle(flat_data)

            batch = []

            for example, generation, status in flat_data:
                if status == 'chosen':
                    batch_element = self.tokenize_batch_element(example.prompt + self.kwargs["chosen_control_token"], generation, example.truncation_mode)
                else:
                    batch_element = self.tokenize_batch_element(example.prompt + self.kwargs["rejected_control_token"], generation, example.truncation_mode)

                batch_element['status'] = status
                batch.append(batch_element)

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


class SimpleKTODataLoader(DataLoader):
    """
    Legacy Dataloader for the original variant of KTO that presumes access to even number of desirable and 
    undesirable examples in each microbatch.. 

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


class UnpairedPreferenceDataLoader(DataLoader):
    """
    Dataloader for losses that do not require pairwise preferences (e.g., KTO).

    Since all the datasets have (or imply) pairwise preferences, this function assumes all preferred/dispreferred
    generations are from the desirable/undesirable conditional generations given x. 
    """
    def get_flat_data(self, prompts):
        """
        Return a flat list of examples given a list of prompts that index self.full_data.
        """
        if self.max_prompt_count:
            num_unique = sum(min(self.max_prompt_count, len(self.full_data[prompt].pairs)) for prompt in prompts)
        else:
            num_unique = sum(len(self.full_data[prompt].pairs) for prompt in prompts)

        allowed_desirable = num_unique * self.kwargs.get('frac_unique_desirable', 1.0)
        allowed_undesirable = num_unique * self.kwargs.get('frac_unique_undesirable', 1.0)
        seen_desirable = 0
        seen_undesirable = 0

        flat_data = []

        for prompt in prompts:
            example = self.full_data[prompt]

            if self.max_prompt_count:
                example.pairs = random.sample(example.pairs, min(self.max_prompt_count, len(example.pairs)))

            for i,j in example.pairs:
                if seen_desirable < allowed_desirable:
                    flat_data.append((example, example.generations[i], 'chosen'))
                    seen_desirable += 1
                
                if seen_undesirable < allowed_undesirable:
                    flat_data.append((example, example.generations[j], 'rejected'))
                    seen_undesirable += 1

        return flat_data

    def __iter__(self):
        prompts = list(self.full_data.keys()) 
        random.shuffle(prompts) # otherwise, will be frontloaded with prompts in same domain
        flat_data = self.get_flat_data(prompts)

        epoch_idx = 0
        example_idx = 0
        done = False

        while True:
            if done: break
            random.shuffle(flat_data)   # so generations in the same preference are not in the same batch
            batch = []
            example_queue = []

            for example, generation, status in flat_data:
                batch_element = self.tokenize_batch_element(example.prompt, generation, example.truncation_mode, prefix='target')
                batch_element['status'] = status 
                batch_element['truncation_mode'] = example.truncation_mode
                example_queue.append(batch_element)
                
                if len(example_queue) >= self.batch_size:
                    while len(batch) < self.batch_size:
                        batch.append(example_queue.pop(0))
                    
                if len(batch) >= self.batch_size:
                    # for estimating the KL term, match up x and y' that are not corresponding input-output pairs in the data
                    # for x_i, get a mismatched y' by just picking the subsequent y_{i+1} in the batch (desirable/undesirable status does not matter)
                    # the respective input IDs, attention mask, and so on will be prefixed by the term KL
                    indices = list(range(1, len(batch))) + [0]
                    for i in range(len(batch)):
                        batch[i].update(self.tokenize_batch_element(
                            batch[i]['prompt_text'],
                            batch[indices[i]]['target_text'],
                            batch[i]['truncation_mode'],
                            prefix='KL'
                        ))

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
            batch = []

            for example, (i,j) in flat_data:
                batch_element = {}
                batch_element.update(self.tokenize_batch_element(example.prompt, example.generations[i], example.truncation_mode, prefix='chosen'))
                batch_element.update(self.tokenize_batch_element(example.prompt, example.generations[j], example.truncation_mode, prefix='rejected'))
                batch.append(batch_element)

                if len(batch) >= self.batch_size:
                    example_idx += len(batch)
                    yield self.collate(batch)
                    batch = []

                    if self.n_examples is not None and example_idx >= self.n_examples:
                        rank0_print(f'Finished {example_idx} examples on {self.split} split')
                        done = True
                        break

            epoch_idx += 1
            if self.n_epochs is not None and epoch_idx >= self.n_epochs:
                done = True
                break
