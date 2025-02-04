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
- for preference feedback data: pairs of indices (i,j) in L, where generation i is preferable to generation j
- for binary feedback data: whether each generation is desirable/chosen or undesirable/rejected
- whether to truncate the beginning or end if the maximum number of tokens is exceeded
- the dataset name
- the unformatted prompt
"""

import train.data as data_module
import torch
import random
import json
from typing import Dict, List, Optional, Tuple
import numpy as np


class DataLoader:
    """
    The base data loader class, similar to the one from the DPO repo.
    Subclass this and overwrite the __iter__ method as needed, since the batch elements will be different depending
    on whether you're doing SFT, aligning with a pairwise loss like DPO, or alignment with an unpaired loss like KTO. 
    """
    def __init__(self, 
                 dataset_names: List[str],
                 tokenizer,
                 process_index: int = 0,
                 num_processes: int = 1,
                 split: str = 'train',
                 microbatch_size: int = 1,
                 max_length: int = 512,
                 max_prompt_length: int = 128,
                 max_prompt_count: int = None,
                 n_epochs: Optional[int] = None,
                 n_examples: Optional[int] = None,
                 seed: int = 0,
                 control_tokens: Dict = {},
                 **kwargs):
        
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.tokenizer = tokenizer
        self.process_index = process_index
        self.num_processes = num_processes
        self.control_tokens = control_tokens
        self.split = split
        self.microbatch_size = microbatch_size
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.max_prompt_count = max_prompt_count
        self.kwargs = kwargs

        assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
        self.n_epochs = n_epochs
        self.epoch_idx = 0
        self.n_examples = n_examples
        
        self.full_data = {} # a dict of Examples

        for name in dataset_names:
            if hasattr(data_module, f"get_{name}"):
                dataset = getattr(data_module, f"get_{name}")(split)
                self.full_data.update(dataset.data)
            else:
                try:
                    with open(name, 'r') as f:
                        data = json.load(f)

                        if data[0]['type'] == 'sample':
                            dataset = data_module.get_sampled_data(name, split)
                        elif data[0]['type'].endswith('feedback'):
                            dataset = data_module.get_feedback(name, split)
                        else:
                            raise IOError("unrecognized data type")
                        
                        self.full_data.update(dataset.data)
                except:
                    raise IOError(f"could not load {name}; neither a local file or a downloadable dataset supported by train.data")

        self.num_training_steps = self.get_num_training_steps()

    def collate(self, batch: Dict[str, List]) -> Dict:
        """
        Takes a list of examples and returns a batch of examples with consistent padding across all processes.
        Uses a fixed maximum length for padding to ensure consistency across batches and processes.
        """
        if self.tokenizer.pad_token_id is None:
            raise Exception("tokenizer's pad_token_id is not specified")
        
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

                # Always pad to max_length for consistency across processes
                max_len = self.max_prompt_length if 'prompt' in k else self.max_length

                padded_sequences = []
                for seq in to_pad:
                    if len(seq) > max_len:
                        padded_seq = seq[:max_len]
                    else:
                        padding_size = max_len - len(seq)
                        padding = torch.full((padding_size,), padding_value, dtype=seq.dtype)
                        padded_seq = torch.cat([seq, padding])
                    padded_sequences.append(padded_seq)

                padded_batch[k] = torch.stack(padded_sequences)
                if 'prompt' in k:
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    def tokenize_batch_element(self, conversation: List[Dict[str, str]], generation: List[Dict[str, str]], prefix: str='target') -> Dict:
        """
        Tokenize a single batch element and truncate if prompt + generation is too long. Batch element is turned into Pytorch 
        tensors in self.collate. Create the labels for the generation, which are of length equal to the sum of the length of 
        the prompt and the generation, with -100 for the prompt tokens.

        Args:
        - conversation: list of previous turns, each resembling dict {"role": "assistant", "content": generation}
        - generation: list of current turns, each resembling dict {"role": "assistant", "content": generation}
        - prefix: the prefix corresponding to the generation (e.g., 'chosen', 'rejected', 'target')

        Returns:
            A dict of the tokenized prompt and the concatenation of the two on all relevant elements (e.g., tokens, 
            attention mask, etc.). The generation elements will have keys starting with '{prefix}_' and the concatenated 
            elements will have keys starting with '{prefix}_combined_'. 'prompt' will map to the raw conversation history,
            as a list of dicts, and the prefix key alone will map to the untemplated output.
        """
        untruncated_prompt_string = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True) # for inference-time generation
        
        filter_out_bos_eos = lambda x: [ t for t in x if t not in [ self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id] ]
        # truncate the prompt if necessary
        total_length = 0

        # truncate history to fit in self.max_prompt_length
        for i, turn in enumerate(conversation):
            content_token_ids = filter_out_bos_eos(self.tokenizer.encode(turn['content']))
            # we're only modifying the text in content but need to consider the formatted length
            templated_length = len(self.tokenizer.apply_chat_template([turn], tokenize=True, add_generation_prompt=True))
            
            if total_length + templated_length > self.max_prompt_length:
                turn['content'] = self.tokenizer.decode(content_token_ids[:self.max_prompt_length - (total_length + templated_length)])
                total_length = self.max_prompt_length
                break
            else:
                total_length += templated_length

        untruncated_conversation = conversation
        conversation = conversation[:(i+1)]

        # truncate the generation if necessary 
        for i, turn in enumerate(generation):
            content_token_ids = filter_out_bos_eos(self.tokenizer.encode(turn['content']))
            # we're only modifying the text in content but need to consider the formatted length
            templated_length = len(self.tokenizer.apply_chat_template([turn], tokenize=True, add_generation_prompt=False))
            
            if total_length + templated_length > self.max_length:
                turn['content'] = self.tokenizer.decode(content_token_ids[:self.max_length - (total_length + templated_length)])
                total_length = self.max_length
                break
            else:
                total_length += templated_length

        generation = generation[:(i+1)]

        tokenized_prompt = self.tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True)
        tokenized_prompt_and_generation_string = self.tokenizer.apply_chat_template(conversation + generation, tokenize=False, add_generation_prompt=False)
        tokenized_prompt_and_generation = self.tokenizer.apply_chat_template(
            conversation + generation, 
            tokenize=True, 
            add_generation_prompt=False
        )

        # Prepare the batch element
        batch_element = {
            'untruncated_conversation': untruncated_conversation,
            'prompt': conversation,
            f'{prefix}': generation,
            'prompt_text': untruncated_prompt_string,
            'prompt_input_ids': tokenized_prompt,
            'prompt_attention_mask': [1] * len(tokenized_prompt),
            f'{prefix}_text': self.tokenizer.apply_chat_template(generation, tokenize=False),
            f'{prefix}_combined_text': tokenized_prompt_and_generation_string,
            f'{prefix}_combined_input_ids': tokenized_prompt_and_generation,
            f'{prefix}_combined_attention_mask': [1] * len(tokenized_prompt_and_generation),
        }

        # Prepare labels
        tokenized_prompt = self.tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True)
        if tokenized_prompt[-1] in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]:
            tokenized_prompt.pop()
        
        labels = tokenized_prompt_and_generation[:]
        labels[:len(tokenized_prompt)] = [-100] * len(tokenized_prompt)
        batch_element[f'{prefix}_labels'] = labels

        return batch_element
    
    def __iter__(self):
        """Create a flat version of the data and yield batches."""
        raise NotImplementedError

    def get_num_training_steps(self):
        """Get the number of training steps."""
        raise NotImplementedError
    

class SFTDataLoader(DataLoader):
    """
    Dataloader for supervised fine-tuning.
    """
    def __iter__(self):
        flat_data = []
        prompts = list(self.full_data.keys())
        
        for prompt in prompts:
            flat_data.append(self.full_data[prompt])

        if self.num_processes == 1: # for eval usually
            usable_size = len(flat_data)
        else: # to avoid hanging with uneven batches
            global_batch_size = int(self.num_processes * self.microbatch_size)
            usable_size = len(flat_data) // global_batch_size * global_batch_size
        
        self.rng.shuffle(flat_data)
        flat_data = [d for i, d in enumerate(flat_data[:usable_size]) if i % self.num_processes == self.process_index]

        epoch_idx = 0
        example_idx = 0
        done = False
        
        while True:
            if done: break
            self.rng.shuffle(flat_data)

            batch = []

            for example in flat_data:
                # Assuming example.prompt is now a list of conversation turns
                conversation = example.prompt
                if not isinstance(conversation[0], dict):
                    # Convert to the new format if it's not already
                    conversation = [{"role": "user", "content": conversation[0]}]
                    for i, message in enumerate(conversation[1:]):
                        role = "assistant" if i % 2 == 0 else "user"
                        conversation.append({"role": role, "content": message})

                # Get the target generation (last turn from assistant)
                target_generation = example.generations[example.sft_index]

                # Add control token if specified
                if self.control_tokens.get('chosen'):
                    target_generation = self.control_tokens['chosen'] + target_generation

                batch_element = self.tokenize_batch_element(
                    conversation,
                    target_generation,
                )
                batch_element['original_prompt'] = example.original_prompt
                batch_element['prompt_id'] = example.prompt_id
                batch_element['dataset_name'] = example.dataset_name
                batch.append(batch_element)

                if len(batch) == self.microbatch_size:
                    example_idx += len(batch) * self.num_processes
                    yield self.collate(batch)
                    batch = []

                    if self.n_examples is not None and example_idx >= self.n_examples:
                        print(f'Finished generating {self.n_examples} examples on {self.split} split')
                        done = True
                        break

            if self.num_processes == 1 and batch != []: # flush for eval, sampling
                yield self.collate(batch) 
                batch = []

            epoch_idx += 1
            if self.n_epochs is not None and epoch_idx >= self.n_epochs:
                done = True
                break

    def get_num_training_steps(self):
        """Get the number of training steps."""
        return len(self.full_data) // self.num_processes


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

        if self.control_tokens.get('chosen') is None:
            raise KeyError("control token for chosen outputs not specified")
        
        if self.control_tokens.get('rejected') is None:
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
                example.pairs = self.rng.sample(example.pairs, min(self.max_prompt_count, len(example.pairs)))

            for i,j in example.pairs:
                flat_data.append((example, example.generations[i], 'chosen'))
                flat_data.append((example, example.generations[j], 'rejected'))

        return flat_data
    
    def __iter__(self):
        prompts = list(self.full_data.keys())
        flat_data = self.get_flat_data(prompts)

        if self.num_processes == 1: # for eval usually
            usable_size = len(flat_data)
        else: # to avoid hanging with uneven batches
            global_batch_size = int(self.num_processes * self.microbatch_size)
            usable_size = len(flat_data) // global_batch_size * global_batch_size
        
        self.rng.shuffle(flat_data) # shuffle before splitting across processes, otherwise some processes will only get chosen examples
        flat_data = [d for i, d in enumerate(flat_data[:usable_size]) if i % self.num_processes == self.process_index]
      
        epoch_idx = 0
        example_idx = 0
        done = False
        
        while True:
            if done: break
            self.rng.shuffle(flat_data)

            batch = []

            for example, generation, status in flat_data:
                # Convert prompt to conversation format if it's not already
                conversation = example.prompt
                if not isinstance(conversation[0], dict):
                    conversation = [{"role": "user", "content": conversation[0]}]
                    for i, message in enumerate(conversation[1:]):
                        role = "assistant" if i % 2 == 0 else "user"
                        conversation.append({"role": role, "content": message})

                # Add control token to the generation
                if status == 'chosen':
                    conditioned_generation = self.control_tokens["chosen"] + generation
                else:
                    conditioned_generation = self.control_tokens["rejected"] + generation

                batch_element = self.tokenize_batch_element(
                    conversation,
                    conditioned_generation,
                )
                batch_element['status'] = status
                batch_element['prompt_id'] = example.prompt_id
                batch.append(batch_element)

                if len(batch) >= self.microbatch_size:
                    example_idx += len(batch) * self.num_processes
                    yield self.collate(batch)
                    batch = []

                    if self.n_examples is not None and example_idx >= self.n_examples:
                        print(f'Finished generating {example_idx} examples on {self.split} split')
                        done = True
                        break

            if self.num_processes == 1 and batch != []: # flush for eval, sampling
                yield self.collate(batch)
                batch = []

            epoch_idx += 1
            if self.n_epochs is not None and epoch_idx >= self.n_epochs:
                done = True
                break

    def get_num_training_steps(self):
        max_prompt_count = min(float("inf"), self.max_prompt_count) if self.max_prompt_count else float("inf")
        num_pairs = int(sum(min(max_prompt_count, len(example.pairs)) for _, example in self.full_data.items()))
        num_training_steps = num_pairs * 2
        return num_training_steps // self.num_processes


class UnpairedPreferenceDataLoader(DataLoader):
    """
    Dataloader for losses that do not require pairwise preferences (e.g., KTO).

    This assumes that if an example has no pairs, then it is naturally unpaired, using the 'desirable' field
    to infer its label. If an example has pairs, then it is assumed to be from a naturally paired dataset, and 
    the preferred/dispreferred generations are from the desirable/undesirable conditional generations given x. 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.microbatch_size * self.num_processes <= 1:
            raise ValueError("can't use batch size of 1 with UnpairedPreferenceDataLoader")
        
    def get_flat_data(self, prompts):
        """
        Return a flat list of examples given a list of prompts that index self.full_data.
        """
        if self.max_prompt_count:
            num_unique = sum(min(self.max_prompt_count, len(self.full_data[prompt].generations)) for prompt in prompts)
        else:
            num_unique = sum(len(self.full_data[prompt].generations) for prompt in prompts)

        allowed_desirable = num_unique * self.kwargs.get('frac_unique_desirable', np.inf)
        allowed_undesirable = num_unique * self.kwargs.get('frac_unique_undesirable', np.inf)
        seen_desirable = 0
        seen_undesirable = 0

        flat_data = []

        for prompt in prompts:
            example = self.full_data[prompt]

            # naturally unpaired feedback
            if example.pairs == [] and example.desirable != []:
                for i in range(len(example.desirable)):
                    if seen_desirable < allowed_desirable and example.desirable[i]:
                        flat_data.append((example, example.generations[i], 'chosen'))
                        seen_desirable += 1

                    if seen_undesirable < allowed_undesirable and not example.desirable[i]:
                        flat_data.append((example, example.generations[i], 'rejected'))
                        seen_undesirable += 1
            # getting unpaired data out of pairs
            elif example.pairs != []:
                if self.max_prompt_count:
                    example.pairs = self.rng.sample(example.pairs, min(self.max_prompt_count, len(example.pairs)))

                for i,j in example.pairs:
                    if seen_desirable < allowed_desirable:
                        flat_data.append((example, example.generations[i], 'chosen'))
                        seen_desirable += 1
                    
                    if seen_undesirable < allowed_undesirable:
                        flat_data.append((example, example.generations[j], 'rejected'))
                        seen_undesirable += 1
            else:
                raise IOError("data is neither paired nor has desirability labels")

        return flat_data

    def __iter__(self):
        prompts = list(self.full_data.keys())
        flat_data = self.get_flat_data(prompts)

        if self.num_processes == 1: # for eval usually
            usable_size = len(flat_data)
        else: # to avoid hanging with uneven batches
            global_batch_size = int(self.num_processes * self.microbatch_size)
            usable_size = len(flat_data) // global_batch_size * global_batch_size
        
        self.rng.shuffle(flat_data) # shuffle before splitting across processes, otherwise some processes will only get chosen examples
        flat_data = [d for i, d in enumerate(flat_data[:usable_size]) if i % self.num_processes == self.process_index]

        epoch_idx = 0
        example_idx = 0
        done = False

        while True:
            if done: break
            self.rng.shuffle(flat_data)   # so generations in the same preference are not in the same batch
            batch = []
            example_queue = []

            for example, generation, status in flat_data:
                batch_element = self.tokenize_batch_element(example.prompt, generation, prefix='target')
                batch_element['status'] = status 
                batch_element['conversation'] = example.prompt
                batch_element['generation'] = generation
                batch_element['prompt_id'] = example.prompt_id
                example_queue.append(batch_element)
                
                if len(example_queue) >= self.microbatch_size:
                    while len(batch) < self.microbatch_size:
                        batch.append(example_queue.pop(0))
                    
                if len(batch) >= self.microbatch_size:
                    # for estimating the KL term, match up x and y' that are not corresponding input-output pairs in the data
                    # for x_i, get a mismatched y' by just picking the subsequent y_{i+1} in the batch (desirable/undesirable status does not matter)
                    # the respective input IDs, attention mask, and so on will be prefixed by the term KL
                    indices = list(range(1, len(batch))) + [0]
                    for i in range(len(batch)):
                        batch[i].update(self.tokenize_batch_element(
                            batch[i]['conversation'],
                            batch[indices[i]]['generation'],
                            prefix='KL'
                        ))

                    example_idx += len(batch) * self.num_processes
                    yield self.collate(batch)
                    batch = []

                    if self.n_examples is not None and example_idx >= self.n_examples:
                        print(f'Finished generating {example_idx} examples on {self.split} split')
                        done = True
                        break

            if self.num_processes == 1 and batch != []: # flush for eval, sampling
                yield self.collate(batch)
                batch = []

            epoch_idx += 1
            if self.n_epochs is not None and epoch_idx >= self.n_epochs:
                done = True
                break

    def get_num_training_steps(self):
        max_prompt_count = min(float("inf"), self.max_prompt_count) if self.max_prompt_count else float("inf")
        num_pairs = int(sum(min(max_prompt_count, len(example.pairs)) for _, example in self.full_data.items()))
        num_training_steps = num_pairs * self.kwargs.get('frac_unique_desirable', 1.0) + num_pairs * self.kwargs.get('frac_unique_undesirable', 1.0)
        return num_training_steps // self.num_processes


class ScoreDataLoader(UnpairedPreferenceDataLoader):
    def get_flat_data(self, prompts):
        """
        Return a flat list of examples given a list of prompts that index self.full_data.
        Assumes that there are a list of scores.
        """
        flat_data = []
        prev_status = 'rejected'

        for prompt in prompts:
            example = self.full_data[prompt]

            if self.max_prompt_count:
                example.pairs = self.rng.sample(example.pairs, min(self.max_prompt_count, len(example.pairs)))

            # for oasst, lower scores are better, so rank 0 is the best response and rank n is the worst
            if prev_status == 'rejected':
                flat_data.append((example, example.generations[np.argmin(example.scores)], 'chosen'))
            else:
                flat_data.append((example, example.generations[np.argmax(example.scores)], 'rejected'))

            prev_status = flat_data[-1][-1]

        return flat_data


class HalfPrefDataLoader(UnpairedPreferenceDataLoader):
    """
    Dataloader for training on only one output per input.
    This throws out at least half the data (more than half if there are multiple pairs per input).
    For this reason, this should ONLY be used for training.
    """
    def get_flat_data(self, prompts):
        """
        Return a flat list of examples given a list of prompts that index self.full_data.
        Only use one preference pair per input.
        """
        flat_data = []
        prev_status = 'rejected'

        for prompt in prompts:
            example = self.full_data[prompt]

            if self.max_prompt_count:
                example.pairs = self.rng.sample(example.pairs, min(self.max_prompt_count, len(example.pairs)))

            for i,j in example.pairs:
                if prev_status == 'rejected':
                    flat_data.append((example, example.generations[i], 'chosen'))
                else:
                    flat_data.append((example, example.generations[j], 'rejected'))

                prev_status = flat_data[-1][-1]
                break # only use one pair

        return flat_data


class PairedPreferenceDataLoader(DataLoader):
    """
    Dataloader for losses that do require pairwise preferences (e.g., DPO).
    """
    def __iter__(self):
        prompts = list(self.full_data.keys())
        flat_data = []

        for prompt in prompts:
            example = self.full_data[prompt]

            if self.max_prompt_count:
                example.pairs = self.rng.sample(example.pairs, min(self.max_prompt_count, len(example.pairs)))

            for pair in example.pairs:
                flat_data.append((example, pair))

        if self.num_processes == 1: # for eval, sampling
            usable_size = len(flat_data)
        else: # to avoid hanging with uneven batches
            global_batch_size = int(self.num_processes * self.microbatch_size)
            usable_size = len(flat_data) // global_batch_size * global_batch_size

        self.rng.shuffle(flat_data) # shuffle before splitting across processes, otherwise some processes will only get chosen examples
        flat_data = [d for i, d in enumerate(flat_data[:usable_size]) if i % self.num_processes == self.process_index]
        
        epoch_idx = 0
        example_idx = 0
        done = False

        while True:
            if done: break
            self.rng.shuffle(flat_data)
            batch = []

            for example, (i, j) in flat_data:
                batch_element = {}
                batch_element.update(self.tokenize_batch_element(example.prompt, example.generations[i], prefix='chosen'))
                batch_element.update(self.tokenize_batch_element(example.prompt, example.generations[j], prefix='rejected'))
                batch_element['prompt_id'] = example.prompt_id
                batch.append(batch_element)

                if len(batch) >= self.microbatch_size:
                    example_idx += len(batch) * self.num_processes
                    yield self.collate(batch)
                    batch = []

                    if self.n_examples is not None and example_idx >= self.n_examples:
                        print(f'Finished {example_idx} examples on {self.split} split')
                        done = True
                        break

            if self.num_processes == 1 and batch != []: # flush for eval, sampling
                yield self.collate(batch)
                batch = []

            epoch_idx += 1
            if self.n_epochs is not None and epoch_idx >= self.n_epochs:
                done = True
                break

    def get_num_training_steps(self):
        max_prompt_count = min(float("inf"), self.max_prompt_count) if self.max_prompt_count else float("inf")
        all_data = int(sum(min(max_prompt_count, len(example.pairs)) for _, example in self.full_data.items()))
        return all_data // self.num_processes
