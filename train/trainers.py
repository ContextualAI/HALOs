# Copyright (c) 2023 Contextual AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Extendable Trainer classes for aligning LLMs.
The specific class that should be used should be specified in the loss file under config/loss.

- The BasicTrainer contains the core methods (e.g., sharding, basic training loop, etc.).
- The SFTTrainer, PairedPreferenceTrainer, and UnpairedPreferenceTrainer all subclass BasicTrainer
  and override the get_batch_metrics() and (optionally) forward() methods.
- The PPOTrainer is a little different, since it also uses a reward model to judge examples before 
  updating the policy---note that there is no active sampling, as in standard online PPO.
- The BradleyTerryTrainer is used for training a Bradley-Terry reward model over paired preferences.

The trainer for each loss should subclass either PairedPreferenceTrainer or UnpairedPreferenceTrainer.
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import contextlib
import gc
from .models import AutoModelForCausalLM, AutoModelForCausalLMWithValueHead, AutoModelForBradleyTerry
from omegaconf import OmegaConf, DictConfig
from transformers import AutoTokenizer, GenerationConfig
from accelerate import Accelerator

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist


from . import dataloader
from .utils import (
    formatted_dict,
    pad_to_length,
    masked_mean,
    masked_var,
    entropy_from_logits,
    delete_dicts,
    rowwise_product,
    get_base_model_state_dict_from_peft
)
import numpy as np
import wandb
from tqdm import tqdm

import random
import os
from collections import defaultdict
import time
import json
from typing import Optional, Dict, List, Union, Tuple
from contextlib import nullcontext


class BasicTrainer(object):
    policy_hf_model_class = AutoModelForCausalLM
    reference_hf_model_class = AutoModelForCausalLM
    use_reference_model = True

    def __init__(self, 
                 tokenizer: AutoTokenizer, 
                 config: DictConfig, 
                 train_iterator: dataloader.DataLoader, 
                 eval_iterator: dataloader.DataLoader, 
                 accelerator: Accelerator,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler,
                 policy: nn.Module, 
                 reference_model: Optional[nn.Module] = None,
                 **kwargs):
        """A trainer for a language model, supporting either SFT, HALO, or offline PPO training."""
        self.seed = config.seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.accelerator = accelerator
        
        self.config = config
        self.run_dir = config.local_run_dir

        self.tokenizer = tokenizer
        self.example_counter = 0
        self.batch_counter = 0

        self.policy = policy
        self.policy_dtype = getattr(torch, config.model.policy_dtype)
        
        self.reference_model = reference_model
        self.train_iterator = train_iterator
        self.eval_iterator = eval_iterator
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.reward_model = kwargs.get('reward_model', None)
        self.reward_tokenizer = kwargs.get('reward_tokenizer', None)
        if self.reward_model is not None:
            assert self.reward_tokenizer is not None, "reward_tokenizer must be provided when using reward_model"
            self.reward_model.eval()

        self.prepare_accelerator()

    def prepare_accelerator(self):
        """Prepare the Accelerator."""
        self.policy, self.reference_model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.policy,
            self.reference_model,
            self.optimizer,
            self.scheduler,
        )

        if self.reward_model:
            self.reward_model = self.accelerator.prepare(self.reward_model)

    def get_batch_logps(self, logits: torch.FloatTensor, labels: torch.LongTensor):
        """Compute the token-level log probabilities of the given labels under the given logits."""
        # ignoring vocab size, batch size x length should be equal
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != -100)

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == -100] = 0

        distribution_logps = logits.float().log_softmax(-1)
        per_token_logps = torch.gather(distribution_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return per_token_logps * loss_mask
    
    def get_humanline_mask(self, policy_logps, reference_logps):
        """
        Return a boolean mask over tokens, where True means that the token has been rejected under humanline sampling.

        Args:
            policy_logps: token-level probabilities according to policy (microbatch_size, maximum sequence length)
            reference_logps: token-level probabilities according to reference model (microbatch_size, maximum sequence length)

        Returns:
            The rejection mask (microbatch_size, sequence length)
        """
        forward_rv = torch.distributions.beta.Beta(self.config.humanline_gamma_R, self.config.humanline_beta_R)
        forward_M = (reference_logps - policy_logps).exp().max().item()
        forward_sample = forward_rv.sample(policy_logps.shape).to(self.accelerator.device)
        forward_token_mask = (reference_logps - policy_logps).exp() < forward_M * forward_sample

        backward_rv = torch.distributions.beta.Beta(self.config.humanline_gamma_P, self.config.humanline_beta_P)
        backward_sample = backward_rv.sample(policy_logps.shape).to(self.accelerator.device)
        backward_M = (policy_logps - reference_logps).exp().max().item()
        backward_token_mask = (policy_logps - reference_logps).exp() < backward_M * backward_sample

        humanline_mask = forward_token_mask | backward_token_mask
        
        return humanline_mask
    
    def get_ratios(self, policy_logps, reference_logps, sequence_level=False):
        """
        Return the probability ratio under the policy vs the reference [policy(y|x)/reference(y|x)].
        Apply humanline sampling if specified.

        Args:
            policy_logps: token-level probabilities according to policy (microbatch_size, maximum sequence length)
            reference_logps: token-level probabilities according to reference model (microbatch_size, maximum sequence length)
            sequence_level: if true, return the probability for the entire sequence; otherwise, per-token

        Returns:
            The probability ratios (microbatch_size, sequence length) if sequence_level; otherwise, (microbatch_size, 1)
        """
        logratio = policy_logps - reference_logps

        if self.config.humanline:
            logratio = torch.where(self.get_humanline_mask(policy_logps, reference_logps), logratio.detach(), logratio)

        if sequence_level:
            logratio = logratio.sum(-1)

        ratio = logratio.exp()

        return ratio

    def get_sequence_rewards(self, policy_logps, reference_logps, length_normalized=False):
        """
        If regular alignment, return the HALO-defined reward for the sequence (log [policy(y|x)/reference(y|x)]).

        For humanline alignment, do zero-shot rejection sampling. Assume that the examples are drawn from the 
        reference model, zero-out the tokens that don't meet the rejection sampling criterion, then sum over what 
        remains to get the sequence-level rewards.
        
        Args:
            policy_logps: token-level probabilities according to policy (microbatch_size, maximum sequence length)
            reference_logps: token-level probabilities according to reference model (microbatch_size, maximum sequence length)
            length_normalized: divide the sequence reward by the number of non-rejected tokens

        Returns:
            The sequence-level rewards (microbatch_size, 1).
        """
        mask = self.get_humanline_mask(policy_logps, reference_logps) if self.config.humanline else torch.zeros_like(policy_logps).bool()
        token_rewards = torch.where(mask, (policy_logps - reference_logps).detach(), policy_logps - reference_logps)
        
        normalization_factor = (token_rewards.abs() != 0).float().sum(-1) if length_normalized else 1
        sequence_rewards = token_rewards.sum(-1) / normalization_factor

        return sequence_rewards

    def loss(self,
             batch: Dict,
             policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Args:
            batch: batch of data, mapping keys to Tensors
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (microbatch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (microbatch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (microbatch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (microbatch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the losses, one for each example (sif chosen_only or rejected_only, only n/2 losses).
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively, for reporting.
            Note that rejected responses do not factor into the loss, only the reward calculation.
        """
        raise NotImplementedError

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str=None) -> Tuple[torch.FloatTensor, Dict]:
        """Compute the loss and other metrics for the given batch of inputs.
        
        Arg:
            batch: dictionary of inputs for the batch (what is required will vary depending on the trainer)
            mode: one of 'train', 'eval', 'sample'

        Returns:
            A tuple of a scalar loss and a dict of metrics.
        """
        raise NotImplementedError

    def get_reward_scores(self, batch: Dict[str, torch.LongTensor]) -> torch.FloatTensor:
        """Get reward scores from reward model.
        
        Args:
            batch: Dictionary containing batch data
            
        Returns:
            torch.FloatTensor of shape (microbatch_size,) containing reward scores
        """
        if 'score' in batch:
            print(f"Using score from batch: {batch['score']}")
            reward_scores = batch['score']
        elif self.reward_model is not None:
            # Decode the sequences using policy tokenizer
            sequences = self.tokenizer.batch_decode(batch['target_combined_input_ids'], skip_special_tokens=True)
            # Encode with reward model tokenizer
            reward_inputs = self.reward_tokenizer(
                sequences,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=self.config.model.max_length
            ).to(self.accelerator.device)

            with torch.no_grad():
                # Get reward model scores
                outputs = self.reward_model(reward_inputs['input_ids'], attention_mask=reward_inputs['attention_mask'])
                # Use the positive class logit as the reward score
                if self.config.model.reward_model_path == "RLHFlow/ArmoRM-Llama3-8B-v0.1":
                    reward_scores = outputs.score.cpu().float()
                else:
                    reward_scores = outputs.logits[:, 1]
        else:
            # Use binary labels (1 for chosen, -1 for rejected)
            reward_scores = torch.tensor([(1 if batch['status'][i] == 'chosen' else -1) for i in range(len(batch['status']))])

        return reward_scores

    def eval(self) -> Dict[str, Dict]:
        """
        Run evaluation on all the examples in the test data and return the metrics from get_batch_metrics.
        This is close-ended evaluation and measures the performance of a single model on a single dataset. 
        It does not compare two models to each other.

        Returns:
            A dict of form:
            {
                'metadata': the Hydra config
                'results': a dict of batch metrics (averaged across all of the test data)
            }
        """
        self.accelerator.print(f'Running evaluation after {self.example_counter} train examples')
        self.policy.eval()

        if self.reference_model is not None:
            self.reference_model.eval()

        all_eval_metrics = defaultdict(list)
    
        for eval_batch in (tqdm(self.eval_iterator, desc='Computing eval metrics') if self.accelerator.is_main_process else self.eval_iterator):
            eval_batch = {k: v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in eval_batch.items()}
            with torch.no_grad():
                _, eval_metrics = self.get_batch_metrics(eval_batch, mode='eval')

            for k, v in eval_metrics.items():
                all_eval_metrics[k].extend(torch.as_tensor(v).reshape(-1).float().cpu().numpy().tolist())

        # Compute mean metrics
        mean_eval_metrics = {}
        for k, v in all_eval_metrics.items():
            if len(v) > 0:
                mean_eval_metrics[k] = sum(v) / len(v)

        delete_dicts(eval_batch, eval_metrics, all_eval_metrics)
        self.free_memory()

        if self.accelerator.is_main_process and self.config.wandb.enabled:
            wandb.log(mean_eval_metrics, step=self.example_counter)
        else:
            results = None

        results = {
            'metadata': OmegaConf.to_container(self.config),
            'results': formatted_dict(mean_eval_metrics),
        }
        
        return results

    def train(self):
        """Begin either SFT or HALO training, with periodic evaluation. This is subclassed when implementing PPO."""
        self.accelerator.print(f'Using {self.config.optimizer} optimizer with learning rate {self.config.lr}')

        if self.reference_model is not None:
            self.reference_model.eval()

        last_log = None
        batch_metrics = defaultdict(list)
        accumulated_batches = []

        for train_batch in self.train_iterator:
            # EVALUATION
            if accumulated_batches == [] and ((self.example_counter % self.config.eval_every == 0) or (self.example_counter == 0 and self.config.do_first_eval)):
                results = self.eval()

                if self.example_counter > 0:
                    if self.config.debug:
                        self.accelerator.print('skipping save in debug mode')
                    elif self.config.intermediate_checkpoints:
                        output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                        self.accelerator.print(f'creating checkpoint to write to {output_dir}...')
                        self.save(output_dir, results['results'], final_save=False)

                self.accelerator.print(results['results'])
                delete_dicts(results) 
            
            # TRAINING
            accumulated_batches.append(train_batch)
            if len(accumulated_batches) < self.config.model.gradient_accumulation_steps:
                continue

            self.policy.train()
            start_time = time.time()
            
            for i in range(self.config.humanline_iters if self.config.humanline else 1):
                self.optimizer.zero_grad()

                for batch_idx, batch in enumerate(accumulated_batches):  
                    # only synchronize gradients on the last batch to avoid slowdown from unnecessary communication
                    with contextlib.nullcontext() if batch_idx + 1 == len(accumulated_batches) else self.accelerator.no_sync(self.policy):
                        batch = {k: v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        loss, metrics = self.get_batch_metrics(batch)
                        loss = loss / self.config.model.gradient_accumulation_steps
                        self.accelerator.backward(loss)
                        delete_dicts(batch)
                    
                        for k, v in metrics.items():
                            batch_metrics[k].extend(torch.as_tensor(v).reshape(-1).float().cpu().numpy().tolist())

                grad_norm = self.accelerator.clip_grad_norm_(self.policy.parameters(), self.config.model.max_grad_norm)
                batch_metrics['grad_norm'].extend(torch.as_tensor(grad_norm).reshape(-1).float().cpu().numpy().tolist())

                self.optimizer.step()
                if self.config.sync_reference or self.config.humanline:
                    self.sync_reference_with_policy()

            self.scheduler.step()
            step_time = time.time() - start_time
            examples_per_second = self.config.model.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            
            self.batch_counter += 1
            self.example_counter += self.config.model.batch_size

            if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                mean_train_metrics = {}
                for k, v in batch_metrics.items():
                    if len(v) > 0:
                        mean_train_metrics[k] = sum(v) / len(v)

                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                mean_train_metrics['counters/lr'] = self.scheduler.get_last_lr()[0]
                self.accelerator.print(f'train stats after {self.batch_counter} steps: {formatted_dict(mean_train_metrics)}')

                if self.config.wandb.enabled and self.accelerator.is_main_process:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
                batch_metrics = defaultdict(list)
            else:
                self.accelerator.print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')

            delete_dicts(metrics, batch_metrics, mean_train_metrics)
            accumulated_batches = []
            self.free_memory()

    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = {}, final_save=True):
        """Save tokenizer, policy model, optimizer, scheduler state to disk."""
        self.accelerator.print(f"Saving...")
        if output_dir is None:
            output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')

        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            self.accelerator.print(f"Saving tokenizer...")
            self.tokenizer.save_pretrained(output_dir)

            with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
                metrics['counter'] = self.example_counter
                json.dump(metrics, f)
        
        self.accelerator.wait_for_everyone()
        self.accelerator.print(f"Saving state...")
        optimizer = self.accelerator.unwrap_model(self.optimizer)
        scheduler = self.accelerator.unwrap_model(self.scheduler)
        if self.accelerator.is_main_process:
            optimizer_state = {
                'state_dict': optimizer.state_dict(),
                'class': optimizer.__class__.__name__,
            }
            torch.save(optimizer_state, os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        
        self.accelerator.wait_for_everyone()
        self.accelerator.print(f"Saving model...")

        if self.config.model.use_peft and final_save:
            state_dict = get_base_model_state_dict_from_peft(
                self.accelerator.get_state_dict(self.policy),
                self.config.model.peft.lora_alpha,
                self.config.model.peft.lora_r,
            )
            unwrapped_model = self.accelerator.unwrap_model(self.policy).base_model
        else:
            # by default, get_state_dict unwraps model before getting the state_dict
            state_dict = self.accelerator.get_state_dict(self.policy)
            unwrapped_model = self.accelerator.unwrap_model(self.policy)

        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=state_dict,
            safe_serialization=False
        )
            
        self.accelerator.wait_for_everyone()

    def sync_reference_with_policy(self):
        """
        Update the reference model to have the policy weights.
        """
        state_dict = self.accelerator.unwrap_model(self.policy).state_dict()
        self.accelerator.unwrap_model(self.reference_model).load_state_dict(state_dict)
        self.accelerator.wait_for_everyone()

    def sample(self, model, batch: Dict[str, Union[List, torch.LongTensor]], temp: float=0.7) -> str:
        """
        Sample from the given model. NOTE: If the policy is being trained with FSDP, then sampling from it 
        directly will produce gibberish. If you want to sample from the policy, you should sync the reference 
        model with the policy via sync_reference_with_policy, then sample from reference model. If you are 
        inserting this back into the dataloader, remember that the generation should be in the format 
        { "role": "assistant", "content": x }.
        
        Args:
            model: the model to sample from (the reference model in most cases)
            batch: the sample batch returned from a data loader
            temp: temperature for sampling
        Returns:
            Completion for the prompt.
        """
        generation_config = GenerationConfig(
            max_new_tokens=(self.config.model.max_length - self.config.model.max_prompt_length),
            do_sample=True,
            temperature=temp,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        batch_elements = []

        for prompt in batch['prompt_text']:
            batch_element = self.train_iterator.tokenize_batch_element(
                [{"role": "user", "content" : prompt}], 
                [{"role": "assistant", "content": ""}], 
            )
            batch_elements.append(batch_element)

        model.eval()

        if self.accelerator.state.fsdp_plugin is not None:
            context = FSDP.summon_full_params(model)
        else:
            context = nullcontext

        with context:
            with torch.no_grad():
                batch_elements = self.train_iterator.collate(batch_elements)
                batch_completion_ids = model.generate(
                    batch_elements['prompt_input_ids'].to(self.accelerator.device), 
                    attention_mask=batch_elements['prompt_attention_mask'].to(self.accelerator.device),
                    generation_config=generation_config
                )
                # truncate prompt
                batch_completion_ids = [ x[len(batch_elements['prompt_input_ids'][i]):] for i,x in enumerate(batch_completion_ids) ]
                batch_completions = self.tokenizer.batch_decode(batch_completion_ids, skip_special_tokens=True)
        
        return batch_completions

    def update_batch(self, batch: Dict[str, Union[List, torch.LongTensor]], completions: List[str]) -> Dict[str, Union[List, torch.LongTensor]]:
        
        """
        Args:
            batch: the original offline batch
            completions: the online generations from the updated reference model (after syncing weights with policy model)
            
        Returns:
            updated_batch: the updated online batch
        """
        updated_batch = []
        for i, prompt in enumerate(batch['prompt']):
            batch_element = self.train_iterator.tokenize_batch_element(
                prompt, 
                [{"role": "assistant", "content": completions[i]}], 
                prefix='target'
            )
            for k in ['status', 'conversation', 'generation']:
                batch_element[k] = batch[k][i]
            updated_batch.append(batch_element)
        
        for i in range(len(updated_batch)):
            updated_batch[i].update(self.train_iterator.tokenize_batch_element(
                updated_batch[i]['prompt'], 
                [{"role": "assistant", "content": completions[i]}], 
                prefix='KL'
            ))
        
        updated_batch = self.train_iterator.collate(updated_batch)
        delete_dicts(batch)
        
        return updated_batch

    def free_memory(self):
        torch.cuda.empty_cache()
        self.accelerator.free_memory()
        gc.collect()


class SFTTrainer(BasicTrainer):
    use_reference_model = False

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str='train'):
        """Compute the loss and other metrics for the given batch of inputs.
        
        Args:
            batch: dictionary of inputs for the batch (should contain 'target_attention_mask', 'target_input_input_ids', 
                'target_labels' where 'target' corresponds to the SFT example)
            mode: one of 'train', 'eval', 'sample'
        """
        metrics = {}
        
        with self.accelerator.autocast():
            policy_chosen_logits = self.policy(
                batch['target_combined_input_ids'], 
                attention_mask=batch['target_combined_attention_mask'],
            ).logits.to(self.policy_dtype)
            
            policy_chosen_logps = self.get_batch_logps(policy_chosen_logits, batch['target_labels'])
            token_mask = (policy_chosen_logps != 0).float().detach()

        # Normalize the loss by the number of tokens before returning for backpropagation
        normalized_loss = -policy_chosen_logps / token_mask.sum()

        # Gather losses and logps from all processes
        metrics[f'logps/{mode}'] = self.accelerator.gather((policy_chosen_logps * token_mask).detach().sum() / token_mask.sum())
        metrics[f'loss/{mode}'] = self.accelerator.gather(normalized_loss.detach())

        return normalized_loss.sum(), metrics


class HumanlineSFTTrainer(BasicTrainer):
    use_reference_model = True

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str='train'):
        """Compute the loss and other metrics for the given batch of inputs.
        
        Args:
            batch: dictionary of inputs for the batch (should contain 'target_attention_mask', 'target_input_input_ids', 
                'target_labels' where 'target' corresponds to the SFT example)
            mode: one of 'train', 'eval', 'sample'
        """
        metrics = {}
        
        with self.accelerator.autocast():
            policy_chosen_logits = self.policy(
                batch['target_combined_input_ids'], 
                attention_mask=batch['target_combined_attention_mask'],
            ).logits.to(self.policy_dtype)
            
            policy_chosen_logps = self.get_batch_logps(policy_chosen_logits, batch['target_labels'])
            policy_chosen_logps = policy_chosen_logps.view(-1)
            token_mask = (policy_chosen_logps != 0).float().detach()

            with torch.no_grad():
                reference_chosen_logits = self.reference_model(
                    batch['target_combined_input_ids'], 
                    attention_mask=batch['target_combined_attention_mask'],
                ).logits.to(self.policy_dtype)
                
                reference_chosen_logps = self.get_batch_logps(reference_chosen_logits, batch['target_labels'])
                reference_chosen_logps = reference_chosen_logps.view(-1)

            humanline_mask = self.get_humanline_mask(policy_chosen_logps, reference_chosen_logps)
            normalized_loss = torch.where(humanline_mask, -policy_chosen_logps.detach(), -policy_chosen_logps) / token_mask.sum()

        # Gather losses and logps from all processes
        metrics[f'unmasked/{mode}'] = ((1 - humanline_mask.float()) * token_mask).sum() / token_mask.sum()
        metrics[f'tokens/{mode}'] = self.accelerator.gather(token_mask.sum()).sum()
        metrics[f'logps/{mode}'] = self.accelerator.gather((policy_chosen_logps * token_mask).detach().sum() / token_mask.sum())
        metrics[f'loss/{mode}'] = self.accelerator.gather(normalized_loss.detach())

        del policy_chosen_logits, policy_chosen_logps, reference_chosen_logits, reference_chosen_logps, humanline_mask, token_mask

        return normalized_loss.sum(), metrics


class UnpairedPreferenceTrainer(BasicTrainer):
    use_reference_model = True

    """A trainer for any loss that doesn't use paired preference, like KTO."""
    def forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], use_cache: bool=False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.BoolTensor]:
        """Run the given model on the given batch of inputs.
        
        Returns:
            chosen_logps: log probabilities of chosen examples 
            rejected_logps: log probabilities of rejected examples
            use_cache: if true, expecte to get cached logprobs from the model
        """
        with self.accelerator.autocast():
            if use_cache:
                all_logps = model(batch['target_combined_input_ids']).to(self.policy_dtype).to(self.accelerator.device)
            else:
                all_logits = model(
                    batch['target_combined_input_ids'], 
                    attention_mask=batch['target_combined_attention_mask'],
                ).logits.to(self.policy_dtype)
            
                all_logps = self.get_batch_logps(all_logits, batch['target_labels'])

        assert all_logps.shape[0] == len(batch['status'])
        chosen_idx = [i for i in range(all_logps.shape[0]) if batch['status'][i] == 'chosen']
        rejected_idx = [i for i in range(all_logps.shape[0]) if batch['status'][i] == 'rejected']

        chosen_logps = all_logps[chosen_idx, ...]
        rejected_logps = all_logps[rejected_idx, ...]
        return chosen_logps, rejected_logps

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str='train'):
        """Compute the loss and other metrics for the given batch of inputs."""
        metrics = {}

        if self.reference_model is None:
            policy_chosen_logps, policy_rejected_logps = self.forward(self.policy, batch)
            losses, chosen_rewards, rejected_rewards = self.loss(batch, policy_chosen_logps, policy_rejected_logps)
        else:
            policy_chosen_logps, policy_rejected_logps = self.forward(self.policy, batch)
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.forward(self.reference_model, batch, use_cache=self.config.cache_reference_logprobs)
            losses, chosen_rewards, rejected_rewards = self.loss(batch, policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps)

        # all_gather treats empty lists/tensors poorly, and empty lists can occur because a batch can contain all chosen or all rejected example
        # therefore, concatenate chosen + rejected rewards before all_gather
        combined_rewards = torch.cat((chosen_rewards.detach(), rejected_rewards.detach()), 0)
        combined_statuses = torch.Tensor([1] * len(chosen_rewards) + [0] * len(rejected_rewards)).to(self.accelerator.device)

        all_rewards = self.accelerator.gather(combined_rewards.detach())
        all_statuses = self.accelerator.gather(combined_statuses.detach())
        chosen_rewards_idx = [ i for i in range(len(all_statuses)) if all_statuses[i].item() == 1 ]
        rejected_rewards_idx = [ i for i in range(len(all_statuses)) if all_statuses[i].item() == 0 ]

        metrics[f'rewards_{mode}/chosen'] = all_rewards[chosen_rewards_idx]
        metrics[f'rewards_{mode}/rejected'] = all_rewards[rejected_rewards_idx]
        metrics[f'rewards_{mode}/margins'] = torch.Tensor([(all_rewards[chosen_rewards_idx].mean().nan_to_num(0) - all_rewards[rejected_rewards_idx].mean().nan_to_num(0)).item()])
        metrics[f'loss/{mode}'] = self.accelerator.gather(losses.mean().detach()).mean()

        del policy_chosen_logps, policy_rejected_logps
        del combined_rewards, combined_statuses, all_rewards, all_statuses, chosen_rewards_idx, rejected_rewards_idx, all_devices_losses

        if self.reference_model:
            del reference_chosen_logps, reference_rejected_logps

        return losses.mean(), metrics
        return losses.mean(), metrics


class PairedPreferenceTrainer(BasicTrainer):
    use_reference_model = True

    """A trainer for any loss that uses paired preference, like DPO."""
    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor. The first half is chosen outputs, the second half is rejected.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (microbatch_size, sequence_length).
            
        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        max_length = max(batch['chosen_combined_input_ids'].shape[1], batch['rejected_combined_input_ids'].shape[1])
        concatenated_batch = {}

        for k in batch:
            if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
                pad_value = -100 if 'labels' in k else 0
                concatenated_key = k.replace('chosen', 'concatenated')
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)

        for k in batch:
            if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
                pad_value = -100 if 'labels' in k else 0
                concatenated_key = k.replace('rejected', 'concatenated')
                concatenated_batch[concatenated_key] = torch.cat((
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ), dim=0)

        return concatenated_batch

    def forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], use_cache: bool=False) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
           Return two tensors of shape (batch size), one of the chosen examples, another of the rejected ones.

           Returns:
            chosen_logps: log probabilities of chosen examples 
            rejected_logps: log probabilities of rejected examples 
            use_cache: if true, expecte to get cached logprobs from the model
        """
        with self.accelerator.autocast():
            concatenated_batch = self.concatenated_inputs(batch)

            if use_cache:
                all_logps = model(batch['concatenated_combined_input_ids']).to(self.policy_dtype).to(self.accelerator.device)
            else:
                all_logits = model(
                    concatenated_batch['concatenated_combined_input_ids'], 
                    attention_mask=concatenated_batch['concatenated_combined_attention_mask'],
                ).logits.to(self.policy_dtype)
                
                all_logps = self.get_batch_logps(all_logits, concatenated_batch['concatenated_labels'])
        
        chosen_logps = all_logps[:batch['chosen_combined_input_ids'].shape[0], ...]
        rejected_logps = all_logps[batch['chosen_combined_input_ids'].shape[0]:, ...]
        return chosen_logps, rejected_logps

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str='train'):
        """Compute the loss and other metrics for the given batch of inputs."""
        metrics = {}

        if self.reference_model is None:
            policy_chosen_logps, policy_rejected_logps = self.forward(self.policy, batch)
            losses, chosen_rewards, rejected_rewards = self.loss(batch, policy_chosen_logps, policy_rejected_logps)
        else:
            policy_chosen_logps, policy_rejected_logps = self.forward(self.policy, batch)
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.forward(self.reference_model, batch, use_cache=self.config.cache_reference_logprobs)
            losses, chosen_rewards, rejected_rewards = self.loss(batch, policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps)

        # accuracy calculated on paired examples (for apples-to-apples comparison with UnpairedPreferenceTrainer)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        metrics[f'rewards_{mode}/chosen'] = self.accelerator.gather(chosen_rewards.detach())
        metrics[f'rewards_{mode}/rejected'] = self.accelerator.gather(rejected_rewards.detach())
        metrics[f'rewards_{mode}/accuracies'] = self.accelerator.gather(reward_accuracies.detach())
        metrics[f'rewards_{mode}/margins'] = self.accelerator.gather((chosen_rewards - rejected_rewards).detach())
        metrics[f'logps_{mode}/rejected'] = self.accelerator.gather(policy_rejected_logps.detach())
        metrics[f'logps_{mode}/chosen'] = self.accelerator.gather(policy_chosen_logps.detach())
        metrics[f'loss/{mode}'] = self.accelerator.gather(losses.mean().detach()).mean()

        del chosen_rewards, rejected_rewards, reward_accuracies, policy_chosen_logps, policy_rejected_logps
        if self.reference_model:
            del reference_chosen_logps, reference_rejected_logps

        return losses.mean(), metrics
        return losses.mean(), metrics


class DPOTrainer(PairedPreferenceTrainer):
    def loss(self,
        batch: Dict,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        *args,
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model token-level log probabilities."""
        chosen_rewards = self.config.loss.beta * self.get_sequence_rewards(policy_chosen_logps, reference_chosen_logps)
        rejected_rewards = self.config.loss.beta * self.get_sequence_rewards(policy_rejected_logps, reference_rejected_logps)

        losses = -F.logsigmoid(chosen_rewards - rejected_rewards)

        return losses, chosen_rewards.detach(), rejected_rewards.detach()


class CDPOTrainer(PairedPreferenceTrainer):
    def loss(self,
        batch: Dict,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        *args,
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the CDPO loss for a batch of policy and reference model token-level log probabilities."""
        chosen_rewards = self.config.loss.beta * self.get_sequence_rewards(policy_chosen_logps, reference_chosen_logps)
        rejected_rewards = self.config.loss.beta * self.get_sequence_rewards(policy_rejected_logps, reference_rejected_logps)
        
        forward_losses = -F.logsigmoid(chosen_rewards - rejected_rewards)
        reverse_losses = -F.logsigmoid(rejected_rewards - chosen_rewards)
        losses = (1 - self.config.loss.epsilon) * forward_losses + self.config.loss.epsilon * reverse_losses

        return losses, chosen_rewards.detach(), rejected_rewards.detach()


class IPOTrainer(PairedPreferenceTrainer):
    def loss(self,
        batch: Dict,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        *args,
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the IPO loss for a batch of policy and reference model token-level log probabilities."""
        chosen_rewards = self.get_sequence_rewards(policy_chosen_logps, reference_chosen_logps, length_normalized=True)
        rejected_rewards = self.get_sequence_rewards(policy_rejected_logps, reference_rejected_logps, length_normalized=True)
        
        losses = (chosen_rewards - rejected_rewards - (1/(2 * self.config.loss.tau))).pow(2)

        return losses, chosen_rewards.detach(), rejected_rewards.detach()
    

class SimPOTrainer(PairedPreferenceTrainer):
    def loss(self,
        batch: Dict,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        *args,
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the SimPO loss for a batch of policy and reference model token-level log probabilities."""
        # implicit reference model that assigns probability 1 (logp = 0) to all tokens
        chosen_rewards = self.get_sequence_rewards(policy_chosen_logps, torch.zeros_like(policy_chosen_logps).to(self.accelerator.device), length_normalized=True)
        rejected_rewards = self.get_sequence_rewards(policy_rejected_logps, torch.zeros_like(policy_rejected_logps).to(self.accelerator.device), length_normalized=True)
        losses = -F.logsigmoid(self.config.loss.beta * (chosen_rewards - rejected_rewards - self.config.loss.gamma_beta_ratio))

        return losses, chosen_rewards.detach(), rejected_rewards.detach()


class SLiCTrainer(PairedPreferenceTrainer):
    use_reference_model = False

    def loss(self,
        batch: Dict,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        *args,
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the SLIC loss as defined by Zhao et al. in https://arxiv.org/pdf/2305.10425.pdf

        Calibration loss defined as:
            L(x, y) := max(0, beta - log p_policy(y_chosen|x) + log p_rejected(y|x))
        For the cross-entropy loss, just use the NLL of the chosen sequence (equivalent to SFT).
        """
        # implicit reference model that assigns probability 1 (logp = 0) to all tokens, as in SimPO
        chosen_rewards = self.get_sequence_rewards(policy_chosen_logps, torch.zeros_like(policy_chosen_logps).to(self.accelerator.device))
        rejected_rewards = self.get_sequence_rewards(policy_rejected_logps, torch.zeros_like(policy_rejected_logps).to(self.accelerator.device))
        
        cal_loss = torch.clamp(self.config.loss.beta - chosen_rewards + rejected_rewards, min=0)
        reg_loss = -policy_chosen_logps

        losses = cal_loss + self.config.loss.lambda_coef * reg_loss

        return losses, chosen_rewards.detach(), rejected_rewards.detach()


class KTOTrainer(UnpairedPreferenceTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss(self,
        batch: Dict,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
        *args,
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the Kahneman-Tversky loss for a batch of policy and reference model log probabilities.

        If generation y ~ p_desirable, we have the 'desirable' loss:
            L(x, y) := 1 - sigmoid(beta * ([log p_policy(y|x) - log p_reference(y|x)] - KL(p_policy || p_reference)))
        If generation y ~ p_undesirable, we have the 'undesirable' loss:
            L(x, y) := 1 - sigmoid(beta * (KL(p_policy || p_reference) - [log p_policy(y|x) - log p_reference(y|x)]))

        The desirable losses are weighed by config.loss.desirable_weight.
        The undesirable losses are weighed by config.loss.undesirable_weight.
        This should be used to address imbalances in the ratio of desirable:undesirable examples respectively.
        The KL term is estimated by matching x with unrelated outputs y', then calculating the average log ratio
        log p_policy(y'|x) - log p_reference(y'|x).

        If humanline alignment, do the following:
        - for rejected examples, use a KL estimate of zero
        - zero-shot rejection sampling (handled by get_sequence_rewards)
        - adjust for entire chosen(rejected) sequences that have been sampled out by upweighting other chosen(rejected) examples
        """
        if policy_chosen_logps.shape[0] != 0:
            chosen_rewards = self.get_sequence_rewards(policy_chosen_logps, reference_chosen_logps)
        else:
            chosen_rewards = torch.Tensor([]).to(self.policy_dtype).to(self.accelerator.device)

        if policy_rejected_logps.shape[0] != 0:
            rejected_rewards = self.get_sequence_rewards(policy_rejected_logps, reference_rejected_logps)
        else:
            rejected_rewards = torch.Tensor([]).to(self.policy_dtype).to(self.accelerator.device)

        KL_rewards = self.get_sequence_rewards(policy_KL_logps.detach(), reference_KL_logps.detach())

        stats = self.accelerator.reduce(torch.Tensor([
            (chosen_rewards.abs() != 0).float().sum().item(),   # non-empty sequences after rejection sampling
            len(chosen_rewards),                                
            (rejected_rewards.abs() != 0).float().sum().item(), # non-empty sequences after rejection sampling
            len(rejected_rewards),
            KL_rewards.sum(),                                   # sum of non-empty KL examples
            (KL_rewards.abs() != 0).float().sum().item(),       # number of non-empty KL examples
        ]).to(self.accelerator.device), reduction="sum")

        KL = (stats[4] / stats[5].clamp(min=1)).clamp(min=0)
        
        if policy_chosen_logps.shape[0] != 0:
            chosen_losses = self.config.loss.desirable_weight * (1 - F.sigmoid(self.config.loss.beta * (chosen_rewards - KL)))
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            chosen_losses = torch.Tensor([]).to(self.policy_dtype).to(self.accelerator.device)
        
        if policy_rejected_logps.shape[0] != 0:
            rejected_losses = self.config.loss.undesirable_weight * (1 - F.sigmoid(self.config.loss.beta * (KL - rejected_rewards)))
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            rejected_losses = torch.Tensor([]).to(self.policy_dtype).to(self.accelerator.device)

        losses = torch.cat((chosen_losses, rejected_losses), 0)

        return losses, chosen_rewards.detach(), rejected_rewards.detach(), KL.detach()
    
    def forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], use_cache: bool=False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs.
        
        Args:
            - model: the model to use for the forward pass
            - batch: the microbatch (should have the input ids, attention mask, and labels)
            - use_cache: if true, can get cached logprobs instead

        Returns:
            chosen_logps: log probabilities of chosen examples
            rejected_logps: log probabilities of rejected examples
            KL_logps: log probabilities of the unmatched y'|x (used to estimate the KL divergence between policy and reference)
        """
        with self.accelerator.autocast():
            with torch.no_grad():
                if use_cache:
                    KL_logps = model(batch[f'KL_combined_input_ids']).to(self.policy_dtype).to(self.accelerator.device)
                else:
                    KL_logits = model(
                        batch[f'KL_combined_input_ids'],
                        attention_mask=batch[f'KL_combined_attention_mask']
                    ).logits.to(self.policy_dtype)

                    KL_logps = self.get_batch_logps(KL_logits, batch[f'KL_labels'])

            if use_cache:
                target_logps = model(batch[f'target_combined_input_ids']).to(self.policy_dtype).to(self.accelerator.device)
            else:
                target_logits = model(
                    batch[f'target_combined_input_ids'],
                    attention_mask=batch[f'target_combined_attention_mask']
                ).logits.to(self.policy_dtype)

                target_logps = self.get_batch_logps(target_logits, batch[f'target_labels'])

        assert target_logps.shape[0] == len(batch['status'])
        chosen_idx = [i for i in range(target_logps.shape[0]) if batch['status'][i] == 'chosen']
        rejected_idx = [i for i in range(target_logps.shape[0]) if batch['status'][i] == 'rejected']
        chosen_logps = target_logps[chosen_idx, ...]
        rejected_logps = target_logps[rejected_idx, ...]

        return chosen_logps, rejected_logps, KL_logps
    
    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str='train'):
        """Compute the loss and other metrics for the given batch of inputs."""
        metrics = {}
        
        policy_chosen_logps, policy_rejected_logps, policy_KL_logps = self.forward(self.policy, batch)
        with torch.no_grad():    
            reference_chosen_logps, reference_rejected_logps, reference_KL_logps = self.forward(self.reference_model, batch, use_cache=self.config.cache_reference_logprobs)
        
        losses, chosen_rewards, rejected_rewards, KL = self.loss(
            batch,
            policy_chosen_logps,
            policy_rejected_logps,
            policy_KL_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            reference_KL_logps,
        )

        combined_rewards = torch.cat((chosen_rewards.detach(), rejected_rewards.detach()), 0)
        combined_statuses = torch.Tensor([1] * len(chosen_rewards) + [0] * len(rejected_rewards)).to(self.accelerator.device)

        all_rewards = self.accelerator.gather(combined_rewards)
        all_statuses = self.accelerator.gather(combined_statuses)
        all_KL = self.accelerator.gather(KL)
        chosen_rewards_idx = [ i for i in range(len(all_statuses)) if all_statuses[i].item() == 1 ]
        rejected_rewards_idx = [ i for i in range(len(all_statuses)) if all_statuses[i].item() == 0 ]

        metrics[f'rewards_{mode}/chosen'] = all_rewards[chosen_rewards_idx]
        metrics[f'rewards_{mode}/rejected'] = all_rewards[rejected_rewards_idx]
        metrics[f'rewards_{mode}/margins'] = torch.Tensor([(all_rewards[chosen_rewards_idx].mean().nan_to_num(0) - all_rewards[rejected_rewards_idx].mean().nan_to_num(0)).item()])
        metrics[f'rewards_{mode}/KL_estimate'] = all_KL
        metrics[f'loss/{mode}'] = self.accelerator.gather(losses.mean().detach()).mean()

        del policy_chosen_logps, policy_rejected_logps, policy_KL_logps, reference_chosen_logps, reference_rejected_logps, reference_KL_logps
        del combined_rewards, combined_statuses, all_rewards, all_statuses, chosen_rewards_idx, rejected_rewards_idx, all_KL

        return losses.mean(), metrics
        return losses.mean(), metrics


class GRPOTrainer(BasicTrainer):
    def loss(self, batch: Dict, policy_logps: torch.FloatTensor, reference_logps: torch.FloatTensor, advantages: torch.FloatTensor, group_size: torch.FloatTensor):
        """
        Compute the GRPO loss.

        Args:
            policy_logps: log probability of the output under the policy (microbatch_size, sequence_length) 
            reference_logps: log probability of the output under the reference model (microbatch_size, sequence_length) 
            advantages: sequence level advantages (microbatch_size,)
            group_size: number of outputs (in entire batch) belonging to prompt associated with sequence (microbatch_size,)

        Returns:
            sequence-level losses (microbatch_size,), sequence-level KL (microbatch_size,), weighted advantages (microbatch_size,)
        """
        ratio = self.get_ratios(policy_logps, reference_logps, sequence_level=self.config.loss.sequence_level)
        KL = (-ratio.log()).exp() + ratio.log() - 1

        if not self.config.loss.sequence_level:
            advantages = advantages.unsqueeze(-1)
            group_size = group_size.unsqueeze(-1)

        weighted_adv = advantages * ratio
        weighted_adv_clipped =  advantages * ratio.clamp(1 - self.config.loss.epsilon, 1 + self.config.loss.epsilon)
        losses = -1 * (torch.min(weighted_adv, weighted_adv_clipped) - self.config.loss.beta * KL) / group_size

        return losses, KL.detach(), weighted_adv.detach()

    def forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], use_cache: bool=False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs.
        
        Args:
            - model: the model to use for the forward pass
            - batch: the microbatch (should have the input ids, attention mask, and labels)
            - use_cache: if true, can get cached logprobs instead

        Returns:
            logps: log probabilities of examples
        """    
        with self.accelerator.autocast():
            if use_cache:
                all_logps = model(batch['target_combined_input_ids']).to(self.policy_dtype).to(self.accelerator.device)
            else:
                all_logits = model(
                    batch['target_combined_input_ids'], 
                    attention_mask=batch['target_combined_attention_mask'],
                ).logits.to(self.policy_dtype)
            
                all_logps = self.get_batch_logps(all_logits, batch['target_labels'])

        return all_logps
    
    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str='train'):
        """Compute the loss and other metrics for the given batch of inputs."""
        metrics = {}
        
        policy_logps = self.forward(self.policy, batch)
        with torch.no_grad():    
            reference_logps = self.forward(self.reference_model, batch, use_cache=self.config.cache_reference_logprobs)
        
        prompt_ids = self.accelerator.gather_for_metrics(batch['prompt_id'])
        scores = self.accelerator.gather_for_metrics(batch['score'])
        scores_by_prompt_id = defaultdict(list)

        for i in range(len(prompt_ids)):
            scores_by_prompt_id[prompt_ids[i]].append(scores[i])

        group_size = []
        advantages = []

        for i, prompt_id in enumerate(batch['prompt_id']):
            group_size.append(len(scores_by_prompt_id[prompt_id]))
            advantages.append((batch['score'][i] - np.mean(scores_by_prompt_id[prompt_id])) / np.std(scores_by_prompt_id[prompt_id]))

        advantages = torch.Tensor(advantages).to(self.accelerator.device)
        group_size = torch.Tensor(group_size).to(self.accelerator.device)

        losses, KL, weighted_advantage = self.loss(
            batch,
            policy_logps,
            reference_logps,
            advantages,
            group_size
        )

        metrics[f'rewards_{mode}/groupsize'] = group_size
        metrics[f'rewards_{mode}/rewards'] = scores
        metrics[f'rewards_{mode}/weighted_advantage'] = self.accelerator.gather(weighted_advantage)
        metrics[f'rewards_{mode}/KL'] = self.accelerator.gather(KL)
        metrics[f'loss/{mode}'] = self.accelerator.gather(losses.detach()).mean()
        
        del policy_logps, reference_logps, scores, prompt_ids, advantages, group_size, KL, weighted_advantage
        
        return losses.mean(), metrics
        return losses.mean(), metrics


class PPOTrainer(BasicTrainer):
    policy_hf_model_class = AutoModelForCausalLMWithValueHead
    use_reference_model = True
            
    def prepare_accelerator(self):
        """Prepare the Accelerator."""
        self.policy.pretrained_model, self.policy.v_head, self.reference_model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.policy.pretrained_model,
            self.policy.v_head,
            self.reference_model,
            self.optimizer, 
            self.scheduler
        )

        if self.reward_model:
            self.reward_model = self.accelerator.prepare(self.reward_model)

    def forward(self, model: AutoModelForCausalLMWithValueHead, batch: Dict[str, Union[List, torch.LongTensor]], is_policy: bool=True, use_cache: bool=False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs.

        Args:
            model: model to run forward pass on
            batch: input batch (forward pass will be run on keys with prefix 'chosen')
            masks: binary-valued tensor shape (batch size, sequence length)
            is_policy: whether the model is the policy or reference
            use_cache: if true, expecte to get cached logprobs from the model

        Returns: 
            all_logps: batch log probabilities at the token level of shape (batch size, sequence length)
            all_logits: corresponding logits of shape (batch size, sequence length)
            all_values: values predicted for each token, of shape (batch size, sequence length)
        """
        if is_policy:
            # here the prefix 'chosen' is a misnomer, since it can refer to the dispreferred generations
            # the 'status' field contains the actual status of the generations
            all_logits, _, all_values = model(batch['target_combined_input_ids'], attention_mask=batch['target_combined_attention_mask'])
            all_values = all_values[:, :-1].contiguous()
        else: # if reference
            if use_cache:
                all_logps = model(batch['target_combined_input_ids']).to(self.policy_dtype).to(self.accelerator.device)
            else:
                all_logits = model(batch['target_combined_input_ids'], attention_mask=batch['target_combined_attention_mask']).logits.to(self.policy_dtype)
                all_values = None

        all_logps = self.get_batch_logps(all_logits.to(self.policy_dtype), batch['target_labels'])
        # Returned tensors will have sequence length that is one less than the inputs (to account for label shifting).
        all_logits = all_logits[:, :-1].contiguous()
        all_logps = all_logps.contiguous()

        return all_logps, all_logits, all_values

    def get_reward_scores(self, batch: Dict[str, torch.LongTensor]) -> torch.FloatTensor:
        """Get reward scores either from reward model or binary labels.
        
        Args:
            batch: Dictionary containing batch data
            
        Returns:
            torch.FloatTensor of shape (microbatch_size,) containing reward scores
        """
        if self.reward_model is not None:
            # Decode the sequences using policy tokenizer
            sequences = self.tokenizer.batch_decode(batch['target_combined_input_ids'], skip_special_tokens=True)
            # Encode with reward model tokenizer
            reward_inputs = self.reward_tokenizer(
                sequences,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=self.config.model.max_length
            ).to(self.accelerator.device)

            with torch.no_grad():
                # Get reward model scores
                outputs = self.reward_model(reward_inputs['input_ids'], attention_mask=reward_inputs['attention_mask'])
                # Use the positive class logit as the reward score
                reward_scores = outputs.logits[:, 1]
        else:
            # Use binary labels (1 for chosen, -1 for rejected)
            reward_scores = torch.tensor([(1 if batch['status'][i] == 'chosen' else -1) for i in range(len(batch['status']))])

        return reward_scores
    
    def compute_advantages(self, values: torch.FloatTensor, rewards: torch.FloatTensor, masks: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Estimate the advantages and rewards for every token taken.

        Args:
            values: the estimated values of the tokens. Should already be detached from graph.
            rewards: signal from the environment as to whether the generation is good or bad.
                In the basic implementation, this is only one nonzero reward, on the last unpadded token of each sequence.
                torch tensor of shape (batch size, sequence length)
            masks: torch tensor of shape (batch size, sequence length); 1 if token should be considered and 0 otherwise

        Returns:
            advantages: torch tensor of shape (batch size, sequence length)
            returns: Also called 'rewards-to-go'.
                Only tokens after the current token are used to calculate this: http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf
                torch tensor of shape (batch size, sequence length)
        """
        values = values * masks
        rewards = rewards * masks
        gae = 0 # generalized advantage estimation
        seq_len = rewards.shape[-1]
        advantages_reversed = []
        
        discounted_future_reward = torch.zeros_like(rewards[:,0])
        discounted_future_rewards_reversed = []

        for t in reversed(range(seq_len)):
            # see https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-fe1b3c5549e8
            delta = rewards[:, t] + self.config.loss.gamma * (values[:, t + 1] if t < seq_len - 1 else 0.0) - values[:, t]
            gae = delta + self.config.loss.gamma * self.config.loss.lam * gae
            advantages_reversed.append(gae)
            
            discounted_future_reward = rewards[:, t] + self.config.loss.gamma * discounted_future_reward
            discounted_future_rewards_reversed.append(discounted_future_reward)

        advantages = (torch.stack(advantages_reversed[::-1]).transpose(0, 1) * masks)
        returns = (advantages + values).contiguous()
        discounted_future_rewards = (torch.stack(discounted_future_rewards_reversed[::-1]).transpose(0, 1) * masks).contiguous()

        # normalizing advantages leads to more stable learning
        mean_adv, var_adv = masked_mean(advantages, masks), masked_var(advantages, masks)
        normalized_advantages = (advantages - mean_adv) * torch.rsqrt(var_adv + 1e-8)
        normalized_advantages = (normalized_advantages * masks).detach().contiguous()

        return normalized_advantages, returns, discounted_future_rewards

    def loss(self, batch: Dict, episode: Dict) -> Tuple[torch.FloatTensor, Dict]:
        """
        Given the batch statistics and the current episode's values, calculate the loss and return some loss statistics.

        Args:
            batch: dictionary containing batch data (shoud have keys 'values', 'returns', 'advantages', 'logprobs', 'masks')
            episode: dictionary containing the episode data (should have keys 'logits', 'values', 'logprobs')

        Returns:
            loss: combined policy and critic loss of shape (1,)
            loss_stats: dictionary of episode/batch statistics
        """
        value_losses = (episode['values'] - batch['discounted_future_rewards'].detach()) ** 2
        critic_loss = 0.5 * masked_mean(value_losses, batch['masks'])
        
        ratio = self.get_ratios(episode['logprobs'], batch['logprobs'])
        policy_losses = -batch['advantages'] * ratio
        policy_losses_clipped = -batch['advantages'] * torch.clamp(ratio, 1 - self.config.loss.cliprange, 1 + self.config.loss.cliprange)
        policy_loss = masked_mean(torch.max(policy_losses, policy_losses_clipped), batch['masks'])

        KL_penalty = masked_mean(episode['logprobs'] - batch['logprobs'], batch['masks'])

        loss = policy_loss + self.config.loss.critic_coef * critic_loss + self.config.loss.KL_coef * KL_penalty

        loss_stats = {
            'loss/total': loss.detach(),
            'loss/critic': critic_loss.detach(),
            'loss/policy': policy_loss.detach(),
            'clipfrac/policy': masked_mean(torch.gt(policy_losses_clipped, policy_losses).float(), batch['masks']).detach(),
            'loss/entropy': entropy_from_logits(episode['logits'], batch['masks']).detach(),
            'loss/policykl': masked_mean(batch['logprobs'] - episode['logprobs'], batch['masks']).detach(),
            'loss/seqratio': rowwise_product(ratio, batch['masks']).mean().detach(),
        }

        return loss, loss_stats
    
    def get_global_batch_dict(self, batch):
        """
        Get the processed dict for the entire batch.

        Args:
            batch: dictionary containing batch data (shoud have keys 'values', 'returns', 'advantages', 'logprobs', 'masks')

        Returns:
            global_batch_dict: dictionary containing processed batch data
        """
        batch_size = len(batch['prompt_text'])
        batch = {k: v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            masks = (batch['target_labels'][:, 1:] != -100).clone().to(self.policy_dtype)
            logprobs, _, _ = self.forward(self.reference_model, batch, is_policy=False)
            _, _, values = self.forward(self.policy, batch)
            # Get reward scores from either reward model or binary labels
            scores = self.get_reward_scores(batch)
            rewards = torch.zeros_like(masks) 
            for row in range(rewards.shape[0]):
                rewards[row, masks[row].nonzero()[-1]] += scores[row]

            rewards = rewards * masks
            advantages, returns, discounted_future_rewards = self.compute_advantages(values, rewards, masks)
            
        global_batch_dict = {
            "target_combined_input_ids": batch['target_combined_input_ids'],
            "target_labels": batch['target_labels'],
            "target_combined_attention_mask": batch['target_combined_attention_mask'],
            "logprobs": logprobs,
            "rewards": scores,
            "values": values,
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
            "discounted_future_rewards": discounted_future_rewards,
        }
        global_batch_dict = {k: v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in global_batch_dict.items()}

        return global_batch_dict

    def eval(self) -> Dict[str, Dict]:
        """
        Run evaluation on all the examples in the test data and return the metrics from get_batch_metrics.
        This is close-ended evaluation and measures the performance of a single model on a single dataset. 
        It does not compare two models to eacch other.

        Returns:
            A dict of form:
            {
                'metadata': the Hydra config
                'results': a dict of batch metrics (averaged across all of the test data)
            }
        """
        self.accelerator.print(f'Running evaluation after {self.example_counter} train examples')
        self.policy.eval()

        for eval_batch in (tqdm(self.eval_iterator, desc='Computing eval metrics') if self.accelerator.is_main_process else self.eval_iterator):
            eval_batch = {k: v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in eval_batch.items()}
            global_batch_dict = self.get_global_batch_dict(eval_batch)

            with torch.no_grad():
                _, eval_metrics = self.get_batch_metrics(global_batch_dict, mode='eval')

            delete_dicts(eval_batch)

        # Compute mean metrics
        mean_eval_metrics = {}
        for k, v in eval_metrics.items():
            if len(v) > 0:
                mean_eval_metrics[k] = sum(v) / len(v)

        if self.accelerator.is_main_process and self.config.wandb.enabled:
            wandb.log(mean_eval_metrics, step=self.example_counter)
        else:
            results = None

        results = {
            'metadata': OmegaConf.to_container(self.config),
            'results': formatted_dict(mean_eval_metrics),
        }

        delete_dicts(eval_metrics, mean_eval_metrics)
        self.accelerator.free_memory()
        torch.cuda.empty_cache()
        
        return results

    def train(self):
        """Train with PPO."""
        self.policy.train()
        self.reference_model.eval()
        
        last_log = None
        batch_metrics = defaultdict(list)
        accumulated_batches = []

        for train_batch in self.train_iterator:
            # EVALUATION
            if accumulated_batches == [] and ((self.example_counter % self.config.eval_every == 0) or (self.example_counter == 0 and self.config.do_first_eval)):
                results = self.eval()

                if self.example_counter > 0:
                    if self.config.debug:
                        self.accelerator.print('skipping save in debug mode.')
                    elif self.config.intermediate_checkpoints:
                        output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                        self.accelerator.print(f'creating checkpoint to write to {output_dir}...')
                        self.save(output_dir, results['results'], final_save=False)

                self.accelerator.print(results['results'])
                delete_dicts(results)

            # TRAINING
            accumulated_batches.append(train_batch)
            if len(accumulated_batches) < self.config.model.gradient_accumulation_steps:
                continue

            start_time = time.time()
            num_update_iters = self.config.humanline_iters if self.config.humanline else self.config.loss.ppo_epochs

            for _ in range(num_update_iters):
                self.optimizer.zero_grad()
           
                for batch_idx, batch in enumerate(accumulated_batches):  
                    # only synchronize gradients on the last batch to avoid slowdown from unnecessary communication
                    microbatch_size = len(batch['prompt_text'])
                    global_batch_dict = self.get_global_batch_dict(batch)
                    loss, local_batch_metrics = self.get_batch_metrics(global_batch_dict, microbatch_size, mode='train')
                    loss = loss / self.config.model.gradient_accumulation_steps
                    delete_dicts(batch)

                    for k, v in local_batch_metrics.items():
                        batch_metrics[k].extend(v)

                    self.accelerator.backward(loss)
                    
                pretrained_norm = self.accelerator.clip_grad_norm_(self.policy.pretrained_model.parameters(), self.config.model.max_grad_norm)
                v_head_norm = self.accelerator.clip_grad_norm_(self.policy.v_head.parameters(), self.config.model.v_head_max_grad_norm)
                batch_metrics['grad_norm'].extend(torch.as_tensor(v_head_norm + pretrained_norm).reshape(-1).float().cpu().numpy().tolist())
                
                self.optimizer.step()
                if self.config.sync_reference or self.config.humanline:
                    self.sync_reference_with_policy()
            
            self.scheduler.step()
            self.batch_counter += 1
            self.example_counter += self.config.model.batch_size

            step_time = time.time() - start_time
            examples_per_second = self.config.model.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)

            delete_dicts(global_batch_dict, batch, local_batch_metrics)

            if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                mean_train_metrics = {}
                for k, v in batch_metrics.items():
                    if len(v) > 0:
                        mean_train_metrics[k] = sum(v) / len(v)

                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                self.accelerator.print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                if self.config.wandb.enabled and self.accelerator.is_main_process:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()

                delete_dicts(batch_metrics, mean_train_metrics)
                batch_metrics = defaultdict(list)    
            else:
                self.accelerator.print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')

            accumulated_batches = []
            self.free_memory()

    def get_batch_metrics(self, global_batch_dict: Dict, microbatch_size: int=0, mode:str='train'):
        """
        Given a batch that has been processed in the outer loop of PPO, return the batch statistics and the loss.
        """
        # for train
        if microbatch_size:
            indices = torch.randperm(microbatch_size).tolist()
            shuffled_global_batch = {k: v[indices] if isinstance(v, torch.Tensor) else [v[i] for i in indices] for k, v in global_batch_dict.items()}
        # for eval
        else:
            shuffled_global_batch = global_batch_dict

        episode_logprobs, episode_logits, episode_values = self.forward(self.policy, shuffled_global_batch)
        episode = {
            'logprobs': episode_logprobs,
            'logits': episode_logits,
            'values': episode_values,
        }
        loss, metrics = self.loss(shuffled_global_batch, episode)

        metrics['rewards'] = shuffled_global_batch['rewards'].detach()
        metrics['returns/mean'] = masked_mean(shuffled_global_batch['returns'], shuffled_global_batch['masks']).detach()
        metrics['returns/var'] = masked_var(shuffled_global_batch['returns'], shuffled_global_batch['masks']).detach()
        metrics['val/mean'] = masked_mean(shuffled_global_batch['values'], shuffled_global_batch['masks']).detach()
        metrics['val/var'] = masked_var(shuffled_global_batch['values'], shuffled_global_batch['masks']).detach()

        batch_metrics = defaultdict(list)
        for k, v in metrics.items():
            v = self.accelerator.gather(v).flatten()
            batch_metrics[k].extend(torch.as_tensor(v).reshape(-1).float().cpu().numpy().tolist())

        delete_dicts(metrics, episode, global_batch_dict, shuffled_global_batch)
        del episode_logprobs, episode_logits, episode_values

        return loss, batch_metrics

    def save(self, output_dir=None, metrics={}, final_save=True):
        """Save tokenizer, policy model, optimizer, scheduler state to disk."""
        self.accelerator.print(f"Saving...")
        if output_dir is None:
            output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')

        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            self.accelerator.print(f"Saving tokenizer...")
            self.tokenizer.save_pretrained(output_dir)

            with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
                metrics['counter'] = self.example_counter
                json.dump(metrics, f)
        
        self.accelerator.wait_for_everyone()
        self.accelerator.print(f"Saving state...")
        optimizer = self.accelerator.unwrap_model(self.optimizer)
        scheduler = self.accelerator.unwrap_model(self.scheduler)
        if self.accelerator.is_main_process:
            optimizer_state = {
                'state_dict': optimizer.state_dict(),
                'class': optimizer.__class__.__name__,
            }
            torch.save(optimizer_state, os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    
        self.accelerator.wait_for_everyone()
        self.accelerator.print(f"Saving model...")

        if self.config.model.use_peft and final_save:
            state_dict = get_base_model_state_dict_from_peft(
                self.accelerator.get_state_dict(self.policy.pretrained_model),
                self.config.model.peft.lora_alpha,
                self.config.model.peft.lora_r,
            )
            unwrapped_model = self.accelerator.unwrap_model(self.policy.pretrained_model).base_model
        else:
            state_dict = self.accelerator.get_state_dict(self.policy.pretrained_model)
            unwrapped_model = self.accelerator.unwrap_model(self.policy.pretrained_model)

        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=state_dict,
            safe_serialization=False
        )
            
        self.accelerator.wait_for_everyone()

        self.accelerator.save(self.policy.v_head.state_dict(), os.path.join(output_dir, "v_head.pt"))
        self.accelerator.wait_for_everyone()


class BradleyTerryTrainer(PairedPreferenceTrainer):
    policy_hf_model_class = AutoModelForBradleyTerry
    use_reference_model = False

    def forward(self, model: AutoModelForBradleyTerry, 
                batch: Dict[str, Union[List, torch.LongTensor]]
                ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Get logits for both chosen and rejected examples.
        
        Args:
            model: The Bradley-Terry model
            batch: Dictionary containing batched inputs
            
        Returns:
            chosen_logits: Raw logits for chosen examples
            rejected_logits: Raw logits for rejected examples
        """
        concatenated_batch = self.concatenated_inputs(batch)
        
        with self.accelerator.autocast():
            all_outputs = model(
                concatenated_batch['concatenated_combined_input_ids'],
                attention_mask=concatenated_batch['concatenated_combined_attention_mask'],
            )
            
        # Split into chosen and rejected based on batch size
        microbatch_size = batch['chosen_combined_input_ids'].shape[0]
        chosen_logits = all_outputs.logits[:microbatch_size]
        rejected_logits = all_outputs.logits[microbatch_size:]
        
        return chosen_logits, rejected_logits

    def loss(self,
             batch: Dict,
             policy_chosen_logits: torch.FloatTensor,
             policy_rejected_logits: torch.FloatTensor,
             *args) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute Bradley-Terry loss given the logits for chosen and rejected examples.
        
        The Bradley-Terry model predicts P(A > B) = sigmoid(score_A - score_B)
        where score_X is the model's predicted score for item X.
        
        Args:
            policy_chosen_logits: Logits from model for chosen examples (microbatch_size, 2)
            policy_rejected_logits: Logits from model for rejected examples (microbatch_size, 2)
            
        Returns:
            losses: The computed losses
            chosen_rewards: Scores for chosen examples 
            rejected_rewards: Scores for rejected examples
        """
        # Extract the scores (logits[:, 1] represents the positive class logit)
        chosen_scores = policy_chosen_logits[:, 1]
        rejected_scores = policy_rejected_logits[:, 1]
        
        # Compute probability of chosen being preferred over rejected
        logits = chosen_scores - rejected_scores
        
        # Binary cross entropy loss with labels of 1 (chosen should be preferred)
        labels = torch.ones_like(logits)
        losses = F.binary_cross_entropy_with_logits(logits, labels)
        
        return losses, chosen_scores.detach(), rejected_scores.detach()

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str = 'train'):
        """Compute metrics for a batch of examples.
        
        Args:
            batch: The input batch
            mode: Either 'train' or 'eval'
            
        Returns:
            loss: The total loss for the batch
            metrics: Dictionary of metrics
        """
        metrics = {}
        
        policy_chosen_logits, policy_rejected_logits = self.forward(self.policy, batch)
        losses, chosen_rewards, rejected_rewards = self.loss(batch, policy_chosen_logits, policy_rejected_logits)
        
        # Compute accuracy
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        # Gather metrics across all processes
        metrics[f'rewards_{mode}/chosen'] = self.accelerator.gather(chosen_rewards.detach())
        metrics[f'rewards_{mode}/rejected'] = self.accelerator.gather(rejected_rewards.detach())
        metrics[f'rewards_{mode}/accuracies'] = self.accelerator.gather(reward_accuracies.detach())
        metrics[f'rewards_{mode}/margins'] = self.accelerator.gather((chosen_rewards - rejected_rewards).detach())
        metrics[f'loss/{mode}'] = self.accelerator.gather(losses.detach()).mean()

        return losses.mean(), metrics
