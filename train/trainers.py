# Copyright (c) 2023 Contextual AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Extendable Trainer classes for aligning LLMs.
The specific class that should be used should be specified in the loss file under config/loss.

The BasicTrainer contains the core methods (e.g., sharding, basic training loop, etc.).
The SFTTrainer, PairedPreferenceTrainer, and UnpairedPreferenceTrainer all subclass BasicTrainer
and override the get_batch_metrics() and (optionally) forward() methods.

The trainer for each loss should subclass either PairedPreferenceTrainer or UnpairedPreferenceTrainer.
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
import gc
from .models import AutoModelForCausalLM, AutoModelForCausalLMWithValueHead
from omegaconf import OmegaConf, DictConfig
from transformers import AutoTokenizer
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
import functools
from typing import Optional, Dict, List, Union, Tuple


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
                 num_skip_batches=0):
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
        self.num_skip_batches = num_skip_batches # when loading from checkpoint
        self.prepare_accelerator()

    def prepare_accelerator(self):
        """Prepare the Accelerator."""
        self.policy, self.reference_model, self.train_iterator, self.eval_iterator, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.policy,
            self.reference_model,
            self.train_iterator, 
            self.eval_iterator, 
            self.optimizer, 
            self.scheduler
        )

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
        
    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy."""
        with self.accelerator.autocast():
            policy_output = self.policy.generate(
                batch['prompt_input_ids'],
                attention_mask=batch['prompt_attention_mask'],
                max_length=self.config.model.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                top_p=self.config.top_p,
            )
        
        policy_output = pad_to_length(policy_output, self.config.model.max_length, self.tokenizer.pad_token_id)
        policy_output = self.accelerator.gather(policy_output)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        return policy_output_decoded

    def loss(self,
             policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

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

        if self.reference_model is not None:
            self.reference_model.eval()

        all_eval_metrics = defaultdict(list)
    
        # Wrap the eval_iterator with accelerator.prepare
        eval_dataloader = self.accelerator.prepare(self.eval_iterator)

        for eval_batch in (tqdm(eval_dataloader, desc='Computing eval metrics') if self.accelerator.is_main_process else eval_dataloader):
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

        for batch in self.train_iterator:
            if self.batch_counter < self.num_skip_batches:
                self.batch_counter += 1
                self.example_counter += self.config.model.batch_size
                continue

            # EVALUATION
            if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval):
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
            self.policy.train()
            accumulated = 0
            start_time = time.time()
            
            with self.accelerator.accumulate(self.policy):
                batch = {k: v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                loss, metrics = self.get_batch_metrics(batch)
                self.accelerator.backward(loss)

                for k, v in metrics.items():
                    batch_metrics[k].extend(torch.as_tensor(v).reshape(-1).float().cpu().numpy().tolist())

                grad_norm = self.accelerator.clip_grad_norm_(self.policy.parameters(), self.config.model.max_grad_norm)
                batch_metrics['grad_norm'].extend(torch.as_tensor(grad_norm).reshape(-1).float().cpu().numpy().tolist())
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                accumulated += 1

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
                self.accelerator.print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                if self.config.wandb.enabled and self.accelerator.is_main_process:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
                batch_metrics = defaultdict(list)
            else:
                self.accelerator.print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')

            delete_dicts(batch, metrics, batch_metrics, mean_train_metrics)

            if accumulated >= self.config.model.gradient_accumulation_steps:
                self.free_memory()
                accumulated = 0

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
            policy_chosen_logps = policy_chosen_logps.view(-1)
            losses = -policy_chosen_logps

        # Gather losses and logps from all processes
        total_nonzero_elements = self.accelerator.gather((policy_chosen_logps != 0).sum().detach()).sum()
        metrics[f'logps_{mode}/chosen'] = self.accelerator.gather(policy_chosen_logps.detach()).sum() / total_nonzero_elements
        metrics[f'loss/{mode}'] = self.accelerator.gather(losses.sum().detach()).sum() / total_nonzero_elements

        del policy_chosen_logits, policy_chosen_logps

        return losses.sum(), metrics


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
            losses, chosen_rewards, rejected_rewards = self.loss(policy_chosen_logps, policy_rejected_logps)
        else:
            policy_chosen_logps, policy_rejected_logps = self.forward(self.policy, batch)
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.forward(self.reference_model, batch, use_cache=self.config.cache_reference_logprobs)
            losses, chosen_rewards, rejected_rewards = self.loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps)

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

        return losses.sum(), metrics


class PairedPreferenceTrainer(BasicTrainer):
    use_reference_model = True

    """A trainer for any loss that uses paired preference, like DPO."""
    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor. The first half is chosen outputs, the second half is rejected.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            
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
            losses, chosen_rewards, rejected_rewards = self.loss(policy_chosen_logps, policy_rejected_logps)
        else:
            policy_chosen_logps, policy_rejected_logps = self.forward(self.policy, batch)
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.forward(self.reference_model, batch, use_cache=self.config.cache_reference_logprobs)
            losses, chosen_rewards, rejected_rewards = self.loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps)

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

        return losses.sum(), metrics


class DPOTrainer(PairedPreferenceTrainer):
    def loss(self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        *args,
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model token-level log probabilities."""
        chosen_rewards = self.config.loss.beta * (policy_chosen_logps.sum(-1) - reference_chosen_logps.sum(-1))
        rejected_rewards = self.config.loss.beta * (policy_rejected_logps.sum(-1) - reference_rejected_logps.sum(-1))

        losses = -F.logsigmoid(chosen_rewards - rejected_rewards)

        return losses, chosen_rewards.detach(), rejected_rewards.detach()


class CDPOTrainer(PairedPreferenceTrainer):
    def loss(self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        *args,
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the CDPO loss for a batch of policy and reference model token-level log probabilities."""
        chosen_rewards = self.config.loss.beta * (policy_chosen_logps.sum(-1) - reference_chosen_logps.sum(-1))
        rejected_rewards = self.config.loss.beta * (policy_rejected_logps.sum(-1) - reference_rejected_logps.sum(-1))
        
        forward_losses = -F.logsigmoid(chosen_rewards - rejected_rewards)
        reverse_losses = -F.logsigmoid(rejected_rewards - chosen_rewards)
        losses = (1 - self.config.loss.epsilon) * forward_losses + self.config.loss.epsilon * reverse_losses

        return losses, chosen_rewards.detach(), rejected_rewards.detach()


class SLiCTrainer(PairedPreferenceTrainer):
    use_reference_model = False

    def loss(self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        *args,
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the SLIC loss as defined by Zhao et al. in https://arxiv.org/pdf/2305.10425.pdf

        Calibration loss defined as:
            L(x, y) := max(0, beta - log p_policy(y_chosen|x) + log p_rejected(y|x))
        For the cross-entropy loss, just use the NLL of the chosen sequence (equivalent to SFT).
        """
        cal_loss = torch.clamp(self.config.loss.beta - policy_chosen_logps.sum(-1) + policy_rejected_logps.sum(-1), min=0)
        reg_loss = -policy_chosen_logps

        losses = cal_loss + self.config.loss.lambda_coef * reg_loss

        chosen_rewards = policy_chosen_logps.detach()
        rejected_rewards = policy_rejected_logps.detach()

        return losses, chosen_rewards, rejected_rewards


class KTOTrainer(UnpairedPreferenceTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous_KL = 0

    def loss(self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_chosen_KL_logps: torch.FloatTensor,
        policy_rejected_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_chosen_KL_logps: torch.FloatTensor,
        reference_rejected_KL_logps: torch.FloatTensor,
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
        log p_policy(y'|x) - log p_reference(y'|x). Doing so avoids the requirement that there be equal numbers of 
        desirable and undesirable examples in the microbatch.
        """
        if self.config.loss.tokenwise_KL:
            KL = torch.Tensor([]).to(self.accelerator.device)

            if policy_chosen_logps.shape[0] != 0:
                KL_chosen = ((policy_chosen_KL_logps - reference_chosen_KL_logps).clamp(min=0).max(-1)).values
                KL = torch.cat((KL, KL_chosen), 0)

            if policy_rejected_logps.shape[0] != 0:
                KL_rejected = ((policy_rejected_KL_logps - reference_rejected_KL_logps).clamp(min=0).max(-1)).values
                KL = torch.cat((KL, KL_rejected), 0)
        else:
            KL_rewards = torch.cat((policy_chosen_KL_logps, policy_rejected_KL_logps), 0).sum(-1) - torch.cat((reference_chosen_KL_logps, reference_rejected_KL_logps), 0).sum(-1)
            # take mean of the KL estimates across all devices in this step
            KL = self.accelerator.gather(KL_rewards.detach()).mean().clamp(min=0)
            KL_chosen, KL_rejected = KL, KL

        if policy_chosen_logps.shape[0] != 0:
            chosen_rewards = (policy_chosen_logps.sum(-1) - reference_chosen_logps.sum(-1))
            chosen_losses = 1 - F.sigmoid(self.config.loss.beta * (chosen_rewards - KL_chosen))
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            chosen_losses = torch.Tensor([]).to(self.policy_dtype).to(self.accelerator.device)
            chosen_rewards = torch.Tensor([]).to(self.policy_dtype).to(self.accelerator.device)
        
        if policy_rejected_logps.shape[0] != 0:
            rejected_rewards = (policy_rejected_logps.sum(-1) - reference_rejected_logps.sum(-1))
            rejected_losses = 1 - F.sigmoid(self.config.loss.beta * (KL_rejected - rejected_rewards))
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            rejected_losses = torch.Tensor([]).to(self.policy_dtype).to(self.accelerator.device)
            rejected_rewards = torch.Tensor([]).to(self.policy_dtype).to(self.accelerator.device)

        losses = torch.cat((self.config.loss.desirable_weight * chosen_losses, self.config.loss.undesirable_weight * rejected_losses), 0)

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
            chosen_KL_logps: log probabilities of the unmatched y'|x (used to estimate the KL divergence between policy and reference
            rejected_KL_logps: log probabilities of the unmatched y'|x (used to estimate the KL divergence between policy and reference
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

        return chosen_logps, rejected_logps, KL_logps[chosen_idx, ...], KL_logps[rejected_idx, ...]
    
    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str='train'):
        """Compute the loss and other metrics for the given batch of inputs."""
        metrics = {}

        policy_chosen_logps, policy_rejected_logps, policy_chosen_KL_logps, policy_rejected_KL_logps = self.forward(self.policy, batch)
        with torch.no_grad():
            reference_chosen_logps, reference_rejected_logps, reference_chosen_KL_logps, reference_rejected_KL_logps = self.forward(self.reference_model, batch, use_cache=self.config.cache_reference_logprobs)
        
        losses, chosen_rewards, rejected_rewards, KL = self.loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_KL_logps,
            policy_rejected_KL_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            reference_chosen_KL_logps,
            reference_rejected_KL_logps,
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

        del policy_chosen_logps, policy_rejected_logps, policy_chosen_KL_logps, policy_rejected_KL_logps, reference_chosen_logps, reference_rejected_logps, reference_chosen_KL_logps, reference_rejected_KL_logps
        del combined_rewards, combined_statuses, all_rewards, all_statuses, chosen_rewards_idx, rejected_rewards_idx, all_KL

        return losses.sum(), metrics


class KTOZeroTrainer(UnpairedPreferenceTrainer):
    def loss(self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        *args
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute a variant of the Kahneman-Tversky loss where the reference point is 0 instead of the expected reward
        (i.e., the human reference point remains what it is at initialization, when policy = reference). This should NOT
        be used for purposes other than to understand the importance of the KL term.

        One can also think of this as a variant of unlikelihood training (Welleck et al., 2023). The purpose of this is to understand 
        the importance of the KL term in the standard variant of the KTO loss. We do *not* reecommend using this in practice as its
        performance is usually inferior. For each batch of n/2 chosen examples and n/2 rejected examples (belonging to n different 
        inputs), calculate the loss as follows.

        If generation y ~ p_chosen, where x' ~ are the examples with rejected generations, we have the 'chosen' loss:
            L(x, y) := 1 - sigmoid(beta * ([log p_policy(y|x) - log p_reference(y|x)] - 0))
        If generation y ~ p_rejected, , where x' ~ are the examples with chosen generations, we have the 'rejected' loss:
            L(x, y) := 1 - sigmoid(beta * (0 - [log p_policy(y|x) - log p_reference(y|x)]))
        """
        if policy_chosen_logps.shape[0] != 0:
            chosen_rewards = (policy_chosen_logps.sum(-1) - reference_chosen_logps.sum(-1))
            chosen_losses = 1 - F.sigmoid(self.config.loss.beta * (chosen_rewards - 0))
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            chosen_losses = torch.Tensor([]).to(self.policy_dtype).to(self.accelerator.device)
            chosen_rewards = torch.Tensor([]).to(self.policy_dtype).to(self.accelerator.device)
        
        if policy_rejected_logps.shape[0] != 0:
            rejected_rewards = (policy_rejected_logps.sum(-1) - reference_rejected_logps.sum(-1))
            rejected_losses = 1 - F.sigmoid(self.config.loss.beta * (0 - rejected_rewards))
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            rejected_losses = torch.Tensor([]).to(self.policy_dtype).to(self.accelerator.device)
            rejected_rewards = torch.Tensor([]).to(self.policy_dtype).to(self.accelerator.device)

        losses = torch.cat((self.config.loss.desirable_weight * chosen_losses, self.config.loss.undesirable_weight * rejected_losses), 0)

        return losses, chosen_rewards, rejected_rewards


class PPOTrainer(BasicTrainer):
    policy_hf_model_class = AutoModelForCausalLMWithValueHead
    use_reference_model = True

    """One-step, offline variant of PPO."""
    def prepare_accelerator(self):
        """Prepare the Accelerator."""
        self.policy.pretrained_model, self.policy.v_head, self.reference_model, self.train_iterator, self.eval_iterator, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.policy.pretrained_model,
            self.policy.v_head,
            self.reference_model,
            self.train_iterator, 
            self.eval_iterator, 
            self.optimizer, 
            self.scheduler
        )

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
            
            discounted_future_rewards_reversed.append(discounted_future_reward)
            discounted_future_reward = rewards[:, t] + self.config.loss.gamma * discounted_future_reward

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
        
        ratio = torch.exp(episode['logprobs'] - batch['logprobs'])
        policy_losses = -batch['advantages'] * ratio
        policy_losses_clipped = -batch['advantages'] * torch.clamp(ratio, self.config.loss.cliprange, 1 / self.config.loss.cliprange)
        policy_loss = masked_mean(torch.max(policy_losses, policy_losses_clipped), batch['masks'])

        KL_penalty = masked_mean(batch['logprobs'] - episode['logprobs'], batch['masks'])

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
        batch['scores'] = torch.Tensor([(1 if batch['status'][i] == 'chosen' else -1) for i in range(batch_size)])
        batch = {k: v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            masks = (batch['target_labels'][:, 1:] != -100).clone().to(self.policy_dtype)
            logprobs, _, _ = self.forward(self.reference_model, batch, is_policy=False)
            _, _, values = self.forward(self.policy, batch)
            
            rewards = torch.zeros_like(masks) 
            for row in range(rewards.shape[0]):
                rewards[row, masks[row].nonzero()[-1]] += batch['scores'][row]

            rewards = rewards * masks
            advantages, returns, discounted_future_rewards = self.compute_advantages(values, rewards, masks)
            
        global_batch_dict = {
            "target_combined_input_ids": batch['target_combined_input_ids'],
            "target_labels": batch['target_labels'],
            "target_combined_attention_mask": batch['target_combined_attention_mask'],
            "logprobs": logprobs,
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

        # Wrap the eval_iterator with accelerator.prepare
        eval_dataloader = self.accelerator.prepare(self.eval_iterator)

        for eval_batch in (tqdm(eval_dataloader, desc='Computing eval metrics') if self.accelerator.is_main_process else eval_dataloader):
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
        self.accelerator.print(f'Using {self.config.optimizer} optimizer with learning rate {self.config.lr}')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))

        self.policy.train()
        self.reference_model.eval()
        
        last_log = None
        batch_metrics = defaultdict(list)

        for batch in self.train_iterator:
            if self.batch_counter < self.num_skip_batches:
                self.batch_counter += 1
                self.example_counter += self.config.model.batch_size
                continue

            # EVALUATION
            if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval):
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
            start_time = time.time()

            batch_size = len(batch['prompt_text'])
            global_batch_dict = self.get_global_batch_dict(batch)

            for ppo_epoch in range(self.config.loss.ppo_epochs):
                with self.accelerator.accumulate(self.policy):
                    loss, local_batch_metrics = self.get_batch_metrics(global_batch_dict, batch_size, mode='train')

                    for k, v in local_batch_metrics.items():
                        batch_metrics[k].extend(v)

                    self.accelerator.backward(loss)
                    v_head_norm = self.accelerator.clip_grad_norm_(self.policy.pretrained_model.parameters(), self.config.model.max_grad_norm)
                    pretrained_norm = self.accelerator.clip_grad_norm_(self.policy.v_head.parameters(), self.config.model.v_head_max_grad_norm)
                    batch_metrics['grad_norm'].extend(torch.as_tensor(v_head_norm + pretrained_norm).reshape(-1).float().cpu().numpy().tolist())
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

            self.batch_counter += 1
            self.example_counter += batch_size

            step_time = time.time() - start_time
            examples_per_second = batch_size / step_time
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

    def get_batch_metrics(self, global_batch_dict: Dict, batch_size: int=0, mode:str='train'):
        """
        Given a batch that has been processed in the outer loop of PPO, return the batch statistics and the loss.
        """
        # for train
        if batch_size:
            indices = torch.randperm(batch_size).tolist()
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

        unwrapped_v_head = self.accelerator.unwrap_model(self.policy.v_head)
        torch.save(unwrapped_v_head.state_dict(), os.path.join(output_dir, "v_head.pt"))
        self.accelerator.wait_for_everyone()