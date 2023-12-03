# Human-Aware Loss Functions (HALOs) :innocent:

This repo allows you to design new **human-aware loss functions (HALOs)** for aligning LLMs with offline human feedback at scale [(read more on our blog)]().
It was used to create Archangel, the largest-ever suite of human-feedback-aligned LLMs, and has been tested at scales from 1B to 30B.

The HALOs repo is a fork of the excellently written [DPO repo](https://github.com/eric-mitchell/direct-preference-optimization) and has preserved many design choices from the original.
However, this repo makes loading data and training models more modular and therefore extensible.

## Quickstart

Let's say we want to implement a new HALO called Kahneman-Tversky optimization (KTO).
This is already implemented in this repo based on the details in our [blog](), but let's pretend that it's not. 
What should we do?

1. First, create and activate the conda environment.

    `conda env create -f environment.yml`
   
    `conda activate halos`

3. Determine whether you need a new dataloader. KTO doesn't use preference pairs, just outputs known to be good or bad.
   This means we can use dataloader.UnpairedPreferenceDataLoader. If you wanted a custom dataloader, you would implement it in the same Python file by extending the base DataLoader class.

4. Write a trainer in trainers.py. This should subclass either UnpairedPreferenceTrainer or PairedPreferenceTrainer depending on whether it uses pairs of preferences or not.
   If you need custom behavior that is not in either of the two, then you can subclass BasicTrainer directly. KTO is simple to implement: we just subclass UnpairedPreferenceTrainer
   and overwrite the loss function definition:

   ```python
   class KTOTrainer(UnpairedPreferenceTrainer):
      def loss(self,
           policy_chosen_logps: torch.FloatTensor,
           policy_rejected_logps: torch.FloatTensor,
           reference_chosen_logps: torch.FloatTensor,
           reference_rejected_logps: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
      """Compute the Kahneman-Tversky loss for a batch of policy and reference model log probabilities. 
      For each batch of n/2 chosen examples and n/2 rejected examples (belonging to n different inputs), calculate the loss as follows.

      If generation y ~ p_chosen, where x' ~ are the examples with rejected generations, we have the 'chosen' loss:
          L(x, y) := 1 - sigmoid(beta * (log p_policy(y|x) - log p_reference(y|x) - KL(p_policy(y_rejected|x') || p_reference(y_rejected|x')))
      If generation y ~ p_rejected, , where x' ~ are the examples with chosen generations, we have the 'rejected' loss:
          L(x, y) := 1 - sigmoid(beta * KL(p_policy(y_chosen|x') || p_reference(y_chosen|x')) - [log p_policy(y|x) - log p_reference(y|x)])
      """
      chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
      rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

      chosen_logratios = (policy_chosen_logps - reference_chosen_logps)
      rejected_logratios = (policy_rejected_logps - reference_rejected_logps)

      losses = torch.cat((1 - F.sigmoid(self.config.loss.beta * (chosen_logratios - rejected_KL)), 1 - F.sigmoid(self.config.loss.beta * (chosen_KL - rejected_logratios))), 0)

      chosen_rewards = self.config.loss.beta * (policy_chosen_logps - reference_chosen_logps).detach()
      rejected_rewards = self.config.loss.beta * (policy_rejected_logps - reference_rejected_logps).detach()

      return losses, chosen_rewards, rejected_rewards
   ```

