# **H**um**a**n-Centered **Lo**ss Functions (HALOs) :innocent:

This repo allows you to design new **HumAn-centered LOss functions (HALOs)** for aligning LLMs with offline human feedback at scale [(read more on our blog)]().
It was used to create Archangel, the largest-ever suite of human-feedback-aligned LLMs, and has been tested at scales from 1B to 30B.

This repo draws from the excellently written [DPO repo](https://github.com/eric-mitchell/direct-preference-optimization) and has preserved many design choices from the original.
Some of the key changes we introduced are:
- making data loading more modular, so that you can easily write your own dataloader
- making trainers more modular, so that each HALO has its own trainer subclass
- adding code for doing open-ended evaluation with GPT-4 as a judge
- supporting losses beyond SFT and DPO (including KTO, PPO (offline, off-policy variant), and SLiC)


## Quickstart

Let's say we want to implement a new HALO called Kahneman-Tversky optimization (KTO).
This is already implemented in this repo based on the details in our [blog](), but let's pretend that it's not. 
What should we do?

1. First, create and activate the conda environment.

    `conda env create -f environment.yml`
   
    `conda activate halos`

3. Determine whether you need a new dataloader. KTO doesn't use preference pairs, just knowledge of whether outputs are desirable or undesirable.
   This means we use `dataloader.UnpairedPreferenceDataLoader`. However, that dataloader assumes that you're working with datasets that originally contain preference pairs, like Anthropic HH or SHP.
   If you wanted a custom dataloader, you would implement it in the same Python file by extending the base `DataLoader` class.

5. Write a trainer in `trainers.py`. This should subclass either `UnpairedPreferenceTrainer` or `PairedPreferenceTrainer` depending on whether it uses pairs of preferences or not.
   If you need highly custom behavior that is not in either, then you can subclass `BasicTrainer` directly.

   KTO is simple to implement: we just subclass `trainers.UnpairedPreferenceTrainer` as `trainers.KTOTrainer` and overwrite the loss function definition. KTO has one hyperparameter, beta, which we can access via `self.config.loss.beta`:

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

6. Add a file to the config/loss folder specifying the details of the loss:

   ```yaml
    name: kto
    beta: 0.1 # the temperature parameter for KTO; lower values mean we care less about the reference model
    trainer: KTOTrainer # implemented in trainers.py
    dataloader: UnpairedPreferenceDataLoader # already exists in dataloaders.py
    use_reference_model: true # true because the loss definition includes a reference model
    ```

7. Now we can start training a model! Let's train a Llama-7B model on the SHP, Anthropic HH, and Open Assistant datasets.
   Since the corresponding entry for Llama-7B is config/model/llama7b.yaml, we run a command with [Hydra](https://hydra.cc/docs/intro/):

   `python train.py loss=kto model=llama7b datasets=[shp,hh,oasst] exp_name=kto_llama7b mode=train ++cache_dir=/data/models`

   which will align a Llama-7B model from scratch. If we want to align a model that we've already finetuned with the HALOs repo,
   we can add something like `++model.load_from=/data/models/sft_llama7b/LATEST/policy.pt` to the end of the command.

   That's it! Your model will be saved to `/data/models/kto_llama7b/LATEST/policy.pt`.


8. Let's sample some generations from our newly trained model. The sampling configs are in either `config/config.yaml` or under `models/`.
   We can sample 512 generations from our newly trained model in batches of 32 with the command, which will create a JSON file under samples/{exp_name}.json.

   `python eval.py -c /data/models/kto_llama7b/config.yaml -m sample -n 512 -b 32`

9. After setting `OPENAI_API_KEY`, we can evaluate our aligned model with GPT-4 with the following command, which compares the aligned model's generations to the human-chosen response in the data:

    `python compare.py -f samples/sft_llama7b.json -mc 512 -bk chosen -ck policy -r result.jsonl `


## FAQs

1. Do you support multi-node training?

   No, currently the repo only supports single-node training. Multi-node training will be added at some point in the future.
   Every model in the Archangel suite was trained with 8 x A100 GPUs on a single node.

2. How do I save intermediate checkpoints?

   Set intermediate_checkpoints to true in config/config.yaml or on the command line with `++config.intermediate_checkpoints=true`.
   Every `config.eval_every steps`, a checkpoint will be saved in the experiment directory ($cache_dir/$exp_name).
   
   
## Citation

If you find this repo or the technical paper useful in your research, please feel free to cite [our work](http://halos.github.io/):
```
@misc{ethayarajh2023halos,
  url = {http://halos.github.io/},
  author = {Ethayarajh, Kawin and Xu, Winnie, and Jurafsky, Dan and Kiela, Douwe},
  title = {Human-Aware Loss Functions (HALOs)},
  publisher = {Contextual AI Blog},
  year = {2023},
}
``` 
