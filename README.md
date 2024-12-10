
# **H**um**a**n-Centered **Lo**ss Functions (HALOs) :innocent:

This repo allows you to align LLMs with various methods, such as DPO, KTO, and an offline version of PPO.
It was originally released with the KTO paper but has since been significantly revised to support LoRAs, reference logit caching, and easy evaluation (for the original code, see the `legacy` branch of the repo).

Compared to alternatives like TRL or Axlotl, HALOs sacrifices some functionality for:
- modularity: Dataloading, training, and sampling are all separate.
- extensibility: You can quickly write your own dataloader or implement a new alignment loss.
- simplicity: The repo is small enough to hack on.

It has been tested at scales from 1B to 30B LLMs; an earlier version was used to train the Archangel suite of models on Huggingface.

Configs are handled by [Hydra](https://hydra.cc/), jobs are launched with [Accelerate](https://huggingface.co/docs/accelerate/en/index), and all training is done with FSDP by default. To first SFT a model from the Hugginface repo `meta-llama/Meta-Llama-3-8B`, run a command like

```accelerate launch --config_file accelerate_config/fsdp_8gpu.yaml --main_process_port 29500 launch.py loss=sft model=llama datasets=[ultrabin] exp_name=llama3-8b_sft ++cache_dir=/data/models ++model.name_or_path=meta-llama/Meta-Llama-3-8B```

which will save a model to `/data/models/llama3-8b_sft/FINAL/`. To then align the SFT model with KTO, run a command like

```accelerate launch --config_file accelerate_config/fsdp_8gpu.yaml --main_process_port 29500 launch.py loss=kto model=llama datasets=[ultrabin] exp_name=llama3-8b_sft_kto ++cache_dir=/data/models ++model.name_or_path=meta-llama/Meta-Llama-3-8B ++model.load_from=/data/models/llama3-8b_sft/FINAL/```

which will save a model to `/data/models/llama3-8b_sft_kto/FINAL`.


## Quickstart

1. First, clone the repo and install the dependencies. This might take a while. The package versions are important---if you change them, there is no guarantee the code will run.

   ```console
   . install.sh
   ```

2. Determine whether you need a new dataset. If you have a dataset that you want to refer to as `foo` when you launch jobs, add a function called `get_foo` in `dataloader.py` that will return a `Dataset` instance. This function should have the following signature, where `split` should be either `train` or `test`:

   ```def get_foo(split: str, *args, **kwargs) -> Dataset:```
    
   Determine whether you need a new dataloader. Each loss in `config/loss/` has one corresponding dataloader; for KTO, it is `dataloader.UnpairedPreferenceDataLoader`. You will probably not need to write a new dataloader unless you are doing something creative, like turning score-based data into preferences or binary feedback. 

3. Determine whether you need a new trainer. In most cases, this will subclass either `UnpairedPreferenceTrainer` (i.e., KTO-style) or `PairedPreferenceTrainer` (i.e., DPO-style). If you need highly custom behavior that is not in either, then you can subclass `BasicTrainer` directly.

   We can implement a dummy version of KTO as follows (not that this is different from the proper version of KTO in `KTOTrainer`). To make DummyKTOTrainer, we just subclass `trainers.UnpairedPreferenceTrainer` as `trainers.DummyKTOTrainer` and overwrite the loss function definition. 

   ```python
   class DummyKTOTrainer(UnpairedPreferenceTrainer):
      """A fake version of KTO meant to introduce you to the HALOs repo."""
      def loss(self,
           policy_chosen_logps: torch.FloatTensor,
           policy_rejected_logps: torch.FloatTensor,
           reference_chosen_logps: torch.FloatTensor,
           reference_rejected_logps: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
      chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
      rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

      chosen_logratios = (policy_chosen_logps - reference_chosen_logps)
      rejected_logratios = (policy_rejected_logps - reference_rejected_logps)

      losses = torch.cat((1 - F.sigmoid(self.config.loss.beta * (chosen_logratios - rejected_KL)), 1 - F.sigmoid(self.config.loss.beta * (chosen_KL - rejected_logratios))), 0)

      chosen_rewards = self.config.loss.beta * (policy_chosen_logps - reference_chosen_logps).detach()
      rejected_rewards = self.config.loss.beta * (policy_rejected_logps - reference_rejected_logps).detach()

      return losses, chosen_rewards, rejected_rewards
   ```

4. If we wanted, we could add a file to the `config/loss` folder specifying the details of the Dummy KTO loss:

   ```yaml
    name: dummy-kto
    beta: 0.1 # the temperature parameter for dummy KTO; lower values mean we care less about the reference model
    trainer: DummyKTOTrainer # implemented in trainers.py
    dataloader: UnpairedPreferenceDataLoader # already exists in dataloaders.py
    ```
    
    Similarly, to support a new class of model, we would add a yaml file under `config/model` that inherits from `config/model/base_model.yaml`.

5. Now we can start training a model! Let's align a Llama3-8B model on the Ultrafeedback and SHP datasets. First, setup up logging with `wandb login` and run `wandb offline` if your GPUs are not connected to the Internet. Then to launch a job:

   ```console
   accelerate launch \
      --config_file accelerate_config/fsdp_8gpu.yaml \   # accelerate config for 8-gpu allocation
      --main_process_port 29500 \                        # port for gpu communication
      launch.py \                                        # main file for launching job
      loss=dummy-kto \                                   # must be a file name in config/loss
      model=llama \                                      # must be a file name in config/model
      datasets=[ultrabin,shp] \                          # list of datasets, each with a method (e.g., get_shp) in train/dataloader.py
      exp_name=llama3-8b_sft_dummy-kto \                 # experiment name, also the subfolder in cache dir for saving the model          
      ++cache_dir=/data/models \                               # set the cache directory 
      ++model.name_or_path=meta-llama/Meta-Llama-3-8B \        # HF (or local) repo containing model configs, vocab, etc.
      ++model.load_from=/data/models/llama3-8b_sft/FINAL/ \    # load existing model as starting point; if empty, use model.name_or_path
      ++lr=5e-6 \                                              # set the learning rate
      ++loss.beta=0.1                                          # set a KTO-specific hyperparameter (see config/loss/kto.yaml for details)
   ```

   That's it! Your model will be saved to `/data/models/llama3-8b_sft_dummy-kto/FINAL`.

6. We can now evaluate the aligned model. First, to evaluate on AlpacaEval (you need to set OPENAI_API_KEY for this to work):

   ```console
   python -m train.sample /data/models/llama3-8b_sft_dummy-kto/FINAL --gpu_count 1 --output_file outputs/llama3-8b_sft_dummy-kto.json --datasets alpacaeval
   alpaca_eval evaluate --is_overwrite_leaderboard=True --model_outputs=outputs/llama3-8b_sft_dummy-kto.json
   ```

   Then, we can run the model on various benchmarks from LMEval, which was downloaded during installation:

   ```console
   export MODEL_PATH=/data/models/llama3-8b_sft_dummy-kto/FINAL
   lm_eval --model hf \
   --model_args pretrained="$MODEL_PATH",tokenizer="$MODEL_PATH",parallelize=True \
   --tasks arc_easy,arc_challenge,winogrande,bbh_cot_fewshot,gsm8k_cot \   # can add any task in LMEval
   --batch_size 4    # bug if you use 'auto' with gsm8k_cot
   ```

   These steps are combined in `benchmark.sh`.


## FAQs

1. Do you support multi-node training?

   Yes, see the `scripts/launch_multinode_batch.sh` and `scripts/launch_multinode_interactive.sh` for how to launch jobs across two nodes in a batch or interactive Slurm job. You may need a custom Accelerate configuration depending on how many nodes you have. Use the 2-node examples in `accelerate_config` as a template.

2. How do I save intermediate checkpoints?

   Set `intermediate_checkpoints` to true in `config/config.yaml` or on the command line with `++config.intermediate_checkpoints=true`.
   Every `config.eval_every` steps, a checkpoint will be saved in the experiment directory ($cache_dir/$exp_name).

3. Where do I find all the Archangel models?

   They are all on the [Huggingface Hub](https://huggingface.co/collections/ContextualAI/archangel-65bd45029fa020161b052430).

4. Do you support LoRA training?

   Yes. Set `use_peft` to true in `config/model/base_model.yaml` or on the command line with `++model.use_peft=true`. You can either use the default LoRA hyperparameters in `config/model/base_model.yaml` or override them on the command line (e.g., `++model.peft.lora_r=128`). Note that intermediate checkpoints during LoRA training will only be the LoRA module, but the LoRA weights will be merged with the model before the final save.

5. Do you support FlashAttention?

   Yes, just override `attn_implementation` to `flash_attention_2` in `model/base_model.yaml`, on the command line, or in the any of the files that inherit from `model/base_model.yaml`. This is done by default for certain model classes.

6. Can I precompute the log probabilities of the reference model to save memory?

   Yes. Simply set `++cache_reference_logprobs=true` to precompute the log probabilities from the reference model, which will substantially reduce memory. If you are using the same reference model across multiple jobs, which is common, you can override `++reference model=PATH` to the log probabilities that were cached in a pickle file from a previous job.

7. I am getting an error that looks like [rank1]: `torch.distributed.DistBackendError: [1] is setting up NCCL communicator and retrieving ncclUniqueId from [0] via c10d key-value store by key '0', but store->get('0') got error: Socket Timeout`.

   This is because you did not set up wandb, so machine 0 is waiting for your input to setup wandb while the remaining machines are blocked. Resolve this by doing `wandb login` and then running `wandb offline` if your machines are not connected to the Internet.

   
## Citation

If you find this repo useful, please feel free to cite:

```
@inproceedings{ethayarajhmodel,
  title={Model Alignment as Prospect Theoretic Optimization},
  author={Ethayarajh, Kawin and Xu, Winnie and Muennighoff, Niklas and Jurafsky, Dan and Kiela, Douwe},
  booktitle={Forty-first International Conference on Machine Learning}
}
```
