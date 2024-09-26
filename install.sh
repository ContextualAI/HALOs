conda install pip
pip install packaging ninja
ninja --version
echo $?
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install flash-attn --no-build-isolation
pip install transformers==4.44.0 accelerate==0.33.0 vllm==0.5.5 wandb omegaconf openai peft hydra-core==1.3.2
