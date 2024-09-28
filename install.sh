conda install pip
pip install packaging ninja
ninja --version
echo $?
conda install pytorch=2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install flash-attn==2.6.3 --no-build-isolation
pip install transformers==4.44.0 
pip install peft==0.12.0 datasets=2.20.0
pip install accelerate==0.33.0
pip install vllm==0.5.5
pip install wandb omegaconf openai hydra-core==1.3.2