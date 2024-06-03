import os
import sys
import torch

path = sys.argv[1]

sd_path = os.path.join(path, "policy.pt")

sd = torch.load(sd_path)
# Check if already reformatted by checking if first key has model. prefix
if not "state" in list(sd.keys()):
    print('SD seems already reformatted: ', sd.keys())
    sys.exit(0)
torch.save(sd["state"], sd_path)

# Copy in tokenizer etc
os.system(f"mv {sd_path} {os.path.join(path, 'pytorch_model.bin')}")
os.system(f"cp -r /data/niklas/kto_mistralsft/*tok* {path}/")
os.system(f"cp -r /data/niklas/kto_mistralsft/*json* {path}/")
