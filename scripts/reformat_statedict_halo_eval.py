"""Conversion script for halos trained policy.pt to pytorch_model.bin.
"""
import os
import sys
import torch

# python scripts/reformat_statedict_halo_eval.py /data/models/archangel
path = sys.argv[1]
paths = [path]

for path in paths:
    sd_path = os.path.join(path, "policy.pt")

    sd = torch.load(sd_path)
    # Check if already reformatted by checking if first key has model. prefix
    if not "state" in list(sd.keys()):
        print('SD seems already reformatted: ', sd.keys())
        sys.exit(0)
    # torch.save(sd["state"], sd_path)  # will overwrite original checkpoint = BAD
    torch.save(sd["state"], os.path.join(path, "pytorch_model.bin"))

    # prevent vllm issue: save og state dict but keep the reformatted one a directory up 
    os.system(f"mv {sd_path} {os.path.join(path, '../policy.pt')}")

    # Copy in tokenizer etc
    if 'zephyr' in path:
        print('Copying tokenizer and config for zephyr /data/winnie/zephyr/')
        os.system(f"cp -r /data/winnie/zephyr/*tok* {path}/")
        os.system(f"cp -r /data/winnie/zephyr/*json* {path}/")
    if 'mistral' in path or 'oursft' in path or 'gen_m7' in path:
        print('Copying tokenizer and config from /data/niklas/kto_mistralsft')
        os.system(f"cp -r /data/models/archangel_kto/kto_gen_m7_tulu_ultrabin_shp/LATEST/*tok* {path}/")
        os.system(f"cp -r /data/models/archangel_kto/kto_gen_m7_tulu_ultrabin_shp/LATEST/*json* {path}/")
    elif 'yi34_chat' in path:
        print('Copying tokenizer and config from /data/winnie/yi34b-chat')
        os.system(f"cp -r /data/winnie/yi34b-chat/*tok* {path}/")
        os.system(f"cp -r /data/winnie/yi34b-chat/*json* {path}/")
    elif 'yi34' in path:
        print('Copying tokenizer and config from /data/winnie/yi34b')
        os.system(f"cp -r /data/winnie/yi34b/*tok* {path}/")
        os.system(f"cp -r /data/winnie/yi34b/*json* {path}/")
    elif 'llama' in path:
        print('Copying tokenizer and config from /data/winnie/llama7b')
        os.system(f"cp -r /data/winnie/llama7b/*tok* {path}/")
        os.system(f"cp -r /data/winnie/llama7b/*json* {path}/")
    else:
        raise ValueError(f'Unknown model path: {path}')
    print(f"finished saving {os.path.join(path, 'pytorch_model.bin')}")
