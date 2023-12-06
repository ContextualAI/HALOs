---
license: apache-2.0
datasets:
- stanfordnlp/SHP
- Anthropic/hh-rlhf
- OpenAssistant/oasst1
language:
- en
metrics:
- accuracy
tags:
- human feedback
- rlhf
- preferences
- alignment
- HALO
- halos
- dpo
- rl
---

![halos](https://gist.github.com/assets/29318529/fe2d8391-dbd1-4b7e-9dc4-7cb97e55bc06)

This repo contains the model checkpoints for:
- model family <b>pythia6-9b</b>
- optimized with the loss <b>SFT</b>
- aligned using the SHP, Anthropic HH and Open Assistant datasets.

Please refer to our code repository which contains intructions for training your own HALOs and links to our model cards.

If you find this repo or the technical paper useful in your research, please feel free to cite [our work](http://halos.github.io/):
```
@misc{ethayarajh2023halos,
  url = {http://halos.github.io/},
  author = {Ethayarajh, Kawin and Xu, Winnie, and Jurafsky, Dan and Kiela, Douwe},
  title = {Human-Centered Loss Functions (HALOs)},
  publisher = {Contextual AI Blog},
  year = {2023},
}
```