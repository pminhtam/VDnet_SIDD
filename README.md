# VD Net

[Source](https://github.com/zsyOAOA/VDNet)

Paper [Variational Denoising Network: Toward Blind Noise Modeling and Removal (NeurIPS, 2019)](https://papers.nips.cc/paper/8446-variational-denoising-network-toward-blind-noise-modeling-and-removal.pdf)

## Train

### Train model

```
CUDA_VISIBLE_DEVICES=0 python train.py -n ../image/noise/ -g ../image/gt/ -sz 512 -bs 4 -e 100 -se 100 -le 10 -nw 4 -c -ckpt checkpoint  --restart```
```