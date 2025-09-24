#!/bin/bash
set -euo pipefail

python train.py \
  --vocab vocab_wiki_4k_en.json \
  --signal-dim 64 \
  --latent-dim 32 \
  --num-samples 4096 \
  --epochs 5 \
  --batch-size 32 \
  --save-path checkpoints/vsifu_en.pt
