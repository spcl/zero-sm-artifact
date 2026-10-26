#!/bin/bash
#SBATCH -N 1 -t 04:00:00 -J deepseek-data -o %x-%j.out
# Data of the DeepSeek-V3 recipe, into $WORK/data/deepseek:
#   c4/         MLCommons Llama 3.1 8B pretraining C4 (train shard en_6, 91205 validation samples), Megatron format, 84 GB
#   tokenizer/  Llama 3.1 8B tokenizer files of the same benchmark
#   hf-config/  config.json of deepseek-ai/DeepSeek-V3 (model shape only, no weights)
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/env/cluster.env"
out=$WORK/data/deepseek
mlc=https://training.mlcommons-storage.org/llama3_1
mkdir -p "$out"/{c4,tokenizer,hf-config,npy_index}

fetch() {   # fetch <dir> <url> <md5|sha256> <hash>
  local f
  f=$1/$(basename "$2")
  [[ -s $f ]] || { wget -q -c --tries=5 -O "$f.part" "$2" && mv "$f.part" "$f"; }
  echo "$4  $f" | "${3}sum" -c --quiet
}
fetch "$out/c4" $mlc/datasets/c4/llama3_1_8b/c4-train.en_6_text_document.bin md5 c781dc1a0eeb5a0322ad3b528e25027e
fetch "$out/c4" $mlc/datasets/c4/llama3_1_8b/c4-train.en_6_text_document.idx md5 78b6262d170c7c9839253fdb4f944057
fetch "$out/c4" $mlc/datasets/c4/llama3_1_8b/c4-validation-91205-samples.en_text_document.bin md5 85944abe86263a9a12f283b34af10a2b
fetch "$out/c4" $mlc/datasets/c4/llama3_1_8b/c4-validation-91205-samples.en_text_document.idx md5 32d3afd6f81d320ca88c6675c27cf26f
fetch "$out/tokenizer" $mlc/tokenizers/llama3_1_8b/tokenizer.json md5 c78db1e4b2b306efca2b39555ee69106
fetch "$out/tokenizer" $mlc/tokenizers/llama3_1_8b/tokenizer_config.json md5 53cf4cf12df1da52a733f02170292cbd
fetch "$out/tokenizer" $mlc/tokenizers/llama3_1_8b/special_tokens_map.json md5 7dce6de50917697885e722a430a72073
fetch "$out/hf-config" https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/e815299b0bcbac849fa540c768ef21845365c9eb/config.json \
  sha256 cbf0b95dc614de208a109bb5fd4e7eed11385e9c68411d2c17db5319443035d9
ls -l "$out"/*
