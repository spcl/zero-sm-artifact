#!/bin/bash
# Creates $WORK/src/megatron-pp: Megatron-LM with symmetric-memory pipeline P2P (--use-symmetric-memory-p2p),
# GitHub vasunvidia/Megatron-LM at 9cfaf94465a7660c789ca177e0acff5a1bf97001, plus e2e/pp/megatron/0001-*.patch.
# Idempotent; runs on a login node.
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/env/cluster.env"
dst=$WORK/src/megatron-pp
base=9cfaf94465a7660c789ca177e0acff5a1bf97001
tree=8c43bdeadd6a33748afb0cddb84c3e539869aa6d   # tree of base + patch

if [[ -d $dst/.git && $(git -C "$dst" rev-parse 'HEAD^{tree}') == "$tree" ]]; then
  echo "$dst is ready"; exit 0
fi
[[ ! -e $dst ]] || { echo "$dst exists with other contents; move it away first" >&2; exit 1; }
mkdir -p "$WORK/src"
git clone -q https://github.com/vasunvidia/Megatron-LM.git "$dst"
git -C "$dst" -c advice.detachedHead=false checkout -q --detach "$base"
git -C "$dst" -c user.name=artifact -c user.email=artifact@localhost am -q --whitespace=nowarn "$ROOT"/e2e/pp/megatron/0001-*.patch
[[ $(git -C "$dst" rev-parse 'HEAD^{tree}') == "$tree" ]]
echo "$dst is ready"
