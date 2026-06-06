#!/usr/bin/env bash
# Usage: scripts/train_kenlm.sh <corpus.txt> <out_prefix> <order> <bin_dir>
set -euo pipefail
CORPUS="$1"; OUT="$2"; ORDER="${3:-5}"; BIN="${4:-/tmp/kenlm/build/bin}"
# KenLM CLI는 시스템 libstdc++ 필요 (conda libstdc++ GLIBCXX_3.4.32 충돌 회피)
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"

"$BIN/lmplz" -o "$ORDER" --discount_fallback < "$CORPUS" > "${OUT}.arpa"
"$BIN/build_binary" "${OUT}.arpa" "${OUT}.binary"
echo "built ${OUT}.binary (order=$ORDER)"
