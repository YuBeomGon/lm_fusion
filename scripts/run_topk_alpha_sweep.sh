#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/poc.yaml}"
LM_COND="${2:-honest}"
ALPHAS="${ALPHAS:-0.00,0.05,0.10,0.15,0.20,0.30}"
NON_BASELINE_ALPHAS="${NON_BASELINE_ALPHAS:-0.05,0.10,0.15,0.20,0.30}"
TOPKS="${TOPKS:-32 50 100}"
MODE="${MODE:-topk_strict}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_REPEATED_BASELINE="${SKIP_REPEATED_BASELINE:-1}"

mkdir -p "${LOG_DIR}" docs/report data

echo "[START] $(date -Is)"
echo "[CONFIG] ${CONFIG}"
echo "[LM_COND] ${LM_COND}"
echo "[MODE] ${MODE}"
echo "[TOPKS] ${TOPKS}"
echo "[ALPHAS] ${ALPHAS}"
echo "[NON_BASELINE_ALPHAS] ${NON_BASELINE_ALPHAS}"
echo "[RUN_ID] ${RUN_ID}"
echo "[PYTHON_BIN] ${PYTHON_BIN}"

first_topk=1
for topk in ${TOPKS}; do
  run_alphas="${ALPHAS}"
  if [ "${SKIP_REPEATED_BASELINE}" = "1" ] && [ "${first_topk}" = "0" ]; then
    run_alphas="${NON_BASELINE_ALPHAS}"
  fi
  suffix="sweep_${MODE}_topk${topk}_${RUN_ID}"
  echo "[RUN] topk=${topk} alphas=${run_alphas} suffix=${suffix} $(date -Is)"
  PYTHONPATH=. "${PYTHON_BIN}" scripts/run_eval.py \
    --config "${CONFIG}" \
    --lm-cond "${LM_COND}" \
    --fusion-mode "${MODE}" \
    --asr-topk "${topk}" \
    --alpha-grid "${run_alphas}" \
    --output-suffix "${suffix}"
  echo "[DONE_TOPK] topk=${topk} $(date -Is)"
  first_topk=0
done

PYTHONPATH=. "${PYTHON_BIN}" scripts/write_sweep_report.py \
  --run-id "${RUN_ID}" \
  --lm-cond "${LM_COND}" \
  --mode "${MODE}" \
  --topks "${TOPKS}" \
  --alphas "${ALPHAS}" \
  --output "docs/report/topk_alpha_sweep_${RUN_ID}.md"

echo "[DONE] $(date -Is)"
