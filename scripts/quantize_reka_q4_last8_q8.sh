#!/usr/bin/env bash
set -euo pipefail

# Quantize text decoder GGUF with:
#   - default Q4_0
#   - last 8 transformer blocks (24..31) overridden to Q8_0
#
# Usage: ./scripts/quantize_reka_q4_last8_q8.sh [INPUT_GGUF] [OUTPUT_GGUF]
# Optional: QUANTIZE_BIN (default: build/bin or build_linux/bin/llama-quantize)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

INPUT_GGUF="${1:-}"
OUTPUT_GGUF="${2:-}"
THREADS="${THREADS:-16}"

if [[ -z "$INPUT_GGUF" || -z "$OUTPUT_GGUF" ]]; then
  echo "Usage: $0 <INPUT_F16_GGUF> <OUTPUT_GGUF>" >&2
  exit 1
fi

if [[ -z "${QUANTIZE_BIN:-}" ]]; then
  if [[ -x "$REPO_ROOT/build_linux/bin/llama-quantize" ]]; then
    QUANTIZE_BIN="$REPO_ROOT/build_linux/bin/llama-quantize"
  else
    QUANTIZE_BIN="$REPO_ROOT/build/bin/llama-quantize"
  fi
fi

"$QUANTIZE_BIN" \
  --tensor-type 'blk\.(2[4-9]|3[01])\..*=Q8_0' \
  "$INPUT_GGUF" \
  "$OUTPUT_GGUF" \
  Q4_0 \
  "$THREADS"
