#!/usr/bin/env bash
# Validate the profile -> extract pipeline end to end on real profiler output.
#
# Runs against a tiny random-init Qwen2 model rather than olmOCR-2 itself, so
# it finishes in about a minute with nothing to download, while still
# exercising the code path that matters: how torch.profiler actually records
# input shapes, and whether the parser reads them back correctly.
#
# Run this BEFORE committing an A100 slot to the 7B profiling run.
#
#     bash tests/smoke_test.sh
#
# Exits nonzero on the first failure.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p workspace

SEQ=2048

for VARIANT in TinyQwenFA2 TinyQwenSDPA; do
    echo
    echo "==================================================================="
    echo "  $VARIANT"
    echo "==================================================================="

    REPORT="workspace/smoke_${VARIANT}.json"

    uv run profile_model.py \
        --model tests/tiny_qwen.py \
        --class-name "$VARIANT" \
        --input-shape "1,${SEQ}" \
        --dtype bfloat16 \
        --output "$REPORT"

    echo
    echo "--- shape check ---"
    uv run tests/check_report.py "$REPORT" --preset tiny --seq "$SEQ"

    echo
    echo "--- extract ---"
    uv run extract.py --report "$REPORT"

    echo
    echo "--- generated shapes must all differ ---"
    DUPES=$(grep -h '^MODEL_SHAPES' workspace/kernel_*.py | sort | uniq -d || true)
    if [ -n "$DUPES" ]; then
        echo "FAIL: identical MODEL_SHAPES across kernels:"
        echo "$DUPES"
        exit 1
    fi
    grep -h '^MODEL_SHAPES' workspace/kernel_*.py | sort -u | sed 's/^/  /'

    # Fallback shapes must never reach a generated kernel.
    if grep -l 'WARNING: shape source' workspace/kernel_*.py >/dev/null 2>&1; then
        echo "FAIL: kernels generated with fallback shapes:"
        grep -l 'WARNING: shape source' workspace/kernel_*.py
        exit 1
    fi

    rm -f workspace/kernel_*.py workspace/optimization_plan.json
done

echo
echo "==================================================================="
echo "  SMOKE TEST PASSED -- both attention layouts parse correctly."
echo "==================================================================="
echo
echo "Next, the real run:"
echo
echo "  uv run profile_model.py --module transformers \\"
echo "      --class-name Qwen2_5_VLForConditionalGeneration \\"
echo "      --pretrained allenai/olmOCR-2-7B-1025 \\"
echo "      --input-shape 1,${SEQ} --dtype bfloat16"
echo "  uv run tests/check_report.py --preset olmocr2 --seq ${SEQ}"
echo "  uv run extract.py"
