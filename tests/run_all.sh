#!/usr/bin/env bash
# Run the checks. Needs the conda env from environment.yml and a GPU. The reference-pair comparison runs
# only if the reference repository is present at upstream/Dark-sectioning (otherwise SKIPPED). test_consistency
# needs one of your own datasets processed by run_ds.py (pass --dataset/--field/--channel).
PY=python
cd "$(dirname "$0")/.."
status=0; LOG=$(mktemp)
run() { echo "## $1"; $PY "tests/$1" > "$LOG" 2>&1 || status=1; grep -E "$2" "$LOG" || { echo "(no summary lines; full output follows)"; tail -20 "$LOG"; status=1; }; }
run test_cufft_sizes.py        "broken sizes|2612|PASS|FAIL"
run verify_example_data.py     "diff|PASS|FAIL"
run test_upstream_reference.py "vs Dark.tif|PASS|FAIL|SKIPPED"
run test_strength.py           "== |max \|diff\||decreases|PASS|FAIL"
run test_models.py             "slope|neighbour|optical|PASS|FAIL"
run response_curve.py          "^(global|per_slice)|mode "
echo "response curve -> figs/response_curve.png, results/response_curve.csv"
rm -f "$LOG"
exit $status
