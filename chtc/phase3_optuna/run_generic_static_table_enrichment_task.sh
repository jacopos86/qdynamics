#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <record_id>" >&2
  exit 2
fi

RECORD_ID="$1"
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
RECORDS_PATH="${GENERIC_STATIC_TABLE_RECORDS_PATH:-chtc/phase3_optuna/input/generic_static_table_records.tsv}"
INPUT_ROOT="${GENERIC_STATIC_TABLE_INPUT_ROOT:-raw_outputs/generic_static_table}"
OUTPUT_ROOT="${GENERIC_STATIC_TABLE_ENRICHMENT_ROOT:-raw_outputs/generic_static_table_enrichment}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RECORD_ID}/result"

mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/command.sh" <<EOF
PYTHONPATH=. python -m pipelines.exact_bench.generic_static_metric_enrichment \\
  --run-single \\
  --record-id ${RECORD_ID} \\
  --records ${RECORDS_PATH} \\
  --input-root ${INPUT_ROOT} \\
  --output-dir ${OUTPUT_DIR}
EOF

cd "${REPO_ROOT}"
PYTHONPATH=. python -m pipelines.exact_bench.generic_static_metric_enrichment \
  --run-single \
  --record-id "${RECORD_ID}" \
  --records "${RECORDS_PATH}" \
  --input-root "${INPUT_ROOT}" \
  --output-dir "${OUTPUT_DIR}"
