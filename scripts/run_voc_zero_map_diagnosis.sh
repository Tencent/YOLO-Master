#!/usr/bin/env bash
set -euo pipefail

# Server defaults for the current VOC LoRA run. Override any value as an environment variable, for example:
#   RUN_VAL=1 DEVICE=0 BATCH=64 bash scripts/run_voc_zero_map_diagnosis.sh
# Run full validation only after pausing training or confirming enough free GPU memory.

REPO_ROOT="${REPO_ROOT:-/svap_storage/gatilin/workspaces/working/GatilinLAB/YOLO-Master-v260721-experiment}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/detect/yolo-master/lora-sft-exp1}"
DATA_PATH="${DATA_PATH:-/svap_storage/gatilin/workspaces/working/GatilinLAB/ultralytics-8.3.226_softmoe_baseline_v2/ultralytics/cfg/datasets/VOC.yaml}"
EXPECTED_NC="${EXPECTED_NC:-20}"
DEVICE="${DEVICE:-cpu}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-2}"
CONF="${CONF:-0.00001}"
RUN_VAL="${RUN_VAL:-0}"
PROBE_FORWARD="${PROBE_FORWARD:-0}"
SAMPLE_IMAGE="${SAMPLE_IMAGE:-}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPORT="${REPORT:-/tmp/voc_zero_map_diagnosis_${TIMESTAMP}.log}"
JSON_REPORT="${JSON_REPORT:-/tmp/voc_zero_map_diagnosis_${TIMESTAMP}.json}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ultralytics-matplotlib-${USER:-root}}"

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "Repository not found: ${REPO_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${RUN_DIR}/args.yaml" ]]; then
  echo "Run directory is invalid or incomplete: ${RUN_DIR}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${MPLCONFIGDIR}"
export MPLCONFIGDIR

command=(
  python3 scripts/diagnose_voc_zero_map.py
  --run-dir "${RUN_DIR}"
  --data "${DATA_PATH}"
  --expected-nc "${EXPECTED_NC}"
  --device "${DEVICE}"
  --batch "${BATCH}"
  --workers "${WORKERS}"
  --conf "${CONF}"
  --output-json "${JSON_REPORT}"
)

if [[ "${RUN_VAL}" == "1" ]]; then
  command+=(--run-val)
fi
if [[ "${PROBE_FORWARD}" == "1" ]]; then
  command+=(--probe-forward)
fi
if [[ -n "${SAMPLE_IMAGE}" ]]; then
  command+=(--sample-image "${SAMPLE_IMAGE}")
fi

echo "Repository : ${REPO_ROOT}"
echo "Run         : ${RUN_DIR}"
echo "Dataset     : ${DATA_PATH}"
echo "Device      : ${DEVICE}"
echo "Full val    : ${RUN_VAL}"
echo "Report      : ${REPORT}"
echo "JSON report : ${JSON_REPORT}"

"${command[@]}" 2>&1 | tee "${REPORT}"
