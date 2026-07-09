#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Download and validate VisDrone data for YOLO-Master experiments.

Usage:
  scripts/download_visdrone_dataset.sh [official|hf|torrent]

Examples:
  scripts/download_visdrone_dataset.sh official
  DATA_SOURCE=hf HF_MAX_WORKERS=1 scripts/download_visdrone_dataset.sh
  scripts/download_visdrone_dataset.sh torrent

Environment:
  PROJECT_DIR             Repo root. Defaults to this script's parent repo.
  DATA_SOURCE             official, hf, or torrent. Defaults to official.
  DATASET_ROOT            Prepared YOLO dataset root.
  DATASET_YAML            Output Ultralytics data yaml.
  VISDRONE_DOWNLOAD_DIR   Official zip cache directory.
  RETRIES                 Download retries. Defaults to 10.
  OVERWRITE               Set to 1 to recreate prepared dataset.
  HF_DATASET_REPO         Hugging Face dataset repo.
  HF_SNAPSHOT_DIR         Hugging Face snapshot directory.
  HF_ENDPOINT             Hugging Face endpoint. Defaults to https://hf-mirror.com.
  HF_MAX_WORKERS          HF parallel downloads. Defaults to 1 to reduce 429s.
  HF_ALLOW_PATTERNS       Space-separated HF allow patterns.
  TORRENT_FILE            Local .torrent file.
  TORRENT_DOWNLOAD_DIR    Directory for torrent payload.
  TORRENT_STAGE_DIR       Directory where inner VisDrone.zip is extracted.
  TORRENT_SEED_TIME       aria2 seed time after download. Defaults to 0.
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${script_dir}/.." && pwd)}"
DATA_SOURCE="${1:-${DATA_SOURCE:-official}}"

DATASET_ROOT="${DATASET_ROOT:-/jpfs/huangyidan3/datasets/VisDrone}"
DATASET_YAML="${DATASET_YAML:-/jpfs/huangyidan3/datasets/VisDrone.yaml}"
VISDRONE_DOWNLOAD_DIR="${VISDRONE_DOWNLOAD_DIR:-/jpfs/huangyidan3/datasets/downloads/VisDrone}"
RETRIES="${RETRIES:-10}"
OVERWRITE="${OVERWRITE:-0}"

HF_DATASET_REPO="${HF_DATASET_REPO:-banu4prasad/VisDrone-Dataset}"
HF_SNAPSHOT_DIR="${HF_SNAPSHOT_DIR:-/jpfs/huangyidan3/datasets/hf/VisDrone-Dataset}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-1}"
HF_ALLOW_PATTERNS="${HF_ALLOW_PATTERNS:-README.md visdrone.yaml VisDrone2019-DET-train/** VisDrone2019-DET-val/**}"

TORRENT_FILE="${TORRENT_FILE:-${PROJECT_DIR}/VisDrone.torrent}"
TORRENT_DOWNLOAD_DIR="${TORRENT_DOWNLOAD_DIR:-${VISDRONE_DOWNLOAD_DIR}/torrent}"
TORRENT_STAGE_DIR="${TORRENT_STAGE_DIR:-${VISDRONE_DOWNLOAD_DIR}/torrent_stage}"
TORRENT_SEED_TIME="${TORRENT_SEED_TIME:-0}"

count_files() {
  local dir="$1"
  if [[ -d "${dir}" ]]; then
    find "${dir}" -maxdepth 1 \( -type f -o -type l \) | wc -l
  else
    echo 0
  fi
}

validate_layout() {
  local root="$1"
  local train_images=0
  local train_labels=0
  local val_images=0
  local val_labels=0

  if [[ -d "${root}/images/train" ]]; then
    train_images="$(count_files "${root}/images/train")"
    train_labels="$(count_files "${root}/labels/train")"
    val_images="$(count_files "${root}/images/val")"
    val_labels="$(count_files "${root}/labels/val")"
  else
    train_images="$(count_files "${root}/VisDrone2019-DET-train/images")"
    train_labels="$(count_files "${root}/VisDrone2019-DET-train/labels")"
    val_images="$(count_files "${root}/VisDrone2019-DET-val/images")"
    val_labels="$(count_files "${root}/VisDrone2019-DET-val/labels")"
  fi

  echo "Validation counts:"
  echo "  train/images: ${train_images}"
  echo "  train/labels: ${train_labels}"
  echo "  val/images:   ${val_images}"
  echo "  val/labels:   ${val_labels}"

  if (( train_images < 1000 || train_labels < 1000 || val_images < 100 || val_labels < 100 )); then
    echo "ERROR: VisDrone dataset is incomplete and cannot be used for training." >&2
    return 1
  fi

  if [[ ! -s "${DATASET_YAML}" ]]; then
    echo "ERROR: missing dataset yaml: ${DATASET_YAML}" >&2
    return 1
  fi
}

if [[ "${DATA_SOURCE}" == "-h" || "${DATA_SOURCE}" == "--help" ]]; then
  usage
  exit 0
fi

cd "${PROJECT_DIR}"

echo "Project: ${PROJECT_DIR}"
echo "Data source: ${DATA_SOURCE}"
echo "Dataset root: ${DATASET_ROOT}"
echo "Dataset yaml: ${DATASET_YAML}"

case "${DATA_SOURCE}" in
  official)
    args=(
      --root "${DATASET_ROOT}"
      --download-dir "${VISDRONE_DOWNLOAD_DIR}"
      --yaml-out "${DATASET_YAML}"
      --retries "${RETRIES}"
    )
    if [[ "${OVERWRITE}" == "1" ]]; then
      args+=(--overwrite)
    fi
    python scripts/download_visdrone.py "${args[@]}"
    ;;
  hf)
    read -r -a hf_allow_patterns <<< "${HF_ALLOW_PATTERNS}"
    python scripts/download_hf_dataset.py \
      --repo-id "${HF_DATASET_REPO}" \
      --local-dir "${HF_SNAPSHOT_DIR}" \
      --endpoint "${HF_ENDPOINT}" \
      --allow-patterns "${hf_allow_patterns[@]}" \
      --max-workers "${HF_MAX_WORKERS}" \
      --retries "${RETRIES}" \
      --no-force

    prepare_args=(
      --snapshot-dir "${HF_SNAPSHOT_DIR}"
      --out-dir "${DATASET_ROOT}"
      --yaml-out "${DATASET_YAML}"
    )
    if [[ "${OVERWRITE}" == "1" ]]; then
      prepare_args+=(--overwrite)
    fi
    python scripts/prepare_visdrone_hf.py "${prepare_args[@]}"
    ;;
  torrent)
    if ! command -v aria2c >/dev/null 2>&1; then
      echo "ERROR: aria2c is required for torrent downloads." >&2
      exit 1
    fi
    if [[ ! -s "${TORRENT_FILE}" ]]; then
      echo "ERROR: missing torrent file: ${TORRENT_FILE}" >&2
      exit 1
    fi
    if ! command -v unzip >/dev/null 2>&1; then
      echo "ERROR: unzip is required to extract the torrent payload." >&2
      exit 1
    fi

    mkdir -p "${TORRENT_DOWNLOAD_DIR}" "${TORRENT_STAGE_DIR}" "${VISDRONE_DOWNLOAD_DIR}" "${DATASET_ROOT}"
    torrent_zip="$(find "${TORRENT_DOWNLOAD_DIR}" -type f -name 'VisDrone.zip' | head -n 1)"
    if [[ -z "${torrent_zip}" ]]; then
      aria2c \
        --continue=true \
        --seed-time="${TORRENT_SEED_TIME}" \
        --dir="${TORRENT_DOWNLOAD_DIR}" \
        "${TORRENT_FILE}"
      torrent_zip="$(find "${TORRENT_DOWNLOAD_DIR}" -type f -name 'VisDrone.zip' | head -n 1)"
    else
      echo "Using existing torrent payload: ${torrent_zip}"
    fi

    if [[ -z "${torrent_zip}" ]]; then
      echo "ERROR: torrent finished but VisDrone.zip was not found under ${TORRENT_DOWNLOAD_DIR}" >&2
      exit 1
    fi

    marker="${TORRENT_STAGE_DIR}/.extract_complete"
    if [[ "${OVERWRITE}" == "1" || ! -s "${marker}" ]]; then
      if [[ "${OVERWRITE}" == "1" ]]; then
        find "${TORRENT_STAGE_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
      fi
      unzip -q -o "${torrent_zip}" -d "${TORRENT_STAGE_DIR}"
      printf '%s\n' "${torrent_zip}" > "${marker}"
    fi

    train_zip="$(find "${TORRENT_STAGE_DIR}" -type f -name 'VisDrone2019-DET-train.zip' | head -n 1)"
    val_zip="$(find "${TORRENT_STAGE_DIR}" -type f -name 'VisDrone2019-DET-val.zip' | head -n 1)"
    train_dir="$(find "${TORRENT_STAGE_DIR}" -type d -name 'VisDrone2019-DET-train' | head -n 1)"

    if [[ -n "${train_zip}" && -n "${val_zip}" ]]; then
      ln -sf "${train_zip}" "${VISDRONE_DOWNLOAD_DIR}/VisDrone2019-DET-train.zip"
      ln -sf "${val_zip}" "${VISDRONE_DOWNLOAD_DIR}/VisDrone2019-DET-val.zip"
      python scripts/download_visdrone.py \
        --root "${DATASET_ROOT}" \
        --download-dir "${VISDRONE_DOWNLOAD_DIR}" \
        --yaml-out "${DATASET_YAML}" \
        --retries "${RETRIES}" \
        --skip-download
    elif [[ -n "${train_dir}" ]]; then
      raw_root="$(dirname "${train_dir}")"
      if [[ ! -d "${raw_root}/VisDrone2019-DET-val" ]]; then
        echo "ERROR: found train split but not val split under ${raw_root}" >&2
        exit 1
      fi
      for split in train val; do
        target="${DATASET_ROOT}/VisDrone2019-DET-${split}"
        source="${raw_root}/VisDrone2019-DET-${split}"
        if [[ ! -e "${target}" ]]; then
          ln -s "${source}" "${target}"
        fi
      done
      python scripts/download_visdrone.py \
        --root "${DATASET_ROOT}" \
        --download-dir "${VISDRONE_DOWNLOAD_DIR}" \
        --yaml-out "${DATASET_YAML}" \
        --retries "${RETRIES}" \
        --skip-download
    else
      echo "ERROR: could not find VisDrone2019-DET train/val zips or directories in ${TORRENT_STAGE_DIR}" >&2
      exit 1
    fi
    ;;
  *)
    echo "ERROR: unknown data source '${DATA_SOURCE}'. Use official, hf, or torrent." >&2
    usage >&2
    exit 2
    ;;
esac

validate_layout "${DATASET_ROOT}"

echo "VisDrone dataset is ready."
echo "Use DATA=${DATASET_YAML} for training and evaluation."
