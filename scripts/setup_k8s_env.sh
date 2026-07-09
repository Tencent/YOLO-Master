#!/usr/bin/env bash
set -euo pipefail

PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.jd.com/pypi/web/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.jd.com}"
PIP_TIMEOUT="${PIP_TIMEOUT:-120}"
PIP_RETRIES="${PIP_RETRIES:-5}"
PIP_UPGRADE_TOOLS="${PIP_UPGRADE_TOOLS:-0}"
PIP_EXTRAS="${PIP_EXTRAS:-}"
PIP_EXTRA_PACKAGES="${PIP_EXTRA_PACKAGES:-pytest}"

pip_args=(
  --index-url "${PIP_INDEX_URL}"
  --trusted-host "${PIP_TRUSTED_HOST}"
  --timeout "${PIP_TIMEOUT}"
  --retries "${PIP_RETRIES}"
)

echo "Using pip index: ${PIP_INDEX_URL}"
if [[ "${PIP_UPGRADE_TOOLS}" == "1" ]]; then
  python -m pip install "${pip_args[@]}" -U pip setuptools wheel
fi

if [[ -n "${PIP_EXTRAS}" ]]; then
  python -m pip install "${pip_args[@]}" -e ".[${PIP_EXTRAS}]"
else
  python -m pip install "${pip_args[@]}" -e .
fi

if [[ -n "${PIP_EXTRA_PACKAGES}" ]]; then
  read -r -a extra_packages <<< "${PIP_EXTRA_PACKAGES}"
  python -m pip install "${pip_args[@]}" "${extra_packages[@]}"
fi
