#!/usr/bin/env bash
# =============================================================================
# Upgrade / install CANN 8.5.0 (Toolkit + Ops) for MindSpore 2.7.x on Ascend.
#
# Official notes:
#   - MindSpore 2.7.2 ONLY supports CANN 8.5.0
#   - Must install BOTH toolkit and chip-matched ops
#   - Needs root; shared NPU servers: get admin approval first
#
# Usage:
#   sudo bash scripts/upgrade_cann_8.5.sh                 # auto-detect arch + chip
#   sudo bash scripts/upgrade_cann_8.5.sh --chip 910b     # Atlas A2 / 910B
#   sudo bash scripts/upgrade_cann_8.5.sh --chip 910      # Atlas training 910
#   sudo bash scripts/upgrade_cann_8.5.sh --chip A3       # Atlas A3
#   sudo bash scripts/upgrade_cann_8.5.sh --chip 310p     # Atlas inference 310P
#   sudo bash scripts/upgrade_cann_8.5.sh --chip 310b     # Atlas 200I/500 A2
#   sudo bash scripts/upgrade_cann_8.5.sh --download-only # only wget packages
#   sudo bash scripts/upgrade_cann_8.5.sh --yes           # non-interactive install
#
# Packages (Huawei OBS, CANN 8.5.T63 / 8.5.0):
#   toolkit / ops URLs are embedded below (HTTP 200 verified).
# =============================================================================

set -euo pipefail

CANN_VER="8.5.0"
OBS_BASE="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/tmp/cann85_packages}"
INSTALL_PATH="${INSTALL_PATH:-/usr/local/Ascend}"
CHIP=""
DOWNLOAD_ONLY=0
ASSUME_YES=0

usage() {
  sed -n '2,25p' "$0"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --chip) CHIP="$2"; shift 2 ;;
    --download-dir) DOWNLOAD_DIR="$2"; shift 2 ;;
    --install-path) INSTALL_PATH="$2"; shift 2 ;;
    --download-only) DOWNLOAD_ONLY=1; shift ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

detect_arch() {
  local m
  m="$(uname -m)"
  case "$m" in
    x86_64) echo "x86_64" ;;
    aarch64|arm64) echo "aarch64" ;;
    *) echo "Unsupported arch: $m" >&2; exit 1 ;;
  esac
}

detect_chip() {
  if [[ -n "$CHIP" ]]; then
    echo "$CHIP"
    return
  fi
  if ! command -v npu-smi >/dev/null 2>&1; then
    echo "npu-smi not found; please pass --chip {910b|910|A3|310p|310b}" >&2
    exit 1
  fi
  local info
  info="$(npu-smi info 2>/dev/null || true)"
  # Heuristic from product name lines
  if echo "$info" | grep -qiE '910B|A800|A900|A2'; then
    echo "910b"
  elif echo "$info" | grep -qiE 'A3|950'; then
    echo "A3"
  elif echo "$info" | grep -qiE '310P|310p'; then
    echo "310p"
  elif echo "$info" | grep -qiE '310B|310b|200I|500 A2'; then
    echo "310b"
  elif echo "$info" | grep -qiE 'Ascend.?910|910\b'; then
    echo "910"
  else
    echo "Cannot infer chip from npu-smi; pass --chip explicitly." >&2
    echo "$info" >&2
    exit 1
  fi
}

ops_filename() {
  local chip="$1" arch="$2"
  case "$chip" in
    910b|A2|a2) echo "Ascend-cann-910b-ops_${CANN_VER}_linux-${arch}.run" ;;
    910)        echo "Ascend-cann-910-ops_${CANN_VER}_linux-${arch}.run" ;;
    A3|a3)      echo "Ascend-cann-A3-ops_${CANN_VER}_linux-${arch}.run" ;;
    310p|310P)  echo "Ascend-cann-310p-ops_${CANN_VER}_linux-${arch}.run" ;;
    310b|310B)  echo "Ascend-cann-310b-ops_${CANN_VER}_linux-${arch}.run" ;;
    *) echo "Unknown chip '$chip'" >&2; exit 1 ;;
  esac
}

download() {
  local url="$1" out="$2"
  if [[ -f "$out" ]]; then
    echo "[skip] exists: $out"
    return
  fi
  echo "[wget] $url"
  wget -c --progress=dot:giga -O "$out.partial" "$url"
  mv "$out.partial" "$out"
}

ARCH="$(detect_arch)"
CHIP="$(detect_chip)"
TOOLKIT_FILE="Ascend-cann-toolkit_${CANN_VER}_linux-${ARCH}.run"
OPS_FILE="$(ops_filename "$CHIP" "$ARCH")"
TOOLKIT_URL="${OBS_BASE}/${TOOLKIT_FILE}"
OPS_URL="${OBS_BASE}/${OPS_FILE}"

# Optional: driver+toolkit combo package (large). Not required if toolkit+ops used.
COMBO_FILE="Ascend-cann_${CANN_VER}_linux-${ARCH}.run"
COMBO_URL="${OBS_BASE}/${COMBO_FILE}"

echo "=============================================="
echo " CANN ${CANN_VER} install helper"
echo " arch=${ARCH}  chip=${CHIP}"
echo " download_dir=${DOWNLOAD_DIR}"
echo " install_path=${INSTALL_PATH}"
echo "----------------------------------------------"
echo " Toolkit URL:"
echo "   ${TOOLKIT_URL}"
echo " Ops URL:"
echo "   ${OPS_URL}"
echo " (optional combo package)"
echo "   ${COMBO_URL}"
echo "=============================================="

if [[ "${EUID}" -ne 0 && "${DOWNLOAD_ONLY}" -eq 0 ]]; then
  echo "ERROR: install needs root. Re-run with sudo, or use --download-only." >&2
  exit 1
fi

mkdir -p "${DOWNLOAD_DIR}"
cd "${DOWNLOAD_DIR}"

download "${TOOLKIT_URL}" "${DOWNLOAD_DIR}/${TOOLKIT_FILE}"
download "${OPS_URL}" "${DOWNLOAD_DIR}/${OPS_FILE}"

chmod +x "${DOWNLOAD_DIR}/${TOOLKIT_FILE}" "${DOWNLOAD_DIR}/${OPS_FILE}"

if [[ "${DOWNLOAD_ONLY}" -eq 1 ]]; then
  echo "[done] downloaded only. Files in ${DOWNLOAD_DIR}"
  ls -lh "${DOWNLOAD_DIR}/${TOOLKIT_FILE}" "${DOWNLOAD_DIR}/${OPS_FILE}"
  exit 0
fi

if [[ "${ASSUME_YES}" -ne 1 ]]; then
  read -r -p "About to install into ${INSTALL_PATH}. Continue? [y/N] " ans
  [[ "${ans}" == "y" || "${ans}" == "Y" ]] || { echo "Aborted."; exit 1; }
fi

# Free space check (~ toolkit ~1.1G + ops ~2G compressed; install needs more)
avail_kb="$(df -Pk "${INSTALL_PATH%/*}" 2>/dev/null | awk 'NR==2{print $4}')"
if [[ -n "${avail_kb}" && "${avail_kb}" -lt 15000000 ]]; then
  echo "WARNING: less than ~15GB free under ${INSTALL_PATH}; install may fail."
fi

INSTALL_FLAGS=(--install --install-path="${INSTALL_PATH}")
if [[ "${ASSUME_YES}" -eq 1 ]]; then
  # quiet/force accepted by recent CANN run packages
  INSTALL_FLAGS+=(--quiet --force)
fi

echo "[install] toolkit ..."
bash "${DOWNLOAD_DIR}/${TOOLKIT_FILE}" "${INSTALL_FLAGS[@]}"

echo "[install] ops (${CHIP}) ..."
bash "${DOWNLOAD_DIR}/${OPS_FILE}" "${INSTALL_FLAGS[@]}"

# Prefer toolkit set_env; fall back to cann/set_env.sh (layout varies by package)
SET_ENV=""
for cand in \
  "${INSTALL_PATH}/ascend-toolkit/set_env.sh" \
  "${INSTALL_PATH}/cann/set_env.sh" \
  "${INSTALL_PATH}/ascend-toolkit/latest/set_env.sh"
do
  if [[ -f "${cand}" ]]; then
    SET_ENV="${cand}"
    break
  fi
done

echo "=============================================="
echo " Install finished."
if [[ -n "${SET_ENV}" ]]; then
  echo " Load env every shell:"
  echo "   source ${SET_ENV}"
  # shellcheck disable=SC1090
  source "${SET_ENV}"
else
  echo " WARNING: set_env.sh not found under ${INSTALL_PATH}; locate it manually."
fi
echo
echo " Verify:"
echo "   python -c \"import te; print('te OK')\""
echo "   python -c \"import mindspore as ms; ms.set_device('Ascend'); ms.run_check()\""
echo
echo " Also keep ortools away from MindIE libre2 if needed:"
echo "   export LD_LIBRARY_PATH=\\\$CONDA_PREFIX/lib/python*/site-packages/ortools/.libs:\\\$LD_LIBRARY_PATH"
echo "=============================================="
