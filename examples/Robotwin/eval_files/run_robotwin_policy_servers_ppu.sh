#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WALA_ROOT="${WALA_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
WALA_PYTHON="${WALA_PYTHON:-/usr/local/bin/python3}"
CKPT_PATH="${1:-${CKPT_PATH:-}}"
BASE_PORT="${2:-6666}"

if [[ -z "${CKPT_PATH}" ]]; then
    echo "Usage: bash examples/Robotwin/eval_files/run_robotwin_policy_servers_ppu.sh <results/Checkpoints/<run_id>/final_model/pytorch_model.pt> [base_port]" >&2
    exit 1
fi
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-600}"
LOG_DIR="${LOG_DIR:-$(dirname "${CKPT_PATH}")/robotwin_ppu_server_logs/$(date +%Y%m%d_%H%M%S)}"

export PPU_SDK=/usr/local/PPU_SDK
export PPU_HOME=/usr/local/PPU_SDK
export PPU_PATH=/usr/local/PPU_SDK
export CUDA_HOME=/usr/local/PPU_SDK/CUDA_SDK
export CUDA_PATH=/usr/local/PPU_SDK/CUDA_SDK
export LD_LIBRARY_PATH=/usr/local/PPU_SDK/targets/x86_64-linux/lib:/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/lib:${LD_LIBRARY_PATH:-}
export PATH=/usr/local/PPU_SDK/CUDA_SDK/bin:/usr/local/PPU_SDK/bin:/usr/local/PPU_SDK/asight/bin:/usr/local/PPU_SDK/ppu-smi/bin:${PATH}
export PYTHONPATH="${WALA_ROOT}:${PYTHONPATH:-}"

mkdir -p "${LOG_DIR}"

IFS=',' read -ra GPU_IDS <<< "${GPU_IDS_CSV}"
pids=()

cleanup() {
    trap - INT TERM EXIT
    local pid
    for pid in "${pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

port_listening() {
    local port="$1"
    ss -ltnH 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${port}$"
}

wait_for_batch() {
    local elapsed=0
    local idx pid port log_file
    local all_ready

    while (( elapsed < SERVER_START_TIMEOUT )); do
        all_ready=true
        for ((idx = 0; idx < ${#batch_pids[@]}; idx++)); do
            pid="${batch_pids[$idx]}"
            port="${batch_ports[$idx]}"
            log_file="${batch_logs[$idx]}"
            if ! kill -0 "${pid}" 2>/dev/null; then
                if wait "${pid}"; then
                    status=0
                else
                    status=$?
                fi
                echo "[ERROR] Policy server on port ${port} exited during startup with status ${status}." >&2
                echo "[ERROR] See ${log_file}." >&2
                return 1
            fi
            if ! port_listening "${port}"; then
                all_ready=false
            fi
        done
        if ${all_ready}; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done

    echo "[ERROR] Policy-server startup exceeded ${SERVER_START_TIMEOUT}s." >&2
    return 1
}

echo "[INFO] WALA_ROOT=${WALA_ROOT}"
echo "[INFO] WALA_PYTHON=${WALA_PYTHON}"
echo "[INFO] CKPT_PATH=${CKPT_PATH}"
echo "[INFO] BASE_PORT=${BASE_PORT}"
echo "[INFO] GPU_IDS=${GPU_IDS_CSV}"
echo "[INFO] JOBS_PER_GPU=${JOBS_PER_GPU}"
echo "[INFO] SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT}"
echo "[INFO] LOG_DIR=${LOG_DIR}"

slot=0
for ((repeat = 0; repeat < JOBS_PER_GPU; repeat++)); do
    batch_pids=()
    batch_ports=()
    batch_logs=()
    for gpu in "${GPU_IDS[@]}"; do
        port=$((BASE_PORT + slot))
        log_file="${LOG_DIR}/server_gpu${gpu}_port${port}.log"
        echo "[INFO] Launching server gpu=${gpu} port=${port} log=${log_file}"
        (
            cd "${WALA_ROOT}"
            CUDA_VISIBLE_DEVICES="${gpu}" "${WALA_PYTHON}" deployment/model_server/server_policy.py \
                --ckpt_path "${CKPT_PATH}" \
                --port "${port}" \
                --use_bf16 \
                --idle_timeout -1
        ) > "${log_file}" 2>&1 &
        pids+=("$!")
        batch_pids+=("$!")
        batch_ports+=("${port}")
        batch_logs+=("${log_file}")
        slot=$((slot + 1))
    done
    echo "[INFO] Waiting for server batch $((repeat + 1))/${JOBS_PER_GPU} to become ready..."
    wait_for_batch
    echo "[INFO] Server batch $((repeat + 1))/${JOBS_PER_GPU} is ready."
done

echo "[INFO] Started ${#pids[@]} server processes."
echo "[INFO] Press Ctrl-C to stop them."

if wait -n "${pids[@]}"; then
    status=0
else
    status=$?
fi
echo "[ERROR] A policy server exited unexpectedly with status ${status}." >&2
echo "[ERROR] Inspect logs under ${LOG_DIR}." >&2
if (( status == 0 )); then
    exit 1
fi
exit "${status}"
