#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat >&2 <<'EOF'
Usage:
  bash run_robotwin_eval_h20.sh --ppu-host <host> [tunnel options] \
      -m <mode> -n <name> -c <h20_ckpt_path> [start_eval options] <tasks...>

Tunnel options:
  --ppu-host             PPU SSH host (env: ROBOTWIN_PPU_HOST)
  --ppu-ssh-port         PPU SSH port (default: 22, env: ROBOTWIN_PPU_SSH_PORT)
  --ppu-user             PPU SSH user (default: root, env: ROBOTWIN_PPU_USER)
  --identity-file        SSH private key (env: ROBOTWIN_PPU_IDENTITY_FILE)
  --num-servers          Number of forwarded policy-server ports. By default,
                         one server per visible H20 GPU. Client jobs may share
                         these servers.
  --tunnel-timeout       Seconds to wait for SSH forwarding (default: 120)

The remaining options are passed to start_eval.sh. The base port and number of
servers must match run_robotwin_policy_servers_ppu.sh on the PPU machine.

Example:
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash run_robotwin_eval_h20.sh \
      --ppu-host <PPU_SSH_HOST> \
      --ppu-ssh-port <PPU_SSH_PORT> \
      --num-servers 8 \
      -m demo_clean -n full_eval -s 42 -j 4 \
      -c /path/on/h20/to/checkpoint.pt -p 6666 all
EOF
}

detect_gpu_count() {
    local visible="${CUDA_VISIBLE_DEVICES:-}"
    local -a devices=()
    local device=""
    local count=0

    if [[ -n "${visible}" ]]; then
        IFS=',' read -ra devices <<< "${visible}"
        for device in "${devices[@]}"; do
            device="${device//[[:space:]]/}"
            if [[ -n "${device}" ]]; then
                count=$((count + 1))
            fi
        done
    elif command -v nvidia-smi >/dev/null 2>&1; then
        count="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
    fi

    if [[ -z "${count}" || "${count}" == "0" ]]; then
        count=1
    fi
    printf '%s\n' "${count}"
}

require_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] ${name} must be a positive integer, got: ${value}" >&2
        exit 1
    fi
}

local_port_listening() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -ltnH 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${port}$"
    elif command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
    elif command -v netstat >/dev/null 2>&1; then
        netstat -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${port}$"
    else
        return 1
    fi
}

wait_for_tunnel() {
    local timeout_s="$1"
    local elapsed=0
    local port=""
    local all_ready=false

    while (( elapsed < timeout_s )); do
        if ! kill -0 "${TUNNEL_PID}" 2>/dev/null; then
            return 1
        fi

        all_ready=true
        for port in "${FORWARDED_PORTS[@]}"; do
            if ! local_port_listening "${port}"; then
                all_ready=false
                break
            fi
        done

        if ${all_ready}; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    return 1
}

PPU_HOST="${ROBOTWIN_PPU_HOST:-}"
PPU_SSH_PORT="${ROBOTWIN_PPU_SSH_PORT:-22}"
PPU_USER="${ROBOTWIN_PPU_USER:-root}"
IDENTITY_FILE="${ROBOTWIN_PPU_IDENTITY_FILE:-}"
NUM_SERVERS="${ROBOTWIN_PPU_NUM_SERVERS:-}"
TUNNEL_TIMEOUT="${ROBOTWIN_PPU_TUNNEL_TIMEOUT:-120}"
BASE_PORT="${ROBOTWIN_BASE_PORT:-5694}"
JOBS_PER_GPU="${ROBOTWIN_JOBS_PER_GPU:-1}"
START_EVAL_ARGS=()

while (( $# > 0 )); do
    case "$1" in
        --ppu-host)         PPU_HOST="$2"; shift 2 ;;
        --ppu-ssh-port)     PPU_SSH_PORT="$2"; shift 2 ;;
        --ppu-user)         PPU_USER="$2"; shift 2 ;;
        --identity-file)    IDENTITY_FILE="$2"; shift 2 ;;
        --num-servers)      NUM_SERVERS="$2"; shift 2 ;;
        --tunnel-timeout)   TUNNEL_TIMEOUT="$2"; shift 2 ;;
        -p|--base-port)
            BASE_PORT="$2"
            START_EVAL_ARGS+=("$1" "$2")
            shift 2
            ;;
        -j|--jobs-per-gpu)
            JOBS_PER_GPU="$2"
            START_EVAL_ARGS+=("$1" "$2")
            shift 2
            ;;
        -h|--help)          usage; exit 0 ;;
        *)                  START_EVAL_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "${PPU_HOST}" ]]; then
    echo "[ERROR] Missing --ppu-host or ROBOTWIN_PPU_HOST." >&2
    usage
    exit 1
fi

require_positive_integer "PPU SSH port" "${PPU_SSH_PORT}"
require_positive_integer "base port" "${BASE_PORT}"
require_positive_integer "jobs per GPU" "${JOBS_PER_GPU}"
require_positive_integer "tunnel timeout" "${TUNNEL_TIMEOUT}"

if [[ -z "${NUM_SERVERS}" ]]; then
    NUM_SERVERS="$(detect_gpu_count)"
fi
require_positive_integer "number of servers" "${NUM_SERVERS}"

if [[ -n "${IDENTITY_FILE}" && ! -f "${IDENTITY_FILE}" ]]; then
    echo "[ERROR] SSH identity file does not exist: ${IDENTITY_FILE}" >&2
    exit 1
fi

SSH_ARGS=(
    -N
    -T
    -o ExitOnForwardFailure=yes
    -o StrictHostKeyChecking=accept-new
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=3
    -p "${PPU_SSH_PORT}"
)

if [[ -n "${IDENTITY_FILE}" ]]; then
    SSH_ARGS+=(-i "${IDENTITY_FILE}")
fi

FORWARDED_PORTS=()
for (( idx = 0; idx < NUM_SERVERS; ++idx )); do
    port=$((BASE_PORT + idx))
    FORWARDED_PORTS+=("${port}")
    SSH_ARGS+=(-L "${port}:127.0.0.1:${port}")
done
SSH_ARGS+=("${PPU_USER}@${PPU_HOST}")

TUNNEL_PID=""
CLIENT_PID=""
cleanup() {
    trap '' INT TERM EXIT
    if [[ -n "${CLIENT_PID}" ]] && kill -0 "${CLIENT_PID}" 2>/dev/null; then
        echo "[INFO] Stopping RoboTwin clients..."
        kill "${CLIENT_PID}" 2>/dev/null || true
        wait "${CLIENT_PID}" 2>/dev/null || true
    fi
    if [[ -n "${TUNNEL_PID}" ]] && kill -0 "${TUNNEL_PID}" 2>/dev/null; then
        echo "[INFO] Closing PPU SSH tunnel..."
        kill "${TUNNEL_PID}" 2>/dev/null || true
        wait "${TUNNEL_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

last_port=$((BASE_PORT + NUM_SERVERS - 1))
echo "[INFO] Opening ${NUM_SERVERS} PPU policy-server tunnels on H20 ports ${BASE_PORT}-${last_port}"
echo "[INFO] SSH target: ${PPU_USER}@${PPU_HOST}:${PPU_SSH_PORT}"

ssh "${SSH_ARGS[@]}" &
TUNNEL_PID=$!

if ! wait_for_tunnel "${TUNNEL_TIMEOUT}"; then
    if kill -0 "${TUNNEL_PID}" 2>/dev/null; then
        tunnel_status=124
        kill "${TUNNEL_PID}" 2>/dev/null || true
        wait "${TUNNEL_PID}" 2>/dev/null || true
        TUNNEL_PID=""
    elif wait "${TUNNEL_PID}"; then
        tunnel_status=0
    else
        tunnel_status=$?
    fi
    echo "[ERROR] SSH tunnel was not established within ${TUNNEL_TIMEOUT}s (ssh status: ${tunnel_status})." >&2
    exit 1
fi

echo "[INFO] SSH tunnel is running (pid=${TUNNEL_PID}). Starting RoboTwin clients..."
bash "${SCRIPT_DIR}/start_eval.sh" \
    --external-server-host 127.0.0.1 \
    --external-server-count "${NUM_SERVERS}" \
    "${START_EVAL_ARGS[@]}" &
CLIENT_PID=$!

if wait "${CLIENT_PID}"; then
    client_status=0
else
    client_status=$?
fi
CLIENT_PID=""
exit "${client_status}"
