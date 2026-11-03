#!/usr/bin/env bash
# 반드시 bash switch.sh ... 로 실행한다. SSH 셸에서 source하지 않는다.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo 'Use: sudo bash switch.sh v1|v2 [baseline_root] [v2_root]' >&2
    return 2
fi
set -Eeuo pipefail
TARGET="${1:-}"
BASELINE="${2:-/root/ai-agent}"
V2_ROOT="${3:-/root/ai-agent-v2}"
[[ "$TARGET" == v1 || "$TARGET" == v2 ]] || { echo 'Target must be v1 or v2' >&2; exit 2; }
[[ "$EUID" == 0 ]] || { echo 'Run with sudo bash' >&2; exit 2; }
for root in "$BASELINE" "$V2_ROOT"; do
    [[ "$root" =~ ^/[A-Za-z0-9_./-]+$ && -d "$root" ]] || { echo "Invalid app root: $root" >&2; exit 2; }
done
BASELINE="$(realpath "$BASELINE")"
V2_ROOT="$(realpath "$V2_ROOT")"
[[ "$BASELINE" != "$V2_ROOT" && "$V2_ROOT" != "$BASELINE/"* ]] || { echo 'Roots must be separate' >&2; exit 2; }
PYTHON="$BASELINE/app-env/bin/python"
UVICORN="$BASELINE/app-env/bin/uvicorn"
[[ -x "$PYTHON" && -x "$UVICORN" ]] || { echo 'Baseline app-env is required' >&2; exit 2; }
command -v flock >/dev/null
command -v curl >/dev/null
command -v ss >/dev/null
exec 9>/run/lock/ai-agent-switch.lock
flock -n 9 || { echo 'Another switch is running' >&2; exit 2; }
if [[ "$TARGET" == v2 ]]; then
    # 서비스에 손대기 전에 모델 없는 필수 검사를 통과해야 한다.
    bash "$V2_ROOT/deploy/gpu_server/check.sh" "$PYTHON"
fi
for unit in rag_service orchestrator; do
    [[ "$(systemctl show "$unit" -p LoadState --value)" == loaded ]] || { echo "Missing unit: $unit" >&2; exit 2; }
done
for unit in vlm_service vllm_vlm; do
    if systemctl is-active --quiet "$unit"; then
        echo 'VLM/indexing is active. Finish it before switching the answer service.' >&2
        exit 2
    fi
done
for spec in 'rag_service:rag' 'orchestrator:orchestrator'; do
    owner="$(systemctl show "${spec%%:*}" -p User --value)"
    lock="/run/lock/ai-agent-${spec##*:}.lock"
    touch "$lock"
    chown "${owner:-root}" "$lock"
    chmod 600 "$lock"
done
STATE=/var/lib/ai-agent
mkdir -p "$STATE"
PREVIOUS=v1
[[ ! -f "$STATE/active-implementation" ]] || PREVIOUS="$(cat "$STATE/active-implementation")"
[[ "$PREVIOUS" == v1 || "$PREVIOUS" == v2 ]] || { echo 'Invalid active state' >&2; exit 2; }

stop_app() {
    local unit group pid gpu i listeners
    local -a old_pids=()
    for unit in orchestrator rag_service; do
        pid="$(systemctl show "$unit" -p MainPID --value)"
        [[ "$pid" == 0 ]] || old_pids+=("$pid")
        group="$(systemctl show "$unit" -p ControlGroup --value)"
        if [[ -n "$group" && -f "/sys/fs/cgroup${group}/cgroup.procs" ]]; then
            while read -r pid; do old_pids+=("$pid"); done < "/sys/fs/cgroup${group}/cgroup.procs"
        fi
    done
    # 단일 unit 이름을 재사용한다. 다른 구현용 별도 unit은 만들지 않는다.
    systemctl stop orchestrator.service rag_service.service || return 1
    for unit in orchestrator rag_service; do
        [[ "$(systemctl show "$unit" -p MainPID --value)" == 0 ]] || return 1
        case "$(systemctl show "$unit" -p ActiveState --value)" in inactive|failed) ;; *) return 1 ;; esac
        group="$(systemctl show "$unit" -p ControlGroup --value)"
        if [[ -n "$group" && -f "/sys/fs/cgroup${group}/cgroup.procs" ]]; then
            [[ -z "$(cat "/sys/fs/cgroup${group}/cgroup.procs")" ]] || return 1
        fi
    done
    # 수동 uvicorn 등 관리 밖 프로세스도 같은 포트를 점유하면 기동하지 않는다.
    listeners="$(ss -H -ltn 'sport = :8000 or sport = :8001')" || return 1
    [[ -z "$listeners" ]] || return 1
    # flock은 서비스 프로세스 수명 전체 동안 유지된다.
    flock -n /run/lock/ai-agent-rag.lock true || return 1
    flock -n /run/lock/ai-agent-orchestrator.lock true || return 1
    if command -v nvidia-smi >/dev/null && [[ "${#old_pids[@]}" -gt 0 ]]; then
        for ((i=0; i<20; i++)); do
            gpu="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" || return 1
            local remaining=0
            for pid in "${old_pids[@]}"; do
                if grep -qxF "$pid" <<< "$gpu"; then remaining=1; fi
            done
            [[ "$remaining" == 0 ]] && return 0
            sleep 1
        done
        return 1
    fi
    return 0
}

configure() {
    local version="$1" root unit host port lock
    root="$BASELINE"
    [[ "$version" == v1 ]] || root="$V2_ROOT"
    [[ -f "$root/deploy/gpu_server/.env" ]] || return 1
    for unit in rag_service orchestrator; do
        host=127.0.0.1; port=8001; lock=rag
        if [[ "$unit" == orchestrator ]]; then host=0.0.0.0; port=8000; lock=orchestrator; fi
        mkdir -p "/etc/systemd/system/${unit}.service.d" || return 1
        cat > "/etc/systemd/system/${unit}.service.d/90-active-implementation.conf" <<EOF || return 1
[Service]
WorkingDirectory=$root/$unit
EnvironmentFile=
EnvironmentFile=$root/deploy/gpu_server/.env
ExecStart=
ExecStart=/usr/bin/flock --nonblock --no-fork /run/lock/ai-agent-${lock}.lock $UVICORN main:app --host $host --port $port --log-config log_config.json
KillMode=control-group
EOF
    done
    systemctl daemon-reload || return 1
}

wait_ready() {
    local url="$1" tries="$2" i body
    for ((i=0; i<tries; i++)); do
        if body="$(curl -fsS --max-time 5 "$url")"; then
            if [[ "$url" != http://127.0.0.1:8001/health ]] ||
               "$PYTHON" -c 'import json,sys; d=json.load(sys.stdin); ok=d.get("status")=="ok" if d.get("implementation")=="v2" else d.get("retriever",{}).get("status")=="ok" and d.get("generator",{}).get("status")=="ok" and d.get("generator",{}).get("model_available",False); sys.exit(0 if ok else 1)' <<< "$body"; then
                return 0
            fi
        fi
        sleep 2
    done
    return 1
}

start_app() {
    # LLM은 기존 단일 vllm_llm 서비스 공유. 새 모델 서버를 만들지 않는다.
    systemctl start vllm_llm.service || return 1
    wait_ready http://127.0.0.1:9000/v1/models 300 || return 1
    systemctl start rag_service.service || return 1
    wait_ready http://127.0.0.1:8001/health 120 || return 1
    systemctl start orchestrator.service || return 1
    wait_ready http://127.0.0.1:8000/health 30 || return 1
}

rollback() {
    trap - ERR INT TERM
    echo "Switch failed; restoring $PREVIOUS" >&2
    # 중지가 실패하면 겹쳐 올리지 않는다. 두 서비스 중지 상태로 수동 점검한다.
    if stop_app && configure "$PREVIOUS" && start_app; then
        printf '%s\n' "$PREVIOUS" > "$STATE/active-implementation"
        echo "Restored $PREVIOUS" >&2
    else
        systemctl stop orchestrator.service rag_service.service || true
        echo 'Rollback failed; app services stopped. Inspect journalctl.' >&2
    fi
    exit 1
}

echo "Stopping current app before activating $TARGET"
stop_app || { echo 'Shutdown not confirmed; no new service started' >&2; exit 1; }
trap rollback ERR INT TERM
if [[ "$TARGET" == v2 ]]; then
    "$PYTHON" "$V2_ROOT/indexing/snapshot.py" --baseline "$BASELINE" --target "$V2_ROOT"
    "$PYTHON" "$V2_ROOT/deploy/gpu_server/prepare_env.py" "$BASELINE" "$V2_ROOT"
fi
configure "$TARGET"
start_app
printf '%s\n' "$TARGET" > "$STATE/active-implementation"
trap - ERR INT TERM
echo "Active implementation: $TARGET; client URL and ports unchanged"
