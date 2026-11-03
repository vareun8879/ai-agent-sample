#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then echo 'Run with bash, not source' >&2; return 2; fi
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${V2_PYTHON:-/root/ai-agent/app-env/bin/python}"
set -a
source "$ROOT/deploy/gpu_server/.env"
set +a
RUNS=3
if [[ "${1:-}" == -n ]]; then RUNS="$2"; shift 2; fi
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid run count'; exit 2; }
SETS=("$@")
[[ "${#SETS[@]}" -gt 0 ]] || SETS=(mangmi_2026 aica_2026 paichai_2026)
for i in {1..120}; do
    if curl -fsS --max-time 5 http://127.0.0.1:8001/health >/dev/null; then break; fi
    sleep 2
done
curl -fsS --max-time 5 http://127.0.0.1:8001/health | "$PYTHON" -c 'import json,sys; d=json.load(sys.stdin); print(d); sys.exit(0 if d.get("implementation")=="v2" else "v2 is not active")'
cd "$ROOT/rag_service"
STAMP="$(date '+%Y%m%d_%H%M%S')"
REPORTS="$ROOT/reports/$STAMP"
mkdir -p "$REPORTS"
START="$(date '+%Y-%m-%d %H:%M:%S')"
TRACE_FILE="${V2_TRACE_PATH:-$ROOT/rag_service/logs/trace.jsonl}"
TRACE_START=0
if [[ -f "$TRACE_FILE" ]]; then TRACE_START="$(wc -l < "$TRACE_FILE")"; fi
for name in "${SETS[@]}"; do
    [[ "$name" =~ ^[A-Za-z0-9_-]+$ && -f "golden/$name.jsonl" ]] || { echo "Invalid set: $name"; exit 2; }
    for ((run=1; run<=RUNS; run++)); do
        "$PYTHON" golden_eval.py "golden/$name.jsonl" --answer --fail-only --url http://127.0.0.1:8001 \
            --report "$REPORTS/last_report_${name}_${run}.json" || true
    done
done
journalctl -u rag_service --since "$START" --all --no-pager > "$REPORTS/rag_service.log"
if [[ -f "$TRACE_FILE" ]]; then
    tail -n "+$((TRACE_START + 1))" "$TRACE_FILE" > "$REPORTS/trace.jsonl"
else
    : > "$REPORTS/trace.jsonl"
fi
"$PYTHON" -m evaluation.regrade --fail-only "$REPORTS"/last_report_*.json
echo "Reports: $REPORTS"
