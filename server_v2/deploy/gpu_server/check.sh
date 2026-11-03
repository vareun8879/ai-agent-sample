#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then echo 'Run with bash, not source' >&2; return 2; fi
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${1:-/root/ai-agent/app-env/bin/python}"
"$PYTHON" -c 'import ast,pathlib,sys; root=pathlib.Path(sys.argv[1]); files=list(root.rglob("*.py")); [ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p)) for p in files]; print("syntax OK:",len(files),"files")' "$ROOT"
for script in "$ROOT"/deploy/gpu_server/*.sh; do bash -n "$script"; done
cd "$ROOT/rag_service"
"$PYTHON" -m unittest discover -s tests -v
