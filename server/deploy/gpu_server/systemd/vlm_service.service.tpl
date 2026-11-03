[Unit]
Description=VLM service (image -> text; transformers in-process or via vLLM) :10000  [indexing only]
# VLM_BACKEND=vllm 일 때만 vllm_vlm 이 먼저 떠야 한다 — Requires 로 묶으면 transformers 백엔드에서도
# vllm_vlm 이 같이 떠서 VRAM 을 잡아먹으므로 After 만 둔다. 05단계가 백엔드에 맞춰 순서대로 띄운다.
After=vllm_vlm.service

[Service]
Type=simple
User=__USER__
WorkingDirectory=__APP_ROOT__/vlm_service
EnvironmentFile=__ENV_FILE__
Environment=PYTHONUNBUFFERED=1
ExecStart=__APP_ROOT__/app-env/bin/uvicorn main:app --host 127.0.0.1 --port 10000 --log-config log_config.json
Restart=no
# transformers 백엔드는 기동 시 모델(bf16 약 16GB)을 로드한다
TimeoutStartSec=1800
KillSignal=SIGINT
TimeoutStopSec=60

# 상시 기동하지 않는다 — 05_index_documents.sh --with-vlm 이 필요할 때만 start/stop 한다
[Install]
WantedBy=
