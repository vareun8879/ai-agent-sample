[Unit]
Description=RAG QA service (retrieval + vLLM generation) :8001
# 답변 LLM(vllm_llm)은 06단계가 먼저 띄운다. Wants 로 묶지 않는다 — 인덱싱 중엔 vllm_llm 을 내린 채
# rag_service 만 따로 재기동할 수 있어야 한다.
After=network-online.target vllm_llm.service
Wants=network-online.target

[Service]
Type=simple
User=__USER__
WorkingDirectory=__APP_ROOT__/rag_service
EnvironmentFile=__ENV_FILE__
Environment=PYTHONUNBUFFERED=1
ExecStart=__APP_ROOT__/app-env/bin/uvicorn main:app --host 127.0.0.1 --port 8001 --log-config log_config.json
Restart=always
RestartSec=5
# 임베딩·reranker 로딩에 시간이 걸린다 — 기동 실패 판정을 서두르지 않게
TimeoutStartSec=300
KillSignal=SIGINT
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
