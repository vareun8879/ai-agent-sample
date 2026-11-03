[Unit]
Description=vLLM serving answer LLM (A.X-4.0-Light, OpenAI-compatible) :9000
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=__USER__
WorkingDirectory=__APP_ROOT__
EnvironmentFile=__ENV_FILE__
Environment=PYTHONUNBUFFERED=1
# vllm-env/bin 을 PATH 앞에 — vLLM 이 커널 컴파일에 ninja 등 venv 안의 실행 파일을 이름으로 부른다 (없으면 'No such file: ninja' 로 EngineCore 가 죽는다)
Environment=PATH=__APP_ROOT__/vllm-env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# 가중치: 04단계가 받아 둔 HF 원본 디렉터리(LLM_HF_DIR, safetensors). VLLM_LLM_QUANT=fp8 이면
# 로드 시 FP8 로 동적 양자화한다(L4 = Ada, SM 8.9, FP8 지원) — 약 7.5GB. GGUF 변환이 필요 없다.
# --gpu-memory-utilization 0.70 ≈ 16.8GB = 가중치 7.5 + KV 캐시 ≈ 9GB(16k 컨텍스트 × 동시 8~16건).
# 나머지 ≈ 7GB 는 rag_service 의 bge-m3·reranker(2.6GB) 몫. 인덱싱용 vllm_vlm 과는 동시에 못 뜬다.
# --served-model-name 은 rag_service 의 LLM_MODEL 과 같아야 한다 (/health 가 이 이름을 찾는다).
ExecStart=__APP_ROOT__/vllm-env/bin/vllm serve ${LLM_HF_DIR} --served-model-name ${LLM_MODEL} --host 127.0.0.1 --port 9000 --quantization ${VLLM_LLM_QUANT} --dtype auto --max-model-len ${LLM_NUM_CTX} --gpu-memory-utilization ${VLLM_LLM_GPU_UTIL} --max-num-seqs ${VLLM_LLM_MAX_SEQS} --enable-prefix-caching
Restart=always
RestartSec=10
# FP8 변환 · CUDA 그래프 캡처로 첫 기동이 5~10분 걸린다
TimeoutStartSec=1800
KillSignal=SIGINT
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
