[Unit]
Description=vLLM serving vision model for document indexing :10001  [indexing only]
After=network-online.target

[Service]
Type=simple
User=__USER__
WorkingDirectory=__APP_ROOT__
EnvironmentFile=__ENV_FILE__
# vllm-env/bin 을 PATH 앞에 — vLLM 이 커널 컴파일에 ninja 등 venv 안의 실행 파일을 이름으로 부른다 (없으면 'No such file: ninja' 로 EngineCore 가 죽는다)
Environment=PATH=__APP_ROOT__/vllm-env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# 인덱싱은 점검 시간에 답변 LLM(vllm_llm)·rag_service 를 내리고 돌리므로 24GB 를 VLM 이 거의 다 쓴다.
# VLM_GPU_UTIL=0.85 ≈ 20GB = Qwen3-VL-8B FP8 가중치 ≈ 9GB + 32k 컨텍스트(250DPI A4 ≈ 6천 이미지 토큰 + 출력 4k) KV.
# 인덱싱은 페이지를 순차 처리하므로 동시 시퀀스는 4 면 충분하다.
ExecStart=__APP_ROOT__/vllm-env/bin/vllm serve ${VLM_MODEL} --host 127.0.0.1 --port 10001 --gpu-memory-utilization ${VLM_GPU_UTIL} --max-model-len ${VLM_MAX_MODEL_LEN} --max-num-seqs 4
Restart=no
# 첫 기동은 모델 다운로드(≈9GB) + CUDA 그래프 캡처로 오래 걸린다
TimeoutStartSec=1800
KillSignal=SIGINT
TimeoutStopSec=60

# 상시 기동하지 않는다 — 05_index_documents.sh --with-vlm 이 필요할 때만 start/stop 한다
[Install]
WantedBy=
