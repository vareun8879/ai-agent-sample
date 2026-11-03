"""설정 모음 — rag_service의 모든 환경변수와 캘리브레이션 값.

운영자가 배포 환경에서 조정할 수 있는 값은 전부 이 파일에 있다. 각 값에는
"왜 이 값인가"의 실측 근거를 주석으로 남겨 뒀으니, 조정 전에 반드시 읽을 것.

주의: 여기의 검색 관련 값(임베딩 모델·컬렉션·접두사)은 인덱싱 파이프라인
(../../indexing/)과 **반드시 동일**해야 한다. 값이 다르면 벡터 공간 자체가
달라져서 "가까움" 비교가 무의미해진다. 임베딩 모델을 바꾸면
"모델 교체 + 전체 재인덱싱 + MAX_RETRIEVAL_DISTANCE 골든셋 재측정"이 한 세트다.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# [1단계 검색] 인덱싱 파이프라인과 반드시 일치해야 하는 값들
# ──────────────────────────────────────────────────────────────

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "schoolpass_docs")
# bge-m3는 질의 접두사가 필요 없다 (e5 계열은 질문 "query: "/문서 "passage: " 필수 —
# 임베딩 모델을 되돌리면 이 접두사와 indexing/embed_chunks.py를 함께 복원할 것)
QUERY_PREFIX = ""

# 기본 Chroma 경로: 리포지토리의 indexing/chroma_db (store_chroma.py가 생성)
DEFAULT_CHROMA_DIR = Path(__file__).resolve().parents[2] / "indexing" / "chroma_db"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(DEFAULT_CHROMA_DIR)))

# 임베딩 계산 디바이스. 기본 "cpu" 고정 — CUDA torch가 깔린 8GB GPU 환경에서 자동 감지에
# 맡기면 임베딩(+1.2GB)까지 GPU로 올라가 Ollama(≈5.7GB)+reranker(2.3GB)와 합쳐 OOM이 난다.
# 질의 임베딩은 CPU로도 수백 ms라 GPU가 필요 없다. VRAM이 넉넉한 서버에서만 "cuda"로.
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu") or None

# 6: 큰 표·목록이 여러 페이지에 걸칠 때 이어지는 페이지(예: 업무조직 표 p27~28의
# 뒷장이 6위로 검색됨)까지 근거에 들어오도록 4에서 상향
DEFAULT_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "6"))

# 이 코사인 거리보다 먼 조각은 "관련 문서"로 취급하지 않는다.
# bge-m3 실측(2026-07-23, 학교 문서 3종 코퍼스 골든셋):
#   관련 질문(부서 업무·교육목표·감염병 교육 등) 0.32~0.50
#   무관 질문(요리·주식·날씨·IT) 0.58~0.66
# → 갭 중간인 0.50을 기본값으로 한다. (참고: 이전 e5-small은 관련/무관 거리가 겹쳐
#   분리 자체가 불가능했다 — bge-m3 승격의 주된 이유)
# 문서가 크게 늘거나 임베딩 모델을 바꾸면 골든셋으로 반드시 재측정할 것.
MAX_RETRIEVAL_DISTANCE = float(os.getenv("MAX_RETRIEVAL_DISTANCE", "0.5"))

# ──────────────────────────────────────────────────────────────
# [1.2단계 재순위화] cross-encoder reranker (reranker.py)
# ──────────────────────────────────────────────────────────────

# 1: 임베딩 검색 후보를 cross-encoder로 재채점. 0: 임베딩 거리만 사용(MAX_RETRIEVAL_DISTANCE 필터).
#
# 효과(실측 2026-07-24): "생활안전부 상담 담당자는 누구야?"에서 임베딩 단독으로는
# 조직표(p27)가 후보에도 못 들었으나, reranker가 2위(0.595)로 끌어올려 담당자 이름이
# 답변에 정확히 나옴. 관련/무관 분리도 임베딩(0.497 vs 0.5 임계, 여유 0.003)보다
# 훨씬 안전(0.599 vs 0.503, 임계 0.55).
#
# 비용(실측): CPU 재채점은 질의당 약 30초로 부적합. GPU(RTX 3070 Ti)에서는 1.7~2.0초로
# 실측 확인되어 기본 켜짐. 전제: CUDA torch 설치 + 평시 vLLM(:10001) 내려두기
# (VRAM 실측: reranker 2.3GB + Ollama ax4-light ≈ 7.98/8GB — vLLM과는 공존 불가).
RERANK = os.getenv("RERANK", "1") == "1"
# bge-m3(검색)와 짝을 이루는 공식 reranker — 다국어(한국어 포함)
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
# 임베딩 검색에서 reranker에 넘길 후보 수 — top_k(6)보다 넓게 잡아야
# 임베딩이 놓친 정답(예: 조직표 조각이 임베딩 8위)이 재채점 기회를 얻는다.
# 20 → 30: 골든 실측에서 정답 페이지가 후보 20개 밖이던 문항(달력표 p49)이 있었다. GPU 재채점 비용은 무시할 수준.
# 50 (09-18): 산문 조각(prose_facts)이 더해져 문서당 조각이 2,100 → 2,500 이 되자 30개 후보 안에 정답 행이 못 들어가는
# 문항이 생겼다("가정통신은 어떤 앱?" 이 검색 0건). reranker 는 짧은 조각 50개도 수십 ms 라 비용은 없다.
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "50"))
# 낱말 일치 후보 수 (하이브리드 검색). 임베딩 상위 RERANK_CANDIDATES 에 더해, 질문 낱말(한글 2자+ 접두·숫자)이 가장 많이 든
# 짧은 조각(kind=fact) 상위 N 개를 reranker 후보에 합친다. 09-18 실측(rerank_probe): "가정통신은 어떤 앱?" 의 정답 행이
# 임베딩 후보 50개 안에 아예 없었고(상위가 전부 0.015 이하), "몇 교시?" 의 "매일 7교시" 조각도 탈락 — 조각이 늘 때마다
# 임베딩 순위에 밀리는 조각이 생긴다. 낱말로는 들어오므로 reranker 가 판단할 기회를 준다. 컷(임계·상대 하한)은 그대로라
# 오탐이 늘지 않는다. 0 이면 끔.
LEXICAL_CANDIDATES = int(os.getenv("LEXICAL_CANDIDATES", "20"))
# 이 확률(0~1, CrossEncoder 의 sigmoid 출력) 미만은 무관으로 차단.
#
# 주의 — 2026-09-17 이전 값 0.55 는 이중 sigmoid 버그로 뭉개진 척도(0.500~0.731) 기준이었다
# (reranker.py 참고). 실제 확률로는 0.2 였고, 그 때문에 답이 있는 표 페이지 질문 12건이 "찾을 수
# 없음" 으로 떨어졌다. 아래는 버그 제거 후 실제 확률로 다시 잰 값(서버 L4, 망미중 골든 29문항):
#   무관 질문(주식·요리) 후보 최고     : 0.000 ~ 0.012
#   관련 질문 정답 페이지 (통과하던 것)  : 0.245 ~ 0.99
#   관련 질문 정답 페이지 (막히던 것)    : 0.096 ~ 0.164 (7건)  /  0.008 ~ 0.024 (5건 — 통계표·조직도·배치도)
# → 0.05: 무관(≤0.012)과 관련 하위군(≥0.096) 사이.
# 표 행 조각(indexing/chunk_documents.table_facts) 도입 + 제목 접두 제거 후 서버 재측정(09-17 저녁, 후보 30):
#   무관 후보 최고 0.006 / 관련 정답 최저 0.016 (p25 0.239, median 0.668) / 학생수·학급수 행 조각 0.036~0.042
# → 0.03: 관련 25/27 통과, 무관 2/2 차단. 0.02 면 26/27 이지만 무관 표본이 2건뿐이라 여유(5배)를 남긴다.
#   골든셋에 무관 문항을 늘린 뒤 0.02 를 재검토할 것.
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.03"))
# 낱말 구제 하한: 임계를 아무 후보도 못 넘겼는데, 질문 낱말이 **낱말 그대로** 든 낱말 일치 후보가 있으면 이 점수 이상일 때
# 그 하나를 살린다. 실측(09-21 키오스크): "망미중학교 연혁은 어떻게 되지?" — 연혁 표 행 조각("주요 연혁 — 1982.03.01. …")이
# 낱말 일치로 후보에 들었지만 리랭커가 짧은 명사형 질문에 0.03 미만을 줘 전부 버려지고 "찾을 수 없다" 로 끝났다
# ("주요 연혁은?" 은 통과). 0 이면 끈다. 통과한 페이지가 이미 있으면 이 하한 없이 낱말 일치 행 하나를 더한다(reranker 주석).
RERANK_LEXICAL_RESCUE = float(os.getenv("RERANK_LEXICAL_RESCUE", "0.003"))  # 09-21: 0.01 이면 "개교는 언제지?" 의 연혁 행이 탈락 — 낱말 그대로 일치가 조건이라 무관 질문 오탐은 작다
# 인접 페이지 구제 여유: 선택된 페이지의 다음 페이지 후보는 임계에서 이만큼 낮아도 함께 살린다 —
# 페이지에 걸친 표의 뒷장이 아깝게 탈락해 반쪽 답변이 재발하는 것 방지. 실제 확률 척도에 맞춰 0.1 → 0.03.
RERANK_RESCUE_MARGIN = float(os.getenv("RERANK_RESCUE_MARGIN", "0.03"))
# [3번 컨텍스트 다이어트] 상대 하한: 1위 점수 × 이 비율 미만인 후보는 절대 임계를 넘어도 버린다.
# 실측(09-17, 표 행 조각 도입 후 골든 06): 검색은 27/29 맞는데 답이 틀리는 문항이 생겼다 —
#   보건실 몇 층? : p123 배치도 0.124 가 정답인데 p10 교실 수 표 0.039 가 딸려 들어가 "2층"
#   입학식 언제? : p49 달력이 정답인데 p50 (졸업식 2월 3일) 이 딸려 들어가 "2월 3일 화요일"
# 7B 모델은 관련 낮은 페이지가 같이 있으면 섞어 읽는다. 절대 임계(0.03)는 "무관 차단"용이고, 이 비율은
# "답이 있는 질문에서 곁다리 페이지 제거"용이다. 프로브 데이터로 비율을 정했다(1위 대비 2위 이하 비율):
#   유지해야 하는 것: 학생수 0.78·0.67 (p9 행 조각들), 학적 담당 0.46 (p29 위원회), 학급수 1.0
#   잘라야 하는 것: 보건실 0.31 (p10 교실 수 표), 도서실 0.32 (p10), 시간표 0.09 (p58)
# → 0.4 (0.3 이면 보건실 0.31 이 살아남는다). 인접 페이지 구제(RESCUE)는 이 비율과 별개로 유지된다.
# 개교 질문의 달력 페이지(0.63)처럼 reranker 자체가 높게 준 곁다리는 못 자른다 — 그건 모델 몫. 0 이면 끔.
RERANK_RELATIVE_MIN = float(os.getenv("RERANK_RELATIVE_MIN", "0.4"))
# reranker 전용 디바이스 — 임베딩(EMBED_DEVICE)과 분리한 이유: 8GB GPU에서 Ollama(LLM)와
# 공존하려면 VRAM을 아껴야 한다. reranker(fp16 ~1.4GB)만 GPU에 올리고 임베딩은 CPU 유지가
# 안전한 구성이다 (질의 임베딩은 CPU로도 수백 ms). 비우면 자동 감지.
# 주의: GPU reranker 운영 시 vLLM(:10001, 6.8GB 예약)은 내려둬야 한다 — 인덱싱 때만 띄울 것.
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "") or None

# GPU에 올리는 모델(bge-m3 임베딩, reranker)을 fp16으로 로드한다. fp32 대비 VRAM 절반
# (각 2.3GB → 1.2GB), 속도 약 2배, 검색 품질 차이는 무시할 수준(bge 계열 공식 권장 dtype).
# CPU에서는 자동으로 무시된다 — CPU fp16 연산은 느리거나 미지원. 서버(L4 24GB) 배포 기본 1.
MODEL_FP16 = os.getenv("MODEL_FP16", "1") == "1"


def fp16_model_kwargs(device):
    """sentence-transformers 로더에 넘길 model_kwargs. GPU + MODEL_FP16 일 때만 fp16."""
    if not MODEL_FP16:
        return {}
    try:
        import torch
    except ImportError:  # torch 없이는 GPU도 없다
        return {}
    on_cuda = device.startswith("cuda") if device else torch.cuda.is_available()
    return {"model_kwargs": {"torch_dtype": torch.float16}} if on_cuda else {}

# ──────────────────────────────────────────────────────────────
# [1.5단계 컨텍스트 확장] small-to-big 페이지 병합 (expansion.py)
# ──────────────────────────────────────────────────────────────

# 검색은 조각 단위(임베딩 정확도)로 하고, LLM에 주는 컨텍스트는 그 조각이 속한
# 페이지 전체(내용 완전성)로 넓힌다. 0으로 끄면 조각 그대로 전달.
PAGE_EXPAND = os.getenv("PAGE_EXPAND", "1") == "1"

# 페이지 병합 컨텍스트의 총량 상한(문자). LLM 컨텍스트 창(LLM_NUM_CTX) 예산 안에
# 들어가야 한다 — 10,000자 ≈ 한국어 7,500토큰 수준 (num_ctx 16384 기준).
# 9,000이었을 때 두 페이지짜리 표(p27+28 = 9,027자)가 27자 차이로 잘린 사례가 있어 상향.
CONTEXT_CHAR_BUDGET = int(os.getenv("CONTEXT_CHAR_BUDGET", "10000"))

# ──────────────────────────────────────────────────────────────
# [2단계 생성] 답변 LLM 호출 (generator.py) — Ollama(개발 PC) / vLLM(운영 서버)
# ──────────────────────────────────────────────────────────────

# 답변 LLM 서빙 백엔드.
#   ollama: 개발 PC(8GB, Windows)·소규모·폴백. Ollama /api/chat.
#   vllm:   운영 서버(L4). 납품 키오스크 40대+가 서버 하나를 같이 쓰므로 동시 요청을
#           배치로 묶는 continuous batching 이 필요하다. vLLM OpenAI 호환 /v1/chat/completions.
# 두 백엔드는 같은 모델(A.X-4.0-Light)·같은 프롬프트를 쓰고 호출 형식만 다르다 (generator.py).
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
# vLLM OpenAI 호환 API 베이스(…/v1). 서버는 --served-model-name 을 LLM_MODEL 과 같게 띄운다.
VLLM_LLM_URL = os.getenv("VLLM_LLM_URL", "http://127.0.0.1:9000/v1")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# Ollama에 올라간 모델 태그. A.X-4.0-Light를 미리 `ollama pull`/`ollama create` 해두고
# 그 태그를 여기(또는 LLM_MODEL 환경변수)에 맞춰야 한다.
LLM_MODEL = os.getenv("LLM_MODEL", "ax4-light")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300"))

# 0: 문서 내용을 그대로 옮기는 키오스크 용도라 창의성이 불필요하고, 긴 표 나열의
# 회차별 편차(어떤 실행에서는 특정 부서가 빠지는 문제)를 없애기 위해 탐욕 디코딩을 쓴다
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# Ollama의 컨텍스트 기본값(2048토큰)은 페이지 병합 컨텍스트(최대 10,000자 ≈ 7,500토큰)가
# 조용히 잘려 나간다 — 반드시 명시한다. 16384 = 컨텍스트 예산 + 질문 + 출력(4096)까지 수용
# (A.X-4.0-Light의 최대 컨텍스트 16,384와 일치).
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "16384"))

# 표 전체 설명처럼 긴 답변이 중간에 끊기지 않도록 출력 상한도 명시.
# 2048에서는 두 페이지짜리 표 답변이 앞 페이지만 쓰고 끝나는 사례가 있어 4096으로 상향.
# 주의: 이 값을 더 올려도 "긴 나열이 중간에 끊기는" 문제는 해결되지 않는다 —
# 실측에서 4096/8192/16000 어느 한도든 모델은 같은 지점(예: 1,488토큰)에서 스스로
# 멈췄다(done_reason=stop). 그 문제는 자동 이어쓰기(continuation.py)가 담당한다.
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "4096"))

# 답변 모드 (요청의 answer_mode 가 우선, 없으면 이 기본값).
#   voice: 음성 안내용 — 첫 문장에 핵심, 전체 3문장 이내, 목록은 3항목까지, 이어쓰기 없음, 출력 상한 LLM_NUM_PREDICT_VOICE.
#          키오스크는 답을 TTS 로 읽어 주므로 45초짜리 조직표 낭독(실측 1,500토큰)은 쓸모가 없다 — 실측: 같은 질문이 45.6초 → 수 초.
#   full : 표·목록을 빠짐없이 나열 + 자동 이어쓰기 (개발·화면 확인용. 골든셋의 '완전 나열' 검증도 이 모드로).
LLM_ANSWER_MODE = os.getenv("LLM_ANSWER_MODE", "voice")
LLM_NUM_PREDICT_VOICE = int(os.getenv("LLM_NUM_PREDICT_VOICE", "512"))

# ──────────────────────────────────────────────────────────────
# [2.5단계 검증] 답변의 숫자가 근거에 있는지 대조 (verify.py)
# ──────────────────────────────────────────────────────────────
# 1: 답변의 3자리 이상 숫자(연도·날짜·인원·금액)가 근거 텍스트에 없으면 답변을 막고 확인 안내문을 낸다.
# 실측(09-17): 근거에 '1982.03.01. 개교' 가 있는데 모델이 매번 "1980년" 이라 답함 — 프롬프트만으로는 못 막는다.
VERIFY_NUMBERS = os.getenv("VERIFY_NUMBERS", "1") == "1"
# 검증 실패 시 재생성 횟수: 근거에 없는 숫자를 지적하는 메시지를 붙여 한 번 더 답하게 한다.
# 실측(09-17): "언제 개교?" 에 1980(근거 1982), "학생 수?" 에 2555(근거 255(5)) — 같은 근거로 다시 물으면
# 모델이 맞게 고쳐 쓰는 경우가 많다(1980/1982 는 호출마다 흔들리던 값). 실패할 때만 LLM 1회 추가(1~2초).
# 재생성도 실패하면 확인 안내문(UNVERIFIED_ANSWER)이 나간다. 0 이면 끔.
VERIFY_RETRY = int(os.getenv("VERIFY_RETRY", "1"))
# 인용 우선 생성(voice 모드): 모델이 답하기 전에 '근거:' 줄에 [문서] 구절을 그대로 복사하고 '답변:' 줄에 답한다.
# 실측(09-17, A.X-4.0-Light 7B): 인용은 정확한데 옮겨 쓰는 단계에서 틀린다 — "근거: 1982.03.01. 개교 / 답변: 1980년",
# "근거: 계 255(5) / 답변: 2,555". 그래서 (1) 답변의 날짜는 인용 구절에 있어야 통과시키고(verify.unsupported_dates),
# (2) 재생성까지 실패하면 모델 문장 대신 인용 구절을 그대로 읽어 준다(QUOTE_ANSWER_PREFIX) — 인용이 [문서]에
# 실제로 있는 구절일 때만. 1 이면 켬. 스트리밍은 '답변:' 이후만 화면에 흘린다.
QUOTE_FIRST = os.getenv("QUOTE_FIRST", "1") == "1"
QUOTE_ANSWER_PREFIX = os.getenv("QUOTE_ANSWER_PREFIX", "자료에는 다음과 같이 적혀 있습니다. ")
# 핵심 근거: 검색에서 고른 표 행·도해 설명 조각(kind=fact)을 페이지 전문 앞에 따로 보여준다 — 최대 개수.
# 실측(09-17): 행 조각으로 정답 줄을 찾아 놓고도 LLM 에는 2,700자 달력 페이지만 주니 7B 가 표 속에서 헤맸다
# ("입학식은?" → 졸업식 날짜와 섞어 "2월 3일 화요일"). 0 이면 끔.
HIGHLIGHT_MAX = int(os.getenv("HIGHLIGHT_MAX", "5"))
# 1: 고른 페이지의 표 행 조각 중 질문 낱말이 (검색된 행보다) 확실히 더 겹치는 줄을 [핵심 근거] 맨 앞에 보탠다.
# 실측(09-18): 정답 행이 색인은 됐지만 reranker 상위에 못 들어 근거에 없던 문항(예산 총액·동아리 배정인원) 대응.
# 같은 점수의 이웃 행까지 넣으면 오히려 오답 후보가 늘어 30문항 세트가 흔들렸으므로 '확실히 높은 줄만'. 0 이면 끔.
HIGHLIGHT_FILL = os.getenv("HIGHLIGHT_FILL", "1") == "1"
# 실측(2026-10-30, 261030_6_log): 낱말 점수 보강이 무관한 수치 행을 우선함.
# 기존 리랭커로 복합 질문별 사실 청크를 평가한다. log/on/off; 기본은 비교 로그만.
COMPOUND_HIGHLIGHT_MODE = os.getenv("COMPOUND_HIGHLIGHT_MODE", "log").lower()
# 과도한 추가 추론을 피한다. 한도를 넘으면 일부를 임의로 잘라 평가하지 않고 보강을 생략한다.
COMPOUND_HIGHLIGHT_MAX_FACTS = int(os.getenv("COMPOUND_HIGHLIGHT_MAX_FACTS", "200"))
# 사실 행의 대상·값 대조. retry는 두 항목의 근거가 확정될 때만 재생성 1회 후 항목별 검증.
# off/log/retry; 기본 log. 재생성이 나아지지 않으면 원래 답변을 유지한다.
ROW_VALUE_AUDIT_MODE = os.getenv("ROW_VALUE_AUDIT_MODE", "log").lower()
# 1 이면 LLM 에 보내는 프롬프트 전문을 INFO 로그에 남긴다 (골든 실패 원인 분석용 — 운영에서는 0)
LOG_PROMPT = os.getenv("LOG_PROMPT", "0") == "1"

# ──────────────────────────────────────────────────────────────
# [2단계 보정] 자동 이어쓰기 (continuation.py)
# ──────────────────────────────────────────────────────────────

# 7B급 모델은 긴 나열(조직표 전체 등)을 출력 한도에 닿기 전에 스스로 끝내버린다
# (실측: 731/4096토큰에서 done_reason=stop, 프롬프트 강제도 무효). 그래서 1차 답변이
# 나열형이고 출력이 MIN_TOKENS 이상이면 "이어서 써라"를 최대 ROUNDS회 자동 요청한다.
#
# 발동 조건은 "1차 답변의 실제 출력 토큰 수"만 쓴다. 예전에 "근거 컨텍스트가 크면"도
# OR 조건으로 발동시켰으나 역효과가 실측돼 제거했다: 1차 답변이 283토큰으로 완결됐는데
# 느슨하게 관련된 페이지 5개가 병합되어 트리거가 발동했고, 이을 내용이 없으니 모델이
# 직전 대화 이력에서 다른 주제 내용을 끌어와 지어냈다. 컨텍스트가 큰 건 "느슨한 페이지가
# 많이 검색됐다"는 뜻이지 "답변이 길어야 한다"는 뜻이 아니다.
LLM_CONTINUE_ROUNDS = int(os.getenv("LLM_CONTINUE_ROUNDS", "3"))
LLM_CONTINUE_MIN_TOKENS = int(os.getenv("LLM_CONTINUE_MIN_TOKENS", "400"))

# ──────────────────────────────────────────────────────────────
# [서비스] API 계층 (main.py)
# ──────────────────────────────────────────────────────────────

# 검색 결과가 전부 임계값에 걸러졌을 때(= 매뉴얼과 무관한 질문) LLM을 태우지 않고
# 내보내는 고정 답변. LLM 프롬프트의 "근거 없음" 문구와 앞부분("제공된 자료에서")이
# 일치해야 continuation.py의 안내문 혼입 필터가 함께 동작한다.
NO_CONTEXT_ANSWER = os.getenv(
    "NO_CONTEXT_ANSWER",
    "죄송합니다. 제공된 자료에서 관련 내용을 찾을 수 없습니다. 안내데스크에 문의해 주세요.",
)
