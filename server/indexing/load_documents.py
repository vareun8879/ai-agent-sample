"""1단계 — PDF 텍스트 추출.

디지털 PDF는 텍스트 레이어에서 그대로 추출하고, 텍스트 레이어가 없는
스캔본(이미지) 페이지만 vlm_service(비전 모델)로 전사한다. 텍스트 레이어는 원본
그 자체라 이미지 렌더링·모델 추론을 거칠 이유가 없고, VLM은 텍스트 레이어가 없는
페이지에만 쓴다 — 표·레이아웃 구조까지 살린 전사를 기대할 수 있다.

텍스트 레이어는 pypdf와 PyMuPDF **둘 다**로 뽑아 페이지마다 덜 깨진 쪽을 쓴다.
두 추출기는 깨지는 방식이 정반대다 — 자간을 글자별로 배치한 표 페이지는 pypdf가
글자마다 공백을 끼우고("부서업 무총괄"), 세로쓰기 셀이 있는 페이지는 PyMuPDF가
한 글자씩 줄바꿈한다("미/술/실"). 망미중 교육계획서 123쪽 실측: pypdf만 쓰면 6쪽,
PyMuPDF만 쓰면 8쪽이 깨지지만 페이지별 선택이면 조직도(도해) 1쪽만 남는다.

백지 페이지(장 구분용 빈 면)는 VLM에 보내지 않는다 — 빈 이미지를 받은 VLM은
"이미지를 제공받지 못했습니다" 같은 사과문을 지어내고, 그것이 그대로 인덱싱되어
검색 결과에 섞이는 사고가 실측으로 확인됐다. 잉크 비율로 백지를 걸러 빈 텍스트로
남기고(청킹 단계에서 조각이 생기지 않는다), 혹시 VLM이 사과문을 돌려주면 그것도 버린다.

상용 운영 원칙 — "조용한 실패 금지":
- 스캔본 페이지를 처리해야 할 수 있는데 VLM이 준비 안 됐으면 시작 자체를 거부한다(exit 1).
- 페이지 전사 실패(재시도 소진)나 파일 추출 실패가 하나라도 있으면 exit 1로 끝나,
  run_all.py가 이후 단계(청킹·임베딩·저장)를 중단한다 — 검색이 안 되는 문서가 섞인
  불완전한 인덱스를 만들어놓고 아무도 모르는 상황을 막는다.
- VLM 없이 텍스트 레이어만으로 돌려보는 개발 환경에서만 VLM_REQUIRED=0으로 완화한다.
"""

import base64
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF — 텍스트 레이어 추출(pypdf와 병행) + 페이지 렌더링(백지 판정, VLM 입력)
import httpx
from pypdf import PdfReader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("load_documents")

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
# 부분 인덱싱: docs/ 바로 밑 폴더 이름(common, <기관ID>)을 쉼표로 — 그 폴더의 문서만 처리하고 나머지 산출물·색인은 그대로 둔다.
# run_all.py --only … / 05_index_documents.sh --only … 가 환경변수로 넘긴다. 비어 있으면 전체(기존 동작).
INDEX_ONLY = [s.strip().strip("/") for s in os.getenv("INDEX_ONLY", "").split(",") if s.strip()]

# 페이지당 추출 글자 수가 이보다 적으면 스캔본(이미지) 페이지로 보고 VLM 전사로 넘어간다
LOW_TEXT_THRESHOLD = int(os.getenv("LOW_TEXT_THRESHOLD", "20"))
# 그림 페이지: 텍스트 레이어가 머리말·쪽 번호 정도(IMAGE_PAGE_TEXT_MAX 자 미만)뿐이고 한 장의 그림이 페이지의
# IMAGE_PAGE_MIN_RATIO 이상을 덮으면 스캔본과 같이 VLM 으로 전사한다. 실측(09-21, 망미중 계획서 p17·18 "부산교육 정책 방향"
# 인포그래픽): 텍스트 83·47자(제목·쪽 번호)에 그림이 63~67% 를 덮는데 스캔본 기준(20자 미만)에 안 걸려 내용이 통째로 빠졌고,
# "부산교육 방향은?" 이 목차 페이지로 답했다. 일반 페이지의 그림은 최대 8% (교목·교화 사진).
IMAGE_PAGE_TEXT_MAX = int(os.getenv("IMAGE_PAGE_TEXT_MAX", "150"))
IMAGE_PAGE_MIN_RATIO = float(os.getenv("IMAGE_PAGE_MIN_RATIO", "0.3"))
# 그림 페이지의 원본 그림 해상도(표시 크기 기준 유효 DPI)가 이 값 미만이면 글자가 뭉개져 어떤 VLM도 정확히 못 읽는다 —
# 로그에 경고를 남겨 문서 제공처에 원본 이미지를 요청할 수 있게 한다. 실측(09-21): 망미중 계획서 p17 그림 721×948px(유효 95DPI)
# 은 정확히 전사됐고, p18 그림 438×639px(유효 59DPI)은 "믿음과 존중" → "인성과 정의" 처럼 비슷한 낱말로 오독됐다.
IMAGE_PAGE_LOW_DPI = int(os.getenv("IMAGE_PAGE_LOW_DPI", "90"))
# 텍스트 레이어가 "있어도" 깨진 경우가 있다 — 글자 사이 자간을 개별 배치한 PDF에서
# 추출기가 글자마다 가짜 공백·줄바꿈을 끼워 넣는 문제("학 교 일 지"). 한글 조각(공백 없는
# 연속 한글) 중 한 글자짜리의 비율이 이 값 이상이면 깨진 레이어로 보고 VLM으로 재추출한다.
# 비율 계산에서 학사일정표의 요일(월·화·수…)처럼 원래 한 글자인 흔한 단어는 뺀다 —
# 안 빼면 정상적인 달력 페이지가 0.55로 깨진 것으로 오판된다.
# 실측 캘리브레이션(두 추출기 중 좋은 쪽 기준): 깨진 표 페이지 0.55~0.79,
# 정상 페이지 0.25 이하, 조직도(상자 도해) 0.36 → 0.45로 분리.
BROKEN_LAYER_RATIO = float(os.getenv("BROKEN_LAYER_RATIO", "0.45"))
_HANGUL_RUN = re.compile(r"[가-힣]+")
_SINGLE_CHAR_WORDS = set("월화수목금토일주년반회및등교시학생")
# 페이지를 저해상도로 렌더링했을 때 잉크(어두운 픽셀) 비율이 이 값(%) 미만이면 백지로 본다.
# 실측: 백지 0.00%, 글자가 있는 페이지 최소 5.3% → 0.2%면 여유 있게 갈린다.
BLANK_PAGE_INK_PCT = float(os.getenv("BLANK_PAGE_INK_PCT", "0.2"))
_BLANK_CHECK_DPI = 40
# VLM이 이미지를 못 읽었을 때 내는 사과문 — 전사 결과로 인정하지 않는다
_VLM_REFUSAL = re.compile(r"죄송|이미지(를|가) (제공|첨부|전달)|제공(받지|되지) (못|않)|볼 수 없|확인할 수 없")
_VLM_REFUSAL_MAX_LEN = 200
# 표 추출: 괘선 좌표로 셀을 잡는 PyMuPDF find_tables()로 표를 마크다운 표로 옮긴다 — 글자는 레이어
# 원문 그대로이면서 행·열 관계가 보존된다(달력형 학사일정표의 날짜↔행사 대응 등). VLM이 필요 없다.
TABLE_MODE = os.getenv("TABLE_MODE", "1") == "1"
# 표 버전 텍스트의 글자 수가 레이어 원문 대비 이 비율 미만이면 표 감지가 내용을 잃은 것 — 레이어를 유지
TABLE_MIN_KEEP_RATIO = float(os.getenv("TABLE_MIN_KEEP_RATIO", "0.85"))
# 도해(조직도) 판정: 3행·3열 이하의 작은 "상자" 표가 이 개수 이상이면 괘선 없는 도해로 본다.
# 상자 안 글자는 표 추출로 들어가지만 상자 사이의 관계(소속·연결선)는 텍스트에 남지 않으므로
# VLM 설명문을 덧붙인다. 실측: 조직도 페이지 상자 8개, 그 외 페이지 최대 1개.
DIAGRAM_MIN_BOXES = int(os.getenv("DIAGRAM_MIN_BOXES", "5"))
# 특정 페이지를 도해로 강제: "파일명(확장자 제외):페이지" 쉼표 구분.
# 예) VLM_DIAGRAM_PAGES="2026학년도 망미중학교 교육계획서(탑재용):26"
VLM_DIAGRAM_PAGES = {s.strip() for s in os.getenv("VLM_DIAGRAM_PAGES", "").split(",") if s.strip()}
# VLM은 이미지가 크면 이미지 토큰이 프롬프트 예산을 다 먹어 답변이 잘릴 수 있다.
# 250 기준 A4 이미지 토큰 약 6,000 — 서버 vLLM --max-model-len 32768(Qwen3-VL-8B)에서 출력 4,096과 공존.
# 대안인 A.X-4.0-VL-Light(transformers)는 모델이 내부에서 리사이즈하므로 DPI 를 과하게 올려도 이득이 없다.
# 150에서는 빽빽한 표의 부서명(병합 셀)을 오독하는 사례가 있어 200 이상을 쓴다. 3B·12k 구성이면 200 으로 낮출 것.
VLM_RENDER_DPI = int(os.getenv("VLM_RENDER_DPI", "250"))
# 렌더링 픽셀 상한 — VLM_RENDER_DPI 는 A4 세로(2,067×2,924 ≈ 604만 px) 기준으로 잡은 값이다. 16:9 와이드 슬라이드
# (실측 09-21 SCHOOL-PASS·ZERO-PASS.pdf, 1440×810pt)는 250DPI 면 5,000×2,813 = 1,400만 px 가 돼 PNG 20MB(413)·
# 이미지 토큰 1.4만 개로 과하다. 이 값을 넘는 페이지는 DPI 를 자동으로 낮춘다 (슬라이드는 글자가 커서 손해 없음).
VLM_RENDER_MAX_PIXELS = int(os.getenv("VLM_RENDER_MAX_PIXELS", str(2100 * 2970)))

# vlm_service 연결 설정
VLM_SERVICE_URL = os.getenv("VLM_SERVICE_URL", "http://localhost:10000")
# 빽빽한 스캔 페이지는 출력이 4,096토큰까지 나오고 8B 모델은 ~35tok/s 라 생성만 2분 가까이 걸린다 —
# 120초면 타임아웃→재시도 3회로 페이지당 6분을 버리므로 넉넉히 둔다 (vlm_service 쪽 VLM_TIMEOUT 과 맞출 것).
VLM_TIMEOUT = float(os.getenv("VLM_TIMEOUT", "360"))
VLM_RETRIES = int(os.getenv("VLM_RETRIES", "3"))
VLM_RETRY_BACKOFF = float(os.getenv("VLM_RETRY_BACKOFF", "2"))  # 대기 = BACKOFF ** 시도횟수 (초)
# 1(기본): VLM 미준비 시 기동 거부. 0: 텍스트 레이어만으로 진행(개발 전용 — 스캔본 페이지는 비게 됨)
VLM_REQUIRED = os.getenv("VLM_REQUIRED", "1") == "1"

# VLM 전사 결과 캐시 — 같은 PDF(내용 해시)·페이지·DPI·프롬프트면 다시 묻지 않고 파일에서 읽는다.
# 이유(실측 09-18): 조각 규칙(chunk_documents)만 바꿔도 재인덱싱마다 --with-vlm 20분(대부분 Qwen3-VL-8B 적재 시간)이
# 들었다. 캐시가 있으면 --with-vlm 없이도 도해·스캔본 페이지의 전사가 그대로 들어가 3분이면 끝난다.
# 키에 프롬프트 해시가 들어 있어 프롬프트를 고치면 자동으로 새로 전사한다. 캐시를 버리려면 폴더를 지우면 된다.
VLM_CACHE = os.getenv("VLM_CACHE", "1") == "1"
VLM_CACHE_DIR = Path(os.getenv("VLM_CACHE_DIR", str(Path(__file__).resolve().parent / "vlm_cache")))

VLM_INTERNAL_KEY = os.getenv("INTERNAL_API_KEY", "")
if not VLM_INTERNAL_KEY and os.getenv("ALLOW_DEV_KEYS") == "1":
    VLM_INTERNAL_KEY = "dev-only-shared-key"

VLM_PROMPT = (
    "이 문서 페이지 이미지의 내용을 한국어로 옮겨 적어라. "
    "표가 있으면 행과 열의 관계를 살려 정리하고, 글자는 최대한 정확하게 그대로 옮겨 적어라. "
    "표에서 세로로 병합된 셀(여러 행에 걸친 부서명·분류명 등)은 생략하지 말고 "
    "해당하는 각 행에 그 값을 반복해서 적어라."
)
# 그림 페이지(인포그래픽·포스터) 전용: 소형 VLM은 흐릿한 글자를 비슷한 낱말로 바꿔 읽는다(위 IMAGE_PAGE_LOW_DPI 실측).
# 지어낸 낱말이 답에 섞이는 것보다 "[판독불가]" 가 낫다 — 답변 LLM 이 그 자리를 "자료에서 판독되지 않음" 으로 처리한다.
VLM_PROMPT_IMAGE = (
    "이 이미지에 적힌 모든 글자를 한국어로 그대로 옮겨 적어라. "
    "제목·소제목·말풍선·상자 안의 문구를 위에서 아래, 왼쪽에서 오른쪽 순서로 한 줄에 하나씩 적고, "
    "같은 상자·같은 열에 속한 문구는 그 묶음의 제목 뒤에 '제목: 문구1 / 문구2' 형태로 적어라. "
    "글자를 바꾸거나 비슷한 말로 고쳐 쓰지 마라. 흐릿하거나 확실하지 않은 글자는 추측하지 말고 그 낱말 자리에 [판독불가] 라고 적어라. "
    "그림의 의미를 해석하거나 설명을 덧붙이지 마라."
)
# 2차 시도(깨진 레이어 페이지 전용): 1차 전사에서 빠진 단어들을 힌트로 주입한다.
# 소형 VLM은 세로쓰기 병합 셀(부서명 등)을 스스로 못 읽지만, 이름을 알려주면
# 병합 범위의 올바른 위치에 배치하는 것이 실측으로 확인됐다.
VLM_PROMPT_RETRY_PREFIX = (
    "이 문서 페이지의 내용을 한국어로 옮겨 적어라. "
    "표가 있으면 마크다운 표로 행과 열을 정확히 살려 옮기고, 어떤 열도 생략하지 마라. "
    "세로쓰기로 적혔거나 여러 행에 걸쳐 병합된 셀 값은 병합 범위의 모든 행에 반복해서 채워라. "
    "참고: 다음 단어들이 이 페이지에 있으나 놓치기 쉽다 — 실제 이미지에서 보이는 것만 정확한 위치에 반영하라: "
)
# 힌트로 주입할 최대 단어 수 — 프롬프트 비대화 방지
VLM_RETRY_MAX_HINTS = int(os.getenv("VLM_RETRY_MAX_HINTS", "40"))
_HANGUL_WORD = re.compile(r"[가-힣]{4,}")
# 도해(조직도·흐름도) 전용: 글자는 표 추출이 이미 정확히 가져오므로 VLM에는 "관계"만 묻는다.
# 출력은 "한 줄에 사실 하나" 로 받는다 — chunk_documents.diagram_facts 가 줄마다 검색용 조각을 만들기 때문.
# 직책과 이름이 한 상자에 있으면 붙여 적게 한다("교장 홍길동") — "교장 선생님 성함이 뭐야?" 가 걸리려면
# 직책과 이름이 같은 줄에 있어야 한다(실측: 조직도 p26 은 레이어 텍스트에서 "교장" 과 "허OO" 가 다른 줄).
VLM_PROMPT_DIAGRAM = (
    "이 문서 페이지에 있는 도해(조직도·흐름도)의 내용을 한국어로 설명하라. "
    "한 줄에 사실 하나씩만 적고, 글머리표나 번호 없이 문장으로 써라. "
    "상자 안의 글자는 보이는 그대로 정확히 옮기고, 직책과 사람 이름이 한 상자에 있으면 "
    "'직책 이름' 순서로 한 줄에 붙여 적어라(예: 교장 홍길동). "
    "상자 사이의 선과 배치가 나타내는 관계(소속, 상하, 담당)를 '~는 ~에 속한다', "
    "'~ 아래에 ~가 있다' 형태로 빠짐없이 나열하라. 보이지 않는 내용을 추측해서 적지 마라."
)
# 도해 페이지에서 VLM 설명문과 표/레이어 원문을 나누는 표식 (chunk_documents 가 이 앞부분을 줄 단위 조각으로 만든다)
DIAGRAM_SPLIT_MARKER = "[페이지 원문]"

# 파일·페이지마다 커넥션을 새로 맺지 않도록 프로세스 전체가 하나의 클라이언트를 공유한다
_client = httpx.Client(timeout=VLM_TIMEOUT)


def check_vlm_ready() -> tuple[bool, str]:
    """vlm_service가 떠 있고 비전 모델이 준비됐는지 확인한다. (성공 여부, 사유)를 돌려준다."""
    if not VLM_INTERNAL_KEY:
        return False, "INTERNAL_API_KEY 미설정 (로컬 개발은 ALLOW_DEV_KEYS=1)"
    try:
        resp = _client.get(f"{VLM_SERVICE_URL}/health", timeout=5.0)
        resp.raise_for_status()
        gen = resp.json().get("generator", {})
    except Exception as exc:
        return False, f"vlm_service({VLM_SERVICE_URL}) 연결 실패: {exc}"
    if not gen.get("model_available"):
        # 흔한 원인: VLM_MODEL이 vLLM의 GET /v1/models가 반환하는 id와 다름
        return False, f"vlm_service는 응답하지만 비전 모델이 준비되지 않음: {gen}"
    return True, f"모델 {gen.get('model')} 준비됨 ({VLM_SERVICE_URL})"


# PDF 글꼴 자체 기호(글머리표 등)는 유니코드 사설 영역(U+E000~U+F8FF)으로 뽑혀 화면·TTS 에서 "□" 가 된다.
# 실측: "전\uf000편입" (원문 "전·편입"), 줄 앞 "\uf000학교폭력…" (글머리표). 글자 사이면 가운뎃점, 아니면 공백으로.
_PUA_BETWEEN = re.compile(r"(?<=\S)[\ue000-\uf8ff]+(?=\S)")
_PUA_ANY = re.compile(r"[\ue000-\uf8ff]+")


def _clean_pua(text: str) -> str:
    return _PUA_ANY.sub(" ", _PUA_BETWEEN.sub("·", text))


def _despace_hangul(text: str) -> str:
    """깨진 레이어의 한글 글자 사이 공백을 제거한다("부서업 무총괄" → "부서업무총괄").

    실제 단어 경계 공백도 함께 사라지지만("업무 내용" → "업무내용"), 이 텍스트는
    검색 보조용이라 부분 문자열 매칭("교무기획부")이 보존되는 것이 더 중요하다.
    """
    return re.sub(r"(?<=[가-힣])[ \t]+(?=[가-힣])", "", text)


def _broken_ratio(text: str) -> float:
    """한글 조각 중 '원래 한 글자 단어가 아닌' 한 글자짜리의 비율. 표본이 20개 미만이면 0."""
    runs = _HANGUL_RUN.findall(text)
    if len(runs) < 20:  # 표본이 적으면 판단 보류 (짧은 페이지는 LOW_TEXT_THRESHOLD가 담당)
        return 0.0
    single = sum(1 for r in runs if len(r) == 1 and r not in _SINGLE_CHAR_WORDS)
    return single / len(runs)


def _is_broken_text_layer(text: str) -> bool:
    """자간 개별 배치 PDF에서 추출기가 글자마다 공백을 끼워 넣은 깨진 레이어인지 판별한다.

    깨진 텍스트는 그대로 인덱싱하면 임베딩·검색 품질이 크게 떨어진다("교과협의회"로
    질문해도 인덱스에는 "교 과 협 의 회"로 들어 있어 매칭이 약해짐).
    """
    return _broken_ratio(text) >= BROKEN_LAYER_RATIO


_DAY_CHARS = set("월화수목금토일")


def _join_vertical_runs(text: str) -> str:
    """PyMuPDF가 세로쓰기 셀을 한 글자씩 줄바꿈한 것("교\\n무\\n기\\n획\\n부")을 한 단어로 합친다.

    한 글자 한글만 있는 줄이 2개 이상 연속되면 이어 붙인다. 학사일정표 헤더처럼
    요일만 한 줄씩 나열된 경우(월/화/수/목/금)는 원래 별개의 셀이므로 건드리지 않는다.
    """
    lines = text.split("\n")
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if len(run) >= 2 and not all(c in _DAY_CHARS for c in run):
            out.append("".join(run))
        else:
            out.extend(run)
        run.clear()

    for line in lines:
        s = line.strip()
        if len(s) == 1 and "가" <= s <= "힣":
            run.append(s)
            continue
        flush()
        out.append(line)
    flush()
    return "\n".join(out)


# ── 추출기가 지어낸 공백 억제 ──────────────────────────────────────────────
# 디자인 PDF 는 소제목에 자간을 크게 준다. PyMuPDF 는 글자 간격이 임계보다 넓으면 공백을 **합성**해 넣으므로
# "TALENT DEVELOPMENT" 가 "TA L E N T D E V E LO P M E N T" 로 나온다 (같은 부류: "AT A G L A N C E", "SIGNATURE PROGRAM S",
# "CAM PUS"). 실측(09-22, AICA p10): 모델이 이 문자열을 프로그램 이름으로 골라 "대표 프로그램은 TA L E N T … 입니다" 로 답했다.
# TEXT_INHIBIT_SPACES 는 합성 공백만 안 넣고 문서에 실제로 있는 공백 문자는 그대로 둔다. 다만 공백 문자 없이 위치로만 띄운
# PDF(일부 조판·표 안 텍스트)에서는 한글 낱말까지 붙어 버리므로, **한글 사이 공백 수가 원본과 거의 같은 페이지에서만** 쓴다.
_INHIBIT_FLAGS = fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP | fitz.TEXT_INHIBIT_SPACES
# 한글·숫자 경계 공백이 이 비율 넘게 줄면 억제본을 버린다. 기본 0 = 하나라도 줄면 버림 — 실측(09-22 망미중): 2% 허용에서
# 배치도 p123 의 "교육반2 컴퓨터실" 이 "교육반2컴퓨터실" 로 붙은 채 채택됐다(123쪽 중 7쪽). 영문 소제목만 있는 페이지는 손실 0.
INHIBIT_SPACE_TOLERANCE = float(os.getenv("INHIBIT_SPACE_TOLERANCE", "0"))
# 한글·숫자·구두점 뒤에 오는 공백 경계 — "가. 학교생활", "3. 예산계획" 의 번호 뒤 공백도 합성 공백일 수 있어 같이 센다
_HANGUL_GAP = re.compile(r"[가-힣0-9).:,]\s+[가-힣0-9(]")


def _hangul_gaps(text: str) -> int:
    """한글·숫자 사이의 공백 경계 수 — 억제본이 한글 낱말을 붙여 버렸는지 재는 자."""
    return len(_HANGUL_GAP.findall(text))


def _inhibited_text(fitz_page) -> str | None:
    """합성 공백을 억제한 PyMuPDF 텍스트. 원본과 같거나(억제할 게 없음) 한글 낱말이 붙어 버리면 None."""
    try:
        plain = fitz_page.get_text("text") or ""
        inh = fitz_page.get_text("text", flags=_INHIBIT_FLAGS) or ""
    except Exception:
        return None
    if not inh.strip() or inh == plain:
        return None
    base = _hangul_gaps(plain)
    if base and _hangul_gaps(inh) < base * (1 - INHIBIT_SPACE_TOLERANCE):
        return None
    return inh


def _extract_layer_text(pypdf_page, fitz_page) -> tuple[str, str, bool]:
    """pypdf·PyMuPDF 두 추출기로 텍스트 레이어를 뽑아 덜 깨진 쪽을 (텍스트, 추출기명, 합성 공백 억제 여부)로 돌려준다.

    둘은 깨지는 방식이 반대다(모듈 docstring 참고). 비율이 같으면 pypdf(기존 동작)를 유지한다 — 단 합성 공백 억제본이
    안전하게 만들어진 페이지(_inhibited_text)에서는 그 억제본을 같은 비율일 때 우선한다(pypdf 도 같은 자간 공백을 만든다).
    한쪽 추출기가 예외를 내면 다른 쪽 결과를 쓴다.
    """
    cands: list[tuple[str, str]] = []
    inhibited = _inhibited_text(fitz_page)
    if inhibited is not None:
        cands.append(("pymupdf-inhibit", _join_vertical_runs(inhibited.strip())))
    try:
        cands.append(("pypdf", (pypdf_page.extract_text() or "").strip()))
    except Exception as exc:  # pypdf는 손상된 폰트/인코딩에서 종종 예외를 낸다
        logger.warning("pypdf 추출 실패 — PyMuPDF 결과만 사용: %s", exc)
    try:
        cands.append(("pymupdf", _join_vertical_runs((fitz_page.get_text("text") or "").strip())))
    except Exception as exc:
        logger.warning("PyMuPDF 추출 실패 — pypdf 결과만 사용: %s", exc)
    if not cands:
        return "", "none", False
    # 글자가 아예 없는 쪽(추출기 미지원 폰트)은 제외하고, 깨짐 비율이 낮은 쪽을 고른다 (동률이면 앞쪽 후보)
    nonempty = [c for c in cands if len(c[1]) >= LOW_TEXT_THRESHOLD] or cands
    engine, text = min(nonempty, key=lambda c: _broken_ratio(c[1]))
    return text, engine, engine == "pymupdf-inhibit"


_SUPERSCRIPT_FLAG = 1          # PyMuPDF span flags 의 위첨자 비트
_SUP_MARK = re.compile(r"^[\d*†‡]{1,2}$")   # 각주 표식으로 볼 위첨자: 한두 자리 숫자·기호


def _superscript_lines(fitz_page) -> list[tuple[str, str]]:
    """각주 위첨자가 든 (원본 줄, 표식을 뺀 줄) 목록 — 글꼴 정보(span flags)로만 판정한다.

    실측(09-22, AICA 기관소개): "출범1", "착공1", "AI집적단지1", "(NHN클라우드)2" — 각주 번호가 본문 숫자로 붙어 나와
    답변에 그대로 샜다("AI집적단지1에 있습니다"). 텍스트만 보고 "한글 뒤 한 자리 숫자" 를 지우면 고1·중3 같은 표기를
    잘못 지우므로 위첨자 글꼴로 찍힌 것만 지운다.

    **줄 단위**로 돌려주는 이유(09-23 실측): 앞 글자 몇 자(anchor)로 정규식 치환하던 방식은 같은 anchor 가 먼저 나오는
    다른 자리를 지웠다 — AICA p10 에서 각주는 "광주인공지능사관학교1" 인데 anchor "다." 가 "…양성합니다." 에 걸려
    **다음 줄 "1,528명" 의 1** 이 사라지고 줄이 통째로 밀렸다(골든 a36 답변 ",528명"). 줄을 통째로 바꾸면 그 줄 밖은
    건드릴 수 없다."""
    out: list[tuple[str, str]] = []
    try:
        d = fitz_page.get_text("dict")
    except Exception:
        return out
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            keep, full, found = [], [], False
            for sp in spans:
                t = sp.get("text") or ""
                full.append(t)
                if t.strip() and (sp.get("flags", 0) & _SUPERSCRIPT_FLAG) and _SUP_MARK.match(t.strip()):
                    found = True   # 위첨자 표식 — 뺀다
                else:
                    keep.append(t)
            if found:
                out.append(("".join(full).strip(), "".join(keep).strip()))
    return out


def _strip_superscripts(text: str, fitz_page) -> str:
    """위첨자 각주 표식을 페이지 텍스트에서 지운다 — 표식이 든 **줄 전체**를 표식 뺀 줄로 바꾼다 (_superscript_lines 참고).

    페이지 텍스트에 그 줄이 정확히 한 번 나올 때만 바꾼다. 여러 번 나오거나(반복되는 머리말) 줄이 달리 쪼개져
    못 찾으면 그냥 둔다 — 원문을 잘못 건드리느니 각주가 남는 편이 안전하다."""
    removed = 0
    for full, kept in _superscript_lines(fitz_page):
        if not full or full == kept:
            continue
        if text.count(full) != 1:
            continue
        # 줄이 표식뿐이었으면(광주인공지능사관학교 다음 줄의 "1") 줄바꿈까지 지운다
        text = text.replace(full, kept, 1) if kept else text.replace(full + "\n", "", 1)
        removed += 1
    if removed:
        logger.debug("위첨자 각주 표식 %d줄 정리", removed)
    return text


def _big_image_dpi(fitz_page, min_ratio: float = IMAGE_PAGE_MIN_RATIO) -> int | None:
    """페이지 면적의 min_ratio 이상을 덮는 그림들 중 **가장 낮은** 유효 DPI(원본 픽셀 폭 ÷ 표시 폭 pt × 72).
    그런 그림이 없으면 None (인포그래픽·도해 한 장짜리 페이지 판정 + 저해상도 경고용).
    가장 낮은 값을 쓰는 이유: 실측 p18 은 배경 그림(115DPI)과 인포그래픽(59DPI)이 겹쳐 있어 큰 쪽만 보면 문제를 놓친다."""
    try:
        area = fitz_page.rect.width * fitz_page.rect.height
        if area <= 0:
            return None
        dpis = []
        for info in fitz_page.get_image_info():
            r = fitz.Rect(info["bbox"])
            if r.get_area() / area < min_ratio or r.width <= 0:
                continue
            dpis.append(int(info.get("width", 0) / r.width * 72))
        return min(dpis) if dpis else None
    except Exception:
        return None


def _has_big_image(fitz_page, min_ratio: float = IMAGE_PAGE_MIN_RATIO) -> bool:
    return _big_image_dpi(fitz_page, min_ratio) is not None


def _is_blank_page(fitz_page) -> bool:
    """저해상도 흑백 렌더링에서 어두운 픽셀 비율이 BLANK_PAGE_INK_PCT 미만이면 백지다."""
    pix = fitz_page.get_pixmap(dpi=_BLANK_CHECK_DPI, colorspace=fitz.csGRAY)
    samples = pix.samples
    if not samples:
        return True
    ink = sum(1 for b in samples if b < 240) / len(samples) * 100
    return ink < BLANK_PAGE_INK_PCT


def _is_vlm_refusal(text: str) -> bool:
    """VLM이 내용 대신 '이미지를 못 받았다' 류의 사과문을 돌려줬는지 판별한다."""
    return len(text) < _VLM_REFUSAL_MAX_LEN and bool(_VLM_REFUSAL.search(text))


# VLM 전송 이미지 포맷 — 컬러 스캔본을 250DPI PNG 로 보내면 base64 가 vlm_service 상한(20MB)을 넘어 413 이 난다(실측
# 09-21 ZERO-PASS.pdf p3). JPEG 품질 90 이면 5~10배 작고 글자 판독에는 차이가 없다.
VLM_IMAGE_JPEG_QUALITY = int(os.getenv("VLM_IMAGE_JPEG_QUALITY", "90"))


def _effective_dpi(fitz_page, dpi: int) -> int:
    """VLM_RENDER_MAX_PIXELS 를 넘지 않도록 큰 페이지의 DPI 를 낮춘다."""
    w_in, h_in = fitz_page.rect.width / 72, fitz_page.rect.height / 72
    if w_in * h_in * dpi * dpi <= VLM_RENDER_MAX_PIXELS:
        return dpi
    return max(72, int((VLM_RENDER_MAX_PIXELS / (w_in * h_in)) ** 0.5))


def _render_page_png(fitz_page, dpi: int) -> bytes:
    """페이지를 VLM 전송용 이미지 바이트로 렌더링한다 (JPEG; 품질 0 이면 PNG). 큰 페이지는 픽셀 상한에 맞춰 DPI 를 낮춘다."""
    eff = _effective_dpi(fitz_page, dpi)
    if eff != dpi:
        logger.info("큰 페이지(%.0f×%.0fpt) — 렌더링 %dDPI → %dDPI (픽셀 상한 %d)",
                    fitz_page.rect.width, fitz_page.rect.height, dpi, eff, VLM_RENDER_MAX_PIXELS)
    pix = fitz_page.get_pixmap(dpi=eff)
    if VLM_IMAGE_JPEG_QUALITY <= 0:
        return pix.tobytes("png")
    return pix.tobytes("jpeg", jpg_quality=VLM_IMAGE_JPEG_QUALITY)


# 자간 맞춤용 공백이 낀 한글 낱말("학 급 수", "1 학 년"): 한 글자 한글이 공백 하나로 2개 이상 이어진 구간
_SPACED_HANGUL = re.compile(r"(?<![가-힣])(?:[가-힣] )+[가-힣](?![가-힣])")


# 자간 공백 제거를 적용하는 셀 길이 상한 — 표 머리·행 머리(낱말 하나)만 대상. 긴 문장 셀은 "알 수 있음" 같은
# 진짜 한 글자 낱말 나열이 있어 건드리지 않는다.
_UNSPACE_MAX_CELL = 12


def _unspace_hangul_word(text: str) -> str:
    """짧은 셀의 자간 공백을 없앤다("학 급 수" → "학급수", "1 학 년" → "1 학년")."""
    if len(text) > _UNSPACE_MAX_CELL:
        return text
    return _SPACED_HANGUL.sub(lambda m: m.group().replace(" ", ""), text)


def _cell_text(cell) -> str:
    """표 셀 문자열 정리 — 세로쓰기 한 글자 줄 합치기, 자간 공백 제거, 줄바꿈·연속 공백 → 한 칸, 파이프 치환."""
    if not cell:
        return ""
    return _unspace_hangul_word(" ".join(_join_vertical_runs(str(cell).strip()).split())).replace("|", "／")


# 학사일정표(달력형)의 날짜 칸: "24(목)", "10/1(목6)", "5/1(금9)" — 일(/월) + 괄호 안 요일(+시수)
_CAL_DATE_CELL = re.compile(r"^\d{1,2}(?:/\d{1,2})?\([월화수목금토일]\d*\)")


def _merge_continuation_rows(rows: list[list[str]]) -> list[list[str]]:
    """학사일정표(달력형)의 행사 행을 바로 위 날짜 행의 칸에 합친다.

    학사일정표는 한 주가 두 행이다 — 날짜 행("24(목) | 25(금)") 아래에 행사 행("추석연휴 | 추석")이
    따로 있어, 행 단위로 읽으면 날짜와 행사가 끊긴다(실측: "추석 연휴는 며칠이야?" 검색 실패).
    윗 행에 날짜 칸이 2개 이상 있고, 이 행에는 날짜 칸이 없으며 둘째 칸(주)이 비어 있으면 행사 행으로
    보고 칸마다 윗 행 값 뒤에 이어 붙인다("24(목) 추석연휴"). 날짜 칸 조건 덕에 세로 병합 표(교과 시수표·
    평가 항목표 등)의 일반 행은 건드리지 않는다 — 실측에서 조건 없이 합치면 그 표들이 뭉개졌다.
    """
    if not rows or len(rows[0]) < 4:
        return rows
    out: list[list[str]] = [rows[0]]
    for r in rows[1:]:
        prev = out[-1]
        prev_dates = sum(1 for c in prev if _CAL_DATE_CELL.match(c))
        cur_dates = sum(1 for c in r if _CAL_DATE_CELL.match(c))
        if len(out) > 1 and prev_dates >= 2 and cur_dates == 0 and not r[1] and r[0] == prev[0] and any(r[1:]):
            out[-1] = [prev[0]] + [(p + " " + c).strip() if c else p for p, c in zip(prev[1:], r[1:])]
        else:
            out.append(r)
    return out


def _table_to_markdown(tab) -> str | None:
    """PyMuPDF Table → 마크다운 표. 세로 병합된 첫 열(부서명·월 등)은 아래 행에 값을 채운다.

    2행·2열 미만이거나 내용이 없으면 None — 괘선 조각을 표로 오인한 경우를 거른다."""
    rows = [[_cell_text(c) for c in r] for r in tab.extract()]
    rows = [r for r in rows if any(r)]
    if len(rows) < 2:
        return None
    ncol = max(len(r) for r in rows)
    if ncol < 2:
        return None
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    carry = ""
    for r in rows[1:]:
        if r[0]:
            carry = r[0]
        else:
            r[0] = carry
    rows = _merge_continuation_rows(rows)
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * ncol]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _page_text_with_tables(fitz_page, inhibit_spaces: bool = False) -> tuple[str, int, int] | None:
    """표 영역은 마크다운 표로, 나머지는 텍스트 블록으로 읽기 순서(위→아래)대로 합친 페이지 텍스트.

    (텍스트, 표 개수, 작은 상자 표 개수)를 돌려주고, 표가 하나도 없으면 None.
    inhibit_spaces: 레이어 추출에서 합성 공백 억제본을 골랐으면 블록 텍스트도 같은 플래그로 뽑는다(_inhibited_text 참고)."""
    try:
        found = fitz_page.find_tables()
    except Exception as exc:  # 특이한 페이지에서 find_tables가 예외를 낼 수 있다 — 레이어로 진행
        logger.warning("find_tables 실패 — 레이어 텍스트 사용: %s", exc)
        return None
    tables: list[tuple[tuple, str]] = []
    boxes = 0
    for t in found.tables:
        if t.row_count <= 3 and t.col_count <= 3:
            boxes += 1
        md = _table_to_markdown(t)
        if md:
            tables.append((tuple(t.bbox), md))
    if not tables:
        return None
    blocks: list[list] = []   # [x0, y0, x1, y1, 텍스트]
    blocks_raw = fitz_page.get_text("blocks", flags=_INHIBIT_FLAGS) if inhibit_spaces else fitz_page.get_text("blocks")
    for x0, y0, x1, y1, txt, _bno, btype in blocks_raw:
        if btype != 0:
            continue
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if any(bx0 <= cx <= bx1 and by0 <= cy <= by1 for (bx0, by0, bx1, by1), _ in tables):
            continue  # 표 안의 글자는 마크다운 표가 담당
        s = _join_vertical_runs(txt.strip())
        if s:
            blocks.append([x0, y0, x1, y1, s])
    blocks = _merge_stacked_blocks(blocks)
    items: list[tuple[float, float, str]] = [(b[1], b[0], b[4]) for b in blocks]
    for (bx0, by0, _bx1, _by1), md in tables:
        items.append((by0, bx0, md))
    return "\n".join(it[2] for it in _sort_reading_order(items)), len(tables), boxes


# 상자(도해) 레이아웃: 나란히 놓인 상자들의 글이 줄 단위로 섞이지 않게 한다.
# 실측(09-21): 경영목표 페이지의 상자 3개("맞춤교육 / 창의성과 문제해결력을 / 갖춘 미래역량 교육" 등)가 PyMuPDF 블록에서
# 위 두 줄과 셋째 줄로 갈라졌고, y 좌표순 정렬(안심교육 y=396 < 맞춤교육 y=401)에 셋째 줄들이 뒤로 몰려
# "안심교육 / 존중과 배려의 / 맞춤교육 / … / 갖춘 미래역량 교육 / 인성 교육" 이 됐다 → 모델이 경영목표 1·2·3 을 뒤섞어 답함.
_STACK_GAP_PT = 6.0      # 위 블록 아래끝과 아래 블록 위끝 사이가 이 이하면 같은 상자의 이어지는 줄
_ROW_TOL_PT = 10.0       # 위끝 y 차이가 이 이하면 같은 행(나란한 상자들) — 행 안에서는 왼쪽부터


def _merge_stacked_blocks(blocks: list[list]) -> list[list]:
    """가로로 겹치고 세로로 바로 이어지는 블록을 한 상자로 합친다 (블록: [x0, y0, x1, y1, 텍스트])."""
    merged: list[list] = []
    for b in sorted(blocks, key=lambda b: (b[1], b[0])):
        for m in merged:
            overlap = min(b[2], m[2]) - max(b[0], m[0])
            narrower = min(b[2] - b[0], m[2] - m[0])
            if narrower > 0 and overlap >= narrower * 0.5 and -1.0 <= b[1] - m[3] <= _STACK_GAP_PT:
                m[0], m[2], m[3] = min(m[0], b[0]), max(m[2], b[2]), max(m[3], b[3])
                m[4] = m[4] + "\n" + b[4]
                break
        else:
            merged.append(list(b))
    return merged


def _sort_reading_order(items: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """(y0, x0, 텍스트) 를 행(위끝 y 가 _ROW_TOL_PT 안이면 같은 행) → 행 안에서 x 순으로."""
    out: list[tuple[float, float, str]] = []
    row: list[tuple[float, float, str]] = []
    row_y = None
    for it in sorted(items, key=lambda it: (it[0], it[1])):
        if row_y is not None and it[0] - row_y > _ROW_TOL_PT:
            out.extend(sorted(row, key=lambda r: r[1]))
            row, row_y = [], None
        if row_y is None:
            row_y = it[0]
        row.append(it)
    out.extend(sorted(row, key=lambda r: r[1]))
    return out


# 표 정규화 대상 헤더의 첫 칸 — 이런 표는 첫 열이 세로 병합된 분류(부서명)다
_DEPT_HEADER_FIRST_CELLS = ("부", "부서", "구분", "분류")
# 3칸짜리 행의 첫 칸이 이런 접미로 끝나면 "부서 칸이 빠진 행"이 아니라 "새 부서의 시작"이다
_DEPT_SUFFIXES = ("부", "실", "업무")


def _fill_table_dept(text: str) -> str:
    """VLM 전사 마크다운 표에서 병합 셀(부서명) 열을 모든 행에 명시적으로 채운다.

    소형 VLM은 세로 병합된 부서명을 첫 행에만 적거나(빈 칸 행), 아예 열을 빼먹는다
    (3칸 행). 그대로 인덱싱하면 답변 LLM이 소속을 추론해야 해서 "상담·보건이 독립
    부서로 표시"되는 오류가 났다 — 여기서 결정론적으로 채워 추론을 없앤다.
    4열(부/업무/담당자/내용) 헤더의 표에만 적용한다.
    """
    lines = text.split("\n")
    out = []
    expected = 0  # 현재 표의 열 수 (0이면 표 밖)
    carry = ""  # 마지막으로 본 부서명

    for line in lines:
        s = line.strip()
        if not s.startswith("|"):
            expected, carry = 0, ""
            out.append(line)
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if all(set(c) <= {"-", " "} for c in cells):  # 구분선 행
            out.append(line)
            continue
        if expected == 0:
            # 헤더 행 — 첫 칸이 부서류 표제이고 4열일 때만 정규화 모드 진입
            if len(cells) == 4 and cells[0] in _DEPT_HEADER_FIRST_CELLS:
                expected = 4
            out.append(line)
            continue

        if len(cells) == 4:
            if cells[0]:
                carry = cells[0]
            else:
                cells[0] = carry
        elif len(cells) == 3:
            if cells[0].endswith(_DEPT_SUFFIXES):
                # 새 부서 시작인데 담당자 칸이 빠진 형태 (예: 교과업무/국어과/내용)
                carry = cells[0]
                cells = [cells[0], cells[1], "", cells[2]]
            else:
                # 부서 칸이 빠진 형태 (예: 특수1/오OO/내용) — 직전 부서를 붙인다
                cells = [carry, cells[0], cells[1], cells[2]]
        elif len(cells) == 2:
            # 부서·담당자 칸이 모두 빠진 형태 (예: 수학과/내용)
            cells = [carry, cells[0], "", cells[1]]
        else:
            out.append(line)
            continue
        out.append("| " + " | ".join(cells) + " |")

    return "\n".join(out)


_HANGUL_ONLY = re.compile(r"[가-힣]+")
_KEEP_CHARS = re.compile(r"[가-힣A-Za-z0-9]+")


def _flat(text: str) -> str:
    """한글·영숫자만 남긴다 — 레이어와 전사의 문장부호 표기 차이('예•결산' vs '예·결산')와
    공백 차이를 무시하고 문자열을 대조하기 위한 정규화."""
    return "".join(_KEEP_CHARS.findall(text))


def _repair_roles_from_layer(body: str, layer_text: str) -> str:
    """정규화된 표의 직책(업무) 칸을 레이어 원문으로 교정한다.

    세로로 쌓인 직책명("지방교육행정주사보")을 VLM이 오독해도, 레이어에는
    `...직전행내용꼬리<직책명><담당자이름><이 행 내용>...` 형태로 정확한 글자가 있다.
    담당자 이름 + 내용 머리를 앵커로 위치를 특정하고, 직전 행 내용 꼬리를 시작
    경계로 삼아 그 사이 문자열을 직책명으로 채택한다. 경계를 못 찾거나 결과가
    수상하면(한글 아님/과도한 길이) 원래 값을 유지한다 — 틀린 교정보다 오독 유지가 낫다.
    """
    layer_flat = _flat(layer_text)
    lines = body.split("\n")
    prev_content_tail = ""
    for idx, line in enumerate(lines):
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) != 4 or all(set(c) <= {"-", " "} for c in cells):
            continue
        dept, role, person, content = cells
        role_key = _flat(role)
        person_key = _flat(person)
        content_flat = _flat(content)
        if role_key and person_key and content_flat and role_key not in layer_flat:
            anchor = person_key + content_flat[:6]
            pos = layer_flat.find(anchor)
            if pos > 0 and prev_content_tail:
                p = layer_flat.rfind(prev_content_tail, 0, pos)
                if p >= 0:
                    cand = layer_flat[p + len(prev_content_tail) : pos]
                    if 2 <= len(cand) <= 15 and _HANGUL_ONLY.fullmatch(cand):
                        logger.info("직책명 교정: %r → %r", role, cand)
                        cells[1] = cand
                        lines[idx] = "| " + " | ".join(cells) + " |"
        if content_flat:
            prev_content_tail = content_flat[-8:]
    return "\n".join(lines)


def _missing_words(layer_text: str, vlm_text: str) -> list[str]:
    """레이어 원문에는 있는데 VLM 전사에는 없는 단어들(4자 이상 한글) — 2차 시도 힌트용."""
    return sorted({w for w in _HANGUL_WORD.findall(layer_text) if w not in vlm_text})


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _vlm_cache_path(pdf_sha: str, page_number: int, prompt: str) -> Path | None:
    """캐시 파일 경로: vlm_cache/<PDF 해시 16자>_p<쪽>_dpi<DPI>_<프롬프트 해시 8자>.json. 캐시 꺼짐이면 None."""
    if not VLM_CACHE:
        return None
    ph = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
    return VLM_CACHE_DIR / f"{pdf_sha[:16]}_p{page_number}_dpi{VLM_RENDER_DPI}_{ph}.json"


def _vlm_cache_get(path: Path | None) -> tuple[str, bool] | None:
    if path is None or not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return d["text"], bool(d.get("truncated"))
    except Exception as exc:  # 깨진 캐시는 없는 셈 친다
        logger.warning("VLM 캐시 읽기 실패 — 무시: %s (%s)", path.name, exc)
        return None


def _vlm_cache_put(path: Path | None, text: str, truncated: bool, meta: dict) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"text": text, "truncated": truncated, **meta}, fh, ensure_ascii=False, indent=1)
    except Exception as exc:
        logger.warning("VLM 캐시 저장 실패 — 무시: %s (%s)", path, exc)


def _vlm_page(
    png_bytes: bytes,
    prompt: str = VLM_PROMPT,
    cache_path: Path | None = None,
    allow_call: bool = True,
    cache_meta: dict | None = None,
) -> tuple[str, bool]:
    """페이지 이미지를 vlm_service로 전사해 (텍스트, 잘림 여부)를 돌려준다.

    cache_path 가 있고 파일이 있으면 VLM 을 부르지 않고 그 결과를 쓴다(로그에 '캐시'). 없으면 호출하고 저장한다.
    allow_call=False(VLM 미기동)인데 캐시도 없으면 바로 예외 — 재시도 대기로 시간을 끌지 않는다.
    일시 장애를 넘기기 위해 재시도하고, 재시도를 소진하면 예외를 올린다 —
    호출자가 해당 페이지를 '실패'로 기록해야 한다."""
    cached = _vlm_cache_get(cache_path)
    if cached is not None:
        return cached
    if not allow_call:
        raise RuntimeError("VLM 미사용이고 캐시도 없음")
    payload = {"image_base64": base64.b64encode(png_bytes).decode(), "prompt": prompt}
    last_exc: Exception | None = None
    for attempt in range(1, VLM_RETRIES + 1):
        try:
            resp = _client.post(
                f"{VLM_SERVICE_URL}/describe",
                json=payload,
                headers={"X-Internal-Key": VLM_INTERNAL_KEY},
            )
            resp.raise_for_status()
            data = resp.json()
            text, truncated = data["text"].strip(), bool(data.get("truncated"))
            if data.get("looped"):
                logger.warning("VLM 출력 반복 루프 — vlm_service 가 반복 직전까지 잘라 돌려줌 (%d자). 사진 위주 페이지면 정상", len(text))
            if not truncated:
                _vlm_cache_put(cache_path, text, truncated, cache_meta or {})
            return text, truncated
        except Exception as exc:
            last_exc = exc
            if attempt < VLM_RETRIES:
                wait = VLM_RETRY_BACKOFF**attempt
                logger.warning("VLM 호출 실패 (시도 %d/%d) — %.0f초 후 재시도: %s", attempt, VLM_RETRIES, wait, exc)
                time.sleep(wait)
    raise RuntimeError(f"VLM 호출 {VLM_RETRIES}회 모두 실패: {last_exc}")


def extract_pdf(pdf_path: Path, vlm_enabled: bool) -> dict:
    """PDF 한 파일을 페이지 단위로 추출한다.

    페이지마다 extraction 필드로 어떤 경로로 텍스트를 얻었는지 남긴다:
    - "text_layer": 텍스트 레이어 (디지털 PDF) — layer_engine 필드에 pypdf/pymupdf 중 채택한 쪽
    - "text_layer+tables": 텍스트 레이어 + 표를 마크다운 표로 (tables 필드에 표 개수)
    - "vlm":        vlm_service 전사 (스캔본 페이지)
    - "vlm+layer":  깨진 레이어 페이지 — VLM 전사에 공백 제거한 레이어 원문을 덧붙여 인덱싱
    - "blank":      백지 페이지 — VLM에 보내지 않고 빈 텍스트 (조각이 생기지 않음)
    - "vlm_empty":  VLM이 내용 대신 사과문을 돌려준 스캔본 페이지 — 빈 텍스트, 경고
    - "failed":     VLM 재시도 소진/출력 잘림 — 하나라도 있으면 파이프라인은 실패로 끝난다
    - "skipped":    VLM_REQUIRED=0 개발 모드에서 스캔본 페이지를 건너뜀 (텍스트 비어 있음)

    스캔본(레이어 없음)과 깨진 레이어(있지만 못 씀)는 실패 처리 방식이 다르다:
    스캔본은 VLM이 유일한 소스라 실패 시 파이프라인을 세우지만, 깨진 레이어는
    불완전하게나마 원본 텍스트가 있으므로 경고만 남기고 레이어를 그대로 쓴다.
    """
    reader = PdfReader(str(pdf_path))
    fdoc = fitz.open(str(pdf_path))
    try:
        return _extract_pages(pdf_path, reader, fdoc, vlm_enabled)
    finally:
        fdoc.close()


def _extract_pages(pdf_path: Path, reader: PdfReader, fdoc, vlm_enabled: bool) -> dict:
    pages = []
    vlm_pages = []
    cached_pages = []
    pdf_sha = _file_sha256(pdf_path) if VLM_CACHE else ""
    failed_pages = []
    blank_pages = []
    empty_pages = []
    broken_pages = []
    low_dpi_pages = []   # 저해상도 그림 페이지 (IMAGE_PAGE_LOW_DPI 미만) — 문서 제공처에 원본 요청용
    diagram_pages = []
    engine_counts = {"pypdf": 0, "pymupdf": 0}
    if len(reader.pages) != len(fdoc):
        logger.warning(
            "pypdf(%d)와 PyMuPDF(%d)의 페이지 수가 다름 — PyMuPDF 기준으로 진행: %s",
            len(reader.pages), len(fdoc), pdf_path.name,
        )

    for i, fpage in enumerate(fdoc):
        pypdf_page = reader.pages[i] if i < len(reader.pages) else None
        text, engine, inhibit_spaces = _extract_layer_text(pypdf_page, fpage) if pypdf_page is not None else (
            (fpage.get_text("text") or "").strip(), "pymupdf", False)
        engine_counts[engine] = engine_counts.get(engine, 0) + 1
        extraction = "text_layer"
        is_image_page = len(text) >= LOW_TEXT_THRESHOLD and len(text) < IMAGE_PAGE_TEXT_MAX and _has_big_image(fpage)
        is_scan = len(text) < LOW_TEXT_THRESHOLD or is_image_page
        if is_image_page:
            img_dpi = _big_image_dpi(fpage)
            if img_dpi is not None and img_dpi < IMAGE_PAGE_LOW_DPI:
                logger.warning("저해상도 그림 페이지: %s %d페이지 — 그림 유효 %dDPI (< %d). 글자 오독 가능성이 높으니 "
                               "문서 제공처에 원본 이미지(150DPI 이상) 또는 텍스트를 요청하세요", pdf_path.name, i + 1, img_dpi, IMAGE_PAGE_LOW_DPI)
                low_dpi_pages.append(i + 1)
        n_tables = 0
        is_diagram = f"{pdf_path.stem}:{i + 1}" in VLM_DIAGRAM_PAGES
        if TABLE_MODE and not is_scan:
            found = _page_text_with_tables(fpage, inhibit_spaces)
            if found:
                table_text, n_tables, boxes = found
                keep = len(_flat(table_text)) / max(1, len(_flat(text)))
                if keep >= TABLE_MIN_KEEP_RATIO and not _is_broken_text_layer(table_text):
                    text, extraction = table_text, "text_layer+tables"
                    if boxes >= DIAGRAM_MIN_BOXES:
                        is_diagram = True
                else:
                    logger.info(
                        "표 추출 결과가 레이어보다 부실(글자 %.0f%%) — 레이어 유지: %s %d페이지",
                        keep * 100, pdf_path.name, i + 1,
                    )
                    n_tables = 0
        if not is_scan and text:
            text = _strip_superscripts(text, fpage)   # 각주 위첨자("출범1") 제거 — _superscript_lines 참고
        is_broken = not is_scan and _is_broken_text_layer(text)
        if is_broken:
            broken_pages.append(i + 1)
        if is_diagram:
            diagram_pages.append(i + 1)

        if is_scan and _is_blank_page(fpage):
            # 장 구분용 빈 면 — VLM에 보내면 사과문을 지어내므로 여기서 끝낸다
            pages.append({"page_number": i + 1, "text": "", "extraction": "blank", "layer_engine": engine})
            blank_pages.append(i + 1)
            continue

        diagram_only = is_diagram and not is_scan and not is_broken
        prompt = VLM_PROMPT_DIAGRAM if diagram_only else (VLM_PROMPT_IMAGE if is_image_page else VLM_PROMPT)
        cache_path = _vlm_cache_path(pdf_sha, i + 1, prompt) if pdf_sha else None
        cache_meta = {"source_file": pdf_path.name, "page_number": i + 1, "dpi": VLM_RENDER_DPI, "prompt_head": prompt[:60]}
        has_cache = cache_path is not None and cache_path.exists()
        if (is_scan or is_broken or is_diagram) and (vlm_enabled or has_cache):
            reason = "스캔본" if is_scan else ("도해" if diagram_only else "깨진 텍스트 레이어")
            try:
                png = _render_page_png(fpage, VLM_RENDER_DPI)
                vlm_text, truncated = _vlm_page(png, prompt, cache_path, vlm_enabled, cache_meta)
                if has_cache:
                    cached_pages.append(i + 1)
                    logger.info("VLM 캐시 재사용(%s): %s %d페이지 (%s)", reason, pdf_path.name, i + 1, cache_path.name)
            except Exception as exc:
                if is_scan:
                    logger.error("VLM 전사 실패(%s): %s %d페이지 (%s)", reason, pdf_path.name, i + 1, exc)
                    extraction = "failed"
                    failed_pages.append(i + 1)
                else:
                    logger.warning(
                        "VLM 재추출 실패(%s) — 텍스트 레이어를 그대로 사용: %s %d페이지 (%s)",
                        reason, pdf_path.name, i + 1, exc,
                    )
            else:
                if truncated:
                    if is_scan:
                        # 잘림 원인 판별용으로 글자 수와 끝부분을 남긴다 — 글자 수가 적은데 잘렸거나 끝부분이
                        # 같은 문구의 반복이면 빽빽한 페이지가 아니라 모델의 반복 루프다(토큰 상향으로 해결 안 됨).
                        logger.error(
                            "VLM 출력 잘림(%s): %s %d페이지 — %d자, 끝부분: %r — 빽빽한 페이지면 VLM_MAX_TOKENS 상향, "
                            "반복 루프면 원본 해상도·프롬프트 확인",
                            reason, pdf_path.name, i + 1, len(vlm_text), vlm_text[-160:],
                        )
                        extraction = "failed"
                        failed_pages.append(i + 1)
                    else:
                        logger.warning(
                            "VLM 출력 잘림(%s) — 텍스트 레이어를 그대로 사용: %s %d페이지",
                            reason, pdf_path.name, i + 1,
                        )
                elif _is_vlm_refusal(vlm_text):
                    if is_scan:
                        # 백지 판정은 통과했는데 VLM이 못 읽었다 — 장식 이미지 등. 사과문을
                        # 인덱싱하느니 비워 두고, 요약에 남겨 사람이 확인하게 한다.
                        logger.warning(
                            "VLM이 내용 대신 사과문을 반환(%s) — 빈 텍스트로 처리: %s %d페이지 (%r)",
                            reason, pdf_path.name, i + 1, vlm_text[:60],
                        )
                        text, extraction = "", "vlm_empty"
                        empty_pages.append(i + 1)
                    else:
                        logger.warning(
                            "VLM이 사과문을 반환(%s) — 텍스트 레이어를 그대로 사용: %s %d페이지",
                            reason, pdf_path.name, i + 1,
                        )
                elif is_broken and len(vlm_text) < 0.3 * len(text.replace(" ", "")):
                    # 전사가 원본 대비 지나치게 짧으면 내용을 놓친 것 — 깨졌어도 레이어가 안전
                    logger.warning(
                        "VLM 재추출이 지나치게 짧음(%d자, 레이어 %d자) — 텍스트 레이어 유지: %s %d페이지",
                        len(vlm_text), len(text), pdf_path.name, i + 1,
                    )
                else:
                    if diagram_only:
                        # 도해: VLM 설명문(상자 사이의 관계) + 표/레이어 텍스트(정확한 글자)를 함께 인덱싱
                        if len(vlm_text) >= 50:
                            text = vlm_text + "\n\n" + DIAGRAM_SPLIT_MARKER + "\n" + text
                            extraction = "vlm+layer"
                            vlm_pages.append(i + 1)
                        else:
                            logger.warning(
                                "도해 설명이 지나치게 짧음(%d자) — 표/레이어 텍스트만 인덱싱: %s %d페이지",
                                len(vlm_text), pdf_path.name, i + 1,
                            )
                    elif is_broken:
                        # 1차 전사에서 빠진 단어(세로쓰기 병합 셀의 부서명 등)가 있으면
                        # 그 단어들을 힌트로 주입해 한 번 더 전사한다 — 빠진 단어를 하나라도
                        # 되찾았고 잘리지 않았을 때만 2차 결과를 채택한다.
                        missing = _missing_words(text, vlm_text)
                        if missing:
                            try:
                                retry_prompt = VLM_PROMPT_RETRY_PREFIX + ", ".join(missing[:VLM_RETRY_MAX_HINTS])
                                vlm_text2, truncated2 = _vlm_page(
                                    png, retry_prompt, _vlm_cache_path(pdf_sha, i + 1, retry_prompt) if pdf_sha else None,
                                    vlm_enabled, cache_meta)
                            except Exception as exc:
                                logger.warning("VLM 2차 전사 실패 — 1차 결과 사용: %s %d페이지 (%s)", pdf_path.name, i + 1, exc)
                            else:
                                recovered = sum(1 for w in missing if w in vlm_text2)
                                if recovered and not truncated2 and len(vlm_text2) >= 0.5 * len(vlm_text):
                                    logger.info(
                                        "VLM 2차 전사 채택: %s %d페이지 (빠진 단어 %d개 중 %d개 복구)",
                                        pdf_path.name, i + 1, len(missing), recovered,
                                    )
                                    vlm_text = vlm_text2
                        # VLM 전사(깨끗한 구조) + 레이어 원문(정확한 글자)을 함께 인덱싱.
                        # 3B급 VLM은 세로쓰기 병합 셀(부서명 등)을 놓치거나 글자를 오독할 수
                        # 있는데, 그 값들은 레이어에 정확히 들어 있으므로 둘을 합치면
                        # 어느 쪽으로 검색해도 걸린다.
                        normalized = _fill_table_dept(vlm_text)
                        normalized = _repair_roles_from_layer(normalized, text)
                        text = normalized + "\n\n[텍스트 레이어 원문 — 검색 보조용]\n" + _despace_hangul(text)
                        extraction = "vlm+layer"
                    elif len(vlm_text) > len(text):
                        text = _fill_table_dept(vlm_text)
                        extraction = "vlm"
                    else:
                        extraction = "vlm"
                    if not diagram_only:
                        vlm_pages.append(i + 1)
        elif is_scan:
            extraction = "skipped"
        elif is_broken:
            logger.warning("깨진 텍스트 레이어 감지 — VLM 미사용으로 그대로 인덱싱: %s %d페이지", pdf_path.name, i + 1)
        elif is_diagram:
            logger.info("도해 페이지 감지 — VLM 미사용이라 표/레이어 텍스트만 인덱싱 (--with-vlm 이면 관계 설명 추가): %s %d페이지",
                        pdf_path.name, i + 1)

        pages.append({
            "page_number": i + 1, "text": _clean_pua(text), "extraction": extraction,
            "layer_engine": engine, "tables": n_tables,
        })

    total_chars = sum(len(p["text"]) for p in pages)
    avg_chars_per_page = total_chars / len(pages) if pages else 0

    return {
        "source_file": pdf_path.relative_to(DOCS_DIR).as_posix(),
        "total_pages": len(pages),
        "extracted_chars": total_chars,
        "avg_chars_per_page": round(avg_chars_per_page, 1),
        "layer_engines": engine_counts,
        "broken_layer_pages": broken_pages,
        "low_dpi_image_pages": low_dpi_pages,
        "table_pages": sum(1 for p in pages if p.get("tables")),
        "diagram_pages": diagram_pages,
        "blank_pages": blank_pages,
        "vlm_pages": vlm_pages,
        "vlm_cached_pages": cached_pages,
        "vlm_empty_pages": empty_pages,
        "failed_pages": failed_pages,
        "pages": pages,
    }


def main() -> None:
    if not DOCS_DIR.exists():
        logger.error("docs 폴더를 찾을 수 없습니다: %s", DOCS_DIR)
        sys.exit(1)

    vlm_ok, vlm_detail = check_vlm_ready()
    if vlm_ok:
        logger.info("VLM 준비 확인: %s", vlm_detail)
    elif VLM_REQUIRED:
        logger.error("VLM이 준비되지 않아 기동을 거부합니다 — %s", vlm_detail)
        logger.error("텍스트 레이어만으로 진행하려면(개발 전용) VLM_REQUIRED=0 으로 실행하세요.")
        sys.exit(1)
    else:
        logger.warning("VLM 미준비(%s) — VLM_REQUIRED=0 이므로 텍스트 레이어만으로 진행합니다. "
                       "스캔본 페이지는 빈 텍스트로 남습니다. (VLM 캐시 %s 에 있는 페이지는 캐시로 채웁니다)",
                       vlm_detail, VLM_CACHE_DIR)

    # rglob: docs/ 밑 기관별 서브폴더(docs/common/, docs/<institution_id>/)까지 재귀 탐색
    pdf_files = sorted(DOCS_DIR.rglob("*.pdf"))
    if INDEX_ONLY:
        missing = [f for f in INDEX_ONLY if not (DOCS_DIR / f).is_dir()]
        if missing:
            logger.error("INDEX_ONLY 폴더가 없습니다: %s (docs/ 밑 폴더: %s)", missing,
                         sorted(p.name for p in DOCS_DIR.iterdir() if p.is_dir()))
            sys.exit(1)
        pdf_files = [p for p in pdf_files if p.relative_to(DOCS_DIR).parts[0] in INDEX_ONLY]
        logger.info("부분 인덱싱: 폴더 %s 만 처리 (PDF %d개) — 다른 폴더의 산출물·색인은 그대로 둔다", INDEX_ONLY, len(pdf_files))
    if not pdf_files:
        logger.warning("docs 폴더에 PDF 파일이 없습니다: %s", DOCS_DIR)
        return

    # docs/ 폴더 구조(공통/기관)가 바뀔 수 있으므로 매번 싹 비우고 새로 만든다.
    # 안 지우면 예전 실행의 산출물(옛 파일명·옛 폴더 구조)이 그대로 남아 chunk_documents.py가
    # 같이 읽어버려 중복·오분류로 이어진다 — store_chroma.py가 컬렉션을 지우고 새로 만드는 것과 동일한 이유.
    # 부분 인덱싱이면 해당 폴더의 산출물만 비운다.
    if INDEX_ONLY:
        for f in INDEX_ONLY:
            if (OUTPUT_DIR / f).exists():
                shutil.rmtree(OUTPUT_DIR / f)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    else:
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
        OUTPUT_DIR.mkdir(parents=True)

    results = []
    failed_files = []
    skipped_pages_total = 0

    for pdf_path in pdf_files:
        rel = pdf_path.relative_to(DOCS_DIR)
        logger.info("처리 중: %s", rel.as_posix())
        try:
            doc = extract_pdf(pdf_path, vlm_enabled=vlm_ok)
        except Exception as exc:
            logger.error("추출 실패: %s (%s)", rel.as_posix(), exc)
            failed_files.append(rel.as_posix())
            continue

        # docs/의 서브폴더 구조를 output/에도 그대로 미러링 — 다른 기관 폴더에 같은
        # 파일명이 있어도 output/에서 서로 덮어쓰지 않게 하기 위함
        out_path = OUTPUT_DIR / rel.with_suffix(".json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

        results.append(doc)
        logger.info(
            "텍스트 레이어 추출기: %s — pypdf %d쪽, PyMuPDF %d쪽",
            rel.as_posix(), doc["layer_engines"].get("pypdf", 0), doc["layer_engines"].get("pymupdf", 0),
        )
        if doc["blank_pages"]:
            logger.info("백지 페이지(건너뜀): %s — %d페이지 %s", rel.as_posix(), len(doc["blank_pages"]), doc["blank_pages"])
        if doc["table_pages"]:
            logger.info("표 추출 페이지: %s — %d쪽 (마크다운 표로 인덱싱)", rel.as_posix(), doc["table_pages"])
        if doc["diagram_pages"]:
            logger.info("도해 페이지: %s — %s", rel.as_posix(), doc["diagram_pages"])
        if doc["broken_layer_pages"]:
            logger.info("깨진 레이어 페이지: %s — %s", rel.as_posix(), doc["broken_layer_pages"])
        if doc["vlm_pages"]:
            logger.info("VLM 전사: %s — %d페이지 %s", rel.as_posix(), len(doc["vlm_pages"]), doc["vlm_pages"])
        if doc.get("low_dpi_image_pages"):
            logger.warning("★ 저해상도 그림 페이지: %s — %s (원본 이미지·텍스트 교체 요청 대상)", rel.as_posix(), doc["low_dpi_image_pages"])
        if doc.get("vlm_cached_pages"):
            logger.info("VLM 캐시 재사용: %s — %d페이지 %s (캐시: %s)", rel.as_posix(),
                        len(doc["vlm_cached_pages"]), doc["vlm_cached_pages"], VLM_CACHE_DIR)
        if doc["vlm_empty_pages"]:
            logger.warning("VLM이 내용을 못 읽어 비워 둔 페이지(사람 확인 필요): %s — %s", rel.as_posix(), doc["vlm_empty_pages"])
        if doc["failed_pages"]:
            logger.error("전사 실패 페이지: %s — %s", rel.as_posix(), doc["failed_pages"])
            failed_files.append(rel.as_posix())
        skipped_pages_total += sum(1 for p in doc["pages"] if p["extraction"] == "skipped")

    total_pages = sum(d["total_pages"] for d in results)
    total_chars = sum(d["extracted_chars"] for d in results)
    total_vlm_pages = sum(len(d["vlm_pages"]) for d in results)
    total_blank_pages = sum(len(d["blank_pages"]) for d in results)
    total_empty_pages = sum(len(d["vlm_empty_pages"]) for d in results)

    logger.info("-" * 50)
    logger.info(
        "완료: %d/%d개 파일, 총 %d페이지 (텍스트 레이어 %d, VLM 전사 %d, 백지 %d), %d자 추출",
        len(results), len(pdf_files), total_pages,
        total_pages - total_vlm_pages - skipped_pages_total - total_blank_pages - total_empty_pages,
        total_vlm_pages, total_blank_pages, total_chars,
    )
    logger.info("결과 저장 위치: %s", OUTPUT_DIR)

    if skipped_pages_total:
        logger.warning("VLM 미사용으로 건너뛴 스캔본 페이지: %d개 (VLM_REQUIRED=0 개발 모드)", skipped_pages_total)

    if failed_files:
        # 실패가 하나라도 있으면 비0으로 종료 — run_all.py가 이후 단계를 중단해
        # 불완전한 인덱스가 만들어지는 것을 막는다
        logger.error("추출 실패가 있어 파이프라인을 실패로 종료합니다: %s", sorted(set(failed_files)))
        sys.exit(1)


if __name__ == "__main__":
    main()
