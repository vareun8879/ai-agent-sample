import json
import logging
import os
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("chunk_documents")

INPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_PATH = Path(__file__).resolve().parent / "output_chunks" / "chunks.jsonl"
INDEX_ONLY = [s.strip().strip("/") for s in os.getenv("INDEX_ONLY", "").split(",") if s.strip()]   # load_documents.py 참고

# 1200자: 임베딩 모델이 bge-m3(입력 8192토큰)로 바뀌면서 상향 — 표가 많은 페이지가
# 잘게 조각나 "표 전체" 질문에 일부만 검색되던 문제를 완화한다.
# (e5 계열로 되돌리면 512토큰 입력 한도 때문에 400자로 복원해야 한다)
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
# 하드 컷 지점(CHUNK_SIZE)에서 이 범위만큼 앞을 살펴 문장 끝/공백을 찾는다
BOUNDARY_SEARCH_WINDOW = 120
SENTENCE_ENDING = re.compile(r"[.!?。！？]\s|\n")
# 마크다운 표(1단계 find_tables 출력)가 조각 경계에 걸리면, 뒷조각에도 헤더 행을 붙여
# "이 숫자가 어느 열의 값인지"를 조각 혼자서도 알 수 있게 한다.
TABLE_LINE = re.compile(r"^\s*\|")
TABLE_SEP_LINE = re.compile(r"^\s*\|(\s*-+\s*\|)+\s*$")

# ── 표 행 사실 조각(fact) ──────────────────────────────────────────────
# 페이지 조각(1,200자)과 별개로, 마크다운 표의 **행 하나**를 한 줄 문장으로 편 작은 조각을 함께 색인한다.
# 이유(실측 2026-09-17, 망미중 골든셋): cross-encoder reranker 는 짧고 초점이 있는 지문에는 높은 확률을
# 주지만(예: "학생수: 1학년 83, …, 계 255" 0.6~0.9), 같은 사실이 1,200자짜리 표 조각 속에 묻히면
# 0.01 아래로 떨어져 무관 질문과 구분이 안 됐다 — 달력·통계표·조직표 질문 5건이 그래서 "찾을 수 없음"
# 이었다. 행 조각은 검색·재채점용이고, LLM 컨텍스트는 expansion(small-to-big)이 페이지 전문으로 바꿔
# 준다(expansion.page_text 는 kind="fact" 를 제외한다).
# 행 하나가 "{표 제목} — {행 머리}: {열 이름} {값}, …" 이 된다. 열 이름은 표 헤더에서 오되, 병합 헤더
# ("요일, 날짜(시수)" 뒤에 빈 칸이 이어지는 것)는 칸마다 붙이지 않는다. 헤더 행에 칸이 하나만 찬 표
# ("| 교무기획부 |  |")는 그 칸을 표 제목으로 쓴다(조직도 페이지의 소제목 표).
FACT_HEADING_MAX_LEN = 60   # 표 위의 이 길이 이하 줄을 표 제목으로 쓴다
FACT_CELL_SPLIT_LEN = 200   # 이보다 긴 셀(주요 일정처럼 항목이 여럿 든 셀)은 글머리표 항목으로 쪼갠다
FACT_MIN_ITEM_LEN = 6       # 쪼갠 항목이 이보다 짧으면 버린다 ("* 5" 같은 찌꺼기)
FACT_MAX_LEN = 400          # 너무 긴 행(설문 문항 등)은 검색에 도움이 안 되므로 앞부분만 남긴다
FACT_LABEL_MAX_LEN = 12     # 열 이름으로 쓰는 헤더 칸의 최대 길이
# 글머리표: "- ", "* ", "· " 는 뒤에 공백이 있어야 하고(숫자·마크다운과 구분), 특수 글머리표(∙•▪◦)는 "∙학생들의…" 처럼
# 공백 없이 붙어도 항목 시작이다 — 실측(09-21): 설문 분석 표의 "∙…86%… ∙…80%…" 가 한 덩어리로 남아 항목별 조각이 안 됐다
_BULLET_SPLIT = re.compile(r"(?:^|\s)(?:[-*·]\s+|[•▪◦∙]\s*)")
# 연혁처럼 한 셀에 "1982.03.01. … 2001.03.01. … 2024.03.01. …" 이 이어진 경우 — 날짜 토큰 앞에서 자른다
_DATE_SPLIT = re.compile(r"(?=(?:19|20)\d{2}\.\s?\d{1,2}\.\s?\d{1,2}\.?)")


def _split_cell(v: str) -> list[str]:
    """긴 셀을 항목으로 쪼갠다: 글머리표가 2개 이상이면 글머리표로, 날짜 토큰이 2개 이상이면 날짜 앞에서,
    아니면 통째로. 실측(09-17): 연혁 셀이 1,000자 한 덩어리라 "언제 개교?" 의 핵심 근거가 안 잡혔고,
    페이지 전문만 받은 7B 가 1980 을 지어냈다. "1982.03.01. 망미여자중학교 개교 …" 가 한 조각이 되면 잡힌다."""
    if len(_BULLET_SPLIT.findall(v)) >= 2:
        parts = _BULLET_SPLIT.split(v)
    elif len(_DATE_SPLIT.findall(v)) >= 2:
        parts = _DATE_SPLIT.split(v)
    else:
        parts = [v]
    return [it.strip() for it in parts if len(it.strip()) >= FACT_MIN_ITEM_LEN]
_HEADING_NUMBERING = re.compile(r"^(?:[가-힣]|\d{1,2}|[ⅠⅡⅢⅣⅤⅥⅦⅧ])[.)]\s*")
# 학사일정표(달력형) 날짜 칸: "18(수3) 학부모총회", "10/1(목6) 유해약물…", "24(목) 추석연휴" → (월?, 일, 요일, 행사)
_CAL_CELL = re.compile(r"^(?:(\d{1,2})/)?(\d{1,2})\(([월화수목금토일])\d*\)\s*(.*)$")


def _calendar_days(cells: list[str], min_dates: int = 2) -> list[tuple[int, int, str, str]]:
    """달력 행(첫 칸이 월, 날짜 칸 2개 이상)의 날짜 칸을 (월, 일, 요일, 행사)로 푼다. 달력 행이 아니면 빈 리스트.

    월 경계 주(週)의 월 추정: 주 행의 월 칸은 그 주에 '들어 있는' 달 하나만 적힌다 — "| 7 | 18 | 29(월) 기말고사 | 30(화) |
    7/1(수16) | …" 처럼 6월 29·30일이 7월 행에 온다. 이때 새 달의 첫날만 "7/1" 로 월이 명시되므로, 그 앞 칸들은 (명시된
    월 − 1)로 본다. 실측(09-19, 1000문항): "1학년 응급심폐소생술은?" 에 모델이 주 행의 월 칸을 읽어 "7월 30일" 로 답했다
    (정답 6월 30일). 명시 칸이 없으면 전부 월 칸의 달이다."""
    head = cells[0].strip()
    if not head.isdigit() or not 1 <= int(head) <= 12:
        return []
    matches = [_CAL_CELL.match(c.strip()) for c in cells[1:]]
    if sum(1 for m in matches if m) < min_dates:
        return []
    head_mon = int(head)
    explicit = next((int(m.group(1)) for m in matches if m and m.group(1)), None)
    before_boundary = explicit is not None
    out = []
    for m in matches:
        if not m:
            continue
        month, day, dow, event = m.groups()
        if not 1 <= int(day) <= 31:
            continue
        if month:
            before_boundary = False
            mon = int(month)
        elif before_boundary:
            mon = explicit - 1 if explicit > 1 else 12
        else:
            mon = head_mon
        out.append((mon, int(day), dow, event.strip()))
    return out


def _calendar_day_facts(heading: str, cells: list[str]) -> list[str]:
    """달력 행을 날짜 하나 = 사실 하나로 편다: "{표 제목} — 3월 18일(수) 학부모총회".

    행 조각("3: 주 3, 16(월2) 학부모상담주간, 17(화3), 18(수3) 학부모총회, …, 수업일수 5")은 검색에는 걸리지만
    7B 가 옆 칸(17)이나 시수(3)·수업일수를 답으로 집는 실측 결함(학부모총회 → "17(화3)", 추석 연휴 → "3일")이 있어,
    행사 있는 날짜마다 짧은 조각을 따로 둔다. 행사 없는 날짜 칸("21(월5)")은 만들지 않는다."""
    pre = f"{heading} — " if heading else ""
    return [f"{pre}{mon}월 {day}일({dow}) {event}" for mon, day, dow, event in _calendar_days(cells) if event]


# ── 페이지 조각용 달력 풀어쓰기 ────────────────────────────────────────
# 날짜 조각(fact)은 검색에만 쓰이고 LLM 은 페이지 전문(expansion.page_text = kind="page" 조각)을 읽는다. 그런데 달력형
# 학사일정표는 전문으로 읽으면 한 주가 한 행이라 옆 칸 값·시수 숫자·주 행의 월 칸이 섞인다 — 실측(09-19, 1000문항 중
# 날짜 오독 12건: 6/30 → "7/30", 4/30 → "5/30" 등). 페이지 조각에 넣는 텍스트에서는 달력 표를 "7월 21일(화) 여름방학식"
# 식의 날짜별 줄로 풀어 쓴다. 행사 없는 날은 생략하고, 주별 수업일수는 줄 끝에 남긴다. 달력이 아닌 표는 그대로 둔다.
def linearize_calendar_tables(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("|"):
            out.append(lines[i])
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].lstrip().startswith("|"):
            j += 1
        block = lines[i:j]
        rows = [_split_row(l) for l in block]
        cal_rows = [(cells, _calendar_days(cells)) for cells in rows]
        n_cal = sum(1 for _, d in cal_rows if d)
        if n_cal < 3:
            out.extend(block)          # 달력이 아닌 표: 그대로
            i = j
            continue
        cal_rows = [(cells, _calendar_days(cells, min_dates=1)) for cells in rows]  # 달력 블록이면 날짜 1칸짜리 주도 포함
        header = rows[0] if rows and not _is_sep_row(rows[0]) else []
        # "수업 일수" 열 위치 — 헤더에 그 이름이 있으면 주별 수업일수를 붙인다
        # 주별 합계 열(수업일수·영업일수 등 "…일수" 머리) — 있으면 줄 끝에 그 이름 그대로 붙인다
        days_col = next((k for k, h in enumerate(header) if h.replace(" ", "").endswith("일수")), -1)
        days_name = header[days_col].replace(" ", "") if days_col >= 0 else ""
        out.append("(달력형 일정표 — 날짜별 정리. 각 줄은 '월 일(요일) 내용' 이며 내용 없는 날은 생략)")
        for cells, days in cal_rows:
            if _is_sep_row(cells):
                continue
            if days:
                events = [f"{mon}월 {day}일({dow}) {ev}" for mon, day, dow, ev in days if ev]
                week = cells[1].strip() if len(cells) > 1 and cells[1].strip().isdigit() else ""
                tail = ""
                if days_col >= 0 and days_col < len(cells) and cells[days_col].strip():
                    tail = f" [이 주 {days_name} {cells[days_col].strip()}]"
                if events:
                    out.append((f"{week}주차: " if week else "") + " / ".join(events) + tail)
                elif tail:
                    out.append((f"{week}주차: " if week else "") + "내용 없음" + tail)
            elif cells is header:
                continue
            elif cells and cells[0] and any(cells[1:]):
                # 달력 밖 행(수업 일수 합계, 주요 일정): "라벨: 값, 값"
                out.append(cells[0] + ": " + ", ".join(c for c in cells[1:] if c))
            elif cells and any(cells[1:]):
                # 머리 칸이 빈 행(요일별 고정 수업·교시 안내): 값만 이어 붙인다
                out.append(", ".join(c for c in cells[1:] if c))
        i = j
    return "\n".join(out)


# 도해(조직도) 페이지: load_documents 가 VLM 설명문 + 이 표식 + 표/레이어 원문 으로 만든다
DIAGRAM_SPLIT_MARKER = "[페이지 원문]"
_LINE_LEAD = re.compile(r"^[\s\-*•▪◦·∙\d.)]+")


def diagram_facts(text: str) -> list[str]:
    """도해 페이지의 VLM 설명문(표식 앞부분)을 한 줄 = 사실 하나로 편다. 도해 페이지가 아니면 빈 리스트.

    조직도 p26 실측: 레이어 텍스트에는 "교장" 과 "허OO" 가 다른 줄이라 "교장 선생님 성함" 질문이 어떤 조각에도
    걸리지 않았다(reranker 0.015). VLM 은 '교장 허OO 아래에 교감 서OO 가 있다' 같은 한 줄 문장을 내므로,
    그 줄을 표 행 조각과 같은 방식(짧고 초점 있는 조각)으로 색인한다."""
    if DIAGRAM_SPLIT_MARKER not in text:
        return []
    prose = text.split(DIAGRAM_SPLIT_MARKER, 1)[0]
    facts: list[str] = []
    seen: set[str] = set()
    for line in prose.split("\n"):
        s = _LINE_LEAD.sub("", line.strip()).strip()
        if len(s) < FACT_MIN_ITEM_LEN or len(s) > FACT_MAX_LEN or s.startswith("|"):
            continue
        if s not in seen:
            seen.add(s)
            facts.append(s)
    return facts


# ── 산문 조각(prose fact) ──────────────────────────────────────────────
# 표가 거의 없는 소개자료·인포그래픽형 문서(실측 09-18, AICA 기관소개)는 표 행 조각이 없어 700~1,000자 페이지 조각만
# 남고, reranker 는 긴 조각에 낮은 확률을 줘 "최대 3개월 무상", "총사업비 약 105억원" 같은 값이 검색 0건이 됐다.
# 표 밖 본문에서도 **숫자가 든 문장 하나** 또는 **연속된 짧은 줄 묶음**(인포그래픽의 "자원 이용 / 최대 3개월 / 무상 제공")을
# 표 행 조각과 같은 짧은 조각으로 색인한다. 출처·각주·URL·머리말꼬리말은 뺀다.
PROSE_SHORT_LINE = 24        # 이 길이 이하 줄이 2개 이상 이어지면 한 묶음
PROSE_FACT_MIN_LEN = 12
PROSE_FACT_MAX_LEN = 160
_PROSE_NOISE = re.compile(r"^(?:출처|※|각주|주\)|▶|file:)")
_PROSE_URL = re.compile(r"https?://|www\.|file:///|\.(?:kr|com|net|org)\b")
_PROSE_SENT_END = re.compile(r"[.!?。]\s*\d?$")           # 문장 끝 (각주 번호 하나가 붙은 "…합니다.2" 포함)
_PROSE_SENT_SPLIT = re.compile(r"(?<=[.!?。])\s+")
_PROSE_STEP = re.compile(r"^(?:\d{1,2}|[›▶→])$")           # 흐름도의 단계 번호·화살표 줄 — 묶음을 끊는다
_PROSE_LEAD_NUM = re.compile(r"^(?:\d{1,2}[.)]|[가-힣][.)]|[①-⑳])\s*")  # 목록 번호 — 숫자 판정에서 제외
# 조각으로 삼을 만한 '값'이 든 줄: 숫자, 영문 낱말(PPST·aTOPS 같은 약칭), 따옴표로 묶인 이름(‘협성회’) —
# 실측(09-18, 배재고): "신입생 예비과정(PPST 예비시험 등)" 은 숫자가 없어 조각이 안 됐고 검색 0건이었다
_PROSE_VALUE = re.compile(r"\d|[A-Za-z]{2,}|[‘'\"“][^’'\"”]{2,20}[’'\"”]")
# 정의 줄: 짧은 라벨 + "—"/":" + 설명 ("철인의 메스 — 철학·인문학·… 융합형 프로그램", "학교 유형: 사립"). 값(숫자·영문·따옴표)이
# 없어도 "X 가 뭐야?" 의 답이므로 조각으로 만든다 — 실측(09-21 키오스크, 배재고 p8): "철인의 메스가 뭐야" 가 페이지 조각(776자)
# 만으로는 리랭커 0.03 미만이라 1초 거절, 긴 문장 질문만 통과했다. 라벨을 스스로 갖고 있으므로 소제목을 앞에 붙이지 않는다.
_PROSE_DEF = re.compile(r"^([^—:：]{2,24}?)\s*[—:：]\s+\S.{7,}$")
# 한 줄을 온전히 차지하는 날짜만 연혁 경계로 취급한다. 본문 속 숫자는 해당하지 않는다.
_PROSE_DATE_LINE = re.compile(
    r"\d{4}(?:\s*년(?:\s*(?:0?[1-9]|1[0-2])\s*월"
    r"(?:\s*(?:0?[1-9]|[12]\d|3[01])\s*일)?)?"
    r"|[./-](?:0?[1-9]|1[0-2])(?:[./-](?:0?[1-9]|[12]\d|3[01]))?\.?)?\s*[~～]?"
)


def prose_date_pairs(lines: list[str]) -> dict[int, tuple[int, str]]:
    """날짜·한 줄 설명이 반복되는 구조만 후보로 삼고, 불명확한 줄은 기존 처리에 남긴다."""
    pairs: dict[int, tuple[int, str]] = {}
    group: list[tuple[int, int, str, str]] = []

    def flush() -> None:
        # 실측(2026-10-30, a38): 연도만 있는 줄과 연·월 줄이 섞인 연혁에서 날짜가 누락됐다.
        # 숫자만 있는 목록을 연혁으로 단정하지 않도록 명시적 날짜 표기도 요구한다.
        if len(group) >= 2 and any(re.search(r"년|[./-]", date) for _, _, date, _ in group):
            for start, end, date, body in group:
                pairs[start] = (end, f"{date} {body}")
        group.clear()

    i = 0
    while i < len(lines):
        date = lines[i].strip()
        j = i + 1
        if j < len(lines) and not lines[j].strip():
            j += 1  # 날짜와 설명 사이 빈 줄 하나만 허용한다.
        body = lines[j].strip() if j < len(lines) else ""
        if (_PROSE_DATE_LINE.fullmatch(date) and body
                and not _PROSE_DATE_LINE.fullmatch(body)
                and not TABLE_LINE.match(body) and not _PROSE_NOISE.match(body)
                and not _PROSE_URL.search(body)
                and not re.match(r"^(?:[#•▪◦∙*▶]|[-·]\s|\d{1,2}[.)]\s)", body)
                and re.search(r"[가-힣A-Za-z]", body)
                and PROSE_FACT_MIN_LEN <= len(date + " " + body) <= PROSE_FACT_MAX_LEN):
            group.append((i, j + 1, date, body))
            i = j + 1
            continue
        if date:  # 항목 사이 빈 줄은 허용하되 다른 본문이 나오면 반복 구조를 끊는다.
            flush()
        i += 1
    flush()
    return pairs


def page_line_key(line: str) -> str:
    """머리말·꼬리말 판정용 정규화 — 쪽 번호 등 숫자를 지운 형태 ("망미중학교 ⧕ 3" 과 "… ⧕ 57" 이 같은 키)."""
    return re.sub(r"\d+", "#", line.strip())


def repeated_lines(pages: list[dict], min_pages: int = 3) -> set[str]:
    """페이지 가장자리에서 반복되는 텍스트만 머리말·꼬리말 후보로 삼는다.

    숫자가 있는 줄은 연도·수량을 잃지 않도록 보존한다. 본문 내부에서도 나타나는
    줄은 반복 제목일 수 있으므로 제외하지 않는다.
    """
    count: dict[str, int] = {}
    interior: set[str] = set()
    for page in pages:
        lines = [l.strip() for l in page.get("text", "").split("\n") if l.strip()]
        if len(lines) <= 6:
            interior.update(page_line_key(l) for l in lines)
            continue  # 본문과 가장자리를 구분할 수 없는 짧은 페이지는 제거 근거로 쓰지 않는다.
        interior.update(page_line_key(l) for l in lines[3:-3])
        edge_keys = {page_line_key(l) for l in lines[:3] + lines[-3:]
                     if re.search(r"[가-힣A-Za-z]", l) and not re.search(r"\d", l)
                     and not TABLE_LINE.match(l)}
        for k in edge_keys:
            count[k] = count.get(k, 0) + 1
    return {k for k, n in count.items() if n >= min_pages and k not in interior}


def prose_facts(text: str, skip_keys: set[str] | None = None) -> list[str]:
    """표 밖 본문에서 숫자가 든 문장·짧은 줄 묶음을 한 줄 조각으로. 도해 페이지(VLM 설명문)는 diagram_facts 가 맡는다."""
    if DIAGRAM_SPLIT_MARKER in text:
        return []
    skip_keys = skip_keys or set()
    facts: list[str] = []
    seen: set[str] = set()

    # 소제목: 숫자·값이 없는 짧은 줄 하나("광주인공지능사관학교", "신청 절차") — 뒤따르는 문장·묶음 조각 앞에 "소제목 — " 로
    # 붙인다. 표 행 조각의 "표 제목 — " 과 같은 역할. 실측(09-18, AICA): "사관학교 7기 총사업비?" 의 정답 문장에는
    # "제7기·총사업비" 만 있고 "사관학교" 는 위 소제목에만 있어 검색 후보에 못 들었다.
    label = ""

    def add(s: str) -> None:
        s = s.strip()
        body = _PROSE_LEAD_NUM.sub("", s)
        is_def = bool(_PROSE_DEF.match(s))
        # 소제목 아래 문장은 값이 없어도 조각으로 — "동아리 · 비교과 — 자율탐구프로그램과 … 활동이 활발합니다" 는
        # "동아리 활동은 뭐가 있어?" 의 답이다 (같은 실측). 소제목도 정의 줄도 없는 값 없는 문장은 여전히 뺀다.
        if not (is_def or label or _PROSE_VALUE.search(body)) or not PROSE_FACT_MIN_LEN <= len(s) <= PROSE_FACT_MAX_LEN:
            return
        if label and not is_def and label not in s:
            s = f"{label} — {s}"
        s = re.sub(r"[ 	]{2,}", " ", s)   # "철인의 메스 —  철학…" 처럼 원문의 겹공백 정리
        s = s[:FACT_MAX_LEN]
        if s not in seen:
            seen.add(s)
            facts.append(s)

    para: list[str] = []   # 문장 부호로 안 끝난 긴 줄들 (PDF 줄바꿈 복원)
    run: list[str] = []    # 연속된 짧은 줄들 (인포그래픽 묶음)

    def flush_para() -> None:
        if para:
            for sent in _PROSE_SENT_SPLIT.split(" ".join(para)):
                if len(sent) > PROSE_FACT_MAX_LEN:
                    # 마침표 없는 "라벨 값" 줄들(소개자료의 "학교 유형 …", "운영 특징 … 7교시 …")이 한 덩어리로 붙은 것 —
                    # 문장이 아니므로 원래 줄 단위로 되돌려 조각화한다 (실측 09-18 배재고 p2: 덩어리째 버려져 검색 0건)
                    for line in para:
                        add(line)
                else:
                    add(sent)
            para.clear()

    def flush_run() -> None:
        nonlocal label
        if len(run) >= 2:
            add(" ".join(run))
        # 소제목 = 홀로 선 값 없는 짧은 줄("신청 절차"), 또는 값 줄들 뒤에 붙은 제목("1,528명 76% 7기 광주인공지능사관학교").
        # 흐름도 단계 묶음("사업공고 확인 / 홈페이지· / aTOPS 공고")의 끝 줄은 소제목이 아니다 — 실측(09-18): 앞 단계의
        # "우수성 평가" 가 "자원 이용 최대 3개월 무상 제공" 에 붙어 검색이 떨어졌다
        if (run and not _PROSE_VALUE.search(run[-1])
                and not _PROSE_SENT_END.search(run[-1]) and len(run[-1]) >= 4
                and (len(run) == 1 or all(_PROSE_VALUE.search(l) for l in run[:-1]))):
            label = run[-1]
        run.clear()

    lines = text.split("\n")
    date_pairs = prose_date_pairs(lines)
    if date_pairs:
        logger.debug("날짜·설명 조각 %d개: %s", len(date_pairs), list(date_pairs.values()))
    consumed_until = 0
    for li, raw in enumerate(lines + [""]):
        if li < consumed_until:
            continue
        if li in date_pairs:
            flush_para()
            flush_run()
            label = ""
            consumed_until, fact = date_pairs[li]
            add(fact)
            continue
        line = raw.strip()
        if not line or TABLE_LINE.match(line) or _PROSE_NOISE.match(line) or _PROSE_URL.search(line) \
                or page_line_key(line) in skip_keys or _PROSE_STEP.match(line):
            flush_para()
            flush_run()
            continue
        if para and not _PROSE_SENT_END.search(para[-1]):
            para.append(line)                      # 앞 줄이 문장 중간에서 잘렸다 — 이어 붙인다 (짧아도)
            if _PROSE_SENT_END.search(line):
                flush_para()
            continue
        if len(line) <= PROSE_SHORT_LINE:
            flush_para()
            run.append(line)
            continue
        flush_run()
        para.append(line)
        if _PROSE_SENT_END.search(line):
            flush_para()
    return facts


_EMPTY_CELL = re.compile(r"^[․.\-–—·ㆍ]+$")   # "․", "-", "—" 만 든 칸 = 빈 칸 (실측 09-21: "교감 ․" 가 그대로 읽혔다)


def _split_row(line: str) -> list[str]:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return ["" if _EMPTY_CELL.match(c) else c for c in cells]


def _is_sep_row(cells: list[str]) -> bool:
    return all(set(c) <= {"-", " "} for c in cells)


def _header_labels(header: list[str]) -> list[str]:
    """열 이름 — 병합 헤더(뒤에 빈 칸이 이어지는 칸)는 각 열의 이름이 아니므로 비운다.
    긴 헤더 칸(FACT_LABEL_MAX_LEN 초과)은 헤더가 아니라 데이터 첫 행이 헤더로 잡힌 것(명단 표)이라 비운다.
    모서리 칸이 단위 이름이고 나머지가 숫자인 헤더("학년 | 1 | 2 | 3 | 계")는 숫자 칸에 단위를 붙인다
    ("1학년, 2학년, 3학년") — 실측(09-18): "1학년 학생은 몇 명?" 이 "학생수: 1 83, 2 81" 조각과 안 겹쳐 검색이
    동아리표(1학년 12명)로 빠졌다. 문서 특정 값이 아니라 통계표 헤더의 일반 형태."""
    corner = header[0] if header else ""
    numeric = [h for h in header[1:] if h.isdigit()]
    unit = corner if corner and len(corner) <= 4 and len(numeric) >= 2 else ""
    labels = []
    for i, h in enumerate(header):
        # 뒤에 빈 칸이 이어지면 병합 칸 — 단, 그 뒤가 전부 빈 칸(표 끝의 남는 열)이면 병합이 아니라 마지막 열 이름이다
        spans = i + 1 < len(header) and header[i + 1] == "" and any(header[i + 1:])
        if spans or len(h) > FACT_LABEL_MAX_LEN:
            labels.append("")
        elif unit and i > 0 and h.isdigit():
            labels.append(h + unit)
        else:
            labels.append(h)
    return labels


# 칸 머리에 자간을 벌려 쓴 짧은 낱말("학 교 발전하는", "교 사 연구하는") — 뒤에 여러 글자 낱말이 이어질 때만
_SPACED_LEAD = re.compile(r"^([가-힣](?: [가-힣]){1,3}) (?=[가-힣]{2,})")
_NUMERIC_CELL = re.compile(r"^[\d,.\s()㎡%~\-]+$")   # 값만 든 칸 ("86", "22 (1)", "1,213.29㎡")
# 배치도 표의 층 행 머리 칸 ("2층", "지하1층", "지하 1층") — table_facts 의 층 행 처리 참고
_FLOOR_HEAD = re.compile(r"^(?:지하\s*)?\d{1,2}\s*층$")
# 단위가 붙은 값 칸("4명", "6명", "3학급", "27일간") — 둘째 머리 행 판정에서 열 이름이 아니라 값으로 본다.
# 실측(09-21 키오스크): 밴드 멤버 표의 값 행 "4명 | 6명 | …" 이 둘째 머리로 오인돼 조각이 하나도 안 생겼다.
_UNIT_VALUE_CELL = re.compile(r"^\d[\d,.]*\s*[가-힣%㎡]{0,3}$")


def _is_subheader(header: list[str], cells: list[str]) -> bool:
    """머리 행 바로 아래 행이 **둘째 머리 행**인가 — 위 머리 행에 병합 칸(뒤에 빈 칸이 이어짐)이 있고, 이 행은 첫 칸이 비었거나
    위와 같고, 찬 칸이 둘 이상이며 값(숫자)만 든 칸이 하나도 없으면 열 이름의 둘째 줄이다.
    실측(09-21 키오스크): 졸업생 진학 표 — 위 머리 "재적 | 일반계고 | (빈 칸×9) | 특목고 | 마이스터고…", 아래 머리
    " | 남일고 | 덕문 여고 | … ". 둘째 줄을 데이터로 읽어 "86: 22 (1), 13, …" 조각이 되면서 모델이 남일고 22 를 "일반계고 22명" 으로 답했다."""
    if not header or len(cells) != len(header):
        return False
    if not any(i + 1 < len(header) and h and header[i + 1] == "" for i, h in enumerate(header)):
        return False
    # 첫 칸이 찬 둘째 줄은 보통 데이터 행이지만, 위 머리의 첫 칸 자체가 병합 그룹("공연부 | (빈)×4 | 합계")이면 첫 칸도
    # 그 그룹의 열 이름이다. 실측(09-21 키오스크): 밴드 멤버 구성 표 "보컬 | 일렉기타 | 베이스 | …" 가 데이터로 읽혀
    # "멤버 구성 — 4명 6명 3명 …: 합계 19명" 조각만 남았고 "보컬은 몇 명?" 이 엉뚱한 행을 읽었다.
    if cells[0] and cells[0] != header[0] and not (header[0] and len(header) > 1 and header[1] == ""):
        return False
    filled = [c for c in cells[1:] if c]
    if len(filled) < 2 or any(_NUMERIC_CELL.match(c) or _UNIT_VALUE_CELL.match(c) for c in filled):
        return False
    # 열 이름은 짧다 — 찬 칸의 2/3 이상이 FACT_LABEL_MAX_LEN 이하여야 한다. 실측(09-21): 달력표의 요일별 수업 안내 행
    # ("월 2년 정보 순회(1,2,3) | 화 … | …", 전부 긴 칸)이 둘째 머리로 잡혀 주 행 조각마다 "요일, 날짜(시수)" 가 붙었다
    short = sum(1 for c in filled if len(c) <= FACT_LABEL_MAX_LEN)
    return short * 3 >= len(filled) * 2


def _is_repeated_header(cells: list[str]) -> bool:
    filled = [c for c in cells[1:] if c]
    return len(filled) >= 2 and not any(_NUMERIC_CELL.match(c) for c in filled)


def _merge_header_labels(header: list[str], sub: list[str]) -> list[str]:
    """두 줄 머리 → 열 이름: 병합 칸 이름(앞으로 채움)과 둘째 줄 이름을 "일반계고 남일고" 처럼 잇는다.
    둘째 줄 이름이 너무 길면(학교 나열) 위 이름만 쓴다."""
    labels, group = [], ""
    for i, h in enumerate(header):
        if h:
            group = h
        elif i == 0:
            group = ""
        s = sub[i] if i < len(sub) else ""
        if len(s) > FACT_LABEL_MAX_LEN:
            s = ""
        if group and s and s != group:
            lab = f"{group} {s}"
        else:
            lab = s or group
        labels.append(lab if len(lab) <= FACT_LABEL_MAX_LEN else group)
    return labels


_ORDINAL_CELL = re.compile(r"^(\d{1,2})\s*([가-힣]{1,2})$")


def _ordinal_header(header: list[str]) -> bool:
    """머리 행의 칸이 전부 '숫자+같은 접미'(1호 | 2호 | … | 9호, 1단계 | 2단계 | …)이면 이 표엔 행 이름 열이 없다 —
    이름 열이 있으려면 그 열의 머리 칸이 따로 있어야 하는데, 머리 칸 N개가 모두 순번이면 데이터 행의 N칸은 전부 그 순번의
    값이다. 실측(09-18~19, 3회 연속): "가해자 조치 1호는?" — 조치 표(1호~9호 머리, 값 한 행)를 첫 칸 '서면사과' 를 행 이름으로
    읽어 "서면사과: 2호 접촉, 협박 …" 조각이 되면서 '1호' 가 사라졌고, 모델은 2호 내용을 답했다. 접미가 섞이거나
    순번 아닌 칸이 하나라도 있으면(1학년 | 2학년 | 계) 그대로 행 이름 열이 있는 표다."""
    if len(header) < 3 or not all(header):
        return False
    m = [_ORDINAL_CELL.match(h) for h in header]
    return all(m) and len({x.group(2) for x in m}) == 1


def _merged_column_facts(heading: str, labels: list[str], rows: list[list[str]]) -> list[str]:
    """세로로 병합된 칸 — 어떤 열의 값이 첫 데이터 행에만 있고 나머지 행(2개 이상)은 비어 있으면 표 전체의 값이다
    ("총액(원) 2,000,000" 이 1행에만 찍힌 예산표). 행 조각에는 1행 값처럼 붙어 버리므로 "{표 제목} — {열 이름}: {값}"
    을 따로 만든다. 실측(09-18): "예산 총액은?" 에 1행의 세부 내역과 섞여 답하거나 셋째 행을 읽어 줬다."""
    if len(rows) < 3 or not labels:
        return []
    out = []
    for i, label in enumerate(labels):
        if i == 0 or not label:
            continue
        col = [r[i] if i < len(r) else "" for r in rows]
        if col[0] and not any(col[1:]) and len(col[0]) <= FACT_CELL_SPLIT_LEN:
            out.append((f"{heading} — " if heading else "") + f"{label}: {col[0]}")
    return out


_LOOSE_CELL_MAX_LEN = 80
_VALUE_UNIT = re.compile(r"\d[\d,.]*\s*(?:~\s*\d[\d,.]*\s*)?([가-힣%㎡]{1,4})")


def _value_units(cell: str) -> set[str]:
    """칸에 든 '숫자+단위' 의 단위 집합 ("50 ~ 150백만원" → {백만원}, "30%" → {%})."""
    return set(_VALUE_UNIT.findall(cell))


def _merge_loose_rows(lines: list[str]) -> list[str]:
    """표 바로 뒤에 낱줄로 새어 나온 행을 마크다운 행으로 되돌린다.

    실측(09-22 골든 a23·수동 #253·#293, AICA p8): find_tables 가 지원사업 표의 첫 행만 표로 잡고 둘째 행
    ("AI 제품·서비스 상용화" / "100 ~ 250백만원" / "기존 제품 고도화·실증·국내외 사업화")은 표 뒤 낱줄 3개로 남겼다.
    행 조각이 첫 행(시제품 50~150)뿐이라 [핵심 근거]·폴백 인용이 모두 그 행을 줬고, "상용화 지원 규모?" 에 틀린 행을
    "자료에는 이렇게 적혀 있습니다" 로 읽어 줬다.

    보수적으로만 붙인다: 표의 데이터 행이 1개 이상 있고, 표가 끝난 직후 비표 낱줄이 **정확히 열 수만큼** 이어지며(각 줄
    _LOOSE_CELL_MAX_LEN 자 이하, 문장 부호로 끝나지 않음), 그중 한 칸이 마지막 행의 같은 열과 **같은 숫자 단위**
    (백만원·%·명…)를 가질 때. 한 번 붙이면 같은 조건으로 다음 낱줄들도 이어 본다."""
    out: list[str] = []
    i = 0
    header: list[str] | None = None
    last_row: list[str] | None = None
    while i < len(lines):
        s = lines[i].strip()
        if TABLE_LINE.match(s):
            cells = _split_row(s)
            if not _is_sep_row(cells):
                if header is None:
                    header = cells
                else:
                    last_row = cells
            out.append(lines[i])
            i += 1
            continue
        if header and last_row and len(header) >= 2 and s:
            n = len(header)
            cand = [l.strip() for l in lines[i:i + n]]
            ok = (
                len(cand) == n
                and all(c and not TABLE_LINE.match(c) and len(c) <= _LOOSE_CELL_MAX_LEN and not c.endswith((".", "다", "요", ":")) for c in cand)
                # 첫 칸은 행 이름이어야 한다 — 번호·글머리표로 시작하거나 "지원 대상: …" 처럼 콜론이 든 설명 줄은 행이 아니다
                and not re.match(r"^[\d•▪◦·∙\-*]", cand[0]) and ":" not in cand[0] and "：" not in cand[0]
                and any(_value_units(a) and _value_units(a) == _value_units(b) for a, b in zip(cand, last_row))
            )
            if ok:
                out.append("| " + " | ".join(cand) + " |")
                last_row = cand
                i += n
                continue
        if s:
            header, last_row = None, None   # 표가 끝났다 (빈 줄은 표를 끊지 않는다 — 다음 낱줄도 흡수 후보)
        out.append(lines[i])
        i += 1
    return out


def table_facts(text: str) -> list[str]:
    """페이지 텍스트의 마크다운 표를 행 단위 한 줄 사실로 편다. 표가 없으면 빈 리스트."""
    facts: list[str] = []
    seen: set[str] = set()
    heading = ""
    header: list[str] | None = None
    labels: list[str] = []
    rows: list[list[str]] = []   # 현재 표의 데이터 행 (병합 열 조각용)
    prev_head = ""               # 바로 위 행의 머리 칸 (세로 병합으로 비어 온 머리 칸을 채운다)
    seen_data = False            # 현재 표에서 데이터 행을 처리했는가 (새 표 머리 감지용 — rows 에 안 쌓이는 행도 포함)

    def add(fact: str) -> None:
        fact = fact[:FACT_MAX_LEN]
        if fact not in seen:
            seen.add(fact)
            facts.append(fact)

    def close_table() -> None:
        for f in _merged_column_facts(heading, labels, rows):
            add(f)
        rows.clear()

    lines = _merge_loose_rows(text.split("\n")) + [""]
    for li, line in enumerate(lines):
        s = line.strip()
        if header is not None and 0 < len(s) <= 2:
            # 표 행 사이에 낀 한두 글자 줄("부", "록" — 페이지 옆 세로 라벨이 글자 단위로 흩어진 것)은 표를 끊지 않는다.
            # 실측(09-18): 배치도 표가 "록" 에서 둘로 갈려 2층 행이 새 표의 헤더가 됐다.
            continue
        if not TABLE_LINE.match(s):
            close_table()
            header, prev_head = None, ""
            if s and len(s) <= FACT_HEADING_MAX_LEN:
                cand = _HEADING_NUMBERING.sub("", s)
                # 표 바로 위의 한두 글자 줄("현황", "록")은 페이지 옆 세로 라벨이지 제목이 아니다 — 그 위의 제목 줄을 유지한다.
                # 실측(09-21): 설문 분석 표의 제목이 "2026학년도 교육계획 수립을 위한 설문 분석" 대신 "현황" 이 됐다.
                if len(cand) > 3 or len(heading) <= 3:
                    heading = cand
            continue
        cells = _split_row(s)
        if _is_sep_row(cells):
            continue
        if header is not None and seen_data and li + 1 < len(lines) and _is_sep_row(_split_row(lines[li + 1].strip())):
            # 다음 줄이 구분선(| --- |)이면 이 행은 앞 표의 데이터가 아니라 **새 표의 머리**다 — 표 사이에 빈 줄이 없으면
            # 한 표로 이어 읽힌다. 실측(09-21 키오스크): p19 특색·중점사업 표 뒤에 붙은 "| | 학교장 경영관 | | |" 표가
            # 앞 표의 행이 되어 학교상·학생상·교사상 조각에 앞 표 제목이 통째로 붙었다.
            close_table()
            header, prev_head = None, ""
        if header is None:
            seen_data = False
            filled = [c for c in cells if c]
            if len(filled) == 1:
                # 칸 하나만 찬 헤더 = 표 제목 (조직도의 "| 교무기획부 |  |"). 열 이름은 없다
                heading, header, labels = filled[0], [], []
                continue
            if _FLOOR_HEAD.match(cells[0]):
                # 첫 칸이 "2층"·"지하1층" 인 머리 행 = 배치도의 층 행이다. 층마다 구분선이 끼어 표가 층 단위로 갈라지면 층 행이
                # 새 표의 머리가 되어 (1) 그 행의 시설이 열 이름으로 사라지고 (2) 이어지는 머리 칸 빈 행(같은 층의 둘째 줄)이
                # 층 이름을 못 받는다. 실측(09-22 골든 m21·수동 #21·#95): p123 "2층" 둘째 줄의 도서실이 층 없이 색인돼
                # 모델이 다음 표의 1층으로 답했다(같은 줄의 교무실은 2층으로 맞춰 흔들림). 열 이름 없는 표로 두고 이 행부터
                # 데이터 행으로 편다 → "2층: 과학실 1, …" / 둘째 줄은 prev_head 로 "2층: 과학준비실, 계단, 도서실, …".
                header, labels = [], []
            else:
                header, labels = cells, _header_labels(cells)
                continue
        # 헤더와 칸 수가 다른 행(배치도처럼 줄마다 칸 수가 다른 표)은 열 이름 없이 값만 편다 — 예전엔 버렸는데,
        # 실측(09-18): 교사 배치도 p123 의 층별 행이 전부 빠져 "교무실은 몇 층?" 근거 조각이 없었다.
        if header and not rows and _is_subheader(header, cells):
            labels = _merge_header_labels(header, cells)
            continue
        if header and rows and header[0] and cells[0] == header[0] and _is_repeated_header(cells):
            # 표 안에서 머리 행이 다시 나오는 표(열이 많아 두 단으로 접은 시설표: "구분 | 다목적실 | …" / "실수 | 1 | …") —
            # 이후 행은 새 머리의 열 이름으로 읽는다. 실측(09-21): 둘째 단의 값이 첫 단 이름에 붙어 "회의실 2"(실제는 통합교육반 2)
            # 같은 틀린 조각이 생겼다.
            close_table()
            header, labels = cells, _header_labels(cells)
            continue
        seen_data = True
        if header == [] and sum(1 for c in cells if c) >= 2 and all(len(c) > FACT_LABEL_MAX_LEN for c in cells if c):
            # 제목만 있는 표(머리 "| | 학교장 경영관 | |")의 한 행에 긴 칸이 여럿 = 칸마다 독립된 항목(학교상·학생상·교사상).
            # 첫 칸을 행 이름으로 삼으면 "학교…: 학생…, 교사…" 로 뒤섞인다 — 칸마다 "제목 — 칸" 조각으로.
            # 칸 머리의 자간 벌린 낱말("학 교 발전하는 학교")은 붙이고 그 낱말을 라벨로 쓴다 → "학교장 경영관 — 학교: 발전하는 학교 …"
            for v in cells:
                if not v:
                    continue
                m = _SPACED_LEAD.match(v)
                if m:
                    v = m.group(1).replace(" ", "") + ": " + v[m.end():].strip()
                add((f"{heading} — " if heading else "") + v)
            continue
        aligned = not header or len(cells) == len(header)
        row_labels = labels if aligned else []
        if not cells[0] and prev_head and any(cells[1:]):
            # 머리 칸이 비고 나머지가 찬 행 = 위 행의 머리 칸이 세로 병합된 것("2층" 배치도의 둘째 줄, 부서별 업무표).
            # 실측(09-18): "교무실은 몇 층?" — 교무실이 든 행에 층 이름이 없어 1층으로 답했다.
            cells = [prev_head] + cells[1:]
        if aligned:
            rows.append(cells)
        if aligned and header and _ordinal_header(header):
            # 행 이름 열이 없는 순번 표(_ordinal_header): 행 전체 조각 "1호 서면사과, 2호 …" 와 칸별 조각 "1호: 서면사과" 를
            # 만든다 — 칸별 조각이 있어야 "1호는?" 에 그 칸만 근거로 뽑히고, 순번 대조(verify.ordinal_mismatch)도 선다
            pre = f"{heading} — " if heading else ""
            pairs = [(h, v) for h, v in zip(header, cells) if v]
            if pairs:
                add(pre + ", ".join(f"{h} {v}" for h, v in pairs))
                for h, v in pairs:
                    add(pre + f"{h}: {v}")
            continue
        if not any(cells[1:]):
            if cells[0] and len(cells[0]) <= FACT_HEADING_MAX_LEN:
                heading = cells[0]  # 표 안의 소제목 행 ("| 도덕·사회·역사 |  |")
                continue
            if cells[0]:
                # 첫 칸에만 긴 내용이 든 행(연혁 표: 셀 병합으로 전부 첫 칸에 들어옴) — 항목별 조각으로.
                # 제목은 표 헤더의 마지막 라벨("주요 연혁")을 우선한다 — 표 위 줄은 페이지 옆 라벨("교 가")일 수 있다
                label = next((h for h in reversed(header or []) if h), "") if header else ""
                pre = f"{label or heading} — " if (label or heading) else ""
                for it in _split_cell(cells[0]):
                    add(pre + it)
            continue
        head = cells[0]
        parts: list[str] = []
        items: list[str] = []
        skip: set[int] = set()
        # 머리 칸이 두 열에 걸친 표("구분 | (빈) | 교장 | …" 아래 "현원 | 남 | 1 | …"): 열 이름 없는 앞쪽 글자 칸은 행 이름의 일부다
        # → "현원 남: 교장 1, …". 실측(09-21): "현원: 남, 교장 1, …" 로 성별이 값처럼 떠 있었다.
        for i in range(1, len(cells)):
            lab = row_labels[i] if i < len(row_labels) else None
            v = cells[i]
            if lab == "" and head and v and not _NUMERIC_CELL.match(v) and len(v) <= 4:
                head = f"{head} {v}"
                skip.add(i)
            else:
                break
        if head and (_NUMERIC_CELL.match(head) or _UNIT_VALUE_CELL.match(head)) and row_labels and row_labels[0] and not _calendar_days(cells):
            # (달력 주(週) 행은 예외 — 첫 칸의 월 숫자가 행 이름이어야 rag 쪽 calendar_row_to_days 가 "5: …" 를 읽는다.
            #  실측 09-21: 이 예외가 없어 "개교는 언제야?" 폴백이 주 행 전체를 읽어 줬다)
            # 첫 칸이 값인 행(교지 면적 "1,213.29㎡ | 13,049.01㎡ | …", 진학 현황 "86 | 22 (1) | …") — 첫 칸은 행 이름이
            # 아니라 첫 열의 값이다. 실측(09-21): "교지 면적 — 1,213.29㎡: 운동장부지 …" 로 머리칸이 값으로 밀려 읽혔다.
            parts.append(f"{row_labels[0]} {head}")
            head = ""
        prev_head = head or prev_head
        day_facts = _calendar_day_facts(heading, cells)  # 달력 행이면 날짜별 조각도 (제목이 이미 붙어 있음)
        for i, v in enumerate(cells[1:], start=1):
            if not v or i in skip:
                continue
            label = row_labels[i] if i < len(row_labels) else ""
            if len(v) > FACT_CELL_SPLIT_LEN or len(_BULLET_SPLIT.findall(v)) >= 2:
                items += _split_cell(v)
            else:
                parts.append(f"{label} {v}" if label and label != head else v)
        prefix = (f"{heading} — " if heading else "") + (f"{head}: " if head else "")
        # 긴 칸을 항목으로 쪼갤 때 같은 행의 짧은 칸(대상·담당자 등)을 항목마다 앞에 붙인다 — 실측(09-21 키오스크): 설문 분석 표
        # "교과 활동 | 학부모 | ∙…84%…" 의 항목 조각이 "교과 활동: ∙…84%" 로만 남아 응답 대상이 사라졌고, 모델이 학부모 84% 를
        # 학생 만족도로 답했다.
        item_prefix = prefix + (", ".join(parts) + " — " if parts and items else "")
        for fact in [prefix + ", ".join(parts)] * bool(parts) + [item_prefix + it for it in items] + day_facts:
            add(fact)
    return facts


def _line_start(text: str, pos: int) -> int:
    return text.rfind("\n", 0, pos) + 1


def _table_header_at(text: str, pos: int) -> str:
    """pos가 마크다운 표의 헤더가 아닌 행에 있으면 그 표의 헤더(+구분선)를 돌려준다. 아니면 ''."""
    ls = _line_start(text, pos)
    le = text.find("\n", pos)
    le = len(text) if le < 0 else le
    if not TABLE_LINE.match(text[ls:le]):
        return ""
    above: list[str] = []
    s = ls
    while s > 0:
        ps = _line_start(text, s - 1)
        prev = text[ps:s - 1]
        if not TABLE_LINE.match(prev):
            break
        above.append(prev)
        s = ps
    if not above:
        return ""  # 이 줄이 표의 첫 줄(헤더)
    above.reverse()
    header = above[0]
    if len(above) >= 2 and TABLE_SEP_LINE.match(above[1]):
        return header + "\n" + above[1] + "\n"
    return header + "\n"


def find_boundary(text: str, hard_end: int) -> int:
    """hard_end 근처에서 문장 끝이나 공백을 찾아 단어 중간이 아닌 자연스러운 위치에서 자른다."""
    if hard_end >= len(text):
        return len(text)

    window_start = max(0, hard_end - BOUNDARY_SEARCH_WINDOW)
    window = text[window_start:hard_end]

    best = -1
    for m in SENTENCE_ENDING.finditer(window):
        best = window_start + m.end()
    if best > 0:
        return best

    space = text.rfind(" ", window_start, hard_end)
    if space > 0:
        return space + 1

    return hard_end


def chunk_text(text: str) -> list:
    text = text.strip()
    if not text:
        return []
    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        hard_end = min(start + CHUNK_SIZE, len(text))
        end = find_boundary(text, hard_end)
        if end <= start:
            end = hard_end

        piece = text[start:end].strip()
        if piece:
            chunks.append(_table_header_at(text, start) + piece)

        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
        # 겹침 시작점이 표 행 중간이면 그 행의 처음으로 당긴다 — 행이 반 토막 나지 않게
        if TABLE_LINE.match(text[_line_start(text, start):start + 1]):
            start = _line_start(text, start)

    return chunks


# ── 응답 대상 라벨 ───────────────────────────────────────────────────
# 설문·통계 문서는 "[교사용]", "[고객용]", "[임직원용]" 처럼 대상 표식 뒤에 같은 문항 구조의 표가 여러 쪽 이어진다. 표식은
# 첫 쪽에만 있어 둘째 쪽 조각엔 대상이 남지 않고, 대상만 다른 같은 문항의 수치("만족도 61.5% / 70.8%")를 리랭커·모델이
# 섞었다(실측 09-19, 1000문항 중 9건). 표식 "[…용]" 은 문서 종류를 가리지 않고 잡고, 라벨은 다음 표식이 나오거나
# 설문 내용(퍼센트·설문·문항·응답 낱말)이 끊기는 페이지에서 지운다 — 특정 문서의 장 이름에 기대지 않는다.
_SURVEY_MARK = re.compile(r"\[\s*([가-힣A-Za-z·/ ]{1,12}?)\s*용\s*\]")
_SURVEY_BODY = re.compile(r"%|설문|문항|응답")


def survey_target(text: str, current: str) -> str:
    """이 페이지에 적용할 응답 대상 라벨("교사 설문" 등). 표식이 있으면 그것으로 바꾸고, 설문 내용이 없는 페이지면 지우고,
    아니면 이전 값을 잇는다."""
    m = _SURVEY_MARK.search(text)
    if m:
        return f"{m.group(1).strip()} 설문"
    if current and not _SURVEY_BODY.search(text):
        return ""
    return current


def main() -> None:
    if not INPUT_DIR.exists():
        logger.error("1단계 출력 폴더를 찾을 수 없습니다: %s", INPUT_DIR)
        logger.error("먼저 load_documents.py를 실행하세요.")
        return

    doc_files = sorted(INPUT_DIR.rglob("*.json"))
    if INDEX_ONLY:
        doc_files = [p for p in doc_files if p.relative_to(INPUT_DIR).parts[0] in INDEX_ONLY]
        logger.info("부분 인덱싱: 폴더 %s 의 문서 %d개만 청킹", INDEX_ONLY, len(doc_files))
    if not doc_files:
        logger.warning("%s 안에 JSON 파일이 없습니다. load_documents.py를 먼저 실행하세요.", INPUT_DIR)
        return

    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    total_chunks = 0
    total_facts = 0
    chunk_lengths = []

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for doc_path in doc_files:
            with open(doc_path, encoding="utf-8") as f:
                doc = json.load(f)

            source_file = doc["source_file"]
            # 문서 제목을 조각 앞에 붙인다 — 짧은 조각도 어느 매뉴얼의 내용인지
            # 임베딩에 반영되어 검색 정확도가 올라가고, LLM 프롬프트에서도 출처가 분명해진다
            doc_title = Path(source_file).stem
            # chunk_id는 폴더(기관)까지 포함한 상대경로로 만든다 — 파일명 stem만 쓰면
            # 서로 다른 기관 폴더에 동명 파일이 있을 때 chunk_id가 충돌해 덮어써진다
            doc_key = Path(source_file).with_suffix("").as_posix().replace("/", "_")
            doc_chunk_count = 0

            doc_fact_count = 0
            footer_keys = repeated_lines(doc["pages"])  # 가장자리 반복 텍스트만 산문 조각에서 제외
            survey_label = ""   # "[교사용]" 같은 설문 대상 표식 — 다음 표식이 나올 때까지 이어지는 페이지에 붙인다
            for page in doc["pages"]:
                survey_label = survey_target(page["text"], survey_label)
                tag = f"({survey_label} 결과) " if survey_label else ""
                page_text = linearize_calendar_tables(page["text"])
                if survey_label:
                    page_text = f"※ 이 페이지는 {survey_label} 결과다 — 다른 응답 대상의 수치와 섞지 말 것\n" + page_text
                for i, piece in enumerate(chunk_text(page_text)):
                    record = {
                        "chunk_id": f"{doc_key}_p{page['page_number']}_c{i}",
                        "source_file": source_file,
                        "page_number": page["page_number"],
                        "chunk_index": i,
                        "kind": "page",
                        "text": f"[{doc_title}]\n{piece}",
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    doc_chunk_count += 1
                    chunk_lengths.append(len(piece))
                # 표 행 사실 조각 + 도해 설명 줄 조각 + 산문 조각 — 검색·재채점 전용(컨텍스트는 페이지 전문으로 확장된다)
                for i, fact in enumerate(table_facts(page["text"]) + diagram_facts(page["text"])
                                         + prose_facts(page["text"], footer_keys)):
                    record = {
                        "chunk_id": f"{doc_key}_p{page['page_number']}_f{i}",
                        "source_file": source_file,
                        "page_number": page["page_number"],
                        "chunk_index": i,
                        "kind": "fact",
                        "text": f"[{doc_title}]\n{tag}{fact}",
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    doc_fact_count += 1

            total_chunks += doc_chunk_count
            total_facts += doc_fact_count
            logger.info("%s -> 페이지 조각 %d개 + 표 행 조각 %d개", source_file, doc_chunk_count, doc_fact_count)

    avg_len = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0

    logger.info("-" * 50)
    logger.info(
        "완료: %d개 문서 -> 페이지 조각 %d개 (평균 %.0f자, 최소 %d자, 최대 %d자) + 표 행 조각 %d개",
        len(doc_files), total_chunks, avg_len,
        min(chunk_lengths) if chunk_lengths else 0,
        max(chunk_lengths) if chunk_lengths else 0,
        total_facts,
    )
    logger.info("결과 저장 위치: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
