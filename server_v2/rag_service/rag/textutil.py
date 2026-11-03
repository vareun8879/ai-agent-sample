"""조각 텍스트 공용 유틸."""

import re

# 조각 맨 앞의 문서 제목 줄 "[2026학년도 망미중학교 교육계획서(탑재용)]" (indexing/chunk_documents 가 붙임)
_TITLE_LINE = re.compile(r"^\[[^\]\n]{1,120}\]\n")


# 통계표의 "255(5)", "11(2)" — 숫자 바로 뒤 괄호 안 숫자(부가 값, 문서에 뜻이 없음). 음성 답변에서는 뗀다.
# 낱말 뒤 괄호 안이 숫자 범위뿐인 "학생회선거(5~6)", "진로캠프(1~6)" 도 같다 — 실측(09-18): 모델이 "(5~6)" 을
# "5학년과 6학년" 으로 지어냈다. 뜻이 표기상 정해지지 않은 값이라 답에서 뺀다.
# "입학식(10학급, 651명)", "5/1(금9)", "2026.3.3.(화)" 처럼 괄호 안이 숫자만이 아닌 것은 건드리지 않는다.
# 콤마형 "학생회선거(5,6)" 도 같다 — 실측(09-22 수동 #58): 모델이 "5학년과 6학년을 대상으로" 라고 지어냈다(중학교에 5·6학년은
# 없다). 프롬프트에 넣기 전(prompt.build_prompt)에도 지워 모델이 아예 보지 못하게 한다.
_PAREN_NUM = re.compile(r"(?<=\d)\s*\(\d{1,4}\)|\s*\(\d{1,2}\s*[~\-,]\s*\d{1,2}\)")


def strip_paren_numbers(text: str) -> str:
    """모델 답변의 '숫자(숫자)' 에서 괄호 부분을 뗀다 — 실측(09-17): 규칙을 줘도 7B 가 "11(2)개" 로 그대로 옮기거나
    "괄호의 5는 학급 수" 처럼 뜻을 지어냈다. 문서에 뜻이 없는 값이라 답에서 빼는 것이 안전하다."""
    return _PAREN_NUM.sub("", text)


_HANGUL_TOKEN = re.compile(r"[가-힣]{2,}")
# 질문의 숫자("17대", "1호", "1365")도 낱말이다 — 실측(09-18): 숫자를 안 세니 초대~17대 취임 행, 1호~9호 조치 행이
# 전부 같은 점수라 정답 행이 뒤로 밀렸다. 줄에서는 다른 숫자의 일부가 아닌 온전한 수로만 맞춘다.
_NUM_TOKEN = re.compile(r"\d+")
# "총액·합계" 류 질문 낱말은 표에서 "계/합계/총계" 칸으로 적힌다 — 낱말 자체가 안 겹쳐도 같은 뜻으로 친다.
# 실측(09-18): "예산 총액은?" 폴백이 "총액(원)" 칸이 없는 셋째 행을 골랐다. 문서 특정 값이 아니라 한국어 합계 표기 일반.
_TOTAL_Q = ("총액", "총계", "합계", "총합", "전체", "모두", "전부", "다해서", "합쳐서", "통틀어", "총원", "총인원")
# 행 머리(줄 처음 또는 "제목 — " 뒤)가 계/합계/총액… 인 줄만 — 행 속의 "계 83" 칸까지 세면 학년별 행이 진짜 계 행을 이긴다
# (실측 09-18 m17: "1학년: 배산초 78, 기타 5, 계 83" 이 "계: 배산초 249, 기타 7, 255" 를 밀어냈다)
_TOTAL_LINE = re.compile(r"(?:^|— )(?:합계|총계|총합|총원|계|총액(?:\([^)]*\))?)\s*:")


# 질문에서 낱말로 세지 않는 것 — 실측(09-20): 같은 뜻의 질문이 표현에 따라 다른 결과를 냈다.
#   "교직원 수는?" → 43명 / "전체 교직원 수는?" → 거절 / "망미중학교 전체 교직원 수는?" → 6쪽이 검색돼 인용문 나열.
#   "전체"·"수는" 은 어느 줄에도 없어 거절 재질문 조건(낱말 2개 겹침)을 못 채웠고, "망미중학교" 는 문서 제목이라
#   모든 조각에 있어 낱말 일치 후보를 엉뚱한 쪽(p3 연혁·p61…)으로 벌렸다.
#   (1) 군더더기 낱말: 양·범위·의문·요청 표현. 합계 보너스(_TOTAL_Q)는 별도로 유지된다.
#   (2) 문서 제목 낱말: retriever 가 load() 때 색인된 문서 제목("2026학년도 망미중학교 교육계획서")에서 뽑아 넣는다.
#       질문 낱말이 제목 낱말의 앞부분("망미중" ⊂ "망미중학교")이면 제외.
STOP_TOKENS = frozenset((
    "전체", "전부", "모두", "총", "총원", "다해서", "합쳐서", "통틀어",
    "대해", "대해서", "대한", "관련", "관해", "관한", "경우", "정도",
    "알려줘", "알려주세요", "알려", "말해줘", "말해", "가르쳐줘", "궁금해", "궁금합니다", "설명해줘", "설명",
    "뭐야", "뭐예요", "뭐에요", "뭐지", "뭔지", "뭐냐", "뭐니", "무엇", "무엇인가요", "무엇이지", "무엇이야", "무엇이냐", "무엇인지", "어떻게", "어떤", "언제", "어디", "어디야", "누구", "누구야", "얼마", "얼마야",
    "있어", "있어요", "있나요", "있습니까", "인가요", "이야", "인지", "해줘", "해주세요", "주세요", "좀", "그럼", "그리고", "우리",
    "몇", "몇시", "몇명", "며칠", "언제야", "언제예요", "얼마나", "얼마나요",
    "수는", "수가", "수를", "수도", "수야", "명이", "명은", "명이야", "명인가요", "개야", "개는", "개인가요",
    # 정도·태도 부사 — 실측(09-21 키오스크): "aTops가 정확히 무엇이지" 의 '정확히' 가 스쿨패스 "가장 정확하게" 와 겹쳐
    # 무관 문서 2쪽이 근거에 끼고 10초가 걸렸다
    "정확히", "정확하게", "정확한", "자세히", "자세하게", "자세한", "간단히", "간단하게", "구체적으로", "대략", "대충",
    "정말", "진짜", "혹시", "그냥", "다시", "제발", "빨리",
))
# 영문 낱말(aTops, BASS, PAPS, Wee) — 대소문자를 가리지 않고 낱말 전체로만 맞춘다. 실측(09-21): "aTops가 정확히 무엇이지" 는
# 한글 낱말이 군더더기뿐이라 질문의 진짜 주어(aTops)가 검색·폴백 점수에 들어가지 않았다.
_EN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]{1,}")


def _is_en(t: str) -> bool:
    return bool(t) and t[0].isascii() and t[0].isalpha()
TITLE_TOKENS: set[str] = set()   # retriever.load() 가 채운다


def _is_title_token(t: str) -> bool:
    return len(t) >= 2 and any(tt.startswith(t) for tt in TITLE_TOKENS)


def question_tokens(question: str, drop_stop: bool = True) -> list[str]:
    """질문의 낱말(한글 2자+, 숫자) — 중복은 한 번만. 실측(09-18): 질문+가짜 인용을 합쳐 힌트로 쓸 때 "제품·지원" 이
    세 번씩 들어가 그 낱말이 든 옆 행이 정답 줄을 이겼다.
    drop_stop=True(기본)면 군더더기 낱말·문서 제목 낱말을 뺀다. 다 빠지면(질문이 "전체 알려줘" 뿐) 원래 낱말을 돌려준다."""
    raw = list(dict.fromkeys(_HANGUL_TOKEN.findall(question) + _NUM_TOKEN.findall(question) + _EN_WORD.findall(question)))
    if not drop_stop:
        return raw
    kept = [t for t in raw if t not in STOP_TOKENS and not _is_title_token(t)]
    return kept or raw


# 사람을 묻는 질문의 군더더기 — "교장 선생님 성함이 뭐야?" 에서 문서 줄에 그대로 있을 리 없는 낱말(선생님·성함·누구·담당자).
# 낱말 구제(reranker)가 "질문 낱말 전부가 낱말 그대로 든 줄" 을 요구할 때 이 낱말은 세지 않는다 — 실측(09-22 골든 m09·m10):
# p26 조직도 줄 "교장 허00은 학교의 최고 책임자이다." 에 '교장' 만 있고 '선생님·성함' 은 없어 구제가 안 됐다(3회 연속).
_PERSON_Q_PREFIX = ("선생님", "선생", "쌤", "성함", "이름", "누구", "누가", "담당자", "담당", "분이", "분은", "분의")


def subject_tokens(question: str) -> list[str]:
    """질문 낱말에서 사람을 묻는 군더더기(_PERSON_Q_PREFIX 로 시작하는 낱말)를 뺀 것 — 답의 주어("교장", "보건", "학적")."""
    toks = question_tokens(question)
    core = [t for t in toks if not t.startswith(_PERSON_Q_PREFIX)]
    return core or toks


def search_question(question: str, scope_titles: "list[set[str]] | None" = None) -> str:
    """임베딩·리랭커에 넣을 질문 — 문서 제목 낱말(기관명 등)만 지운다. 실측(09-20): "망미중학교 전체 교직원 수는?" 은
    제목 낱말 때문에 임베딩이 연혁(p3)·조직표(p28·61)·설문(p119) 등 6쪽을 끌어와 답변이 느려졌다("교직원 수는?" 은 1쪽).
    군더더기 낱말은 문장의 뜻(전체·언제)을 담고 있으므로 임베딩 질문에서는 지우지 않는다. 지울 게 없으면 원문 그대로.

    scope_titles(검색 범위에 든 문서들의 제목 낱말 집합 목록)를 주면 **모든 문서 제목에 다 있는 낱말만** 지운다 —
    범위에 문서가 여럿이면 기관명은 어느 문서를 볼지 가르는 낱말이라 남겨야 한다. 실측(09-20, 1000문항 v2):
    공통 문서(A학교 계획서) + 기관 문서(B학교 소개서) 범위에서 "B학교 교훈이 뭐야?" 의 "B학교" 를 지우자
    A학교 교훈을 답했다(2건). 문서가 하나뿐인 범위에서는 예전처럼 다 지운다.

    제목에 한글 낱말이 없는 문서(파일명이 "2026 SCHOOL-PASS" 처럼 영문·숫자뿐)도 **범위의 한 문서로 센다** —
    실측(09-23 골든 b26·b27): 빈 제목 집합을 후보에서 빼는 바람에 교집합이 배재고 제목 낱말 전체가 되어
    "배재고등학교 대표 전화번호가 뭐야?" 가 "대표 전화번호가 뭐야" 로 검색됐고, 배재고 p12 가 공통 문서 페이지들에
    밀려(4·5위) 답이 "찾을 수 없음" 이 됐다. 빈 집합이 하나라도 있으면 교집합은 공집합 = 아무것도 안 지운다."""
    if not TITLE_TOKENS:
        return question
    if scope_titles is None:
        strip = set(TITLE_TOKENS)
    else:
        strip = set.intersection(*scope_titles) if scope_titles else set()
    if not strip:
        return question
    out = question
    for t in sorted(strip, key=len, reverse=True):
        if len(t) >= 3 and t in out:
            out = out.replace(t, " ")
    # "망미중" 처럼 제목 낱말의 앞부분만 쓴 표기: 질문 낱말 중 제목 낱말의 접두(3자 이상)인 것도 지운다
    for w in _HANGUL_TOKEN.findall(out):
        if len(w) >= 3 and any(tt.startswith(w) for tt in strip):
            out = out.replace(w, " ")
    out = re.sub(r"\s{2,}", " ", out).strip(" ,.?!")
    return out if len(out) >= 2 else question


# 설문·통계 사실 조각의 응답 대상 꼬리표 — 인덱싱(chunk_documents.survey_target)이 "(학부모 설문 결과) …" 처럼 붙인다.
# 라벨 자체는 문서의 "[○○용]" 표식에서 오므로 여기에 대상 이름을 적지 않는다. retriever.load() 가 색인에서 모아 채운다.
_SURVEY_TAG = re.compile(r"^\(([^()]{1,20}?) 설문 결과\)\s*")
SURVEY_GROUPS: set[str] = set()   # 색인에 실제로 있는 응답 대상 라벨("학부모", "교사"…)


_TARGET_CELL = re.compile(r"대상\s*[:：]?\s*([^,，—]{1,20})")


def survey_group(line: str) -> str:
    """줄의 응답 대상 라벨 — 머리 꼬리표 "(학부모 설문 결과)" 또는 표 칸 "대상 학부모". 없거나 여럿이면 빈 문자열.
    실측(09-21 키오스크): 설문 분석 요약표(대상 칸이 있는 행)의 학부모 84% 를 학생 만족도로 답했다."""
    m = _SURVEY_TAG.match(line)
    if m:
        return m.group(1).strip()
    if SURVEY_GROUPS:
        for t in _TARGET_CELL.finditer(line):
            found = [g for g in SURVEY_GROUPS if g in t.group(1)]
            if len(found) == 1:
                return found[0]
    return ""


def filter_survey_lines(question: str, lines: list[str]) -> list[str]:
    """질문이 특정 응답 대상(라벨 낱말)을 말하면 다른 대상의 설문 줄을 뺀다 — 실측(09-20, 1000문항 v2): "학생이 원하는
    선택교과 1위는?" 의 [핵심 근거]에 "(학부모 설문 결과) 진로와 직업 46.2%" 가 먼저 와서 학부모 값을 답했다(4건).
    질문에 대상이 없거나 여럿이면 그대로 둔다. 라벨 없는 줄은 항상 남긴다."""
    if not SURVEY_GROUPS or not lines:
        return lines
    mentioned = {g for g in SURVEY_GROUPS if any(part and part in question for part in re.split(r"[·/ ]+", g))}
    if not mentioned:
        return lines
    kept = [l for l in lines if not survey_group(l) or survey_group(l) in mentioned]
    return kept


_LIST_MARK = re.compile(r"(?<![\d,.])\d{1,2}\)|[⧕⧔]\s*\d+\s*$|^\s*\d{1,2}\.\s|\(\d\)")   # "1)", 쪽 표시 "⧕ 3", "1. ", 각주 "(1)"


_YEAR_MARK = re.compile(r"(?:19|20)\d{2}\s*(?:학년도|년도|년)?")
_DATE_Q = re.compile(r"언제|날짜|며칠|일정|기간|몇\s*월|몇\s*일")


def has_value_digit(line: str, question: str = "") -> bool:
    """줄에 값으로 볼 숫자가 있는가 — 목록 번호("1) 2) 3)")·쪽 번호·각주 번호는 값이 아니다. 날짜를 묻는 질문이 아니면
    연도("2025학년도")도 값이 아니다.
    실측(09-21 키오스크): "학생 수는?" 거절-재질문에 소제목 줄 "학생 현황 1) 학급 편제 및 학생수 2) … 3) 2025학년도 졸업생
    진학 현황 ⧕ 3" 이 '숫자 든 줄' 로 통과해 그 소제목을 읽어 줬다(5회 연속)."""
    body = _LIST_MARK.sub(" ", line)
    if not _DATE_Q.search(question or ""):
        body = _YEAR_MARK.sub(" ", body)
    return any(ch.isdigit() for ch in body)


def is_value_question(question: str) -> bool:
    """수·명·몇·얼마·언제·날짜·비율처럼 '값'을 묻는 질문 — 폴백으로 읽어 줄 줄에 숫자가 있어야 답이 된다.
    실측(09-20): "망미중학교 전체 교직원 수는?" 폴백이 숫자 없는 소제목 줄("교직원 현황 1) 교직원 정원 및 현원 …")을 읽어 줬다."""
    return bool(re.search(r"(수는|수가|수를|명이|명은|몇|얼마|언제|며칠|날짜|시간|비율|퍼센트|%|금액|예산|면적|인원|일수|기간|수량|개수|횟수)", question))


def question_overlap(question: str, line: str, toks: list[str] | None = None) -> int:
    """질문 낱말이 줄에 얼마나 겹치는지 — 낱말마다 **앞부분 일치 길이**(2자 이상)를 더한다.

    조사가 붙은 채 잘린 낱말("입학식은")은 "입학식" 이 있는 줄에 3점, "개학식" 이 있는 줄에는 0점
    (뒤쪽 조각 "학식" 으로 세면 개학식이 잡힌다 — 09-17 실측 오답). 총액·합계류 질문이면 "계/합계/총액" 칸이
    있는 줄에 2점 보너스. 폴백 줄 고르기(pick_evidence_line)와 핵심 근거 보강(expansion)이 같이 쓴다."""
    toks = question_tokens(question) if toks is None else toks
    pts = 0
    # 표 머리 칸이 두 줄로 나뉘어 "상담 교사" 처럼 띄어 붙는 낱말도 질문의 "상담교사" 와 같은 낱말이다 — 실측(09-21 키오스크):
    # "상담교사는 몇 명?" 에 정원표 행("… 상담 교사 1 …")이 2점, 위원회 행("… 상담교사, 학부모 2명 …")이 4점이라 위원회 행을 읽어 줬다.
    line_ns = line.replace(" ", "")
    for t in toks:
        if t.isdigit():
            if re.search(r"(?<!\d)" + t + r"(?!\d)", line):
                pts += len(t)
            continue
        if _is_en(t):
            if re.search(r"(?<![A-Za-z])" + re.escape(t) + r"(?![A-Za-z])", line, re.I):
                pts += len(t)
            continue
        for n in range(len(t), 1, -1):  # 긴 접두부터: "입학식은" → 입학식은, 입학식, 입학
            if t[:n] in line or t[:n] in line_ns:
                pts += n
                break
    if pts and any(t.startswith(w) for t in question_tokens(question, drop_stop=False) for w in _TOTAL_Q) and _TOTAL_LINE.search(line):
        pts += 2
    return pts


def pick_evidence_line(question: str, lines: list[str], hint: str = "", rank: dict[str, int] | None = None) -> str:
    """폴백으로 읽어 줄 문서 줄 하나 — 질문 낱말과 가장 많이 겹치는 줄. 동점이면 페이지 순위(rank: 줄 → 검색 순위,
    작을수록 앞)가 앞선 것, 그래도 같으면 짧은 것. 하나도 안 겹치면 ''.
    hint(모델이 쓴 문장들)를 주면 그 낱말도 세되 질문의 절반 무게 — 실측(09-18): 모델이 옆 행(2학년)을 보고 쓴 문장을
    질문과 같은 무게로 세니 옆 행이 계 행을 이겼다. 질문이 우선, 모델 문장은 동점을 가르는 정도.

    실측 배경: 검색 1위 줄을 그대로 쓰니 "추석 연휴는 며칠?" 에 달력 행 전체를 읽어 줬다; 같은 페이지의
    "9월 24일(목) 추석연휴" 가 질문 낱말을 담고 더 짧다 (달력 행은 calendar_row_to_days 가 날짜 칸으로 줄인다).
    동점 규칙은 두 실측이 정반대를 요구한다 — (09-19, 각 3회 연속)
      · "입학식은 언제야?": 근거 5줄 중 4줄이 '입학식' 으로 동점. 3위 페이지(연혁)의 "1982.03.05. 제1회 입학식" 이
        1위 페이지(학사일정)의 "입학식 및 시업식: 2026.3.3.(화)" 보다 두 글자 짧아 길이로 이겼다 → 페이지 순위가 맞다
      · "상용화 지원 규모?": 같은 p8 의 시제품 행(101자, 표 머리말 지원·규모·기업당 일치)과 상용화 행(47자, 모델 문장의
        기존·고도화·실증 일치)이 34:34. 줄 순서로 가르면 시제품 행이 이긴다 → 같은 페이지 안에선 짧은 쪽이 맞다
    그래서 페이지 순위(reranker 판단)를 먼저, 같은 페이지 안에서만 길이를 본다. 줄 순서는 쓰지 않는다.
    0점 줄은 돌려주지 않는다 — 실측(09-18): 겹치는 낱말이 없는데도 표 머리 행을 읽어 줬다. 그럴 땐 안내문이 낫다."""
    if not lines:
        return ""
    toks = question_tokens(question)
    # 모델 문장의 숫자는 빼고 센다 — 검증에 걸린 바로 그 값(옆 행의 80, ×10 한 2500)이라 옆 행을 가리키기 때문
    htoks = [t for t in question_tokens(hint) if t not in toks and not t.isdigit()] if hint else []
    rank = rank or {}

    def key(l: str):
        return (2 * question_overlap(question, l, toks) + (question_overlap(hint, l, htoks) if htoks else 0),
                whole_word_hits(l, toks), -rank.get(l, 0), -len(l))

    best = max(lines, key=key)
    if question_overlap(question, best, toks) == 0:
        return ""
    return calendar_row_to_days(best, toks) or best


def whole_word_hits(line: str, toks: list[str]) -> int:
    """질문 낱말이 줄에 **낱말 그대로**(앞뒤에 한글·숫자가 붙지 않게) 들어 있는 개수 — 동점 가르기용.
    실측(09-21): "개교는 언제야?" 에 "1982.03.01. 개교"(연혁) 와 "5월 4일(월) 개교기념일 재량휴업일" 이 접두 일치로 동점,
    짧은 쪽이 이겨 기념일을 읽어 줬다. '개교' 가 낱말로 끝나는 줄이 '개교기념일' 의 일부인 줄보다 질문의 항목이다.
    앞쪽도 본다 — 실측(09-21 키오스크): 뒤쪽만 보니 스쿨패스 문서의 "전국 50개교" 가 '개교' 낱말 일치로 잡혀 낱말 구제가
    연혁 행 대신 스쿨패스 페이지를 살렸다."""
    hits = 0
    line_ns = line.replace(" ", "")   # "상담 교사" 도 "상담교사" 낱말 — question_overlap 주석 참고
    for t in toks:
        if t.isdigit():
            continue
        if _is_en(t):
            if re.search(r"(?<![A-Za-z])" + re.escape(t) + r"(?![A-Za-z])", line, re.I):
                hits += 1
            continue
        for n in range(len(t), 1, -1):
            pre = t[:n]
            pat = r"(?<![가-힣0-9])" + re.escape(pre) + r"(?![가-힣])"
            if re.search(pat, line) or re.search(pat, line_ns):
                hits += 1
                break
            if pre in line or pre in line_ns:
                break   # 더 짧은 접두는 같은 자리의 부분 일치일 뿐
    return hits


_TABLE_LINE = re.compile(r"^\s*\|")
_TITLE_ANYWHERE = re.compile(r"\[[^\]\n]{1,120}\]")
_PROSE_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n{2,}")


def prose_sentences(page_text: str, min_len: int = 8, max_len: int = 200) -> list[str]:
    """페이지 전문에서 표가 아닌 본문을 문장 단위로 잘라 돌려준다 — 핵심 근거 줄(표 행)이 없을 때의 폴백 후보.
    실측(09-18): "학교 경계에서 몇 미터?" 근거가 표가 아닌 본문 문장("…직선거리 200미터의 범위…")이라 폴백 후보가
    없어 안내문으로 끝났다. 줄바꿈으로 잘린 문장은 이어 붙이고, 너무 짧거나(번호·제목 조각) 긴 것은 뺀다."""
    # 조각 경계마다 끼어 있는 문서 제목 줄 "[2026학년도 … 교육계획서]" 는 본문이 아니다 — 실측(09-18): 폴백이
    # "…운영계획 [2026학년도 망미중학교 교육계획서(탑재용)] Ⅳ." 를 읽어 줬다
    page_text = _TITLE_ANYWHERE.sub(" ", page_text)
    prose = " ".join(l.strip() for l in page_text.split("\n") if l.strip() and not _TABLE_LINE.match(l))
    out = []
    for sent in _PROSE_SPLIT.split(prose):
        sent = sent.strip()
        if min_len <= len(sent) <= max_len:
            out.append(sent)
    return out


# 달력 행 조각: "2학기 학사일정 — 9: 주 6, 21(월5), 22(화6), 23(수6), 24(목) 추석연휴, 25(금) 추석, 수업일수 3"
_CAL_ROW = re.compile(r"^(?:(?P<title>.*?) — )?(?P<month>\d{1,2}): (?P<body>.*)$")
_CAL_CELL = re.compile(r"^(?:(\d{1,2})/)?(\d{1,2})\(([월화수목금토일])\d*\)\s*(.*)$")


def calendar_row_to_days(line: str, question_tokens: list[str]) -> str:
    """달력 행 조각이면 질문 낱말이 든 날짜 칸만 "9월 24일(목) 추석연휴, 9월 25일(금) 추석" 로 줄인다.
    달력 행이 아니거나 맞는 칸이 없으면 ''. (indexing 의 날짜별 조각과 같은 규칙 — 검색에 그 조각이 못 들어와
    행 조각만 있을 때의 표현용.) 실측(09-17): 폴백이 행 전체를 음성으로 읽어 줬다."""
    m = _CAL_ROW.match(line.strip())
    if not m or not (1 <= int(m.group("month")) <= 12):
        return ""
    head_month = int(m.group("month"))
    scored = []  # (칸 점수 = 질문 낱말과 가장 길게 맞는 접두 길이, 날짜 문장)
    for cell in m.group("body").split(", "):
        c = _CAL_CELL.match(cell.strip())
        if not c:
            continue
        month, day, dow, event = c.groups()
        if not event:
            continue
        best = max((n for t in question_tokens for n in range(len(t), 1, -1) if t[:n] in event), default=0)
        if best:
            scored.append((best, f"{int(month) if month else head_month}월 {int(day)}일({dow}) {event.strip()}"))
    if not scored:
        return ""
    # 최고점 칸만 — "학부모총회는" 은 학부모총회(5) 만, 학부모상담주간(3) 은 제외. 추석연휴/추석은 같은 점수라 둘 다
    top = max(sc for sc, _ in scored)
    picked = [txt for sc, txt in scored if sc == top]
    title = m.group("title")
    return (f"{title} — " if title else "") + ", ".join(picked)


_SENT_SPLIT = re.compile(r"(?<=[.!?。])\s+")
_LABEL_ECHO = re.compile(r"^(?:답변|답|근거)\s*[은는:]\s*['\"‘“「].*['\"’”」]\s*(?:입니다|이다|임)?\s*[.!]?$")


def clean_answer(text: str) -> str:
    """음성 답변 최종 정리: 괄호 부가 숫자 제거 + 표 기호('|')가 든 문장 제거.
    실측(09-17): "보건실은 1층에 있습니다. 보건실은 1층에 있으며, 이는 문서에서 \"| 1층 | 가사실 | …\" 라는 부…"
    — 첫 문장이 답인데 둘째 문장에 표 행을 기호째 붙였다. 표 문장을 떼어도 답이 남으면 그렇게, 전부 표 문장이면
    기호만 쉼표로 바꾼다."""
    text = strip_paren_numbers(text)
    # 형식 라벨이 답에 새는 경우 — 실측(09-22 수동 #288): "… 3946입니다. 답변은 '사업 문의 전화번호는 …입니다'입니다." 처럼
    # 앞 문장을 '답변은 …' 으로 한 번 더 되풀이. 그 문장은 뗀다(앞 문장이 남으면).
    sents = _SENT_SPLIT.split(text.strip())
    kept = [s for s in sents if not _LABEL_ECHO.match(s)]
    if kept and len(kept) < len(sents):
        text = " ".join(kept).strip()
    if "|" not in text:
        return text
    sents = _SENT_SPLIT.split(text.strip())
    kept = [s for s in sents if "|" not in s]
    if kept:
        return " ".join(kept).strip()
    return re.sub(r"\s*\|\s*", ", ", text).strip(", ")


def strip_title(text: str) -> str:
    """조각 앞의 문서 제목 줄을 뗀다. 제목은 임베딩·출처 표시에는 필요하지만 reranker 입력과
    핵심 근거 표시에는 방해가 된다(실측: 짧은 지문의 reranker 확률이 4~10배 떨어짐)."""
    return _TITLE_LINE.sub("", text, count=1)


# ── 복합 질문 ────────────────────────────────────────────────────────────
# "A와 B는 각각 …?", "…몇 점이고, …몇 점부터야?" — 실측(09-22 수동 #94·#96·#97·#100·#195·#296): 두 항목을 물으면 모델이
# 앞 항목만 답하거나, 검증 실패 폴백이 한 줄만 골라 엉뚱한 줄(교과협의회 10월, 안전대책위원회)을 읽어 줬다.
# 폴백은 항목마다 줄을 하나씩 고르도록 질문을 둘로 나눈다. 표지가 없으면 나누지 않는다(오분할 방지).
_COMPOUND_MARK = re.compile(r"각각|둘 다|둘다|같이|함께|모두|두 가지|두가지")
_CONJ = re.compile(r"(?<=[가-힣A-Za-z0-9)])(?:와|과|랑|이랑|하고)\s+")
_CLAUSE = re.compile(r"(?<=[가-힣0-9])(?:이고|고),\s*")


def split_compound_question(question: str) -> list[str]:
    """복합 질문을 항목별 부분 질문 두 개로 나눈다. 복합이 아니면 []."""
    q = question.strip()
    m = _CLAUSE.search(q)
    if m and len(q[:m.start()]) >= 4 and len(q[m.end():]) >= 4:
        return [q[:m.start()], q[m.end():]]          # "…몇 점이고, 표창 추천은 몇 점부터야?"
    mk = _COMPOUND_MARK.search(q)
    if not mk:
        return []
    left, tail = q[:mk.start()], q[mk.end():]
    conj = list(_CONJ.finditer(left))
    if not conj:
        return []
    c = conj[-1]                                       # 마지막 접속 조사에서 나눈다 ("교과 활동과 봉사활동은" → 교과 활동 | 봉사활동은)
    a, b = left[:c.start()].strip(), left[c.end():].strip()
    if len(a) < 2 or len(b) < 2:
        return []
    return [f"{a} {tail}".strip(), f"{b} {tail}".strip()]  # 술어(몇 층에 있어?)는 양쪽에 붙인다 — 값 질문 판정에 쓰인다
