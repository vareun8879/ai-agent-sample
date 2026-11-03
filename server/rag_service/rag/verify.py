"""2.5단계 — 답변 사실 검증 (grounding check).

LLM 답변에 나온 **숫자·연도·날짜**가 근거 컨텍스트에 실제로 있는지 코드로 대조한다.
LLM 을 믿지 않고 프로그램이 검사하는 마지막 관문이다.

배경(실측 2026-09-17): "망미중학교는 언제 개교했어?" 에 근거 p3 에는 '1982.03.01. 개교' 가 또렷이
있는데 모델(A.X-4.0-Light, temperature 0)이 매번 "1980년 3월 1일" 이라고 답했다. 컨텍스트 어디에도
1980 은 없다 — 모델이 문서를 무시하고 자기 기억으로 지어낸 것이다. 민원 안내에서 "자신 있게 틀린
숫자" 는 "모릅니다" 보다 훨씬 나쁘므로, 근거에 없는 숫자가 답에 있으면 그 답을 내보내지 않는다.

검사 대상은 숫자만이다 — 이름·장소는 형태가 자유로워 오탐이 많고, 숫자(연도·날짜·인원·층·금액)는
질문의 핵심 사실이면서 문자열 대조가 정확하다.

한계: 컨텍스트에 우연히 같은 숫자가 다른 의미로 있으면 통과시킨다(예: 1980 이 다른 항목에 있는 경우).
그건 이 검사가 잡을 수 없는 종류이고, 그 방어는 프롬프트·골든셋·더 큰 모델의 몫이다.
"""

import logging
import re

from .config import VERIFY_NUMBERS

logger = logging.getLogger("rag.verify")

# 답변에서 뽑는 숫자 토큰: 4자리 연도, 그 외 1자리 이상 정수. 천 단위 구분(1,115,683)은 콤마를 뗀 형태로도 대조한다.
_NUM = re.compile(r"\d[\d,]*")
# 근거 없는 숫자가 있을 때 사용자에게 나가는 문구. NO_CONTEXT_ANSWER 와 구분해 "찾긴 했는데 확신이 없다" 를 전한다.
UNVERIFIED_ANSWER = "죄송합니다. 자료에서 확인이 필요한 내용이라 정확히 안내드리기 어렵습니다. 안내데스크에 문의해 주세요."


def _norm(s: str) -> str:
    return s.replace(",", "")


def _numbers(text: str) -> set[str]:
    out = set()
    for m in _NUM.finditer(text):
        n = _norm(m.group()).lstrip("0") or "0"  # 앞자리 0만 무시("03" 과 "3" 동일). 뒷자리 0은 값이다(1980 ≠ 198)
        out.add(n)
    return out


# 검사에서 제외하는 흔한 숫자: 1~2자리 소수(목록 번호·문장 속 '3문장'·'1회' 등)는 오탐이 잦고
# 사실 오류로 이어질 위험이 작다. 3자리 이상(연도·인원·금액·번호)만 본다.
def _significant(n: str) -> bool:
    return len(n) >= 3


# 단위가 붙은 수량("48명", "20개", "1층", "200미터", "40%")은 자릿수와 상관없이 질문의 핵심 값이다.
# 실측(09-18, 확장 골든): "1학년 학생은 몇 명?" 에 동아리표의 12명 네 줄을 더해 "48명" — 컨텍스트 어디에도 48 은 없다.
# 3자리 규칙이 놓친 2자리 합산 환각이라, 단위 뒤 수량은 1~2자리라도 근거에 그 수가 있어야 한다.
# 기간("2일간")·순서("3월")는 단위 목록에 없으므로 그대로 둔다(연휴 며칠 계산 답을 막지 않기 위해).
# 09-22 수동 #58·#100: "5학년과 6학년"(달력 칸 "(5,6)" 해석), "상점 1점"(정답 2점) — 학년·점도 질문의 핵심 값이라 대조한다
_UNIT_NUM = re.compile(r"(?<!\d)(\d{1,2})\s*(?:명|개|층|미터|퍼센트|%|원|회|학급|교시|대|건|점|학년)")


def unsupported_numbers(answer: str, context_text: str) -> list[str]:
    """답변에는 있는데 근거 텍스트에 없는 유의미한 숫자들 — 3자리 이상 전부 + 단위가 붙은 1~2자리 수량."""
    ctx_nums = _numbers(context_text)
    # 숫자 토큰 집합끼리만 대조한다 — 부분 문자열 대조("198" ⊂ "1982")는 환각을 통과시킨다
    bad = {n for n in _numbers(answer) if _significant(n) and n not in ctx_nums}
    bad |= {n.lstrip("0") or "0" for n in _UNIT_NUM.findall(answer) if (n.lstrip("0") or "0") not in ctx_nums}
    return sorted(bad)


_EN_ANSWER_WORD = re.compile(r"[A-Za-z]{3,}")
UNSUPPORTED_EN = "근거에 없는 영문"


def unsupported_english(answer: str, context_text: str) -> list[str]:
    """답변의 영문 낱말(3자 이상)이 근거에 없으면 그 낱말들 — 실측(09-21 키오스크): "aTops가 무엇이지" 에 모델이
    "AI Transformation Platform for Startups의 약자" 라고 지어냈는데 문서에는 "통합 플랫폼(aTOPS)" 뿐이었다.
    한국어 문서의 답에 영문이 나온다면 문서에 있는 고유명사(PAPS·Wee·English Cafe)를 옮긴 것이어야 한다."""
    ctx = context_text.lower()
    return sorted({w for w in _EN_ANSWER_WORD.findall(answer) if w.lower() not in ctx})


# URL·도메인 토큰 — 답변의 주소는 컨텍스트에 그대로 있어야 한다. 실측(09-22 수동 #205): "aTOPS 포털 주소가 뭐야?" 에
# "포털.atops.or.kr" — 모델이 portal 을 한글로 옮겨 적었다. 숫자·영문 검사는 이걸 못 잡는다(영문 낱말 검사는 3자 이상
# 영문 '낱말' 이라 "atops"·"or"·"kr" 은 각각 컨텍스트에 있다). 주소는 통째로 대조한다.
_URL_TOKEN = re.compile(r"(?i)(?<![A-Za-z0-9.])(?:https?://)?(?:[\w가-힣-]+\.)+(?:go\.kr|or\.kr|ac\.kr|co\.kr|kr|com|net|org|io|edu)(?![A-Za-z0-9])(?:/[^\s'\"()]*)?")
UNSUPPORTED_URL = "근거에 없는 주소"


def unsupported_urls(answer: str, context_text: str) -> list[str]:
    """답변의 URL/도메인 중 컨텍스트에 그대로(대소문자·앞 http:// 무시) 없는 것."""
    ctx = context_text.lower().replace("http://", "").replace("https://", "").replace("www.", "")
    bad = []
    for m in _URL_TOKEN.finditer(answer):
        u = m.group().lower().replace("http://", "").replace("https://", "").replace("www.", "").rstrip("./")
        if u and u not in ctx:
            bad.append(m.group())
    return sorted(set(bad))


def context_text(chunks: list[dict]) -> str:
    """검증 기준이 되는 근거 텍스트 — 페이지 전문 + 핵심 근거 줄(표 행 조각, 프롬프트에도 같이 들어간다)."""
    parts = []
    for c in chunks:
        parts.append(c.get("text", ""))
        parts.extend(c.get("highlights") or [])
    return "\n".join(parts)


# ── 날짜 대조: 답변의 "M월 D일" 은 인용 구절의 날짜여야 한다 ──────────────
# 실측(09-17): 입학식(2026.3.3.)을 물었는데 옆 페이지 졸업식(2027.2.3.)과 섞어 "2월 3일" — 두 날짜가 다
# 컨텍스트에 있어 숫자 대조로는 못 잡는다. 인용 구절(모델이 답의 근거라고 복사한 한 줄)에 있는 날짜만 허용한다.
# 인용 구절의 날짜 표기: 2026.3.3.(화) / 3.3 / 3/3 / 3월 3일 / 달력 행 "3: … 18(수3) …"(행 머리 월 + 칸 일)
# 일(日) 자리는 4자리까지 잡는다 — 실측: 모델이 "2월 2026일" 처럼 연도를 일 자리에 쓴 답을 2자리 규칙은 놓쳤다
_ANS_DATE = re.compile(r"(\d{1,2})\s*월\s*(\d{1,4})\s*일")
_DOTTED = re.compile(r"(?:\d{4}\s*[.\-/]\s*)?(\d{1,2})\s*[.\-/]\s*(\d{1,2})\b")
_CAL_HEAD = re.compile(r"(?:^|\s)(\d{1,2})\s*:")
_CAL_DAY = re.compile(r"\b(\d{1,2})\s*\([월화수목금토일]")


def _quote_dates(quote: str) -> set[tuple[int, int]]:
    out = {(int(m), int(d)) for m, d in _ANS_DATE.findall(quote)}
    out |= {(int(m), int(d)) for m, d in _DOTTED.findall(quote) if 1 <= int(m) <= 12 and 1 <= int(d) <= 31}
    head = _CAL_HEAD.search(quote)
    if head:
        month = int(head.group(1))
        if 1 <= month <= 12:
            out |= {(month, int(d)) for d in _CAL_DAY.findall(quote) if 1 <= int(d) <= 31}
    return out


# 문서 쪽 완전한 날짜(연·월·일): "2026.12.31.", "2027-1-28", "2026년 3월 3일", "1982.03.01."
_CTX_FULL_DATE = re.compile(r"(\d{4})\s*[.\-/년]\s*(\d{1,2})\s*[.\-/월]\s*(\d{1,2})")


def _context_full_dates(ctx: str) -> set[tuple[int, int, int]]:
    return {(int(y), int(m), int(d)) for y, m, d in _CTX_FULL_DATE.findall(ctx) if 1 <= int(m) <= 12 and 1 <= int(d) <= 31}


def unsupported_dates(answer: str, quote: str, extra: str = "", ctx: str = "") -> list[str]:
    """답변의 날짜 중 근거가 없는 것.

    - 연도가 붙은 완전한 날짜("2026년 1월 31일")는 문서 전체(ctx)에 (연,월,일)이 그대로 있어야 한다.
      실측: 모델이 "2026년 1월 31일부터 2027년 1월 2일까지" 라고 지어냈고(문서는 12.31~1.28) 둘 다 문서에 없다.
      반대로 "1982년 3월 1일" 은 문서의 "1982.03.01." 과 맞아 통과 — 근거가 핵심 근거 줄이 아니라 본문에 있어도 된다.
    - 연도 없는 날짜("2월 3일")는 인용 구절(+extra: 핵심 근거 줄들)에 (월,일)이 있어야 한다 — 문서 전체로 허용하면
      옆 페이지 졸업식 날짜(2027.2.3.)가 입학식 답으로 통과한다. 허용 날짜가 하나도 없으면 검사하지 않는다.
      extra 를 두는 이유: 답이 두 날짜에 걸치는데(연휴 24·25일) 모델은 한 줄만 인용하는 경우."""
    allowed = _quote_dates(quote) | _quote_dates(extra)
    if not allowed and ctx:
        # 실측(09-18): 인용이 모델 문장(가짜)이고 핵심 근거 줄도 안 잡힌 문항에서 "5/26(화)" 를 "5월 2일 화요일" 로
        # 읽은 답이 검사 없이 나갔다. 인용 기준이 없으면 문서 전체의 (월,일) 이라도 기준으로 삼는다 — 느슨하지만
        # 문서에 아예 없는 날짜는 막는다.
        allowed = _quote_dates(ctx)
    full = _context_full_dates(ctx) if ctx else set()
    bad = []
    for y, m, d in _answer_dates(answer):
        if y is not None:
            if ctx and (y, m, d) not in full:
                bad.append(f"{y}년 {m}월 {d}일")
        elif allowed and (m, d) not in allowed:
            bad.append(f"{m}월 {d}일")
    return sorted(set(bad))


# "27일간", "3일 동안", "5일 이내" 같은 기간·상대 표현은 날짜가 아니다 — 뒤에 붙는 말로 거른다
_ANS_TOKEN = re.compile(
    r"(\d{4})\s*년|(\d{1,2})\s*월|(\d{1,4})\s*일(?!\s*(?:간|동안|이내|이상|이하|전|후|째|정도|마다))"
)


def _answer_dates(answer: str) -> list[tuple]:
    """답변의 날짜들 (연 또는 None, 월, 일). "2026년 12월 31일부터 2027년 1월 28일", "9월 24일과 25일" 처럼
    연·월이 생략된 뒤쪽 값은 앞의 연·월을 이어받는다."""
    out, year, month = [], None, None
    for m in _ANS_TOKEN.finditer(answer):
        if m.group(1):
            year = int(m.group(1))
        elif m.group(2):
            month = int(m.group(2))
        elif month is not None:
            out.append((year, month, int(m.group(3))))
    return out


# ── 담당자 검증: "누구/누가/담당자/성함" 을 물었으면 답에 근거 구절의 이름이 있어야 한다 ─────────────
# 실측(09-17, 3회 연속): "학적 업무는 누가 담당해?" 에 근거 "업무 학적, 담당자 노OO, …" 를 인용하고도 답은
# "학적 담당자가 담당합니다" — 이름 자리에 라벨을 그대로 옮겼다. 이름은 라벨(담당자·부장·교장…) 뒤 토큰으로 찾는다:
# 마스킹된 "노OO" 또는 세 글자 이름. 이름 후보가 하나도 없으면(부서를 묻는 질문 등) 검사하지 않는다.
_WHO_Q = re.compile(r"누구|누가|담당자|성함|이름|담당해|담당하|맡아|맡고")
# 라벨 앞에 한글이 붙으면("학교장") 라벨이 아니다 — 실측(09-18): "학교장 교직원" 을 "교장 + 이름(교직원)" 으로 읽어
# "보건 선생님은 누구?" 에 이름 없는 보고체계 줄을 읽어 줬다. 세 글자 후보가 직책·집단 명사면 이름이 아니다.
_ROLE_WORDS = ("교직원", "교사", "선생", "학생", "학부모", "담당", "부서", "위원", "주무관", "주사", "기사", "실장", "부장", "교감", "교장")
# 라벨 뒤 세 글자가 사람 이름이 아닌 사물·조직 낱말인 경우 — 실측(09-21 키오스크): 위원회 표의 "담당자 장학금 담당" 에서
# "장학금" 이 이름으로 잡혀 "학생선도위원회 담당자는?" 정답(선도위원회 담당)을 막고 폴백이 장학생 행을 읽었다.
_NOT_NAME_SUFFIX = ("금", "비", "실", "부", "과", "회", "팀", "처", "센터", "학교")
# "성명·이름" 은 문장 속에서도 흔한 낱말이라("이름 그대로", "이름이 뭐야") 콜론이 붙은 표 라벨일 때만 이름 라벨로 본다 —
# 실측(09-18, 배재고): 근거 "'인재를 기른다'는 이름 그대로" 에서 "그대로" 를 담당자 이름으로 요구해 정답 "고종" 을 막았다
_NAME_AFTER_LABEL = re.compile(
    r"(?<![가-힣])(?:(?:담당자|담당|부장|실장|학교장|교장|교감|담임|책임자|팀장|과장|원장|소장|국장)\s*[:：]?|(?:성명|이름)\s*[:：])\s*"
    r"((?:[가-힣][O0]{1,3}(?![O0]))|(?:[가-힣]{3}(?![가-힣])))"
)
# 마스킹 표기는 영문 O("허OO", 텍스트 레이어)와 숫자 0("허00", 도해 VLM 전사) 두 가지가 섞여 있다 — 둘 다 이름이다.
# 실측(09-22 골든 m09·m10): p26 조직도 줄 "교장 허00은 …" 이 숫자 0 이라 이름으로 안 잡혀 이름 질문 구제·폴백에서 빠졌다.
_ROLE_WORD = re.compile(r"(?:교사|교감|교장|부장|실장|팀장|과장|원장|소장|국장|담당자|담임|주무관|사서|영양사|위원장)(?![가-힣])")
MISSING_PERSON = "담당자 이름"


def _names(text: str) -> set[str]:
    return {n for n in _NAME_AFTER_LABEL.findall(text)
            if not any(n.endswith(w) or n == w for w in _ROLE_WORDS)
            and not (n.endswith(_NOT_NAME_SUFFIX) and "O" not in n and "0" not in n)}


_MASKED_NAME = re.compile(r"[가-힣][O0]{1,3}(?![A-Za-z0-9])")


def is_who_question(question: str) -> bool:
    return bool(_WHO_Q.search(question))


def has_person_name(text: str) -> bool:
    """텍스트에 사람 이름(라벨 뒤 세 글자 또는 마스킹된 '조OO')이 있는가 — 폴백 구절 선택용.
    실측(09-18): "보건 선생님은 누구야?" 검색이 보건 업무 페이지로 빠져, 이름 없는 감염병 대응 줄을 읽어 줬다."""
    return bool(_names(text) or _MASKED_NAME.search(text))


def has_person_or_role(text: str) -> bool:
    """이름 또는 역할 명사(교사·부장·실장·담당자…)가 있는가 — "누구에게 보고?" 처럼 역할이 답인 질문의 폴백용."""
    return has_person_name(text) or bool(_ROLE_WORD.search(text))


def missing_person(question: str, answer: str, evidence: str) -> list[str]:
    """사람을 묻는 질문인데 근거(인용+핵심 근거)의 이름이 답에 없으면 [MISSING_PERSON]."""
    if not evidence or not _WHO_Q.search(question):
        return []
    names = _names(evidence)
    if not names or any(n in answer for n in names):
        return []
    return [MISSING_PERSON]


# ── 동문서답 검사: 답이 질문의 낱말을 하나도 안 담으면 질문에 대한 답이 아니다 ─────────────
# 실측(09-18): "여기에 질문" 같은 뜻 없는 입력이 설문 페이지의 "1: 질문 …" 행과 낱말로 겹쳐 검색되고, 모델이
# "만족 비율은 61.5%입니다" 라고 답했는데 숫자 대조는 통과했다. 키오스크의 "테스트·여보세요" 입력에서 같은 일이 난다.
# 판정은 textutil.question_overlap(한글 접두 일치·숫자)에 영문 낱말을 더한 것 — 골든 답변은 전부 질문의 주어를 되풀이한다.
_EN_TOKEN = re.compile(r"[A-Za-z]{2,}")
_REFUSAL = re.compile(r"찾을 수 없|확인이 필요한|문의해 주세요")
OFF_TOPIC = "질문과 무관"
# 근거 있는 거절: 모델이 "찾을 수 없다" 고 했는데 [핵심 근거] 에 질문 낱말이 충분히 겹치는 줄이 있는 경우 — 그 줄을 짚어 주며
# 한 번 더 묻는다. 실측(09-18, 3회 연속): "생활체육진로반은 몇 명이 배정돼?" 근거 첫 줄이 "… 생활체육진로반 … 배정인원 25명 …"
# 인데 1초 만에 거절, "사관학교 7기 총사업비?" 도 "광주인공지능사관학교 — 2026년 제7기는 …(총사업비 약 105억원)…" 을 두고 거절.
# 질문 표현(배정돼·7기)과 문서 표현(배정인원·제7기)이 살짝 다를 때 7B 가 포기하는 유형. 무관 질문("테스트")은 겹치는 낱말이
# 없어 재질문이 일어나지 않는다.
REFUSED_WITH_EVIDENCE = "근거 있는 거절"
REFUSAL_EVIDENCE_MIN_SCORE = 5   # 접두 일치 점수 합 (낱말 2개 이상, 5자 이상 겹침)
REFUSAL_EVIDENCE_MIN_TOKENS = 2


def is_refusal(text: str) -> bool:
    return bool(text) and bool(_REFUSAL.search(text))


def refusal_evidence(question: str, lines: list[str]) -> str:
    """거절 답변을 되물을 만한 근거 줄 — 질문 낱말이 REFUSAL_EVIDENCE_MIN_TOKENS 개 이상, 점수 REFUSAL_EVIDENCE_MIN_SCORE 이상
    겹치는 줄 중 최고점. 없으면 ''."""
    from .textutil import question_overlap, question_tokens

    toks = [t for t in question_tokens(question) if not t.isdigit()]
    if not toks:
        return ""
    # 군더더기를 뺀 낱말이 하나뿐인 질문("전체 교직원 수는?" → 교직원)은 그 하나로 판단한다 — 실측(09-20): "전체"·"수는" 이
    # 낱말로 세어져 '2개 겹침' 조건을 못 채우고 재질문 없이 거절이 나갔다. 점수 기준은 그대로.
    need = min(REFUSAL_EVIDENCE_MIN_TOKENS, len(toks))
    need_score = REFUSAL_EVIDENCE_MIN_SCORE if len(toks) >= 2 else min(REFUSAL_EVIDENCE_MIN_SCORE, len(toks[0]))
    best, best_score = "", 0
    for line in lines:
        matched = sum(1 for t in toks if any(t[:n] in line for n in range(len(t), 1, -1)))
        score = question_overlap(question, line, toks)
        if matched >= need and score >= need_score and score > best_score:
            best, best_score = line, score
    return best


def off_topic(question: str, answer: str) -> list[str]:
    from .textutil import question_overlap, question_tokens  # 순환 import 방지

    if not answer or _REFUSAL.search(answer):
        return []
    # 군더더기·기관명을 지우지 않은 낱말로 본다 — 여기서는 "답이 질문과 조금이라도 이어지는가" 만 보면 되므로 기관명이
    # 되풀이된 것도 관련의 증거다. 실측(09-20, 1000문항 v3): "배재고 설립 형태는?" → "배재고등학교는 사립·단설 학교입니다" 가
    # 기관명을 뺀 낱말(설립·형태)과 안 겹쳐 '질문과 무관' 으로 차단됐다(3건).
    from .textutil import STOP_TOKENS, _HANGUL_TOKEN, _is_title_token

    toks = question_tokens(question, drop_stop=False)
    en = {t.lower() for t in _EN_TOKEN.findall(question)}
    if not toks and not en:
        return []
    # 군더더기·"질문·여기" 류를 빼면 내용 낱말이 하나도 없는 입력("여기에 질문", "테스트")은 무엇을 물은 게 아니다 —
    # 실측(09-21 골든 x07): 모델이 설문 행을 읽고 "11번 질문에 대한 답변은 …" 이라 답해 '질문' 낱말이 겹쳐 통과했다.
    if not en and not [t for t in toks if t not in STOP_TOKENS and t not in _NOISE_TOKENS and not _is_title_token(t)]:
        return [OFF_TOPIC]
    # 답이 질문 낱말을 되받은 것뿐이면(새 낱말·숫자 없음) 답이 아니다 — 실측(09-21 골든 x09): "여보세요" → 답 "여보세요".
    a_toks = set(_HANGUL_TOKEN.findall(answer))
    if a_toks and a_toks <= set(toks) and not any(ch.isdigit() for ch in answer):
        return [OFF_TOPIC]
    if any(t in answer.lower() for t in en):
        return []
    if len(toks) == 1 and not en:
        # 낱말 하나짜리 입력("사랑해", "안녕", "교훈")은 접두 2자 일치로는 부족하다 — 실측(09-21 키오스크): "사랑해" 가
        # "사랑이 넘치는 교육복지 활동" 행과 '사랑' 으로 겹쳐 담당자 이름까지 읽어 줬다. 낱말 전체가 답에 있거나,
        # 조사를 뗀 어간("개교는" → 개교, "교훈이" → 교훈)이 있어야 그 낱말의 답이다. '해·야' 같은 어미는 조사가 아니다.
        t = toks[0]
        stems = {t} | {t[:-len(j)] for j in _JOSA if t.endswith(j) and len(t) - len(j) >= 2}
        return [] if any(st in answer for st in stems) else [OFF_TOPIC]
    if question_overlap(question, answer, toks) > 0:
        return []
    return [OFF_TOPIC]


# 질문의 뜻을 담지 않는 낱말 — 이것만으로 된 입력은 무관 입력으로 본다 (STOP_TOKENS 는 textutil 이 공용으로 관리)
_NOISE_TOKENS = frozenset(("질문", "여기", "여기에", "저기", "거기", "테스트", "시험", "확인", "입력"))
# 낱말 끝의 조사 — 긴 것부터 대조한다 (off_topic 의 한 낱말 입력 판정용)
_JOSA = ("에서는", "에서", "으로", "이랑", "에는", "은요", "는요", "이요", "은", "는", "이", "가", "을", "를", "의", "에", "도", "와", "과", "로", "요", "랑")


# ── 복합 질문의 항목 누락: 부분 질문의 주어가 답에 없으면(또는 값 질문인데 그 자리에 숫자가 없으면) 한 항목만 답한 것 ──
# 실측(09-23 골든 a38): "착공과 1단계 준공은 각각 언제?" 에 연혁 줄 둘을 이어 읽었지만 준공 쪽 연도가 빠졌는데 통과했다.
# m32 "전체 학생 수와 학급 수" 도 한쪽만 말한 답이 검증을 통과한다. 재생성 한 번의 근거로만 쓰고 답을 막지는 않는다 —
# 답의 표현이 문서와 달라 못 찾는 오탐이 있을 수 있고, 한 항목이라도 맞은 답이 안내문보다는 낫다.
MISSING_PART = "항목 누락"
# 답을 절(節)로 나누는 자리 — 문장 끝(…입니다) · 쉼표 · 가운뎃점 · 줄바꿈 · 이어주는 어미(…했고, …되며)
# 어미까지 나누는 이유: "2021년에 착공했고 실증동이 준공되었습니다" 를 한 절로 보면 앞 항목의 연도가 뒤 항목의 값으로 통한다
_ANS_CLAUSE = re.compile(r"(?<=[다요])\s+|[,·\n]|(?<=[가-힣])(?:했고|하고|이고|되고|되었고|였고|하며|이며|되며|지만)\s+")


def missing_part(question: str, answer: str, parts: list[str]) -> list[str]:
    """복합 질문에서 답이 빠뜨린 항목들 — 부분 질문마다 주어 낱말이 답에 있는지, 값 질문이면 그 절에 숫자가 있는지 본다."""
    from .textutil import is_value_question, question_tokens

    if not parts or not answer or _REFUSAL.search(answer):
        return []
    clauses = [c.strip() for c in _ANS_CLAUSE.split(answer) if c and c.strip()]
    if not clauses:
        return []
    # 부분마다 **그 부분에만 있는** 낱말로 본다 — 술어는 양쪽에 똑같이 붙어 있어("… 부장은 각각 누구야?") 그것까지 세면
    # 앞 항목만 답한 문장도 뒤 항목의 답으로 통한다. 실측(09-23 골든 m33): "교육과정부 부장은 유OO입니다" 가 '부장' 으로 통과.
    tok_sets = [[t for t in question_tokens(p) if len(t) >= 2 and not t.isdigit() and not _is_en_tok(t)] for p in parts]
    shared = set.intersection(*(set(s) for s in tok_sets)) if len(tok_sets) > 1 else set()
    bad = []
    for part, all_toks in zip(parts, tok_sets):
        toks = [t for t in all_toks if t not in shared]
        if not toks:
            continue
        # 그중에서도 가장 긴 낱말이 그 항목의 이름이다 — 짧은 낱말은 술어의 잔재("부장은")라 옆 항목 문장에도 있다.
        # 실측(09-23 골든 m33): "교육과정부 부장은 유OO입니다" 가 '부장은' 으로 두 항목 다 답한 것처럼 통과했다.
        longest = max(len(t) for t in toks)
        toks = [t for t in toks if len(t) == longest]
        hit = [c for c in clauses if any(_line_has_token(c, t) for t in toks)]
        if not hit:
            bad.append(f"{MISSING_PART}({toks[0]})")
        elif is_value_question(part) and not any(any(ch.isdigit() for ch in c) for c in hit):
            bad.append(f"{MISSING_PART}({toks[0]} 값)")
    return bad


def _is_en_tok(t: str) -> bool:
    return bool(t) and t[0].isascii() and t[0].isalpha()


# ── 행 단위 수량 대조: "N층/N명/N회…" 는 질문 대상이 적힌 줄의 값이어야 한다 ─────────────
# 실측(09-18): "행정실은 몇 층?" 근거 맨 앞에 "1층: … 행정실 …" 행이 있는데 답은 "2층" — 2 는 문서 어딘가에 있어
# 숫자 대조를 통과했다. 질문의 핵심 낱말(3자 이상, 접두 일치)이 든 줄들만 모아, 그 줄들이 같은 단위의 값을 갖고 있는데
# 답의 값은 없으면 불일치로 본다. 그 줄들에 단위가 없으면(표 값이 단위 없이 적힘) 맨숫자로 대조하고, 아무 값도 없으면 검사하지 않는다.
_UNIT_VAL = re.compile(r"(?<![\d.])(\d{1,2})\s*(명|개|층|미터|퍼센트|%|원|회|학급|교시|대|건|점|학년)")  # 소수 "1.4%" 의 4 는 값이 아니다
ROW_MISMATCH = "≠"
# 단위 없는 맨숫자를 값으로 인정하지 않는 단위 — "층" 은 위치이고 표의 맨숫자는 개수다.
# 시설 배치도에서 층을 확인할 때 교실 수 표의 맨숫자를 층수로 인정하지 않는다.
# "명·개·점" 은 표에 단위 없이 적히는 값이라(m30 "상점 2") 맨숫자 규칙을 그대로 둔다 — 층만 뺀다.
_NO_BARE_UNITS = frozenset(("층",))


def _key_tokens(question: str) -> list[str]:
    from .textutil import question_tokens

    toks = [t for t in question_tokens(question) if not t.isdigit()]
    long = [t for t in toks if len(t) >= 3]
    return long or sorted(toks, key=len, reverse=True)[:1]


def _line_has_token(line: str, tok: str) -> bool:
    """질문 낱말(접두)이 줄에 있는가 — 3자 낱말은 조사를 뗀 2자("보컬은" → 보컬)까지 본다(_line_has_token_loose 와 같은 규칙).
    실측(09-21 키오스크): "밴드 동아리 보컬은 몇 명?" — 3자만 보니 "보컬은" 이 든 줄이 없어 멤버 행(보컬 4명)이 대조 대상에서
    빠지고, "동아리" 가 든 조직 행(인원 19명)만 남아 정답 4명을 '4명≠19명' 으로 거부했다."""
    line_ns = line.replace(" ", "")   # "상담 교사" 행도 "상담교사" 질문의 행이다 (textutil.question_overlap 주석 참고)
    floor = 1 if len(tok) <= 3 else 2
    return any(tok[:n] in line or tok[:n] in line_ns for n in range(len(tok), floor, -1))


def audit_answer_values(question: str, answer: str, facts: list[str], *, with_evidence: bool = False) -> list:
    """질문 대상의 사실 행과 답의 값을 대조한다. 모호하면 판정하지 않는다.

    with_evidence=True이면 불일치가 있고 두 항목 모두 근거 행이 유일할 때만
    두 항목의 (부분 질문, 행, 기대 값)을 돌려준다.

    실측(261101_1, m29·m30): 다른 대상의 층수·상점이 같은 페이지에
    있어 기존의 문서 전체 숫자 대조를 통과했다. 사실 행에서 대상과 값이 함께
    확인되는 경우에만 불일치를 기록한다.
    실측(261102_2, m14): 개교 당시와 현재의 기관명이 달라 연도 단독 대조는
    정답을 오탐했다. 연도 검사는 이 함수에서 하지 않는다.
    실측(261102_4, m29·m30): 불일치 항목만 재생성에 주니 다른 항목을 누락했다.
    재생성에는 두 항목의 근거가 모두 필요하다.
    """
    from .textutil import question_tokens, split_compound_question

    if not question or not answer or not facts or is_refusal(answer):
        return []
    parts = split_compound_question(question)
    if len(parts) == 2:
        clauses = [c.strip() for c in _ANS_CLAUSE.split(answer) if c and c.strip()]
        token_sets = [set(question_tokens(p)) for p in parts]
        issues = []
        paired = []
        for i, part in enumerate(parts):
            unique = [t for t in token_sets[i] - token_sets[1 - i] if len(t) >= 2 and not t.isdigit()]
            if not unique:
                continue
            longest = max(map(len, unique))
            targets = [t for t in unique if len(t) == longest]
            hit_clauses = [c for c in clauses if any(_line_has_token(c, t) for t in targets)]
            if len(hit_clauses) != 1:
                continue
            said = {(n.lstrip("0") or "0", u) for n, u in _UNIT_VAL.findall(hit_clauses[0])}
            if len(said) != 1:
                continue
            said_num, unit = next(iter(said))
            expected = set()
            evidence = {}
            ambiguous = False
            for fact in facts:
                if " — " not in fact:
                    continue
                tail = fact.rsplit(" — ", 1)[-1]
                if ":" not in tail:
                    continue
                label, value = tail.split(":", 1)
                in_label = any(_line_has_token(label, t) for t in targets)
                in_value = any(_line_has_token(value, t) for t in targets)
                if in_label:
                    # 표의 한 항목이 숫자 하나인 행만. 긴 혼합 셀은 판정하지 않는다.
                    bare = re.fullmatch(r"\s*(\d[\d,]*)\s*", value)
                    if bare:
                        values = {(_norm(bare.group(1)).lstrip("0") or "0", unit)}
                        if unit in _NO_BARE_UNITS:
                            ambiguous = True
                    else:
                        values = {(n.lstrip("0") or "0", u) for n, u in _UNIT_VAL.findall(value) if u == unit}
                    if len(values) > 1:
                        ambiguous = True
                    if len(values) == 1:
                        expected.update(values)
                        evidence.setdefault(next(iter(values)), set()).add(fact)
                elif in_value:
                    # 배치도처럼 층수가 행 라벨이고 시설이 값 칸에 나열된 경우.
                    values = {(n.lstrip("0") or "0", u) for n, u in _UNIT_VAL.findall(label) if u == unit}
                    if len(values) > 1:
                        ambiguous = True
                    if len(values) == 1:
                        expected.update(values)
                        evidence.setdefault(next(iter(values)), set()).add(fact)
            if len(expected) == 1:
                value = next(iter(expected))
                value_num, _ = value
                if not ambiguous and len(evidence.get(value, ())) == 1:
                    paired.append((part.strip(), next(iter(evidence[value])), f"{value_num}{unit}"))
                if (said_num, unit) not in expected:
                    issues.append(f"{part.strip()}: {said_num}{unit}≠{value_num}{unit}")
        if with_evidence:
            return paired if issues and len(paired) == 2 and paired[0][1] != paired[1][1] else []
        return issues
    return []


def paired_value_problems(answer: str, evidence: list[tuple[str, str, str]]) -> list[str]:
    """재생성 답변의 각 항목을 해당 근거 값과 대조한다. 모호하면 채택하지 않는다."""
    from .textutil import question_tokens

    if not answer or is_refusal(answer) or len(evidence) != 2:
        return ["답변 없음·거절 또는 근거 부족"]
    clauses = [c.strip() for sentence in re.split(r"(?<=[.!?。])\s+", answer)
               for c in _ANS_CLAUSE.split(sentence) if c and c.strip()]
    tokens = [set(question_tokens(part)) for part, _, _ in evidence]
    bad = []
    for i, (part, _, expected) in enumerate(evidence):
        unique = [t for t in tokens[i] - tokens[1 - i] if len(t) >= 2 and not t.isdigit()]
        if not unique:
            bad.append(f"항목 식별 불가: {part}")
            continue
        longest = max(map(len, unique))
        targets = [t for t in unique if len(t) == longest]
        hit = [c for c in clauses if any(_line_has_token(c, t) for t in targets)]
        if len(hit) != 1:
            bad.append(f"항목 누락·중복: {part}")
            continue
        values = {f"{n.lstrip('0') or '0'}{u}" for n, u in _UNIT_VAL.findall(hit[0])}
        if values != {expected}:
            bad.append(f"항목 값 불일치·모호: {part} (기대 {expected})")
    return bad


def unsupported_row_numbers(question: str, answer: str, lines: list[str]) -> list[str]:
    keys = _key_tokens(question)
    if not keys:
        return []
    rows = [l for l in lines if l and any(_line_has_token(l, k) for k in keys)]
    if not rows:
        return []
    bad = []
    found = _UNIT_VAL.findall(answer)
    for num, unit in found:
        n = num.lstrip("0") or "0"
        if len({v.lstrip("0") or "0" for v, u in found if u == unit}) > 1:
            continue  # 같은 단위 값이 여럿인 답("1학기 40%, 2학기 60%")은 내역 나열 — 한 줄 대조 대상이 아니다
        with_unit = re.compile(r"(?<![\d.])0*" + n + r"\s*" + re.escape(unit))
        if any(with_unit.search(r) for r in rows):
            continue
        # 같은 단위의 다른 값이 든 줄 — 답의 값이 그 줄에 없으면 불일치
        others = sorted({m.group(1).lstrip("0") or "0" for r in rows for m in with_unit_any(unit).finditer(r)})
        # 단위 없는 표 값은 맨숫자로 — 단, 낱말에 붙은 숫자("교육반2", "3-1")는 값이 아니고, 질문 낱말 **바로 옆**(12자 안)에
        # 있어야 한다: "행정실 1, 회의실 1, … 영어 교과 실 2" 에서 2 는 행정실의 값이 아니다
        bare = re.compile(r"0*" + n)
        unitless_rows = [] if unit in _NO_BARE_UNITS else [r for r in rows if not with_unit_any(unit).search(r)]
        if others and not any(_near_key(r, keys, bare) for r in unitless_rows):
            bad.append(f"{n}{unit}{ROW_MISMATCH}{'/'.join(o + unit for o in others)}")
    return bad


def _line_has_token_loose(line: str, tok: str) -> bool:
    """접두 일치를 2자까지 허용 — "개교는" 처럼 조사가 붙은 3자 낱말은 "개교" 로 대조해야 한다."""
    floor = 1 if len(tok) <= 3 else 2
    return any(tok[:n] in line for n in range(len(tok), floor, -1))


def unsupported_row_dates(question: str, answer: str, lines: list[str]) -> list[str]:
    """답의 날짜는 **질문 낱말이 든 줄**의 날짜여야 한다 — 실측(09-21 키오스크): "망미중학교 개교는 언제야?" 에
    달력 줄 "8월 18일(화) 개학" 을 읽고 "개교는 8월 18일" 이라고 답했다. 8/18 은 문서에 있는 날짜라 날짜 대조를 통과했고,
    개교≠개학 은 낱말 대조로만 잡힌다 (같은 유형: 시업식 → 종업식 2/3, SQ3R 4월 → 옆 행 6월).
    질문 낱말이 든 줄에 날짜가 하나도 없으면 검사하지 않는다(대조 기준이 없음). 불일치면 호출 측이 인용을 버리고
    질문 낱말이 든 줄로 폴백한다(ROW_MISMATCH)."""
    keys = _key_tokens(question)
    if not keys:
        return []
    ans_dates = {(m, d) for _, m, d in _answer_dates(answer)}
    if not ans_dates:
        return []
    rows = [l for l in lines if l and any(_line_has_token_loose(l, k) for k in keys)]
    allowed: set[tuple[int, int]] = set()
    for r in rows:
        allowed |= _quote_dates(r)
        allowed |= {(m, d) for _, m, d in _context_full_dates(r)}
    if not allowed:
        return []
    if ans_dates & allowed:
        return []
    shown = "/".join(f"{m}월{d}일" for m, d in sorted(allowed)[:3])
    return [f"{m}월{d}일{ROW_MISMATCH}{shown}" for m, d in sorted(ans_dates)]


def _near_key(line: str, keys: list[str], num: re.Pattern, gap: int = 6) -> bool:
    """줄에서 질문 낱말(접두) **바로 뒤에 오는 값**이 그 숫자인가 — 사이에 숫자 아닌 글자 gap 개까지 허용
    ("행정실 1", "1학년 83", "학년: 83"). "행정실 1, 회의실 2" 에서 2 는 행정실의 값이 아니다."""
    for text in (line, line.replace(" ", "")):   # "상담 교사 1" 도 "상담교사" 바로 뒤의 값 (textutil.question_overlap 주석 참고)
        for k in keys:
            for n in range(len(k), 2, -1):
                for m in re.finditer(re.escape(k[:n]) + r"[^\d\n]{0," + str(gap) + r"}?(\d[\d,.]*)", text):
                    if num.fullmatch(m.group(1).rstrip(".,")):
                        return True
    return False


_UNIT_ANY_CACHE: dict[str, re.Pattern] = {}


def with_unit_any(unit: str) -> re.Pattern:
    if unit not in _UNIT_ANY_CACHE:
        _UNIT_ANY_CACHE[unit] = re.compile(r"(?<![\d.])(\d{1,2})\s*" + re.escape(unit))
    return _UNIT_ANY_CACHE[unit]


# ── 순번 대조: 질문의 "1호/3학년/2학기/17대…" 를 물었는데 인용한 줄이 같은 단위의 다른 순번(2호)뿐이면 옆 항목을 읽은 것 ──
# 실측(09-18): "가해자 조치 1호는?" 에 2호 행("접촉, 협박, 보복행위의 금지")을 인용하고 "1호는 …" 이라고 답했다.
# 숫자 대조·수량 대조는 "호" 를 단위로 안 보므로 통과했다. 순번은 값이 아니라 항목 이름이라 따로 본다.
_ORDINAL = re.compile(r"(?<![\d.])(\d{1,2})\s*(호|단계|차|항|조|번|학년|학기|층|기|대|회차|주차|교시)")  # 뒤에 조사("1호는")가 붙어도 잡는다


def ordinal_mismatch(question: str, quote: str) -> list[str]:
    """질문의 순번(N단위)이 인용에 없고 같은 단위의 다른 순번만 있으면 ["1호≠2호"]. 인용이 없거나 순번이 없으면 []."""
    if not quote:
        return []
    q_ord = {(n.lstrip("0") or "0", u) for n, u in _ORDINAL.findall(question)}
    if not q_ord:
        return []
    c_ord = {(n.lstrip("0") or "0", u) for n, u in _ORDINAL.findall(quote)}
    bad = []
    for n, u in sorted(q_ord):
        if (n, u) in c_ord:
            continue
        others = sorted(m for m, v in c_ord if v == u)
        if others:
            bad.append(f"{n}{u}{ROW_MISMATCH}{'/'.join(o + u for o in others)}")
    return bad


_YEAR_Q = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*(?:년|학년도)")


def year_mismatch(question: str, answer: str) -> list[str]:
    """질문이 연도를 짚었는데 답이 다른 연도만 말하면 ["2021년≠2020년"]. 답에 연도가 없거나 질문 연도가 들어 있으면 [].
    실측(09-21 키오스크, 망미 179문항): "2021년에 받은 표창은?" → "2020년에 받은 표창은 '생활교육 및 인성교육 …'" — 표창
    이름은 2021.12.31. 행에서 맞게 옮겼는데 연도를 옆 행(2020.12.01.)의 것으로 썼고, 2020 은 근거에 있어 숫자 검증을 통과했다."""
    q_years = {y for y in _YEAR_Q.findall(question)}
    if not q_years:
        return []
    a_years = {y for y in _YEAR_Q.findall(answer)}
    if not a_years or q_years & a_years:
        return []
    return [f"{'/'.join(sorted(q_years))}년{ROW_MISMATCH}{'/'.join(sorted(a_years))}년"]


_FLAT = re.compile(r"[^\w()]+")  # 글자·숫자·괄호만 남긴다 — 공백·표 기호('|')·구두점 차이는 무시


def quote_in_context(quote: str, ctx: str, min_len: int = 6) -> bool:
    """인용 구절이 컨텍스트에 실제로 있는지 — 공백·표 기호·구두점 차이만 무시하고 부분 문자열로 대조."""
    q = _FLAT.sub("", quote)
    if len(q) < min_len:
        return False
    return q in _FLAT.sub("", ctx)


def check(answer: str, chunks: list[dict]) -> tuple[str, bool, list[str]]:
    """(최종 답변, 검증 통과 여부, 근거 없는 숫자 목록).

    VERIFY_NUMBERS=0 이면 항상 통과. 통과 못 하면 답변을 UNVERIFIED_ANSWER 로 바꾼다 —
    잘못된 숫자가 섞인 답을 일부만 고쳐 내보내는 것보다 통째로 막는 편이 민원 안내에 안전하다.
    """
    if not VERIFY_NUMBERS or not answer or not chunks:
        return answer, True, []
    bad = unsupported_numbers(answer, context_text(chunks))
    if bad:
        logger.warning("검증 실패 — 근거에 없는 숫자 %s. 답변 차단: %r", bad, answer[:120])
        return UNVERIFIED_ANSWER, False, bad
    return answer, True, []
