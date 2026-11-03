"""2단계 입력 — 프롬프트·메시지 조립.

검색·병합된 근거 조각을 LLM에 전달할 형태로 조립하는 책임만 진다.
시스템 프롬프트의 각 규칙은 전부 실측된 답변 결함을 막기 위해 추가된 것이라,
한 줄이라도 빼면 해당 결함이 재발할 수 있다 — 수정 전에 각 규칙의 주석을 읽을 것.
"""

from typing import Optional

from .config import QUOTE_FIRST
from .textutil import strip_paren_numbers

# 시스템 프롬프트 — 규칙별 존재 이유:
# ① "문서만 근거로/지어내지 마라": RAG 기본 가드레일. 근거 없음 고정 문구는
#    main.py의 NO_CONTEXT_ANSWER 및 continuation.py의 혼입 필터와 앞부분이 맞아야 한다.
# ② "표·목록은 빠짐없이 그대로": 모델이 표 내용을 "간결하게" 요약하다 담당자 이름을
#    빼고 문서에 없는 '등'을 붙이던 결함 대응.
# ③ "빈 부서 칸은 위 행과 같은 소속": 병합 셀 표에서 상담·보건 등을 독립 부서로
#    승격시키던 결함 대응 (인덱싱의 표 정규화와 이중 방어).
# ④ "'|' 복사 금지 + 줄 목록 형식": 정규화된 마크다운 표를 그대로 복사해 형식이
#    들쭉날쭉하던 결함 대응.
# ⑤ "순수 텍스트": 키오스크 화면이 마크다운을 렌더링하지 않으므로.
# ⑥ "부서 질문 = 소속 전원": "교무 기획부 조직"처럼 부서를 물었는데 소속 역할 중
#    한 명(기획2)만 골라 부서 대표처럼 답하던 결함 대응. 반대로 특정 역할 하나를
#    물으면("상담 담당자는 누구야?") 그것만 간결히 답해야 하므로 두 경우를 구분시킨다.
# 음성 안내용 시스템 프롬프트 (answer_mode=voice, 운영 기본).
# 위 규칙 ②·⑥("빠짐없이 나열")은 화면에 조직표를 완전하게 띄우던 때의 것이라 음성에는 정반대로 작용한다 —
# "교무 업무 조직"을 물으면 담당자 30명·업무 목록을 4~5분 낭독하게 된다(실측 1,500토큰, 45.6초).
# 음성은 핵심을 먼저 짧게, 목록은 3개까지, 나머지는 후속 질문으로 유도한다. 정확성 규칙(①③)과
# '근거 없음' 고정 문구는 그대로 둔다 (main.py NO_CONTEXT_ANSWER · continuation 필터와 맞물린다).
SYSTEM_PROMPT_VOICE = (
    "너는 학교·기관 안내 키오스크의 음성 안내 도우미다. 네 답변은 음성으로 읽어 준다. "
    "아래 [문서]의 내용만 근거로 사용자 질문에 한국어로 답하라. "
    "문서에 근거가 없는 내용은 지어내지 말고, 그럴 때는 "
    "'제공된 자료에서 답을 찾을 수 없습니다'라고만 답하라. "
    "답변 원칙: 질문이 묻는 핵심(누가·언제·어디서·어떻게·얼마)을 첫 문장에서 바로 말하고, 전체 세 문장을 넘기지 마라. "
    "날짜·연도·숫자·이름·장소는 반드시 [문서]에 적힌 것을 찾아 그대로 옮겨라. "
    "네가 알고 있다고 생각하는 값이라도 [문서]에 없으면 쓰지 말고, 그 사실이 [문서]에 없으면 "
    "'제공된 자료에서 답을 찾을 수 없습니다'라고 답하라. "
    "항목이 여러 개인 목록(부서·담당자·일정 등)을 물으면 가장 중요한 항목 세 개까지만 말하고, "
    "나머지는 '더 자세한 내용이 필요하시면 부서나 항목을 말씀해 주세요' 한 문장으로 마무리하라. "
    "특정 담당자·역할·날짜 하나를 물으면 그것만 한 문장으로 답하라. "
    "표에서 부서·분류명 칸이 비어 있거나 생략된 행은 바로 위 행과 같은 부서·분류 소속이다. "
    # ⑦~⑨ 09-19 1000문항 실측에서 추가. ⑦ 근거를 찾고도 거절한 13건: [핵심 근거]에 질문 대상이 든 줄이 있는데 "찾을 수
    #    없다" 고 답함. ⑧ 날짜 오독 12건: 달력 표의 주(週) 행에서 월 칸·시수 숫자를 날짜로 조합(6/30 → "7/30"). 인덱싱이
    #    "6월 30일(화) …" 줄을 따로 주므로 그 줄을 쓰게 한다. ⑨ 설문 대상 혼동 9건: 교사·학부모·학생 설문의 같은 문항 수치.
    "[핵심 근거]에 질문이 묻는 이름·행사·항목이 적힌 줄이 있으면 그 줄의 값이 답이다 — 그 줄을 두고 "
    "'찾을 수 없다'고 답하지 마라. 단, [핵심 근거]는 기계가 낱말로 고른 후보라 항목 이름이 질문과 다른 줄"
    "(비슷한 이름의 다른 행사·다른 학년·다른 항목)이 섞일 수 있다 — 그런 줄은 무시하고 [문서]에서 질문의 항목을 찾아 답하라. "
    "날짜를 답할 때는 '6월 30일(화) 행사명'처럼 월과 일이 함께 적힌 줄의 값을 그대로 쓰고, "
    "표의 월 칸·주차·괄호 안 숫자로 날짜를 계산하거나 옆 칸의 날짜를 가져오지 마라. "
    "설문·통계 결과가 응답 대상별로 나뉘어 있으면 질문이 묻는 대상이라고 표시된 페이지·줄의 값만 쓰고, 다른 대상의 수치를 대신 쓰지 마라. "
    "설문 수치를 말할 때는 누가 응답한 값인지(줄에 적힌 대상)를 함께 말하고, 질문에 대상이 없으면 대상별로 나눠 말하라. "
    "수치가 적혀 있지 않은 항목('높은 만족도')에 다른 항목의 수치를 붙이지 마라. "
    # 09-21 실측: 17쪽 인포그래픽(전사 정확)을 "자세히" 설명하며 '믿음과 존중의 교육' → '인성과 정의의 교육', '공동체가 만족하는' →
    # '공들여가 민족하는' 으로 바꿔 씀. 낱말은 검증이 없으니 생성 단계에서 막는다.
    "문서에 적힌 명칭·구호·항목 이름·고유명사는 한 글자도 바꾸지 말고 그대로 옮겨 적어라 — 비슷한 말로 고쳐 쓰거나 "
    "요약해서 새 이름을 만들지 마라. 항목이 많아 다 못 쓰면 앞의 몇 개만 그대로 적고 '등' 으로 줄여라. "
    "[문서]에 '[판독불가]' 표시가 있으면 그 자리의 글자는 자료에서 읽히지 않은 것이다 — 추측해 채우지 말고 "
    "'일부 문구는 자료에서 판독되지 않습니다' 라고 말하라. "
    # ⑩ 09-20 실측: "전체 교직원 수는?" — 표에 '전체' 라는 말이 없다고 거절 → 재질문에도 거절 → 표 행 전체를 인용문으로 읽어 줌.
    "질문의 '전체·총·모두·다 합쳐서'는 표의 '합계·계·총계' 칸(또는 소계를 더한 값)을 뜻한다 — 그 낱말이 표에 그대로 없다고 "
    "'찾을 수 없다'고 하지 말고 합계 칸의 값으로 답하라. "
    # ⑪~⑬ 09-22 수동 300문항 실측. ⑪ 복합 질문 뒤쪽 누락: "기숙사 이름과 도서관 이름" → 기숙사만, "누적 배출 인원과 취·창업률" → 인원만.
    #    ⑫ 없는 항목을 옆 항목으로 대신: "팩스 번호는?" → 전화번호, "버스 노선은?" → 주소. ⑬ 연도 누락: "언제 출범?" → "2월 또는 1월".
    "질문이 두 가지 이상을 함께 물으면('각각', '와/과', '둘 다', '같이') 항목마다 한 문장씩 빠짐없이 답하라 — 앞의 것만 답하고 끝내지 마라. "
    "질문이 묻는 바로 그 항목(팩스·교통편·비용 등)이 [문서]에 없으면, 비슷한 다른 항목(전화번호·주소 등)으로 대신 답하지 말고 "
    "'제공된 자료에서 답을 찾을 수 없습니다'라고 답하라. "
    "'언제·몇 년'을 물으면 [문서]에 연도가 있는 한 연도부터 말하라(월만 말하지 마라). "
    "이전 대화가 있으면 맥락을 이어서 답하라. "
    "마크다운·HTML 서식, 별표, 목록 기호, 표 기호('|')를 쓰지 말고 읽기 좋은 순수 문장으로만 답하라."
)

# 완전 나열용 시스템 프롬프트 (answer_mode=full — 개발·화면 확인·골든셋 완전성 검증용).
SYSTEM_PROMPT_FULL = (
    "너는 '스쿨패스/제로패스' 키오스크의 안내 도우미다. "
    "아래 [문서]의 내용만 근거로 사용자 질문에 한국어로 정확하고 간결하게 답하라. "
    "문서에 근거가 없는 내용은 지어내지 말고, 그럴 때는 "
    "'제공된 자료에서 답을 찾을 수 없습니다'라고만 답하라. "
    "날짜·연도·숫자·이름은 반드시 [문서]에 적힌 것을 찾아 그대로 옮기고, 네 기억 속 값을 쓰지 마라. "
    "조직표·명단·일정표 같은 표나 목록의 내용을 답할 때는 문서에 있는 항목을 빠짐없이 "
    "그대로 옮겨라 — 담당자 이름 등 세부 정보를 생략하거나, 문서에 없는 '등'을 붙여 "
    "임의로 줄이지 마라. "
    "표에서 부서·분류명 칸이 비어 있거나 생략된 행은 바로 위 행과 같은 부서·분류 "
    "소속이다 — 그런 행을 새로운 부서로 만들지 마라. "
    "특정 부서에 대해 물으면 그 부서에 속한 모든 역할(부장·기획 등)과 담당자를 "
    "빠짐없이 나열하라 — 일부 역할만 골라 부서 전체인 것처럼 답하지 마라. "
    "특정 역할이나 담당자 한 명을 물으면 그것만 간결히 답하라. "
    "문서의 표를 '|' 기호 그대로 복사하지 말고, 조직표는 부서별로 묶어 "
    "'역할(담당자 이름): 업무 내용' 형태의 줄 목록으로 일관되게 정리하라. "
    "이전 대화가 있으면 맥락을 이어서 답하라."
    "마크다운이나 HTML 서식(별표, 굵게, 목록 기호, 태그 등)을 쓰지 말고 순수 텍스트로만 답하라."
)
# 하위 호환 — 옛 코드가 SYSTEM_PROMPT 를 import 한다
SYSTEM_PROMPT = SYSTEM_PROMPT_FULL


def system_prompt_for(mode: str) -> str:
    return SYSTEM_PROMPT_FULL if mode == "full" else SYSTEM_PROMPT_VOICE


def build_prompt(question: str, chunks: list[dict], mode: str = "voice") -> str:
    """검색된 근거 조각 + 질문을 하나의 프롬프트 문자열로 조합한다.

    같은 문서의 연속 페이지는 하나의 [문서 N] 블록으로 합친다 — 별개 블록으로 주면
    LLM이 페이지에 걸친 표를 서로 다른 문서로 취급해 뒷페이지를 누락하는 것이 실측됐다.
    """
    if chunks:
        blocks = []
        i, n = 0, 1
        while i < len(chunks):
            j = i
            while (
                j + 1 < len(chunks)
                and chunks[j + 1].get("source_file") == chunks[i].get("source_file")
                and isinstance(chunks[j].get("page_number"), int)
                and isinstance(chunks[j + 1].get("page_number"), int)
                and chunks[j + 1]["page_number"] == chunks[j]["page_number"] + 1
            ):
                j += 1
            group = chunks[i : j + 1]
            src = group[0].get("source_file", "?")
            if len(group) > 1:
                label = (
                    f'{src} p{group[0]["page_number"]}~p{group[-1]["page_number"]}'
                    " — 같은 내용이 여러 페이지에 걸쳐 이어짐"
                )
            else:
                label = f'{src} p{group[0].get("page_number", "?")}'
            # 뜻이 정해지지 않은 괄호 숫자("학생회선거(5,6)", "255(5)")는 모델에게 보이지 않게 지운다 — 실측(09-22 수동 #58):
            # 규칙("괄호 값은 해석하지 마라")을 줘도 "5학년과 6학년을 대상으로" 라고 지어냈다. 검증 컨텍스트(verify.context_text)는
            # 원문 그대로라 숫자 대조에는 영향이 없다.
            body = "\n".join(strip_paren_numbers(c["text"]) for c in group)
            blocks.append(f"[문서 {n}] ({label})\n{body}")
            n += 1
            i = j + 1
        context = "\n\n".join(blocks)
        # 핵심 근거: 검색이 질문과 가장 잘 맞춘 표 행·설명 줄 (expansion 이 페이지마다 붙임). 페이지 전문보다
        # 앞에 두어 모델이 긴 표에서 헤매지 않게 한다 — 실측: "입학식은?" 에 옆 페이지 졸업식 날짜를 섞던 결함.
        lines = [strip_paren_numbers(h) for c in chunks for h in (c.get("highlights") or [])]
        if lines:
            hl = "\n".join(f"- {h}" for h in lines)
            context = f"[핵심 근거] (아래 [문서] 중 질문과 가장 잘 맞는 줄들 — 답은 여기서 먼저 찾아라)\n{hl}\n\n{context}"
    else:
        context = "(관련 문서를 찾지 못함)"

    tail = QUOTE_FORMAT_RULE if (mode == "voice" and QUOTE_FIRST) else "[답변]"
    return f"[문서]\n{context}\n\n[질문]\n{question}\n\n{tail}"


# 인용 우선 형식 — 모델이 답하기 전에 [문서] 구절을 그대로 복사하게 한다 (config.QUOTE_FIRST 주석 참고).
# 실측(09-17): 7B 는 인용은 정확한데 옮겨 쓰는 단계에서 틀린다. 인용을 프로그램이 검증·활용한다.
QUOTE_LABEL = "근거:"
ANSWER_LABEL = "답변:"
QUOTE_FORMAT_RULE = (
    f"먼저 '{QUOTE_LABEL}' 줄에 질문의 답이 적힌 [문서]의 구절(한 줄, 표의 행이면 그 행)을 한 글자도 바꾸지 말고 "
    f"그대로 복사하라. 그 다음 줄 '{ANSWER_LABEL}' 에는 그 구절에 적힌 값을 그대로 옮기되, 질문을 되받아 "
    "'무엇은 무엇입니다' 형태의 완전한 문장으로 답하라 — 값이나 표 칸만 달랑 쓰지 마라. "
    "그 구절에 값과 함께 적힌 단서(당시 명칭, 조건, 괄호 안 설명, 대상)가 있으면 빠뜨리지 말고 같이 말하라. "
    "숫자 바로 뒤 괄호 안에 적힌 숫자는 별개의 부가 값이다 — 앞 숫자와 붙여 읽지 말고, "
    "그 괄호 값은 뜻을 알 수 없으니 답에 넣지도, 해석하지도 마라.\n"
    f"{QUOTE_LABEL}"
)
# 하위 호환 이름
RETRY_QUOTE_LABEL = QUOTE_LABEL
RETRY_ANSWER_LABEL = ANSWER_LABEL


def retry_prompt(bad_items: list[str], evidence_lines: Optional[list[str]] = None) -> str:
    """검증 실패 후 재생성 요청 메시지 — 직전 답변 다음 user 턴으로 붙인다.
    bad_items: 근거에 없는 숫자·날짜, 또는 "담당자 이름"(verify.MISSING_PERSON — 답에 빠진 것)."""
    missing = [b for b in bad_items if b == "담당자 이름"]
    off = [b for b in bad_items if b == "질문과 무관"]
    rows = [b for b in bad_items if "≠" in b]          # "2층≠1층": 질문 대상이 적힌 줄의 값이 아님
    refused = [b for b in bad_items if b.startswith("근거 있는 거절: ")]   # verify.REFUSED_WITH_EVIDENCE + 근거 줄
    lack = [b for b in bad_items if b.startswith("항목 누락(")]           # verify.MISSING_PART — 복합 질문의 한 항목만 답함
    wrong = [b for b in bad_items if b not in missing + off + rows + refused + lack]
    parts = []
    for r in refused:
        line = r.split(": ", 1)[1]
        parts.append(f"방금 '찾을 수 없다'고 답했지만 [핵심 근거]에 질문과 관련된 줄이 있다: '{line}'. "
                     "그 줄에 질문의 답이 적혀 있으면 그 값으로 답하고, 정말 없으면 그대로 답하라.")
    if wrong:
        parts.append(f"방금 답변에 쓴 {', '.join(wrong)} 은(는) [문서]의 근거 구절에 없다.")
    if rows:
        got = ", ".join(r.split("≠")[0] for r in rows)
        parts.append(f"방금 답변의 {got} 은(는) 질문이 묻는 대상이 적힌 근거 줄의 값이 아니다 — 질문의 대상과 같은 줄에 적힌 값으로 답하라.")
        if evidence_lines:
            parts.append("아래는 원래 질문의 두 항목과 각각의 근거다.\n" + "\n\n".join(evidence_lines) +
                         "\n틀린 항목만 답하지 말고 두 항목을 모두 포함한 완성 답변을 작성하라. "
                         "각 항목의 이름과 해당 근거의 값을 별도 문장으로 말하라.")
    if lack:
        items = ", ".join(b[len("항목 누락("):-1].replace(" 값", "") for b in lack)
        parts.append(f"이 질문은 두 가지를 묻는데 방금 답변에는 '{items}' 쪽의 답이 빠졌다 — 항목마다 [문서]에서 그 항목이 "
                     "적힌 줄을 따로 찾아 '가: …, 나: …' 처럼 각각 답하라. 한 항목이 [문서]에 없으면 그 항목만 자료에 없다고 답하라.")
    if missing:
        parts.append("방금 답변에는 질문이 묻는 사람의 이름이 빠졌다 — 근거 구절에 적힌 담당자 이름을 그대로 넣어 답하라.")
    if off:
        parts.append("방금 답변은 질문에 대한 답이 아니다 — 질문이 묻는 것이 [문서]에 있으면 그것으로 답하고, "
                     "없으면 '제공된 자료에서 답을 찾을 수 없습니다'라고만 답하라.")
    return " ".join(parts) + f" 기억으로 답하지 말고 [문서]를 다시 읽어라. {QUOTE_FORMAT_RULE}"


def split_quote_answer(text: str) -> tuple[str, str]:
    """모델 출력을 (답변, 인용 구절)로 나눈다. '답변:' 이 없으면 (원문, '')."""
    text = text.strip()
    idx = text.rfind(ANSWER_LABEL)
    if idx < 0:
        return text, ""
    quote = text[:idx]
    if quote.startswith(QUOTE_LABEL):
        quote = quote[len(QUOTE_LABEL):]
    return text[idx + len(ANSWER_LABEL):].strip(), quote.strip()


def split_retry_answer(text: str) -> str:
    """하위 호환 — 답변 부분만."""
    return split_quote_answer(text)[0]


# 한 줄 재답: 검증에 걸려 문서 한 줄을 그대로 읽어 주기 직전에, 그 한 줄만 주고 문장으로 바꿔 달라고 한 번 더 묻는다.
# 실측(09-21 키오스크): "전체 학생수는?" — 모델이 페이지 전체를 보고는 두 번 거절했지만 답은 "출신 학교별 학생수 — 계: … 255"
# 한 줄에 있었다. 표 행을 그대로 읽어 주면("자료에는 다음과 같이 적혀 있습니다. …") 값은 맞아도 안내문으로 어색하다.
LINE_ANSWER_SYSTEM = (
    "너는 안내 도우미다. 아래 [자료 한 줄]만 근거로 [질문]에 한 문장으로 답하라. "
    "줄에 적힌 값(숫자·날짜·이름)을 글자 그대로 쓰고, 계산하거나 다른 값을 더하지 마라. "
    "표 제목이나 열 이름을 나열하지 말고 질문이 묻는 값만 말하라. "
    "질문이 '전체·총·합계'를 물으면 '계·합계·총계' 값이 답이다. "
    "줄에 답이 없으면 정확히 '없음' 이라고만 답하라. 마크다운·기호 없이 순수 문장으로."
)


def line_answer_messages(question: str, line: str) -> list[dict]:
    return [
        {"role": "system", "content": LINE_ANSWER_SYSTEM},
        {"role": "user", "content": f"[자료 한 줄]" + chr(10) + line + chr(10) + chr(10) + f"[질문] {question}"},
    ]


def build_messages(
    question: str, chunks: list[dict], history: Optional[list[dict]] = None, mode: str = "voice"
) -> list[dict]:
    """system 지침(mode 별) + 직전 문답(멀티턴) + 근거·질문 프롬프트를 채팅 메시지 배열로 조합한다."""
    messages = [{"role": "system", "content": system_prompt_for(mode)}]
    for turn in history or []:
        q, a = turn.get("question", ""), turn.get("answer", "")
        if q and a:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": build_prompt(question, chunks, mode)})
    return messages
