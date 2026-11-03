"""2단계 보정 — 자동 이어쓰기의 판단과 회차 정리.

7B급 LLM은 긴 나열(조직표 전체 등)을 출력 한도에 닿기 전에 스스로 끝내버린다
(실측: 731/4096토큰에서 done_reason=stop, 출력 한도를 16000으로 올려도 동일 지점에서
멈춤, 프롬프트 강제도 무효). 유일하게 통한 우회가 "직전 답변 다음부터 이어서 써라"를
자동으로 재요청하는 것인데, 이 이어쓰기가 만들어내는 부산물들이 실측으로 여럿
확인됐다. 이 모듈은 그 판단(언제 이어쓸지)과 방어(회차 결과 정리)를 담당한다.

이어쓰기 루프 자체(LLM 호출 반복)는 generator.py에 있다.

실측된 이어쓰기 부산물과 대응:
- 회차가 "끝" 마커를 내용과 함께 씀        → 마커만 떼고 다음 회차가 재확인
- 이을 게 없는데 근거 없음 안내문을 냄      → 폐기하고 종료
- "모든 항목을 다루었습니다" 메타 발언       → 폐기하고 종료
- 이미 답한 내용을 제목만 바꿔 그대로 반복   → 문자열 창 매칭으로 감지, 폐기
- 같은 항목을 표현만 바꿔(축약↔상세) 재설명 → 항목 라벨 재등장으로 감지, 해당 줄 제거
- 1차 답변이 나열형이 아닌데 이어쓰기 발동   → looks_enumeration으로 사전 차단
  (근거 없음 안내문·짧은 서술형에 이어쓰기를 걸면 기준점 없이 표 중간부터 쏟아내거나
  대화 이력의 다른 주제를 끌어와 지어내는 기형 답변이 된다)
"""

import re
from typing import Optional

# "마지막 항목 바로 다음부터 순서대로"가 안정적이다 — "빠진 항목을 대조해 찾아라"는
# 모델이 중간 항목(1차가 멈춘 지점 직후)을 일관되게 건너뛰는 문제가 실측됐다.
# [문서]에만 근거하라는 재확인도 명시 — 이전 대화의 다른 주제 내용을 끌어오는 것을 막는다.
CONTINUE_PROMPT = (
    "직전 답변이 다룬 마지막 항목의 바로 다음 항목부터, [문서]에 나온 순서대로 나머지 "
    "전부를 이어서 같은 형식으로 설명하라. 오직 이번 질문의 [문서]에 있는 내용만 사용하고 "
    "이전 대화의 다른 주제 내용을 가져오지 마라. "
    '남은 항목이 없으면 "끝"이라고만 답하라.'
)

_WS = re.compile(r"\s+")
# 이어쓰기 회차의 줄이 "라벨: ..." 또는 "라벨(..." 형태로 시작하면 그 라벨을 추출한다.
# 이미 다룬 항목을 표현만 바꿔(축약↔상세) 다시 설명하는 회차를 잡기 위한 것 —
# 문자열 그대로는 안 겹치지만 "같은 항목을 또 설명한다"는 신호는 라벨 재등장으로 잡힌다.
_ITEM_LABEL = re.compile(r"^[\s\-*#]*\**([가-힣A-Za-z0-9·/ ]{2,20}?)\**\s*[:(（]")


def is_done_marker(text: str) -> bool:
    """이어쓰기 요청에 모델이 '더 없음'으로 답했는지 판별한다 ("끝", '"끝"', "### 끝" 등)."""
    s = text.strip().strip('"').strip(".")
    return len(s) <= 3 and "끝" in s


def looks_enumeration(first_piece: str) -> bool:
    """1차 답변이 이어쓰기 대상(나열형)인지 판별한다.

    근거 없음 안내문이나 짧은 서술형 답변에 이어쓰기를 걸면, "마지막 항목 다음부터"
    라는 지시가 기준점 없이 실행되어 표 중간부터 쏟아내는 기형 답변이 된다(실측).
    """
    s = first_piece.strip()
    return s.count("\n") >= 5 and not s.startswith("제공된 자료에서")


def clean_continuation(piece: str, accumulated: str) -> tuple[Optional[str], bool]:
    """이어쓰기 회차의 결과를 정리한다 → (덧붙일 텍스트 또는 None, 루프 중단 여부).

    accumulated는 지금까지 사용자에게 나간 답변 전체 — 중복 판정의 기준이 된다.
    None 반환은 "이 회차는 노출할 내용이 없음"을 뜻하고, 이때 루프도 중단된다.
    """
    s = piece.strip()
    if not s or is_done_marker(s):
        return None, True
    if s.startswith("제공된 자료에서"):
        return None, True
    # "모든 항목을 이미 다루었습니다" 같은 메타 발언으로 끝났음을 알리는 회차 —
    # 내용이 아니므로 노출하지 않고 종료한다
    if len(s) < 200 and re.search(r"(모두 다루|이미.*답변|더 이상.*(없습니다|않습니다))", s):
        return None, True

    # 내용 말미의 '끝' 마커는 노출만 막고 루프는 계속 돈다 — 모델이 "끝"이라 해놓고
    # 실제로는 항목이 남은 경우가 많아서, 진짜 끝인지는 다음 회차가 재확인한다
    # (정말 끝이면 다음 회차가 단독 '끝'(2토큰)을 반환해 저비용으로 종료된다)
    lines = s.splitlines()
    while lines and is_done_marker(lines[-1].lstrip("#").strip()):
        lines.pop()
    s = "\n".join(lines).strip()
    if not s:
        return None, True

    # 중복 검사 ① 문자열 그대로 반복: 회차 앞부분(정규화 120자)을 40자 창 3개로 잘라
    # 기존 답변에서 찾는다 — 제목만 바꿔 달고("...(계속)") 본문을 그대로 반복하는 사례.
    norm_piece, norm_acc = _WS.sub("", s), _WS.sub("", accumulated)
    windows = [norm_piece[i : i + 40] for i in (0, 40, 80) if len(norm_piece) >= i + 40]
    if windows and sum(1 for w in windows if w in norm_acc) >= max(1, len(windows) - 1):
        return None, True

    # 중복 검사 ② 같은 항목을 표현만 바꿔 재설명: 라벨(항목명)이 이미 accumulated에
    # "라벨:" 형태로 등장했다면 그 줄만 걷어낸다. 남은 새 항목이 하나도 없으면 회차 폐기.
    kept, any_item, any_new_item = [], False, False
    for line in s.splitlines():
        stripped = line.strip()
        # 헤더 줄(#로 시작)은 항목이 아니라 구조 표시일 뿐이라 라벨 판정에서 제외한다 —
        # 안 그러면 "교과업무 (계속)"이 그 자체로 "새 항목" 취급된다
        if stripped.startswith("#"):
            kept.append(line)
            continue
        m = _ITEM_LABEL.match(stripped)
        if m:
            any_item = True
            label = re.escape(m.group(1).strip())
            if re.search(label + r"\s*[:(（]", accumulated):
                continue  # 이미 다룬 항목 — 이 줄은 버린다
            any_new_item = True
        kept.append(line)
    if any_item and not any_new_item:
        return None, True

    # 후처리: 위 필터링으로 내용이 다 빠져 헤더만 남은 절은 헤더째로 제거한다.
    # 다음 헤더(또는 끝)까지의 구간에 실질 내용(빈 줄이 아닌 줄)이 하나도 없으면 빈 절.
    final = []
    for i, line in enumerate(kept):
        if line.strip().startswith("#"):
            section = []
            for nxt in kept[i + 1 :]:
                if nxt.strip().startswith("#"):
                    break
                section.append(nxt)
            if not any(x.strip() for x in section):
                continue
        final.append(line)
    s = "\n".join(final).strip()
    if not s:
        return None, True

    return s, False
