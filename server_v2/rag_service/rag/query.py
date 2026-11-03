from .contracts import Part
from .textutil import split_compound_question


def plan(question: str) -> list[Part]:
    # 기존의 실측된 분해 규칙만 이관. 분해 실패 시 원문을 보존한다.
    parts = split_compound_question(question) or [question]
    return [Part(f"q{i + 1}", q.strip()) for i, q in enumerate(parts)]
