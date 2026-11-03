"""새 채점은 pass/fail/review. 불명확한 자연어를 억지로 자동 통과시키지 않는다."""
import re
from rag.validation import quantities


def assess(answer: str, rules: list[dict]) -> tuple[str, list[str]]:
    if not answer.strip():
        return "fail", ["빈 답변"]
    errors, review = [], []
    # 규칙의 대상이 등장하는 문장에 값이 앞뒤로 놓인 경우를 모두 허용한다.
    sentences = [s.strip() for s in re.split(r"(?<=[다요][.!?])\s+|\n+|[,，](?!\d{3})", answer) if s.strip()]
    for rule in rules:
        subject = rule.get("subject", [])
        hits = [s for s in sentences if any(x.replace(" ", "") in s.replace(" ", "") for x in subject)] if subject else [answer]
        if not hits:
            errors.append(f"대상 누락: {subject}")
            continue
        if subject and any(any(other.replace(" ", "") in h.replace(" ", "")
                               for r in rules if r is not rule for other in r.get("subject", [])) for h in hits):
            review.append(f"같은 문장에 여러 대상: {subject}")
            continue
        actual = quantities(" ".join(hits))
        expected = set().union(*(quantities(v) for v in rule["values"]))
        found = [value in actual or (value[1] != "원" and (value[0], "") in actual) for value in expected]
        if not (any(found) if rule.get("alternatives") else all(found)):
            errors.append(f"기대 값·단위 없음: {subject} {rule['values']}")
        units = {unit for _, unit in expected}
        extra = {v for v in actual if v[1] in units and v not in expected}
        if extra:
            errors.append(f"다른 값 동시 포함: {subject} {sorted(map(str, extra))}")
    return ("fail" if errors else "review" if review else "pass"), errors + review


def refusal(answer: str) -> tuple[str, list[str]]:
    if not answer.strip():
        return "fail", ["빈 응답은 거절 답변이 아님"]
    # 근거가 없다는 메타데이터만 보고 통과시키지 않는다. 부분 거절은 수동 검토.
    if re.search(r"(?:확인하지 못|찾을 수 없|확인하기 어렵|안내드리기 어렵|확인이 필요)", answer):
        if re.search(r"\d|http|[A-Za-z]{2,}", answer):
            return "review", ["거절과 구체적 정보가 섞임"]
        if len(answer) > 140:
            return "review", ["긴 거절문 내용 확인 필요"]
        return "pass", []
    return "review", ["거절·인사·관련 없는 응답인지 내용 확인 필요"]
