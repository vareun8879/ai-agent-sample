from dataclasses import asdict
from .contracts import CheckedItem, Outcome, Part, Status, ValidationCode, ValidationIssue

NO_ANSWER = "제공된 자료에서 답을 확인하지 못했습니다."


def render(parts: list[Part], checked: list[CheckedItem], evidence: dict, mode: str) -> Outcome:
    by_id = {c.item.part_id: c for c in checked}
    sentences, sources, errors, states = [], {}, [], []
    all_supported = True
    for part in parts:
        c = by_id.get(part.id)
        status = c.item.status if c else Status.INVALID
        issues = c.errors if c else ["출력 파싱 실패"]
        if issues:
            errors.extend(f"{part.id}: {e}" for e in issues)
            status = Status.INVALID
        accepted = c and c.item.status == Status.SUPPORTED and not issues
        # log에서는 검사 결과를 기록하되 문자열을 바꾸지 않는다. 정상 판정은 하지 않는다.
        visible = accepted or (mode == "log" and c and c.item.status == Status.SUPPORTED)
        if visible:
            sentences.append(c.item.answer)
            for claim in c.item.claims:
                if claim.evidence_id in evidence:
                    ev = evidence[claim.evidence_id]
                    sources[ev.id] = ev.source()
        else:
            prefix = f"‘{part.question}’ 항목은 " if len(parts) > 1 else ""
            sentences.append(prefix + ("자료의 근거가 서로 달라 확인이 필요합니다."
                                      if status == Status.CONFLICT else NO_ANSWER))
        all_supported = bool(all_supported and accepted)
        details = c.issues if c else [ValidationIssue(ValidationCode.PARSE_FAILURE, "출력 파싱 실패")]
        states.append({"part_id": part.id, "status": status.value, "errors": issues,
                       "issues": [asdict(issue) for issue in details]})
    return Outcome(" ".join(sentences), list(sources.values()), all_supported, errors, states)
