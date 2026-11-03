"""원문 위치를 보존한 값 후보 관측. 정오 판정이나 답변 치환에는 사용하지 않는다."""
import re

from .validation import normalized


# 실측(261102_7, m17/b01/a22): 숫자 결합·연도 변형·금액 단위 오류.
# 범위·괄호는 원문 그대로 묶는다. 생략된 단위나 괄호의 의미는 추론하지 않는다.
_NUMBER = r"\d+(?:[.,]\d+)*"
_UNIT = r"(?:백만원|천만원|억원|만원|천원|원|퍼센트|%|학급|개월|교시|학년|호선|층|명|개|점|년|월|일|회|종|실|TB|PB|PF|㎡)"
_VALUE = re.compile(
    rf"(?<![\d.,]){_NUMBER}(?:\s*{_UNIT})?"
    rf"(?:\s*(?:~|～|∼|–|부터|-)\s*{_NUMBER}(?:\s*{_UNIT})?)*"
    rf"(?:\s*\(\s*{_NUMBER}\s*\))?", re.I)


def collect(evidence):
    """후보 ID는 원문 근거 ID와 문자 위치에 종속된다. 숫자 의미는 판정하지 않는다."""
    result = {}
    for alias, ev in evidence.items():
        rows = []
        for match in _VALUE.finditer(ev.text):
            start, end = match.span()
            rows.append({"id": f"{ev.id}:{start}:{end}", "start": start, "end": end,
                         "raw": ev.text[start:end],
                         "context": ev.text[max(0, start - 80):min(len(ev.text), end + 80)]})
        result[alias] = rows
    return result


def audit(items, candidates, allowed):
    """문자열 대응만 기록. 불일치는 오답을 뜻하지 않으며 단위 환산도 하지 않는다."""
    rows = []
    for item in items:
        for index, claim in enumerate(item.claims):
            permitted = claim.evidence_id in allowed.get(item.part_id, [])
            matches = [c["id"] for c in candidates.get(claim.evidence_id, [])
                       if normalized(c["raw"]) == normalized(claim.value)] if permitted else []
            status = ("outside_allowed_evidence" if not permitted else
                      "unique_exact_candidate" if len(matches) == 1 else
                      "multiple_exact_candidates" if matches else "no_exact_candidate")
            rows.append({"part_id": item.part_id, "claim_index": index,
                         "evidence_id": claim.evidence_id, "value": claim.value,
                         "status": status, "candidate_ids": matches})
    return rows
