"""실시간 평가와 보고서 재채점의 공통 판정. 서비스 답변에는 관여하지 않는다."""
from .rubric import assess, refusal


GRADING_VERSION = 3


def evaluate(case, answer, legacy_ok, rules=None, *, call_failed=False, truncated=False):
    if call_failed:
        status, reasons = "fail", ["서비스 호출 실패"]
    elif truncated:
        status, reasons = "review", ["답변이 잘렸을 가능성으로 전체 내용 확인 필요"]
    elif not case.get("grounded", True):
        status, reasons = refusal(answer)
    elif rules:
        status, reasons = assess(answer, rules)
    else:
        status, reasons = "legacy_only", []
    if call_failed or truncated:
        adjusted = False
    elif not case.get("grounded", True):
        adjusted = status == "pass"
    else:
        adjusted = bool(legacy_ok) and status not in ("fail", "review")
    return status, reasons, adjusted
