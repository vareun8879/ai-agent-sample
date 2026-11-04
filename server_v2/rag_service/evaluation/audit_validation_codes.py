"""이전 trace를 새 검증기로 다시 읽어 오류 문자열·최종 응답의 동등성을 확인한다. 모델 호출 없음."""
import argparse
from collections import Counter
import json
from pathlib import Path

from rag.contracts import CheckedItem, Evidence, ItemAnswer, Part, Status
from rag.parsing import ParseError, parse
from rag.rendering import render
from rag.validation import check
from .regrade import regrade_row


def audit_trace(row):
    parts = [Part(**p) for p in row["parts"]]
    evidence = {key: Evidence(**value) for key, value in row["evidence"].items()}
    allowed = row["allowed"]
    best, differences, codes = {}, [], Counter()
    for index, attempt in enumerate(row.get("attempts", []), 1):
        try:
            items = parse(attempt.get("raw", ""), parts)
        except ParseError:
            if "error" not in attempt:
                differences.append(f"attempt {index}: previously parsed output cannot be parsed")
            continue
        checked = check(items, evidence, allowed)
        problems = [f"{c.item.part_id}: {e}" for c in checked for e in c.errors]
        # 집합 순회 순서는 프로세스마다 달라질 수 있어 오류의 내용으로 비교한다.
        if "error" in attempt or set(problems) != set(attempt.get("errors", [])):
            differences.append(f"attempt {index}: validation messages changed")
        for c in checked:
            codes.update(issue.code.value for issue in c.issues)
            if c.item.part_id not in best or (c.item.status == Status.SUPPORTED and not c.errors):
                best[c.item.part_id] = c
    if not evidence:
        best = {p.id: CheckedItem(ItemAnswer(p.id, Status.INSUFFICIENT, "")) for p in parts}
    result = render(parts, list(best.values()), evidence, row["validation_mode"]).response()
    previous = row["result"]
    for field in ("answer", "sources", "grounded"):
        if result[field] != previous.get(field):
            differences.append(f"final {field} changed")
    if set(result["unverified"]) != set(previous.get("unverified", [])):
        differences.append("final unverified messages changed")
    return differences, codes


def compare_first_attempt(row):
    """검사 변경의 영향만 비교한다. 새 재시도의 답변은 예측하지 않는다."""
    attempts = row.get("attempts", [])
    if not attempts:
        return {"state": "no_attempt", "changes": []}
    first = attempts[0]
    parts = [Part(**p) for p in row["parts"]]
    try:
        items = parse(first.get("raw", ""), parts)
    except ParseError:
        return {"state": "parse_failure", "changes": []}
    if "issues" not in first:
        raise ValueError("comparison requires a trace with structured validation issues")
    evidence = {key: Evidence(**value) for key, value in row["evidence"].items()}
    checked = check(items, evidence, row["allowed"])
    changes = []
    for c in checked:
        before = {(i["code"], i["message"]) for i in first["issues"]
                  if i.get("part_id") == c.item.part_id}
        after = {(i.code.value, i.message) for i in c.issues}
        if before != after:
            changes.append({"part_id": c.item.part_id, "answer": c.item.answer,
                            "removed": sorted(before - after), "added": sorted(after - before),
                            "before_clean": not before, "after_clean": not after})
    old_errors, new_errors = bool(first.get("errors")), any(c.errors for c in checked)
    return {"state": "parsed", "items": len(items),
            "supported_items": sum(item.status == Status.SUPPORTED for item in items),
            "changes": changes, "retry_skip_candidate": len(attempts) > 1 and not new_errors,
            "new_retry_needed": not old_errors and new_errors,
            "retry_feedback_changed": old_errors and new_errors and bool(changes)}


def compare_reports(paths):
    root = Path(__file__).resolve().parents[1]
    rubrics = json.loads((root / "evaluation/rubrics.json").read_text(encoding="utf-8"))
    changes, questions = [], {}
    total = rule_rows = 0
    for path in paths:
        name = path.name.removeprefix("last_report_").rsplit("_", 1)[0]
        cases = {c["id"]: c for c in (json.loads(line) for line in
                 (root / "golden" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
        for row in json.loads(path.read_text(encoding="utf-8")):
            case = cases[row["id"]]
            if case.get("optional"):
                continue
            if row.get("grading_version") != 3:
                raise ValueError("use baseline reports graded with version 3")
            total += 1
            rules = rubrics.get(name, {}).get(row["id"])
            rule_rows += bool(rules)
            updated = regrade_row(case, row, rules)
            key = (row["question"], case.get("institution"))
            questions.setdefault(key, []).append(bool(row["adjusted_answer_ok"]))
            if (row["adjusted_answer_ok"] != updated["adjusted_answer_ok"] or
                    row["rubric_status"] != updated["rubric_status"]):
                changes.append({"report": path.name, "id": row["id"], "answer": row["answer"],
                                "before": row["adjusted_answer_ok"], "after": updated["adjusted_answer_ok"],
                                "before_status": row["rubric_status"], "after_status": updated["rubric_status"],
                                "reasons": updated["rubric_reasons"]})
    return {"rows": total, "rule_rows": rule_rows, "changes": changes}, questions


def comparison_summary(args):
    reports, baseline = compare_reports(args.reports)
    counts, changes = Counter(), []
    for number, line in enumerate(args.trace.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        counts["records"] += 1
        observed = compare_first_attempt(row)
        counts[observed["state"]] += 1
        for key in ("items", "supported_items", "retry_skip_candidate", "new_retry_needed", "retry_feedback_changed"):
            counts[key] += observed.get(key, 0)
        if observed["changes"]:
            previous = baseline.get((row["question"], row.get("institution_id")))
            changes.append({"line": number, "request_id": row.get("request_id"),
                            "question": row["question"],
                            "baseline_all_pass": all(previous) if previous else None,
                            **observed})
    return {"mode": "comparison", "counts": dict(counts), "changed_requests": changes,
            "report_comparison": reports,
            "note": "First-output comparison only. New retry outputs and final latency require live measurement."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--compare", action="store_true", help="검사 변경 전후의 첫 시도와 기존 보고서 판정 비교")
    ap.add_argument("--reports", nargs="*", type=Path, default=[])
    args = ap.parse_args()
    if args.output.exists() or args.output.resolve() == args.trace.resolve():
        raise SystemExit(f"Refusing to overwrite: {args.output}")
    if args.compare:
        if not args.reports:
            ap.error("--compare requires --reports")
        summary = comparison_summary(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
        print(json.dumps({"counts": summary["counts"],
                          "changed_requests": len(summary["changed_requests"]),
                          "report_changes": len(summary["report_comparison"]["changes"]),
                          "output": str(args.output)}, ensure_ascii=False, indent=2))
        if not summary["counts"].get("parsed"):
            raise SystemExit(1)
        return
    checked = skipped = 0
    mismatches, codes = [], Counter()
    with args.trace.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not all(key in row for key in ("result", "parts", "evidence", "allowed", "validation_mode")):
                skipped += 1
                continue
            differences, observed = audit_trace(row)
            checked += 1
            codes.update(observed)
            if differences:
                mismatches.append({"line": number, "request_id": row.get("request_id"),
                                   "question": row.get("question"), "differences": differences})
    summary = {"checked": checked, "skipped": skipped, "mismatch_count": len(mismatches),
               "mismatches": mismatches, "issue_counts": dict(codes)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if mismatches or not checked:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
