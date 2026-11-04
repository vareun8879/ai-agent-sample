"""이전 trace를 새 검증기로 다시 읽어 오류 문자열·최종 응답의 동등성을 확인한다. 모델 호출 없음."""
import argparse
from collections import Counter
import json
from pathlib import Path

from rag.contracts import CheckedItem, Evidence, ItemAnswer, Part, Status
from rag.parsing import ParseError, parse
from rag.rendering import render
from rag.validation import check


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists() or args.output.resolve() == args.trace.resolve():
        raise SystemExit(f"Refusing to overwrite: {args.output}")
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
