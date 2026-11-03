"""지난 골든 답변을 색인 사실 행과 모델 호출 없이 대조한다.

사용법: python audit_row_values.py golden/last_report_*_[123].json
       python audit_row_values.py --with-evidence golden/last_report_*_[123].json
보고서에는 source_file 이 없으므로 골든 문항의 doc·institution 과 보고서의
페이지 번호로 사실 행을 복원한다. 결과는 배포 전 오탐 조사용이며 답변을 바꾸지 않는다.
"""

import argparse
import json
import re
from pathlib import Path

from rag.verify import audit_answer_values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--chunks", type=Path,
                        default=Path(__file__).resolve().parents[1] / "indexing/output_chunks/chunks.jsonl")
    parser.add_argument("--with-evidence", action="store_true",
                        help="두 항목 모두 근거 행이 유일한 재생성 대상만 표시")
    args = parser.parse_args()

    facts: dict[tuple[str, int], list[str]] = {}
    with args.chunks.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("kind") == "fact":
                key = (row["source_file"], row["page_number"])
                facts.setdefault(key, []).append(row["text"].split("\n", 1)[-1].strip())

    checked_pass = flagged_pass = checked_fail = flagged_fail = 0
    for report in args.reports:
        match = re.fullmatch(r"last_report_(.+)_\d+\.json", report.name)
        if not match:
            parser.error(f"보고서 파일명 형식을 확인하세요: {report}")
        cases_path = Path(__file__).resolve().parent / "golden" / f"{match.group(1)}.jsonl"
        with cases_path.open(encoding="utf-8") as stream:
            cases = {c["id"]: c for c in (json.loads(line) for line in stream if line.strip())}
        rows = json.loads(report.read_text(encoding="utf-8"))
        for row in rows:
            case = cases.get(row["id"])
            if not case or not row.get("answer"):
                continue
            doc = case.get("doc", "")
            institution = case.get("institution", "")
            pages = set(row.get("pages") or [])
            lines = [text for (src, page), items in facts.items()
                     if page in pages and (not doc or doc in src)
                     and (not institution or src.startswith(f"{institution}/") or src.startswith("common/"))
                     for text in items]
            issues = audit_answer_values(row["question"], row["answer"], lines,
                                         with_evidence=args.with_evidence)
            passed = bool(row.get("answer_ok"))
            if passed:
                checked_pass += 1
                flagged_pass += bool(issues)
            else:
                checked_fail += 1
                flagged_fail += bool(issues)
            if issues:
                mark = "PASS" if passed else "FAIL"
                shown = [f"{issue} [{line}]" for issue, line, _ in issues] if args.with_evidence else issues
                print(f"{report.name} {row['id']} {mark}: {'; '.join(shown)}")
    print(f"통과 답변: {flagged_pass}/{checked_pass}건 표시, 실패 답변: {flagged_fail}/{checked_fail}건 표시")


if __name__ == "__main__":
    main()
