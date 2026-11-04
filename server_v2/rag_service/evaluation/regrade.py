"""모델 없이 기존 보고서를 새 기준으로 재채점. 원본 보고서는 수정하지 않는다.

python -m evaluation.regrade /path/to/last_report_*_[123].json
"""
import argparse
import json
from pathlib import Path
import re
from .answer_grading import GRADING_VERSION, evaluate


def regrade_row(case, row, rules=None):
    updated = dict(row)
    truncated = row.get("answer_truncated", len(row.get("answer") or "") >= 300)
    status, reasons, adjusted = evaluate(
        case, row.get("answer") or "", row.get("answer_ok"), rules,
        call_failed=bool(row.get("call_failed") or "호출 실패" in row.get("why", "")),
        truncated=truncated)
    updated.update(rubric_status=status, rubric_reasons=reasons,
                   adjusted_answer_ok=adjusted, grading_version=GRADING_VERSION,
                   answer_truncated=truncated)
    # 예전 보고서에서 누락된 메타데이터를 정상/빈 값으로 추정하지 않는다.
    for field in ("sources", "grounded", "unverified"):
        updated.setdefault(field, None)
    updated.setdefault("citation_ok", row.get("retrieval_ok"))
    updated.setdefault("retrieval_basis", "answer_sources_legacy")
    updated.setdefault("retrieval_observed", False)
    updated.setdefault("actual_retrieval_ok", None)
    return updated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", type=Path)
    ap.add_argument("--fail-only", action="store_true", help="화면의 문항별 출력은 실패만 표시")
    ap.add_argument("--output-dir", type=Path, help="재채점 보고서를 별도 디렉토리에 저장 (기존 파일 덮어쓰기 금지)")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    rubrics = json.loads((Path(__file__).parent / "rubrics.json").read_text(encoding="utf-8"))
    for path in args.reports:
        destination = args.output_dir / path.name if args.output_dir else None
        if destination and (destination.resolve() == path.resolve() or destination.exists()):
            raise SystemExit(f"Refusing to overwrite report: {destination}")
        match = re.fullmatch(r"last_report_(.+)_\d+\.json", path.name)
        if not match:
            raise SystemExit(f"Unexpected report name: {path.name}")
        name = match[1]
        cases = {c["id"]: c for c in (json.loads(line) for line in
                 (root / "golden" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
        counts = {"pass": 0, "fail": 0, "review": 0, "legacy_only": 0}
        legacy = 0
        adjusted = 0
        changed = 0
        output_rows = []
        for row in json.loads(path.read_text(encoding="utf-8")):
            case = cases[row["id"]]
            updated = regrade_row(case, row, rubrics.get(name, {}).get(row["id"]))
            output_rows.append(updated)
            if case.get("optional"):
                continue
            legacy += bool(row["answer_ok"])
            status, reasons = updated["rubric_status"], updated["rubric_reasons"]
            counts[status] += 1
            adjusted += updated["adjusted_answer_ok"]
            old_adjusted = row.get("adjusted_answer_ok", row.get("answer_ok"))
            changed += old_adjusted != updated["adjusted_answer_ok"]
            if (not updated["adjusted_answer_ok"] and status != "review") or (
                    not args.fail_only and (status == "review" or old_adjusted != updated["adjusted_answer_ok"])):
                print(f"{path.name} {row['id']} adjusted={old_adjusted}->{updated['adjusted_answer_ok']} "
                      f"status={status}: {reasons or row.get('why', '')}")
        print(f"{path.name}: legacy_pass={legacy}, adjusted_pass={adjusted}, changed={changed}, new={counts}"
              " (review는 확인 필요로 분리하며 통과에는 포함하지 않음)")
        if destination:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as stream:
                json.dump(output_rows, stream, ensure_ascii=False, indent=2)
            print(f"Regraded report: {destination}")


if __name__ == "__main__":
    main()
