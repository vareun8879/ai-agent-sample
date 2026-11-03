"""모델 없이 기존 보고서를 새 기준으로 재채점. 원본 보고서는 수정하지 않는다.

python -m evaluation.regrade /path/to/last_report_*_[123].json
"""
import argparse
import json
from pathlib import Path
import re
from .rubric import assess, refusal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", type=Path)
    ap.add_argument("--fail-only", action="store_true", help="화면의 문항별 출력은 실패만 표시")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    rubrics = json.loads((Path(__file__).parent / "rubrics.json").read_text(encoding="utf-8"))
    for path in args.reports:
        match = re.fullmatch(r"last_report_(.+)_\d+\.json", path.name)
        if not match:
            raise SystemExit(f"Unexpected report name: {path.name}")
        name = match[1]
        cases = {c["id"]: c for c in (json.loads(line) for line in
                 (root / "golden" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
        counts = {"pass": 0, "fail": 0, "review": 0, "legacy_only": 0}
        legacy = 0
        adjusted = 0
        for row in json.loads(path.read_text(encoding="utf-8")):
            case = cases[row["id"]]
            if case.get("optional"):
                continue
            legacy += bool(row["answer_ok"])
            rules = rubrics.get(name, {}).get(row["id"])
            if "호출 실패" in row.get("why", ""):
                status, reasons = "fail", ["서비스 호출 실패"]
            elif not case.get("grounded", True):
                status, reasons = refusal(row.get("answer") or "")
            elif rules:
                status, reasons = assess(row.get("answer") or "", rules)
            else:
                status, reasons = "legacy_only", []
            if status in ("pass", "fail") and len(row.get("answer") or "") >= 300:
                status, reasons = "review", ["기존 보고서 답변이 300자에서 잘렸을 가능성"]
            counts[status] += 1
            adjusted += bool(row["answer_ok"]) and status != "fail"
            if status == "fail" or (status == "review" and not args.fail_only):
                print(f"{path.name} {row['id']} old={row['answer_ok']} new={status}: {reasons}")
        print(f"{path.name}: legacy_pass={legacy}, adjusted_pass={adjusted}, new={counts}"
              " (legacy_only는 기존 판정 유지, review는 확정 실패로 세지 않음)")


if __name__ == "__main__":
    main()
