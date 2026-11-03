"""새 검사 로그를 기존 통과 답변 문항과 대조한다. 모델을 호출하지 않는다."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("reports", nargs="+", type=Path)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    cases = {}
    for path in (root / "golden").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cases[(row["id"], row["question"])] = row.get("institution")
    baseline = {}
    for path in args.reports:
        for row in json.loads(path.read_text(encoding="utf-8")):
            key = (row["question"], cases.get((row["id"], row["question"])))
            baseline.setdefault(key, []).append(bool(row.get("answer_ok")))
    checked = flagged = 0
    for line in args.trace.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (row["question"], row.get("institution_id"))
        if key not in baseline or not all(baseline[key]):
            continue
        checked += 1
        errors = row.get("result", {}).get("unverified", [])
        if errors:
            flagged += 1
            print(f"기존 통과 문항 검출: {row['question']} {errors}")
    print(f"기존 보고서에서 항상 통과한 문항의 새 응답: {flagged}/{checked}건 검출")
    print("검출은 오탐 확정이 아닙니다. 새 답변 자체의 오류인지 인용과 함께 검토하세요.")


if __name__ == "__main__":
    main()
