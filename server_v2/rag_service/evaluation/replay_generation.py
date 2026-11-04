"""저장된 근거로 생성 입력을 비교한다. 검색·서비스 설정은 변경하지 않는다."""
import argparse
import asyncio
from dataclasses import asdict, replace
import json
from pathlib import Path
import re
import time

import httpx

from rag.config import Settings
from rag.contracts import Evidence, Part
from rag.generation import Generator, SYSTEM
from rag.parsing import parse
from rag.validation import check


def variant_input(original, allowed, variant, drop_ids):
    evidence = dict(original)
    if variant == "fact-first":
        evidence = dict(sorted(evidence.items(), key=lambda pair: pair[1].kind != "fact"))
    elif variant == "drop":
        missing = set(drop_ids) - evidence.keys()
        if missing or not drop_ids:
            raise ValueError(f"drop requires existing evidence IDs: {sorted(missing)}")
        evidence = {key: value for key, value in evidence.items() if key not in drop_ids}
    elif variant == "paren-space":
        evidence = {key: replace(value, text=re.sub(r"(?<=\d)\((?=\d)", " (", value.text))
                    for key, value in evidence.items()}
    permitted = {key: [eid for eid in ids if eid in evidence] for key, ids in allowed.items()}
    if not evidence or any(not ids for ids in permitted.values()):
        raise ValueError("variant leaves a question without evidence")
    if variant != "control" and evidence == original and list(evidence) == list(original):
        raise ValueError(f"{variant}: input unchanged; no model calls made")
    return evidence, permitted


async def run(args):
    cases = {}
    for path in sorted(Path("golden").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            case = json.loads(line)
            if case["id"] in args.cases:
                if case["id"] in cases:
                    raise ValueError(f"ambiguous case ID: {case['id']}")
                cases[case["id"]] = case
    if set(args.cases) != cases.keys():
        raise ValueError(f"unknown cases: {set(args.cases) - cases.keys()}")
    traces = [json.loads(line) for line in args.trace.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    jobs = []
    for case_id in args.cases:
        case = cases[case_id]
        matches = [(n, row) for n, row in enumerate(traces, 1)
                   if row["question"] == case["question"]
                   and row.get("institution_id") == case.get("institution")]
        if len(matches) < args.occurrence:
            raise ValueError(f"{case_id}: trace occurrence {args.occurrence} missing")
        line_number, trace = matches[args.occurrence - 1]
        original = {key: Evidence(**value) for key, value in trace["evidence"].items()}
        parts = [Part(**part) for part in trace["parts"]]
        for variant in dict.fromkeys(["control", *args.variants]):
            evidence, allowed = variant_input(original, trace["allowed"], variant, args.drop_ids)
            jobs.append((case_id, line_number, trace, parts, original, evidence, allowed, variant))
    settings = Settings.from_env()
    calls = len(jobs) * args.repeat
    print(f"Planned calls: {calls}; no retries; mode={args.mode}; history=[]")
    print("Historical history/mode are not recorded: this is a controlled comparison, not exact replay.")
    if args.dry_run:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 기존 결과를 덮어쓰지 않고, 중간 실패 시에도 완료된 호출 원문을 보존한다.
    with args.output.open("x", encoding="utf-8") as output:
        async with httpx.AsyncClient() as client:
            generator = Generator(settings, client)
            for case_id, line_number, trace, parts, original, evidence, allowed, variant in jobs:
                for repeat in range(1, args.repeat + 1):
                    row = {"case_id": case_id, "variant": variant, "repeat": repeat,
                           "trace": str(args.trace), "trace_line": line_number,
                           "request_id": trace.get("request_id"), "question": trace["question"],
                           "parts": [asdict(p) for p in parts], "allowed": allowed,
                           "original_evidence": {k: asdict(v) for k, v in original.items()},
                           "display_evidence": {k: asdict(v) for k, v in evidence.items()},
                           "mode": args.mode, "history": [], "system": SYSTEM,
                           "model": settings.llm_model, "temperature": 0,
                           "max_tokens": settings.max_tokens, "raw": ""}
                    started = time.monotonic()
                    try:
                        row["raw"] = await generator.generate(trace["question"], parts, evidence,
                                                              allowed, [], args.mode)
                        items = parse(row["raw"], parts)
                        row["items"] = [asdict(item) for item in items]
                        row["errors"] = [f"{c.item.part_id}: {error}"
                                         for c in check(items, original, allowed) for error in c.errors]
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    row["seconds"] = round(time.monotonic() - started, 3)
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    answer = " / ".join(item["answer"] for item in row.get("items", []))
                    print(f"{case_id} {variant} {repeat}/{args.repeat} {row['seconds']:.1f}s: {answer}")
                    print(f"  original-source validation: {row.get('error', row.get('errors'))}")
                    if "error" in row:
                        raise RuntimeError(f"Replay stopped; partial results saved: {args.output}")
    print(f"Results: {args.output}; correctness requires source review (validation pass is not grading).")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", choices=["control", "fact-first", "drop", "paren-space"], required=True)
    ap.add_argument("--drop-ids", nargs="*", default=[])
    ap.add_argument("--occurrence", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--mode", choices=["voice", "full"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", help="validate inputs and print call count without calling LLM")
    args = ap.parse_args()
    if args.repeat < 1 or args.occurrence < 1:
        ap.error("repeat and occurrence must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
