"""골든 질문 세트로 RAG 품질을 채점한다 — "이 PDF가 정확하게 들어갔는가"의 실제 기준.

사용:
  cd rag_service
  python golden_eval.py golden/mangmi_2026.jsonl                # 검색만 채점 (엔진 in-process, LLM 불필요)
  python golden_eval.py golden/mangmi_2026.jsonl --answer       # 답변까지 채점 (rag_service HTTP :8001)
  python golden_eval.py golden/*.jsonl --answer --url http://127.0.0.1:8001 --key <INTERNAL_API_KEY>
  python golden_eval.py golden/mangmi_2026.jsonl --report golden/last_report.json

문항 형식(JSONL, 한 줄에 하나):
  {"id": "m01", "question": "입학식은 언제야?", "doc": "망미중학교", "pages": [49],
   "keywords": ["3.3|3월 3일"], "optional": false, "grounded": true, "note": "..."}
  - doc:      근거 source_file 에 포함돼야 하는 문자열 (생략 시 검사 안 함)
  - institution: 이 문항을 보낼 때의 기관 ID (docs/<기관ID>/ 문서용 — 키오스크가 보내는 institution_id 와 같은 값).
                 검색 범위가 공통 문서 + 그 기관 문서로 좁혀진다. 생략하면 전체 문서 (공통 문서 세트용)
  - pages:    이 중 하나라도 근거 페이지(페이지 병합 후)에 있으면 검색 통과
  - keywords: 각 항목은 "a|b|c" — 그중 하나가 답변에 있으면 그 항목 통과. 모두 통과해야 답변 통과
  - exclude:  이 값이 답변에 하나라도 있으면 실패 — 옆 행의 값과 헷갈리기 쉬운 문항용 (예: 상용화 100~250 을 물었는데
              시제품 행의 150 이 섞여 나오면 오답). 09-18 실측: 두 행 값을 다 읽어 주는 답이 keywords 만으로는 통과했다
  - grounded: false 면 검색은 "근거 0건" 이어야 통과, 답변은 근거 0건이거나 거절문("찾을 수 없…")이면 통과 — 무관 질문 차단 검사
  - optional: true 면 통과율 계산에서 제외 (예: VLM 전사가 있어야 맞출 수 있는 문항)

채점:
  검색 통과 = doc·pages 조건 충족 (또는 grounded=false 문항이 근거 0건)
  답변 통과 = 검색 통과 + keywords 모두 충족           (--answer 일 때만)
종료 코드: 필수 문항의 (답변 모드면 답변, 아니면 검색) 통과율이 --min-pass(기본 0.9) 미만이면 1.
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent


def load_cases(patterns: list[str]) -> list[dict]:
    cases = []
    for pat in patterns:
        files = sorted(glob.glob(pat)) or [pat]
        for f in files:
            with open(f, encoding="utf-8") as fh:
                for ln, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        cases.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        sys.exit(f"{f}:{ln} JSON 오류: {exc}")
    if not cases:
        sys.exit("문항이 없습니다")
    return cases


# ── 백엔드: in-process 검색 / HTTP 답변 ─────────────────────


class LocalRetrieval:
    """rag_service 엔진을 이 프로세스에 띄워 검색만 한다 (Ollama 불필요)."""

    def __init__(self) -> None:
        sys.path.insert(0, str(HERE))
        from rag.engine import RagEngine  # noqa: WPS433

        t = time.time()
        self.engine = RagEngine()
        self.engine.load()
        print(f"[엔진 로드 {time.time() - t:.1f}s]")

    def run(self, question: str, institution_id: Optional[str] = None) -> dict:
        chunks = self.engine.retrieve_context(question, institution_id=institution_id)
        return {"answer": None, "grounded": bool(chunks), "sources": chunks}


class HttpAnswer:
    def __init__(self, url: str, key: str, timeout: float, mode: Optional[str] = None) -> None:
        import httpx

        self.client = httpx.Client(base_url=url.rstrip("/"), timeout=timeout, headers={"X-Internal-Key": key})
        self.mode = mode

    def run(self, question: str, institution_id: Optional[str] = None) -> dict:
        body = {"question": question}
        if self.mode:
            body["answer_mode"] = self.mode
        if institution_id:
            body["institution_id"] = institution_id
        r = self.client.post("/answer", json=body)
        r.raise_for_status()
        data = r.json()
        return {"answer": data.get("answer", ""), "grounded": bool(data.get("grounded")), "sources": data.get("sources", []),
                "unverified": data.get("unverified") or []}


# ── 채점 ─────────────────────────────────────────────────────


def grade(case: dict, result: dict, with_answer: bool) -> dict:
    sources = result["sources"]
    pages = sorted({(s.get("source_file") or "", s.get("page_number")) for s in sources})
    want_grounded = case.get("grounded", True)
    doc = case.get("doc")
    want_pages = set(case.get("pages", []))

    if not want_grounded:
        retrieval_ok = not sources
        why = "" if retrieval_ok else f"근거 {len(sources)}건이 잡힘"
    else:
        hits = [(f, p) for f, p in pages if (not doc or doc in f) and (not want_pages or p in want_pages)]
        retrieval_ok = bool(hits) if (doc or want_pages) else bool(sources)
        why = "" if retrieval_ok else f"기대 p{sorted(want_pages)} / 실제 {[p for _, p in pages]}"

    answer_ok = None
    if with_answer:
        ans = result["answer"] or ""
        if not want_grounded:
            # 근거 0건이거나, 근거 후보가 잡혔어도 모델이 "찾을 수 없다" 로 거절했으면 통과 — 사용자에게 나간 답이 기준.
            # 실측(09-18): "오늘 날씨 어때?" 에 미세먼지 페이지가 후보로 잡혔지만 답은 거절문이었다.
            answer_ok = retrieval_ok or not result["grounded"] or "찾을 수 없" in ans
        else:
            missing = [grp for grp in case.get("keywords", []) if not any(k in ans for k in grp.split("|"))]
            banned = [b for b in case.get("exclude", []) if b in ans]
            answer_ok = retrieval_ok and not missing and not banned
            if missing:
                why = (why + "; " if why else "") + f"키워드 없음 {missing}"
            if banned:
                why = (why + "; " if why else "") + f"금지 값 포함 {banned} (다른 행의 값이 섞임)"
            if result.get("unverified"):
                # 검색은 됐지만 모델이 근거에 없는 숫자를 지어내 검증기가 막은 경우 — '못 찾음' 과 구분해 표시
                why = (why + "; " if why else "") + f"환각 차단(근거에 없는 숫자 {result['unverified']})"
    return {
        "id": case.get("id"),
        "question": case["question"],
        "optional": bool(case.get("optional")),
        "retrieval_ok": retrieval_ok,
        "answer_ok": answer_ok,
        "pages": [p for _, p in pages],
        "answer": (result["answer"] or "")[:300] if with_answer else None,
        "why": why,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="골든 JSONL 파일(글롭 가능)")
    ap.add_argument("--answer", action="store_true", help="rag_service HTTP 로 답변까지 채점")
    ap.add_argument("--mode", choices=["voice", "full"], default=None,
                    help="답변 모드. 생략하면 서버 기본값(LLM_ANSWER_MODE, 운영은 voice). full 은 표·목록 완전 나열 검증용")
    ap.add_argument("--url", default=os.getenv("RAG_SERVICE_URL", "http://127.0.0.1:8001"))
    ap.add_argument("--key", default=os.getenv("INTERNAL_API_KEY", "dev-only-shared-key"))
    ap.add_argument("--timeout", type=float, default=330.0)
    ap.add_argument("--min-pass", type=float, default=0.9)
    ap.add_argument("--report", help="결과 JSON 저장 경로")
    ap.add_argument("--only", help="이 id 들만 채점 (쉼표 구분, 예: m01,m14) — 실패 문항 재확인용")
    args = ap.parse_args()

    cases = load_cases(args.files)
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        cases = [c for c in cases if c.get("id") in wanted]
    backend = HttpAnswer(args.url, args.key, args.timeout, args.mode) if args.answer else LocalRetrieval()

    rows = []
    for c in cases:
        t = time.time()
        try:
            res = backend.run(c["question"], c.get("institution"))
        except Exception as exc:  # 서비스 장애도 한 문항 실패로 기록하고 계속
            res = {"answer": "", "grounded": False, "sources": []}
            c = dict(c, _error=str(exc))
        row = grade(c, res, args.answer)
        row["seconds"] = round(time.time() - t, 1)
        if "_error" in c:
            # 서비스 장애는 어떤 문항이든 실패 — 무관 문항이 빈 답을 '거절' 로 통과하지 않게 (09-18 실측: 재시작 중 돌린 보고서)
            row["why"] = f"호출 실패: {c['_error']}"
            row["retrieval_ok"] = False
            row["answer_ok"] = False if args.answer else None
        rows.append(row)
        mark = ("답변 " + ("PASS" if row["answer_ok"] else "FAIL")) if args.answer else ("검색 " + ("PASS" if row["retrieval_ok"] else "FAIL"))
        opt = " (선택)" if row["optional"] else ""
        print(f"{row['id']:>4} {mark}{opt} {row['seconds']:>5}s  {row['question']}  → p{row['pages']}  {row['why']}")
        if args.answer and row["answer"]:
            print(f"       ↳ {row['answer'][:140].replace(chr(10), ' ')}")

    required = [r for r in rows if not r["optional"]]
    key = "answer_ok" if args.answer else "retrieval_ok"
    passed = sum(1 for r in required if r[key])
    rate = passed / len(required) if required else 1.0
    r_pass = sum(1 for r in required if r["retrieval_ok"])
    print("-" * 60)
    print(f"검색 통과 {r_pass}/{len(required)}" + (f", 답변 통과 {passed}/{len(required)}" if args.answer else "")
          + f"  → 통과율 {rate:.0%} (기준 {args.min_pass:.0%})"
          + (f"  [선택 문항 {len(rows) - len(required)}개 별도]" if len(rows) != len(required) else ""))
    if args.report:
        Path(args.report).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print("보고서:", args.report)
    sys.exit(0 if rate >= args.min_pass else 1)


if __name__ == "__main__":
    main()
