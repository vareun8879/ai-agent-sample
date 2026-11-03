"""골든 질문 세트 자체 검사 — "이 문항이 채점 가능한가" 를 문서 텍스트로 확인한다.

골든셋이 틀리면(기대 페이지에 키워드가 없거나 페이지 번호가 어긋나면) 채점 결과가 서비스 품질과 무관해진다.
골든 문항을 새로 만들거나 문서를 바꿨을 때 06(채점) 전에 먼저 돌린다. LLM·서버 불필요, 1초.

사용:
  cd rag_service
  python golden_lint.py golden/mangmi_2026_ext.jsonl                  # indexing/output/**/*.json 에서 페이지 텍스트를 읽음
  python golden_lint.py golden/*.jsonl --pages /path/to/pages.json     # load_documents 결과 JSON 을 직접 지정

검사 항목 (문항마다):
  1. id 중복 / 필수 필드(question) 누락
  2. grounded=true 문항: pages 의 각 페이지가 문서에 있고 비어 있지 않은가
  3. keywords 의 각 항목("a|b|c")에 대해 대안 중 하나라도 기대 페이지 텍스트에 있는가
     — 공백·표 기호를 무시하고 대조한다("14,262.30" 과 "14,262" 처럼 부분 일치면 통과)
  4. grounded=false 문항: pages/keywords 가 비어 있는가

종료 코드: 문제가 하나라도 있으면 1.
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_GLOB = str(HERE.parent / "indexing" / "output" / "**" / "*.json")

_FLAT = re.compile(r"[\s|]+")


def flat(s: str) -> str:
    return _FLAT.sub("", s)


def load_cases(patterns: list[str]) -> list[tuple[str, dict]]:
    out = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)) or [pat]:
            with open(f, encoding="utf-8") as fh:
                for ln, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        out.append((f"{Path(f).name}:{ln}", json.loads(line)))
                    except json.JSONDecodeError as exc:
                        out.append((f"{Path(f).name}:{ln}", {"_error": f"JSON 파싱 실패: {exc}"}))
    return out


def load_docs(pages_json: str | None) -> list[tuple[str, dict[int, str]]]:
    """(문서 이름, 페이지 번호 → 텍스트) 목록 — indexing/output/**/*.json 전부."""
    files = [pages_json] if pages_json else sorted(glob.glob(DEFAULT_OUTPUT_GLOB, recursive=True))
    if not files:
        sys.exit(f"페이지 텍스트를 찾을 수 없습니다: {DEFAULT_OUTPUT_GLOB} (load_documents.py 를 먼저 실행하거나 --pages 지정)")
    docs = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        if "pages" in d:
            docs.append((d.get("source_file", f), {p["page_number"]: p.get("text", "") for p in d["pages"]}))
    return docs


def pick_pages(docs: list[tuple[str, dict[int, str]]], doc_filter: str | None) -> dict[int, str]:
    """doc_filter(파일명 부분 문자열)에 맞는 문서의 페이지. 여럿이면 첫 번째. 없으면 빈 dict.
    세트 하나에 여러 기관 문서가 섞여 있어도(paths_2026 처럼) 문항마다 자기 doc 으로 대조한다."""
    cands = [(n, d) for n, d in docs if not doc_filter or doc_filter in n]
    if not cands:
        return {}
    if len(cands) > 1 and doc_filter:
        print(f"[주의] doc {doc_filter!r} 에 문서 {len(cands)}개가 잡혔습니다 — 첫 번째만 사용: {cands[0][0]}")
    return cands[0][1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="골든 JSONL (글롭 가능)")
    ap.add_argument("--pages", help="load_documents 출력 JSON 경로 (기본: indexing/output/**/*.json)")
    ap.add_argument("--doc", help="여러 문서 중 고를 파일명 부분 문자열 (기본: 골든의 doc 값)")
    args = ap.parse_args()

    cases = load_cases(args.files)
    docs = load_docs(args.pages)
    default_filter = args.doc or next((c.get("doc") for _, c in cases if c.get("doc")), None)
    cache: dict[str | None, dict[int, str]] = {}

    problems: list[str] = []
    seen_ids: set[str] = set()
    n_rel = n_irr = 0
    for where, c in cases:
        if "_error" in c:
            problems.append(f"{where}: {c['_error']}")
            continue
        cid = c.get("id", "?")
        if cid in seen_ids:
            problems.append(f"{where} [{cid}]: id 중복")
        seen_ids.add(cid)
        if not c.get("question"):
            problems.append(f"{where} [{cid}]: question 없음")
        if c.get("grounded", True) is False:
            n_irr += 1
            if c.get("pages") or c.get("keywords"):
                problems.append(f"{where} [{cid}]: 무관 문항(grounded=false)에 pages/keywords 가 있음")
            continue
        n_rel += 1
        want = c.get("pages") or []
        if not want:
            problems.append(f"{where} [{cid}]: pages 없음")
            continue
        doc_filter = c.get("doc") or default_filter
        if doc_filter not in cache:
            cache[doc_filter] = pick_pages(docs, doc_filter)
        pages = cache[doc_filter]
        if not pages:
            problems.append(f"{where} [{cid}]: doc {doc_filter!r} 에 맞는 문서가 indexing/output 에 없음")
            continue
        texts = []
        for p in want:
            t = pages.get(p, "")
            if not t.strip():
                problems.append(f"{where} [{cid}]: p{p} 가 문서에 없거나 빈 페이지")
            texts.append(t)
        joined = flat("\n".join(texts))
        for kw in c.get("keywords") or []:
            alts = [a for a in kw.split("|") if a]
            if not any(flat(a) in joined for a in alts):
                problems.append(f"{where} [{cid}]: 키워드 {kw!r} 가 기대 페이지 p{want} 텍스트에 없음")
        if not c.get("keywords"):
            problems.append(f"{where} [{cid}]: keywords 없음 (검색만 채점됨)")

    n_pages = sum(len(v) for v in cache.values())
    print(f"문항 {len(cases)}개 (관련 {n_rel} / 무관 {n_irr}) — 문서 {len([v for v in cache.values() if v])}개 {n_pages}쪽")
    if problems:
        print(f"\n문제 {len(problems)}건:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("문제 없음 — 모든 키워드가 기대 페이지에 있습니다.")


if __name__ == "__main__":
    main()
