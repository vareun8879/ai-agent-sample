"""reranker 임계값 캘리브레이션 도구 — 골든 문항마다 '정답 페이지의 최고 점수' 와 '무관 문항의 최고 점수' 를 찍는다.

  cd rag_service && ../app-env/bin/python rerank_probe.py golden/*.jsonl
  cd rag_service && ../app-env/bin/python rerank_probe.py golden/*.jsonl --only e20,b30   # 특정 문항만
  (기관 문서 문항은 골든의 "institution" 값으로 서비스와 같은 범위 — 공통 + 그 기관 — 를 검색한다)

출력:
  - 문항별: 기대 페이지가 후보에 있었는지, 그 페이지의 reranker 점수, 후보 전체 최고 점수, 현재 임계값 통과 여부
  - 요약: 정답 페이지 점수 분포(관련 문항) vs 최고 점수 분포(무관 문항) → 둘을 가르는 임계값 후보와, 그 값으로 바꿨을 때
    예상 통과/차단 수

임계값(RERANK_MIN_SCORE)은 "무관 질문을 차단" 하면서 "답이 있는 질문은 통과" 시켜야 한다. 이 둘의 점수 분포가
겹치지 않으면 그 사이 값이 답이고, 겹치면 임계값만으로는 안 되니 검색성 보강(표 제목 등)으로 넘어간다.
엔진을 in-process 로 띄우므로 LLM 은 필요 없다 (rag_service 가 떠 있어도 무관, GPU 면 30초 안쪽).
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rag.config import DEFAULT_TOP_K, RERANK_CANDIDATES, RERANK_MIN_SCORE, RERANK_RELATIVE_MIN  # noqa: E402
from rag.engine import RagEngine  # noqa: E402
from rag.reranker import _passage  # noqa: E402


def load_cases(patterns):
    cases = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)) or [pat]:
            with open(f, encoding="utf-8") as fh:
                cases += [json.loads(l) for l in fh if l.strip() and not l.startswith("#")]
    return cases


def main():
    args = sys.argv[1:]
    only = None
    if "--only" in args:
        i = args.index("--only")
        only = {x.strip() for x in args[i + 1].split(",") if x.strip()}
        args = args[:i] + args[i + 2:]
    cases = load_cases(args or ["golden/*.jsonl"])
    if only:
        cases = [c for c in cases if c.get("id") in only]
    eng = RagEngine()
    eng.load()
    assert eng.reranker and eng.reranker.ready, "RERANK=1 이어야 한다"
    model = eng.reranker._model

    def snip(x, s):
        """조각 한 줄 요약: p{페이지}/{종류} {점수} '{제목 접두 뗀 앞 60자}'"""
        t = _passage(x["text"]).replace("\n", " ")[:60]
        return f"p{x.get('page_number')}/{x.get('kind', 'page')} {s:.3f} '{t}'"

    rel_scores, irr_scores, rows, details = [], [], [], []
    for c in cases:
        q = c["question"]
        cands = eng.retriever.retrieve(q, top_k=RERANK_CANDIDATES, institution_id=c.get("institution"))
        raw = model.predict([(q, _passage(x["text"])) for x in cands]) if cands else []  # 서비스와 같은 입력(제목 접두 제거)
        scores = np.asarray(raw, dtype="float64") if len(raw) else np.array([])  # predict 가 이미 0~1 확률
        best = float(scores.max()) if len(scores) else 0.0
        want = set(c.get("pages", []))
        doc = c.get("doc")
        order = sorted(range(len(cands)), key=lambda i: -scores[i]) if len(scores) else []
        top3 = [snip(cands[i], float(scores[i])) for i in order[:3]]
        # 서비스가 실제로 고르는 페이지 (절대 임계 + 상대 하한 + 인접 구제 적용)
        chosen = sorted({x["page_number"] for x in eng.reranker.select(q, cands, DEFAULT_TOP_K)}) if cands else []
        top3.append(f"→ 선택 페이지 {chosen}")
        if c.get("grounded", True) is False:
            irr_scores.append(best)
            rows.append((c["id"], "무관", "-", None, best, best >= RERANK_MIN_SCORE, q))
            details.append((c["id"], q, None, top3))
            continue
        # 기대 페이지 조각 중 최고 점수
        hit = [(float(s), x) for x, s in zip(cands, scores) if x.get("page_number") in want and (not doc or doc in x.get("source_file", ""))]
        gold = max(h[0] for h in hit) if hit else None
        gold_snip = snip(max(hit, key=lambda h: h[0])[1], gold) if hit else None
        if gold is not None:
            rel_scores.append(gold)
        rows.append((c["id"], "관련", f"p{sorted(want)}", gold, best, (gold or 0) >= RERANK_MIN_SCORE, q))
        details.append((c["id"], q, gold_snip, top3))

    print(f"\n임계값(RERANK_MIN_SCORE) 현재 {RERANK_MIN_SCORE}  |  상대 하한 비율 {RERANK_RELATIVE_MIN}  |  후보 {RERANK_CANDIDATES}개\n")
    print(f"{'id':>4} {'종류':<3} {'기대':<12} {'정답p점수':>9} {'후보최고':>8} {'통과':>4}  질문")
    for id_, kind, want, gold, best, ok, q in rows:
        g = f"{gold:.3f}" if gold is not None else "후보없음"
        print(f"{id_:>4} {kind:<3} {want:<12} {g:>9} {best:>8.3f} {'O' if ok else 'X':>4}  {q}")

    print("\n── 문항별 상세: 정답 페이지 최고 조각 / 후보 상위 3 (p페이지/종류 점수 '내용') ──")
    for id_, q, gold_snip, top3 in details:
        print(f"{id_} {q}")
        print(f"     정답: {gold_snip or '후보 안에 없음 (임베딩 검색에서 탈락)'}")
        for t in top3:
            print(f"     상위: {t}")

    rel = np.array(rel_scores); irr = np.array(irr_scores)
    print("\n── 분포 ──")
    if len(rel):
        print(f"관련 문항 정답페이지 점수: n={len(rel)}  min {rel.min():.3f}  p25 {np.percentile(rel,25):.3f}  median {np.median(rel):.3f}  max {rel.max():.3f}")
    print(f"관련 문항 중 기대 페이지가 후보 {RERANK_CANDIDATES}개에 아예 없음: {sum(1 for r in rows if r[1]=='관련' and r[3] is None)}건 → 임계값이 아니라 임베딩 검색(후보 수/검색성) 문제")
    if len(irr):
        print(f"무관 문항 후보최고 점수:   n={len(irr)}  max {irr.max():.3f}")

    print("\n── 임계값 후보별 예상 ──")
    print(f"{'임계값':>6} {'관련 통과':>8} {'무관 차단':>8}")
    for t in (0.30, 0.20, 0.10, 0.05, 0.03, 0.02, 0.01):
        pass_rel = int((rel >= t).sum()) if len(rel) else 0
        block_irr = int((irr < t).sum()) if len(irr) else 0
        print(f"{t:>6.2f} {pass_rel:>4}/{len(rel):<3} {block_irr:>4}/{len(irr):<3}")
    if len(rel) and len(irr):
        gap_lo, gap_hi = float(irr.max()), float(rel.min())
        if gap_hi > gap_lo:
            print(f"\n무관 최고({gap_lo:.3f}) < 관련 최저({gap_hi:.3f}) — 그 사이 값(예: {(gap_lo+gap_hi)/2:.2f})이면 둘 다 만족")
        else:
            print(f"\n분포가 겹침 (무관 최고 {gap_lo:.3f} ≥ 관련 최저 {gap_hi:.3f}) — 임계값만으로는 전부 못 가른다. 겹치는 문항은 검색성 보강 대상")


if __name__ == "__main__":
    main()
