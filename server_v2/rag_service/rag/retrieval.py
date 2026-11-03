"""기존 Chroma 스냅샷 연결. 검색 범위는 항상 common + 요청 기관으로 제한."""
from dataclasses import replace
import math
from pathlib import Path
import threading
from .config import Settings
from .contracts import Evidence, Part, evidence_id
from .fact_anchor import anchor_probe, select_anchor_rescue
from .textutil import question_overlap, strip_title, search_question
from . import textutil


def in_scope(meta: dict, institution: str | None) -> bool:
    return meta.get("scope") == "common" or bool(institution and meta.get("institution_id") == institution)


class Search:
    def __init__(self, settings: Settings):
        self.s = settings
        self.collection = None
        self.embed = None
        self.rerank = None
        self.lex = []
        self.titles = []
        self.lock = threading.Lock()

    def load(self):
        import chromadb
        import torch
        from sentence_transformers import SentenceTransformer, CrossEncoder
        if not self.s.chroma_dir.is_dir():
            raise RuntimeError("Chroma snapshot missing")
        # 이 객체를 교체하면서 모델을 두 벌 올리지 않는다. reload는 DB만 다시 연다.
        client = chromadb.PersistentClient(path=str(self.s.chroma_dir))
        self.collection = client.get_collection(self.s.collection)
        if not self.collection.count():
            raise RuntimeError("Chroma collection is empty")
        def kwargs(device):
            return {"model_kwargs": {"torch_dtype": torch.float16}} if device.startswith("cuda") else {}
        self.embed = SentenceTransformer(self.s.embedding_model, device=self.s.embed_device,
                                         **kwargs(self.s.embed_device))
        self.rerank = CrossEncoder(self.s.rerank_model, device=self.s.rerank_device,
                                  **kwargs(self.s.rerank_device))
        self.reload()

    def reload(self):
        with self.lock:
            data = self.collection.get(where={"kind": "fact"}, include=["documents", "metadatas"])
            self.lex = list(zip(data["ids"], data["documents"], data["metadatas"]))
            import re
            all_meta = self.collection.get(include=["metadatas"])["metadatas"]
            unique = {m.get("source_file", ""): m for m in all_meta}
            self.titles = [(m, set(re.findall(r"[가-힣]{2,}", Path(source).stem)))
                           for source, m in unique.items()]
            textutil.TITLE_TOKENS.clear()
            textutil.TITLE_TOKENS.update(t for _, tokens in self.titles for t in tokens)

    def _evidence(self, cid, text, meta, distance=None):
        source, page = meta.get("source_file", ""), int(meta.get("page_number", 0))
        return Evidence(evidence_id(self.s.revision, source, page, cid, text), source,
                        page, int(meta.get("chunk_index", 0)), strip_title(text),
                        meta.get("kind", "page"), distance)

    def retrieve(self, parts: list[Part], institution: str | None, top_k: int | None = None):
        with self.lock:
            return self._retrieve(parts, institution, top_k)

    def retrieve_with_trace(self, parts: list[Part], institution: str | None, top_k: int | None = None):
        """동일한 검색 결과와 각 선택 단계의 메타데이터를 함께 반환한다."""
        with self.lock:
            return self._retrieve(parts, institution, top_k, diagnostic=True)

    def _retrieve(self, parts, institution, top_k, diagnostic=False):
        where = ({"$or": [{"scope": "common"}, {"institution_id": institution}]}
                 if institution else {"scope": "common"})
        titles = [t for m, t in self.titles if in_scope(m, institution)]
        queries = [search_question(p.question, titles) or p.question for p in parts]
        vectors = self.embed.encode(queries, normalize_embeddings=True).tolist()
        result = self.collection.query(query_embeddings=vectors, where=where,
                                       n_results=min(self.s.candidates, self.collection.count()),
                                       include=["documents", "metadatas", "distances"])
        by_part = {}
        stages = {} if diagnostic else None
        for i, (part, query) in enumerate(zip(parts, queries)):
            candidates = {}
            candidate_meta = {}
            chunk_ids = {} if diagnostic else None
            vector_hits = []
            for cid, doc, meta, dist in zip(result["ids"][i], result["documents"][i],
                                           result["metadatas"][i], result["distances"][i]):
                if in_scope(meta, institution):
                    e = self._evidence(cid, doc, meta, float(dist))
                    candidates[e.id] = e
                    candidate_meta[e.id] = meta
                    if diagnostic:
                        chunk_ids[e.id] = cid
                        vector_hits.append({"chunk_id": cid, "source_file": e.source_file,
                                            "page_number": e.page_number, "distance": e.distance})
            lexical = [(question_overlap(query, strip_title(doc)), cid, doc, meta)
                       for cid, doc, meta in self.lex if in_scope(meta, institution)]
            lexical.sort(key=lambda x: (-x[0], len(x[2]), x[1]))
            lexical_hits = []
            for score, cid, doc, meta in lexical[:self.s.lexical]:
                if score > 0:
                    e = self._evidence(cid, doc, meta)
                    candidates.setdefault(e.id, e)
                    candidate_meta[e.id] = meta
                    if diagnostic:
                        chunk_ids[e.id] = cid
                        lexical_hits.append({"chunk_id": cid, "source_file": e.source_file,
                                             "page_number": e.page_number, "overlap": score})
            rows = list(candidates.values())
            if not rows:
                by_part[part.id] = []
                if diagnostic:
                    stages[part.id] = {"query": query, "vector": vector_hits,
                                       "lexical": lexical_hits, "ranked": [], "floor": None}
                continue
            # CrossEncoderは単一ラベルでsigmoid済み。二重変換しない。
            scores = self.rerank.predict([(query, e.text) for e in rows], show_progress_bar=False)
            if len(scores) != len(rows) or not all(math.isfinite(float(s)) for s in scores):
                raise RuntimeError("Invalid reranker output")
            ranked = sorted((replace(e, score=float(s)) for e, s in zip(rows, scores)),
                            key=lambda e: (-e.score, e.id))
            floor = max(self.s.min_score, ranked[0].score * self.s.relative_min)
            limit = top_k or self.s.top_k
            by_part[part.id] = [e for e in ranked if e.score >= floor][:limit]
            rescue = None
            if self.s.fact_anchor_mode == "on":
                # 실측(261103_4, m09·m10): 정답 사실 행이 절대 점수는 넘지만
                # 다른 문서 조각의 높은 점수로 상대 기준에서 탈락했다.
                rescue = select_anchor_rescue(query, ranked, by_part[part.id],
                                              candidate_meta, institution, self.s.min_score, limit)
                if rescue is not None:
                    by_part[part.id] = [rescue, *by_part[part.id]]
            if diagnostic:
                selected = {e.id for e in by_part[part.id]}
                stages[part.id] = {"query": query, "vector": vector_hits,
                                   "lexical": lexical_hits, "floor": floor,
                                   "ranked": [{"evidence_id": e.id, "source_file": e.source_file,
                                               "chunk_id": chunk_ids[e.id],
                                               "page_number": e.page_number, "score": e.score,
                                               "selected": e.id in selected}
                                              for e in ranked]}
                if rescue is not None:
                    stages[part.id]["anchor_rescue"] = {
                        "chunk_id": chunk_ids[rescue.id], "source_file": rescue.source_file,
                        "page_number": rescue.page_number, "score": rescue.score}
                if self.s.fact_anchor_mode == "log":
                    # 실측(261103_3, m09·m10·m34): 정답 행은 벡터 후보에 있었지만
                    # 최종 점수 하한에서 빠졌다. 값·대상 관계를 단정하기 전에 필드
                    # 일치 후보와 기존 통과 문항의 충돌을 전체 골든에서 관측한다.
                    try:
                        probe = anchor_probe(query, ranked, by_part[part.id])
                        snippets = [row["snippet"] for row in probe["matches"]]
                        if snippets:
                            snippet_scores = self.rerank.predict(
                                [(query, snippet) for snippet in snippets], show_progress_bar=False)
                            if len(snippet_scores) != len(snippets) or not all(
                                    math.isfinite(float(score)) for score in snippet_scores):
                                raise ValueError("invalid snippet score")
                            for row, score in zip(probe["matches"], snippet_scores):
                                row["snippet_score"] = float(score)
                        stages[part.id]["fact_anchor"] = probe
                    except Exception as exc:
                        # 진단 실패가 검색·답변을 중단시키지 않는다.
                        stages[part.id]["fact_anchor_error"] = type(exc).__name__
        return (by_part, stages) if diagnostic else by_part
