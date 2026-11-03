"""질의응답 1단계 — 검색(Retrieval).

질문 텍스트를 인덱싱 때와 동일한 임베딩 모델로 벡터화하고, indexing/이 만들어 둔
Chroma 컬렉션에서 의미가 가장 가까운 근거 조각 top-k개를 찾아온다.

- 임베딩 모델·컬렉션·접두사 등 인덱싱과 일치해야 하는 값은 config.py에 있다.
- 검색된 조각의 페이지 병합(small-to-big)은 expansion.py가 담당하고, 이 클래스의
  expand_pages()는 그 위임 창구다 — 호출 측(main.py)은 Retriever 하나만 알면 된다.
"""

import logging
import re
from pathlib import Path
from typing import Optional

from . import expansion
from .config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBED_DEVICE,
    EMBEDDING_MODEL,
    LEXICAL_CANDIDATES,
    MAX_RETRIEVAL_DISTANCE,
    QUERY_PREFIX,
)
from . import textutil
from .textutil import question_overlap, question_tokens, strip_title, whole_word_hits

logger = logging.getLogger("rag.retriever")


def filter_relevant(chunks: list[dict], max_distance: float = MAX_RETRIEVAL_DISTANCE) -> list[dict]:
    """거리 임계값을 넘는(= 의미가 먼) 조각을 걸러낸다.

    전부 걸러지면 빈 리스트가 반환되고, 호출 측은 LLM을 태우지 않고
    "자료에서 찾을 수 없음" 고정 답변을 내보내야 한다 — 억지 답변(환각) 방지.
    """
    return [c for c in chunks if c.get("distance") is not None and c["distance"] <= max_distance]


class Retriever:
    """질문 → 근거 조각 top-k. 무거운 모델/DB 핸들은 load()에서 한 번만 준비한다."""

    def __init__(self) -> None:
        self._model = None
        self._collection = None
        self._lex: list[dict] = []   # 낱말 일치용 짧은 조각(kind=fact) 전부 — load() 에서 메모리에 올린다 (수천 개, 수 MB)

    def load(self) -> None:
        """임베딩 모델과 Chroma 컬렉션을 로드한다. 서버 기동 시 한 번만 호출한다.

        인덱싱 재실행 후에는 이 메서드를 다시 호출(/reload)하면 재기동 없이 새 인덱스가
        반영된다 — 단, 임베딩 모델 자체가 바뀐 경우는 프로세스 재기동이 필요하다
        (config 상수가 import 시점에 고정되므로).
        """
        # import를 메서드 안에 둬서, 검색을 안 쓰는 경우 무거운 의존성 로딩을 피한다
        import chromadb
        from sentence_transformers import SentenceTransformer

        from .config import fp16_model_kwargs

        extra = fp16_model_kwargs(EMBED_DEVICE)
        logger.info("임베딩 모델 로드: %s (device=%s, fp16=%s)", EMBEDDING_MODEL, EMBED_DEVICE or "auto", bool(extra))
        self._model = SentenceTransformer(EMBEDDING_MODEL, device=EMBED_DEVICE, **extra)

        if not CHROMA_DIR.exists():
            raise RuntimeError(
                f"Chroma DB를 찾을 수 없습니다: {CHROMA_DIR}. "
                "먼저 indexing/ 파이프라인(store_chroma.py까지)을 실행하세요."
            )

        logger.info("Chroma 컬렉션 열기: %s (%s)", COLLECTION_NAME, CHROMA_DIR)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._collection = client.get_collection(COLLECTION_NAME)
        self._lex = self._load_lexical() if LEXICAL_CANDIDATES > 0 else []
        self._register_title_tokens()
        logger.info("검색기 준비 완료 (조각 %d개, 낱말 일치용 짧은 조각 %d개, 제목 낱말 %d개)",
                    self._collection.count(), len(self._lex), len(textutil.TITLE_TOKENS))

    def _register_title_tokens(self) -> None:
        """색인된 문서 제목의 낱말을 textutil.TITLE_TOKENS 에 넣는다 — 질문의 "망미중학교", "배재고" 는 낱말 일치·거절
        재질문·폴백 줄 고르기에서 세지 않는다 (모든 조각에 있는 말이라 변별력이 없고 후보만 벌린다)."""
        try:
            got = self._collection.get(include=["metadatas"], limit=100000)
            stems = {Path(m.get("source_file", "")).stem for m in (got.get("metadatas") or [])}
        except Exception as exc:
            logger.warning("제목 낱말 등록 실패 (계속 진행): %s", exc)
            return
        toks: set[str] = set()
        self._doc_titles = []   # (scope, institution_id, 제목 낱말 집합) — 검색 범위별로 지울 제목 낱말을 고르는 데 쓴다
        seen: set[str] = set()
        for m in (got.get("metadatas") or []):
            stem = Path(m.get("source_file", "")).stem
            if stem in seen:
                continue
            seen.add(stem)
            title_toks = set(re.findall(r"[가-힣]{2,}", stem))
            toks.update(title_toks)
            self._doc_titles.append((m.get("scope"), m.get("institution_id"), title_toks))
        textutil.TITLE_TOKENS.clear()
        textutil.TITLE_TOKENS.update(toks)

    def search_text(self, question: str, institution_id: Optional[str]) -> str:
        """임베딩·리랭커에 넣을 질문 문장 — 검색 범위(공통 + 해당 기관)에 든 문서 제목에 **모두** 있는 낱말만 지운다.
        범위에 문서가 여럿이면 기관명이 문서를 가르는 낱말이므로 남긴다 (textutil.search_question 참고)."""
        titles = [t for sc, inst, t in getattr(self, "_doc_titles", [])
                  if sc == "common" or (institution_id and inst == institution_id)]
        return textutil.search_question(question, titles)

    def _load_lexical(self) -> list[dict]:
        """kind=fact 조각을 전부 읽어 (본문, 메타) 목록으로 — 옛 인덱스(kind 없음)면 빈 목록."""
        try:
            r = self._collection.get(where={"kind": "fact"}, include=["documents", "metadatas"])
        except Exception as exc:
            logger.warning("낱말 일치 조각 적재 실패 — 하이브리드 검색 없이 진행: %s", exc)
            return []
        out = []
        groups: set[str] = set()
        for chunk_id, doc, meta in zip(r["ids"], r["documents"], r["metadatas"]):
            body = strip_title(doc)
            out.append({"chunk_id": chunk_id, "text": doc, "body": body, "meta": meta})
            g = textutil.survey_group(body)
            if g:
                groups.add(g)
        textutil.SURVEY_GROUPS.clear()
        textutil.SURVEY_GROUPS.update(groups)
        if groups:
            logger.info("설문 응답 대상 라벨 %d개 등록: %s", len(groups), sorted(groups))
        return out

    def lexical_candidates(self, question: str, institution_id: Optional[str], limit: int) -> list[dict]:
        """질문 낱말이 가장 많이 든 짧은 조각 상위 limit 개. 검색 범위는 임베딩과 동일.
        점수가 같으면 질문 낱말이 **낱말 그대로** 든 조각을 먼저, 그다음 짧은 것 — 실측(09-21 키오스크): "개교는 언제야?" 의
        접두 일치("개교") 2점이 달력의 "개교기념일" 줄 여러 개와 연혁 행("1982.03.01. … 개교 초대 …")에 똑같이 붙고 짧은
        달력 줄들이 상위 20개를 다 채워 연혁 행이 후보에서 빠졌다 — 후보에 없으면 리랭커의 낱말 구제도 걸 게 없다."""
        toks = question_tokens(question)
        if not toks or not self._lex:
            return []
        scored = []
        for item in self._lex:
            m = item["meta"]
            if m.get("scope") != "common" and (not institution_id or m.get("institution_id") != institution_id):
                continue
            sc = question_overlap(question, item["body"], toks)
            if sc > 0:
                scored.append((sc, whole_word_hits(item["body"], toks), len(item["body"]), item))
        scored.sort(key=lambda s: (-s[0], -s[1], s[2]))
        return [
            {
                "chunk_id": it["chunk_id"],
                "text": it["text"],
                "source_file": it["meta"].get("source_file"),
                "page_number": it["meta"].get("page_number"),
                "chunk_index": it["meta"].get("chunk_index"),
                "kind": it["meta"].get("kind", "fact"),
                "distance": 1.0,   # 임베딩 거리 없음 — reranker 가 rank_key 를 새로 매긴다 (reranker 없는 경로에선 걸러진다)
                "lexical": sc,
            }
            for sc, _, _, it in scored[:limit]
        ]

    @property
    def ready(self) -> bool:
        return self._model is not None and self._collection is not None

    def retrieve(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        institution_id: Optional[str] = None,
        search_text: Optional[str] = None,
    ) -> list[dict]:
        """질문과 의미가 가까운 근거 조각 top-k개를 반환한다 (거리 오름차순).

        institution_id를 주면 공통 문서 + 해당 기관 문서만 검색 대상이 된다.
        (기본값 None이면 scope="common"만 검색한다.)
        """
        if not self.ready:
            raise RuntimeError("Retriever가 아직 load되지 않았습니다")

        # 인덱싱 때와 동일하게 정규화 → 코사인 유사도가 단순 내적으로 계산된다
        if search_text is None:
            search_text = self.search_text(question, institution_id)
        query_vec = self._model.encode(
            [QUERY_PREFIX + search_text], normalize_embeddings=True
        )[0].tolist()

        if institution_id:
            where = {"$or": [{"scope": "common"}, {"institution_id": institution_id}]}
        else:
            where = {"scope": "common"}

        results = self._collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=where,
        )

        chunks: list[dict] = []
        # query 결과는 배치(질문 1개)라 [0] 인덱스로 꺼낸다
        for chunk_id, doc, meta, dist in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": doc,
                    "source_file": meta.get("source_file"),
                    "page_number": meta.get("page_number"),
                    "chunk_index": meta.get("chunk_index"),
                    "kind": meta.get("kind", "page"),  # page: 본문 조각 / fact: 표 행 사실 조각
                    "distance": dist,  # 코사인 거리 (0에 가까울수록 유사)
                }
            )
        # 하이브리드: 낱말 일치 후보를 합친다 (이미 임베딩 후보에 있는 조각은 제외). reranker 가 최종 순위·컷을 정한다
        if LEXICAL_CANDIDATES > 0 and self._lex:
            # page와 fact는 chunk_index가 겹칠 수 있으므로 Chroma의 고유 ID로 병합한다.
            have = {c["chunk_id"]: c for c in chunks}
            added = []
            for lc in self.lexical_candidates(question, institution_id, LEXICAL_CANDIDATES):
                key = lc["chunk_id"]
                if key in have:
                    # 임베딩 후보에 이미 있는 조각에도 낱말 점수를 붙인다 — 리랭커의 낱말 구제는 이 점수가 있는 조각만 본다.
                    # 실측(09-21 키오스크): "개교는 언제야?" 의 연혁 행("1982.03.01. … 개교")이 임베딩 후보에 들어 있어
                    # '추가' 목록에서 빠졌고, 낱말 점수가 없으니 리랭커 탈락 후 구제도 안 돼 모델이 개학 날짜로 답했다.
                    have[key]["lexical"] = lc["lexical"]
                else:
                    added.append(lc)
                    have[key] = lc
            if added:
                logger.info("낱말 일치 후보 %d개 추가 (최고 %d점): %s", len(added), added[0]["lexical"],
                            [f"p{c['page_number']} {strip_title(c['text'])[:40]!r}" for c in added[:3]])
                chunks.extend(added)
        return chunks

    def expand_pages(self, chunks: list[dict], question: str = "", reranker=None) -> list[dict]:
        """검색된 조각들을 페이지 단위로 병합한다 — 로직은 expansion.expand_pages 참고.
        question 을 주면 고른 페이지의 표 행 조각 중 질문 낱말이 겹치는 줄을 핵심 근거에 보강한다."""
        if not self.ready:
            return chunks
        return expansion.expand_pages(self._collection, chunks, question, reranker)

    def fact_lines_on_pages(self, chunks: list[dict]) -> list[str]:
        """최종 컨텍스트 페이지의 색인 사실 행 — 답변 값 대조에 쓴다."""
        if not self.ready:
            return []
        lines = []
        for src, page in dict.fromkeys((c["source_file"], c["page_number"]) for c in chunks):
            lines.extend(expansion.page_facts(self._collection, src, page))
        return list(dict.fromkeys(lines))
