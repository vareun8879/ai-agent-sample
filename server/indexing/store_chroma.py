import json
import logging
import os
from pathlib import Path

import chromadb

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("store_chroma")

INPUT_PATH = Path(__file__).resolve().parent / "output_embeddings" / "embeddings.jsonl"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "schoolpass_docs"

BATCH_SIZE = 200
INDEX_ONLY = [s.strip().strip("/") for s in os.getenv("INDEX_ONLY", "").split(",") if s.strip()]   # load_documents.py 참고
DEMO_QUERY = "스쿨패스는 어떻게 신청하나요?"
DEMO_TOP_K = 3


def classify(source_file: str) -> tuple[str, str]:
    """source_file(docs/ 기준 상대경로)의 첫 폴더명으로 scope/institution_id를 정한다.

    docs/common/... -> 공통 문서, docs/<institution_id>/... -> 해당 기관 전용 문서.
    Chroma 메타데이터는 None을 지원하지 않아 "없음"은 빈 문자열로 표현한다.
    """
    folder = source_file.split("/")[0]
    if folder == "common":
        return "common", ""
    return "institution", folder


def load_records() -> list:
    records = []
    with open(INPUT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    if not INPUT_PATH.exists():
        logger.error("3단계 출력을 찾을 수 없습니다: %s", INPUT_PATH)
        logger.error("먼저 embed_chunks.py를 실행하세요.")
        return

    records = load_records()
    if not records:
        logger.warning("임베딩 데이터가 없습니다: %s", INPUT_PATH)
        return

    embedding_model = records[0]["embedding_model"]

    logger.info("Chroma 클라이언트 초기화: %s", CHROMA_DIR)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if INDEX_ONLY:
        # 부분 인덱싱: 컬렉션은 두고 해당 폴더(scope/기관)의 조각만 지운 뒤 새 조각을 넣는다 — 다른 기관 문서는 그대로.
        # 기존 컬렉션이 없으면(첫 실행) 전체 인덱싱과 같이 새로 만든다.
        try:
            collection = client.get_collection(COLLECTION_NAME)
        except Exception:
            collection = None
        if collection is None:
            collection = client.create_collection(
                name=COLLECTION_NAME, metadata={"hnsw:space": "cosine", "embedding_model": embedding_model})
            logger.info("컬렉션이 없어 새로 만듦: %s", COLLECTION_NAME)
        else:
            stored_model = (collection.metadata or {}).get("embedding_model")
            if stored_model and stored_model != embedding_model:
                logger.error("기존 컬렉션의 임베딩 모델(%s)과 다릅니다(%s) — 부분 인덱싱 불가, 전체 재인덱싱(--only 없이)을 하세요",
                             stored_model, embedding_model)
                raise SystemExit(1)
            before = collection.count()
            for folder in INDEX_ONLY:
                where = {"scope": "common"} if folder == "common" else {"institution_id": folder}
                collection.delete(where=where)
            logger.info("부분 인덱싱: 폴더 %s 의 기존 조각 삭제 (%d → %d개)", INDEX_ONLY, before, collection.count())
        # 같은 chunk_id 가 남아 있으면 add 가 실패하므로 한 번 더 지운다
        existing = set(collection.get(ids=[r["chunk_id"] for r in records], include=[]).get("ids") or [])
        if existing:
            collection.delete(ids=sorted(existing))
            logger.info("같은 chunk_id 의 옛 조각 %d개 삭제", len(existing))
    else:
        # 재실행 시 중복 저장을 막기 위해 기존 컬렉션은 지우고 새로 만든다
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info("기존 컬렉션 삭제: %s", COLLECTION_NAME)
        except Exception:
            pass

        collection = client.create_collection(
            name=COLLECTION_NAME,
            # 임베딩을 정규화해서 저장했으므로 거리 계산도 코사인으로 맞춘다
            metadata={"hnsw:space": "cosine", "embedding_model": embedding_model},
        )

    ids = [r["chunk_id"] for r in records]
    embeddings = [r["embedding"] for r in records]
    documents = [r["text"] for r in records]
    metadatas = []
    for r in records:
        scope, institution_id = classify(r["source_file"])
        metadatas.append({
            "source_file": r["source_file"],
            "page_number": r["page_number"],
            "chunk_index": r["chunk_index"],
            # "page": 페이지 본문 조각 / "fact": 표 행 사실 조각(검색 전용 — expansion 이 페이지 전문에서 제외)
            "kind": r.get("kind", "page"),
            "scope": scope,
            "institution_id": institution_id,
        })

    for i in range(0, len(ids), BATCH_SIZE):
        end = min(i + BATCH_SIZE, len(ids))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )
        logger.info("저장 중... %d/%d", end, len(ids))

    logger.info("-" * 50)
    logger.info("완료: %d개 조각 저장 -> %s (컬렉션 전체 %d개, 컬렉션: %s)", len(ids), CHROMA_DIR, collection.count(), COLLECTION_NAME)

    institution_counts: dict[str, int] = {}
    for m in (collection.get(include=["metadatas"], limit=200000).get("metadatas") or []):
        key = "common" if m.get("scope") == "common" else f"institution:{m.get('institution_id')}"
        institution_counts[key] = institution_counts.get(key, 0) + 1
    logger.info("scope/institution별 조각 수 (컬렉션 전체): %s", institution_counts)

    # ---- 검증: 저장된 DB에서 실제로 검색이 되는지 확인 ----
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embedding_model)
    # bge-m3는 질의 접두사가 필요 없다 (e5 계열이었다면 "query: " 필요)
    query_vec = model.encode([DEMO_QUERY], normalize_embeddings=True)[0].tolist()

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=DEMO_TOP_K,
        where={"scope": "common"},
    )

    logger.info("-" * 50)
    logger.info('Chroma 검색 검증: "%s"', DEMO_QUERY)
    for rank, (dist, meta, doc) in enumerate(
        zip(results["distances"][0], results["metadatas"][0], results["documents"][0]), start=1
    ):
        preview = doc[:60].replace("\n", " ")
        logger.info("  %d. [거리 %.4f] %s (p%s) — %s...", rank, dist, meta["source_file"], meta["page_number"], preview)


if __name__ == "__main__":
    main()
