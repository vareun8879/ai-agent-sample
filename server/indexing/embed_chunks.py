import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("embed_chunks")

INPUT_PATH = Path(__file__).resolve().parent / "output_chunks" / "chunks.jsonl"
OUTPUT_PATH = Path(__file__).resolve().parent / "output_embeddings" / "embeddings.jsonl"

# bge-m3: 다국어(한국어 포함) 검색 성능이 e5-small보다 크게 좋고, 입력 한도가
# 8192토큰이라 큰 청크(chunk_documents.py의 1200자)도 잘리지 않고 임베딩된다.
# 주의: 모델을 바꾸면 rag_service의 EMBEDDING_MODEL과 반드시 함께 바꾸고
# 전체 재인덱싱 + MAX_RETRIEVAL_DISTANCE 재측정까지 한 세트로 진행할 것.
MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 8  # bge-m3(568M)는 e5-small보다 무거워 CPU 메모리 대비 배치를 줄인다
# 임베딩 계산 디바이스. 비우면 자동 감지(CUDA 가능 시 GPU). rag_service의 EMBED_DEVICE와 동일 규칙.
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "") or None

# bge-m3는 접두사가 필요 없다 (e5 계열은 "passage: "/"query: " 필수였음 —
# e5로 되돌릴 일이 있으면 이 접두사도 함께 복원해야 한다)
PASSAGE_PREFIX = ""
QUERY_PREFIX = ""

# 파이프라인이 실제로 의미 검색을 하는지 확인하기 위한 데모 질문
DEMO_QUERY = "스쿨패스는 어떻게 신청하나요?"
DEMO_TOP_K = 3


def load_chunks() -> list:
    chunks = []
    with open(INPUT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main() -> None:
    if not INPUT_PATH.exists():
        logger.error("2단계 출력을 찾을 수 없습니다: %s", INPUT_PATH)
        logger.error("먼저 chunk_documents.py를 실행하세요.")
        return

    chunks = load_chunks()
    if not chunks:
        logger.warning("청크가 없습니다: %s", INPUT_PATH)
        return

    logger.info("임베딩 모델 로드 중: %s", MODEL_NAME)
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, device=EMBED_DEVICE)
    logger.info("모델 로드 완료 (%.1fs)", time.time() - t0)

    texts = [PASSAGE_PREFIX + c["text"] for c in chunks]

    logger.info("%d개 조각 임베딩 계산 중...", len(texts))
    t0 = time.time()
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False, normalize_embeddings=True)
    elapsed = time.time() - t0
    logger.info("임베딩 완료 (%.1fs, 조각당 평균 %.3fs)", elapsed, elapsed / len(texts))

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for chunk, vec in zip(chunks, embeddings):
            record = dict(chunk)
            record["embedding"] = vec.tolist()
            record["embedding_model"] = MODEL_NAME
            record["embedding_dim"] = int(len(vec))
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("-" * 50)
    logger.info("완료: %d개 조각 -> %s (%d차원)", len(chunks), OUTPUT_PATH, embeddings.shape[1])

    # ---- 검증: 실제로 의미 기반 검색이 되는지 데모 질문으로 확인 ----
    query_vec = model.encode([QUERY_PREFIX + DEMO_QUERY], normalize_embeddings=True)[0]
    scores = embeddings @ query_vec  # 정규화된 벡터라 내적 = 코사인 유사도
    top_idx = np.argsort(scores)[::-1][:DEMO_TOP_K]

    logger.info("-" * 50)
    logger.info('데모 검색: "%s"', DEMO_QUERY)
    for rank, i in enumerate(top_idx, start=1):
        c = chunks[i]
        preview = c["text"][:60].replace("\n", " ")
        logger.info("  %d. [%.3f] %s (p%d) — %s...", rank, scores[i], c["source_file"], c["page_number"], preview)


if __name__ == "__main__":
    main()
