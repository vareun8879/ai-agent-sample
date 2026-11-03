"""인덱싱 파이프라인 4단계를 순서대로 실행하는 러너.

문서(docs/)가 바뀌었을 때 이 스크립트 하나만 실행하면 된다:

    python run_all.py                    # docs/ 전체
    python run_all.py --only 7010170     # docs/7010170/ 만 (쉼표로 여러 폴더: --only common,7010170)
                                         #   → 그 폴더의 조각만 지우고 다시 넣는다. 다른 기관 색인은 그대로.

완료 후 rag_service가 떠 있다면 재기동 없이 반영할 수 있다:

    curl -X POST http://localhost:8001/reload -H "X-Internal-Key: <내부 키>"
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("run_all")

HERE = Path(__file__).resolve().parent

STEPS = [
    ("1/4 PDF 텍스트 추출", "load_documents.py"),
    ("2/4 청킹", "chunk_documents.py"),
    ("3/4 임베딩", "embed_chunks.py"),
    ("4/4 Chroma 저장", "store_chroma.py"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="인덱싱 파이프라인 4단계 실행")
    ap.add_argument("--only", default=os.getenv("INDEX_ONLY", ""),
                    help="docs/ 바로 밑 폴더 이름(common 또는 기관ID)을 쉼표로 — 그 폴더만 다시 인덱싱 (기본: 전체)")
    args = ap.parse_args()
    env = dict(os.environ)
    if args.only.strip():
        env["INDEX_ONLY"] = args.only.strip()
        logger.info("부분 인덱싱: %s (다른 폴더의 색인은 그대로 둔다)", args.only.strip())
    else:
        env.pop("INDEX_ONLY", None)

    total_start = time.time()
    for title, script in STEPS:
        logger.info("=" * 50)
        logger.info("%s — %s", title, script)
        start = time.time()
        result = subprocess.run([sys.executable, str(HERE / script)], cwd=str(HERE), env=env)
        if result.returncode != 0:
            logger.error("%s 실패 (exit %d) — 파이프라인 중단", script, result.returncode)
            sys.exit(result.returncode)
        logger.info("%s 완료 (%.1fs)", title, time.time() - start)

    logger.info("=" * 50)
    logger.info("전체 파이프라인 완료 (%.1fs)", time.time() - total_start)
    logger.info("rag_service가 떠 있다면: POST /reload 로 재기동 없이 반영 가능")


if __name__ == "__main__":
    main()
