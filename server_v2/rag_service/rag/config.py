from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    chroma_dir: Path
    revision: str
    collection: str = "schoolpass_docs"
    embedding_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    embed_device: str = "cuda"
    rerank_device: str = "cuda"
    candidates: int = 50
    lexical: int = 20
    top_k: int = 6
    min_score: float = 0.03
    relative_min: float = 0.4
    max_evidence_chars: int = 18000
    llm_url: str = "http://127.0.0.1:9000/v1"
    llm_model: str = "ax4-light"
    llm_timeout: float = 120
    request_timeout: float = 300
    max_calls: int = 2
    max_tokens: int = 1400
    concurrency: int = 8
    validation_mode: str = "log"
    value_candidates_mode: str = "log"
    fact_anchor_mode: str = "on"

    @classmethod
    def from_env(cls):
        mode = os.getenv("V2_VALIDATION_MODE", "log")
        if mode not in ("log", "enforce"):
            raise ValueError("V2_VALIDATION_MODE must be log or enforce")
        candidates_mode = os.getenv("V2_VALUE_CANDIDATES_MODE", "log")
        if candidates_mode not in ("off", "log"):
            raise ValueError("V2_VALUE_CANDIDATES_MODE must be off or log")
        fact_anchor_mode = os.getenv("V2_FACT_ANCHOR_MODE", "on")
        if fact_anchor_mode not in ("off", "log", "on"):
            raise ValueError("V2_FACT_ANCHOR_MODE must be off, log or on")
        revision = os.environ.get("INDEX_REVISION", "").strip()
        if not revision:
            raise ValueError("INDEX_REVISION is required (snapshot manifest revision)")
        return cls(
            chroma_dir=Path(os.environ["CHROMA_DIR"]), revision=revision,
            collection=os.getenv("COLLECTION_NAME", "schoolpass_docs"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            rerank_model=os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
            embed_device=os.getenv("EMBED_DEVICE", "cuda"),
            rerank_device=os.getenv("RERANK_DEVICE", "cuda"),
            candidates=max(1, int(os.getenv("RERANK_CANDIDATES", "50"))),
            top_k=max(1, min(12, int(os.getenv("RETRIEVAL_TOP_K", "6")))),
            min_score=float(os.getenv("RERANK_MIN_SCORE", "0.03")),
            relative_min=float(os.getenv("RERANK_RELATIVE_MIN", "0.4")),
            llm_url=os.getenv("VLLM_LLM_URL", "http://127.0.0.1:9000/v1").rstrip("/"),
            llm_model=os.getenv("LLM_MODEL", "ax4-light"),
            llm_timeout=float(os.getenv("V2_LLM_TIMEOUT", "120")),
            request_timeout=float(os.getenv("V2_REQUEST_TIMEOUT", "300")),
            max_tokens=int(os.getenv("V2_MAX_TOKENS", "1400")),
            concurrency=max(1, int(os.getenv("V2_CONCURRENCY", "8"))),
            validation_mode=mode,
            value_candidates_mode=candidates_mode,
            fact_anchor_mode=fact_anchor_mode,
        )
