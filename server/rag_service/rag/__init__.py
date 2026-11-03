"""질의응답(RAG) 파이프라인 패키지.

질문 하나가 답변이 되기까지의 흐름과 담당 모듈:

    질문 (main.py = HTTP 계층, 로직 없음)
     │
     ▼
  engine.py       파이프라인 파사드 — 아래 단계 전체를 소유·오케스트레이션 (RAG Engine)
     │
     ▼
  retriever.py    1단계 검색 — 질문 임베딩 → Chroma top-k 조각 + 거리 임계값 필터
     │
     ▼
  expansion.py    1.5단계 확장 — 조각이 속한 페이지 전체로 병합 (small-to-big,
     │            연속 페이지 묶음, 컨텍스트 예산 관리)
     ▼
  prompt.py       2단계 입력 — 시스템 프롬프트 + [문서] 블록 + 멀티턴 메시지 조립
     │
     ▼
  generator.py    2단계 생성 — Ollama LLM 호출 + 자동 이어쓰기 루프 + 스트리밍
     │
     ▼
  continuation.py 2단계 보정 — 이어쓰기 발동 판단, 회차 결과 정리(중복·마커·메타 필터)

  config.py       모든 환경변수·캘리브레이션 값 (실측 근거 주석 포함) — 운영 조정은 여기서

인덱싱 파이프라인(../../indexing/)이 만든 chroma_db를 검색 대상으로 쓴다.
검색 관련 설정(임베딩 모델 등)은 인덱싱과 반드시 일치해야 한다 — config.py 참고.
"""

from .engine import RagEngine, RetrieverNotReady
from .generator import Generator
from .retriever import Retriever, filter_relevant

__all__ = ["RagEngine", "RetrieverNotReady", "Retriever", "Generator", "filter_relevant"]
