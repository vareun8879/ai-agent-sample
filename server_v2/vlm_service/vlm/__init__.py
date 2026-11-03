"""VLM 백엔드 선택 — VLM_BACKEND=vllm(기본) | transformers.

vllm:         별도 vLLM 서버(OpenAI 호환)에 위임 (generator.Generator). 개발 PC·Qwen 계열용.
transformers: 이 프로세스가 모델을 직접 로드 (generator_hf.HFGenerator). vLLM 이 못 받는
              커스텀 아키텍처(SKT A.X-4.0-VL-Light)용 — 운영 서버 인덱싱 기본.
두 클래스는 같은 인터페이스(describe → (텍스트, 잘림), health)라 main.py 는 차이를 모른다.
"""
import os

_BACKEND = os.getenv("VLM_BACKEND", "vllm")
if _BACKEND == "transformers":
    from .generator_hf import HFGenerator as Generator
elif _BACKEND == "vllm":
    from .generator import Generator
else:
    raise ValueError(f"VLM_BACKEND 값이 잘못됐습니다: {_BACKEND!r} (vllm | transformers)")

__all__ = ["Generator"]
