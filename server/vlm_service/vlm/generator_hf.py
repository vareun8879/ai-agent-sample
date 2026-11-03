"""VLM 생성 — transformers 인프로세스 백엔드 (VLM_BACKEND=transformers).

vLLM 이 받지 않는 커스텀 아키텍처 비전 모델을 이 프로세스 안에서 직접 로드해 돌린다.
대상: SKT A.X-4.0-VL-Light (`AX4VLForConditionalGeneration`, custom_code, Apache-2.0) —
한국어 문서 이미지 이해가 강해(한국어 이미지 벤치마크 79.4, Qwen2.5-VL-32B 73.4) 인덱싱
VLM 으로 골랐지만, vLLM 모델 레지스트리에 없어 OpenAI 호환 서버로는 못 띄운다.

인덱싱은 페이지를 순차 처리하는 배치라 vLLM 의 동시 처리(continuous batching)가 필요 없다.
transformers 의 generate 로도 페이지당 수십 초면 충분하고, 답변 LLM(vllm_llm)을 내린 점검
시간에만 도니까 bf16 원본(약 16GB)을 그대로 올린다.

호출 형식은 모델 카드의 quickstart 를 그대로 따른다:
  messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
  text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  inputs = processor(images=[image], text=[text], padding=True, return_tensors="pt")
"""

import asyncio
import base64
import io
import logging
import os
import threading
from typing import Optional

from .generator import DEFAULT_PROMPT, VLM_MAX_TOKENS, VLM_MODEL, VLM_TEMPERATURE

logger = logging.getLogger("vlm.generator_hf")

# 로컬에 이미 받아 둔 가중치 디렉터리(04/05단계가 내려받음). 없으면 VLM_MODEL(HF id)로 받는다 (HF_HOME 캐시).
VLM_HF_DIR = os.getenv("VLM_HF_DIR", "")
VLM_DEVICE = os.getenv("VLM_DEVICE", "cuda")
# 전사 용도라 기본은 탐욕 디코딩(VLM_TEMPERATURE=0). 샘플링을 켜면 모델 카드 권장값(top_p 0.8, top_k 20, rep 1.05)을 쓴다.
_SAMPLING_KWARGS = {"top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05}


class HFGenerator:
    """이미지(+선택 프롬프트) → transformers 로 로드한 비전 모델의 텍스트 응답.

    vLLM 백엔드(generator.Generator)와 같은 인터페이스: describe() → (텍스트, 잘림 여부), health().
    모델 호출은 스레드에서 락으로 한 번에 하나만 — GPU 메모리를 예측 가능하게 쓰기 위해서다.
    """

    backend = "transformers"

    def __init__(self) -> None:
        if not VLM_MODEL and not VLM_HF_DIR:
            raise RuntimeError("VLM_MODEL(HF id) 또는 VLM_HF_DIR(로컬 디렉터리)가 필요합니다 (예: VLM_MODEL=skt/A.X-4.0-VL-Light).")
        self.model_name = VLM_MODEL or VLM_HF_DIR
        self.model = self.model_name  # main.py 가 로그·health 에 쓰는 이름
        self.url = "in-process (transformers)"
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._load()

    # ── 로드 ──────────────────────────────────────────────

    def _source(self) -> str:
        if VLM_HF_DIR and os.path.isfile(os.path.join(VLM_HF_DIR, "config.json")):
            return VLM_HF_DIR
        return VLM_MODEL

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        src = self._source()
        logger.info("비전 모델 로드 (transformers, trust_remote_code): %s → %s", src, VLM_DEVICE)
        self._processor = AutoProcessor.from_pretrained(src, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            src, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device=VLM_DEVICE)
        self._model.eval()
        if VLM_DEVICE.startswith("cuda") and torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 2**30
            logger.info("비전 모델 로드 완료 — VRAM %.1fGB", used)
        else:
            logger.info("비전 모델 로드 완료 (%s)", VLM_DEVICE)

    # ── 추론 ──────────────────────────────────────────────

    def _generate_sync(self, image_bytes: bytes, prompt: str) -> tuple[str, bool]:
        import torch
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(images=[image], text=[text], padding=True, return_tensors="pt").to(VLM_DEVICE)

        gen_kwargs: dict = {"max_new_tokens": VLM_MAX_TOKENS}
        if VLM_TEMPERATURE > 0:
            gen_kwargs.update(_SAMPLING_KWARGS, temperature=VLM_TEMPERATURE, do_sample=True)
        else:
            gen_kwargs["do_sample"] = False

        with self._lock, torch.inference_mode():
            generated = self._model.generate(**inputs, **gen_kwargs)
        in_len = inputs["input_ids"].shape[1]
        new_ids = generated[0][in_len:]
        # 출력이 max_new_tokens 를 다 채웠으면 한도에 걸려 잘린 것으로 본다 (vLLM 의 finish_reason=length 에 해당)
        truncated = int(new_ids.shape[0]) >= VLM_MAX_TOKENS
        out = self._processor.batch_decode([new_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return out.strip(), truncated

    async def describe(self, image_base64: str, prompt: Optional[str] = None) -> tuple[str, bool]:
        """(전사 텍스트, 잘림 여부). GPU 추론은 워커 스레드에서 — 이벤트 루프(/health)가 막히지 않게."""
        image_bytes = base64.b64decode(image_base64)
        return await asyncio.to_thread(self._generate_sync, image_bytes, prompt or DEFAULT_PROMPT)

    # ── 상태 ──────────────────────────────────────────────

    async def health(self) -> dict:
        ready = self._model is not None
        info = {"status": "ok" if ready else "loading", "backend": self.backend, "model": self.model, "model_available": ready}
        try:
            import torch

            if VLM_DEVICE.startswith("cuda") and torch.cuda.is_available():
                info["vram_gb"] = round(torch.cuda.memory_allocated() / 2**30, 1)
        except Exception:  # torch 미설치 등 — health 는 항상 응답해야 한다
            pass
        return info
