"""VLM 생성 — 이미지 + (선택) 프롬프트를 비전 모델에 넘기고 텍스트 응답을 받는다.

백엔드는 vLLM(OpenAI 호환 API) 하나다. 모델에 대해서는 아무 가정도 하지 않는다 —
Qwen2.5-VL이든 A.X-4.0-VL-Light든, vLLM으로 서빙되는 비전 모델이면
`VLLM_URL` + `VLM_MODEL` 두 환경변수만 맞추면 그대로 쓸 수 있다.

    vllm serve Qwen/Qwen3-VL-8B-Instruct-FP8 --port 10001

`VLM_MODEL`은 vLLM의 `GET /v1/models`가 반환하는 id(`--served-model-name`을 줬다면
그 별칭)와 **정확히 일치**해야 한다. 불일치하면 `/health`의 `model_available`이
false가 되고, 인덱싱 배치는 기동을 거부한다.
"""

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger("vlm.generator")

# 현재 운영 중인 vLLM 서버(Qwen2.5-VL, :10001)를 기본값으로 둔다
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:10001/v1")
# 모델 id는 배포 환경마다 다르므로 기본값을 두지 않는다 — 미설정 시 기동을 거부해
# "엉뚱한 모델 이름으로 조용히 실패"하는 상황을 막는다
VLM_MODEL = os.getenv("VLM_MODEL", "")

# 빽빽한 페이지는 4,096토큰 출력에 2분 가까이 걸린다 — 인덱서 쪽 VLM_TIMEOUT 과 맞춘다
VLM_TIMEOUT = float(os.getenv("VLM_TIMEOUT", "360"))
VLM_TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0.2"))


def _mime_of(image_base64: str) -> str:
    """base64 앞머리로 이미지 포맷을 판별한다 — 인덱서는 JPEG(용량), 다른 호출자는 PNG 를 보낼 수 있다."""
    head = image_base64[:8]
    if head.startswith("/9j/"):
        return "image/jpeg"
    if head.startswith("iVBOR"):
        return "image/png"
    return "image/png"
# 텍스트가 많은 문서 페이지가 서버 기본값에 걸려 중간에 잘리지 않도록 요청에 명시한다.
# 밀도 높은 표 페이지는 전사에 3천 토큰 가까이 필요할 수 있다 — vLLM --max-model-len이
# (이미지 토큰 + 이 값)을 감당할 만큼 커야 한다 (250 DPI A4 기준 이미지 약 6천 토큰, 서버는 32768).
VLM_MAX_TOKENS = int(os.getenv("VLM_MAX_TOKENS", "4096"))

# 문서 이미지(스캔본 페이지, 표, 화면 캡처 등) 이해를 기본 용도로 잡은 프롬프트.
# 호출자가 DescribeRequest.prompt로 용도에 맞게 덮어쓸 수 있다.
DEFAULT_PROMPT = (
    "이미지에 있는 내용을 한국어로 옮겨 적어라. "
    "표가 있으면 행과 열의 관계를 살려 정리하고, "
    "글자가 있으면 최대한 정확하게 그대로 옮겨 적어라. "
    "표에서 세로로 병합된 셀(여러 행에 걸친 부서명·분류명 등)은 생략하지 말고 "
    "해당하는 각 행에 그 값을 반복해서 적어라. "
    "의미 있는 텍스트나 표가 없는 사진/도해라면 무엇을 보여주는지 한 문장으로 요약하라."
)


class Generator:
    """이미지(+선택 프롬프트) → vLLM(OpenAI 호환) 비전 모델의 텍스트 응답."""

    def __init__(self) -> None:
        if not VLM_MODEL:
            raise RuntimeError(
                "VLM_MODEL 환경변수가 설정되지 않았습니다. "
                "vLLM의 GET /v1/models가 반환하는 모델 id와 동일하게 지정하세요 "
                "(예: VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct-FP8)."
            )
        self.model = VLM_MODEL
        self.url = VLLM_URL

    async def describe(self, image_base64: str, prompt: Optional[str] = None) -> tuple[str, bool]:
        """(전사 텍스트, 잘림 여부)를 돌려준다.

        잘림(finish_reason == 'length')은 호출자에게 반드시 전달해야 한다 — 표 절반만
        전사된 텍스트가 정상 결과인 것처럼 인덱싱되는 조용한 실패를 막기 위해서다.
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or DEFAULT_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:{_mime_of(image_base64)};base64,{image_base64}"}},
                    ],
                }
            ],
            "temperature": VLM_TEMPERATURE,
            "max_tokens": VLM_MAX_TOKENS,
        }
        async with httpx.AsyncClient(timeout=VLM_TIMEOUT) as client:
            resp = await client.post(f"{self.url}/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        choice = data["choices"][0]
        truncated = choice.get("finish_reason") == "length"
        return choice["message"]["content"].strip(), truncated

    async def health(self) -> dict:
        """vLLM 서버가 떠 있고 대상 모델이 로드돼 있는지 확인한다."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.url}/models")
                resp.raise_for_status()
                data = resp.json()
            available = [m.get("id", "") for m in data.get("data", [])]
            model_ready = self.model in available
            result = {"status": "ok", "backend": "vllm", "model": self.model, "model_available": model_ready}
            if not model_ready:
                # 호출자가 원인을 바로 알 수 있게 서버가 실제로 서빙 중인 목록을 함께 준다
                result["served_models"] = available
            return result
        except Exception as exc:  # vLLM 미기동 등
            return {"status": "unreachable", "backend": "vllm", "model": self.model, "error": str(exc)}
