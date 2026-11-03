"""클라우드에서 실행하는 모델 없는 TextQaClient 계약 검사."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient


class ClientContractTests(unittest.TestCase):
    def test_rag_json_and_sse_have_same_final_answer(self):
        from rag.contracts import Outcome
        with patch.dict(os.environ, {"INTERNAL_API_KEY": "test-internal",
                                    "INDEX_REVISION": "test-revision", "CHROMA_DIR": "/unused"}):
            path = Path(__file__).resolve().parents[1] / "main.py"
            spec = importlib.util.spec_from_file_location("v2_test_rag", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            result = Outcome("검사한 최종 답변입니다.", [], True, [])
            with patch.object(module.Search, "load"), patch.object(module.Pipeline, "answer", AsyncMock(return_value=result)):
                with TestClient(module.app) as client:
                    headers = {"X-Internal-Key": "test-internal"}
                    normal = client.post("/answer", headers=headers, json={"question": "질문"})
                    stream = client.post("/answer/stream", headers=headers, json={"question": "질문"})
                    self.assertEqual(normal.status_code, 200)
                    events = [json.loads(line[6:]) for line in stream.text.splitlines() if line.startswith("data: ")]
                    self.assertEqual([e["type"] for e in events], ["delta", "done"])
                    self.assertEqual(events[-1]["answer"], normal.json()["answer"])
                    self.assertEqual(events[0]["text"], normal.json()["answer"])

    def test_text_client_contract_and_institution_sessions(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            "INTERNAL_API_KEY": "test-internal", "KIOSK_API_KEY": "test-kiosk",
            "QA_LOG_PATH": str(Path(tmp) / "qa.jsonl")
        }):
            path = Path(__file__).resolve().parents[2] / "orchestrator/main.py"
            spec = importlib.util.spec_from_file_location("v2_test_orchestrator", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            reply = {"answer": "확인된 답입니다.", "grounded": True, "unverified": [],
                     "sources": [{"source_file": "a.pdf", "page_number": 1, "distance": 0.1}]}
            fake = AsyncMock(return_value=reply)
            with patch.object(module, "_answer", fake), TestClient(module.app) as client:
                headers = {"X-API-Key": "test-kiosk"}
                payload = {"question": "시설 위치 알려줘", "session_id": "same",
                           "institution_id": "A", "model": "a.x"}
                response = client.post("/ask/text", headers=headers, json=payload)
                self.assertEqual(response.status_code, 200)
                data = response.json()
                for key in ("request_id", "question", "answer", "grounded", "sources", "needs_retry"):
                    self.assertIn(key, data)
                self.assertIsInstance(data["sources"][0]["page_number"], int)
                client.post("/ask/text", headers=headers, json={**payload, "institution_id": "B"})
                self.assertEqual(fake.call_args.args[1], [])
                self.assertEqual(client.post("/ask/text", headers={"X-API-Key": "bad"}, json=payload).status_code, 403)
                blank = client.post("/ask/text", headers=headers, json={"question": ""}).json()
                self.assertTrue(blank["needs_retry"])


if __name__ == "__main__":
    unittest.main()
