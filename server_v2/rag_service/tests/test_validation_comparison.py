import json
import unittest
from dataclasses import asdict

from evaluation.audit_validation_codes import compare_first_attempt
from rag.contracts import Evidence


class ValidationComparisonTests(unittest.TestCase):
    def test_unit_fix_is_reported_without_predicting_retry_answer(self):
        raw = json.dumps({"items": [{"part_id": "q1", "status": "supported",
            "answer": "준공은 2024년입니다.", "claims": [{"evidence_id": "E1",
            "quote": "2024 실증", "value": "2024년"}]}]}, ensure_ascii=False)
        issues = [{"part_id": "q1", "code": "value_not_in_quote", "message": "선택 값이 인용에 없음"},
                  {"part_id": "q1", "code": "answer_quantity_not_in_quote", "message": "인용에 없는 값·단위: 2024년"}]
        row = {"parts": [{"id": "q1", "question": "언제 준공했어?"}],
               "evidence": {"E1": asdict(Evidence("hash", "a.pdf", 1, 0, "2024 실증", "fact"))},
               "allowed": {"q1": ["E1"]}, "attempts": [
                   {"raw": raw, "issues": issues, "errors": [i["message"] for i in issues]},
                   {"raw": "not used"}]}
        result = compare_first_attempt(row)
        self.assertEqual(result["changes"][0]["removed"],
                         [("answer_quantity_not_in_quote", "인용에 없는 값·단위: 2024년")])
        self.assertFalse(result["retry_skip_candidate"])
        self.assertTrue(result["retry_feedback_changed"])

    def test_malformed_first_output_is_not_counted_as_clean(self):
        row = {"parts": [{"id": "q1", "question": "질문"}],
               "attempts": [{"raw": '{"items": [] , "items": []}'}]}
        self.assertEqual(compare_first_attempt(row)["state"], "parse_failure")


if __name__ == "__main__":
    unittest.main()
