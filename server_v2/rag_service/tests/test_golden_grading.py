import unittest

from golden_eval import grade
from evaluation.regrade import regrade_row


def result(answer):
    return {"answer": answer, "grounded": True,
            "sources": [{"source_file": "document.pdf", "page_number": 8}],
            "unverified": []}


class GoldenGradingTests(unittest.TestCase):
    def test_refusal_content_not_grounded_flag_controls_adjusted_grade(self):
        case = {"id": "outside", "question": "자료 밖의 질문", "grounded": False}
        for answer, status, accepted in (
            ("자료에서 답을 확인하지 못했습니다.", "pass", True),
            ("담당자의 이름은 제공된 자료에서 확인할 수 없습니다.", "pass", True),
            ("자료에서 확인할 수 없습니다. 문의 번호는 1234입니다.", "review", False),
            ("", "fail", False),
            ("공사는 2099년에 시작했습니다.", "review", False),
            ("자료에서 확인하지 못했지만 2099년입니다.", "review", False),
        ):
            with self.subTest(answer=answer):
                data = dict(result(answer), grounded=False, sources=[])
                live = grade(case, data, True)
                saved = regrade_row(case, live)
                self.assertEqual(live["rubric_status"], status)
                self.assertEqual(live["adjusted_answer_ok"], accepted)
                self.assertEqual(saved["adjusted_answer_ok"], accepted)
                self.assertEqual(saved["rubric_reasons"], live["rubric_reasons"])

    def test_report_keeps_full_answer_and_marks_actual_retrieval_unknown(self):
        case = {"id": "example", "question": "설명해줘", "keywords": []}
        answer = "문서의 설명입니다. " * 40
        data = dict(result(answer), grounded=False, unverified=["검증 오류"])
        row = grade(case, data, True)
        self.assertEqual(row["answer"], answer)
        self.assertFalse(row["answer_truncated"])
        self.assertEqual(row["sources"], data["sources"])
        self.assertFalse(row["grounded"])
        self.assertEqual(row["unverified"], ["검증 오류"])
        self.assertIsNone(row["actual_retrieval_ok"])
        self.assertEqual(row["retrieval_basis"], "answer_sources")
        self.assertEqual(regrade_row(case, row)["rubric_status"], row["rubric_status"])

    def test_old_truncated_report_is_review_and_missing_metadata_stays_unknown(self):
        old = {"id": "example", "answer": "가" * 300, "answer_ok": True, "retrieval_ok": True}
        row = regrade_row({"grounded": True}, old)
        self.assertEqual(row["rubric_status"], "review")
        self.assertFalse(row["adjusted_answer_ok"])
        self.assertIsNone(row["grounded"])
        self.assertIsNone(row["sources"])
        self.assertIsNone(row["unverified"])

    def test_call_failure_is_not_refusal_pass(self):
        case = {"id": "outside", "question": "자료 밖의 질문", "grounded": False, "_error": "unavailable"}
        row = grade(case, dict(result(""), sources=[], grounded=False), True)
        self.assertFalse(row["answer_ok"])
        self.assertFalse(row["adjusted_answer_ok"])
        self.assertFalse(row["citation_ok"])
        self.assertEqual(regrade_row(case, row)["rubric_status"], "fail")

    def test_legacy_false_pass_is_preserved_and_adjusted(self):
        case = {"id": "example", "question": "지원 규모는?", "doc": "document",
                "pages": [8], "keywords": ["100", "250"]}
        rules = [{"subject": [], "values": ["100백만원", "250백만원"]}]
        row = grade(case, result("지원 규모는 1000만원에서 2500만원입니다."), True, rules)
        self.assertTrue(row["answer_ok"])
        self.assertFalse(row["adjusted_answer_ok"])
        self.assertEqual(row["rubric_status"], "fail")
        self.assertEqual(regrade_row(case, row, rules)["adjusted_answer_ok"], row["adjusted_answer_ok"])

    def test_correct_values_and_uncovered_case(self):
        case = {"id": "example", "question": "지원 규모는?", "doc": "document",
                "pages": [8], "keywords": ["100", "250"]}
        rules = [{"subject": [], "values": ["100백만원", "250백만원"]}]
        correct = grade(case, result("지원 규모는 100~250백만원입니다."), True, rules)
        self.assertTrue(correct["adjusted_answer_ok"])
        self.assertEqual(correct["rubric_status"], "pass")
        self.assertEqual(regrade_row(case, correct, rules)["adjusted_answer_ok"], correct["adjusted_answer_ok"])
        uncovered = grade(case, result("지원 규모는 1000만원에서 2500만원입니다."), True)
        self.assertEqual(uncovered["rubric_status"], "legacy_only")
        self.assertTrue(uncovered["adjusted_answer_ok"])

    def test_matching_digits_with_wrong_unit_stays_false(self):
        case = {"id": "example", "question": "지원 규모는?", "doc": "document",
                "pages": [8], "keywords": ["50", "150"]}
        rules = [{"subject": [], "values": ["50백만원", "150백만원"]}]
        row = grade(case, result("지원 규모는 50만원에서 150백만원입니다."), True, rules)
        self.assertTrue(row["answer_ok"])
        self.assertFalse(row["adjusted_answer_ok"])


if __name__ == "__main__":
    unittest.main()
