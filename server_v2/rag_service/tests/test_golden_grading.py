import unittest

from golden_eval import grade


def result(answer):
    return {"answer": answer, "grounded": True,
            "sources": [{"source_file": "document.pdf", "page_number": 8}],
            "unverified": []}


class GoldenGradingTests(unittest.TestCase):
    def test_legacy_false_pass_is_preserved_and_adjusted(self):
        case = {"id": "example", "question": "지원 규모는?", "doc": "document",
                "pages": [8], "keywords": ["100", "250"]}
        rules = [{"subject": [], "values": ["100백만원", "250백만원"]}]
        row = grade(case, result("지원 규모는 1000만원에서 2500만원입니다."), True, rules)
        self.assertTrue(row["answer_ok"])
        self.assertFalse(row["adjusted_answer_ok"])
        self.assertEqual(row["rubric_status"], "fail")

    def test_correct_values_and_uncovered_case(self):
        case = {"id": "example", "question": "지원 규모는?", "doc": "document",
                "pages": [8], "keywords": ["100", "250"]}
        rules = [{"subject": [], "values": ["100백만원", "250백만원"]}]
        correct = grade(case, result("지원 규모는 100~250백만원입니다."), True, rules)
        self.assertTrue(correct["adjusted_answer_ok"])
        self.assertEqual(correct["rubric_status"], "pass")
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
