import unittest

from rag.contracts import Claim, Evidence, ItemAnswer, Part, Status, ValidationCode
from rag.rendering import render
from rag.validation import check


class ValidationCodeTests(unittest.TestCase):
    def test_value_and_answer_errors_keep_legacy_messages(self):
        evidence = {"E1": Evidence("hash", "a.pdf", 1, 0, "개관 2017년", "fact")}
        item = ItemAnswer("q1", Status.SUPPORTED, "개관 연도는 20117년입니다.",
                          (Claim("E1", "개관 2017년", "20117년"),))
        checked = check([item], evidence, {"q1": ["E1"]})[0]
        self.assertEqual(checked.errors, ["선택 값이 인용에 없음", "인용에 없는 값·단위: 20117년"])
        self.assertEqual([i.code for i in checked.issues],
                         [ValidationCode.VALUE_NOT_IN_QUOTE, ValidationCode.ANSWER_QUANTITY_NOT_IN_QUOTE])
        self.assertEqual(checked.issues[0].evidence_id, "E1")
        self.assertEqual(checked.issues[0].claim_index, 0)
        response = render([Part("q1", "언제 개관했어?")], [checked], evidence, "log").response()
        self.assertEqual(set(response), {"answer", "sources", "grounded", "unverified"})
        self.assertEqual(response["answer"], item.answer)
        self.assertFalse(response["grounded"])

    def test_duplicate_messages_keep_claim_locations(self):
        evidence = {"E1": Evidence("hash", "a.pdf", 1, 0, "시설 안내", "fact")}
        item = ItemAnswer("q1", Status.SUPPORTED, "안내입니다.",
                          (Claim("E1", "없는 구절", "값"), Claim("E1", "다른 구절", "값")))
        checked = check([item], evidence, {"q1": ["E1"]})[0]
        self.assertEqual(checked.errors, ["인용이 원문에 없음"])
        self.assertEqual([i.claim_index for i in checked.issues], [0, 1])
        self.assertTrue(all(i.code == ValidationCode.QUOTE_NOT_IN_SOURCE for i in checked.issues))

    def test_missing_output_is_explicit_parse_failure_in_internal_state(self):
        outcome = render([Part("q1", "질문")], [], {}, "log")
        self.assertEqual(outcome.unverified, ["q1: 출력 파싱 실패"])
        self.assertEqual(outcome.items[0]["issues"][0]["code"], ValidationCode.PARSE_FAILURE)


if __name__ == "__main__":
    unittest.main()
