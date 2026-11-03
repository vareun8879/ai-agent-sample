import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from rag.config import Settings
from rag.contracts import Part, Evidence, Status, evidence_id
from rag.parsing import parse, ParseError
from rag.pipeline import Pipeline
from rag.rendering import render
from rag.validation import check, quantities
from evaluation.rubric import assess


PARTS = [Part("q1", "진료실은 몇 층인가?"), Part("q2", "자료실은 몇 층인가?")]
EVIDENCE = {
    "E1": Evidence("hash1", "a.pdf", 1, 0, "1층: 진료실", "fact"),
    "E2": Evidence("hash2", "a.pdf", 1, 1, "2층: 자료실", "fact"),
}
ALLOWED = {"q1": ["E1"], "q2": ["E2"]}


def output():
    return {"items": [
        {"part_id": "q1", "status": "supported", "answer": "진료실은 1층입니다.",
         "claims": [{"evidence_id": "E1", "quote": "1층: 진료실", "value": "1층"}]},
        {"part_id": "q2", "status": "supported", "answer": "자료실은 2층입니다.",
         "claims": [{"evidence_id": "E2", "quote": "2층: 자료실", "value": "2층"}]}]}


class CoreTests(unittest.TestCase):
    def test_both_answers_survive(self):
        items = parse(json.dumps(output()), PARTS)
        result = render(PARTS, check(items, EVIDENCE, ALLOWED), EVIDENCE, "enforce")
        self.assertTrue(result.grounded)
        self.assertIn("진료실은 1층", result.answer)
        self.assertIn("자료실은 2층", result.answer)

    def test_multiple_answer_labels_are_not_silently_truncated(self):
        with self.assertRaises(ParseError):
            parse("근거: 1층\n답변: 진료실 1층\n근거: 2층\n답변: 자료실 2층", PARTS)

    def test_missing_or_duplicate_part(self):
        for rows in ([output()["items"][0]], [output()["items"][0]] * 2):
            with self.assertRaises(ParseError):
                parse(json.dumps({"items": rows}), PARTS)

    def test_duplicate_json_key(self):
        with self.assertRaises(ParseError):
            parse('{"items": [], "items": []}', PARTS)

    def test_unknown_evidence_id(self):
        data = output()
        data["items"][0]["claims"][0]["evidence_id"] = "E2"
        checked = check(parse(json.dumps(data), PARTS), EVIDENCE, ALLOWED)
        self.assertTrue(checked[0].errors)

    def test_bad_answer_never_has_grounded_true(self):
        data = output()
        data["items"][1]["answer"] = "자료실은 1층입니다."
        checked = check(parse(json.dumps(data), PARTS), EVIDENCE, ALLOWED)
        for mode in ("log", "enforce"):
            result = render(PARTS, checked, EVIDENCE, mode)
            self.assertFalse(result.grounded)
            self.assertTrue(result.unverified)
        self.assertNotIn("자료실은 1층", render(PARTS, checked, EVIDENCE, "enforce").answer)

    def test_amount_units_and_parenthesized_values(self):
        self.assertNotEqual(quantities("50만원"), quantities("50백만원"))
        self.assertEqual(quantities("50~150백만원"), quantities("5천만원 1.5억원"))
        self.assertNotEqual(quantities("255(5)"), quantities("2555"))

    def test_stable_versioned_identity(self):
        args = ("rev1", "a.pdf", 1, "f0", "text")
        self.assertEqual(evidence_id(*args), evidence_id(*args))
        self.assertNotEqual(evidence_id(*args), evidence_id("rev2", *args[1:]))
        self.assertNotEqual(evidence_id(*args), evidence_id("rev1", "b.pdf", 1, "f0", "text"))

    def test_rubric_reversed_targets_and_wrong_units(self):
        rules = [{"subject": ["진료실"], "values": ["1층"]},
                 {"subject": ["자료실"], "values": ["2층"]}]
        self.assertEqual(assess("진료실은 2층입니다. 자료실은 1층입니다.", rules)[0], "fail")
        self.assertEqual(assess("진료실은 1층입니다. 자료실은 2층입니다.", rules)[0], "pass")
        self.assertEqual(assess("50만원부터 150백만원입니다.",
                               [{"values": ["50백만원", "150백만원"]}])[0], "fail")


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_call_budget_and_parse_repair(self):
        class Search:
            def retrieve(self, parts, institution, top_k):
                return {parts[0].id: [EVIDENCE["E1"]]}
        class Generator:
            calls = 0
            async def generate(self, *args):
                self.calls += 1
                return "invalid"
        gen = Generator()
        pipeline = Pipeline(Settings(Path("unused"), "rev"), Search(), gen)
        with patch.object(Pipeline, "record"):
            result = await pipeline.answer("진료실은 몇 층인가?")
        self.assertEqual(gen.calls, 2)
        self.assertFalse(result.grounded)
        self.assertTrue(result.unverified)

    async def test_no_evidence_no_model_calls(self):
        class Search:
            def retrieve(self, parts, institution, top_k):
                return {p.id: [] for p in parts}
        class Generator:
            async def generate(self, *args):
                raise AssertionError("must not call LLM")
        with patch.object(Pipeline, "record"):
            result = await Pipeline(Settings(Path("unused"), "rev"), Search(), Generator()).answer("없는 질문")
        self.assertFalse(result.grounded)
        self.assertEqual(result.calls, 0)


if __name__ == "__main__":
    unittest.main()
