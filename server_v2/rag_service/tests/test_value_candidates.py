import unittest
import json
from pathlib import Path
from unittest.mock import patch
from rag.config import Settings
from rag.pipeline import Pipeline
from rag.contracts import Evidence, ItemAnswer, Claim, Status
from rag.value_candidates import collect, audit


class ValueCandidateTests(unittest.TestCase):
    def evidence(self, text):
        return {"E1": Evidence("hash", "example.pdf", 1, 0, text, "fact")}

    def test_offsets_repeated_values_and_empty_input(self):
        text = "학생 18명, 교원 18명"
        candidates = collect(self.evidence(text))["E1"]
        self.assertEqual([c["raw"] for c in candidates], ["18명", "18명"])
        self.assertNotEqual(candidates[0]["id"], candidates[1]["id"])
        for c in candidates:
            self.assertEqual(text[c["start"]:c["end"]], c["raw"])
        self.assertEqual(collect({}), {})
        self.assertEqual(collect(self.evidence("English only"))["E1"], [])

    def test_parentheses_range_and_date_are_not_rewritten(self):
        text = "계 147(3), 예산 20~70백만원, 개관 2019.04.02"
        raw = [c["raw"] for c in collect(self.evidence(text))["E1"]]
        self.assertEqual(raw, ["147(3)", "20~70백만원", "2019.04.02"])

    def test_matching_does_not_accept_subnumbers_or_foreign_evidence(self):
        candidates = collect(self.evidence("연도 2017년, 인원 18명, 교원 18명"))
        def observe(value, allowed):
            item = ItemAnswer("q1", Status.SUPPORTED, "unchanged",
                              (Claim("E1", "unchanged", value),))
            result = audit([item], candidates, allowed)[0]
            self.assertEqual(item.answer, "unchanged")
            return result["status"]
        self.assertEqual(observe("2017년", {"q1": ["E1"]}), "unique_exact_candidate")
        self.assertEqual(observe("18명", {"q1": ["E1"]}), "multiple_exact_candidates")
        self.assertEqual(observe("17년", {"q1": ["E1"]}), "no_exact_candidate")
        self.assertEqual(observe("2017년", {"q1": ["E2"]}), "outside_allowed_evidence")


class CandidateObservationTests(unittest.IsolatedAsyncioTestCase):
    async def test_log_mode_preserves_response_and_generation_input(self):
        evidence = Evidence("hash", "example.pdf", 1, 0, "인원: 18명", "fact")
        class Search:
            def retrieve(self, parts, institution, top_k):
                return {parts[0].id: [evidence]}
        class Generator:
            def __init__(self):
                self.inputs = []
            async def generate(self, *args):
                self.inputs.append(args)
                return json.dumps({"items": [{"part_id": "q1", "status": "supported",
                    "answer": "인원은 18명입니다.", "claims": [{"evidence_id": "E1",
                    "quote": "인원: 18명", "value": "18명"}]}]})
        responses, inputs, traces = [], [], []
        for mode in ("off", "log"):
            generator = Generator()
            pipeline = Pipeline(Settings(Path("unused"), "rev", value_candidates_mode=mode),
                                Search(), generator)
            with patch.object(Pipeline, "record") as record:
                result = await pipeline.answer("인원은 몇 명인가요?")
                traces.append(record.call_args.args[0])
            responses.append(result.response())
            inputs.append(generator.inputs)
        self.assertEqual(responses[0], responses[1])
        self.assertEqual(inputs[0], inputs[1])
        self.assertNotIn("value_candidates", traces[0])
        self.assertEqual(traces[1]["value_candidate_audit"][0]["status"], "unique_exact_candidate")


if __name__ == "__main__":
    unittest.main()
