"""검색 진단이 검색 결과를 바꾸지 않는지 검사한다."""
from pathlib import Path
import unittest
from unittest.mock import patch

from rag.config import Settings
from rag.contracts import Part
from rag.pipeline import Pipeline
from rag.retrieval import Search


class _Vectors:
    def tolist(self):
        return [[0.1, 0.2]]


class _Embed:
    def encode(self, queries, normalize_embeddings=False):
        return _Vectors()


class _Collection:
    def count(self):
        return 2

    def query(self, **kwargs):
        return {
            "ids": [["page-19", "person-row-26"]],
            "documents": [["교사는 학교장 경영관에 속한다", "교무부장: 홍길동"]],
            "metadatas": [[
                {"scope": "institution", "institution_id": "school", "source_file": "a.pdf", "page_number": 19},
                {"scope": "institution", "institution_id": "school", "source_file": "a.pdf", "page_number": 26,
                 "kind": "fact"},
            ]],
            "distances": [[0.1, 0.2]],
        }


class _Reranker:
    def predict(self, pairs, show_progress_bar=False):
        return [0.8, 0.1] if len(pairs) == 2 else [0.5]


class RetrievalTraceTests(unittest.TestCase):
    def make_search(self, reranker=None):
        search = Search(Settings(Path("unused"), "rev", top_k=1, fact_anchor_mode="log"))
        search.collection = _Collection()
        search.embed = _Embed()
        search.rerank = reranker or _Reranker()
        return search

    def test_trace_preserves_result_and_shows_filtered_candidate(self):
        search = self.make_search()
        parts = [Part("q1", "교무부장은 누구야?")]

        ordinary = search.retrieve(parts, "school")
        traced, stages = search.retrieve_with_trace(parts, "school")

        self.assertEqual(ordinary, traced)
        self.assertEqual([e.page_number for e in traced["q1"]], [19])
        self.assertEqual([hit["page_number"] for hit in stages["q1"]["vector"]], [19, 26])
        filtered = next(hit for hit in stages["q1"]["ranked"] if hit["page_number"] == 26)
        self.assertEqual(filtered["chunk_id"], "person-row-26")
        self.assertFalse(filtered["selected"])
        self.assertGreater(stages["q1"]["floor"], filtered["score"])
        probe = stages["q1"]["fact_anchor"]["matches"]
        self.assertEqual(probe[0]["page_number"], 26)
        self.assertEqual(probe[0]["snippet_score"], 0.5)

    def test_failed_observation_does_not_fail_search(self):
        class BrokenSnippetReranker(_Reranker):
            def predict(self, pairs, show_progress_bar=False):
                if len(pairs) == 1:
                    raise RuntimeError("diagnostic failure")
                return super().predict(pairs, show_progress_bar=show_progress_bar)

        search = self.make_search(BrokenSnippetReranker())
        parts = [Part("q1", "교무부장은 누구야?")]
        ordinary = search.retrieve(parts, "school")
        traced, stages = search.retrieve_with_trace(parts, "school")
        self.assertEqual(ordinary, traced)
        self.assertEqual(stages["q1"]["fact_anchor_error"], "RuntimeError")

    def test_active_rescue_adds_one_fact_without_changing_main_scores(self):
        search = self.make_search()
        search.s = Settings(Path("unused"), "rev", top_k=6, fact_anchor_mode="on")
        parts = [Part("q1", "교무부장은 누구야?")]
        result, stages = search.retrieve_with_trace(parts, "school")
        self.assertEqual([e.page_number for e in result["q1"]], [26, 19])
        self.assertEqual(stages["q1"]["anchor_rescue"]["page_number"], 26)
        self.assertNotIn("fact_anchor", stages["q1"])


class PipelineTraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_pipeline_records_search_stages(self):
        stages = {"q1": {"query": "질문", "vector": [], "lexical": [], "ranked": [], "floor": None}}

        class EmptySearch:
            def retrieve_with_trace(self, parts, institution, top_k):
                return {p.id: [] for p in parts}, stages

        class NoGenerator:
            async def generate(self, *args):
                raise AssertionError("empty evidence should not call the model")

        with patch.object(Pipeline, "record") as record:
            await Pipeline(Settings(Path("unused"), "rev"), EmptySearch(), NoGenerator()).answer("질문")
        self.assertEqual(record.call_args.args[0]["retrieval_stages"], stages)


if __name__ == "__main__":
    unittest.main()
