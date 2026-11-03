"""사실 행의 필드 위치 관측은 사람·사물 종류와 무관하다."""
from dataclasses import replace
import unittest

from rag.contracts import Evidence
from rag.fact_anchor import anchor_fields, anchor_probe, has_leading_anchor, select_anchor_rescue


def fact(key, text, score, page=1):
    return Evidence(key, "a.pdf", page, 0, text, "fact", score=score)


class FactAnchorTests(unittest.TestCase):
    def test_empty_text_has_no_fields(self):
        self.assertEqual(anchor_fields(""), [])

    def test_fields_preserve_both_sides_of_label(self):
        self.assertEqual(anchor_fields("[문서]\n조직 — 부서: 업무 보건, 담당자 조OO"),
                         ["조직", "부서", "업무 보건", "담당자 조OO"])

    def test_observes_target_field_without_classifying_answer_type(self):
        direct = fact("direct", "[문서]\n조직 — 부서: 업무 보건, 담당자 조OO", 0.002)
        list_member = fact("list", "[문서]\n체육·음악·미술·보건 부장 이OO", 0.01)
        unrelated = fact("unrelated", "[문서]\n예방 교육: 담당자 보건교사", 0.2)
        result = anchor_probe("보건 선생님은 누구야?", [unrelated, list_member, direct], [])
        self.assertEqual(result["matching_count"], 1)
        self.assertEqual(result["matches"][0]["evidence_id"], "direct")
        self.assertEqual(result["matches"][0]["field"], "업무 보건")
        self.assertEqual(result["matches"][0]["snippet"], "업무 보건, 담당자 조OO")

    def test_selected_state_is_observation_only(self):
        row = fact("place", "[문서]\n기숙사: 우남학사", 0.9)
        selected = [replace(row, score=0.9)]
        result = anchor_probe("기숙사 이름이 뭐야?", [row], selected)
        self.assertTrue(result["matches"][0]["selected"])
        self.assertEqual(selected, [row])

    def test_unique_head_fact_can_pass_relative_cut_only(self):
        generic = fact("generic", "[문서]\n교사는 학교장 지침을 따른다", 0.6)
        direct = fact("direct", "[문서]\n관장 김OO은 시설을 운영한다", 0.06, page=2)
        mention = fact("mention", "[문서]\n부관장 이OO은 관장 김OO을 돕는다", 0.04, page=3)
        metadata = {e.id: {"institution_id": "A"} for e in (generic, direct, mention)}
        self.assertTrue(has_leading_anchor("관장 성함이 뭐야?", direct))
        self.assertFalse(has_leading_anchor("관장 성함이 뭐야?", mention))
        self.assertEqual(select_anchor_rescue("관장 성함이 뭐야?", [generic, direct, mention],
                                              [generic], metadata, "A", 0.03, 6), direct)

    def test_ambiguity_scope_and_limit_keep_original_selection(self):
        selected = fact("selected", "[문서]\n다른 항목 설명", 0.7)
        first = fact("first", "[문서]\n관장 김OO", 0.06, page=2)
        second = fact("second", "[문서]\n관장 박OO", 0.05, page=3)
        metadata = {e.id: {"institution_id": "A"} for e in (selected, first, second)}
        args = ("관장 성함이 뭐야?", [selected, first, second], [selected])
        self.assertIsNone(select_anchor_rescue(*args, metadata, "A", 0.03, 6))
        self.assertIsNone(select_anchor_rescue(*args, metadata, "A", 0.03, 1))
        self.assertIsNone(select_anchor_rescue("관장 성함이 뭐야?", [selected, first],
                                               [selected], metadata, "B", 0.03, 6))
        self.assertIsNone(select_anchor_rescue("관장 성함이 뭐야?", [selected, first],
                                               [selected], metadata, "A", 0.07, 6))
        self.assertIsNone(select_anchor_rescue("관장 성함이 뭐야?", [first, second],
                                               [first], metadata, "A", 0.03, 6))

    def test_same_page_is_not_added_twice(self):
        selected = fact("selected", "[문서]\n다른 항목 설명", 0.7)
        candidate = fact("candidate", "[문서]\n관장 김OO", 0.06)
        metadata = {e.id: {"institution_id": "A"} for e in (selected, candidate)}
        self.assertIsNone(select_anchor_rescue("관장 성함이 뭐야?", [selected, candidate],
                                               [selected], metadata, "A", 0.03, 6))


if __name__ == "__main__":
    unittest.main()
