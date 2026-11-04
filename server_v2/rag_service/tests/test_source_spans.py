import unittest

from rag.contracts import Evidence
from rag.source_spans import collect


class SourceSpanTests(unittest.TestCase):
    def test_split_numeric_line_from_previous_number_and_preserve_source(self):
        text = ("140년의 전통을 이어 온 학교입니다.\n"
                "1885년\n개교 (설립)\n"
                "약 1,300명\n재학생 수\n")
        evidence = {"E1": Evidence("page", "school.pdf", 1, 0, text, "page")}
        rows = collect("학교는 언제 개교했어?", evidence, ["E1"])
        self.assertEqual([r["value"] for r in rows], ["1885년"])
        self.assertEqual(rows[0]["raw"], "1885년\n개교 (설립)")
        self.assertEqual(text[rows[0]["start"]:rows[0]["end"]], rows[0]["raw"])

    def test_multiple_values_in_one_fact_row_are_not_split_by_guess(self):
        evidence = {"E1": Evidence("fact", "school.pdf", 1, 0,
                                    "1885년 개교, 1,300명 학생 수", "fact")}
        self.assertEqual(collect("언제 개교했어?", evidence, ["E1"]), [])
        self.assertEqual(collect("언제 개교했어?", {}, []), [])


if __name__ == "__main__":
    unittest.main()
