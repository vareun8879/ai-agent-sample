from decimal import Decimal
import unittest

from rag.validation import quantities
from rag.validation import check
from rag.contracts import Claim, Evidence, ItemAnswer, Status


class UnitBoundaryTests(unittest.TestCase):
    def test_following_word_is_not_a_unit(self):
        for text, number in (("2024 실증·창업동 준공", "2024"),
                             ("2025\n개교", "2025"), ("1 회의실", "1")):
            with self.subTest(text=text):
                self.assertEqual(quantities(text), {(Decimal(number), "")})

    def test_attached_and_spaced_units_keep_particles(self):
        for text, number, unit in (("83명이", "83", "명"), ("83 명이", "83", "명"),
                                   ("5층에", "5", "층"), ("5 층에", "5", "층"),
                                   ("3 학년", "3", "학년"), ("27 일간", "27", "일"),
                                   ("50 백만원입니다", "50000000", "원")):
            with self.subTest(text=text):
                self.assertEqual(quantities(text), {(Decimal(number), unit)})

    def test_standalone_unit_on_next_line_is_preserved(self):
        for gap in ("\n", "\r\n"):
            self.assertEqual(quantities("83" + gap + "명"), {(Decimal("83"), "명")})
            self.assertEqual(quantities("2025" + gap + "개교"), {(Decimal("2025"), "")})

    def test_ranges_amounts_and_wrong_digits_remain_distinct(self):
        self.assertEqual(quantities("50 ~ 150 백만원"), quantities("50~150백만원"))
        self.assertEqual(quantities("100 ~ 250 백만원"), quantities("1억원 2.5억원"))
        self.assertNotEqual(quantities("50만원"), quantities("50백만원"))
        self.assertNotEqual(quantities("2,555명"), quantities("255명"))
        self.assertNotEqual(quantities("18885년"), quantities("1885년"))
        self.assertNotEqual(quantities("1888년"), quantities("1885년"))
        self.assertNotEqual(quantities("255(5)"), quantities("2555"))

    def test_known_numeric_errors_remain_validation_errors(self):
        for source, answer in (("1885년", "18885년"), ("1885년", "1888년"),
                               ("255명", "2,555명"), ("50백만원", "50만원"),
                               ("100~250백만원", "1000만원~2500만원"),
                               ("100~250백만원", "1000~2500백만원"),
                               ("2023년 11월", "2020년 11월")):
            with self.subTest(source=source, answer=answer):
                evidence = {"E1": Evidence("hash", "a.pdf", 1, 0, source, "fact")}
                item = ItemAnswer("q1", Status.SUPPORTED, answer, (Claim("E1", source, source),))
                self.assertTrue(check([item], evidence, {"q1": ["E1"]})[0].errors)


if __name__ == "__main__":
    unittest.main()
