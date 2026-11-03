"""원문 인용·값 검사. 의미적 관련성의 완전한 증명으로 취급하지 않는다.

실측(261102_5, m29): 파싱으로 첫 항목이 사라졌으며 기존 답을 정상으로 반환했다.
실측(261102_4, a22): 숫자가 같아도 만원/백만원 단위가 틀릴 수 있다.
새 값 검사는 먼저 log로 측정하고, 과거 통과 문항 오탐 검토 후 enforce로 전환한다.
"""
from decimal import Decimal
import re
import unicodedata
from .contracts import CheckedItem, Evidence, ItemAnswer

NUMBER = re.compile(r"(?<![\d.])\d[\d,]*(?:\.\d+)?")
QUANTITY = re.compile(r"(?<![\d.])(\d[\d,]*(?:\.\d+)?)\s*(백만원|천만원|억원|만원|천원|원|퍼센트|%|학급|개월|교시|학년|호선|층|명|개|점|년|월|일|회|종|실|TB|PB|PF|㎡)?", re.I)
SCALE = {"원": 1, "천원": 1000, "만원": 10000, "백만원": 1000000,
         "천만원": 10000000, "억원": 100000000}


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def quantities(text: str) -> set[tuple[Decimal, str]]:
    # 명확한 연.월(.일)만 날짜 표기로 바꾼다. 일반 소수는 보존한다.
    text = re.sub(r"(?<!\d)(\d{4})\.(\d{1,2})(?:\.(\d{1,2}))?\.?",
                  lambda m: f"{m[1]}년 {m[2]}월" + (f" {m[3]}일" if m[3] else ""), text)
    matches = list(QUANTITY.finditer(text))
    out = set()
    for i, m in enumerate(matches):
        num = Decimal(m[1].replace(",", ""))
        unit = m[2] or ""
        # 범위의 단위 생략만 허용: 50~150백만원. 임의의 주변 숫자에는 적용하지 않는다.
        if not unit and i + 1 < len(matches):
            following = matches[i + 1]
            if re.fullmatch(r"\s*(?:~|～|∼|–|부터)\s*", text[m.end():following.start()]):
                unit = following[2] or ""
        if unit in SCALE:
            num, unit = num * SCALE[unit], "원"
        if unit == "퍼센트":
            unit = "%"
        out.add((num, unit.lower()))
    return out


def check(items: list[ItemAnswer], evidence: dict[str, Evidence],
          allowed: dict[str, list[str]]) -> list[CheckedItem]:
    checked = []
    for item in items:
        errors = []
        quotes = []
        for claim in item.claims:
            ev = evidence.get(claim.evidence_id)
            if ev is None or claim.evidence_id not in allowed.get(item.part_id, []):
                errors.append("근거 ID가 해당 항목에 제공되지 않음")
                continue
            quote = normalized(claim.quote)
            if quote not in normalized(ev.text):
                errors.append("인용이 원문에 없음")
                continue
            quotes.append(claim.quote)
            if normalized(claim.value) not in quote:
                errors.append("선택 값이 인용에 없음")
            if quantities(claim.value):
                wanted = quantities(claim.value)
                actual = quantities(item.answer)
                if not wanted.issubset(actual):
                    errors.append("선택 값·단위가 항목 답변에 없음")
            elif normalized(claim.value) not in normalized(item.answer):
                errors.append("선택한 이름·설명 구절이 항목 답변에 없음")
        quoted = quantities("\n".join(quotes))
        for num, unit in quantities(item.answer):
            if (num, unit) not in quoted and (unit == "원" or (num, "") not in quoted):
                errors.append(f"인용에 없는 값·단위: {num}{unit}")
        checked.append(CheckedItem(item, list(dict.fromkeys(errors))))
    return checked
