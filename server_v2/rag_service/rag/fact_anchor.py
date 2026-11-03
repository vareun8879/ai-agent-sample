"""질문 대상이 사실 행의 첫 필드에 있는지 확인하고 후보를 관측한다.

사람·사물 종류나 답 값을 추측하지 않는다. 행 본문에 낱말이 우연히 나온
경우와 필드 머리에서 질문 대상을 가리키는 경우를 구별한다.
"""
import re

from .contracts import Evidence
from .textutil import question_tokens, strip_title, whole_word_hits


_FIELD_SPLIT = re.compile(r"\s+[—–]\s+|\s*[:：,|]\s*")
_ENUMERATION = re.compile(r"[·ㆍ・/]")


def anchor_fields(text: str) -> list[str]:
    """원문 앞부분의 표제·필드 값을 순서대로 돌려준다."""
    lines = strip_title(text).strip().splitlines()
    if not lines:
        return []
    body = lines[0]
    return [field.strip() for field in _FIELD_SPLIT.split(body, maxsplit=5)
            if field.strip()][:4]


def has_leading_anchor(question: str, evidence: Evidence) -> bool:
    """사실 행 첫 낱말이 질문의 온전한 낱말인지 확인한다."""
    if evidence.kind != "fact":
        return False
    fields = anchor_fields(evidence.text)
    if not fields:
        return False
    head = fields[0]
    if _ENUMERATION.search(head) and head not in question:
        return False
    first_word = re.match(r"[가-힣A-Za-z0-9]+", head)
    return bool(first_word and whole_word_hits(first_word.group(), question_tokens(question)))


def select_anchor_rescue(question: str, ranked: list[Evidence], selected: list[Evidence],
                         metadata: dict[str, dict], institution: str | None,
                         min_score: float, limit: int) -> Evidence | None:
    """절대 점수는 넘지만 상대 점수 때문에 빠진 단일 사실 행만 구제한다."""
    if len(selected) >= limit or any(has_leading_anchor(question, e) for e in selected):
        return None
    selected_ids = {e.id for e in selected}
    selected_pages = {(e.source_file, e.page_number) for e in selected}
    eligible = []
    for e in ranked:
        if e.id in selected_ids or e.score < min_score or not has_leading_anchor(question, e):
            continue
        if (e.source_file, e.page_number) in selected_pages:
            continue
        meta = metadata.get(e.id, {})
        if institution and meta.get("institution_id") != institution:
            continue
        eligible.append(e)
        if len(eligible) > 1:
            return None
    return eligible[0] if eligible else None


def anchor_probe(question: str, ranked: list[Evidence], selected: list[Evidence]) -> dict:
    """전체 사실 행 후보에서 질문 낱말과 필드가 만나는 위치를 기록한다."""
    tokens = question_tokens(question)
    selected_ids = {e.id for e in selected}
    matches = []
    for e in ranked:
        if e.kind != "fact":
            continue
        fields = anchor_fields(e.text)
        for position, field in enumerate(fields):
            # 나열된 여러 대상 중 일부만 맞는 행은 단독 대상의 근거로 삼지 않는다.
            if _ENUMERATION.search(field) and field not in question:
                continue
            hits = whole_word_hits(field, tokens)
            if hits:
                snippet = ", ".join(fields[position:position + 2])[:160]
                matches.append({"evidence_id": e.id, "source_file": e.source_file,
                                "page_number": e.page_number, "score": e.score,
                                "field": field[:160], "field_position": position,
                                "snippet": snippet,
                                "anchor_hits": hits, "selected": e.id in selected_ids})
                break
    matches.sort(key=lambda row: (-row["anchor_hits"], row["field_position"],
                                  -row["score"], row["evidence_id"]))
    return {"tokens": tokens, "matches": matches[:10],
            "matching_count": len(matches)}
