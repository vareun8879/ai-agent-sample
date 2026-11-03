"""모델 출력은 JSON 전체로 파싱한다. 마지막 답변만 선택하는 폴백은 없다."""
import json
import re
from .contracts import Claim, ItemAnswer, Part, Status


class ParseError(ValueError):
    pass


def _unique_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ParseError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def parse(text: str, parts: list[Part]) -> list[ItemAnswer]:
    text = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1)
    try:
        payload = json.loads(text, object_pairs_hook=_unique_pairs)
    except (ValueError, TypeError) as exc:
        raise ParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"items"} or not isinstance(payload["items"], list):
        raise ParseError("expected an object with items array")
    expected = {p.id for p in parts}
    seen = set()
    items = []
    for row in payload["items"]:
        if not isinstance(row, dict) or set(row) != {"part_id", "status", "answer", "claims"}:
            raise ParseError("item fields must be part_id, status, answer, claims")
        pid = row["part_id"]
        if not isinstance(pid, str) or pid not in expected or pid in seen:
            raise ParseError("unknown or repeated part_id")
        seen.add(pid)
        if row["status"] not in ("supported", "insufficient", "conflict"):
            raise ParseError("unknown status")
        if not isinstance(row["answer"], str) or not isinstance(row["claims"], list):
            raise ParseError("answer/claims type mismatch")
        claims = []
        for c in row["claims"]:
            if not isinstance(c, dict) or set(c) != {"evidence_id", "quote", "value"}:
                raise ParseError("invalid claim fields")
            if not all(isinstance(v, str) and v.strip() for v in c.values()):
                raise ParseError("claim fields must be nonempty strings")
            claims.append(Claim(c["evidence_id"], c["quote"], c["value"]))
        if row["status"] == "supported" and (not row["answer"].strip() or not claims):
            raise ParseError("supported item requires answer and claims")
        if row["status"] != "supported" and (row["answer"].strip() or claims):
            raise ParseError("unresolved item must have empty answer and claims")
        items.append(ItemAnswer(pid, Status(row["status"]), row["answer"].strip(), tuple(claims)))
    if seen != expected:
        raise ParseError("missing question part")
    by_id = {i.part_id: i for i in items}
    return [by_id[p.id] for p in parts]
