from .contracts import Evidence, Part


def arrange(parts: list[Part], candidates: dict[str, list[Evidence]], budget: int):
    """항목별 순환 배분. 원문을 중간에서 자르지 않고 넣을 수 있는 근거만 제공한다."""
    registry = {}
    aliases = {}
    allowed = {p.id: [] for p in parts}
    used = 0
    depth = max((len(v) for v in candidates.values()), default=0)
    for rank in range(depth):
        for part in parts:
            rows = candidates.get(part.id, [])
            if rank >= len(rows):
                continue
            e = rows[rank]
            if e.id not in aliases:
                if used + len(e.text) > budget:
                    continue
                alias = f"E{len(registry) + 1}"
                aliases[e.id] = alias
                registry[alias] = e
                used += len(e.text)
            allowed[part.id].append(aliases[e.id])
    return registry, allowed
