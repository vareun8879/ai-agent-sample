"""질문 대상과 숫자가 같은 원문 구절에 있는지 관측한다. 답변에는 사용하지 않는다."""
from dataclasses import replace

from .textutil import question_tokens, whole_word_hits
from .value_candidates import collect as collect_values


def _lines(text):
    offset = 0
    for raw in text.splitlines(keepends=True):
        body = raw.rstrip("\r\n")
        left = len(body) - len(body.lstrip())
        right = len(body.rstrip())
        if right > left:
            yield offset + left, offset + right
        offset += len(raw)


def _values(ev, text):
    return collect_values({"source": replace(ev, text=text)})["source"]


def collect(question, evidence, allowed):
    """한 값의 원문 줄 또는 값만 있는 줄+다음 설명 줄을 위치와 함께 기록한다.

    실측(261104, b01): 값 후보의 넓은 문맥에 140년과 1885년 개교가 함께 보여
    모델이 140년을 골랐다. 한 구절에 값이 여러 개면 관측 후보에서 제외한다.
    """
    tokens = question_tokens(question)
    out = []
    for alias in allowed:
        ev = evidence.get(alias)
        if ev is None:
            continue
        lines = list(_lines(ev.text))
        for index, (start, end) in enumerate(lines):
            raw = ev.text[start:end]
            values = _values(ev, raw)
            if len(values) != 1:
                continue
            # 숫자 한 개만 적힌 PDF 줄의 바로 다음 줄이 설명일 때만 결합한다.
            # 기존 원문의 시작·끝 위치는 그대로 보존한다.
            if (values[0]["raw"] == raw and index + 1 < len(lines)):
                next_start, next_end = lines[index + 1]
                next_raw = ev.text[next_start:next_end]
                if not _values(ev, next_raw):
                    end = next_end
                    raw = ev.text[start:end]
            hits = whole_word_hits(raw, tokens)
            if not hits:
                continue
            out.append({"evidence_id": alias, "start": start, "end": end,
                        "raw": raw, "value": values[0]["raw"], "question_hits": hits,
                        "source_file": ev.source_file, "page_number": ev.page_number})
    return out
