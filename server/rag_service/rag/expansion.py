"""1.5단계 — 컨텍스트 확장 (small-to-big 페이지 병합).

검색(retriever.py)은 400~1,200자 조각 단위로 해야 임베딩이 정확하지만, 그 조각만
LLM에 주면 표처럼 페이지 하나가 한 덩어리인 내용이 반쪽만 전달된다. 그래서 검색된
조각이 속한 **페이지 전체**를 Chroma에서 다시 가져와 병합해 LLM 컨텍스트로 쓴다
— "작게 검색하고 크게 읽기(small-to-big retrieval)" 패턴.

추가로 **같은 문서의 연속 페이지는 한 단위로 묶는다**. 큰 표가 페이지를 넘어
이어지는 경우(실측: 업무조직 표 p27~28) 앞장만 검색돼도 뒷장이 함께 들어가야
"전체 설명" 질문에 완전한 답이 나온다.

컨텍스트 총량은 CONTEXT_CHAR_BUDGET으로 제한한다 — LLM 컨텍스트 창(num_ctx)을
넘으면 Ollama가 프롬프트 앞부분(시스템 프롬프트!)부터 조용히 잘라내기 때문이다.
"""

import logging
import time

from .config import (CONTEXT_CHAR_BUDGET, HIGHLIGHT_FILL, HIGHLIGHT_MAX, PAGE_EXPAND,
                     COMPOUND_HIGHLIGHT_MODE, COMPOUND_HIGHLIGHT_MAX_FACTS, RERANK_MIN_SCORE)
from .textutil import (filter_survey_lines, question_overlap,
                       question_tokens, split_compound_question, strip_title, whole_word_hits)

logger = logging.getLogger("rag.expansion")


def _rank(c: dict) -> float:
    """정렬용 관련도 키 — reranker가 붙인 rank_key(1-점수)가 있으면 그것을,
    없으면 임베딩 거리를 쓴다. 어느 쪽이든 작을수록 관련도가 높다."""
    return c.get("rank_key", c["distance"])


def page_text(collection, src: str, page: int) -> str:
    """해당 페이지의 본문 조각(kind=page)을 chunk_index 순으로 이어 붙인 전문을 돌려준다.

    표 행 사실 조각(kind=fact — chunk_documents.table_facts)은 검색·재채점 전용이라 제외한다:
    같은 표를 행 문장으로 한 번 더 붙이면 컨텍스트만 두 배가 된다. 옛 인덱스(kind 없음)는 전부 본문이다."""
    r = collection.get(
        where={"$and": [{"source_file": src}, {"page_number": page}]},
        include=["documents", "metadatas"],
    )
    parts = sorted(
        ((m, d) for m, d in zip(r["metadatas"], r["documents"]) if m.get("kind", "page") == "page"),
        key=lambda x: x[0].get("chunk_index", 0),
    )
    return "\n".join(doc for _, doc in parts)


def page_facts(collection, src: str, page: int) -> list[str]:
    """해당 페이지의 표 행·도해 사실 조각(kind=fact) 본문 목록 — 핵심 근거 보강용. 옛 인덱스(kind 없음)면 빈 목록."""
    r = collection.get(
        where={"$and": [{"source_file": src}, {"page_number": page}, {"kind": "fact"}]},
        include=["documents", "metadatas"],
    )
    parts = sorted(zip(r["metadatas"], r["documents"]), key=lambda x: x[0].get("chunk_index", 0))
    return [strip_title(d).strip() for _, d in parts if d and d.strip()]


def _extra_highlights(collection, src: str, page: int, question: str, have: list[str], room: int, above: int = 0) -> list[str]:
    """검색에는 안 걸렸지만 질문 낱말과 검색된 행(above 점)보다 **더 많이** 겹치는 그 페이지의 fact 줄 — 점수순, room 개까지.
    같은 점수의 이웃 행은 넣지 않는다 — 실측(09-18): 같은 표의 다른 행(교원 여 13, 2호 조치)이 같이 들어가 오답이 늘었다.

    실측(09-18): "예산 총액은?" 의 정답 조각 "예산계획 — 총액(원): 2,000,000" 이 색인은 됐는데 reranker 상위에
    못 들어 [핵심 근거]에도, 폴백 후보에도 없었다(같은 유형: 수업공개주간 5/26 행, 동아리 20번 행, 학교폭력 담당 행).
    페이지는 이미 맞게 골랐으니, 그 페이지 안에서 질문 낱말이 든 행을 찾는 건 검색이 아니라 문자열 대조로 충분하다."""
    if room <= 0 or not question or not HIGHLIGHT_FILL:
        return []
    toks = question_tokens(question)
    if not toks:
        return []
    try:
        facts = page_facts(collection, src, page)
    except Exception as exc:
        logger.warning("페이지 fact 조회 실패 (%s p%s) — 보강 생략: %s", src, page, exc)
        return []
    # 점수 같으면 문서 순서(표의 위 행부터) — "언제부터 시작?" 처럼 첫 행이 답인 질문에서 뒷 행이 먼저 오지 않게.
    # (폴백 pick_evidence_line 은 짧은 줄 우선이지만, 여기는 모델에게 보여 줄 후보라 원문 순서가 자연스럽다)
    scored = [(question_overlap(question, f, toks), f) for f in facts if f not in have]
    # 이미 근거 줄이 있는 페이지는 그보다 **높은** 줄만(같은 페이지 이웃 행 과보강 방지 — 09-18 3차 회귀), 아직 근거 줄이 없는
    # 페이지는 다른 페이지 최고점과 **동점**이어도 허용 — 실측(09-21 골든 회귀 e34): "보건 선생님은 누구야?" 에 p98~100 의
    # 보건 활동 줄(2점)이 p27 의 "업무 보건, 담당자 조OO"(2점)를 동점으로 막아 이름 요구가 안 걸리고 이름 없는 답이 나갔다.
    floor = max(above, 0)
    scored = sorted((s for s in scored if (s[0] > floor if have else s[0] >= floor and s[0] > 0)), key=lambda s: -s[0])
    if not scored:
        return []
    # 최고점 줄(동점 포함)만 — 검색된 행이 없는 페이지(above=0)에서 2점짜리 이웃 행까지 들어오지 않게
    top = scored[0][0]
    return [f for sc, f in scored if sc == top][:room]


def compound_highlights(collection, chunks: list[dict], question: str, reranker) -> list[dict]:
    """최종 컨텍스트에 남은 페이지의 사실 청크를 부분 질문별로 재평가한다."""
    if (COMPOUND_HIGHLIGHT_MODE not in ("log", "on") or not HIGHLIGHT_FILL
            or HIGHLIGHT_MAX <= 0 or not chunks or reranker is None or not reranker.ready):
        return chunks
    parts = split_compound_question(question)
    if len(parts) != 2:
        return chunks
    started = time.perf_counter()
    facts = {}
    before = [(c["source_file"], c["page_number"], h) for c in chunks for h in c.get("highlights", [])]
    try:
        for src, page in dict.fromkeys((c["source_file"], c["page_number"]) for c in chunks):
            r = collection.get(where={"$and": [{"source_file": src}, {"page_number": page},
                                               {"kind": "fact"}]}, include=["documents", "metadatas"])
            for cid, doc in zip(r["ids"], r["documents"]):
                text = strip_title(doc or "").strip()
                if text and filter_survey_lines(question, [text]):
                    facts.setdefault(cid, {"chunk_id": cid, "source_file": src,
                                           "page_number": page, "text": text})
        candidates = list(facts.values())
        if not candidates or len(candidates) > COMPOUND_HIGHLIGHT_MAX_FACTS:
            logger.info("부분근거 생략 question=%r 후보=%d 한도=%d", question, len(candidates), COMPOUND_HIGHLIGHT_MAX_FACTS)
            return chunks
        # 두 질문을 한 배치로 평가한다. 원래 질문을 유지해 부분별 대상과 술어를 함께 전달한다.
        scores = reranker.score_pairs([(part, c["text"]) for part in parts for c in candidates])
        n = len(candidates)
        selected = []
        for i, part in enumerate(parts):
            ranked = sorted(({**c, "score": scores[i * n + j]} for j, c in enumerate(candidates)),
                            key=lambda c: -c["score"])
            top = ranked[0]
            # 같은 최고점에 다른 본문이 있으면 선택을 유보한다.
            ambiguous = any(c["score"] == top["score"] and c["text"] != top["text"] for c in ranked[1:])
            accepted = top["score"] >= RERANK_MIN_SCORE and not ambiguous
            logger.info("부분근거 평가 question=%r part=%r 통과=%s 임계=%s 상위=%s",
                        question, part, accepted, RERANK_MIN_SCORE, ranked[:3])
            if accepted:
                selected.append(top)
        # 두 항목 모두 근거가 있을 때만 대안을 만든다. 실패한 한쪽을 다른 항목 근거로 채우지 않는다.
        if len(selected) != len(parts):
            logger.info("부분근거 유지 question=%r 선택=%d/%d 추가시간=%.3fs",
                        question, len(selected), len(parts), time.perf_counter() - started)
            return chunks
        unique = {c["chunk_id"]: c for c in selected}
        priority = list(dict.fromkeys((c["source_file"], c["page_number"], c["text"]) for c in unique.values()))
        if len(priority) > HIGHLIGHT_MAX:
            return chunks
        proposed = list(dict.fromkeys(priority + before))[:HIGHLIGHT_MAX]
        logger.info("부분근거 비교 question=%r mode=%s 기존=%s 제안=%s 추가시간=%.3fs",
                    question, COMPOUND_HIGHLIGHT_MODE, before, proposed, time.perf_counter() - started)
        if COMPOUND_HIGHLIGHT_MODE != "on":
            return chunks
        return [{**c, "highlights": [h for src, page, h in proposed
                                     if (src, page) == (c["source_file"], c["page_number"])]} for c in chunks]
    except Exception:
        logger.exception("부분근거 평가 실패 question=%r — 기존 근거 유지 (추가시간=%.3fs)",
                         question, time.perf_counter() - started)
        return chunks


def expand_pages(collection, chunks: list[dict], question: str = "", reranker=None) -> list[dict]:
    """검색된 조각들을 페이지 단위로 병합해 돌려준다 (small-to-big).

    - 각 조각이 속한 (문서, 페이지)의 모든 조각을 가져와 페이지 전문으로 병합한다.
    - 같은 문서의 연속 페이지는 한 단위로 묶는다 — 반쪽만 검색돼도 나머지가 함께 들어간다.
    - 단위는 검색 순위(최소 거리)순으로 CONTEXT_CHAR_BUDGET까지 채운다. 첫 단위는
      예산을 넘어도 반드시 포함하고(단, 뒤쪽 페이지부터 덜어내 상한을 맞춘다),
      이후 단위는 예산 안에 들면 페이지 전문으로, 조각만이라도 들어가면 조각으로,
      둘 다 안 되면 제외한다.
    - PAGE_EXPAND=0이면 그대로 반환.

    반환 형태는 retriever.retrieve()의 조각 dict와 동일하다(text/source_file/
    page_number/chunk_index/distance) — 호출 측(main.py)이 구분 없이 다루게 하기 위함.
    """
    if not (PAGE_EXPAND and chunks):
        return chunks

    # 같은 페이지의 조각이 여럿 검색됐으면 가장 가까운 것 하나로 대표 (입력은 거리 오름차순)
    pages: dict[tuple, dict] = {}
    for c in chunks:
        pages.setdefault((c["source_file"], c["page_number"]), c)
    # 핵심 근거: 페이지별로 고른 행 조각(kind=fact)의 본문 — 관련도 순, 전체 HIGHLIGHT_MAX 개까지.
    # 페이지 전문은 표가 길어 7B 가 헤매므로, 검색이 실제로 맞춘 줄을 프롬프트 앞에 따로 보여준다 (prompt.build_prompt)
    highlights: dict[tuple, list[str]] = {}
    n_hl = 0
    for c in chunks:
        if c.get("kind") != "fact" or n_hl >= HIGHLIGHT_MAX:
            continue
        line = strip_title(c["text"]).strip()
        key = (c["source_file"], c["page_number"])
        if line and line not in highlights.setdefault(key, []):
            highlights[key].append(line)
            n_hl += 1
    # 보강: 고른 페이지의 fact 중 검색된 행들보다 질문 낱말이 확실히 더 겹치는 줄을 그 페이지 목록 맨 앞에 둔다.
    # 총량이 상한을 넘으면 뒤(관련도 낮은 검색 행)부터 덜어낸다 — 정답 행이 상한에 막혀 못 들어오는 일이 없게.
    if question and HIGHLIGHT_FILL:
        toks = question_tokens(question)
        have_all = [h for hs in highlights.values() for h in hs]
        above = max((question_overlap(question, h, toks) for h in have_all), default=0)
        for key in pages:
            extra = _extra_highlights(collection, key[0], key[1], question, highlights.get(key, []), HIGHLIGHT_MAX, above)
            if extra:
                highlights[key] = extra + highlights.get(key, [])
                n_hl += len(extra)
                logger.info("핵심 근거 보강 (p%s, 검색 행 최고 %d점 초과): %s", key[1], above, [e[:60] for e in extra])
        # 질문이 응답 대상(학부모·학생 등 문서의 [○○용] 표식)을 말하면 다른 대상의 설문 줄은 핵심 근거에서 뺀다
        for key in list(highlights):
            before = highlights[key]
            after = filter_survey_lines(question, before)
            if len(after) != len(before):
                logger.info("핵심 근거에서 다른 응답 대상의 설문 줄 %d개 제외 (p%s)", len(before) - len(after), key[1])
                n_hl -= len(before) - len(after)
                highlights[key] = after
        # 상한 초과분은 관련도 낮은 페이지의 마지막 줄부터 덜어내되, 질문 낱말이 **낱말 그대로** 든 줄은 마지막까지 남긴다.
        # 실측(09-21): "개교일은 언제야?" — 리랭커 낱말 구제로 붙은 연혁 행("… 개교 초대 …")이 맨 뒤 페이지라 먼저 잘려 나가고,
        # 달력 페이지 줄(개교기념일)만 남아 모델이 1980년을 지어내고 폴백도 기념일 줄을 읽었다.
        keep_toks = question_tokens(question)
        def _droppable(lines: list[str]) -> int:
            for i in range(len(lines) - 1, -1, -1):
                if whole_word_hits(lines[i], keep_toks) == 0:
                    return i
            return -1
        while n_hl > HIGHLIGHT_MAX:
            dropped = False
            for key in reversed(list(pages)):
                i = _droppable(highlights.get(key, []))
                if i >= 0:
                    highlights[key].pop(i)
                    n_hl -= 1
                    dropped = True
                    break
            if not dropped:
                for key in reversed(list(pages)):
                    if highlights.get(key):
                        highlights[key].pop()
                        n_hl -= 1
                        break

    # 문서별로 페이지 번호를 정렬해 연속 구간(단위)으로 묶는다
    by_source: dict[str, list] = {}
    for (src, page), best in pages.items():
        by_source.setdefault(src, []).append((page, best))
    units = []  # (대표 거리, 문서, [(페이지, 대표 조각), ...])
    for src, items in by_source.items():
        items.sort(key=lambda x: x[0])
        group = [items[0]]
        for page, best in items[1:]:
            if page == group[-1][0] + 1:
                group.append((page, best))
            else:
                units.append((min(_rank(b) for _, b in group), src, group))
                group = [(page, best)]
        units.append((min(_rank(b) for _, b in group), src, group))
    units.sort(key=lambda u: u[0])

    merged: list[dict] = []
    used = 0
    for _, src, group in units:
        entries = []
        for page, best in group:
            try:
                text = page_text(collection, src, page)
            except Exception as exc:
                logger.warning("페이지 병합 실패 (%s p%s) — 조각만 사용: %s", src, page, exc)
                text = best["text"]
            entries.append(
                {
                    "text": text or best["text"],
                    "source_file": src,
                    "page_number": page,
                    "chunk_index": best.get("chunk_index"),
                    "distance": best["distance"],
                    "highlights": highlights.get((src, page), []),
                }
            )
        unit_len = sum(len(e["text"]) for e in entries)
        best_of_unit = min((b for _, b in group), key=_rank)

        if not merged:
            # 첫 단위는 반드시 포함한다. 단, 긴 연속 구간(예: 4페이지 연속 검색)으로
            # 예산을 넘으면 뒤쪽 페이지부터 덜어내 상한 안에 맞춘다 — 표의 연속성을
            # 위해 순서 앞쪽을 우선 보존한다.
            # 덜어내는 순서: 질문 낱말이 **낱말 그대로** 든 핵심 근거가 없는 페이지를 뒤에서부터 먼저, 그런 페이지가 다 빠지면
            # 마지막 페이지. 실측(09-21 키오스크): "학생선도위원회 담당자는?" — p26~30 연속 구간(9천 자 초과)에서 정답 행이 든
            # p29 가 잘리고 근거 없는 p27 이 남아, 폴백이 p28 의 엉뚱한 부서 행을 읽어 줬다.
            keep_toks = question_tokens(question) if question else []
            def _no_hit(e: dict) -> bool:
                return not any(whole_word_hits(h, keep_toks) for h in e.get("highlights") or [])
            while len(entries) > 1 and sum(len(e["text"]) for e in entries) > CONTEXT_CHAR_BUDGET:
                idx = next((i for i in range(len(entries) - 1, -1, -1) if _no_hit(entries[i])), len(entries) - 1)
                dropped = entries.pop(idx)
                if idx != len(entries):
                    logger.info("컨텍스트 예산 초과 — 근거 없는 p%s 를 먼저 제외 (p%s 보존)", dropped["page_number"],
                                [e["page_number"] for e in entries if not _no_hit(e)])
            used += sum(len(e["text"]) for e in entries)
            merged.extend(entries)
        elif used + unit_len <= CONTEXT_CHAR_BUDGET:
            used += unit_len
            merged.extend(entries)
        elif used + len(best_of_unit["text"]) <= CONTEXT_CHAR_BUDGET:
            # 단위 전체는 예산 초과 — 대표 조각만이라도 근거로 유지
            used += len(best_of_unit["text"])
            merged.append(best_of_unit)
        # 둘 다 초과면 제외 — 상위 근거의 완전성이 하위 근거의 개수보다 중요하다
    return compound_highlights(collection, merged, question, reranker)
