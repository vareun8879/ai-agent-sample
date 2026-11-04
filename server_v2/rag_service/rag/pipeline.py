import asyncio
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import time
import uuid
from .config import Settings
from .contracts import CheckedItem, ItemAnswer, Status, ValidationCode
from .evidence import arrange
from .generation import Generator
from .parsing import ParseError, parse
from .query import plan
from .rendering import render
from .source_spans import collect as collect_source_spans
from .validation import check
from .value_candidates import collect as collect_values, audit as audit_values

logger = logging.getLogger("rag.v2")


class Pipeline:
    def __init__(self, settings: Settings, search, generator: Generator):
        self.s, self.search, self.generator = settings, search, generator

    async def answer(self, question, institution=None, history=None, mode="voice", top_k=None):
        trace = {"request_id": uuid.uuid4().hex, "question": question,
                 "institution_id": institution, "revision": self.s.revision,
                 "validation_mode": self.s.validation_mode, "attempts": []}
        started = time.monotonic()
        parts = plan(question)
        trace["parts"] = [asdict(p) for p in parts]
        if hasattr(self.search, "retrieve_with_trace"):
            candidates, stages = await asyncio.to_thread(
                self.search.retrieve_with_trace, parts, institution, top_k)
            trace["retrieval_stages"] = stages
        else:
            candidates = await asyncio.to_thread(self.search.retrieve, parts, institution, top_k)
        evidence, allowed = arrange(parts, candidates, self.s.max_evidence_chars)
        trace["evidence"] = {k: asdict(e) for k, e in evidence.items()}
        trace["allowed"] = allowed
        trace["retrieval_seconds"] = time.monotonic() - started
        best = {}
        feedback = None
        calls = 0
        if evidence:
            for attempt in range(self.s.max_calls):
                calls += 1
                call_started = time.monotonic()
                raw = ""
                try:
                    raw = await self.generator.generate(question, parts, evidence, allowed,
                                                        history or [], mode, feedback)
                    items = parse(raw, parts)
                    checked = check(items, evidence, allowed)
                    # 두 번째 실패가 첫 번째에 확인된 항목을 지우지 않도록 항목 단위로 유지.
                    for c in checked:
                        old = best.get(c.item.part_id)
                        if old is None or (c.item.status == Status.SUPPORTED and not c.errors):
                            best[c.item.part_id] = c
                    problems = [f"{c.item.part_id}: {e}" for c in checked for e in c.errors]
                    trace["attempts"].append({"raw": raw, "errors": problems,
                                              "issues": [{"part_id": c.item.part_id, **asdict(issue)}
                                                         for c in checked for issue in c.issues],
                                              "seconds": time.monotonic() - call_started})
                    if not problems:
                        break
                    feedback = {"errors": problems, "previous_output": raw,
                                "instruction": ("모든 항목을 다시 검토해 JSON 전체를 작성하라. "
                                                "각 quote는 해당 evidence.text의 연속된 원문을 생략·변형 없이 복사하고, "
                                                "value는 quote 안의 원문 부분을 그대로 복사하라. "
                                                "답변의 숫자와 단위가 인용에 실제 있는지 확인하라.")}
                except (ParseError, ValueError) as exc:
                    trace["attempts"].append({"raw": raw, "error": str(exc),
                                              "issues": [{"code": (ValidationCode.PARSE_FAILURE if
                                                  isinstance(exc, ParseError) else ValidationCode.GENERATION_OUTPUT_ERROR),
                                                  "message": str(exc)}],
                                              "seconds": time.monotonic() - call_started})
                    feedback = {"errors": [str(exc)], "previous_output": raw,
                                "instruction": "모든 항목을 포함한 JSON 객체 하나로 작성하라."}
                except Exception:
                    # 네트워크/서빙 장애를 '문서에 답 없음'으로 바꾸지 않는다.
                    trace["error"] = "generation transport failure"
                    self.record(trace)
                    raise
        else:
            best = {p.id: CheckedItem(ItemAnswer(p.id, Status.INSUFFICIENT, "")) for p in parts}
        outcome = render(parts, list(best.values()), evidence, self.s.validation_mode)
        if self.s.value_candidates_mode == "log":
            # 관측 실패가 답변·검증 판정에 영향을 주지 않게 한다.
            try:
                candidates = collect_values(evidence)
                trace["value_candidates"] = candidates
                trace["value_candidate_audit"] = audit_values(
                    [c.item for c in best.values()], candidates, allowed)
            except Exception:
                logger.exception("Value candidate observation failed")
                trace["value_candidate_error"] = "observation failed"
        if self.s.source_span_mode == "log":
            try:
                trace["source_spans"] = {}
                for part in parts:
                    rows = collect_source_spans(part.question, evidence, allowed.get(part.id, []))
                    trace["source_spans"][part.id] = {"total": len(rows), "rows": rows[:50]}
            except Exception:
                logger.exception("Source span observation failed")
                trace["source_span_error"] = "observation failed"
        outcome.calls = calls
        trace.update({"seconds": time.monotonic() - started, "calls": calls,
                      "result": outcome.response(), "items": outcome.items})
        self.record(trace)
        return outcome

    @staticmethod
    def record(trace):
        # 후보별 진단은 JSONL 원본에 남긴다. journal의 단일 행 크기 제한을 피한다.
        journal_trace = {key: value for key, value in trace.items()
                         if key not in ("retrieval_stages", "source_spans")}
        logger.info("trace=%s", json.dumps(journal_trace, ensure_ascii=False))
        path = Path(os.getenv("V2_TRACE_PATH", "logs/trace.jsonl"))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(trace, ensure_ascii=False) + "\n")
        except OSError:
            logger.exception("Trace file write failed")
