from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json


class Status(str, Enum):
    SUPPORTED = "supported"
    INSUFFICIENT = "insufficient"
    CONFLICT = "conflict"
    INVALID = "invalid"


@dataclass(frozen=True)
class Part:
    id: str
    question: str


@dataclass(frozen=True)
class Evidence:
    id: str
    source_file: str
    page_number: int
    chunk_index: int
    text: str
    kind: str
    distance: float | None = None
    score: float = 0.0

    def source(self) -> dict:
        return {"source_file": self.source_file, "page_number": self.page_number,
                "chunk_index": self.chunk_index, "distance": self.distance}


def evidence_id(revision: str, source: str, page: int, chunk_id: str, text: str) -> str:
    # 순번만을 영구 ID로 취급하지 않는다. 동일 버전·원문·위치에서는 항상 같은 ID.
    raw = json.dumps([revision, source, page, chunk_id, text], ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class Claim:
    evidence_id: str
    quote: str
    value: str


@dataclass(frozen=True)
class ItemAnswer:
    part_id: str
    status: Status
    answer: str
    claims: tuple[Claim, ...] = ()


class ValidationCode(str, Enum):
    EVIDENCE_NOT_ALLOWED = "evidence_not_allowed"
    QUOTE_NOT_IN_SOURCE = "quote_not_in_source"
    VALUE_NOT_IN_QUOTE = "value_not_in_quote"
    VALUE_QUANTITY_MISSING_IN_ANSWER = "value_quantity_missing_in_answer"
    VALUE_TEXT_MISSING_IN_ANSWER = "value_text_missing_in_answer"
    ANSWER_QUANTITY_NOT_IN_QUOTE = "answer_quantity_not_in_quote"
    PARSE_FAILURE = "parse_failure"
    GENERATION_OUTPUT_ERROR = "generation_output_error"


@dataclass(frozen=True)
class ValidationIssue:
    code: ValidationCode
    message: str
    evidence_id: str | None = None
    claim_index: int | None = None


@dataclass
class CheckedItem:
    item: ItemAnswer
    errors: list[str] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)


@dataclass
class Outcome:
    answer: str
    sources: list[dict]
    grounded: bool
    unverified: list[str]
    items: list[dict] = field(default_factory=list)
    calls: int = 0

    def response(self) -> dict:
        return {"answer": self.answer, "sources": self.sources,
                "grounded": self.grounded, "unverified": self.unverified}
