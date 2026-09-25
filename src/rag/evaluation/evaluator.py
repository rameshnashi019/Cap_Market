from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from rag.llm.generator import GeneratedAnswer


class AnsweringPipeline(Protocol):
    def ask(self, question: str, k: int = 4) -> GeneratedAnswer:
        ...


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    expected_answer: str | None = None
    expected_sources: tuple[str, ...] = ()
    required_facts: tuple[str, ...] = ()
    should_decline: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationCase":
        question = str(data.get("question", "")).strip()
        if not question:
            raise ValueError("evaluation case question must not be empty")
        sources = tuple(str(source) for source in data.get("expected_sources", []))
        facts = tuple(str(fact) for fact in data.get("required_facts", []))
        return cls(
            question=question,
            expected_answer=data.get("expected_answer"),
            expected_sources=sources,
            required_facts=facts,
            should_decline=bool(data.get("should_decline", False)),
        )


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    """Load one golden Q&A case per line from a JSONL file."""
    cases: list[EvaluationCase] = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid evaluation JSON on line {line_number}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"evaluation line {line_number} must contain an object")
            cases.append(EvaluationCase.from_dict(data))
    if not cases:
        raise ValueError("evaluation file must contain at least one case")
    return cases


@dataclass
class CaseEvaluation:
    question: str
    answer: str
    source_recall: float
    fact_coverage: float
    answer_match: bool
    citation_present: bool
    correctly_declined: bool
    passed: bool
    sources: list[str] = field(default_factory=list)


class RAGEvaluator:
    """Evaluate retrieval and grounded-answer behavior against golden cases."""

    def evaluate_case(self, pipeline: AnsweringPipeline, case: EvaluationCase) -> CaseEvaluation:
        result = pipeline.ask(case.question)
        answer_lower = result.answer.lower()
        expected_sources = {source.lower() for source in case.expected_sources}
        actual_sources = {source.lower() for source in result.sources}
        source_recall = (
            len(expected_sources & actual_sources) / len(expected_sources)
            if expected_sources
            else 1.0
        )
        fact_coverage = (
            sum(fact.lower() in answer_lower for fact in case.required_facts)
            / len(case.required_facts)
            if case.required_facts
            else 1.0
        )
        answer_match = (
            case.expected_answer is None
            or case.expected_answer.lower() in answer_lower
        )
        citation_present = not case.expected_sources or bool(result.sources)
        correctly_declined = (
            not case.should_decline
            or "not have enough information" in answer_lower
        )
        passed = (
            source_recall == 1.0
            and fact_coverage == 1.0
            and answer_match
            and citation_present
            and correctly_declined
        )
        return CaseEvaluation(
            question=case.question,
            answer=result.answer,
            source_recall=source_recall,
            fact_coverage=fact_coverage,
            answer_match=answer_match,
            citation_present=citation_present,
            correctly_declined=correctly_declined,
            passed=passed,
            sources=result.sources,
        )

    def evaluate(
        self,
        pipeline: AnsweringPipeline,
        cases: Sequence[EvaluationCase],
    ) -> dict[str, Any]:
        if not cases:
            raise ValueError("at least one evaluation case is required")
        results = [self.evaluate_case(pipeline, case) for case in cases]
        return {
            "case_count": len(results),
            "passed_count": sum(result.passed for result in results),
            "pass_rate": sum(result.passed for result in results) / len(results),
            "average_source_recall": sum(result.source_recall for result in results) / len(results),
            "average_fact_coverage": sum(result.fact_coverage for result in results) / len(results),
            "results": results,
        }

    def evaluate_file(
        self,
        pipeline: AnsweringPipeline,
        path: str | Path,
    ) -> dict[str, Any]:
        """Load and evaluate a JSONL golden question-answer dataset."""
        return self.evaluate(pipeline, load_evaluation_cases(path))
