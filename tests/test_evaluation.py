from rag.evaluation.evaluator import EvaluationCase, RAGEvaluator
from rag.llm.generator import GeneratedAnswer


class FakePipeline:
    def __init__(self, answer: GeneratedAnswer) -> None:
        self.answer = answer

    def ask(self, question: str, k: int = 4) -> GeneratedAnswer:
        return self.answer


def test_evaluator_checks_sources_facts_and_answer() -> None:
    pipeline = FakePipeline(
        GeneratedAnswer(
            answer="Authentication uses secure tokens.",
            sources=["security.md"],
            retrieved_chunks=1,
        )
    )
    case = EvaluationCase(
        question="How does authentication work?",
        expected_answer="secure tokens",
        expected_sources=("security.md",),
        required_facts=("authentication", "secure tokens"),
    )

    result = RAGEvaluator().evaluate_case(pipeline, case)

    assert result.passed is True
    assert result.source_recall == 1.0
    assert result.fact_coverage == 1.0
    assert result.citation_present is True


def test_evaluator_detects_wrong_source_and_missing_fact() -> None:
    pipeline = FakePipeline(
        GeneratedAnswer(
            answer="Authentication uses passwords.",
            sources=["unrelated.md"],
            retrieved_chunks=1,
        )
    )
    case = EvaluationCase(
        question="How does authentication work?",
        expected_sources=("security.md",),
        required_facts=("secure tokens",),
    )

    result = RAGEvaluator().evaluate_case(pipeline, case)

    assert result.passed is False
    assert result.source_recall == 0.0
    assert result.fact_coverage == 0.0


def test_evaluator_aggregates_golden_cases() -> None:
    pipeline = FakePipeline(
        GeneratedAnswer(
            answer="I do not have enough information in the documents to answer that question.",
            sources=[],
            retrieved_chunks=0,
        )
    )
    cases = [
        EvaluationCase(question="Known?", should_decline=True),
        EvaluationCase(question="Unknown?", should_decline=True),
    ]

    report = RAGEvaluator().evaluate(pipeline, cases)

    assert report["case_count"] == 2
    assert report["passed_count"] == 2
    assert report["pass_rate"] == 1.0
