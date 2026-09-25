from pathlib import Path

import pytest

from rag.evaluation.evaluator import EvaluationCase, load_evaluation_cases


def test_load_evaluation_cases_from_jsonl(tmp_path: Path) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(
        '# comments are allowed\n'
        '{"question":"Where is the office?","expected_sources":["company.md"],"required_facts":["London"]}\n'
        '{"question":"Unknown?","should_decline":true}\n',
        encoding="utf-8",
    )

    cases = load_evaluation_cases(dataset)

    assert cases == [
        EvaluationCase(
            question="Where is the office?",
            expected_sources=("company.md",),
            required_facts=("London",),
        ),
        EvaluationCase(question="Unknown?", should_decline=True),
    ]


def test_load_evaluation_cases_rejects_invalid_json(tmp_path: Path) -> None:
    dataset = tmp_path / "invalid.jsonl"
    dataset.write_text("not json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        load_evaluation_cases(dataset)
