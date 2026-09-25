# RAG Evaluation Dataset

Add one JSON object per line to a `.jsonl` file. Create each case from a real document in `data/docs`.

```json
{"question":"How long are access tokens valid?","expected_answer":"24 hours","expected_sources":["security.md"],"required_facts":["24 hours"]}
{"question":"What is the office location?","should_decline":true}
```

Fields:

- `question`: user question sent to the RAG pipeline
- `expected_answer`: optional answer phrase that must appear
- `expected_sources`: source filenames or paths that must be retrieved
- `required_facts`: important facts that must appear in the answer
- `should_decline`: set to `true` when the documents do not contain the answer

Keep this dataset versioned with the documents it evaluates. Do not use an answer from general knowledge; write expected facts only from the source document.

Run it from Python:

```python
from rag.evaluation.evaluator import RAGEvaluator

report = RAGEvaluator().evaluate_file(pipeline, "data/evaluation/questions.jsonl")
print(report["pass_rate"])
```
