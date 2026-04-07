# evaluate.py

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from eval_dataset import datasets
from evaluator import collect_rag_output
# your test questions
test_data = [
    {
        "question":     "What is backpropagation?",
        "ground_truth": "Backpropagation computes gradients using chain rule..."
    },
    # more questions...
]

# collect RAG outputs for each question
eval_data = {
    "question":   [],
    "answer":     [],
    "contexts":   [],
    "ground_truth": []
}

for item in datasets:
    output = collect_rag_output(item["question"])

    eval_data["question"].append(item["question"])
    eval_data["answer"].append(output["answer"])
    eval_data["contexts"].append(output["contexts"])
    eval_data["ground_truth"].append(item["ground_truth"])

# convert to HuggingFace dataset format (RAGAS needs this)
dataset = datasets.from_dict(eval_data)

# run evaluation
results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)

print(results)