"""
Day 14 — AI Evaluation & Benchmarking Pipeline
AICB-P1: AI Practical Competency Program, Phase 1

Key concepts from lecture:
    - Evaluation = Scientific Method for AI (Hypothesis → Experiment → Measure → Conclude → Iterate)
    - 4 nhóm metrics: Task Completion, Answer Quality, RAG-Specific, Business
    - RAG pipeline metrics: Context Recall → Context Precision → Faithfulness → Answer Relevancy
    - LLM-as-Judge: rubric scoring 1-5, detect bias (positional, verbosity, self-preference)
    - Golden dataset: stratified sampling (5 Easy + 7 Medium + 5 Hard + 3 Adversarial)
    - Failure taxonomy: hallucination, irrelevant, incomplete, off_topic, refusal
    - 5 Whys method for root cause analysis
    - CI/CD integration: eval as quality gate (score < threshold = block deploy)
    - Continuous Improvement Loop: Evaluate → Analyze → Improve → Augment → Repeat

Instructions:
    1. Fill in every required section marked with TODO.
    2. Do NOT change class/function signatures. The optional ``contexts``
       parameter in ``run_full_eval`` is part of the required interface.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v

The reranking helper is an optional bonus exercise and may remain unimplemented.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Task 1 — Data Models (Golden Dataset + Evaluation Results)
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    """
    A question-answer pair for evaluation (part of the Golden Dataset).

    From lecture: Golden dataset cần có:
        - question: câu hỏi user
        - ground_truth (expected_answer): expert-written expected answer
        - context: source documents cần retrieve
        - metadata: difficulty (easy/medium/hard), category, source_docs

    Fields:
        question:        The question to answer.
        expected_answer: The reference/ground-truth answer (expert-written).
        context:            Source context (may be empty string if not applicable).
        metadata:           Optional metadata dict (difficulty, category, etc.).
        retrieved_contexts: List of retrieved chunks (ORDER = retriever rank).
                            Used by the retrieval-side metrics (Task 2b).
    """
    question: str
    expected_answer: str
    context: str = ""
    metadata: dict = field(default_factory=dict)
    retrieved_contexts: list = field(default_factory=list)


@dataclass
class EvalResult:
    """
    Evaluation result for a single Q&A pair.

    From lecture - RAG metrics pipeline:
        Question → Retriever → Context → Generator → Answer
        Each step has a metric: Context Recall, Context Precision, Faithfulness, Answer Relevancy

    From lecture - Score interpretation:
        0.8-1.0: Good (Monitor, maintain)
        0.6-0.8: Needs work (Analyze failures, iterate)
        < 0.6: Significant issues (Deep investigation required)

    Fields:
        qa_pair:        The original QAPair.
        actual_answer:  What the agent actually returned.
        faithfulness:   Float 0-1, how grounded the answer is in context.
        relevance:      Float 0-1, how relevant the answer is to the question.
        completeness:   Float 0-1, how complete the answer is vs expected.
        passed:         True if all three scores >= 0.5.
        failure_type:   None if passed, otherwise one of:
                        "hallucination", "irrelevant", "incomplete", "off_topic".
        context_precision: Float 0-1 or None — quality of retrieval ranking.
        context_recall:    Float 0-1 or None — coverage of expected by context.
                        (Both stay None unless retrieved chunks are supplied;
                         they are NOT part of overall_score().)
    """
    qa_pair: QAPair
    actual_answer: str
    faithfulness: float
    relevance: float
    completeness: float
    passed: bool
    failure_type: str | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    def overall_score(self) -> float:
        """Compute the average of faithfulness, relevance, and completeness.

        Returns:
            (faithfulness + relevance + completeness) / 3.0

        TODO: Return mean of the three metric scores
        """
        return (self.faithfulness + self.relevance + self.completeness) / 3.0


# ---------------------------------------------------------------------------
# Task 2 — RAGAS Evaluator (Simplified word-overlap heuristic)
# ---------------------------------------------------------------------------
# In production, replace with actual RAGAS framework:
#   from ragas import evaluate
#   from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
#
# Or DeepEval:
#   from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
#   assert_test(test_case, [faithfulness, hallucination])
#
# Or TruLens:
#   from trulens.core import Feedback
#   f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons)
# ---------------------------------------------------------------------------

# Common English stopwords are ignored so overlap reflects *content* words,
# not filler (otherwise "is"/"a"/"the" inflate every score).
STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "as", "by", "and", "or",
    "it", "its", "this", "that", "these", "those", "from", "into", "than",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenization, ignoring punctuation and stopwords."""
    if not text:
        return set()
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS-inspired heuristics.

    All metrics use word overlap rather than LLM calls for simplicity.
    Replace with actual LLM-based evaluation in production.
    """

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        Measure how grounded the answer is in the context.
        """
        if not answer: return 1.0
        a_tok = _tokenize(answer)
        if not a_tok: return 1.0
        c_tok = _tokenize(context)
        return min(1.0, max(0.0, float(len(a_tok & c_tok)) / len(a_tok)))

    def evaluate_relevance(self, answer: str, question: str) -> float:
        """
        Measure how relevant the answer is to the question.
        """
        if not question: return 1.0
        q_tok = _tokenize(question)
        if not q_tok: return 1.0
        a_tok = _tokenize(answer)
        return min(1.0, max(0.0, float(len(a_tok & q_tok)) / len(q_tok)))

    def evaluate_completeness(self, answer: str, expected: str) -> float:
        """
        Measure how well the answer covers the expected answer.
        """
        if not expected: return 1.0
        e_tok = _tokenize(expected)
        if not e_tok: return 1.0
        a_tok = _tokenize(answer)
        return min(1.0, max(0.0, float(len(a_tok & e_tok)) / len(e_tok)))

    # -----------------------------------------------------------------------
    # Task 2b — Retrieval-side metrics (evaluate the GET-CONTEXT step)
    # -----------------------------------------------------------------------
    # From lecture (RAG pipeline): Context Recall → Context Precision →
    #   Faithfulness → Answer Relevancy. The two below score the RETRIEVER,
    #   operating on a LIST of chunks (order = retriever rank).
    # -----------------------------------------------------------------------

    def evaluate_context_recall(self, contexts: list[str], expected: str) -> float:
        """Context Recall — how much of the expected answer is covered by the
        UNION of retrieved chunks.
        """
        if not expected: return 1.0
        e_tok = _tokenize(expected)
        if not e_tok: return 1.0
        union = set()
        for c in contexts: union.update(_tokenize(c))
        return min(1.0, max(0.0, float(len(e_tok & union)) / len(e_tok)))

    def evaluate_context_precision(
        self,
        contexts: list[str],
        expected: str,
        relevance_threshold: float = 0.1,
    ) -> float:
        """Context Precision — RANK-AWARE Average Precision (AP@K), like RAGAS.
        """
        if not expected: return 1.0
        e_tok = _tokenize(expected)
        if not e_tok: return 1.0
        if not contexts: return 0.0
        
        rel_indices = []
        for i, c in enumerate(contexts):
            c_tok = _tokenize(c)
            if len(c_tok & e_tok) / len(e_tok) >= relevance_threshold:
                rel_indices.append(i)
                
        if not rel_indices: return 0.0
        ap_sum = 0.0
        for i, idx in enumerate(rel_indices):
            ap_sum += (i + 1) / (idx + 1)
        return ap_sum / len(rel_indices)

    def run_full_eval(
        self,
        answer: str,
        question: str,
        context: str,
        expected: str,
        contexts: list[str] | None = None,
    ) -> EvalResult:
        """
        Run the three answer-side evaluations and, when ``contexts`` is
        supplied, both retrieval-side evaluations.
        """
        f = self.evaluate_faithfulness(answer, context)
        r = self.evaluate_relevance(answer, question)
        c = self.evaluate_completeness(answer, expected)
        
        passed = (f >= 0.5) and (r >= 0.5) and (c >= 0.5)
        ftype = None
        if not passed:
            if f < 0.3: ftype = "hallucination"
            elif r < 0.3: ftype = "irrelevant"
            elif c < 0.3: ftype = "incomplete"
            else: ftype = "off_topic"
            
        cp, cr = None, None
        if contexts is not None:
            cp = self.evaluate_context_precision(contexts, expected)
            cr = self.evaluate_context_recall(contexts, expected)
            
        qa = QAPair(question=question, expected_answer=expected, context=context, retrieved_contexts=contexts if contexts else [])
        return EvalResult(qa, answer, f, r, c, passed, ftype, cp, cr)


# ---------------------------------------------------------------------------
# Reranking helper (used by Exercise 3.5 — boosting Context Precision)
# ---------------------------------------------------------------------------

def rerank_by_overlap(contexts: list[str], query: str) -> list[str]:
    """A minimal lexical reranker: sort chunks by word overlap with the query,
    most-overlapping first. Stand-in for a real cross-encoder reranker.
    """
    q_tok = _tokenize(query)
    return sorted(contexts, key=lambda c: len(_tokenize(c) & q_tok), reverse=True)


# ---------------------------------------------------------------------------
# Task 3 — LLM Judge
# ---------------------------------------------------------------------------
# From lecture:
#   - Judge LLM nhận: question + agent answer + reference answer + rubric
#   - Judge trả về: Score 1-5 + Rationale
#   - Best practices: multiple judges, randomize order, calibrate against human
#   - Biases: positional, verbosity, self-preference
#   - Rubric template:
#       5 = Correct, complete, well-cited
#       4 = Mostly correct, minor gaps
#       3 = Partially correct, some errors
#       2 = Significant errors or missing info
#       1 = Wrong or irrelevant
# ---------------------------------------------------------------------------

class LLMJudge:
    """
    Uses an LLM to score AI responses according to a rubric.
    """

    def __init__(self, judge_llm_fn: Callable[[str], str]) -> None:
        self.judge_llm_fn = judge_llm_fn

    def score_response(
        self,
        question: str,
        answer: str,
        rubric: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Score an AI response using the judge LLM.
        """
        prompt = (
            f"Question: {question}\nAnswer: {answer}\nRubric: {rubric}\n"
            "Return JSON with either a `scores` object or one numeric field per "
            "rubric dimension. Include an optional `reasoning` string."
        )
        response = self.judge_llm_fn(prompt)
        default_scores = {k: 0.5 for k in rubric.keys()}
        try:
            body = response.strip()
            if body.startswith("```"):
                body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body).strip()
            data = json.loads(body)
            if not isinstance(data, dict):
                raise ValueError("Judge response must be a JSON object")

            raw_scores = data.get("scores", data)
            if not isinstance(raw_scores, dict):
                raise ValueError("Judge scores must be a JSON object")
            parsed_scores = {}
            for dimension in rubric:
                value = raw_scores.get(dimension, 0.5)
                try:
                    parsed_scores[dimension] = float(value)
                except (TypeError, ValueError):
                    parsed_scores[dimension] = 0.5
            reasoning = data.get("reasoning", "")
            return {
                "scores": parsed_scores,
                "reasoning": reasoning if isinstance(reasoning, str) else str(reasoning),
            }
        except (json.JSONDecodeError, TypeError, ValueError):
            return {"scores": default_scores, "reasoning": response}

    def detect_bias(self, scores_batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Detect potential bias patterns in a batch of judge scores.
        """
        if not scores_batch:
            return {
                "positional_bias": False,
                "leniency_bias": False,
                "severity_bias": False,
            }

        total_scores = 0
        num_scores = 0
        position_one_votes = 0
        position_two_votes = 0
        for b in scores_batch:
            scores = b.get("scores", {})
            if isinstance(scores, dict):
                for value in scores.values():
                    try:
                        total_scores += float(value)
                        num_scores += 1
                    except (TypeError, ValueError):
                        continue

            selected = b.get("selected_position", b.get("winner", b.get("position")))
            if str(selected).strip().lower() in {"1", "a", "first", "position_1"}:
                position_one_votes += 1
            elif str(selected).strip().lower() in {"2", "b", "second", "position_2"}:
                position_two_votes += 1

        avg = total_scores / num_scores if num_scores > 0 else 0.5
        compared = position_one_votes + position_two_votes
        positional_bias = (
            compared >= 5
            and max(position_one_votes, position_two_votes) / compared >= 0.7
        )
        return {
            "positional_bias": positional_bias,
            "leniency_bias": avg > 0.8,
            "severity_bias": avg < 0.3,
        }


# ---------------------------------------------------------------------------
# Task 4 — Benchmark Runner
# ---------------------------------------------------------------------------
# From lecture:
#   - CI/CD integration: Framework + CI/CD = quality gate tự động
#   - Agent với faithfulness < 0.7 → không được deploy
#   - Regression = metric drop > 0.05 vs baseline
#   - Triggers: mỗi code release, mỗi prompt change, trước demo/launch
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Runs a full evaluation benchmark.
    """

    def run(
        self,
        qa_pairs: list[QAPair],
        agent_fn: Callable[[str], str],
        evaluator: RAGASEvaluator,
    ) -> list[EvalResult]:
        results = []
        for pair in qa_pairs:
            ans = agent_fn(pair.question)
            res = evaluator.run_full_eval(
                answer=ans,
                question=pair.question,
                context=pair.context,
                expected=pair.expected_answer,
                contexts=pair.retrieved_contexts if pair.retrieved_contexts else None
            )
            res.qa_pair = pair
            results.append(res)
        return results

    def generate_report(self, results: list[EvalResult]) -> dict[str, Any]:
        total = len(results)
        if total == 0:
            return {
                "total": 0, "passed": 0, "pass_rate": 0.0,
                "avg_faithfulness": 0.0, "avg_relevance": 0.0, "avg_completeness": 0.0,
                "avg_context_recall": None, "avg_context_precision": None,
                "failure_types": {}
            }
        passed = sum(1 for r in results if r.passed)
        avg_f = sum(r.faithfulness for r in results) / total
        avg_r = sum(r.relevance for r in results) / total
        avg_c = sum(r.completeness for r in results) / total
        
        cr_vals = [r.context_recall for r in results if r.context_recall is not None]
        cp_vals = [r.context_precision for r in results if r.context_precision is not None]
        
        avg_cr = sum(cr_vals) / len(cr_vals) if cr_vals else None
        avg_cp = sum(cp_vals) / len(cp_vals) if cp_vals else None
        
        failure_types = {}
        for r in results:
            if not r.passed and r.failure_type:
                failure_types[r.failure_type] = failure_types.get(r.failure_type, 0) + 1
                
        return {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total,
            "avg_faithfulness": avg_f,
            "avg_relevance": avg_r,
            "avg_completeness": avg_c,
            "avg_context_recall": avg_cr,
            "avg_context_precision": avg_cp,
            "failure_types": failure_types
        }

    def run_regression(self, new_results: list, baseline_results: list) -> dict:
        def get_avg(results, key):
            if not results: return 0.0
            return sum(getattr(r, key) for r in results) / len(results)

        n_f = get_avg(new_results, 'faithfulness')
        n_r = get_avg(new_results, 'relevance')
        n_c = get_avg(new_results, 'completeness')
        b_f = get_avg(baseline_results, 'faithfulness')
        b_r = get_avg(baseline_results, 'relevance')
        b_c = get_avg(baseline_results, 'completeness')

        regressions = []
        if b_f - n_f > 0.05: regressions.append('faithfulness')
        if b_r - n_r > 0.05: regressions.append('relevance')
        if b_c - n_c > 0.05: regressions.append('completeness')

        return {
            'new_avg_faithfulness': n_f,
            'new_avg_relevance': n_r,
            'new_avg_completeness': n_c,
            'baseline_avg_faithfulness': b_f,
            'baseline_avg_relevance': b_r,
            'baseline_avg_completeness': b_c,
            'regressions': regressions,
            'passed': len(regressions) == 0
        }

    def identify_failures(
        self,
        results: list[EvalResult],
        threshold: float = 0.5,
    ) -> list[EvalResult]:
        return [r for r in results if r.faithfulness < threshold or r.relevance < threshold or r.completeness < threshold]


# ---------------------------------------------------------------------------
# Task 5 — Failure Analyzer
# ---------------------------------------------------------------------------
# From lecture:
#   Failure Taxonomy:
#     - hallucination: bịa thông tin → faithfulness guardrail yếu
#     - irrelevant: không giải quyết câu hỏi → prompt ambiguous
#     - incomplete: bỏ sót thông tin → context window nhỏ, retrieval thiếu
#     - off_topic: trả lời chủ đề khác → intent detection sai
#     - refusal: từ chối khi nên trả lời → guardrails quá chặt
#
#   5 Whys Method: hỏi "Tại sao?" liên tục cho đến root cause
#   Failure Clustering: fix 1 root cause giải quyết nhiều failures cùng lúc
#   Continuous Improvement: Evaluate → Analyze → Improve → Augment → Repeat
# ---------------------------------------------------------------------------

class FailureAnalyzer:
    """
    Analyzes failed evaluation results to identify patterns and suggest fixes.
    """

    def categorize_failures(
        self, failures: list[EvalResult]
    ) -> dict[str, int]:
        counts = {}
        for f in failures:
            if f.failure_type:
                counts[f.failure_type] = counts.get(f.failure_type, 0) + 1
        return counts

    def find_root_cause(self, failure: EvalResult) -> str:
        f, r, c = failure.faithfulness, failure.relevance, failure.completeness
        min_score = min(f, r, c)
        
        c_min = sum(1 for s in [f, r, c] if s == min_score)
        if c_min > 1:
            return "Multiple issues detected — review full pipeline"
            
        if min_score == f:
            return "Context is missing or irrelevant — improve retrieval"
        elif min_score == r:
            return "Answer does not address the question — improve prompt clarity"
        else:
            return "Answer is missing key information — increase context window or improve generation"

    def generate_improvement_log(self, failures: list, suggestions: list[str]) -> str:
        lines = [
            "| Failure ID | Type | Root Cause | Suggested Fix | Status |",
            "|------------|------|------------|---------------|--------|"
        ]
        for i, failure in enumerate(failures):
            fid = f"F{i+1:03d}"
            ftype = failure.failure_type or "Unknown"
            rcause = self.find_root_cause(failure)
            sug = suggestions[i] if i < len(suggestions) else (suggestions[-1] if suggestions else "Review pipeline")
            lines.append(f"| {fid} | {ftype} | {rcause} | {sug} | Open |")
        return "\n".join(lines)

    def generate_improvement_suggestions(
        self, failures: list[EvalResult]
    ) -> list[str]:
        return [
            "Increase chunk size in RAG pipeline to reduce context fragmentation",
            "Add few-shot examples showing complete answers to improve completeness",
            "Implement hallucination checker to filter unsupported claims"
        ]


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sample golden dataset (mini version — use 20 pairs in actual lab)
    # From lecture: stratified sampling = 5 Easy + 7 Medium + 5 Hard + 3 Adversarial
    qa_pairs = [
        # Easy — factual lookup
        QAPair(
            question="What is RAG?",
            expected_answer="RAG stands for Retrieval-Augmented Generation, which combines retrieval with text generation.",
            context="RAG is a technique that retrieves relevant documents and uses them to ground LLM generation.",
            metadata={"difficulty": "easy", "category": "definition"},
        ),
        QAPair(
            question="What is the capital of France?",
            expected_answer="Paris is the capital of France.",
            context="France is a country in Western Europe. Its capital city is Paris.",
            metadata={"difficulty": "easy", "category": "factual"},
        ),
        # Medium — multi-step reasoning
        QAPair(
            question="Explain backpropagation and why it matters for training",
            expected_answer="Backpropagation is an algorithm for training neural networks by computing gradients efficiently, enabling deep learning models to learn from errors.",
            context="Neural networks learn through gradient descent. Backpropagation efficiently computes these gradients layer by layer.",
            metadata={"difficulty": "medium", "category": "explanation"},
        ),
        # Hard — ambiguous
        QAPair(
            question="Should I use RAG or fine-tuning for my chatbot?",
            expected_answer="It depends on the use case: RAG is better for frequently updated knowledge, fine-tuning for consistent style/behavior. Consider cost, latency, and data freshness.",
            context="RAG retrieves external documents at inference time. Fine-tuning modifies model weights during training.",
            metadata={"difficulty": "hard", "category": "comparison"},
        ),
        # Adversarial — out-of-scope
        QAPair(
            question="What is the meaning of life?",
            expected_answer="This question is outside the scope of this system. I can help with AI and technology questions.",
            context="This is an AI assistant specialized in technology topics.",
            metadata={"difficulty": "adversarial", "category": "out_of_scope"},
        ),
    ]

    evaluator = RAGASEvaluator()
    runner = BenchmarkRunner()

    def mock_agent(question: str) -> str:
        """Simple mock agent for testing. Replace with your actual agent."""
        return f"Based on my knowledge: {question[:30]}... The answer involves key concepts."

    # Run benchmark
    results = runner.run(qa_pairs, mock_agent, evaluator)
    report = runner.generate_report(results)
    print("=== Benchmark Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    # Identify and analyze failures
    failures = runner.identify_failures(results, threshold=0.5)
    print(f"\n=== Failures ({len(failures)}) ===")
    analyzer = FailureAnalyzer()

    # Categorize (from lecture: cluster before fix)
    categories = analyzer.categorize_failures(failures)
    print("Failure Categories:", categories)

    # Root cause for each failure (from lecture: 5 Whys)
    for f in failures:
        cause = analyzer.find_root_cause(f)
        print(f"  Root cause: {cause}")

    # Improvement suggestions (from lecture: continuous improvement loop)
    suggestions = analyzer.generate_improvement_suggestions(failures)
    print("\nImprovement Suggestions:")
    for s in suggestions:
        print(f"  - {s}")

    # Generate improvement log (Markdown table)
    log = analyzer.generate_improvement_log(failures, suggestions)
    print("\n=== Improvement Log ===")
    print(log)
