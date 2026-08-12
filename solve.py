import re

with open('template.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace QAPair
qapair_repl = r"""@dataclass
class QAPair:
    question: str
    expected_answer: str
    context: str = ""
    metadata: dict = field(default_factory=dict)
    retrieved_contexts: list = field(default_factory=list)"""
content = re.sub(r'@dataclass\nclass QAPair:.*?(?=\n\n@dataclass)', qapair_repl, content, flags=re.DOTALL)

# Replace EvalResult
evalresult_repl = r"""@dataclass
class EvalResult:
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
        return (self.faithfulness + self.relevance + self.completeness) / 3.0"""
content = re.sub(r'@dataclass\nclass EvalResult:.*?(?=\n\n# ---)', evalresult_repl, content, flags=re.DOTALL)

# RAGASEvaluator replacements
ragas_repl = r"""    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        if not answer: return 1.0
        a_tok = _tokenize(answer)
        if not a_tok: return 1.0
        c_tok = _tokenize(context)
        return min(1.0, max(0.0, float(len(a_tok & c_tok)) / len(a_tok)))

    def evaluate_relevance(self, answer: str, question: str) -> float:
        if not question: return 1.0
        q_tok = _tokenize(question)
        if not q_tok: return 1.0
        a_tok = _tokenize(answer)
        return min(1.0, max(0.0, float(len(a_tok & q_tok)) / len(q_tok)))

    def evaluate_completeness(self, answer: str, expected: str) -> float:
        if not expected: return 1.0
        e_tok = _tokenize(expected)
        if not e_tok: return 1.0
        a_tok = _tokenize(answer)
        return min(1.0, max(0.0, float(len(a_tok & e_tok)) / len(e_tok)))

    def evaluate_context_recall(self, contexts: list[str], expected: str) -> float:
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
        return EvalResult(qa, answer, f, r, c, passed, ftype, cp, cr)"""

content = re.sub(r'    def evaluate_faithfulness\(self, answer: str, context: str\) -> float:.*?(?=\n\n# ---)', ragas_repl, content, flags=re.DOTALL)

# rerank_by_overlap
rerank_repl = r"""def rerank_by_overlap(contexts: list[str], query: str) -> list[str]:
    q_tok = _tokenize(query)
    return sorted(contexts, key=lambda c: len(_tokenize(c) & q_tok), reverse=True)"""
content = re.sub(r'def rerank_by_overlap\(contexts: list\[str\], query: str\) -> list\[str\]:.*?(?=\n\n# ---)', rerank_repl, content, flags=re.DOTALL)

# LLMJudge replacements
llm_judge_repl = r"""    def __init__(self, judge_llm_fn: Callable[[str], str]) -> None:
        self.judge_llm_fn = judge_llm_fn

    def score_response(
        self,
        question: str,
        answer: str,
        rubric: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = f"Question: {question}\nAnswer: {answer}\nRubric: {rubric}\nPlease provide JSON scores."
        response = self.judge_llm_fn(prompt)
        
        default_scores = {k: 0.5 for k in rubric.keys()}
        import json
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                scores = data.get("scores", default_scores)
                parsed_scores = {}
                for k in rubric.keys():
                    val = scores.get(k, 0.5)
                    try:
                        parsed_scores[k] = float(val)
                    except:
                        parsed_scores[k] = 0.5
                return {"scores": parsed_scores, "reasoning": data.get("reasoning", "")}
            else:
                return {"scores": default_scores, "reasoning": response}
        except:
            return {"scores": default_scores, "reasoning": response}

    def detect_bias(self, scores_batch: list[dict[str, Any]]) -> dict[str, Any]:
        if not scores_batch:
            return {"positional_bias": False, "leniency_bias": False, "severity_bias": False}
        
        total_scores = 0
        num_scores = 0
        for b in scores_batch:
            scores = b.get("scores", {})
            for v in scores.values():
                total_scores += v
                num_scores += 1
                
        avg = total_scores / num_scores if num_scores > 0 else 0.5
        return {
            "positional_bias": False,
            "leniency_bias": avg > 0.8,
            "severity_bias": avg < 0.3,
        }"""
content = re.sub(r'    def __init__\(self, judge_llm_fn: Callable\[\[str\], str\]\) -> None:.*?(?=\n\n# ---)', llm_judge_repl, content, flags=re.DOTALL)


# BenchmarkRunner replacements
bench_runner_repl = r"""    def run(
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
        return [r for r in results if r.faithfulness < threshold or r.relevance < threshold or r.completeness < threshold]"""
content = re.sub(r'    def run\(.*?(?=\n\n# ---)', bench_runner_repl, content, flags=re.DOTALL)


# FailureAnalyzer replacements
fail_analyzer_repl = r"""    def categorize_failures(
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
        ]"""
content = re.sub(r'    def categorize_failures\(.*?(?=\n\n# ---)', fail_analyzer_repl, content, flags=re.DOTALL)


with open('template.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done replacing.")
