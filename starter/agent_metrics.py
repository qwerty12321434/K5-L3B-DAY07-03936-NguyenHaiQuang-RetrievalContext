"""Aggregate agent traces with human labels; never infer grounding from citation IDs."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median

from baseline import hit


def aggregate(cases, traces, reviews):
    """Cases: id/answerable; traces: query_id/status/events/citations.

    Reviews: query_id, answer_grade (0..2 for answerable), citation_support
    (one boolean per citation), and abstention_correct (boolean for unanswerable).
    Provider errors are excluded and reported.
    """
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('case ID trùng')
    if len({r['query_id'] for r in traces}) != len(traces) or len({r['query_id'] for r in reviews}) != len(reviews):
        raise ValueError('trace/review ID trùng')
    trace_by_id = {r['query_id']: r for r in traces}
    review_by_id = {r['query_id']: r for r in reviews}
    out = {'cases': len(cases), 'provider_errors': 0, 'tool_use': {'n': 0, 'N': 0},
           'grounded_answer': {'0': 0, '1': 0, '2': 0, 'N': 0, 'mean': None},
           'citation_validity': {'n': 0, 'N': 0}, 'answer_citation_coverage': {'n': 0, 'N': 0},
           'abstention': {'n': 0, 'N': 0}, 'false_abstention': {'n': 0, 'N': 0}}
    for case in cases:
        cid = case['id']
        if not isinstance(case.get('answerable'), bool): raise ValueError(f'{cid}: thiếu answerable boolean')
        if cid not in trace_by_id: raise ValueError(f'{cid}: thiếu trace')
        t = trace_by_id[cid]
        if t.get('status') == 'error':
            out['provider_errors'] += 1
            continue
        if t.get('status') not in ('answered', 'abstain'): raise ValueError(f'{cid}: status không hợp lệ')
        if cid not in review_by_id: raise ValueError(f'{cid}: thiếu nhãn audit')
        r = review_by_id[cid]
        events = t.get('events')
        citations = t.get('citations')
        support = r.get('citation_support')
        if not isinstance(events, list) or not isinstance(citations, list): raise ValueError(f'{cid}: events/citations không hợp lệ')
        if not isinstance(support, list) or len(support) != len(citations) or any(type(v) is not bool for v in support):
            raise ValueError(f'{cid}: cần một nhãn boolean cho mỗi citation')
        out['tool_use']['N'] += 1
        out['tool_use']['n'] += any(e.get('event') == 'search' for e in events)
        out['citation_validity']['n'] += sum(support)
        out['citation_validity']['N'] += len(citations)
        if t['status'] == 'answered':
            out['answer_citation_coverage']['N'] += 1
            out['answer_citation_coverage']['n'] += bool(citations)
        if case['answerable']:
            grade = r.get('answer_grade')
            if type(grade) is not int or grade not in (0, 1, 2): raise ValueError(f'{cid}: answer_grade phải là 0, 1 hoặc 2')
            if t['status'] == 'abstain' and grade != 0: raise ValueError(f'{cid}: abstain trên câu có đáp án phải được chấm 0')
            out['grounded_answer'][str(grade)] += 1
            out['grounded_answer']['N'] += 1
            out['false_abstention']['N'] += 1
            out['false_abstention']['n'] += t['status'] == 'abstain'
        else:
            correct = r.get('abstention_correct')
            if type(correct) is not bool:
                raise ValueError(f'{cid}: cần nhãn abstention_correct do người chấm xác nhận')
            if correct and t['status'] != 'abstain':
                raise ValueError(f'{cid}: nhãn từ chối đúng không khớp status')
            out['abstention']['N'] += 1
            out['abstention']['n'] += correct
    g = out['grounded_answer']
    if g['N']: g['mean'] = (g['1'] + 2*g['2']) / g['N']
    return out


def evaluate_details(cases, traces, reviews):
    """Tổng hợp retrieval, nhãn context và chi phí của một cấu hình."""
    by_trace = {trace["query_id"]: trace for trace in traces}
    by_review = {review["query_id"]: review for review in reviews}
    context_grades = {"0": 0, "1": 0, "2": 0}
    ranking = Counter()
    errors = []
    durations = []
    tool_counts = Counter()
    token_counts = Counter()
    for case in cases:
        case_id = case["id"]
        trace = by_trace[case_id]
        if trace["status"] == "error":
            errors.append({"query_id": case_id, "error": trace.get("error"), "elapsed_ms": trace["elapsed_ms"]})
            continue
        review = by_review[case_id]
        if not isinstance(review.get("answer_reason"), str) or not review["answer_reason"].strip():
            raise ValueError(f"{case_id}: thiếu lý do chấm câu trả lời")
        citation_reasons = review.get("citation_reasons")
        if (not isinstance(citation_reasons, list) or len(citation_reasons) != len(trace["citations"])
                or any(not isinstance(reason, str) or not reason.strip() for reason in citation_reasons)):
            raise ValueError(f"{case_id}: cần lý do cho từng citation")
        searches = [event for event in trace["events"] if event["event"] == "search"]
        durations.append(trace["elapsed_ms"])
        tool_counts[str(len(searches))] += 1
        for call in trace["model_calls"]:
            token_counts["prompt"] += call["prompt_tokens"] or 0
            token_counts["output"] += call["output_tokens"] or 0
        if not case["answerable"]:
            continue

        grade = review.get("context_grade")
        if type(grade) is not int or grade not in (0, 1, 2) or not review.get("context_reason", "").strip():
            raise ValueError(f"{case_id}: cần context_grade 0/1/2 và lý do")
        context_grades[str(grade)] += 1
        # Xếp hạng dùng gold cố định. Context và citation do người đọc trace chấm.
        first = searches[0]["results"] if searches and searches[0]["query"] == case["question"] else []
        ranking["original_hit_at_1"] += any(hit(chunk, case["gold"]) for chunk in first[:1])
        ranking["original_evidence_hit_at_3"] += any(hit(chunk, case["gold"]) for chunk in first[:3])
        ranking["any_search_evidence_hit_at_3"] += any(
            hit(chunk, case["gold"]) for event in searches for chunk in event["results"][:3]
        )

    total = sum(case["answerable"] for case in cases)

    def rate(n):
        return {"n": n, "N": total, "percent": round(100 * n / total, 2) if total else None}

    return {
        "retrieval": {name: rate(ranking[name]) for name in (
            "original_hit_at_1", "original_evidence_hit_at_3", "any_search_evidence_hit_at_3",
        )},
        "context_grades": context_grades,
        "context_evidence_rate": rate(context_grades["2"]),
        "search_call_distribution": dict(tool_counts),
        "llm_tokens_successful_queries": dict(token_counts),
        "median_ms": round(median(durations), 3) if durations else None,
        "api_cost_usd": sum(trace["api_cost_usd"] for trace in traces),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description='Tổng hợp trace agent và nhãn audit thủ công')
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--traces', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--split', choices=('dev', 'test'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--details', action='store_true', help='Tổng hợp thêm retrieval, context và thời gian.')
    args = parser.parse_args()
    def load(path): return json.loads(path.read_text(encoding='utf-8'))
    cases = [c for c in load(args.cases) if c.get('split') == args.split]
    if not cases: raise ValueError('không có case trong split')
    traces, reviews = load(args.traces), load(args.reviews)
    report = {'split': args.split, **aggregate(cases, traces, reviews)}
    if args.details:
        report.update(evaluate_details(cases, traces, reviews))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    main()
