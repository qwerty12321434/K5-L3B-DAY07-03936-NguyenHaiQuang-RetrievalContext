"""TF-IDF offline và benchmark so sánh retrieval trên kho Day06."""
import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path


def tokenize(text):
    return re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold())


def validate(chunks, questions):
    if not isinstance(chunks, list) or not chunks:
        raise ValueError('chunks phải là JSON array không rỗng')
    if not isinstance(questions, list) or not questions:
        raise ValueError('questions phải là JSON array không rỗng')
    ids = set()
    for c in chunks:
        for key in ('id', 'text', 'source_id', 'source', 'page', 'metadata'):
            if key not in c or c[key] is None:
                raise ValueError(f'chunk thiếu {key}')
        if not all(isinstance(c[k], str) and c[k].strip() for k in ('id', 'text', 'source_id', 'source')):
            raise ValueError('chunk id/text/source_id/source phải là chuỗi không rỗng')
        if not isinstance(c['metadata'], dict) or c['id'] in ids:
            raise ValueError('metadata phải là object và ID chunk phải duy nhất')
        ids.add(c['id'])
    ids.clear()
    for q in questions:
        if not isinstance(q.get('id'), str) or not q['id'].strip() or q['id'] in ids:
            raise ValueError('question id phải là chuỗi không rỗng và duy nhất')
        ids.add(q['id'])
        if q.get('split') not in ('dev', 'test') or not isinstance(q.get('question'), str) or not q['question'].strip():
            raise ValueError(f"question {q['id']}: split hoặc question không hợp lệ")
        if not isinstance(q.get('expected_answer'), str) or not q['expected_answer'].strip():
            raise ValueError(f"question {q['id']}: expected_answer phải là chuỗi không rỗng")
        if not isinstance(q.get('gold'), list) or not q['gold']:
            raise ValueError(f"question {q['id']}: gold phải là danh sách không rỗng")
        for g in q['gold']:
            if not all(key in g and g[key] is not None for key in ('source_id', 'page', 'evidence')) or not isinstance(g['evidence'], str) or not g['evidence']:
                raise ValueError(f"question {q['id']}: gold không hợp lệ")
            if not any(hit(c, [g]) for c in chunks):
                raise ValueError(f"question {q['id']}: gold không nằm trọn trong chunk đúng nguồn/trang")


def hit(chunk, gold):
    return any(chunk['source_id'] == g['source_id'] and chunk['page'] == g['page'] and g['evidence'] in chunk['text'] for g in gold)


class TfidfIndex:
    def __init__(self, chunks):
        self.chunks = chunks
        counts = [Counter(tokenize(c['text'])) for c in chunks]
        df = Counter(term for row in counts for term in row)
        self.idf = {term: math.log((1 + len(chunks)) / (1 + count)) + 1 for term, count in df.items()}
        self.vectors = [self._vector(row) for row in counts]

    def _vector(self, counts):
        vector = {term: count * self.idf[term] for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1
        return {term: value / norm for term, value in vector.items()}

    def search(self, question, top_k=3):
        query = self._vector(Counter(tokenize(question)))
        results = []
        for chunk, vector in zip(self.chunks, self.vectors):
            score = sum(value * vector.get(term, 0) for term, value in query.items())
            if score > 0:
                results.append({**chunk, 'score': score, 'strategy': 'tfidf'})
        return sorted(results, key=lambda c: (-c['score'], c['id']))[:top_k]


def pack_context(results, budget=800):
    blocks, ids, seen = [], [], set()
    for c in results:
        if c['id'] in seen:
            continue
        seen.add(c['id'])
        block = f"[{c['id']} | {c['source']} | page {c['page']}]\n{c['text']}"
        if len('\n\n'.join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(c['id'])
    return '\n\n'.join(blocks), ids


def evaluate(index, questions, budget=800):
    traces = []
    for q in questions:
        start = time.perf_counter()
        results = index.search(q['question'], top_k=3)
        context, ids = pack_context(results, budget)
        elapsed_ms = (time.perf_counter() - start) * 1000
        marked = [{**c, 'hit': hit(c, q['gold'])} for c in results]
        traces.append({'query_id': q['id'], 'split': q['split'], 'question': q['question'],
                       'strategy': 'tfidf', 'top_k': 3, 'budget_chars': budget,
                       'gold': q['gold'], 'results': marked, 'context': context, 'context_ids': ids,
                       'context_chars': len(context), 'elapsed_ms': elapsed_ms,
                       'hit1': bool(marked and marked[0]['hit']),
                       'hit3': any(c['hit'] for c in marked),
                       'context_hit': any(c['hit'] and c['id'] in ids for c in marked)})
    return traces


def run_day6_comparison(
    day6_root: Path, database_url: str, ollama_url: str, questions_path: Path, output: Path,
):
    # Chỉ nạp code kết nối Day06 khi chạy so sánh. TF-IDF offline vẫn chạy độc lập.
    from day6_retriever import connect_day6
    from retrieval_benchmark import (
        CANDIDATES, CONTEXT_BUDGET, RRF_K, TOP_K, load_chunks, run_case, summarize,
    )

    # Runner này dành cho dev. Test được giữ riêng để chạy sau khi chốt cấu hình.
    raw_cases = questions_path.read_bytes()
    cases = json.loads(raw_cases)
    if any(case.get("split") != "dev" for case in cases):
        raise ValueError("Chỉ nhận tập dev. Giữ nguyên tập test cho giai đoạn đánh giá cuối.")
    answerable = [case for case in cases if case["answerable"] is True]
    absent = [case for case in cases if case["answerable"] is False]
    if len(answerable) < 15 or len(absent) < 2 or len(cases) != len(answerable) + len(absent):
        raise ValueError("Dev cần ít nhất 15 câu có đáp án và 2 câu thiếu bằng chứng.")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("ID câu hỏi phải duy nhất.")

    with connect_day6(day6_root, database_url, ollama_url) as retriever:
        # Giữ ảnh chụp nhất quán dù kho Day06 có một lần ingest khác chạy đồng thời.
        with retriever.store.connection.transaction():
            retriever.store.connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            chunks = load_chunks(retriever)
            validate(chunks, answerable)
            started = perf_counter()
            lexical = TfidfIndex(chunks)
            build_ms = (perf_counter() - started) * 1000
            # Làm nóng embedding trước khi đo truy vấn, ghi riêng thời gian làm nóng.
            started = perf_counter()
            retriever.search("kiểm tra kết nối kho dữ liệu cá nhân", top_k=1)
            warmup_ms = (perf_counter() - started) * 1000
            traces = [row for case in cases for row in run_case(case, retriever, lexical)]

        chunk_bytes = json.dumps(chunks, ensure_ascii=False, sort_keys=True).encode("utf-8")
        summary = {
            "run_at_utc": datetime.now(timezone.utc).isoformat(),
            "split": "dev",
            "answerable": len(answerable),
            "insufficient_evidence": len(absent),
            "questions_sha256": hashlib.sha256(raw_cases).hexdigest(),
            "corpus": {
                "index": "day6_chunks/B",
                "chunks": len(chunks),
                "chunks_sha256": hashlib.sha256(chunk_bytes).hexdigest(),
                "embedding_model": retriever.embedder.model_name,
                "model_digest": retriever.embedder.digest,
                "embedding_dimensions": retriever.embedder.dimensions,
                **retriever.store.versions(),
            },
            "settings": {
                "query": "original",
                "filters": {},
                "candidate_limit_per_branch": CANDIDATES,
                "ranking_cutoff": TOP_K,
                "rrf_k": RRF_K,
                "context_budget_chars": CONTEXT_BUDGET,
                "context_sent_to_llm": False,
                "llm_tokens": None,
                "tfidf_build_ms": round(build_ms, 3),
                "embedding_warmup_ms": round(warmup_ms, 3),
            },
            "methods": summarize(traces),
        }

    output.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    (output / "summary.json").write_text(serialized, encoding="utf-8")
    with (output / "traces.jsonl").open("w", encoding="utf-8") as file:
        for row in traces:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Chạy TF-IDF offline hoặc so sánh retrieval trên kho Day06.")
    parser.add_argument("--compare", action="store_true", help="So sánh dense, TF-IDF và hybrid trên dev.")
    parser.add_argument("--day6-root", type=Path, default=os.environ.get("DAY6_ROOT"))
    parser.add_argument('--chunks', type=Path)
    parser.add_argument('--questions', type=Path)
    parser.add_argument('--split', choices=('dev', 'test'))
    parser.add_argument('--output', type=Path, help='Thư mục output; file cùng tên sẽ bị ghi đè')
    args = parser.parse_args(argv)

    if args.compare:
        database_url = os.environ.get("DATABASE_URL")
        if not args.day6_root or not database_url:
            parser.error("Cần đặt DAY6_ROOT (hoặc --day6-root) và DATABASE_URL.")
        if args.chunks or args.split == "test":
            parser.error("--compare dùng chunk B trực tiếp từ Day06 và chỉ chạy dev.")
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        questions = args.questions or Path("data/benchmark_dev.json")
        output = args.output or Path("results/retrieval_dev")
        try:
            summary = run_day6_comparison(args.day6_root, database_url, ollama_url, questions, output)
        except Exception as error:
            parser.exit(1, f"So sánh Day06 thất bại: {error}\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if any(method["errors"] for method in summary["methods"].values()):
            parser.exit(1, "Một số lượt truy xuất lỗi. Xem trường error trong traces.jsonl.\n")
        return

    if not (args.chunks and args.questions and args.split and args.output):
        parser.error("TF-IDF offline cần --chunks, --questions, --split và --output.")
    chunks = json.loads(args.chunks.read_text(encoding='utf-8'))
    questions = json.loads(args.questions.read_text(encoding='utf-8'))
    # Do not inspect held-out test evidence during dev work.
    selected = [q for q in questions if q.get('split') == args.split]
    validate(chunks, selected)
    started = time.perf_counter()
    index = TfidfIndex(chunks)
    index_seconds = time.perf_counter() - started
    traces = evaluate(index, selected)
    summary = {'strategy': 'tfidf', 'split': args.split, 'n': len(traces),
               'hit1': sum(t['hit1'] for t in traces), 'hit3': sum(t['hit3'] for t in traces),
               'context_hit': sum(t['context_hit'] for t in traces),
               'avg_context_chars': statistics.mean(t['context_chars'] for t in traces),
               'median_ms': statistics.median(t['elapsed_ms'] for t in traces),
               'index_seconds': index_seconds, 'service_cost_usd': 0}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, obj in (('traces.json', traces), ('summary.json', summary)):
        (args.output / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    main()
