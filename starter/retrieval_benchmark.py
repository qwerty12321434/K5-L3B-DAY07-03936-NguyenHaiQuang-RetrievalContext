"""Xếp hạng hybrid RRF và các hàm tính metric dùng bởi baseline.py."""

from __future__ import annotations

import statistics
import sys
from time import perf_counter

from baseline import hit, pack_context


TOP_K = 3
CANDIDATES = 10
RRF_K = 60
CONTEXT_BUDGET = 1500
METHODS = ("dense", "tfidf", "hybrid_rrf")


def load_chunks(retriever):
    # TF-IDF đọc trực tiếp các chunk B từ cùng DB và cùng model digest với dense.
    # Chỉ lấy văn bản và metadata vì TF-IDF không cần cột embedding.
    return retriever.store.connection.execute("""
        SELECT chunk_id AS id, document_id AS source_id, source, page, text, metadata
        FROM day6_chunks
        WHERE strategy = 'B' AND model_digest = %s
        ORDER BY chunk_id
    """, (retriever.embedder.digest,)).fetchall()


def fuse_rankings(dense, lexical):
    # RRF cộng điểm theo thứ hạng; hai điểm cosine gốc không cần cùng thang đo.
    chunks, ranks = {}, {}
    for method, results in (("dense", dense), ("tfidf", lexical)):
        for rank, chunk in enumerate(results, start=1):
            chunk_id = chunk["id"]
            chunks[chunk_id] = chunk
            ranks.setdefault(chunk_id, {})[method] = rank
    fused = [
        {
            **chunks[chunk_id],
            "score": sum(1 / (RRF_K + rank) for rank in method_ranks.values()),
            "component_ranks": method_ranks,
        }
        for chunk_id, method_ranks in ranks.items()
    ]
    return sorted(fused, key=lambda chunk: (-chunk["score"], chunk["id"]))


class AgentRetriever:
    """Dùng đúng ba cách xếp hạng đã so sánh ở benchmark retrieval."""

    def __init__(self, method, dense, lexical):
        self.method = method
        self.dense = dense
        self.lexical = lexical

    def search(self, query, *, filters=None, top_k=3):
        # Cùng một chính sách không lọc metadata cho cả ba phương pháp ở benchmark.
        if filters:
            raise ValueError("Benchmark agent so sánh truy vấn không filter.")
        if self.method == "dense":
            return self.dense.search(query, top_k=top_k)
        if self.method == "tfidf":
            return self.lexical.search(query, top_k=top_k)
        dense = self.dense.search(query, top_k=max(CANDIDATES, top_k))
        lexical = self.lexical.search(query, top_k=max(CANDIDATES, top_k))
        return fuse_rankings(dense, lexical)[:top_k]


def trace_result(case, method, candidates, elapsed_ms, error=None):
    # Cắt top 3 trước khi đóng gói context; dùng cùng budget cho mọi phương pháp.
    started = perf_counter()
    top = candidates[:TOP_K]
    context, context_ids = pack_context(top, budget=CONTEXT_BUDGET)
    elapsed_ms += (perf_counter() - started) * 1000

    # Gold chỉ được đọc sau khi tìm kiếm. Câu thiếu bằng chứng không có nhãn hit.
    gold = case.get("gold")
    results = [
        {
            **{key: value for key, value in chunk.items() if key != "strategy"},
            "rank": rank,
            "evidence_hit": hit(chunk, gold) if gold else None,
        }
        for rank, chunk in enumerate(top, start=1)
    ]
    evidence_ids = {chunk["id"] for chunk in results if chunk["evidence_hit"]}
    return {
        "query_id": case["id"],
        "answerable": case["answerable"],
        "method": method,
        "query": case["question"],
        "filters": {},
        "candidate_count": len(candidates),
        "results": results,
        "context": context,
        "context_ids": context_ids,
        "context_chars": len(context),
        "hit_at_1": bool(top and top[0]["id"] in evidence_ids) if gold else None,
        "evidence_hit_at_3": bool(evidence_ids) if gold else None,
        "packed_context_has_evidence": bool(evidence_ids.intersection(context_ids)) if gold else None,
        "elapsed_ms": round(elapsed_ms, 3),
        "api_cost_usd": 0,
        "error": error,
    }


def run_case(case, retriever, lexical):
    # Chỉ truyền nguyên câu hỏi vào retrieval. Nếu một nhánh lỗi, nhánh còn lại vẫn chạy.
    branches = {}
    for method, search in (("dense", retriever.search), ("tfidf", lexical.search)):
        started = perf_counter()
        candidates, error = [], None
        try:
            candidates = search(case["question"], top_k=CANDIDATES)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        branches[method] = (candidates, (perf_counter() - started) * 1000, error)

    dense, dense_ms, dense_error = branches["dense"]
    tfidf, tfidf_ms, tfidf_error = branches["tfidf"]
    started = perf_counter()
    error = dense_error or tfidf_error
    hybrid = [] if error else fuse_rankings(dense, tfidf)
    # Tính đủ thời gian của hai nhánh cho hybrid, dù dùng lại kết quả của chúng.
    hybrid_ms = dense_ms + tfidf_ms + (perf_counter() - started) * 1000
    traces = [trace_result(case, method, *branch) for method, branch in branches.items()]
    traces.append(trace_result(case, "hybrid_rrf", hybrid, hybrid_ms, error))
    return traces


def summarize(traces):
    def rate(key):
        # Query lỗi vẫn nằm trong mẫu số các câu có đáp án và được tính là miss.
        n = sum(row[key] is True for row in answerable)
        total = len(answerable)
        return {"n": n, "total": total, "percent": round(n / total * 100, 2)}

    methods = {}
    for name in METHODS:
        rows = [row for row in traces if row["method"] == name]
        answerable = [row for row in rows if row["answerable"]]
        successful = [row for row in rows if row["error"] is None]
        methods[name] = {
            "queries": len(rows),
            "errors": len(rows) - len(successful),
            "hit_at_1": rate("hit_at_1"),
            "evidence_hit_at_3": rate("evidence_hit_at_3"),
            "packed_context_evidence_rate": rate("packed_context_has_evidence"),
            "median_ms": round(statistics.median(row["elapsed_ms"] for row in successful), 3)
            if successful else None,
            "api_cost_usd": 0,
        }
    return methods


if __name__ == "__main__":
    # Giữ lệnh cũ để chạy kiểm tra. Điểm vào chính của benchmark là baseline.py.
    from baseline import main

    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main(["--compare", *sys.argv[1:]])
