"""Read the Day 6 chunk index and expose its dense search to the Day 7 starter."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any


# Nạp code Day06 từ DAY6_ROOT để dùng lại cách tạo embedding và câu SQL có sẵn.
def load_day6_modules(day6_root: Path):
    # Đặt tên riêng cho package Day06 để hai repo có thể cùng dùng tên thư mục src.
    package = day6_root.resolve() / "starter" / "src" / "__init__.py"
    if not package.is_file():
        raise FileNotFoundError(f"Không tìm thấy code Day06 tại {package}. Kiểm tra DAY6_ROOT.")
    name = "_day6_ingestion"
    # Python lưu module theo tên trong tiến trình. Kiểm tra để tránh dùng nhầm checkout.
    if name in sys.modules and Path(sys.modules[name].__file__) != package:
        raise ValueError("Tiến trình này đã nạp một bản Day06 khác.")
    if name not in sys.modules:
        # Chỉ định thư mục package để các import tương đối của Day06 vẫn hoạt động.
        spec = spec_from_file_location(name, package, submodule_search_locations=[str(package.parent)])
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return import_module(f"{name}.embeddings"), import_module(f"{name}.store")


class Day6Retriever:
    """Adapt the shared Day 6 B index to the Day 7 Retriever contract."""

    def __init__(self, embedder, store) -> None:
        self.embedder = embedder
        self.store = store

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        if not query.strip() or top_k < 1:
            raise ValueError("Cần có câu truy vấn và top_k phải lớn hơn 0.")
        # Ollama chuyển câu hỏi thành vector theo cùng model và định dạng query của Day06.
        vector = self.embedder.embed_query(query)
        # PgVectorStore của Day06 chạy SQL trên bảng day6_chunks, xếp theo độ tương đồng cosine.
        # B tìm trên toàn bộ chunk B. C thêm filter metadata trên chính các vector B đó.
        rows = self.store.search("C" if filters else "B", vector, top_k=top_k, filters=filters)
        # Đổi tên chunk_id và document_id theo schema Day07. URL nguồn và trang lấy từ DB.
        return [
            {
                "id": row["chunk_id"],
                "source_id": row["document_id"],
                "source": row["source"],
                "page": row["page"],
                "text": row["text"],
                "metadata": row["metadata"],
                "score": row["score"],
            }
            for row in rows
        ]


@contextmanager
def connect_day6(day6_root: Path, database_url: str, ollama_url: str):
    # DAY6_ROOT trỏ tới code Python; DATABASE_URL và OLLAMA_BASE_URL trỏ tới hai dịch vụ.
    embeddings, storage = load_day6_modules(day6_root)
    embedder = embeddings.OllamaEmbedder(ollama_url)
    # Store mở kết nối PostgreSQL qua psycopg và dùng digest để chọn vector đã nạp cùng model.
    # Khối with đóng kết nối khi chạy xong hoặc có lỗi.
    with storage.PgVectorStore(database_url, embedder.dimensions, embedder.digest) as store:
        # Các giao dịch của Day07 trên kết nối này chỉ được đọc kho Day06.
        store.connection.read_only = True
        yield Day6Retriever(embedder, store)


# CLI để kiểm tra thủ công. Agent về sau có thể gọi connect_day6() và search() trực tiếp.
def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm tra truy xuất Day07 từ kho vector Day06.")
    parser.add_argument("--day6-root", type=Path, default=os.environ.get("DAY6_ROOT"))
    parser.add_argument(
        "--query", default="Trong phiếu yêu cầu cung cấp dữ liệu, ghi cách nhận hồ sơ ở mục nào?"
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    database_url = os.environ.get("DATABASE_URL")
    if args.day6_root is None or not database_url:
        parser.error("Cần đặt DAY6_ROOT (hoặc --day6-root) và DATABASE_URL.")
    if not args.query.strip() or args.top_k < 1:
        parser.error("--query phải có nội dung và --top-k phải lớn hơn 0.")

    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    try:
        with connect_day6(args.day6_root, database_url, ollama_url) as retriever:
            # Phân biệt kho thiếu vector tương thích với truy vấn tìm kiếm không có kết quả.
            indexed_chunks = retriever.store.counts().get("B", 0)
            if not indexed_chunks:
                raise RuntimeError(
                    "Kho Day06 không có chunk B khớp digest của model Ollama đang chạy. "
                    "Kiểm tra database, dữ liệu đã ingest và phiên bản model."
                )
            # CLI gọi search trực tiếp để kiểm tra Ollama và pgvector trong một lượt chạy.
            results = retriever.search(args.query, top_k=args.top_k)
            if not results:
                raise RuntimeError("pgvector không trả về chunk cho truy vấn kiểm tra.")
            # In thông tin kho và model cùng top-k để đối chiếu với snapshot Day06.
            report = {
                "status": "ok",
                "storage": {
                    **retriever.store.versions(),
                    "table": "day6_chunks",
                    "strategy": "B",
                    "indexed_chunks": indexed_chunks,
                },
                "embedding": {
                    "model": retriever.embedder.model_name,
                    "dimensions": retriever.embedder.dimensions,
                    "digest": retriever.embedder.digest,
                    "quantization": retriever.embedder.quantization,
                    "ollama_version": retriever.embedder.version,
                },
                "query": args.query,
                "results": [
                    {"rank": rank, **chunk} for rank, chunk in enumerate(results, start=1)
                ],
            }
    except Exception as error:
        parser.exit(1, f"Kiểm tra Day06 thất bại: {error}\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # Giữ tiếng Việt trong JSON và báo lỗi khi chuyển hướng đầu ra trên Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
