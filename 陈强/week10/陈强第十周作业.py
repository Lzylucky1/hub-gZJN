import argparse
import logging
from typing import Optional

from rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def print_result(question: str, result: dict) -> None:
    print("\n" + "=" * 60)
    print(f"问题：{question}")
    print("=" * 60)
    print(result["answer"])
    if result.get("citations"):
        print("\n── 来源 ──")
        for citation in result["citations"]:
            print(f"  {citation['source']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="年报问答系统 CLI")
    parser.add_argument("--query", type=str, default=None, help="要问的问题")
    parser.add_argument("--stock", type=str, default=None, help="股票代码，例如 600519")
    parser.add_argument("--year", type=str, default=None, help="年份，例如 2023")
    parser.add_argument("--query-rewrite", action="store_true", help="使用查询改写增强检索")
    parser.add_argument("--no-bm25", action="store_true", help="关闭 BM25 检索")
    parser.add_argument("--no-rerank", action="store_true", help="关闭 CrossEncoder rerank")
    args = parser.parse_args()

    pipeline = RAGPipeline(
        use_bm25=not args.no_bm25,
        use_rerank=not args.no_rerank,
        use_query_rewrite=args.query_rewrite,
    )

    filter_meta: Optional[dict] = {}
    if args.stock:
        filter_meta["stock_code"] = args.stock
    if args.year:
        filter_meta["year"] = args.year
    if not filter_meta:
        filter_meta = None

    if args.query:
        result = pipeline.query(args.query, filter_meta=filter_meta, verbose=True)
        print_result(args.query, result)
        return

    print("年报问答系统 CLI")
    print("请输入问题，输入 exit 退出。")
    while True:
        try:
            question = input("问题：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n已退出。")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("已退出。")
            break
        result = pipeline.query(question, filter_meta=filter_meta, verbose=True)
        print_result(question, result)


if __name__ == "__main__":
    main()
