r"""
交互式问答命令行：检索 + （可选）LLM 生成 = RAG

注意：开头把 stdout 设成 utf-8，保证 Windows 控制台能正常显示中文。
"""

import argparse
import os
import sys

# 让 import 能找到同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Windows 控制台默认 GBK，重配成 utf-8，避免中文乱码
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from qa_system import QASystem, format_answer
from config import llm_config_summary

ALL_METHODS = ["overlap", "tfidf", "bm25", "embedding"]


def print_result(results, topk: int):
    """打印检索结果。"""
    if not results:
        print("（未命中任何知识库条目）")
        return
    shown = results[:topk]
    if len(shown) == 1:
        print(format_answer(shown[0]))
    else:
        for r in shown:
            print("-" * 60)
            print(format_answer(r))
        print("-" * 60)
        print(f"共展示 {len(shown)} 个候选，最佳为 [{shown[0].qa_id}]。")


def interactive(qa: QASystem, topk: int, use_generate: bool):
    print("=" * 60)
    print("  Python 编程问答系统（RAG）")
    print(f"  检索方法：{qa.retriever_name}   知识库：{len(qa.kb)} 条")
    print(f"  生成模式：{'RAG(LLM 生成)' if use_generate else '检索式(直接抽取)'}")
    if use_generate:
        print(f"  LLM 后端：{llm_config_summary()}")
        print("  （Ollama 不可用时自动转本地模型，首次生成需加载模型几秒~十几秒）")
    print("  :method/:topk/:gen 调参；quit 退出。")
    print("=" * 60)
    while True:
        try:
            query = input("\n❓ 请输入问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not query:
            continue
        low = query.lower()
        if low in ("quit", "exit", ":q"):
            print("再见！")
            break
        if low.startswith(":method"):
            name = query.split(None, 1)
            if len(name) < 2 or name[1] not in ALL_METHODS:
                print(f"用法：:method {'|'.join(ALL_METHODS)}")
                continue
            qa = QASystem(method=name[1])
            print(f"已切换检索方法 → {qa.retriever_name}")
            continue
        if low.startswith(":topk"):
            parts = query.split()
            if len(parts) < 2 or not parts[1].isdigit():
                print("用法：:topk <数字>")
                continue
            topk = int(parts[1])
            print(f"展示候选数 → {topk}")
            continue
        if low.startswith(":gen"):
            parts = query.split()
            if len(parts) < 2 or parts[1] not in ("on", "off"):
                print("用法：:gen on|off  （开关 RAG 生成模式）")
                continue
            use_generate = parts[1] == "on"
            print(f"生成模式 → {'RAG(LLM 生成)' if use_generate else '检索式(直接抽取)'}")
            if use_generate:
                print(f"  LLM 后端：{llm_config_summary()}")
            continue
        if query in ("help", ":h", "?"):
            print(f"命令：:method {'|'.join(ALL_METHODS)}  /  :topk N  /  :gen on|off  /  quit")
            continue

        if use_generate:
            # RAG：检索 top-k → LLM 生成
            n = max(topk, 3)
            results = qa.search(query, topk=n)
            print("\n--- 检索到的参考 ---")
            for r in results:
                print(f"  [{r.qa_id}] {r.question}  (score={r.score:.4f})")
            print("\n--- LLM 生成答案 ---")
            ans = qa.generate(query, topk=n, use_llm=True)
            print(ans)
        else:
            results = qa.search(query, topk=max(topk, 3))
            print_result(results, topk)


def main():
    parser = argparse.ArgumentParser(description="Python 编程问答系统（RAG）")
    parser.add_argument("--method", default="embedding",
                        choices=ALL_METHODS,
                        help="检索方法（默认 embedding 语义向量检索）；"
                             "overlap/tfidf/bm25 为词法检索；不可用时自动降级 bm25")
    parser.add_argument("--topk", type=int, default=1,
                        help="展示候选数量（默认 1；RAG 生成时实际检索 max(topk,3) 条）")
    parser.add_argument("--once", default=None,
                        help="单问模式：直接回答该问题后退出，不进入交互")
    parser.add_argument("--generate", action=argparse.BooleanOptionalAction, default=True,
                        help="启用 RAG 生成模式：检索 top-k → LLM 生成最终答案"
                             "（默认启用；用 --no-generate 关闭，退化为检索式）")
    args = parser.parse_args()

    qa = QASystem(method=args.method)

    if args.once:
        if args.generate:
            # 单问 + RAG 生成
            print("--- 检索到的参考 ---")
            results = qa.search(args.once, topk=max(args.topk, 3))
            for r in results:
                print(f"  [{r.qa_id}] {r.question}  (score={r.score:.4f})")
            print("\n--- LLM 生成答案 ---")
            print(qa.generate(args.once, topk=max(args.topk, 3), use_llm=True))
        else:
            results = qa.search(args.once, topk=max(args.topk, 1))
            print_result(results, args.topk)
        return

    interactive(qa, args.topk, args.generate)


if __name__ == "__main__":
    main()
