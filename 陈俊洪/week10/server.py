"""
启动：
  cd src
  python terminal_chat.py                       # 交互式问答
  python terminal_chat.py --debug               # 默认开启逐步调试展示
  python terminal_chat.py --stock 600519 --year 2023   # 限定公司/年份
  python terminal_chat.py --query "茅台2023年营收"      # 单次提问后退出

交互命令（问答过程中输入）：
  /debug         切换逐步调试展示（显示向量/BM25/RRF/上下文各阶段）
  /filter 600519 2023   设置股票代码+年份过滤（/filter 清空）
  /mode          查看当前配置
  /help          查看命令帮助
  exit / quit    退出

依赖：与 rag_pipeline 相同（faiss-cpu rank_bm25 jieba openai numpy）
"""

import argparse
import importlib.util
import logging
from pathlib import Path
from typing import Optional

# 与 serve.py 一致：用 importlib 动态加载 rag_pipeline，保持与启动目录无关
_PIPELINE_PATH = Path(__file__).parent / "rag_pipeline.py"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_pipeline_module():
    spec = importlib.util.spec_from_file_location("rag_pipeline", _PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── ANSI 颜色（终端美化，Windows 10+ 终端默认支持）─────────────────────────────

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    CYAN   = "\033[36m"
    GREEN  = "\033[32m"
    YELLOW = "\033[33m"
    BLUE   = "\033[34m"
    MAGENTA= "\033[35m"
    GREY   = "\033[90m"


def _build_source(item: dict) -> str:
    """把 chunk metadata 格式化为可读来源字符串。"""
    s = f"{item.get('stock_code', '')} {item.get('year', '')}年报"
    section = item.get("section", "")
    if section:
        parts = section.split(" > ")
        s += " · " + " > ".join(parts[-2:])
    page = item.get("page_num", -1)
    if page and page != -1:
        s += f" · 第{page}页"
    return s


def _preview(text: str, n: int = 150) -> str:
    text = text.strip().replace("\n", " ")
    return text[:n] + "…" if len(text) > n else text


# ── 逐步调试展示（对应 serve.py 的 /query/debug）───────────────────────────────

def run_debug(module, pipeline, question: str, filter_meta: Optional[dict]):
    """
    逐步执行 RAG 流水线并在终端打印每一步的中间结果：
      ① 向量检索  ② BM25 检索  ③ RRF 融合  ④ 输入 LLM 的上下文  ⑤ 生成答案
    这是把 serve.py 的教学调试接口直接搬到终端，方便观察检索过程。
    """
    TOP_K       = module.TOP_K_RETRIEVE   # 10
    TOP_K_FINAL = module.TOP_K_RERANK     # 4

    # ① 向量检索
    vec_results  = pipeline.vec_store.search(question, TOP_K, filter_meta)
    vec_rank_map = {it["chunk_id"]: r for r, it in enumerate(vec_results, 1)}

    print(f"\n{C.BOLD}{C.CYAN}① 向量检索（FAISS Top-{TOP_K}，余弦相似度）{C.RESET}")
    if not vec_results:
        print(f"  {C.GREY}（无结果）{C.RESET}")
    for i, item in enumerate(vec_results[:5], 1):
        print(f"  {C.DIM}[{i}]{C.RESET} score={item.get('vec_score', 0.0):.3f}  "
              f"{C.GREEN}{_build_source(item)}{C.RESET}")
        print(f"      {C.GREY}{_preview(item['content'])}{C.RESET}")

    # ② BM25 检索
    bm25_results  = pipeline.bm25_store.search(question, TOP_K) if pipeline.bm25_store else []
    bm25_rank_map = {it["chunk_id"]: r for r, it in enumerate(bm25_results, 1)}

    print(f"\n{C.BOLD}{C.CYAN}② BM25 关键词检索（Top-{TOP_K}）{C.RESET}")
    if not bm25_results:
        print(f"  {C.GREY}（未启用 BM25 或无结果）{C.RESET}")
    for i, item in enumerate(bm25_results[:5], 1):
        print(f"  {C.DIM}[{i}]{C.RESET} score={item.get('bm25_score', 0.0):.2f}  "
              f"{C.GREEN}{_build_source(item)}{C.RESET}")
        print(f"      {C.GREY}{_preview(item['content'])}{C.RESET}")

    # ③ RRF 融合
    if bm25_results:
        candidates = module.reciprocal_rank_fusion(vec_results, bm25_results)
    else:
        candidates = vec_results

    print(f"\n{C.BOLD}{C.CYAN}③ RRF 融合排名（标注每条来自哪一路）{C.RESET}")
    for i, item in enumerate(candidates[:5], 1):
        vr = vec_rank_map.get(item["chunk_id"])
        br = bm25_rank_map.get(item["chunk_id"])
        tag = f"向量#{vr}" if vr else ""
        tag += (" & " if vr and br else "") + (f"BM25#{br}" if br else "")
        print(f"  {C.DIM}[{i}]{C.RESET} rrf={item.get('rrf_score', 0.0):.4f}  "
              f"{C.YELLOW}({tag}){C.RESET}  {C.GREEN}{_build_source(item)}{C.RESET}")

    # ④ 组装上下文
    final = candidates[:TOP_K_FINAL]
    context, citations = module.build_context(final)

    print(f"\n{C.BOLD}{C.CYAN}④ 输入 LLM 的上下文（Top-{TOP_K_FINAL} 完整原文）{C.RESET}")
    for i, item in enumerate(final, 1):
        content = item.get("parent_content") or item["content"]
        print(f"  {C.DIM}[{i}] {_build_source(item)}{C.RESET}")
        print(f"      {C.GREY}{_preview(content, 300)}{C.RESET}")

    # ⑤ LLM 生成
    print(f"\n{C.BOLD}{C.CYAN}⑤ LLM 生成答案{C.RESET}")
    answer = module.call_llm(question, context, pipeline.client)
    print(f"\n{C.BOLD}{answer}{C.RESET}")
    _print_citations(citations)


# ── 标准展示（对应 serve.py 的 /query）─────────────────────────────────────────

def run_query(pipeline, question: str, filter_meta: Optional[dict]):
    """标准问答：直接输出答案 + 来源引用。"""
    result = pipeline.query(question, filter_meta=filter_meta, verbose=True)
    print(f"\n{C.BOLD}{result['answer']}{C.RESET}")
    _print_citations(result["citations"])


def _print_citations(citations: list[dict]):
    if citations:
        print(f"\n{C.MAGENTA}── 来源 ──{C.RESET}")
        for c in citations:
            print(f"  {C.MAGENTA}{c['source']}{C.RESET}")


# ── 交互循环 ──────────────────────────────────────────────────────────────────

HELP_TEXT = f"""
{C.BOLD}可用命令：{C.RESET}
  {C.CYAN}/debug{C.RESET}                切换逐步调试展示（显示检索流水线各阶段）
  {C.CYAN}/filter <股票代码> <年份>{C.RESET}  设置过滤，如 /filter 600519 2023
  {C.CYAN}/filter{C.RESET}               清空过滤条件
  {C.CYAN}/mode{C.RESET}                 查看当前配置
  {C.CYAN}/help{C.RESET}                 显示本帮助
  {C.CYAN}exit{C.RESET} / {C.CYAN}quit{C.RESET}          退出
其它任意输入将作为问题进行 RAG 问答。
"""


def interactive_loop(module, pipeline, debug: bool, filter_meta: Optional[dict]):
    print(f"\n{C.BOLD}{C.BLUE}═══ 年报 RAG 终端问答系统 ═══{C.RESET}")
    print(f"{C.GREY}模型：{module.LLM_MODEL}  |  向量库：{module.INDEX_PATH}{C.RESET}")
    print(f"{C.GREY}BM25={'on' if pipeline.use_bm25 else 'off'}  "
          f"Rerank={'on' if pipeline.use_rerank else 'off'}  "
          f"调试展示={'on' if debug else 'off'}{C.RESET}")
    print(f"{C.GREY}输入 /help 查看命令，exit 退出{C.RESET}")

    while True:
        try:
            raw = input(f"\n{C.BOLD}{C.BLUE}问题 › {C.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue
        if raw.lower() in ("exit", "quit"):
            break

        # ── 命令处理 ──
        if raw.startswith("/"):
            parts = raw.split()
            cmd = parts[0].lower()

            if cmd == "/help":
                print(HELP_TEXT)
            elif cmd == "/debug":
                debug = not debug
                print(f"{C.YELLOW}逐步调试展示已{'开启' if debug else '关闭'}{C.RESET}")
            elif cmd == "/mode":
                fm = filter_meta or "（无）"
                print(f"{C.YELLOW}BM25={'on' if pipeline.use_bm25 else 'off'}  "
                      f"Rerank={'on' if pipeline.use_rerank else 'off'}  "
                      f"调试展示={'on' if debug else 'off'}  过滤={fm}{C.RESET}")
            elif cmd == "/filter":
                if len(parts) == 1:
                    filter_meta = None
                    print(f"{C.YELLOW}已清空过滤条件{C.RESET}")
                else:
                    filter_meta = {}
                    if len(parts) >= 2:
                        filter_meta["stock_code"] = parts[1]
                    if len(parts) >= 3:
                        filter_meta["year"] = parts[2]
                    print(f"{C.YELLOW}过滤条件设为：{filter_meta}{C.RESET}")
            else:
                print(f"{C.YELLOW}未知命令 {cmd}，输入 /help 查看帮助{C.RESET}")
            continue

        # ── 问答 ──
        try:
            if debug:
                run_debug(module, pipeline, raw, filter_meta)
            else:
                run_query(pipeline, raw, filter_meta)
        except Exception as e:
            logger.error(f"处理失败: {e}", exc_info=True)
            print(f"{C.YELLOW}出错了：{e}{C.RESET}")

    print(f"{C.GREY}再见 👋{C.RESET}")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="年报 RAG 终端问答系统（无 Web 框架）")
    parser.add_argument("--query",  type=str, default=None, help="单次提问，问完即退出")
    parser.add_argument("--stock",  type=str, default=None, help="股票代码，如 600519")
    parser.add_argument("--year",   type=str, default=None, help="年份，如 2023")
    parser.add_argument("--debug",  action="store_true", help="开启逐步调试展示")
    parser.add_argument("--no-bm25",   action="store_true", help="关闭 BM25（消融实验用）")
    parser.add_argument("--rerank",    action="store_true", help="开启 CrossEncoder 精排")
    args = parser.parse_args()

    filter_meta = {}
    if args.stock: filter_meta["stock_code"] = args.stock
    if args.year:  filter_meta["year"] = args.year
    filter_meta = filter_meta or None

    logger.info("初始化 RAG Pipeline...")
    module = _load_pipeline_module()
    pipeline = module.RAGPipeline(
        use_bm25          = not args.no_bm25,
        use_rerank        = args.rerank,
        use_query_rewrite = False,
    )
    logger.info("Pipeline 初始化完成")

    if args.query:
        # 单次问答模式
        if args.debug:
            run_debug(module, pipeline, args.query, filter_meta)
        else:
            run_query(pipeline, args.query, filter_meta)
    else:
        interactive_loop(module, pipeline, args.debug, filter_meta)


if __name__ == "__main__":
    main()
