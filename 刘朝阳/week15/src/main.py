"""
CLI 入口 — 跑一次多 subagent 并行调研，彩色输出展示全程

用法：
  python -m src.main "你的调研问题"
  python -m src.main                    # 用内置示例问题
  python -m src.main --serial "问题"   # 串行模式（对比基线）
  python -m src.main --compare "问题"  # 并行 vs 串行对比

环境变量：
  BAILIAN_API_KEY / ANTHROPIC_API_KEY  — LLM key（必填）
  TAVILY_API_KEY                       — 搜索 key（可选，有则用 Tavily）
"""

import sys, time, argparse, threading, os

# 让 bare import（from orchestrator import ...）在 -m 和直接运行时都生效
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ANSI 彩色（Windows 10+ 终端支持）
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    GRAY    = "\033[90m"

# Windows: 启用 ANSI 转义
if sys.platform == "win32":
    import ctypes
    try:
        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11), 7
        )
    except Exception:
        pass

# 线程安全的 print（多 subagent 并行输出时防交错）
_print_lock = threading.Lock()


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _safe_print(*args, **kw):
    with _print_lock:
        print(*args, **kw)


def make_callbacks():
    """创建一组回调，用彩色打印主 agent / subagent 的每一步。"""

    def on_main_step(step: dict):
        tag = f"{C.BOLD}{C.CYAN}[MAIN {_ts()}]{C.RESET}"
        if step.get("final"):
            _safe_print(f"\n{tag} {C.GREEN}Final Answer{C.RESET} "
                        f"(用时汇总中)")
        elif step.get("observation") is None:
            # pre：决策刚出
            _safe_print(f"\n{tag} Thought: {step['thought'][:100]}")
            _safe_print(f"{tag} {C.YELLOW}Action: {step['action']}{C.RESET}")
            ai = step.get("action_input", "")
            if ai:
                _safe_print(f"{tag} Action Input: {ai[:120]}")
        else:
            # post：工具返回
            obs = step.get("observation", "") or ""
            _safe_print(f"{tag} {C.DIM}Observation: {obs[:150]}...{C.RESET}")

    def on_dispatch(info: dict):
        _safe_print(f"\n{C.BOLD}{C.MAGENTA}{'='*60}{C.RESET}")
        _safe_print(f"{C.BOLD}{C.MAGENTA}[DISPATCH {_ts()}] "
                    f"派发 {len(info['subagent_ids'])} 个子调研员并行执行:{C.RESET}")
        for i, topic in enumerate(info["subtopics"], 1):
            _safe_print(f"  {C.MAGENTA}子课题 {i}: {topic}{C.RESET}")
        _safe_print(f"{C.BOLD}{C.MAGENTA}{'='*60}{C.RESET}\n")

    def on_subagent_step(sid: str, step: dict):
        tag = f"{C.BLUE}[{sid} {_ts()}]{C.RESET}"
        if step.get("final"):
            _safe_print(f"  {tag} {C.GREEN}✓ Final Answer{C.RESET}")
        elif step.get("observation") is None:
            _safe_print(f"  {tag} Thought: {step['thought'][:80]}")
            _safe_print(f"  {tag} {C.YELLOW}→ {step['action']}{C.RESET}"
                        f" ({(step.get('action_input') or '')[:60]})")
        else:
            obs = step.get("observation", "") or ""
            _safe_print(f"  {tag} {C.DIM}obs: {obs[:100]}...{C.RESET}")

    def on_subagent_done(sid: str, duration: float, topic: str):
        _safe_print(f"\n  {C.GREEN}[{sid} 完成] {C.RESET}"
                    f"用时 {duration}s — {topic[:50]}")

    return {
        "on_main_step": on_main_step,
        "on_dispatch": on_dispatch,
        "on_subagent_step": on_subagent_step,
        "on_subagent_done": on_subagent_done,
    }


def run_one(question: str, serial: bool = False) -> dict:
    """跑一次调研并打印过程。"""
    mode = f"{C.RED}串行{C.RESET}" if serial else f"{C.GREEN}并行{C.RESET}"
    _safe_print(f"\n{C.BOLD}{'='*60}{C.RESET}")
    _safe_print(f"{C.BOLD}调研问题: {question}{C.RESET}")
    _safe_print(f"{C.BOLD}执行模式: {mode}{C.RESET}")
    _safe_print(f"{C.BOLD}{'='*60}{C.RESET}")

    from orchestrator import run_research
    cbs = make_callbacks()
    t0 = time.time()
    r = run_research(question, serial=serial, **cbs)
    total = round(time.time() - t0, 2)
    return r, total


def print_report(r: dict, total: float, serial: bool = False):
    """打印最终报告 + 并行统计。"""
    _safe_print(f"\n{C.BOLD}{C.GREEN}{'='*60}{C.RESET}")
    _safe_print(f"{C.BOLD}{C.GREEN}最终报告{C.RESET}")
    _safe_print(f"{C.BOLD}{C.GREEN}{'='*60}{C.RESET}")
    _safe_print(r["final_answer"])

    _safe_print(f"\n{C.BOLD}{'─'*60}{C.RESET}")
    _safe_print(f"{C.BOLD}执行统计:{C.RESET}")
    _safe_print(f"  总耗时: {total}s")
    _safe_print(f"  主 agent 步数: {len(r['main_trace'])}")
    _safe_print(f"  subagent 数: {len(r['subagents'])}")
    _safe_print(f"  派发次数: {len(r['dispatches'])}")
    for ps in r["parallel_stats"]:
        _safe_print(f"  并行统计: {ps['n_subagents']} 个子调研员, "
                    f"wall={ps['wall_clock']}s, "
                    f"串行需={ps['serial_sum']}s, "
                    f"{C.GREEN}加速 {ps['speedup']}×{C.RESET}")
    _safe_print(f"{C.BOLD}{'─'*60}{C.RESET}")


def run_compare(question: str):
    """并行 vs 串行对比。"""
    _safe_print(f"\n{C.BOLD}{C.MAGENTA}并行 vs 串行 A/B 对比{C.RESET}")
    _safe_print(f"{C.BOLD}问题: {question}{C.RESET}\n")

    # 并行
    r_par, t_par = run_one(question, serial=False)
    # 串行
    r_ser, t_ser = run_one(question, serial=True)

    _safe_print(f"\n{C.BOLD}{C.MAGENTA}{'='*60}{C.RESET}")
    _safe_print(f"{C.BOLD}{C.MAGENTA}对比结果{C.RESET}")
    _safe_print(f"{C.BOLD}{C.MAGENTA}{'='*60}{C.RESET}")
    ps_par = r_par["parallel_stats"][0] if r_par["parallel_stats"] else {}
    ps_ser = r_ser["parallel_stats"][0] if r_ser["parallel_stats"] else {}
    _safe_print(f"  并行: total={t_par}s, dispatch wall={ps_par.get('wall_clock','?')}s "
                f"(串行需 {ps_par.get('serial_sum','?')}s, "
                f"{C.GREEN}加速 {ps_par.get('speedup','?')}×{C.RESET})")
    _safe_print(f"  串行: total={t_ser}s, dispatch wall={ps_ser.get('wall_clock','?')}s "
                f"(串行需 {ps_ser.get('serial_sum','?')}s, "
                f"{C.RED}加速 {ps_ser.get('speedup','?')}×{C.RESET})")
    _safe_print(f"{C.BOLD}{C.MAGENTA}{'='*60}{C.RESET}")


DEFAULT_QUESTION = (
    "Research: 1) Overview of the Go programming language "
    "2) Go's concurrency model with goroutines and channels"
)


def main():
    parser = argparse.ArgumentParser(
        description="并行 Subagent 调研系统"
    )
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION,
                        help="调研问题（不填则用内置示例）")
    parser.add_argument("--serial", action="store_true",
                        help="串行模式（对比基线）")
    parser.add_argument("--compare", action="store_true",
                        help="并行 vs 串行 A/B 对比")
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING)

    if args.compare:
        run_compare(args.question)
    else:
        r, total = run_one(args.question, serial=args.serial)
        print_report(r, total, serial=args.serial)


if __name__ == "__main__":
    main()
