"""
多轮对话 CLI 入口

使用方式：
  python src/chat.py
  python src/chat.py --mode fc
  python src/chat.py --mode manual --session my_session_001

交互命令：
  /new     开启新会话（清空记忆）
  /history 打印当前会话历史摘要
  /quit    退出
"""

import os
import sys
import argparse
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent))

from memory import ConversationMemory


def _collect_answer(run_generator):
    """从 ReAct 生成器中收集最终答案，并打印中间步骤。"""
    final_answer = ""
    for step in run_generator:
        stype = step["type"]
        if stype == "action":
            print(
                f"  [Step {step['step']}] {step['action']}({step['action_input']})"
            )
            print(f"    => {step['observation'][:200]}")
        elif stype == "final":
            final_answer = step["answer"]
            print(f"\n[Final] {final_answer}")
        elif stype in ("error", "max_steps"):
            final_answer = step.get("answer", step.get("observation", ""))
            print(f"\n[Error] {final_answer}")
    return final_answer


def main():
    parser = argparse.ArgumentParser(description="ReAct Financial Agent 多轮对话")
    parser.add_argument("--mode", choices=["manual", "fc"], default="manual",
                        help="manual=手写Prompt解析版  fc=Function Calling版")
    parser.add_argument("--session", default=None, help="会话 ID，不填则自动生成")
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()

    if args.mode == "manual":
        from react_manual import run as react_run
    else:
        from react_function_calling import run as react_run

    memory = ConversationMemory(session_id=args.session)
    print(f"\n当前会话 ID: {memory.session_id}")
    print(f"模式: {args.mode}")
    print("输入 /new 开启新会话，/history 查看历史，/quit 退出\n")

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("再见！")
            break

        if user_input == "/new":
            memory.clear()
            print("已开启新会话，历史记忆已清空。\n")
            continue

        if user_input == "/history":
            print("\n--- 当前会话历史摘要 ---")
            print(memory.get_recent_summary(n=10) or "暂无历史")
            print("---\n")
            continue

        print(f"Agent: 正在思考 ...\n")
        _collect_answer(react_run(user_input, max_steps=args.max_steps, memory=memory))
        print()


if __name__ == "__main__":
    main()
