"""
多轮对话能力评估脚本

功能：
  1. 构造一个跨轮指代的对话序列（上下文依赖）
  2. 使用手写 Prompt 解析版逐个问题运行
  3. 验证 Agent 是否能利用前文提到的股票代码 / 公司名称回答后续问题
  4. 将结果保存为 JSON，供作业提交

使用方式：
  python src/evaluate_multi_turn.py
  python src/evaluate_multi_turn.py --output outputs/logs/multi_turn_result.json
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent))

from memory import ConversationMemory
from react_manual import run as manual_run


# 多轮对话测试集：刻意包含指代和省略，考验上下文记忆
MULTI_TURN_DIALOGUES = [
    {
        "id": "MT1",
        "description": '跨轮指代：前文提到两家公司，后文问"它"的毛利率',
        "turns": [
            "贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？",
            "那净利率呢？",
            "它们的股价去年表现如何？",
        ],
    },
    {
        "id": "MT2",
        "description": "跨轮省略：前文提到股票代码，后文直接省略公司名",
        "turns": [
            "宁德时代2023年的营业收入是多少？",
            "2022年呢？",
            "它的净利润增长了多少？",
        ],
    },
    {
        "id": "MT3",
        "description": "话题切换：从公司 A 切换到公司 B，再切回 A",
        "turns": [
            "中国平安的资产负债率是多少？",
            "海康威视2023年有哪些风险因素？",
            "再回到中国平安，它的ROE近3年趋势如何？",
        ],
    },
]


def _run_dialogue(dialogue: dict, max_steps: int = 10) -> dict:
    """运行一个完整的多轮对话，返回每轮结果。"""
    memory = ConversationMemory(session_id=dialogue["id"])
    results = []
    total_time = 0.0

    print(f"\n{'='*60}")
    print(f"对话 {dialogue['id']}: {dialogue['description']}")
    print(f"{'='*60}")

    for turn_idx, question in enumerate(dialogue["turns"], 1):
        print(f"\n[Turn {turn_idx}] User: {question}")
        start = time.time()

        final_answer = ""
        action_steps = []

        for step_data in manual_run(question, max_steps=max_steps, memory=memory):
            stype = step_data["type"]
            if stype == "action":
                action_steps.append({
                    "step": step_data["step"],
                    "action": step_data["action"],
                    "action_input": step_data["action_input"],
                    "observation": step_data["observation"][:200],
                })
            elif stype == "final":
                final_answer = step_data["answer"]
            elif stype in ("error", "max_steps"):
                final_answer = step_data.get("answer", step_data.get("observation", ""))

        elapsed = time.time() - start
        total_time += elapsed

        print(f"[Turn {turn_idx}] Agent: {final_answer[:200]}")
        print(f"  步骤数: {len(action_steps)}  耗时: {elapsed:.1f}s")

        results.append({
            "turn": turn_idx,
            "question": question,
            "answer": final_answer,
            "steps": len(action_steps),
            "elapsed_s": round(elapsed, 1),
        })

    return {
        "id": dialogue["id"],
        "description": dialogue["description"],
        "total_elapsed_s": round(total_time, 1),
        "turns": results,
        "session": memory.to_dict(),
    }


def evaluate(output_path: str | None = None, max_steps: int = 10):
    all_results = []

    for dialogue in MULTI_TURN_DIALOGUES:
        result = _run_dialogue(dialogue, max_steps=max_steps)
        all_results.append(result)

    # 汇总
    print(f"\n{'='*60}")
    print("多轮对话评估汇总")
    print(f"{'='*60}")
    print(f"{'ID':<6} {'Turns':<8} {'Total(s)':<12}")
    print("-" * 30)
    for r in all_results:
        print(f"{r['id']:<6} {len(r['turns']):<8} {r['total_elapsed_s']:<12}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存至 {output_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/logs/multi_turn_result.json",
                        help="保存 JSON 结果的路径")
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()
    evaluate(args.output, args.max_steps)
