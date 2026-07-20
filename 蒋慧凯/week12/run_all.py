"""
week12 作业统一入口：运行多轮对话能力评估

使用方式：
  python run_all.py
  python run_all.py --output outputs/logs/multi_turn_result.json
"""

import os
import sys
import argparse
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluate_multi_turn import evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/logs/multi_turn_result.json",
                        help="评估结果保存路径")
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()

    print("=" * 60)
    print("Week12 作业：为 ReAct Agent 增加多轮对话能力")
    print("=" * 60)
    print(f"输出目录: {args.output}")
    print(f"最大步数: {args.max_steps}\n")

    evaluate(args.output, args.max_steps)

    print("\n" + "=" * 60)
    print("评估完成。请检查 outputs/ 目录下的日志和结果。")
    print("=" * 60)
