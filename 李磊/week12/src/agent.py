"""
ReAct Agent 统一入口（Function Calling 版）

使用方式：
  python agent.py
  python agent.py --question "五粮液近一年股价涨跌幅？"
  python agent.py --question "..." --max_steps 8

环境变量：
  DASHSCOPE_API_KEY  必填（用于 RAG embedding）
  DEEPSEEK_API_KEY   必填（用于 LLM 调用）
  AGENT_MODEL        默认 deepseek-v4-flash
"""

import os
import argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DEFAULT_QUESTION = "贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReAct Financial Agent (Function Calling)")
    parser.add_argument("--question",  default=DEFAULT_QUESTION)
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()

    from react_function_calling import run_and_print

    run_and_print(args.question, args.max_steps)
