"""
Function Calling API 版 ReAct Agent

教学重点：
  1. tool_choice="auto" 让模型自己决定调用哪个工具或直接回答
  2. finish_reason 判断：tool_calls 表示继续调用，stop 表示给出最终答案
  3. 工具集通过 JSON Schema 定义，模型原生理解参数结构

使用方式：
  python react_function_calling.py
  python react_function_calling.py --question "茅台近一年股价涨跌幅如何？"
  python react_function_calling.py --question "..." --max_steps 8

依赖：
  pip install openai faiss-cpu sentence-transformers akshare
  export DASHSCOPE_API_KEY="sk-xxx"
"""

import os
import json
import time
import logging
import argparse
from typing import Generator

from openai import OpenAI

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# client = OpenAI(
#     api_key="sk-xxx",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
# MODEL = "qwen-max"
client = OpenAI(
    api_key="sk-xxx",
    base_url="https://api.deepseek.com",
)
MODEL = "deepseek-v4-pro"

FC_SYSTEM_PROMPT = """你是一个专业的A股金融分析助手。
规则：
- 调用 financial_indicator 或 stock_price 之前，必须先用 company_lookup 获取股票代码
- 数字计算必须使用 calculator 工具，不能心算
- Final Answer 必须引用具体数据来源
- 如果没有合适工具能回答，直接说明原因
"""

# 记忆
messages = []

def msg_size(msg) -> int:
    if isinstance(msg, dict):
        return len(str(msg))
    return len(str(msg.model_dump()))

def get_messages_size(messages:list) -> int:
    sum_size = 0
    for message in messages:
        sum_size += msg_size(message)
    return sum_size

def update_messages(messages: list, thr_size: int):
    cur_size = get_messages_size(messages)
    logger.info(f"📊 清理前记忆大小: {cur_size} 字符 | 阈值: {thr_size}")
    while len(messages) > 1 and cur_size > thr_size:
        # 先删除一个 user（一轮起点）
        cur_size -= msg_size(messages[1])
        del messages[1]

        # 删除后续的 assistant/tool
        while len(messages) > 1:
            # 下一个user对话跳过
            if isinstance(messages[1], dict) and messages[1]["role"] == "user":
                break

            # 删除一个记忆
            cur_size -= msg_size(messages[1])
            del messages[1]

        logger.info(f"🧹 清理 1 轮旧对话 | 剩余 {cur_size}/{thr_size} 字符")
    logger.info(f"📊 清理后记忆大小: {get_messages_size(messages)} 字符")


def run(question: str, max_steps: int = 10) -> Generator[dict, None, None]:
    """
    执行 Function Calling 版 ReAct 循环，yield 每一步结构化结果
    """
    from tools import TOOLS_MAP, TOOLS_SCHEMA

    # 清理久记忆
    update_messages(messages,10000)

    # 第一轮记忆追加系统提示词
    if not messages:  
        messages.append({"role": "system", "content": FC_SYSTEM_PROMPT})
    messages.append({"role": "user", "content": question})

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0,
        )
        msg    = response.choices[0].message
        reason = response.choices[0].finish_reason

        # 模型决定直接回答（无工具调用）
        if reason == "stop" or not msg.tool_calls:
            yield {
                "step":   step,
                "type":   "final",
                "thought": "",
                "answer": msg.content or "（模型返回空内容）",
            }
            messages.append(msg)
            return

        # 模型请求调用工具
        messages.append(msg)

        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            tool_fn = TOOLS_MAP.get(tool_name)
            if tool_fn is None:
                observation = f"未知工具 '{tool_name}'"
            else:
                try:
                    observation = tool_fn(**tool_args)
                except TypeError as e:
                    observation = f"工具参数错误: {e}"

            step_result = {
                "step":         step,
                "type":         "action",
                "thought":      "",   # Function Calling 版 Thought 在模型内部，不可见
                "action":       tool_name,
                "action_input": tool_args,
                "observation":  str(observation),
            }
            yield step_result

            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      str(observation),
            })

    yield {
        "step":   max_steps + 1,
        "type":   "max_steps",
        "answer": f"已达最大步数 {max_steps}，未能得出最终答案",
    }


# ── CLI 打印 ──────────────────────────────────────────────────────────────────

COLORS = {
    "thought": "\033[36m",
    "action":  "\033[33m",
    "obs":     "\033[32m",
    "final":   "\033[35m",
    "error":   "\033[31m",
    "reset":   "\033[0m",
}

def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def run_and_print(question: str, max_steps: int = 10):
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"模型: {MODEL}  实现: Function Calling")
    print('='*60)

    start = time.time()

    for step_data in run(question, max_steps=max_steps):
        stype = step_data["type"]

        if stype == "action":
            print(f"\n[Step {step_data['step']}]")
            # Thought 在 FC 版不可见，显示提示
            print(_c("thought", "🧠 Thought: （模型内部推理，Function Calling 版不可见）"))
            print(_c("action",  f"🔧 Action:  {step_data['action']}"))
            print(_c("action",  f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
            print(_c("obs",     f"👁  Obs:     {step_data['observation'][:300]}"))

        elif stype == "final":
            elapsed = time.time() - start
            print(f"\n{'─'*60}")
            print(_c("final", f"\n✅ Final Answer:\n{step_data['answer']}"))
            print(f"\n共 {step_data['step']} 步，耗时 {elapsed:.1f}s")

        elif stype in ("error", "max_steps"):
            print(_c("error", f"\n⚠️  {step_data.get('answer', '')}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question",  default="贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？")
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()
    run_and_print(args.question, args.max_steps)
