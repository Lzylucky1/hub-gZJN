"""
run_function_call.py — 方式一：Function Call（模型原生函数调用）

教学重点：
  1. 手写 JSON Schema：每个工具的 name/description/parameters 都要开发者自己写
     ——这是 Function Call 的"接入成本"，schema 写得越清楚，模型调用越准
  2. 链式工具调用：模型输出 tool_call → 宿主执行工具 → 结果以 role=tool 回填 → 
     模型可能再次输出 tool_call（如先查经纬度再查天气）→ 循环直到模型不再调用工具
  3. 工具名 → 后端函数的 dispatch 表：业务逻辑（src/）与协议层（本文件）彻底分离

使用方式：
  # 配置环境变量
  #   Windows:  set DEEPSEEK_API_KEY=sk-xxx & set DASHSCOPE_API_KEY=sk-xxx
  #   Linux:    export DEEPSEEK_API_KEY=sk-xxx; export DASHSCOPE_API_KEY=sk-xxx

  # 单个问题
  python run_function_call.py --question "银川今天天气怎么样？"

依赖：
  pip install openai
  环境变量：DASHSCOPE_API_KEY（Embedding，rag_backend 内部用）
            DEEPSEEK_API_KEY（默认 LLM；可在 --provider dashscope 切到 qwen-plus）
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# 把项目根目录加入 sys.path，让 weather_backend 可被导入
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入后端业务函数：get_location（城市名→经纬度）、get_weather（经纬度→天气）
from weather_backend import get_weather, get_location  # noqa: E402

# API Key 配置（实际使用时应从环境变量读取）
DASHSCOPE_API_KEY = "DEEPSEEK_API_KEY"
DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY"

# LLM 提供商配置：支持 deepseek 和 dashscope 两种后端
PROVIDERS = {
    "deepseek": {
        "api_key": DEEPSEEK_API_KEY,
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "dashscope": {
        "api_key": DASHSCOPE_API_KEY,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}


def build_client(provider: str):
    """构建指定提供商的 OpenAI 兼容客户端"""
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


# ── 【教学时刻 1】：手写工具的 JSON Schema ──────────────────────────────────
# Function Call 的核心接入成本：每个工具的参数 schema 必须开发者手写。
# description 直接决定模型"什么时候调这个工具、传什么参数"——写得越具体越准。
# 
# 注意：get_location 的 coordinates_only 参数未在这里暴露，因为它是内部控制参数，
# 模型调用时永远只传 city 参数，coordinates_only=True 由后端函数内部决定，
# 这样返回的结果更精简，模型更容易提取经纬度传给 get_weather。
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_location",
            "description": "查询指定城市的经纬度。城市用中文名，如 '宁德'、'北京'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名，如 '宁德'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "根据经纬度查询当前天气及未来3天预报。需要先调用 get_location 获取经纬度。",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "纬度"},
                    "longitude": {"type": "number", "description": "经度"},
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
]


# ── 【教学时刻 2】：工具名 → 后端函数的 dispatch 表 ─────────────────────────
# 业务逻辑在 weather_backend.py，本文件只负责"协议层"——把模型生成的 tool_call 
# 派发给对应的后端函数执行。
# 
# 新增工具只需：1) 在上面 TOOLS_SCHEMA 写 schema；2) 在这里加一行映射。
# 注意：get_location 调用时固定传入 coordinates_only=True，确保返回精简结果。
TOOL_DISPATCH = {
    "get_location": lambda city: get_location(city, coordinates_only=True),
    "get_weather": get_weather,
}


# 系统提示词：引导模型正确使用工具的关键指令
SYSTEM_PROMPT = (
    "你是一名天气查询助手。回答用户关于天气的问题时，必须先调用 get_location 工具查询城市的经纬度，"
    "然后使用返回的经纬度调用 get_weather 工具查询天气。如果用户只给出城市名，不要直接回答天气，"
    "必须先查询经纬度。"
)


def run(client, model: str, question: str, verbose: bool = True) -> dict:
    """
    单轮闭环：提问 → 模型输出 tool_call → 执行 → 回填 → 循环直到模型不再调用工具。
    返回 {answer, tool_calls, elapsed} 用于对比器汇总。
    """
    # 初始化对话历史：系统提示词 + 用户问题
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []

    # ── 【教学时刻 3】：链式工具调用的 while 循环 ────────────────────────────
    # 为什么用 while 循环而非 if 判断？
    # 因为工具调用可能需要多轮：例如用户问"银川天气怎么样"，
    # 第一轮：模型调用 get_location("银川") → 返回经纬度
    # 第二轮：模型看到经纬度后调用 get_weather(lat, lon) → 返回天气
    # 第三轮：模型看到天气后不再调用工具，生成最终回答
    # 
    # while 循环的终止条件：模型返回的消息中没有 tool_calls（即 msg.tool_calls 为 None 或空列表）
    while True:
        # 调用 LLM，传入对话历史和工具定义，让模型决定是否调用工具
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",  # auto：模型自主决定是否调用工具及调用哪个
        )
        msg = resp.choices[0].message

        # 如果模型没有输出 tool_calls，说明已经可以生成最终回答，退出循环         # 模型输出的工具调用指令
        if not msg.tool_calls:
            break

        # 把带 tool_calls 的消息原样回填到对话历史，保持上下文完整
        messages.append(msg)

        # 逐个执行模型输出的 tool_call（支持并行工具调用，即一次输出多个）
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args})
            
            if verbose:
                print(f"  → [tool] {name}({args})")
            
            # 根据工具名查找对应的后端函数
            fn = TOOL_DISPATCH.get(name)
            if fn is None:
                result = f"未知工具：{name}"
            else:
                try:
                    # 执行工具函数！
                    result = fn(**args)
                except TypeError as e:
                    result = f"参数错误：{e}"
                except Exception as e:
                    result = f"工具执行失败：{e}"
            
            # 打印工具执行结果预览（前120字符）
            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}\n")
            
            # 以 role=tool 把工具结果回填到对话历史，tool_call_id 必须与上面的调用对应
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # 循环结束后，msg.content 就是模型的最终回答
    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（{elapsed:.1f}s）")
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}

DEMO_QUESTIONS = [
    "杭州天气情况",
    "银川的经纬度",
    "银川今天的天气情况",
]



def main():
    """命令行入口"""
    import argparse
    parser = argparse.ArgumentParser(description="方式一：Function Call")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例问题集")
    parser.add_argument("--provider", default="dashscope", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true", help="少输出（被 compare.py 调用时用）")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供 compare.py 解析）")
    args = parser.parse_args()

    if not args.question:
        args.question = "银川今天天气这么样"

    # 构建客户端
    client, model = build_client(args.provider)
    if not args.json:
        print(f"[Function Call] provider={args.provider} model={model}\n")

    questions = DEMO_QUESTIONS if args.demo else ([args.question] if args.question else [DEMO_QUESTIONS[0]])

    results = []
    
    # 逐个处理问题
    for i, q in enumerate(questions, 1):
        if not args.json:
            print("=" * 60)
            print(f"Q{i}：{q}")
            print("=" * 60)
        result = run(client, model, q, verbose=not (args.quiet or args.json))
        result["question"] = q
        results.append(result)
        if not args.json:
            print("\n最终回答：")
            print(result["answer"])
            print()

    # JSON 格式输出（供 compare.py 解析）
    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


if __name__ == "__main__":
    main()
