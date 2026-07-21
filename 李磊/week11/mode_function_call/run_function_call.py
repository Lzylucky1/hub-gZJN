"""
run_function_call.py — Function Call（模型原生函数调用）

教学重点：
  1. 手写 JSON Schema：tools 的 schema 必须开发者手写，描述越清楚模型调用越准
  2. 多轮闭环：循环（模型输出 tool_call → 执行工具 → 结果回填）直到模型不再调工具，生成最终回答
  3. dispatch 表：工具名 → Python 函数，协议层与业务逻辑分离

使用方式：
  python mode_function_call/run_function_call.py --question "北京今天天气怎么样？"
  python mode_function_call/run_function_call.py --demo

依赖：pip install openai httpx
"""

import json
import sys
import time
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.weather_backend import geocode,get_weather_by_coords

# LLM 配置

PROVIDERS = {
    "deepseek": {
        "api_key": "sk-xxx",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "dashscope": {
        "api_key": "sk-xxx",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}

# 工具定义
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "geocode",
            "description": "查询指定城市的经纬度。城市用中文名，如 '北京'、'上海'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名，如 '北京'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_by_coords",
            "description": "根据经纬度查询当前天气及未来3天预报。lat 纬度如 39.90,lon 经度如 116.41",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number", "description": "纬度"},
                    "lon": {"type": "number", "description": "经度"},
                },
                "required": ["lat", "lon"],
            },
        },
    },
]

# 工具名 → 后端函数的映射表
TOOL_DISPATCH = {
    "geocode": geocode,
    "get_weather_by_coords":get_weather_by_coords,
}


SYSTEM_PROMPT = (
    "你是一名天气查询助手。根据用户问题，选择合适的工具来获取信息，"
    "必要时可多次调用不同工具组合完成。只依据工具返回的数据作答，不要编造。"
)

DEMO_QUESTIONS = [
    "北京今天天气怎么样？",
    "经纬度:39.90°N, 116.41°E,未来三天分别什么天气？",
    "上海和经纬度:39.90°N, 116.41°E今天的天气,分别是什么样的?",
]

def build_client(provider: str):
    """根据 provider 名创建 OpenAI 客户端和模型名"""
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]

def run(client, model: str, question: str, verbose: bool = True) -> dict:
    """提问 → 模型决定调工具 → 执行 → 回填 → 最终回答。"""

    # 拼装messages：初始记忆
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []
    round_count = 0

    # 第一次请求
    # model：调用的模型
    # messages：记忆，prompt
    # TOOLS_SCHEMA：工具信息
    # auto：自动判断是否调用工具
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
    )
    msg = resp.choices[0].message

    # 循环调用工具，直至退出
    while msg.tool_calls:
        # 保存这次tool_calls调用的上下文到记忆
        messages.append(msg)  

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args})

            if verbose:
                print(f"  → [tool] {name}({args})")

            # 查dispatch 表 → 调用工具(函数)
            fn = TOOL_DISPATCH.get(name)
            if fn is None:
                result = f"未知工具：{name}"
            else:
                try:
                    result = fn(**args)
                except TypeError as e:
                    result = f"参数错误：{e}"
                except Exception as e:
                    result = f"工具执行失败：{e}"

            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}\n")

            # 保存这次工具调用结果到记忆
            # tool：记忆类型，工具调用
            # tc.id：工具id，和msg上下文对应
            # result：调用结果
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        round_count += 1
        # 工具调用结束，继续下一轮调用
        if round_count <= 10:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
            )
            msg = resp.choices[0].message
        # 到达循环次数不使用工具，强制退出
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="none",
            )
            msg = resp.choices[0].message
            break

    # 输出结果
    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（{elapsed:.1f}s）")
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Function Call — 天气查询")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例问题集")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true", help="少输出")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    # 初始化
    client, model = build_client(args.provider)
    if not args.json:
        print(f"[Function Call] provider={args.provider} model={model}\n")

    # 确定问题列表
    questions = DEMO_QUESTIONS if args.demo else ([args.question] if args.question else [DEMO_QUESTIONS[0]])
    results = []

    # 逐个问题走闭环
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

    # 输出
    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


if __name__ == "__main__":
    main()
