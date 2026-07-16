"""
只需要修改run_function_call.py 中get_weather调用就行，具体代码如下
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.weather_backend import get_weather  # noqa: E402

PROVIDERS = {
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "dashscope": {
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}


def build_client(provider: str):
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气及未来3天预报。城市用中文名，如 '宁德'、'北京'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名，如 '宁德'"},
                },
                "required": ["city"],
            },
        },
    },
]


TOOL_DISPATCH = {
    "get_weather": get_weather,
}


SYSTEM_PROMPT = (
    "你是一名天气查询助手。回答用户关于天气的问题时，必须先调用 get_weather 工具获取实时天气数据，"
    "只依据工具返回的信息作答，不要编造数据。如果用户询问多个城市，逐个调用工具查询。"
)


def run(client, model: str, question: str, verbose: bool = True) -> dict:
    """
    循环闭环：提问 → 模型输出 tool_call → 执行 → 回填 → 重复直到无工具调用 → 最终回答。
    返回 {answer, tool_calls, elapsed}。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []
    round_num = 0

    while True:
        round_num += 1
        if verbose:
            print(f"  ── 第 {round_num} 轮 ──")
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            if verbose:
                print("  → 无工具调用，生成最终回答")
            break

        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args})
            if verbose:
                print(f"  → [tool] {name}({args})")
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
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（{elapsed:.1f}s，共{round_num}轮）")
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="天气查询循环调用演示")
    parser.add_argument("--city", "-c", help="指定单个城市查询天气")
    parser.add_argument("--question", "-q", help="自定义问题（支持多城市）")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互式持续提问")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true", help="少输出")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    if not args.quiet:
        print(f"[天气查询循环调用] provider={args.provider} model={model}\n")

    if args.interactive:
        print("进入交互式模式，输入问题查询天气（输入 quit 退出）\n")
        while True:
            question = input("你想查询哪个城市的天气？").strip()
            if question.lower() == "quit":
                print("退出程序")
                break
            if not question:
                continue
            print("=" * 60)
            result = run(client, model, question, verbose=not args.quiet)
            print("\n最终回答：")
            print(result["answer"])
            print()
    else:
        if args.question:
            question = args.question
        elif args.city:
            question = f"{args.city}的天气如何？"
        else:
            question = "北京的天气如何？"
        print("=" * 60)
        print(f"问题：{question}")
        print("=" * 60)
        result = run(client, model, question, verbose=not args.quiet)
        print("\n最终回答：")
        print(result["answer"])
        print()


if __name__ == "__main__":
    main()
