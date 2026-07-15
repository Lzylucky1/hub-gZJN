"""
function_call_loop.py — 作业核心：Function Call 多轮循环调用

改造点：
  1. 天气查询拆成 get_coordinates → get_weather_by_coords 两步
     （原项目 get_weather(city) 合二为一，单轮就够）
  2. LLM 闭环从"一次调用→执行→二次调用"改为 while 循环：
     每轮模型自己判断是继续调工具还是直接回答
  3. 安全保护：MAX_ROUNDS = 6 防止死循环

教学价值：
  - 链式调用天然需要多轮：模型先调 get_coordinates 拿到坐标，
    再根据结果调 get_weather_by_coords
  - 单轮闭环做不到"先查坐标再查天气"——这就是 Agent 循环的雏形

使用方式：
  python function_call_loop.py -q "宁德天气如何？"
  python function_call_loop.py --demo
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# ── 路径引导：引用参考项目的共享后端 ─────────────────────────────────────
# 参考项目中 src/rag_backend.py 需要 FAISS 索引 + DashScope Embedding，
# 是本作业依赖的外部基础设施，不重复实现。
_HERE = Path(__file__).resolve().parent
_PROJECT = (
    _HERE.parent
    / "week11 工具调用"
    / "week11 工具调用"
    / "function_call_mcp_cli"
)
sys.path.insert(0, str(_PROJECT))

from src.rag_backend import search_annual_report, list_companies  # noqa: E402

# 本作业的 src/weather_backend.py（拆成两函数）
from weather_backend_mod import get_coordinates, get_weather_by_coords  # noqa: E402

# ── LLM 配置 ───────────────────────────────────────────────────────────────

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


# ── 工具定义（JSON Schema） ────────────────────────────────────────────────
# 与原项目的差异：get_weather 拆为 get_coordinates + get_weather_by_coords，
# description 明确写了两者的依赖关系，引导模型链式调用。

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_annual_report",
            "description": (
                "在A股年报语料库中检索与问题最相关的段落。"
                "知识库仅收录 5 家公司：贵州茅台(600519)/五粮液(000858)/"
                "宁德时代(300750)/海康威视(002415)/中国平安(601318)，"
                "年份仅 2021/2022/2023。不在库内的公司请勿调用本工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "检索问题，自然语言。重要：不要包含公司名和年份"
                            "（已由 stock_code/year 参数过滤），只用简短财务术语，"
                            "例如 '营收和净利润'、'研发投入'、'主营业务'。"
                        ),
                    },
                    "stock_code": {
                        "type": "string",
                        "description": "可选，按公司过滤，如 '300750'。不传则跨公司检索",
                    },
                    "year": {
                        "type": "string",
                        "description": "可选，按年份过滤：'2021' / '2022' / '2023'",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回段落数，默认5，建议不超过10",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_companies",
            "description": "列出年报知识库中收录的所有公司、股票代码与可查年份。用于确认目标公司在库内。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_coordinates",
            "description": (
                "查询城市的经纬度坐标（纬度/经度）。"
                "拿到坐标后，再调用 get_weather_by_coords 查询该地天气。"
                "城市用中文名，如 '宁德'、'北京'。"
            ),
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
            "name": "get_weather_by_coords",
            "description": (
                "根据经纬度查询当前天气及未来3天预报。"
                "参数 latitude 和 longitude 应从 get_coordinates 的返回结果中获取，"
                "不要自行猜测坐标值。必须先调 get_coordinates 拿到坐标再调本工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "纬度，如 26.66。必须从 get_coordinates 的结果中获取，不可编造",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "经度，如 119.53。必须从 get_coordinates 的结果中获取，不可编造",
                    },
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "search_annual_report": search_annual_report,
    "list_companies": list_companies,
    "get_coordinates": get_coordinates,
    "get_weather_by_coords": get_weather_by_coords,
}


# ── 多轮循环调用核心逻辑 ──────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一名金融分析助手。回答用户关于A股年报的问题时，必须先调用 search_annual_report 工具检索年报原文，"
    "只依据工具返回的段落作答，不要编造数据。如果用户问的公司不在知识库"
    "（贵州茅台/五粮液/宁德时代/海康威视/中国平安），请明确告知不在库内，不要臆测。"
    "涉及天气时需分两步：先调 get_coordinates 查城市经纬度，"
    "再根据返回的坐标调 get_weather_by_coords 查天气。"
    "坐标值必须来自 get_coordinates 的实际返回结果，不要编造。"
    "你可以在多轮中逐步调用工具——看上一轮的结果决定下一轮要调什么。"
    "当所有信息都获取完毕后，直接给出最终回答，不要再调任何工具。"
)

MAX_ROUNDS = 6


def run(client, model: str, question: str, verbose: bool = True) -> dict:
    """
    多轮循环调用：
      提问 → LLM(工具定义) → tool_calls → 执行 → 回填 →
      LLM(再看结果) → 可能再调 → ... → 无 tool_calls → 最终回答

    返回 {answer, tool_calls, rounds, elapsed}。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []
    total_rounds = 0

    for rnd in range(1, MAX_ROUNDS + 1):
        total_rounds = rnd
        if verbose:
            print(f"\n── 第{rnd}轮 LLM 调用 ──")

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        # 终止条件：模型不再输出 tool_calls → 准备最终回答
        if not msg.tool_calls:
            if verbose:
                print("  ✓ 模型未输出工具调用，进入最终回答")
            break

        # 执行本轮所有工具调用
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args, "round": rnd})
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
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
    else:
        # for-else：达到 MAX_ROUNDS 上限未 break
        if verbose:
            print(f"\n  ⚠ 达到最大轮次({MAX_ROUNDS})，强制结束循环")

    answer = msg.content or ""
    if total_rounds >= MAX_ROUNDS and msg.tool_calls:
        answer += "\n\n[已达到最大推理轮次，可能未完全回答]"

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  → [llm] 最终回答（共{total_rounds}轮，{elapsed:.1f}s）")
    return {
        "answer": answer,
        "tool_calls": tool_call_log,
        "rounds": total_rounds,
        "elapsed": elapsed,
    }


# ── 入口 ───────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "宁德天气如何？",                                         # 链式：坐标→天气，需2轮
    "宁德时代2023年营收和净利润是多少？",                     # 单工具，1轮
    "宁德时代2023年的营收和净利润是多少？另外总部宁德的天气如何？",  # 混合：RAG + 链式天气
    "比亚迪2023年营收是多少？",                                # 幻觉控制：不在知识库
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="作业：Function Call 多轮循环调用")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例问题集")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true", help="少输出")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    if not args.json:
        print(f"[Function Call 循环调用] provider={args.provider} model={model}\n")

    questions = DEMO_QUESTIONS if args.demo else ([args.question] if args.question else [DEMO_QUESTIONS[0]])
    results = []
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

    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


if __name__ == "__main__":
    main()
