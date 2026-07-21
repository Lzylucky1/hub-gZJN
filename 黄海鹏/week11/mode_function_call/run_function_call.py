import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# 把项目根目录加入 sys.path，让 src 可 import（直接 python 运行本脚本也能找到）
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_backend import search_annual_report, list_companies  # noqa: E402
from src.weather_backend import get_weather, get_positioning, get_positioning_str  # noqa: E402

# ── LLM 配置 ───────────────────────────────────────────────────────────────

PROVIDERS = {
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",  # 即 deepseek-v4-flash
    },
    "dashscope": {
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}

# 构建 LLM 客户端
def build_client(provider: str):
    """
    根据 provider 名称构建 LLM 客户端。
    
    Args:
        provider: "deepseek" 或 "dashscope"
    
    Returns:
        (OpenAI 客户端实例, 模型名称字符串)
    
    注意：如果对应的 API Key 环境变量未设置，直接退出程序。
    """
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


# ── 【教学时刻 1】：手写工具的 JSON Schema ──────────────────────────────────
# Function Call 的核心接入成本：每个工具的参数 schema 必须开发者手写。
# description 直接决定模型"什么时候调这个工具、传什么参数"——写得越具体越准。

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
                            "把公司名写进 query 会稀释检索精度。"
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
            "name": "get_city_position",
            "description": "查询指定城市的位置信息（经纬度、行政区）。只返回位置，不返回天气。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名，如 '宁德'、'北京'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_city_weather",
            "description": "查询指定城市的当前天气及未来3天预报。返回位置+天气完整信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名，如 '宁德'、'北京'"},
                },
                "required": ["city"],
            },
        },
    },
]

# ── 【教学时刻 2】：工具名 → 后端函数的 dispatch 表 ─────────────────────────
# 业务逻辑在 src/，本文件只负责"协议层"——把模型生成的 tool_call 派发给后端函数。
# 新增工具只需：1) 在上面写 schema；2) 在这里加一行映射。这是 Function Call 的扩展方式。

def _get_city_weather(city: str) -> str:
    """封装函数：先查位置，再查天气，返回完整信息。"""
    position = get_positioning(city)
    if isinstance(position, str):  # 错误信息
        return position
    return get_weather(position)


TOOL_DISPATCH = {
    "search_annual_report": search_annual_report,
    "list_companies": list_companies,
    "get_city_position": get_positioning_str,
    "get_city_weather": _get_city_weather,
}


# ── 单轮闭环 ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一名金融分析助手。回答用户关于A股年报的问题时，必须先调用 search_annual_report 工具检索年报原文，"
    "只依据工具返回的段落作答，不要编造数据。如果用户问的公司不在知识库"
    "（贵州茅台/五粮液/宁德时代/海康威视/中国平安），请明确告知不在库内，不要臆测。"
    "涉及天气时：如果只要位置信息，调用 get_city_position；如果要天气+位置，调用 get_city_weather。"
    "你可以多次调用工具，直到收集够信息再回答。"
)


def run(client, model: str, question: str, verbose: bool = True) -> dict:
    """
    单轮闭环：提问 → 模型输出 tool_call → 执行 → 回填 → 最终回答。
    
    核心流程（教学时刻 3）：
      1. 第一次 LLM 请求：带上 tools schema，模型决定是否调用工具
      2. 如果模型返回 tool_calls：逐个执行后端函数，以 role=tool 回填结果
      3. 第二次 LLM 请求：模型看到工具结果，生成最终自然语言回答
    
    Args:
        client: OpenAI 客户端实例
        model: 模型名称字符串
        question: 用户问题
        verbose: 是否打印中间过程
    
    Returns:
        {answer: 最终回答, tool_calls: 工具调用日志列表, elapsed: 总耗时秒数}
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []
    rounds = 0

    # 第一次请求：带上 tools，让模型决定是否调用工具
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
    )
    msg = resp.choices[0].message

    # 【教学时刻 3】：模型输出了 tool_calls → 逐个执行后端函数
    if msg.tool_calls:
        # 把 assistant 这条带 tool_calls 的消息原样回填，保持上下文
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
                    # 工具执行！！
                    result = fn(**args)
                except TypeError as e:
                    result = f"参数错误：{e}"
                except Exception as e:
                    result = f"工具执行失败：{e}"
            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}\n")
            # 以 role=tool 把每个工具的结果回填，tool_call_id 必须对上
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        # 第二次请求：模型看到工具结果，生成最终回答（不再调用工具）
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（{elapsed:.1f}s）")
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}


# ── 入口 ───────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "宁德时代2023年营收和净利润是多少？",
    "宁德时代总部在哪个城市？它的经纬度是多少？",
    "宁德时代总部所在城市的天气如何？",
    "宁德时代2023年营收是多少？另外总部天气如何？",
    "比亚迪2023年营收是多少？",  # 幻觉控制
]


def main():
    """
    CLI 入口：解析命令行参数，构建 LLM 客户端，执行问题并输出结果。
    
    支持模式：
      - 单个问题：--question "..."
      - 内置示例集：--demo（跑 4 个预设问题，含并行调用和幻觉控制）
      - JSON 输出：--json（供 compare.py 解析）
    """
    import argparse
    parser = argparse.ArgumentParser(description="方式一：Function Call")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例问题集")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true", help="少输出（被 compare.py 调用时用）")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供 compare.py 解析）")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    if not args.json:
        print(f"[Function Call] provider={args.provider} model={model}\n")

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
        # 单问题输出单对象；demo 输出数组
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


if __name__ == "__main__":
    main()
