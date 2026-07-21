import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent
CLI_DIR = Path(__file__).parent / "cli"
PY = sys.executable

# fincli 真实命令路径：优先用 pip install -e . 注册到 PATH 的 fincli；
# 没装就退回 python mode_cli/cli/main.py（保证不安装也能跑，只是命令不"漂亮"）
_FINCLI = shutil.which("fincli") or None
FINCLI_ARGV = ["fincli"] if _FINCLI else [PY, str(CLI_DIR / "main.py")]
FINCLI_LABEL = "fincli" if _FINCLI else "python mode_cli/cli/main.py"

# ── LLM 配置（与另两方式一致）──────────────────────────────────────────────

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
    """
    根据 provider 名称构建 LLM 客户端（与另两种方式一致）。
    
    Args:
        provider: "deepseek" 或 "dashscope"
    
    Returns:
        (OpenAI 客户端实例, 模型名称字符串)
    """
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


# ── 形态 A：具名 run_cli ───────────────────────────────────────────────────
# 白名单 enum 限定可执行命令集——这是"安全"的来源：模型只能调预先批准的命令。
# 底层统一走 fincli（一条真实命令），而非 python xxx.py，更接近真实 CLI 工具形态。

# command 名 → 实际执行的 argv 模板（参数由 LLM 通过 args JSON 提供）
NAMED_COMMANDS = {
    "rag_search": {
        "argv": FINCLI_ARGV + ["search"],
        "arg_map": {  # LLM args JSON 的 key → fincli flag
            "query": "--query",
            "stock_code": "--stock-code",
            "year": "--year",
            "top_k": "--top-k",
        },
    },
    "rag_list_companies": {
        "argv": FINCLI_ARGV + ["list-companies"],
        "arg_map": {},
    },
    "get_city_position": {
        "argv": FINCLI_ARGV + ["get-city-position"],
        "arg_map": {"city": "--city"},
    },
    "get_city_weather": {
        "argv": FINCLI_ARGV + ["get-city-weather"],
        "arg_map": {"city": "--city"},
    },
}

# 白名单命令
def run_named(command: str, args: dict) -> str:
    """
    形态 A：按白名单拼出 argv，子进程执行，返回 stdout。
    
    安全机制：
      - command 必须在 NAMED_COMMANDS 白名单内
      - 参数通过 arg_map 映射为命令行 flag，不接受任意参数
    
    Args:
        command: 命令名称（如 "rag_search"、"weather"）
        args: 参数字典（如 {"query": "营收", "stock_code": "300750"}）
    
    Returns:
        命令的 stdout 输出字符串；执行失败返回错误信息
    """
    spec = NAMED_COMMANDS.get(command)
    if spec is None:
        return f"[run_cli] 未知命令：{command}（白名单：{list(NAMED_COMMANDS)})"

    argv = list(spec["argv"])
    for key, flag in spec["arg_map"].items():
        val = args.get(key)
        if val is not None:
            argv.extend([flag, str(val)])

    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=30,
            cwd=str(BASE_DIR), env={**os.environ},
        )
    except subprocess.TimeoutExpired:
        return "[run_cli] 命令执行超时（>30s）"
    if proc.returncode != 0:
        return f"[run_cli] 命令失败（code={proc.returncode}）：{proc.stderr[-500:]}"
    return proc.stdout


# ── 形态 B：通用 run_bash（沙箱）──────────────────────────────────────────
# 模型自己拼 shell 命令字符串——最灵活也最危险，沙箱是教学重点。

# 危险命令黑名单（正则，命中即拒绝执行）
DANGEROUS_PATTERNS = [
    r"\brm\b", r"\bdel\b", r"\brmdir\b", r"\bdeltree\b",
    r"\bformat\b", r"\bmkfs\b", r"\bdd\b",
    r"\bshutdown\b", r"\breboot\b", r"\bpoweroff\b",
    r"[>;]\s*(?:rm|del|format)\b",          # 重定向/链式后的删除
    r"\bcurl\b.*\|\s*sh",                    # curl pipe to shell（远程执行）
    r"\bwget\b.*\|\s*sh",
    r"\bsudo\b", r"\bchmod\b.*-R", r"\bchown\b.*-R",
    r"\bnc\b", r"\bnetcat\b",                # 反弹 shell 常用
    r"/etc/passwd", r"/etc/shadow",
    r"\bTaskkill\b", r"\bStop-Process\b",    # Windows 杀进程
]

# 命令白名单：只允许这些可执行文件作为命令头（其余拒绝）
# 形态 B 仍要危险可控：只放行 fincli（本项目工具）+ python + 几个只读命令
ALLOWED_HEADS = {"fincli", "python", "python3", "py", "git", "ls", "dir", "cat", "echo", "type"}

#  sandbox_check() + run_bash() 沙箱机制
def sandbox_check(command: str) -> str | None:
    """
    沙箱安全检查：验证命令是否允许执行。
    
    检查规则：
      1. 危险命令黑名单（正则匹配）：rm/del/format/sudo/curl|sh 等
      2. 可执行文件白名单：只允许 fincli/python/git/ls/cat/echo 等
    
    Args:
        command: 待检查的 shell 命令字符串
    
    Returns:
        None 表示通过检查；字符串表示拒绝原因
    """
    for pat in DANGEROUS_PATTERNS:
        if re.search(pat, command, re.IGNORECASE):
            return f"沙箱拦截：命中危险模式 {pat!r}"
    # 解析命令头：取第一个 token 的文件名
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return "沙箱拦截：命令解析失败"
    if not tokens:
        return "沙箱拦截：空命令"
    head = Path(tokens[0]).name.lower()
    if head not in ALLOWED_HEADS:
        return f"沙箱拦截：{tokens[0]!r} 不在白名单 {sorted(ALLOWED_HEADS)} 中"
    return None


def run_bash(command: str) -> str:
    """
    形态 B：模型生成的 shell 命令，经沙箱检查后在锁定工作目录执行。
    
    安全机制：
      - 先调用 sandbox_check 检查危险模式和白名单
      - 工作目录锁定在项目根（cwd=BASE_DIR）
      - 超时 15s 防止死循环
      - shell=True 让模型可以用管道/重定向
    
    Args:
        command: 完整的 shell 命令字符串
    
    Returns:
        命令的 stdout 输出；被拦截或失败返回错误信息
    """
    blocked = sandbox_check(command)
    if blocked:
        return f"[run_bash] {blocked}"

    try:
        # shell=True 让模型可以用管道/重定向；工作目录锁在项目根；
        # 超时 15s 防止死循环；不继承会话的交互式特性
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=15,
            cwd=str(BASE_DIR), env={**os.environ},
        )
    except subprocess.TimeoutExpired:
        return "[run_bash] 命令执行超时（>15s）"
    out = proc.stdout
    if proc.returncode != 0:
        out += f"\n[run_bash] 退出码 {proc.returncode}，stderr：{proc.stderr[-300:]}"
    return out


# ── 两种形态各自的 tools schema ───────────────────────────────────────────

NAMED_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_cli",
            "description": (
                "执行预批准的命令行工具。command 只能取白名单内的值。"
                "可查 A 股年报（rag_search/list_companies）、位置（get_city_position）和天气（get_city_weather）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": list(NAMED_COMMANDS.keys()),
                        "description": "rag_search（查年报，需 query+可选 stock_code/year/top_k）/"
                                       " rag_list_companies（列公司）/"
                                       " get_city_position（查位置，需 city）/"
                                       " get_city_weather（查天气+位置，需 city）",
                    },
                    "args": {
                        "type": "object",
                        "description": "命令参数。rag_search: {query, stock_code?, year?, top_k?}; get_city_position/get_city_weather: {city}",
                    },
                },
                "required": ["command"],
            },
        },
    },
]

BASH_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "在沙箱里执行一条 shell 命令并返回 stdout。"
                "可用工具 fincli（一条真实命令）："
                "fincli search --query '营收和净利润' --stock-code 300750 --year 2023 --top-k 3；"
                "fincli list-companies；"
                "fincli get-city-position --city 宁德；"
                "fincli get-city-weather --city 宁德。"
                "危险命令（rm/del/format/sudo/curl|sh 等）会被拦截；只允许白名单可执行文件。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "完整的 shell 命令字符串"},
                },
                "required": ["command"],
            },
        },
    },
]

# 形态 → (schema, executor)
MODE_DISPATCH = {
    "named": (NAMED_TOOLS_SCHEMA, lambda args: run_named(args["command"], args.get("args", {}))),
    "bash": (BASH_TOOLS_SCHEMA, lambda args: run_bash(args["command"])),
}


# ── 单轮闭环 ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT_NAMED = (
    "你是一名金融分析助手。通过 run_cli 工具调用预批准命令查 A 股年报与天气。"
    "回答年报问题前必须先 run_cli(command='rag_search', args={...}) 检索原文，只依据返回段落作答，不要编造。"
    "知识库仅含：贵州茅台(600519)/五粮液(000858)/宁德时代(300750)/海康威视(002415)/中国平安(601318)，年份 2021-2023。"
    "rag_search 的 query 不要含公司名/年份（已由 stock_code/year 过滤），用简短术语如 '营收和净利润'。"
    "不在库内的公司请明确告知，不要臆测。"
    "涉及位置/天气时：只要位置用 get_city_position；要天气用 get_city_weather。"
    "你可以多次调用工具，直到收集够信息再回答。"
)

SYSTEM_PROMPT_BASH = (
    "你是一名金融分析助手。通过 run_bash 工具在沙箱里执行 fincli 命令查 A 股年报与天气。"
    "查年报：fincli search --query '营收和净利润' --stock-code 300750 --year 2023 --top-k 3"
    "（query 不要含公司名/年份，用简短财务术语）。"
    "列公司：fincli list-companies。"
    "查位置：fincli get-city-position --city 宁德。"
    "查天气：fincli get-city-weather --city 宁德。"
    "回答必须依据命令返回的原文，不要编造。知识库仅含 5 家公司（茅台/五粮液/宁德时代/海康威视/中国平安），"
    "不在库内的明确告知。你可以多次调用工具，直到收集够信息再回答。"
)


def run(client, model: str, question: str, mode: str, verbose: bool = True) -> dict:
    """
    单轮闭环：提问 → 模型输出 tool_call → 执行命令 → 回填 → 最终回答。
    
    与 Function Call / MCP 的差异（教学时刻 3）：
      - 工具执行：通过 subprocess 调用 CLI 命令，非直接调后端函数或 MCP 协议
      - 两种形态：
        * named: 白名单命令，安全可控
        * bash: 模型自己拼 shell 命令，需沙箱保护
    
    Args:
        client: OpenAI 客户端实例
        model: 模型名称字符串
        question: 用户问题
        mode: "named" 或 "bash"
        verbose: 是否打印中间过程
    
    Returns:
        {answer: 最终回答, tool_calls: 工具调用日志列表, elapsed: 总耗时秒数}
    """
    tools_schema, executor = MODE_DISPATCH[mode]
    sys_prompt = SYSTEM_PROMPT_NAMED if mode == "named" else SYSTEM_PROMPT_BASH

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []

    resp = client.chat.completions.create(
        model=model, messages=messages, tools=tools_schema, tool_choice="auto",
    )
    msg = resp.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": tc.function.name, "args": args})
            if verbose:
                print(f"  → [{mode}] {tc.function.name}({args})")
            try:
                result = executor(args)
            except Exception as e:
                result = f"[{mode}] 执行异常：{e}"
            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}\n")
            messages.append({
                "role": "tool", "tool_call_id": tc.id, "content": result,
            })

        resp = client.chat.completions.create(
            model=model, messages=messages, tools=tools_schema, tool_choice="auto",
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
      - --mode named: 白名单命令（默认）
      - --mode bash: 通用 shell 命令（需沙箱）
      - 单个问题：--question "..."
      - 内置示例集：--demo
      - JSON 输出：--json（供 compare.py 解析）
    """
    import argparse
    parser = argparse.ArgumentParser(description="方式三：CLI")
    parser.add_argument("--mode", default="named", choices=["named", "bash"])
    parser.add_argument("--question", "-q")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供 compare.py 解析）")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    if not args.json:
        print(f"[CLI/{args.mode}] provider={args.provider} model={model}\n", file=sys.stderr)

    questions = DEMO_QUESTIONS if args.demo else ([args.question] if args.question else [DEMO_QUESTIONS[0]])
    results = []
    for i, q in enumerate(questions, 1):
        if not args.json:
            print("=" * 60)
            print(f"Q{i}：{q}")
            print("=" * 60)
        result = run(client, model, q, args.mode, verbose=not (args.quiet or args.json))
        result["question"] = q
        result["mode"] = args.mode
        results.append(result)
        if not args.json:
            print("\n最终回答：")
            print(result["answer"])
            print()

    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


if __name__ == "__main__":
    main()
