
import re
import time
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
MODEL = "qwen2-0.5b"

# ── 任务 1：日期标准化 ────────────────────────────────────────────────
DATE_REGEX = r"\d{4}-\d{2}-\d{2}"
DATE_SYSTEM = "你是日期抽取助手。从用户输入中抽取日期，严格用 YYYY-MM-DD 格式输出，不输出任何其他文字。"
DATE_CASES = [
    "2024年5月12日开服",
    "2023/12/1 上线新活动",
    "三月三号我去打副本",
    "2024.11.30 是赛季截止日期",
    "明天（假设今天是2026-05-11）",
    "2024 年 10 月的第一天有双倍经验",
]

# ── 任务 2：游戏区服代码抽取 ──────────────────────────────────────────
SERVER_REGEX = r"S\d+"
SERVER_SYSTEM = "你是游戏区服抽取助手。从用户输入中找到区服代码（S 加数字），直接输出代码，不输出任何其他文字。"
SERVER_CASES = [
    "我在 S1 服务器",
    "S10 开新区了快来",
    "S250 的兄弟们集合",
    "我是S 999服的",
    "S5 服务器攻城战开始了",
    "有没有S20服的朋友",
]


def run_generate(system: str, user: str, regex: str | None = None) -> tuple[str, float]:
    t0 = time.time()
    extra = {"guided_regex": regex} if regex else {}
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=30,
        extra_body=extra,
    )
    return resp.choices[0].message.content.strip(), time.time() - t0


def matches(pattern: str, text: str) -> bool:
    return bool(re.fullmatch(pattern, text))


def run_section(title: str, system: str, regex: str, cases: list[str]):
    print("=" * 70)
    print(f"  {title}")
    print(f"  正则: {regex}")
    print("=" * 70)
    raw_ok = 0
    guided_ok = 0
    print(f"\n{'输入':<35}{'裸 prompt':<25}{'guided_regex':<15}")
    print("-" * 75)
    for user in cases:
        raw_out, _ = run_generate(system, user)
        guided_out, _ = run_generate(system, user, regex)
        raw_match = matches(regex, raw_out)
        guided_match = matches(regex, guided_out)
        if raw_match:
            raw_ok += 1
        if guided_match:
            guided_ok += 1
        flag_raw = "✓" if raw_match else "✗"
        flag_guided = "✓" if guided_match else "✗"
        # 截断超长输出
        raw_disp = raw_out[:22] + "…" if len(raw_out) > 22 else raw_out
        print(f"{user:<33}  {flag_raw} {raw_disp:<20}  {flag_guided} {guided_out}")
    n = len(cases)
    print("-" * 75)
    print(f"格式合法率：裸 prompt {raw_ok}/{n} ({100*raw_ok/n:.0f}%)  |  "
          f"guided_regex {guided_ok}/{n} ({100*guided_ok/n:.0f}%)\n")


def main():
    run_section("任务 1：日期标准化 → YYYY-MM-DD",
                DATE_SYSTEM, DATE_REGEX, DATE_CASES)
    run_section("任务 2：游戏区服代码抽取 → S 加数字",
                SERVER_SYSTEM, SERVER_REGEX, SERVER_CASES)

    print("=" * 70)
    print("  结论：guided_regex 保证下游解析器永远能拿到合法输入")
    print("       特别适合日期/区服代码/角色ID等有严格格式的字段")
    print("=" * 70)


if __name__ == "__main__":
    main()
