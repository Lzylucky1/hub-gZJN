
import json
import time
from openai import OpenAI
from jsonschema import validate, ValidationError

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
MODEL = "qwen2-0.5b"

# ── JSON Schema 定义 ──────────────────────────────────────────────────
ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "item_name": {
            "type": "string",
            "description": "装备名称，如 屠龙刀、天丛云剑",
        },
        "quality": {
            "type": "string",
            "enum": ["白", "蓝", "紫", "金", "传说"],
        },
        "level": {
            "type": "integer",
            "minimum": 1,
            "maximum": 20,
        },
        "attributes": {
            "type": "array",
            "items": {"type": "string", "enum": ["攻击", "防御", "暴击", "生命", "速度"]},
            "minItems": 1,
        },
    },
    "required": ["item_name", "quality", "level"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = f"""你是游戏装备信息抽取助手。从用户描述中提取装备结构化信息，输出纯 JSON，不要任何解释文字。

字段定义：
  item_name: 装备名称
  quality: 品质，必须是 ['白', '蓝', '紫', '金', '传说'] 之一
  level: 强化等级（1~20 整数）
  attributes: 属性词条数组，可选 ['攻击', '防御', '暴击', '生命', '速度']

示例输出：
{{"item_name": "屠龙刀", "quality": "金", "level": 15, "attributes": ["攻击", "暴击"]}}"""

TEST_CASES = [
    "我这把15级的金色屠龙刀",
    "一把蓝色的5级护甲",
    "传说的武器，20级，加暴击和速度",
    "紫色10级项链，加生命和防御",
    "一个金装20级，三个攻击词条",
    "我的白板1级木剑",
    "传说级武器，lv18，暴击拉满",
    "蓝装8级带防御属性",
    "史诗级的20级装备，双攻击词条",  # "史诗级"不在枚举，模型要映射到"传说"
    "紫武 lv12 加攻击和暴击",
]


def evaluate(output: str) -> dict:
    """评估一个输出：分层校验（JSON 合法 / 字段齐全 / schema 完全通过）"""
    result = {
        "is_json": False,
        "has_all_fields": False,
        "quality_in_enum": False,
        "level_in_range": False,
        "schema_valid": False,
        "parsed": None,
    }
    try:
        obj = json.loads(output)
        result["is_json"] = True
        result["parsed"] = obj
    except json.JSONDecodeError:
        return result

    required = ITEM_SCHEMA["required"]
    if all(k in obj for k in required):
        result["has_all_fields"] = True

    if obj.get("quality") in ITEM_SCHEMA["properties"]["quality"]["enum"]:
        result["quality_in_enum"] = True

    lv = obj.get("level")
    if isinstance(lv, int) and 1 <= lv <= 20:
        result["level_in_range"] = True

    try:
        validate(instance=obj, schema=ITEM_SCHEMA)
        result["schema_valid"] = True
    except ValidationError:
        pass

    return result


def run_generate(user_msg: str, mode: str) -> tuple[str, float]:
    """mode: 'raw' | 'guided_json' | 'response_format'"""
    extra = {}
    kwargs = {}
    if mode == "guided_json":
        extra = {"guided_json": ITEM_SCHEMA}
    elif mode == "response_format":
        kwargs = {"response_format": {"type": "json_object"}}

    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=120,
        extra_body=extra,
        **kwargs,
    )
    return resp.choices[0].message.content.strip(), time.time() - t0


def main():
    print("=" * 78)
    print("  Demo: guided_json（JSON Schema 约束）")
    print(f"  Model: {MODEL}")
    print("  对比三种模式：裸 prompt / response_format / guided_json")
    print("  场景：游戏装备信息抽取")
    print("=" * 78)

    counters = {m: {"json": 0, "fields": 0, "quality": 0, "level": 0, "valid": 0}
                for m in ["raw", "response_format", "guided_json"]}
    n = len(TEST_CASES)

    for user in TEST_CASES:
        print(f"\n▶ {user}")
        for mode in ["raw", "response_format", "guided_json"]:
            out, _ = run_generate(user, mode)
            ev = evaluate(out)
            c = counters[mode]
            if ev["is_json"]:         c["json"] += 1
            if ev["has_all_fields"]:  c["fields"] += 1
            if ev["quality_in_enum"]: c["quality"] += 1
            if ev["level_in_range"]:  c["level"] += 1
            if ev["schema_valid"]:    c["valid"] += 1
            tag = "✓" if ev["schema_valid"] else "✗"
            disp = out[:80] + "…" if len(out) > 80 else out
            print(f"  [{mode:<16}] {tag}  {disp}")

    print("\n" + "=" * 78)
    print(f"  {n} 条测试结果汇总")
    print("=" * 78)
    print(f"{'指标':<24}{'裸 prompt':<18}{'response_format':<20}{'guided_json':<15}")
    print("-" * 78)
    for metric_name, key in [("合法 JSON", "json"),
                              ("字段齐全", "fields"),
                              ("quality 在枚举内", "quality"),
                              ("level 在 1~20", "level"),
                              ("jsonschema 完全通过", "valid")]:
        row = f"{metric_name:<22}"
        for mode in ["raw", "response_format", "guided_json"]:
            v = counters[mode][key]
            row += f"{v}/{n} ({100*v/n:.0f}%)      "
        print(row)

    print()
    print("=" * 78)
    print("  结论：")
    print("    response_format 只保证是 JSON，不保证字段名、类型、枚举正确")
    print("    guided_json     是唯一 100% 保证 schema 合法的方式")
    print("=" * 78)


if __name__ == "__main__":
    main()
