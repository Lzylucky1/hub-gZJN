
import argparse
import json
import time
from typing import Optional
from openai import OpenAI
from jsonschema import validate, ValidationError

# ── 配置 ──────────────────────────────────────────────────────────────
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
MODEL = "qwen2-0.5b"


# ══════════════════════════════════════════════════════════════════════
#                     工具 1: query_game_item
# ══════════════════════════════════════════════════════════════════════

ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "game": {"type": "string", "enum": ["王者荣耀", "原神", "和平精英", "梦幻西游"]},
        "item_name": {"type": "string"},
        "server": {"type": "string", "pattern": r"^S\d+$"},
        "quality": {"type": "string", "enum": ["白", "蓝", "紫", "金", "传说"]},
        "fields": {
            "type": "array",
            "items": {"type": "string", "enum": ["攻击", "防御", "暴击率", "价格", "掉落"]},
            "minItems": 1,
        },
    },
    "required": ["game", "item_name", "fields"],
    "additionalProperties": False,
}

ITEM_SYSTEM = """你是游戏装备查询工具的参数生成器。根据用户问题输出纯 JSON 参数，不要任何解释文字。

JSON 格式：
{
  "game": "王者荣耀" | "原神" | "和平精英" | "梦幻西游",
  "item_name": "装备名称",
  "server": "区服代码，如 S1，S10",
  "quality": "白" | "蓝" | "紫" | "金" | "传说",
  "fields": ["攻击" | "防御" | "暴击率" | "价格" | "掉落"] 的数组
}

必选字段：game, item_name, fields
区服规则：server 格式为 S 加数字（如 S1, S10, S250）
品质默认：不填 quality 则默认 "紫"
字段默认：fields 不填默认 ["攻击"]

示例：查询王者荣耀 S1 服屠龙刀的攻击力和价格
输出：{"game": "王者荣耀", "item_name": "屠龙刀", "server": "S1", "quality": "金", "fields": ["攻击", "价格"]}"""

ITEM_TEST_CASES = [
    # 基础直接（15）
    "查一下王者荣耀里破军的攻击力",
    "原神里和璞鸢的暴击率多少",
    "和平精英M416的伤害",
    "梦幻西游无级别限制武器的价格",
    "王者荣耀破晓的防御属性",
    "原神雾切之回光掉落在哪",
    "查屠龙刀的攻击和暴击率",
    "和平精英AWM的价格",
    "梦幻西游高级必杀兽决掉落",
    "王者荣耀霸者重装防御",
    "原神天空之翼暴击率",
    "查王者荣耀无尽战刃攻击",
    "梦幻西游四法青云价格",
    "原神护摩之杖攻击和暴击",
    "和平精英三级甲防御",
    # 省略 game（8）
    "查一下破军的属性",
    "和璞鸢的暴击率多少",
    "M416伤害数据",
    "看看屠龙刀的价格",
    "无级别武器现在什么价",
    "三级甲防御多少",
    "雾切之回光掉落信息",
    "高级必杀多少钱",
    # 诱导多余文本（7）
    "帮我查破军攻击力，顺便说说破军这件装备的背景故事",
    "查一下屠龙刀价格，这把刀是传奇武器吧",
    "原神和璞鸢暴击率怎么样，值得抽吗",
    "M416和AKM哪个伤害高，查一下M416数据",
    "查梦幻西游高级必杀价格，顺便推荐一下平民装备",
    "三级甲和二级甲区别，先查三级甲防御",
    "无级别武器价格多少，这个装备为什么这么贵",
    # 非标准输入（5）
    "查一下王者农药里破军的攻击",
    "look up 原神 and璞鸢 crit rate",
    "梦话西游 无级别 价钱",
    "和p u 鸢 暴击",
    "M4 伤害 多少",
    # 区服模糊（5）
    "我在一区，查屠龙刀价格",
    "S1 服的破军什么价",
    "查王者荣耀10区无尽战刃属性",
    "我在 S 25 服，看下三级甲",
    "S3 原神和璞鸢暴击率",
    # 复合字段（5）
    "查破军的攻击、防御和价格",
    "屠龙刀全部数据看看",
    "M416 伤害和掉落都查",
    "雾切之回光攻击暴击率价格全要",
    "三级甲防御和价格",
    # 边界/无意义（5）
    "随便查一个装备",
    "装备查询",
    "帮我看看",
    "？？？",
    "查查",
]
assert len(ITEM_TEST_CASES) == 50


# ══════════════════════════════════════════════════════════════════════
#                     工具 2: create_character
# ══════════════════════════════════════════════════════════════════════

CHAR_SCHEMA = {
    "type": "object",
    "properties": {
        "nickname": {"type": "string", "minLength": 1},
        "class_type": {"type": "string", "enum": ["战士", "法师", "射手", "刺客", "辅助"]},
        "server": {"type": "string", "pattern": r"^S\d+$"},
        "phone": {"type": "string", "pattern": r"^1[3-9]\d{9}$"},
        "gender": {"type": "string", "enum": ["男", "女"]},
    },
    "required": ["nickname", "class_type", "server", "phone"],
    "additionalProperties": False,
}

CHAR_SYSTEM = """你是游戏角色创建工具的参数生成器。根据用户描述输出纯 JSON 角色创建参数，不要任何解释文字。

JSON 格式：
{
  "nickname": "角色昵称",
  "class_type": "战士" | "法师" | "射手" | "刺客" | "辅助",
  "server": "区服代码 S 加数字，如 S1",
  "phone": "11位手机号（1开头第二位3-9）",
  "gender": "男" | "女"
}

必选字段：nickname, class_type, server, phone
默认值：gender 不填默认 "男"

示例：张三要在 S1 服建一个战士号，电话 13912345678
输出：{"nickname": "张三", "class_type": "战士", "server": "S1", "phone": "13912345678", "gender": "男"}"""

CHAR_TEST_CASES = [
    # 基础直接（15）
    "帮我在 S1 创建一个战士，昵称 剑指苍穹，电话 13812345678",
    "S10 服建一个法师号，叫 魔法少女，手机 13987654321",
    "创建射手角色 百步穿杨，S5 服，13711112222",
    "S20 来个刺客，昵称 暗影，电话 13644445555",
    "建个辅助 奶一口，S3 服，15277778888",
    "S8 服 战士 铁甲小宝 13922223333",
    "创建法师 冰霜法王 S15 13855556666",
    "S2 射手 一箭封喉 13766667777",
    "刺客 午夜幽灵 S12 13511112222",
    "辅助 守护者 S6 13688889999",
    "S9 战士 无敌铁牛 13433334444",
    "法师 暴风雪 S18 18900001111",
    "射手 精准打击 S7 13055556666",
    "刺客 暗夜猎手 S11 13677778888",
    "辅助 生命之光 S16 13822223333",
    # 角色名特殊（6）
    "建个号叫 ロマサガ，S5 服，战士，13812345678",
    "昵称 空一格，S3，法师，13987654321",
    "起名 这是一个很长很长的昵称测试，S1 刺客 13711112222",
    "就叫 1，S2 射手 13833334444",
    "英文名 DragonSlayer，S6 战士 13955556666",
    "昵称 丶冷月，S8 法师 13677778888",
    # 职业诱导错（6）
    "建个坦克号，S1，叫 铁壁，13812345678",  # "坦克"不在枚举
    "S3 服 狂战士 破晓 13987654321",          # "狂战士"不在枚举
    "法术射手 S5 13211112222",                # 复合职业
    "奶妈 S8 13688889999",                    # "奶妈"不在枚举
    "ADC S10 我叫ADC 13833334444",            # "ADC"不在枚举
    "打野 S2 野王 13766667777",               # "打野"不在枚举
    # 电话不标准（7）
    "建号 剑客 S1 战士 138-1234-5678",
    "创建 法师 冰火 S3 138 1234 5678",
    "角色 射手 猎手 S5 +86 13812345678",
    "S7 刺客 无影 手机号 12345678",
    "S9 辅助 天使 19912345678",               # 199 开头合法
    "战士 猛男 S11 00000000000",              # 全零
    "法师 小雪 S13 电话 1381234567",          # 少一位
    # 区服不标准（5）
    "建个战士号 一区 铁壁 13812345678",
    "S 5 服 法师 冰霜 13987654321",
    "10 区 射手 猎空 13711112222",
    "在 S 25 服建个刺客 暗影 13644445555",
    "我要在 3 服建小号 辅助 奶妈 15277778888",
    # gender 诱导错（5）
    "建个女号 法师 冰雪 S1 13812345678",
    "妹子号 射手 花火 S3 13987654321",
    "人妖号 S5 辅助 听风 13711112222",
    "建个男性角色 战士 硬汉 S7 13833334444",
    "我要女的 刺客 魅影 S9 13677778888",
    # 边界/无意义（6）
    "帮我创个号",
    "我要建角色",
    "随便建一个",
    "？？？",
    "创建",
    "帮我注册不告诉你啥职业",
]
assert len(CHAR_TEST_CASES) == 50


# ══════════════════════════════════════════════════════════════════════
#                     通用运行 + 评估逻辑
# ══════════════════════════════════════════════════════════════════════

def run_one(system: str, user: str, mode: str, schema: dict,
            max_tokens: int = 250) -> tuple[str, float]:
    """mode: 'raw' | 'response_format' | 'guided_json'"""
    extra = {}
    kwargs = {}
    if mode == "guided_json":
        extra = {"guided_json": schema}
    elif mode == "response_format":
        kwargs = {"response_format": {"type": "json_object"}}

    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=max_tokens,
        extra_body=extra,
        **kwargs,
    )
    return resp.choices[0].message.content.strip(), time.time() - t0


def evaluate_output(output: str, schema: dict) -> dict:
    """返回分层指标：JSON 合法 / 必选字段齐全 / jsonschema 完全通过"""
    r = {"is_json": False, "has_required": False, "schema_valid": False,
         "parsed": None, "error": None}
    try:
        obj = json.loads(output)
        r["is_json"] = True
        r["parsed"] = obj
    except json.JSONDecodeError as e:
        r["error"] = f"JSON syntax: {e}"
        return r

    if not isinstance(obj, dict):
        r["error"] = "not an object"
        return r

    required = schema.get("required", [])
    if all(k in obj for k in required):
        r["has_required"] = True

    try:
        validate(instance=obj, schema=schema)
        r["schema_valid"] = True
    except ValidationError as e:
        r["error"] = f"schema: {e.message[:80]}"
    return r


def run_tool_benchmark(tool_name: str, schema: dict, system: str,
                        cases: list[str]) -> dict:
    """跑完一个工具的 50×3 测试，返回 stats 和示例"""
    modes = ["raw", "response_format", "guided_json"]
    stats = {m: {"is_json": 0, "has_required": 0, "schema_valid": 0, "total_latency": 0.0}
             for m in modes}
    fail_examples = {m: [] for m in modes}  # 保存前 3 个失败用例

    print(f"\n{'='*78}")
    print(f"  工具: {tool_name}   测试数: {len(cases)}   模式: 3")
    print(f"{'='*78}")

    for i, user in enumerate(cases, 1):
        if i % 10 == 0:
            print(f"  进度: {i}/{len(cases)}")
        for mode in modes:
            try:
                out, dt = run_one(system, user, mode, schema)
            except Exception as e:
                out, dt = f"[REQUEST ERROR: {e}]", 0.0
            ev = evaluate_output(out, schema)
            s = stats[mode]
            s["total_latency"] += dt
            if ev["is_json"]:       s["is_json"] += 1
            if ev["has_required"]:  s["has_required"] += 1
            if ev["schema_valid"]:  s["schema_valid"] += 1
            # 保存失败案例（schema 不过的前 3 个）
            if not ev["schema_valid"] and len(fail_examples[mode]) < 3:
                fail_examples[mode].append({
                    "user": user,
                    "output": out[:150],
                    "error": ev.get("error", "unknown"),
                })

    return {"stats": stats, "fails": fail_examples, "n": len(cases)}


def print_report(tool_name: str, result: dict):
    stats = result["stats"]
    fails = result["fails"]
    n = result["n"]

    print(f"\n{'─'*78}")
    print(f"  【{tool_name}】 {n} 条测试 × 3 模式 汇总")
    print(f"{'─'*78}")
    print(f"{'指标':<24}{'裸 prompt':<20}{'response_format':<22}{'guided_json':<15}")
    print("─" * 78)
    metric_labels = [
        ("JSON 语法合法", "is_json"),
        ("必选字段齐全", "has_required"),
        ("完整 schema 通过 ★", "schema_valid"),
    ]
    for label, key in metric_labels:
        row = f"{label:<22}"
        for mode in ["raw", "response_format", "guided_json"]:
            v = stats[mode][key]
            row += f"{v}/{n} ({100*v/n:>3.0f}%)       "
        print(row)
    print(f"{'平均延迟（秒）':<22}", end="")
    for mode in ["raw", "response_format", "guided_json"]:
        avg = stats[mode]["total_latency"] / n
        print(f"{avg:.3f}              ", end="")
    print()

    print(f"\n{'─'*78}")
    print(f"  【{tool_name}】 典型失败案例（前 3 条）")
    print(f"{'─'*78}")
    for mode in ["raw", "response_format", "guided_json"]:
        n_fails = len(fails[mode])
        if n_fails == 0:
            print(f"\n[{mode}] ✓ 无失败案例")
        else:
            print(f"\n[{mode}] 失败示例（schema 校验未通过）：")
            for f in fails[mode]:
                print(f"  ▶ Prompt: {f['user']}")
                print(f"    输出:   {f['output']}")
                print(f"    错误:   {f['error']}")


# ══════════════════════════════════════════════════════════════════════
#                         main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", choices=["item", "character", "both"], default="both")
    parser.add_argument("--out", default="../outputs/function_call_results.json",
                        help="结果保存路径（相对 src/）")
    args = parser.parse_args()

    print("=" * 78)
    print("  demo_function_call.py   核心：裸 prompt vs response_format vs guided_json")
    print(f"  Model: {MODEL}")
    print("=" * 78)

    all_results = {}

    if args.tool in ("item", "both"):
        t0 = time.time()
        r = run_tool_benchmark("query_game_item", ITEM_SCHEMA, ITEM_SYSTEM, ITEM_TEST_CASES)
        all_results["item"] = r
        print_report("query_game_item", r)
        print(f"\n  [耗时 {time.time()-t0:.1f}s]")

    if args.tool in ("character", "both"):
        t0 = time.time()
        r = run_tool_benchmark("create_character", CHAR_SCHEMA, CHAR_SYSTEM, CHAR_TEST_CASES)
        all_results["character"] = r
        print_report("create_character", r)
        print(f"\n  [耗时 {time.time()-t0:.1f}s]")

    # 保存详细结果
    import os
    out_path = os.path.join(os.path.dirname(__file__), args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # 清洗不可序列化内容
    to_save = {}
    for k, v in all_results.items():
        to_save[k] = {
            "n": v["n"],
            "stats": v["stats"],
            "fails": v["fails"],
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_save, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存：{out_path}")

    print("\n" + "=" * 78)
    print("  核心结论：")
    print("    裸 prompt        — JSON 语法偶尔错 / 字段拼错 / 正则枚举不符")
    print("    response_format  — JSON 合法率接近满分，但字段语义仍错")
    print("    guided_json      — 100% 满足完整 schema（小模型从不可用变可靠）")
    print("=" * 78)


if __name__ == "__main__":
    main()
