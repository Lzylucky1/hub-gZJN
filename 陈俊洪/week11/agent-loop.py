import os
import subprocess
import httpx

from anthropic import AnthropicBedrock
from dotenv import load_dotenv

load_dotenv(override=True)

MODEL = os.getenv("BEDROCK_MODEL_ID")
api_key = os.getenv("BEDROCK_API_KEY")
aws_region = os.getenv("AWS_REGION")

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODE_MAP = {
    0: "晴天", 1: "大致晴朗", 2: "局部多云", 3: "阴天",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}

client = AnthropicBedrock(
    api_key = api_key,
    aws_region = aws_region,
)

def get_weather(city: str) -> str:
    with httpx.Client(timeout=10.0) as client:
        def _geocode(name: str):
            resp = client.get(GEOCODING_URL, params={
                "name": name, "count": 10, "language": "zh", "format": "json",
            })
            resp.raise_for_status()
            return resp.json().get("results") or []

        results = _geocode(city)
        is_low_admin = all(
            str(r.get("feature_code", "")).startswith("PPL")
            and not str(r.get("feature_code", "")).startswith("PPLA")
            for r in results
        ) if results else True
        has_suffix = any(city.endswith(s) for s in ("市", "县", "区", "镇"))
        if is_low_admin and not has_suffix:
            retry = _geocode(city + "市")
            if retry:
                results = retry

        if not results:
            return f"未找到城市 '{city}'，请尝试其他写法（如'宁德市'改'宁德'）"

        # 在候选里优先取行政级别更高的（feature_code 含 A = 某级政府驻地），
        # 其次取有人口数据的，避免落到同名小村庄
        def _rank(r):
            fc = str(r.get("feature_code", ""))
            admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
            pop = r.get("population") or 0
            return (admin_priority, pop)

        loc = max(results, key=_rank)
        lat = loc["latitude"]
        lon = loc["longitude"]
        city_name = loc.get("name", city)
        country = loc.get("country", "")
        admin1 = loc.get("admin1", "")  # 省/州级行政区

        # Step 2：天气查询
        try:
            weather_resp = client.get(WEATHER_URL, params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                "timezone": "Asia/Shanghai",
                "forecast_days": 3,
            })
            weather_resp.raise_for_status()
        except httpx.RequestError as e:
            return f"天气数据获取失败：{e}"

        data = weather_resp.json()
        cur = data["current"]
        daily = data["daily"]

        # Step 3：格式化输出
        weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")
        location_str = f"{country} {admin1} {city_name}".strip()

        lines = [
            f"【{location_str}】天气报告",
            f"坐标：{lat:.2f}°N, {lon:.2f}°E",
            "",
            f"当前天气：{weather_desc}",
            f"  温度：{cur['temperature_2m']}°C",
            f"  相对湿度：{cur['relative_humidity_2m']}%",
            f"  风速：{cur['wind_speed_10m']} km/h",
            "",
            "未来3天预报：",
        ]
        for i in range(3):
            day_desc = WEATHER_CODE_MAP.get(daily["weather_code"][i], "")
            lines.append(
                f"  {daily['time'][i]}：{day_desc}，"
                f"{daily['temperature_2m_max'][i]}°C / {daily['temperature_2m_min'][i]}°C，"
                f"降水 {daily['precipitation_sum'][i]} mm"
            )

        return "\n".join(lines)

TOOLS_SCHEMA = [
    {
        "name": "get_weather",
        "description": "查询指定城市的天气",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名称"}},
            "required": ["city"]
        }
    }
]

SYSTEM = """你是一名智能 Agent。

当用户需要查询天气时，请使用 get_weather。

可以多次调用工具，直到完成任务，再回复最终答案。
"""

TOOL_HANDLERS = {
    "get_weather": get_weather,
}

def agent_loop(messages):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS_SCHEMA, max_tokens=8000,
        )

        # 大模型处理回复并累积
        messages.append({"role": "assistant", "content": response.content})

        # 判断是否使用工具
        if response.stop_reason != "tool_use":
            return
        
        # 使用工具，收集回复
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\033[33m$ {block.name}\033[0m")
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown: {block.name}"
                print(output[:200])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })

        # 返回工具结果循环继续
        messages.append({"role": "user", "content": results})

if __name__ == "__main__":
    user_query = "请帮我查一下成都的天气，并返回当前和未来三天的天气情况。"
    print("输入问题，回车发送。输入 q 退出。\n")
    print(f"例如: {user_query}")
    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        # Print the model's final text response
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()

# 以下是输出内容
# 【中国 重庆市 重庆】天气报告
# 坐标：29.56°N, 106.56°E

# 当前天气：阴天
#   温度：34.7°C
#   相对湿度：53%
#   风速：6.0 km/h

# 未来3天预报：
#   2026-07-16：小毛毛雨，40.3°C / 32.0°C，降水 0.2 mm
#   2026-07-17：雷暴伴小冰雹，34.0°C / 28.0°C，降水 16.6 mm
#   2026-07-18：雷暴伴小
# 以下是重庆的天气情况：

# ### 🌥️ 重庆当前天气
# | 项目 | 数据 |
# |------|------|
# | 天气状况 | 阴天 |
# | 温度 | 34.7°C |
# | 相对湿度 | 53% |
# | 风速 | 6.0 km/h |

# ### 📅 未来3天预报
# | 日期 | 天气 | 最高/最低温度 | 降水量 |
# |------|------|--------------|--------|
# | 7月16日 | 🌦️ 小毛毛雨 | 40.3°C / 32.0°C | 0.2 mm |
# | 7月17日 | ⛈️ 雷暴伴小冰雹 | 34.0°C / 28.0°C | 16.6 mm |
# | 7月18日 | ⛈️ 雷暴伴小冰雹 | 27.5°C / 26.2°C | 28.0 mm |

# ### 🌡️ 温馨提示
# - 当前重庆天气较热，请注意**防暑降温**！
# - 未来几天将有**雷暴及冰雹**天气，出行请携带雨具，注意安全。
# - 7月17日和18日降水量较大，建议尽量减少户外活动。
