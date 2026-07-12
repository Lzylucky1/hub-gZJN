"""
weather_backend.py — 天气查询后端，拆分为两个独立函数供 Function Call 演示多步调用。

依赖：pip install httpx
API：Open-Meteo，完全免费无需注册
"""

import httpx

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

def _do_geocode(client: httpx.Client, name: str) -> list:
    """单次 geocoding 请求，返回 results 列表。"""
    resp = client.get(GEOCODING_URL, params={
        "name": name, "count": 10, "language": "zh", "format": "json",
    })
    resp.raise_for_status()
    return resp.json().get("results") or []

def geocode(city: str) -> str:
    """
    城市名 → 经纬度。

    Args:
        city: 城市中文名，如 "北京"、"上海"

    Returns:
        包含城市名、经纬度、国家/省份的文字信息，可直接喂给 get_weather_by_coords。
    """
    with httpx.Client(timeout=10.0) as client:
        # 第一步：用用户输入查
        results = _do_geocode(client, city)

        # 地名消歧：如果结果全是低级行政点且用户没带后缀，用"xx市"重试
        is_low = all(
            str(r.get("feature_code", "")).startswith("PPL")
            and not str(r.get("feature_code", "")).startswith("PPLA")
            for r in results
        ) if results else True
        has_suffix = any(city.endswith(s) for s in ("市", "县", "区", "镇"))
        if is_low and not has_suffix:
            retry = _do_geocode(client, city + "市")
            if retry:
                results = retry

        if not results:
            return f"未找到城市 '{city}'，请尝试其他写法"

        # 优先行政级别高的结果
        loc = max(results, key=lambda r: (
            1 if str(r.get("feature_code", "")).startswith(("PPLA", "ADM")) else 0,
            r.get("population") or 0,
        ))

        lat = loc["latitude"]
        lon = loc["longitude"]
        city_name = loc.get("name", city)
        country = loc.get("country", "")
        admin1 = loc.get("admin1", "")
        location_str = f"{country} {admin1} {city_name}".strip()

        return f"【{location_str}】坐标：{lat:.4f}°N, {lon:.4f}°E"


def get_weather_by_coords(lat: float, lon: float) -> str:
    """
    根据经纬度查询当前天气及未来3天预报。

    Args:
        lat: 纬度
        lon: 经度

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述。
    """
    with httpx.Client(timeout=10.0) as client:
        try:
            resp = client.get(WEATHER_URL, params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                "timezone": "Asia/Shanghai",
                "forecast_days": 3,
            })
            resp.raise_for_status()
        except httpx.RequestError as e:
            return f"天气数据获取失败：{e}"

        data = resp.json()
        cur = data["current"]
        daily = data["daily"]

        weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")

        lines = [
            f"坐标 ({lat:.2f}°N, {lon:.2f}°E) 天气报告",
            f"当前：{weather_desc}，{cur['temperature_2m']}°C，"
            f"湿度 {cur['relative_humidity_2m']}%，风速 {cur['wind_speed_10m']} km/h",
            "",
            "未来3天：",
        ]
        for i in range(3):
            day_desc = WEATHER_CODE_MAP.get(daily["weather_code"][i], "")
            lines.append(
                f"  {daily['time'][i]}：{day_desc}，"
                f"{daily['temperature_2m_max'][i]}°C / {daily['temperature_2m_min'][i]}°C，"
                f"降水 {daily['precipitation_sum'][i]} mm"
            )

        return "\n".join(lines)


def get_weather(city: str) -> str:
    """一键查天气（geocode → 天气），保留兼容。"""
    geo_result = geocode(city)
    if geo_result.startswith("未找到"):
        return geo_result

    # 从 geocode 结果中解析经纬度
    import re
    m = re.search(r"(\d+\.\d+)°N,\s*(\d+\.\d+)°E", geo_result)
    if not m:
        return f"坐标解析失败：{geo_result}"
    lat, lon = float(m.group(1)), float(m.group(2))
    return get_weather_by_coords(lat, lon)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    args = parser.parse_args()
    print(get_weather(args.city))
