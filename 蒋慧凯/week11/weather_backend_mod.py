"""
weather_backend_mod.py — 作业版：天气查询拆为两个独立工具

教学点：
  原始项目的 get_weather(city) 把"查坐标"和"查天气"合在一步，
  本作业拆为 get_coordinates + get_weather_by_coords，
  让 LLM 必须链式调用两次才能完成天气查询——这天然验证了多轮循环的必要性。

使用方式：
  from weather_backend_mod import get_coordinates, get_weather_by_coords
  coords = get_coordinates("宁德")
  weather = get_weather_by_coords(26.66, 119.52)
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


def get_coordinates(city: str) -> str:
    """
    查询城市的经纬度坐标。

    Args:
        city: 城市名称，支持中文，例如 "宁德"、"北京"、"上海"

    Returns:
        包含纬度、经度、城市名、省份的坐标信息文本。
        返回的纬度/经度值可直接作为 get_weather_by_coords 的参数传入。
    """
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

        def _rank(r):
            fc = str(r.get("feature_code", ""))
            admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
            pop = r.get("population") or 0
            return (admin_priority, pop)

        loc = max(results, key=_rank)
        lat = loc["latitude"]
        lon = loc["longitude"]
        city_name = loc.get("name", city)
        admin1 = loc.get("admin1", "")

        return (
            f"城市：{city_name}\n"
            f"省份：{admin1}\n"
            f"纬度：{lat}\n"
            f"经度：{lon}"
        )


def get_weather_by_coords(latitude: float, longitude: float) -> str:
    """
    根据经纬度查询当前天气及未来3天预报。

    参数 latitude 和 longitude 应来自 get_coordinates 的返回结果。
    本工具不接收城市名——需先调 get_coordinates 拿到坐标再调本工具。

    Args:
        latitude: 纬度（如 26.66）
        longitude: 经度（如 119.53）

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述
    """
    with httpx.Client(timeout=10.0) as client:
        try:
            weather_resp = client.get(WEATHER_URL, params={
                "latitude": latitude,
                "longitude": longitude,
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

        weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")

        lines = [
            f"坐标 {latitude:.2f}°N, {longitude:.2f}°E 的天气报告",
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
