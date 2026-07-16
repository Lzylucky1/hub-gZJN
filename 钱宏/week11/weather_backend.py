"""
weather_backend.py — 天气查询后端（三种方式共享的业务逻辑）

教学重点：
  1. 同样是"纯业务逻辑"，与 rag_backend 平级，被三种方式复用
  2. 拆分后的架构：get_location（城市名→经纬度）+ get_weather（经纬度→天气）
  3. 错误处理返回可读字符串而非抛异常，方便 LLM 直接消费

使用方式（作为模块）：
  from weather_backend import get_location, get_weather
  loc = get_location("宁德")           # 完整信息
  loc = get_location("宁德", coordinates_only=True)  # 仅经纬度
  weather = get_weather(latitude, longitude)

依赖：
  pip install httpx
  Open-Meteo API 完全免费，无需注册
"""

import json
import httpx

# Geocoding API：城市名转经纬度
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
# 天气 API：经纬度转天气数据
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo 天气代码 → 中文描述映射
WEATHER_CODE_MAP = {
    0: "晴天", 1: "大致晴朗", 2: "局部多云", 3: "阴天",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}


def get_location(city: str, coordinates_only: bool = False) -> str:
    """
    查询指定城市的经纬度。

    Args:
        city: 城市名称，支持中文，例如 "宁德"、"北京"、"上海"
        coordinates_only: 若为 True，仅返回 latitude 和 longitude；
                          若为 False（默认），返回完整信息包括 city_name、country、admin1

    Returns:
        JSON 字符串，包含经纬度信息；若城市未找到，返回错误信息
    """
    with httpx.Client(timeout=10.0) as client:
        # 内部函数：调用 Geocoding API 查询城市地理信息
        def _geocode(name: str):
            resp = client.get(GEOCODING_URL, params={
                "name": name, "count": 10, "language": "zh", "format": "json",
            })
            resp.raise_for_status()
            return resp.json().get("results") or []

        # 首次查询：按用户输入的城市名查询
        results = _geocode(city)

        # 行政级别消歧逻辑：
        # 中国地名常有歧义，例如"宁德"可能命中西藏那曲市的一个小村庄（PPL），
        # 而宁德时代总部所在的福建宁德是地级市"宁德市"（PPLA2）。
        # 判断条件：如果所有候选结果的 feature_code 都是纯 PPL（村庄级），
        # 且用户输入不带"市/县/区/镇"后缀，则尝试用"城市名+市"重查。
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

        # 如果仍无结果，返回错误信息
        if not results:
            return json.dumps({
                "error": f"未找到城市 '{city}'，请尝试其他写法（如'宁德市'改'宁德'）"
            }, ensure_ascii=False)

        # 候选结果排序策略：
        # 优先选择行政级别高的地点（feature_code 含 A，如 PPLA/ADM 表示政府驻地），
        # 其次选择人口多的地点，避免落到同名小村庄。
        def _rank(r):
            fc = str(r.get("feature_code", ""))
            admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
            pop = r.get("population") or 0
            return (admin_priority, pop)

        # 选择最优匹配结果
        loc = max(results, key=_rank)

        # 根据参数决定返回格式
        if coordinates_only:
            return json.dumps({
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "city_name": loc.get("name", city),
                "country": loc.get("country", ""),
                "admin1": loc.get("admin1", ""),
            }, ensure_ascii=False)


def get_weather(latitude: float, longitude: float) -> str:
    """
    根据经纬度查询当前天气及未来3天预报。

    Args:
        latitude: 纬度
        longitude: 经度

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述
    """
    with httpx.Client(timeout=10.0) as client:
        try:
            # 调用 Open-Meteo 天气 API，请求参数说明：
            # - latitude/longitude: 查询坐标
            # - current: 当前天气数据（温度、湿度、风速、天气代码）
            # - daily: 每日预报数据（最高/最低温、降水量、天气代码）
            # - timezone: 使用上海时区，确保时间显示正确
            # - forecast_days: 预报天数（含今天共4天，此处取未来3天）
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

        # 将天气代码转换为中文描述
        weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")

        # 格式化输出结果
        lines = [
            f"坐标：{latitude:.2f}°N, {longitude:.2f}°E",
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--coordinates-only", action="store_true", help="仅返回经纬度")
    args = parser.parse_args()
    loc = get_location(args.city, coordinates_only=args.coordinates_only)
    print("Location:", loc)
    if not args.coordinates_only:
        loc_data = json.loads(loc)
        if "latitude" in loc_data and "error" not in loc_data:
            weather = get_weather(loc_data["latitude"], loc_data["longitude"])
            print("\nWeather:", weather)
