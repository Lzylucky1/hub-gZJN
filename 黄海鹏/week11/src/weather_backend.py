import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
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

def get_positioning(city: str) -> str:
    # 整体流程：Geocoding（城市名→经纬度）→ 天气API查询 → 格式化输出。
    # 被三种方式（Function Call / MCP / CLI）共同复用，是纯业务逻辑层。
    """
    查询指定城市的当前天气及未来3天预报。

    Args:
        city: 城市名称，支持中文，例如 "宁德"、"北京"、"上海"

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述
    """
    # 查询失败时返回可读的错误字符串（不抛异常，方便 LLM 直接消费）。
    
    with httpx.Client(timeout=10.0) as client:
        # Step 1：Geocoding — 城市名 → 经纬度
        # 中国地名常有歧义：裸"宁德"会命中西藏那曲市的一个村（PPL），
        # 而宁德时代总部所在的福建宁德是地级市"宁德市"（PPLA2）。
        # 策略：先按用户输入查；若命中的只是低级行政点（feature_code 纯 PPL），
        # 且用户没带"市/县/区"后缀，就用 city+"市" 重查一次并优先采用。

        # 城市名消歧策略
        def _geocode(name: str):
            """内部辅助：调用 Open-Meteo Geocoding API，将城市名转为经纬度候选列表。
            返回最多 10 个候选结果，按相关度排序。"""
            resp = client.get(GEOCODING_URL, params={
                "name": name, "count": 10, "language": "zh", "format": "json",
            })
            resp.raise_for_status()
            return resp.json().get("results") or []

        # 城市名消歧策略：
        # 中国地名常有歧义——裸"宁德"可能命中西藏那曲市的一个村庄（PPL），
        # 而宁德时代总部所在的福建宁德是地级市"宁德市"（PPLA2）。
        # 策略：先按用户输入查；若所有候选都只是低级居民点（feature_code 为 PPL 但非 PPLA），
        # 且用户输入没有带"市/县/区/镇"后缀，则自动用 city+"市" 重查一次。
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
            """内部辅助：为 Geocoding 候选结果打分，返回 (行政级别权重, 人口数) 元组。"""
            fc = str(r.get("feature_code", ""))
            admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
            pop = r.get("population") or 0
            return (admin_priority, pop)

        loc = max(results, key=_rank)  # 取得分最高的候选
        lat = loc["latitude"] # 纬度
        lon = loc["longitude"] # 经度
        city_name = loc.get("name", city) # 城市名
        country = loc.get("country", "") # 国家
        admin1 = loc.get("admin1", "")  # 省/州级行政区

        return lat, lon, city_name, country, admin1


def get_positioning_str(city: str) -> str:
    """
    查询指定城市的位置信息（经纬度、行政区）。
    
    Args:
        city: 城市名称，支持中文，例如 "宁德"、"北京"、"上海"
    
    Returns:
        包含城市位置信息的格式化字符串
    """
    result = get_positioning(city)
    if isinstance(result, str):  # 错误信息
        return result
    lat, lon, city_name, country, admin1 = result
    location_str = f"{country} {admin1} {city_name}".strip()
    return f"【{location_str}】位置信息\n坐标：{lat:.2f}°N, {lon:.2f}°E"


# 天气查询 （Geocoding -> 天气API -> 格式化）
def get_weather(city_info: tuple) -> str:
    """根据经纬度和城市信息查询天气。"""
    lat, lon, city_name, country, admin1 = city_info  # 解包元组
    with httpx.Client(timeout=10.0) as client:
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, nargs="+", help="支持一个或多个城市，如 --city 北京 广州 上海 宁德")
    args = parser.parse_args()
    for city in args.city:
        getPosition = get_positioning(city)
        print(getPosition)
        # print(type(getPosition))
        print()  # 城市间空一行分隔

        print(get_weather(getPosition))
        
