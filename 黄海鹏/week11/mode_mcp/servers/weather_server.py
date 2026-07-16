import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

# 用 as 别名避免同名 tool 函数遮蔽后端函数导致递归
from src.weather_backend import get_weather as _get_weather  # noqa: E402
from src.weather_backend import get_positioning, get_positioning_str  # noqa: E402


def log(msg: str):
    """日志输出到 stderr（stdout 是 MCP JSON-RPC 通道，不能混入普通文本）。"""
    print(msg, file=sys.stderr, flush=True)


mcp = FastMCP("weather-server")

# 装饰器 如何把后端函数变成协议工具
@mcp.tool()
def get_city_position(city: str) -> str:
    """
    查询指定城市的位置信息（经纬度、行政区）。只返回位置，不返回天气。

    Args:
        city: 城市中文名，如 '宁德'、'北京'。

    Returns:
        包含城市位置信息的格式化字符串。
    """
    return get_positioning_str(city)


@mcp.tool()
def get_city_weather(city: str) -> str:
    """
    查询指定城市的当前天气及未来3天预报。返回位置+天气完整信息。

    Args:
        city: 城市中文名，如 '宁德'、'北京'。

    Returns:
        包含位置、温度、湿度、风速、天气状况和3天预报的文字描述。
    """
    position = get_positioning(city)
    if isinstance(position, str):  # 错误信息
        return position
    return _get_weather(position)


if __name__ == "__main__":
    log("Weather MCP Server 启动中（stdio 模式）...")
    mcp.run(transport="stdio")
