import argparse
import sys
from pathlib import Path

# 让本脚本能 import 项目根的 src/（无论从哪个工作目录 / 是否安装）
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag_backend import search_annual_report, list_companies  # noqa: E402
from src.weather_backend import get_weather, get_positioning, get_positioning_str  # noqa: E402

# fincli命令行入口
def main():
    parser = argparse.ArgumentParser(
        prog="fincli",
        description="fincli — A股年报检索 + 天气查询 命令行工具",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # fincli search ...
    p_search = sub.add_parser("search", help="检索年报段落")
    p_search.add_argument("--query", required=True,
                          help="检索问题（不要含公司名/年份，用简短财务术语，如 '营收和净利润'）")
    p_search.add_argument("--stock-code", default=None, help="按公司过滤，如 300750")
    p_search.add_argument("--year", default=None, help="按年份过滤：2021/2022/2023")
    p_search.add_argument("--top-k", type=int, default=5, help="返回段落数，默认5")

    # fincli list-companies
    sub.add_parser("list-companies", help="列出知识库收录的公司")

    # fincli get-city-position ...
    p_position = sub.add_parser("get-city-position", help="查询城市位置信息（经纬度）")
    p_position.add_argument("--city", required=True, help="城市中文名，如 宁德")

    # fincli get-city-weather ...
    p_weather = sub.add_parser("get-city-weather", help="查询城市位置+天气")
    p_weather.add_argument("--city", required=True, help="城市中文名，如 宁德")

    args = parser.parse_args()

    if args.cmd == "search":
        print(search_annual_report(args.query, args.stock_code, args.year, args.top_k))
    elif args.cmd == "list-companies":
        print(list_companies())
    elif args.cmd == "get-city-position":
        print(get_positioning_str(args.city))
    elif args.cmd == "get-city-weather":
        position = get_positioning(args.city)
        if isinstance(position, str):  # 错误信息
            print(position)
        else:
            print(get_weather(position))


if __name__ == "__main__":
    main()
