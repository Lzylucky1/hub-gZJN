"""
联网搜索 — DuckDuckGo Instant Answer API + Wikipedia API（免费、无 key、零依赖）

教学重点：
  1. 不需要 Tavily key 也能跑通——用 DuckDuckGo 官方 IA API + Wikipedia 搜索 API
     两个免费 JSON 端点拼出"够用的搜索结果"喂给 LLM
  2. 如果有 TAVILY_API_KEY，自动升级到 Tavily（返回更丰富的实时摘要 + 来源）
  3. 用标准库 urllib 调 HTTP，不引 requests/ddgs 等第三方包（少依赖原则）
  4. 失败不抛异常——返回错误字符串，ReAct loop 把它当 Observation 继续兜底

搜索策略：DDG IA API 给摘要 + 相关主题，Wikipedia 给条目搜索片段，
         两者合并去重后格式化成 LLM 可读文本。
"""

import os, json, urllib.request, urllib.parse, logging, re

logger = logging.getLogger(__name__)

_DDG_URL = "https://api.duckduckgo.com/"
_WIKI_URL = "https://en.wikipedia.org/w/api.php"


def _fetch_json(url: str, params: dict, timeout: int = 15) -> dict | None:
    """用 urllib 发 GET 请求取 JSON。失败返回 None。"""
    full = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        full, headers={"User-Agent": "Mozilla/5.0 (ParallelSubagentDemo/1.0)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.warning(f"HTTP 请求失败 {url}: {e}")
        return None


def _ddg_search(query: str, max_results: int = 5) -> list[dict]:
    """DuckDuckGo Instant Answer API。返回 [{title, url, content}]。"""
    data = _fetch_json(_DDG_URL, {
        "q": query, "format": "json", "no_html": 1, "skip_disambig": 1,
    })
    if not data:
        return []
    results = []
    # AbstractText 是 DDG 的主摘要（通常来自 Wikipedia）
    abstract = (data.get("Abstract") or data.get("AbstractText") or "").strip()
    if abstract:
        results.append({
            "title": data.get("Heading") or query,
            "url": data.get("AbstractURL") or "",
            "content": abstract[:600],
        })
    # RelatedTopics 扁平化取前几条
    for topic in (data.get("RelatedTopics") or [])[:max_results * 2]:
        if isinstance(topic, dict) and topic.get("Text"):
            results.append({
                "title": (topic.get("Text") or "")[:60],
                "url": topic.get("FirstURL") or "",
                "content": (topic.get("Text") or "")[:400],
            })
        if len(results) >= max_results:
            break
    return results[:max_results]


def _wiki_search(query: str, max_results: int = 5) -> list[dict]:
    """Wikipedia 搜索 API。返回 [{title, url, content(snippet)}]。"""
    data = _fetch_json(_WIKI_URL, {
        "action": "query", "list": "search",
        "srsearch": query, "format": "json", "srlimit": str(max_results),
    })
    if not data:
        return []
    out = []
    for item in data.get("query", {}).get("search", [])[:max_results]:
        snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))  # 去 HTML 标签
        title = item.get("title", "")
        out.append({
            "title": title,
            "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
            "content": snippet.strip()[:400],
        })
    return out


def _tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """Tavily 搜索（需要 TAVILY_API_KEY）。返回更丰富的实时结果。"""
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return []
    payload = json.dumps({
        "api_key": key, "query": query, "max_results": max_results,
        "search_depth": "basic", "include_answer": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=payload, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.warning(f"Tavily 搜索失败 '{query}': {e}")
        return []
    out = []
    if data.get("answer"):
        out.append({"title": "Tavily 摘要", "url": "", "content": data["answer"][:600]})
    for r in data.get("results", [])[:max_results]:
        out.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": (r.get("content") or "")[:400],
        })
    return out


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """统一搜索入口。优先 Tavily（有 key），否则 DDG + Wikipedia 合并。

    返回 [{title, url, content}] 列表，可能为空（ReAct 兜底处理）。
    """
    # 有 Tavily key → 用 Tavily（结果最好）
    if os.getenv("TAVILY_API_KEY"):
        results = _tavily_search(query, max_results)
        if results:
            return results
    # 否则 / Tavily 失败 → DDG + Wikipedia 合并去重
    ddg = _ddg_search(query, max_results)
    wiki = _wiki_search(query, max_results)
    # 简单去重（按 title 前缀）
    seen = set()
    merged = []
    for r in ddg + wiki:
        key = (r["title"] or "")[:30].lower()
        if key not in seen:
            seen.add(key)
            merged.append(r)
    return merged[:max_results] if merged else (ddg or wiki)


def format_search_result(results: list[dict]) -> str:
    """把搜索结果列表格式化成喂给 LLM 的文本。"""
    if not results:
        return "搜索无结果。请尝试换一个查询词。"
    parts = []
    for i, r in enumerate(results, 1):
        url_str = f" ({r['url']})" if r.get("url") else ""
        parts.append(f"[{i}] {r['title']}{url_str}\n    {r['content'][:300]}")
    return "\n".join(parts)


def search_and_format(query: str, max_results: int = 5) -> str:
    """便捷方法：搜索 + 格式化，一步到位。"""
    return format_search_result(web_search(query, max_results))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    q = "Python programming language"
    print(f"查询: {q}\n{'=' * 50}")
    results = web_search(q, max_results=3)
    print(f"结果数: {len(results)}")
    print(format_search_result(results))
