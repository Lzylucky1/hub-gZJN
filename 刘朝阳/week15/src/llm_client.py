"""
LLM 客户端 — Bailian(百炼/DashScope) GLM-5.2，OpenAI 兼容接口

教学重点：
  1. 主 agent 和 subagent 共用同一个 LLM 客户端，区别只在 system_prompt 和 tools
  2. 用 OpenAI SDK 调 DashScope 兼容接口——换提供商只改 base_url + model
  3. stop 参数是 ReAct 的关键技巧：让 LLM 在 "Observation:" 前停下，
     runner 执行工具后把 Observation 续写回去，再调 LLM 生成下一步

依赖：pip install openai
配置：环境变量 BAILIAN_API_KEY / BAILIAN_BASE_URL / BAILIAN_MODEL，
      也兼容已设的 ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL（opencode 环境同值）
"""

import os, time, logging
from openai import OpenAI

logger = logging.getLogger(__name__)

# ── 从环境变量读配置，兼容多种命名 ──────────────────────────────
API_KEY = (
    os.getenv("BAILIAN_API_KEY")
    or os.getenv("ANTHROPIC_API_KEY")      # opencode 环境已设此变量
    or ""
)
BASE_URL = (
    os.getenv("BAILIAN_BASE_URL")
    or os.getenv("ANTHROPIC_BASE_URL")
    or "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
MODEL = os.getenv("BAILIAN_MODEL") or "glm-5.2"

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """懒加载单例 client。"""
    global _client
    if _client is None:
        if not API_KEY:
            raise EnvironmentError(
                "未找到 API key。请设置 BAILIAN_API_KEY 环境变量"
                "（或复用 ANTHROPIC_API_KEY）。"
            )
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _client


def llm_chat(system: str, user: str, *,
            temperature: float = 0.0, max_tokens: int = 1024,
            stop: list[str] | None = None, retries: int = 3) -> str:
    """单轮 LLM 对话。

    stop: ReAct 用 ["Observation:"] 在工具执行前截断 LLM 输出。
    retries: 指数退避重试（网络抖动 / 限流时兜底）。
    """
    for attempt in range(retries):
        try:
            resp = get_client().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning(f"LLM 调用失败，{wait}s 后重试({attempt + 1}): {str(e)[:80]}")
            time.sleep(wait)
    return ""  # 不可达，retries 耗尽已 raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"模型: {MODEL}  base_url: {BASE_URL}")
    r = llm_chat("你是助手", "回复一句话确认你能工作", max_tokens=30)
    print(f"LLM 回复: {r}")
