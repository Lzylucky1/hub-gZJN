"""
多轮对话记忆模块

核心功能：
  1. 维护同一 session 内的完整对话历史（含 ReAct 中间 Thought/Action/Observation）
  2. 提供上下文窗口管理：超过 max_tokens 时按轮次滑动截断，保留最近 N 轮
  3. 支持跨轮引用：后续问题可自动继承前文提到的股票代码、公司名称等上下文

使用方式：
  from memory import ConversationMemory

  mem = ConversationMemory(session_id="user_001")
  mem.add_user_message("茅台和五粮液2023年毛利率差多少？")
  # ... 一轮 ReAct 结束后 ...
  mem.add_turn_summary(
      question="茅台和五粮液2023年毛利率差多少？",
      answer="茅台 91.96%，五粮液 75.79%，差 16.17 个百分点。"
  )
  # 下一轮
  mem.add_user_message("那净利率呢？")  # 可自动理解"那"指代前文两家公司
"""

import uuid
import copy
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ConversationMemory:
    """会话级记忆，保存多轮对话中的消息历史并提供上下文管理。"""

    def __init__(
        self,
        session_id: str | None = None,
        max_turns: int = 5,
        max_context_chars: int = 20000,
    ):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.max_turns = max_turns
        self.max_context_chars = max_context_chars
        # messages 只保存可直接喂给 LLM 的 role/content 消息
        self.messages: List[Dict[str, Any]] = []
        # turns 保存每轮的用户问题 + 最终答案摘要，便于展示和截断
        self.turns: List[Dict[str, str]] = []

    # ────────────────────────── 基础读写 ──────────────────────────

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self._maybe_evict()

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self._maybe_evict()

    def add_tool_observation(self, observation: str) -> None:
        """手写 Prompt 解析版：将工具返回值作为 user 角色消息追加（Observation: ...）。"""
        self.messages.append({"role": "user", "content": f"Observation: {observation}"})
        self._maybe_evict()

    def add_assistant_tool_calls(self, content: str, tool_calls: list[dict]) -> None:
        """Function Calling 版：保存模型返回的带 tool_calls 的 assistant 消息。"""
        self.messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        })
        self._maybe_evict()

    def add_tool_message(self, content: str, tool_call_id: str) -> None:
        """Function Calling 版所需：将工具结果以 role='tool' 形式存入。"""
        self.messages.append({
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call_id,
        })
        self._maybe_evict()

    def add_turn_summary(self, question: str, answer: str) -> None:
        """记录每轮问答摘要，便于后续做长期记忆或对话展示。"""
        self.turns.append({"question": question, "answer": answer})
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    # ────────────────────────── 上下文组装 ──────────────────────────

    def get_context_messages(self, system_prompt: str | None = None) -> List[Dict[str, Any]]:
        """返回可直接传给 LLM 的 messages 列表，含可选 system prompt。

        当总字符数超过 max_context_chars 时，从最早的消息开始丢弃，
        但始终保留 system prompt 和最近的用户问题。
        """
        result = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})
        result.extend(self.messages)

        # 按字符数做上下文截断：token 数与字符数比例对中文约 1:1~1.5
        # 保留 system prompt 后从第 1 条开始丢弃，避免超长报错
        while len(result) > 2:
            total_chars = sum(len(str(m.get("content", ""))) for m in result)
            if total_chars <= self.max_context_chars:
                break
            removed = result.pop(1)
            logger.debug(f"上下文截断，移除: {removed.get('role')} 长度 {len(str(removed.get('content','')))}")

        return result

    def get_recent_summary(self, n: int = 3) -> str:
        """生成最近 N 轮对话的摘要，用于做跨轮指代消解（如'那它呢'）。"""
        if not self.turns:
            return ""
        recent = self.turns[-n:]
        lines = ["【对话历史】"]
        for idx, turn in enumerate(recent, 1):
            lines.append(f"Q{idx}: {turn['question']}")
            lines.append(f"A{idx}: {turn['answer']}")
        return "\n".join(lines)

    # ────────────────────────── 窗口管理 ──────────────────────────

    def _maybe_evict(self) -> None:
        """按轮次滑动截断：保留最近 max_turns 轮用户问题，避免上下文无限增长。"""
        # 只统计真正的用户问题（不是 ReAct 中间 Observation）
        user_question_indices = [
            i for i, m in enumerate(self.messages)
            if m["role"] == "user" and not m["content"].startswith("Observation:")
        ]
        if len(user_question_indices) <= self.max_turns:
            return
        # 移除最旧的一整轮：从第 0 条到第二个用户问题之前
        cutoff = user_question_indices[1]
        self.messages = self.messages[cutoff:]

    def clear(self) -> None:
        self.messages.clear()
        self.turns.clear()

    # ────────────────────────── 序列化 ──────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "max_turns": self.max_turns,
            "max_context_chars": self.max_context_chars,
            "messages": copy.deepcopy(self.messages),
            "turns": copy.deepcopy(self.turns),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMemory":
        mem = cls(
            session_id=data.get("session_id"),
            max_turns=data.get("max_turns", 5),
            max_context_chars=data.get("max_context_chars", 20000),
        )
        mem.messages = data.get("messages", [])
        mem.turns = data.get("turns", [])
        return mem
