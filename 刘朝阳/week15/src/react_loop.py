"""
通用 ReAct 循环引擎

教学重点：
  1. ReAct = Reason + Act：LLM 输出 Thought(推理) → Action(选工具) → Action Input(参数)，
     runner 执行工具得 Observation，再喂回 LLM 继续，直到 Final Answer
  2. 主 agent 和 subagent 都是 ReAct 循环——区别只在「有哪些工具」：
     主 agent 有 web_search + dispatch_subagents，subagent 只有 web_search
  3. 用 stop=["Observation:"] 让 LLM 在生成完 Action Input 后停下，
     runner 执行工具再补 Observation 续写——ReAct 的经典实现技巧
  4. 完整 trace 捕获：每步 Thought/Action/ActionInput/Observation 存下来，
     供展示「点节点看 ReAct 过程」用

依赖：仅 llm_client + 工具函数，无外部库
"""

import time, re, logging
from typing import Callable, Optional
from llm_client import llm_chat

logger = logging.getLogger(__name__)

REACT_SYSTEM = """你是一个能用工具联网搜索调研的助手。

可用工具：
{tools_desc}

按如下格式严格输出（每轮一次 Thought/Action/Action Input）：
Thought: 你的推理，分析还需查什么
Action: 工具名
Action Input: 工具参数（字符串）

工具执行后会得到 Observation。多轮调用直到能给出完整答案，最后用：
Thought: 我已收集足够信息
Final Answer: 综合答案（带来源要点）

规则：
- Action 必须是上面列出的工具名之一
- Action Input 是该工具的参数字符串
- 每轮只调一次工具，等 Observation 再决定下一步"""


def build_tools_desc(tools: dict) -> str:
    """把 tools 字典格式化成工具说明。tools: {name: (fn, description)}"""
    lines = []
    for name, (fn, desc) in tools.items():
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


class ReActLoop:
    """通用 ReAct 循环。主 agent / subagent 各自实例化一个。"""

    def __init__(self, agent_name: str, tools: dict,
                 max_steps: int = 6,
                 system_prompt: Optional[str] = None,
                 on_step: Optional[Callable] = None):
        """
        tools: {tool_name: (fn(arg, **kw)->str, description_str)}
        system_prompt: 自定义系统提示（主 agent 用 MAIN_SYSTEM 引导派发）。
                       None 时用默认 REACT_SYSTEM。{tools_desc} 占位符会被替换。
        on_step: 每步回调，签名 on_step(step_dict)。
        """
        self.agent_name = agent_name
        self.tools = tools
        self.max_steps = max_steps
        self._system_template = system_prompt or REACT_SYSTEM
        self._on_step = on_step
        self.trace: list[dict] = []

    def run(self, question: str, shared_state: dict | None = None,
            on_step: Optional[Callable] = None) -> dict:
        """执行 ReAct 循环。

        on_step: 本轮回调（优先于构造器设的 self._on_step，供主 agent 派发
                 subagent 时注入带 sid 的闭包回调用）。
        shared_state: 共享状态（主 agent 派发 subagent 时往里塞 subagent trace）。
        返回 {final_answer, trace, duration}。
        """
        cb = on_step if on_step is not None else self._on_step
        self.trace = []
        t0 = time.time()
        system = self._system_template.format(
            tools_desc=build_tools_desc(self.tools)
        )
        history = f"Question: {question}\n\n"
        final_answer = ""

        for step_idx in range(self.max_steps):
            # 调 LLM 生成下一步（停在 Observation: 前）
            llm_out = llm_chat(system, history, temperature=0.0,
                               max_tokens=1024, stop=["Observation:"])
            thought, action, action_input = self._parse(llm_out)

            step = {
                "idx": step_idx, "agent": self.agent_name,
                "thought": thought, "action": action,
                "action_input": action_input, "observation": None,
                "final": False,
            }

            if action == "Final Answer":
                step["final"] = True
                final_answer = action_input
                self.trace.append(step)
                if cb:
                    cb(step)
                break

            # pre 执行：先发 step（observation=None），让调用方马上看到决策
            if cb:
                cb(step)

            # 执行工具
            observation = self._exec_tool(action, action_input, shared_state)

            # post 执行：同一 idx 再发，带真实 observation
            step["observation"] = observation
            step["done"] = True
            self.trace.append(step)
            if cb:
                cb(step)

            # 续写历史（截断防 context 过长）
            history += llm_out + f"Observation: {observation[:1200]}\n"
        else:
            # 超过 max_steps，强制收尾
            final_answer = "（已达最大步数）" + (
                self.trace[-1].get("observation", "") or "" if self.trace else ""
            )
            step = {
                "idx": self.max_steps, "agent": self.agent_name,
                "thought": "达到步数上限", "action": "Final Answer",
                "action_input": final_answer, "observation": None,
                "final": True,
            }
            self.trace.append(step)
            if cb:
                cb(step)

        duration = round(time.time() - t0, 2)
        return {"final_answer": final_answer, "trace": self.trace,
                "duration": duration}

    def _parse(self, text: str) -> tuple[str, str, str]:
        """从 LLM 输出解析 Thought/Action/Action Input。

        返回 (thought, action, action_input)。
        Final Answer 时 action='Final Answer'，内容放 action_input。
        兜底：若没匹配到 Action 也没 Final Answer，但有实质文本，当作 Final Answer
        （LLM 拿到子调研结果后常直接写报告、不带 Final Answer 前缀）。
        """
        thought = ""
        m = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.S)
        if m:
            thought = m.group(1).strip()[:400]

        # Final Answer 优先检测
        mfa = re.search(r"Final Answer:\s*(.*)", text, re.S)
        if mfa:
            return thought, "Final Answer", mfa.group(1).strip()

        # Action / Action Input
        ma = re.search(r"Action:\s*(.*)", text)
        mi = re.search(r"Action Input:\s*(.*)", text)
        if ma:
            action = ma.group(1).strip()
            action_input = (mi.group(1).strip() if mi else "")
            return thought, action, action_input

        # 兜底：有实质文本但无格式标记 → 当作 Final Answer
        if text.strip():
            return thought or "综合调研结果给出报告", "Final Answer", text.strip()
        return thought, "", ""

    def _exec_tool(self, action: str, action_input: str,
                   shared_state: dict | None) -> str:
        """执行工具，返回 observation 文本。"""
        if action not in self.tools:
            return f"工具 '{action}' 不存在，可选: {list(self.tools.keys())}"
        fn, _ = self.tools[action]
        try:
            return str(
                fn(action_input, shared_state=shared_state)
                if shared_state is not None
                else fn(action_input)
            )
        except Exception as e:
            return f"工具执行出错: {type(e).__name__}: {str(e)[:120]}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    from web_search import search_and_format

    def web_search_tool(q, **_):
        return search_and_format(q)

    loop = ReActLoop(
        "test",
        tools={"web_search": (web_search_tool, "联网搜索，参数是查询词")},
        max_steps=4,
    )
    r = loop.run("What is Python programming language?")
    print(f"\n答案: {r['final_answer'][:200]}")
    print(f"trace {len(r['trace'])} 步, 耗时 {r['duration']}s")
    for s in r["trace"]:
        print(f"  [{s['idx']}] {s['action']}({(s['action_input'] or '')[:40]}) "
              f"→ {(s.get('observation') or '')[:50]}")
