"""
主 Agent + 并行 Subagent 编排

教学重点：
  1. 主 agent 自己是 ReAct 循环，有 2 个工具：
     - web_search：单次联网搜索（简单问题直接用）
     - dispatch_subagents：派发多个 subagent 并行调研（多侧面研究问题用）
     主 agent 根据 query 自行决定用哪个——不是固定拓扑，是 LLM 自主路由
  2. 并行优势凸显：dispatch_subagents 一次派发 N 个 subagent，
     ThreadPoolExecutor 并行跑，wall-clock ≈ max(单 agent 时长)，
     而非 sum——这就是 subagent 并行的核心价值
  3. 每个 subagent 也是 ReAct 循环（只 web_search 工具），
     trace 全程捕获存入 shared_state，供展示用

架构对应 Orchestrator-Workers 拓扑（动态：主 agent 决定派几个）。
"""

import time, uuid, logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from react_loop import ReActLoop
from web_search import search_and_format

logger = logging.getLogger(__name__)

# ── 主 agent 的系统提示：引导它「多侧面问题 → 派发 subagent 并行」 ──────
MAIN_SYSTEM = """你是一个市场调研主分析师。你有 2 个工具：
- web_search：联网搜索一次（参数=查询词）。仅用于单一事实可一次答出的问题
- dispatch_subagents：派发多个子调研员并行调研（参数=用 | 分隔的多个子课题）

【关键决策原则】
- 只要问题涉及 2 个及以上侧面（如「市场调研」「竞品分析」「行业分析」「XX 概况/现状/趋势」等），
  必须用 dispatch_subagents 把各侧面拆给子调研员并行处理，不要自己串行 web_search 多次。
  示例："新能源汽车市场调研：销量、竞争、政策" → Action: dispatch_subagents
        Action Input: 2024年中国新能源汽车销量规模 | 主要厂商竞争格局 | 政策与补贴趋势
- 只有单一事实问题（如"2024年比亚迪销量"）才直接 web_search
- 拿到子调研结果后，综合成结构化报告

报告要求：分维度组织，每个要点带来源，末尾给结论与不确定性说明。

【示例】
Question: 2023中国咖啡市场调研：市场规模、主要品牌、消费趋势
Thought: 这是多维度市场调研（3个侧面），必须派发子调研员并行收集，不能自己串行搜索
Action: dispatch_subagents
Action Input: 2023年中国咖啡市场规模与增长 | 中国咖啡主要品牌竞争格局 | 中国咖啡消费趋势与人群
Observation: 并行调研完成：3 个子调研员...（各子课题结果）
Thought: 已收齐三个维度的并行调研结果，综合成报告
Final Answer: （分维度报告）"""


def _dispatch_subagents(
    action_input: str,
    shared_state: dict | None = None,
    *,
    on_subagent_step: Optional[Callable] = None,
    on_subagent_done: Optional[Callable] = None,
    on_dispatch: Optional[Callable] = None,
    serial: bool = False,
) -> str:
    """dispatch_subagents 工具实现。

    action_input: "子课题1 | 子课题2 | ..."（管道分隔）
    派发 N 个 subagent 并行（ThreadPoolExecutor），收齐返回汇总文本。
    serial=True 时改成串行执行（对比基线，凸显并行加速）。

    用真实 subagent id 发 dispatch 事件，确保和后续 subagent_step 事件的 id 一致。
    """
    subtopics = [s.strip() for s in action_input.split("|") if s.strip()][:6]
    if not subtopics:
        return "未解析出子课题"

    shared_state = shared_state if shared_state is not None else {}
    shared_state.setdefault("subagents", {})

    # 构造 (sid, subagent, subtopic) 三元组
    defs = []
    for topic in subtopics:
        sid = f"sub_{uuid.uuid4().hex[:6]}"
        sub = ReActLoop(
            agent_name=sid,
            tools={
                "web_search": (
                    lambda q, **_: search_and_format(q),
                    "联网搜索，参数是查询词",
                )
            },
            max_steps=4,
        )
        defs.append((sid, sub, topic))

    # 记录派发（拓扑可视化用：主→N 个子节点）
    dispatch_info = {
        "subtopics": subtopics,
        "subagent_ids": [sid for sid, _, _ in defs],
    }
    shared_state.setdefault("dispatches", []).append(dispatch_info)
    if on_dispatch:
        on_dispatch(dispatch_info)

    t0 = time.time()
    results: dict[str, tuple] = {}

    def _run_one(sid: str, sub: ReActLoop, topic: str):
        """跑一个 subagent，带 step 回调（闭包捕获 sid，回传给调用方）。"""
        if on_subagent_step:
            return sid, sub.run(
                topic,
                on_step=lambda step, _sid=sid: on_subagent_step(_sid, step),
            )
        return sid, sub.run(topic)

    if serial:
        # 串行：一个接一个（eval A/B 对比基线，凸显并行意义）
        for sid, sub, topic in defs:
            sid, res = _run_one(sid, sub, topic)
            results[sid] = (topic, res)
            shared_state["subagents"][sid] = {
                "subtopic": topic,
                "trace": res["trace"],
                "duration": res["duration"],
                "final_answer": res["final_answer"],
            }
            if on_subagent_done:
                on_subagent_done(sid, res["duration"], topic)
    else:
        # 并行（凸显 subagent 并行优势的核心）
        with ThreadPoolExecutor(max_workers=len(defs)) as pool:
            futs = {
                pool.submit(_run_one, sid, sub, topic): sid
                for sid, sub, topic in defs
            }
            for fut in as_completed(futs):
                sid, res = fut.result()
                topic = next(t for s, _, t in defs if s == sid)
                results[sid] = (topic, res)
                shared_state["subagents"][sid] = {
                    "subtopic": topic,
                    "trace": res["trace"],
                    "duration": res["duration"],
                    "final_answer": res["final_answer"],
                }
                if on_subagent_done:
                    on_subagent_done(sid, res["duration"], topic)

    wall = round(time.time() - t0, 2)
    serial_sum = round(sum(r["duration"] for _, r in results.values()), 2)
    stats = {
        "n_subagents": len(defs),
        "wall_clock": wall,
        "serial_sum": serial_sum,
        "speedup": round(serial_sum / wall, 2) if wall else 0,
    }
    shared_state.setdefault("parallel_stats", []).append(stats)

    # 汇总文本（喂回主 agent 当 Observation，每个子结果截短避免 context 过长）
    parts = [
        f"【子课题: {topic}】(用时{r['duration']}s)\n{r['final_answer'][:500]}"
        for sid, (topic, r) in results.items()
    ]
    return (
        f"并行调研完成：{len(defs)} 个子调研员，wall-clock {wall}s "
        f"(串行需 {serial_sum}s，加速 {stats['speedup']}×)\n\n"
        + "\n\n".join(parts)
    )


def run_research(
    question: str,
    *,
    on_main_step: Optional[Callable] = None,
    on_subagent_step: Optional[Callable] = None,
    on_subagent_done: Optional[Callable] = None,
    on_dispatch: Optional[Callable] = None,
    serial: bool = False,
) -> dict:
    """执行一次市场调研。

    返回 {final_answer, main_trace, subagents, parallel_stats, dispatches}。
    serial=True 时 subagent 串行执行（对比基线）。
    """
    shared_state: dict = {
        "subagents": {}, "dispatches": [], "parallel_stats": [],
    }

    def dispatch_tool(action_input: str, shared_state: dict | None = None):
        return _dispatch_subagents(
            action_input,
            shared_state=shared_state or {},
            on_subagent_step=on_subagent_step,
            on_subagent_done=on_subagent_done,
            on_dispatch=on_dispatch,
            serial=serial,
        )

    main = ReActLoop(
        agent_name="main",
        tools={
            "web_search": (
                lambda q, **_: search_and_format(q),
                "联网搜索一次，参数=查询词",
            ),
            "dispatch_subagents": (
                dispatch_tool,
                "派发多个子调研员并行调研，参数=用 | 分隔的多个子课题",
            ),
        },
        max_steps=8,
        system_prompt=MAIN_SYSTEM,
        on_step=on_main_step,
    )

    result = main.run(question, shared_state=shared_state)
    return {
        "final_answer": result["final_answer"],
        "main_trace": result["trace"],
        "subagents": shared_state["subagents"],
        "parallel_stats": shared_state["parallel_stats"],
        "dispatches": shared_state["dispatches"],
    }
