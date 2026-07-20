# Week12 作业：ReAct Agent 多轮对话能力

## 项目简介

本作业在 ReAct Financial Agent 基础上，增加**多轮对话能力**。核心改动：

- 新增 `src/memory.py` 会话级记忆模块，保存完整对话历史并管理上下文窗口。
- 改造 `src/react_manual.py` 和 `src/react_function_calling.py`，支持传入 `ConversationMemory`。
- 新增 `src/chat.py` 交互式命令行多轮对话入口。
- 改造 `src/serve.py`，新增 `/chat/manual/{session_id}` 和 `/chat/fc/{session_id}` SSE 多轮接口。
- 新增 `src/evaluate_multi_turn.py` 评估脚本，构造跨轮指代、话题切换等场景验证记忆有效性。

## 目录结构

```
week12agent/homework/
├── run_all.py                  # 作业统一入口
├── requirements.txt            # 依赖列表
├── user_guide.md               # 本文件
├── index.html                  # Web UI（多轮对话）
├── src/
│   ├── tools.py                # 5个工具实现
│   ├── memory.py               # 多轮对话记忆模块（新增）
│   ├── react_manual.py         # 手写 Prompt 解析版（支持记忆）
│   ├── react_function_calling.py # Function Calling 版（支持记忆）
│   ├── chat.py                 # 交互式 CLI（新增）
│   ├── serve.py                # FastAPI + SSE 服务（支持多轮）
│   ├── evaluate.py             # 单轮对比评估（原项目保留）
│   └── evaluate_multi_turn.py  # 多轮对话评估（新增）
└── outputs/
    ├── logs/                   # 评估结果 JSON
    └── figures/                # 可视化图表
```

## 环境要求

- Python 3.10+
- 依赖：`openai>=1.0.0`, `faiss-cpu>=1.7.0`, `akshare>=1.10.0`, `fastapi>=0.110.0`, `uvicorn>=0.29.0`, `numpy>=1.24.0`
- 环境变量：
  - `DASHSCOPE_API_KEY`：必填，用于 LLM 推理和 RAG Embedding
  - `AGENT_MODEL`：可选，默认 `qwen-max`

## 安装依赖

```bash
cd week12agent/homework
pip install -r requirements.txt
```

## 运行方式

### 1. 运行多轮对话评估（作业主入口）

```bash
python run_all.py
```

结果保存至 `outputs/logs/multi_turn_result.json`。

### 2. 交互式命令行多轮对话

```bash
python src/chat.py
python src/chat.py --mode fc
```

交互命令：
- `/new`：开启新会话
- `/history`：查看当前会话历史
- `/quit`：退出

### 3. 启动 Web 服务

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

访问 `http://localhost:8000` 使用多轮对话 Web UI。

## 输入输出

- **输入**：多轮对话测试集（`evaluate_multi_turn.py` 中定义），包含跨轮指代、省略、话题切换等场景。
- **输出**：`outputs/logs/multi_turn_result.json`，包含每轮问题、答案、步骤数、耗时、完整会话记忆。
