# Function Call 多轮循环调用 — 作业说明

## 项目简介

将课程参考项目 `function_call_mcp_cli` 中天气查询的工具调用，从"单轮闭环"改造为**多轮循环调用**。

核心改造：
1. 天气查询后端拆为 `get_coordinates` + `get_weather_by_coords` 两个独立工具
2. Function Call 闭环从"一次调用→执行→二次回答"改为 `while` 循环，模型可多轮逐步调工具

## 文件结构

```
homework/
├── run_all.py                     # 统一运行入口
├── function_call_loop.py          # 多轮循环调用版 Function Call（核心作业）
├── weather_backend_mod.py         # 拆为 get_coordinates + get_weather_by_coords
├── requirements.txt               # 依赖
├── user_guide.md                  # 本文件
└── outputs/                       # 运行结果输出
```

## 环境要求

- Python >= 3.10
- 依赖：`openai`、`httpx`
- 环境变量：
  - `DEEPSEEK_API_KEY`（默认 LLM）
  - `DASHSCOPE_API_KEY`（RAG 检索 Embedding 用）

## 使用方式

```bash
# 安装依赖
pip install -r requirements.txt

# 单问题测试
python function_call_loop.py -q "宁德天气如何？"

# 跑全部 demo 问题
python run_all.py

# JSON 输出
python run_all.py --json
```

本作业依赖参考项目的 FAISS 向量索引和 RAG 检索能力。首次运行前需确保参考项目的向量索引已就绪。
