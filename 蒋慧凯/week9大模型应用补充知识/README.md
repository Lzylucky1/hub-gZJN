# Week 9 作业：vLLM 大模型服务部署与约束解码

本作业基于 `vllm_deployment` 教学项目，完成 vLLM 推理服务的本地部署，并验证 4 种约束解码方式在 Function Call 场景下的可靠性提升。

---

## 一、作业目标

1. **部署 vLLM OpenAI 兼容服务**：在本地（WSL2）启动 Qwen2-0.5B-Instruct 推理服务。
2. **验证约束解码**：运行 4 种约束解码演示（choice / regex / json / response_format）。
3. **量化 Function Call 可靠性**：对比裸 prompt、`response_format`、`guided_json` 三模式的 schema 通过率。
4. **可选加分项**：运行吞吐对比 `bench_throughput.py`，生成自己的 `throughput_comparison.png`。

---

## 二、目录结构

```
homework/
├── src/                          # 演示与测试脚本
│   ├── config.py                 # 统一配置（模型路径、API 地址）
│   ├── start_server.sh           # 启动 vLLM server
│   ├── demo_guided_choice.py     # 枚举约束
│   ├── demo_guided_regex.py      # 正则约束
│   ├── demo_guided_json.py       # JSON Schema 约束
│   ├── demo_response_format.py   # OpenAI 标准 JSON 模式
│   ├── demo_function_call.py     # Function Call 可靠性对比（核心）
│   └── bench_throughput.py       # 吞吐对比（需停 server 单独跑）
├── outputs/                      # 运行输出
│   ├── screenshots/              # 部署成功截图（建议提交）
│   ├── function_call_results.json
│   └── throughput_results.json
│   └── throughput_comparison.png
├── requirements.txt              # Python 依赖
├── run_all.py                    # 一键运行 5 个 demo（server 启动后）
├── README.md                     # 项目说明（本文档）
└── user_guide.md                 # 详细使用指南与踩坑记录
```

---

## 三、快速开始

### 1. 环境准备（WSL2 Ubuntu 22.04）

```bash
# 创建虚拟环境
python3 -m venv ~/vllm_env
source ~/vllm_env/bin/activate

# 安装依赖
cd /mnt/.../week9大模型应用补充知识/homework
pip install -r requirements.txt
```

### 2. 配置模型路径

```bash
# 方式一：环境变量（推荐）
export MODEL_PATH="/mnt/d/badou/项目材料准备/pretrain_models/Qwen2-0.5B-Instruct"

# 方式二：修改 src/config.py 中的 DEFAULT_MODEL_PATH
```

### 3. 启动服务并运行

```bash
# 终端 1：启动 vLLM server
bash src/start_server.sh

# 终端 2：运行全部 demo
python run_all.py

# 终端 2：单独跑某个 demo
python src/demo_function_call.py

# 吞吐对比（需先停 server）
pkill -f 'vllm.entrypoints'
python src/bench_throughput.py
```

---

## 四、实验结果（参考）

> 以下数据来自 `vllm_deployment` 示例输出，实际运行结果可能因硬件/驱动/随机性略有差异。

### 4.1 吞吐对比

| 模式 | 50 请求总耗时 | QPS | Generation tok/s |
|------|--------------|-----|------------------|
| transformers 串行 | 60.98s | 0.82 | 60 |
| transformers batch=8 | 12.85s | 3.89 | 289 |
| vLLM continuous batching | 1.03s | 48.59 | 3394 |

### 4.2 Function Call 可靠性

**get_stock_quote（50 条）**

| 指标 | 裸 prompt | response_format | guided_json |
|------|----------|-----------------|-------------|
| JSON 语法合法 | 86% | 100% | 100% |
| 必选字段齐全 | 86% | 100% | 100% |
| 完整 schema 通过 | 60% | 68% | **100%** |

**create_order（50 条）**

| 指标 | 裸 prompt | response_format | guided_json |
|------|----------|-----------------|-------------|
| JSON 语法合法 | 96% | 100% | 100% |
| 必选字段齐全 | 96% | 100% | 100% |
| 完整 schema 通过 | 42% | 42% | **100%** |

### 4.3 核心结论

1. **vLLM 部署可以大幅提升吞吐**：相对 transformers 串行加速约 60×，来自 PagedAttention + Continuous Batching。
2. **response_format 只解决 JSON 语法合法**：字段名、枚举值、正则、数值范围仍可能出错。
3. **guided_json 解决格式+语义约束**：在解码阶段用 FSM 屏蔽非法 token，是 Function Call / Agent 场景的基础设施。

---

## 五、提交说明

按八斗作业规范，需提交到个人目录。请保留以下证据：

- `outputs/screenshots/`：server 启动成功、API 调用成功、约束解码结果等截图。
- `outputs/function_call_results.json`：自己运行生成的结果。
- `outputs/throughput_results.json` 与 `throughput_comparison.png`：可选但加分。
- 源码与文档：`src/`、`README.md`、`user_guide.md`。

大文件（模型权重、tokenizer 文件）不提交。

---

## 六、注意事项

- 本项目必须在 **WSL2/Linux + NVIDIA GPU** 环境下运行，纯 Windows Python 无法安装 vLLM。
- 如果还没有 WSL2，请参考 [`WSL2_INSTALL_GUIDE.md`](WSL2_INSTALL_GUIDE.md)。
- 如果还没有模型权重，可运行 `python src/download_model.py` 下载（或从课程资料中获取）。
- vLLM 版本、torch 版本、CUDA 驱动版本必须匹配，详见 `user_guide.md`。
- 提交前请参考 [`SUBMISSION.md`](SUBMISSION.md)。
- 本作业为教学演示，请勿直接用于生产环境。
