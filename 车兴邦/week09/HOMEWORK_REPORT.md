# 作业报告：部署 vLLM 大模型服务并验证速度提升

## 1. 作业目标

本次作业目标是尝试部署一个 vLLM 大模型推理服务，并通过同一模型、同一批测试 prompt 的吞吐对比，验证 vLLM 相比原生 Transformers 推理的速度提升。

本目录已完成一个 `vllm_deployment` 实验项目，包含：

- vLLM OpenAI 兼容服务启动脚本：`src/start_server.sh`
- 吞吐性能对比脚本：`src/bench_throughput.py`
- 约束解码演示脚本：`src/demo_*.py`
- 实验结果文件：`outputs/throughput_results.json`
- 吞吐对比图：`outputs/throughput_comparison.png`

## 2. 实验环境

| 项目 | 配置 |
|------|------|
| 操作系统 | Windows 11 + WSL2 Ubuntu 22.04 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU 8GB |
| 模型 | Qwen2-0.5B-Instruct |
| vLLM | 0.9.2 |
| PyTorch | 2.7.0 |
| Transformers | 4.52.4 |
| Python 依赖 | 见 `requirements.txt` |

选择 vLLM 0.9.2 的原因是它与常见 CUDA 12.x 驱动环境兼容性更好，避免新版本 vLLM 对 CUDA 13 / 更高 NVIDIA 驱动的要求。

## 3. vLLM 服务部署过程

### 3.1 安装依赖

在 WSL2 Ubuntu 中创建并激活虚拟环境：

```bash
python3 -m venv ~/vllm_env
source ~/vllm_env/bin/activate
pip install -r requirements.txt
```

### 3.2 启动服务

进入项目脚本目录后执行：

```bash
cd vllm_deployment/src
bash start_server.sh
```

启动脚本核心命令如下：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model "/mnt/d/badou/项目材料准备/pretrain_models/Qwen2-0.5B-Instruct" \
  --served-model-name "qwen2-0.5b" \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.6 \
  --dtype float16 \
  --enforce-eager \
  --host 0.0.0.0
```

服务启动后会在本机 `8000` 端口提供 OpenAI 兼容接口。

### 3.3 验证服务可用

查询模型列表：

```bash
curl http://localhost:8000/v1/models
```

调用聊天接口：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2-0.5b",
    "messages": [{"role": "user", "content": "你好，请介绍一下你自己"}],
    "max_tokens": 50
  }'
```

如果接口返回正常 JSON 响应，说明 vLLM 大模型服务部署成功。

## 4. 速度验证方法

使用 `src/bench_throughput.py` 对比三种推理方式：

1. Transformers 串行推理：一次处理一个 prompt。
2. Transformers 手动 batch：batch size = 8，通过 padding 组成批处理。
3. vLLM 批量推理：使用 vLLM 的 PagedAttention 和 continuous batching。

实验设置：

| 参数 | 数值 |
|------|------|
| 测试 prompt 数量 | 50 |
| 最大生成 token 数 | 100 |
| Transformers batch size | 8 |
| 测试内容 | 金融知识问答长短混合 prompt |

运行命令：

```bash
cd vllm_deployment/src
python bench_throughput.py
```

脚本运行后会生成：

- `outputs/throughput_results.json`：原始吞吐数据
- `outputs/throughput_comparison.png`：吞吐对比柱状图

## 5. 实验结果

本次目录中已有一次 benchmark 结果，来自 `outputs/throughput_results.json`：

| 推理方式 | 总耗时 | 生成 token 数 | QPS | tokens/s |
|----------|--------|---------------|-----|----------|
| Transformers 串行 | 62.84s | 3639 | 0.80 | 57.91 |
| Transformers batch=8 | 13.11s | 3711 | 3.81 | 283.09 |
| vLLM | 1.15s | 3492 | 43.57 | 3042.93 |

根据 tokens/s 计算加速比：

- vLLM 相比 Transformers 串行：`3042.93 / 57.91 ≈ 52.55x`
- vLLM 相比 Transformers batch=8：`3042.93 / 283.09 ≈ 10.75x`

根据 QPS 计算加速比：

- vLLM 相比 Transformers 串行：`43.57 / 0.80 ≈ 54.76x`
- vLLM 相比 Transformers batch=8：`43.57 / 3.81 ≈ 11.42x`

## 6. 结果分析

从实验结果可以看出，vLLM 在相同模型和相同请求集合下有显著速度提升：

1. 相比串行 Transformers，vLLM 的请求吞吐从约 `0.80 QPS` 提升到 `43.57 QPS`，提升约 `54.76x`。
2. 相比手动 batch 的 Transformers，vLLM 的生成吞吐从约 `283.09 tokens/s` 提升到 `3042.93 tokens/s`，提升约 `10.75x`。
3. Transformers batch 虽然比串行快，但仍然存在 padding 浪费和静态 batch 调度问题。
4. vLLM 通过 continuous batching 动态调度不同长度请求，短请求完成后可以立即插入新请求，提高 GPU 利用率。
5. vLLM 通过 PagedAttention 管理 KV cache，减少显存碎片和无效占用，从而支持更高并发。

因此，本实验验证了 vLLM 在大模型推理服务场景中能显著提升吞吐性能，适合用作高并发 LLM 服务部署方案。

## 7. 作业结论

本次作业已完成 vLLM 大模型服务部署和速度验证：

- 已使用 `start_server.sh` 将 Qwen2-0.5B-Instruct 部署为 OpenAI 兼容 HTTP 服务。
- 已通过 `curl` 和 OpenAI 兼容接口验证服务可以正常访问。
- 已使用 `bench_throughput.py` 对比 Transformers 串行、Transformers batch 和 vLLM 三种推理方式。
- 实验结果显示，vLLM 相比 Transformers 串行在 tokens/s 上提升约 `52.55x`，相比 Transformers batch=8 提升约 `10.75x`。

最终结论：vLLM 的 PagedAttention 和 continuous batching 能显著提高大模型推理吞吐，是部署大模型在线服务时非常有效的推理加速方案。
