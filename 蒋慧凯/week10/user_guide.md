# Week 10 作业：离线 RAG 问答系统

## 项目简介

基于上市公司年报数据，实现一个完全离线的 RAG 问答系统：

- 使用本地 `BGE-small-zh-v1.5` 做 Embedding 检索
- 使用本地 `Qwen2-0.5B-Instruct` 生成答案
- 无需 API Key，无需联网

## 目录结构

```
src/
└── rag_qa.py          # 主程序
outputs/
└── qa_results.json    # 运行结果
requirements.txt       # Python 依赖
user_guide.md          # 本文件
```

## 环境要求

- Python 3.9+
- PyTorch、Transformers、LangChain、FAISS、Sentence-Transformers
- 推荐 NVIDIA GPU + CUDA；CPU 也可运行

## 输入与输出

- **输入**：`src/rag_qa.py` 中的 `questions` 问题列表
- **输出**：`outputs/qa_results.json`，包含每个问题的答案及引用的年报片段

## 运行步骤

```bash
cd "week10检索增强生成/homework"
pip install -r requirements.txt
python src/rag_qa.py
```

运行完成后，查看 `outputs/qa_results.json`。
