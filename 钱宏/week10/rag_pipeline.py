# -*- coding: utf-8 -*-
"""
RAG 检索增强生成管道

流程：
1. 加载 FAISS 向量索引和元数据
2. 将用户问题转换为向量
3. 在向量数据库中检索相似内容
4. 将检索内容和问题一起传递给 qwen-plus 模型
5. 返回模型回答

使用方式：
  python rag_pipeline.py --question "你的问题"
  或
  python rag_pipeline.py -q "你的问题"

依赖：
  pip install faiss-cpu openai numpy
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
from pathlib import Path


from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

EMBED_MODEL = "text-embedding-v3"
EMBED_DIM = 768
LLM_MODEL = "qwen-plus"
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def get_client() -> OpenAI:
    DASHSCOPE_API_KEY = "DASHSCOPE_API_KEY"
    return OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_URL)


def load_vector_index():
    import faiss

    index_path = VECTORSTORE_DIR / "faiss_index.bin"
    meta_path = VECTORSTORE_DIR / "faiss_meta.json"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS 索引文件不存在: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"元数据文件不存在: {meta_path}")

    logger.info(f"加载 FAISS 索引: {index_path}")
    index = faiss.read_index(str(index_path))
    logger.info(f"索引加载完成，共 {index.ntotal} 条向量")

    logger.info(f"加载元数据: {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        meta_list = json.load(f)

    return index, meta_list


def embed_query(client: OpenAI, query: str) -> np.ndarray:
    """
        将查询语句转换为向量
    """
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=query,
        dimensions=EMBED_DIM,
    )
    vec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)

    norm = np.linalg.norm(vec)
    vec = vec / max(norm, 1e-9)

    return vec


def search(index, meta_list, query_vec, top_k: int = 5):
    """
    在向量数据库中检索相似内容
    :param index:   FAISS 索引
    :param meta_list:   元数据列表
    :param query_vec:    查询向量
    :param top_k:    检索数量
    :return: 相似内容列表
    """
    distances, indices = index.search(query_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx >= 0 and idx < len(meta_list):
            meta = meta_list[idx]
            results.append({
                "score": float(dist),
                "content": meta["content"],
                "title_one": meta.get("title_one", ""),
                "title_two": meta.get("title_two", ""),
                "title_three": meta.get("title_three", ""),
                "source_file": meta.get("source_file", ""),
            })

    return results


def build_prompt(query: str, context_results: list[dict]) -> str:
    context = "\n\n".join([
        f"【来源】{r['source_file']}\n"
        f"【标题】{r['title_one']}{(' - ' + r['title_two']) if r['title_two'] else ''}{(' - ' + r['title_three']) if r['title_three'] else ''}\n"
        f"【内容】{r['content']}\n"
        f"【相关度】{r['score']:.4f}"
        for r in context_results
    ])

    prompt = f"""你是一个专业的AI助手，根据提供的参考资料回答用户问题。

参考资料：
{context}

用户问题：{query}

要求：
1. 仅根据参考资料内容进行回答，不要编造信息
2. 如果参考资料中没有相关信息，请明确说明"参考资料中未找到相关信息"
3. 回答要准确、简洁、有条理
4. 如果有多个相关资料，请综合所有信息进行回答
"""

    return prompt


def generate_answer(client: OpenAI, prompt: str) -> str:
    """
        调用 qwen-plus 模型生成回答
        :param client: OpenAI 客户端
        :param prompt: 提示词
        :return:     模型回答
    """
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=2048,
    )
    return resp.choices[0].message.content.strip()


def rag_pipeline(query: str, top_k: int = 5):
    logger.info(f"\n{'=' * 60}")
    logger.info(f"用户问题: {query}")
    logger.info(f"{'=' * 60}")

    logger.info("步骤1: 加载向量索引...")
    index, meta_list = load_vector_index()

    logger.info("步骤2: 问题向量化...")
    client = get_client()
    query_vec = embed_query(client, query)

    logger.info("步骤3: 向量检索...")
    results = search(index, meta_list, query_vec, top_k)

    logger.info(f"\n检索到 {len(results)} 条相关内容:")
    for i, r in enumerate(results, 1):
        logger.info(f"\n  [{i}] 相关度: {r['score']:.4f}")
        logger.info(f"     来源: {r['source_file']}")
        logger.info(
            f"     标题: {r['title_one']}{(' - ' + r['title_two']) if r['title_two'] else ''}{(' - ' + r['title_three']) if r['title_three'] else ''}")
        logger.info(f"     内容预览: {r['content'][:100]}...")

    logger.info("\n步骤4: 构建提示词...")
    prompt = build_prompt(query, results)

    logger.info("步骤5: 调用 qwen-plus 模型生成回答...")
    answer = generate_answer(client, prompt)

    logger.info(f"\n{'=' * 60}")
    logger.info("模型回答:")
    logger.info(f"{'=' * 60}")
    logger.info(answer)

    return answer


def main():
    parser = argparse.ArgumentParser(description="RAG 检索增强生成系统")
    parser.add_argument("--question", "-q", type=str, help="用户问题")
    parser.add_argument("--top_k", "-k", type=int, default=5, help="检索返回的条数，默认5条")

    args = parser.parse_args()
    if not args.question:
        args.question = "马尔可夫假设"

    rag_pipeline(args.question, args.top_k)


if __name__ == "__main__":
    main()
