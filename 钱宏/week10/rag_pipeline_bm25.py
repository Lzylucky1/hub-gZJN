# -*- coding: utf-8 -*-
"""
RAG 检索增强生成管道（BM25 + 向量混合检索）

流程：
1. 加载 FAISS 向量索引和元数据
2. 构建 BM25 索引
3. 将用户问题转换为向量
4. 同时进行向量检索和 BM25 检索
5. 融合两个检索结果，去重后保留 3 条数据
6. 将检索内容和问题一起传递给 qwen-plus 模型
7. 返回模型回答

使用方式：
  python rag_pipeline_bm25.py --question "你的问题"
  或
  python rag_pipeline_bm25.py -q "你的问题"

依赖：
  pip install faiss-cpu openai numpy rank-bm25
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
from pathlib import Path

from openai import OpenAI
from rank_bm25 import BM25Okapi
import jieba

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


def build_bm25_index(meta_list):
    tokenized_corpus = []
    for meta in meta_list:
        content = meta["content"]
        tokens = jieba.lcut(content)
        tokens = [t for t in tokens if t.strip()]
        tokenized_corpus.append(tokens)

    bm25 = BM25Okapi(tokenized_corpus)
    logger.info(f"BM25 索引构建完成，共 {len(tokenized_corpus)} 条文档")
    return bm25


def embed_query(client: OpenAI, query: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=query,
        dimensions=EMBED_DIM,
    )
    vec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)

    norm = np.linalg.norm(vec)
    vec = vec / max(norm, 1e-9)

    return vec


def vector_search(index, meta_list, query_vec, top_k: int = 5):
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
                "chunk_id": meta.get("chunk_id", str(idx)),
                "retrieval_type": "vector"
            })

    return results


def bm25_search(bm25, meta_list, query: str, top_k: int = 5):
    query_tokens = jieba.lcut(query)
    query_tokens = [t for t in query_tokens if t.strip()]
    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if idx >= 0 and idx < len(meta_list):
            meta = meta_list[idx]
            results.append({
                "score": float(scores[idx]),
                "content": meta["content"],
                "title_one": meta.get("title_one", ""),
                "title_two": meta.get("title_two", ""),
                "title_three": meta.get("title_three", ""),
                "source_file": meta.get("source_file", ""),
                "chunk_id": meta.get("chunk_id", str(idx)),
                "retrieval_type": "bm25"
            })

    return results


def fuse_results(vector_results, bm25_results, final_top_k: int = 4, rrf_k: int = 60):
    rrf_scores = {}

    for rank, r in enumerate(vector_results, start=1):
        chunk_id = r["chunk_id"]
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = {
                "content": r["content"],
                "title_one": r["title_one"],
                "title_two": r["title_two"],
                "title_three": r["title_three"],
                "source_file": r["source_file"],
                "chunk_id": chunk_id,
                "vector_rank": rank,
                "bm25_rank": None,
                "rrf_score": 0.0
            }
        rrf_scores[chunk_id]["vector_rank"] = rank
        rrf_scores[chunk_id]["rrf_score"] += 1.0 / (rrf_k + rank)

    for rank, r in enumerate(bm25_results, start=1):
        chunk_id = r["chunk_id"]
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = {
                "content": r["content"],
                "title_one": r["title_one"],
                "title_two": r["title_two"],
                "title_three": r["title_three"],
                "source_file": r["source_file"],
                "chunk_id": chunk_id,
                "vector_rank": None,
                "bm25_rank": rank,
                "rrf_score": 0.0
            }
        rrf_scores[chunk_id]["bm25_rank"] = rank
        rrf_scores[chunk_id]["rrf_score"] += 1.0 / (rrf_k + rank)

    final_results = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)

    return final_results[:final_top_k]


def build_prompt(query: str, context_results: list[dict]) -> str:
    context = "\n\n".join([
        f"【来源】{r['source_file']}\n"
        f"【标题】{r['title_one']}{(' - ' + r['title_two']) if r['title_two'] else ''}{(' - ' + r['title_three']) if r['title_three'] else ''}\n"
        f"【内容】{r['content']}"
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
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=2048,
    )
    return resp.choices[0].message.content.strip()


def rag_pipeline(query: str, top_k: int = 5, final_top_k: int = 3):
    logger.info(f"\n{'=' * 60}")
    logger.info(f"用户问题: {query}")
    logger.info(f"{'=' * 60}")

    logger.info("步骤1: 加载向量索引...")
    index, meta_list = load_vector_index()

    logger.info("步骤2: 构建 BM25 索引...")
    bm25 = build_bm25_index(meta_list)

    logger.info("步骤3: 问题向量化...")
    client = get_client()
    query_vec = embed_query(client, query)

    logger.info("步骤4: 向量检索...")
    vector_results = vector_search(index, meta_list, query_vec, top_k)

    logger.info(f"向量检索结果 ({len(vector_results)} 条):")
    for i, r in enumerate(vector_results[:3], 1):
        logger.info(f"  [{i}] 得分: {r['score']:.4f} - {r['source_file']}")

    logger.info("步骤5: BM25 检索...")
    bm25_results = bm25_search(bm25, meta_list, query, top_k)

    logger.info(f"BM25 检索结果 ({len(bm25_results)} 条):")
    for i, r in enumerate(bm25_results[:3], 1):
        logger.info(f"  [{i}] 得分: {r['score']:.4f} - {r['source_file']}")

    logger.info("步骤6: 融合检索结果并去重...")
    final_results = fuse_results(vector_results, bm25_results, final_top_k)

    logger.info(f"\n融合后保留 {len(final_results)} 条相关内容:")
    for i, r in enumerate(final_results, 1):
        logger.info(f"\n  [{i}] RRF分数: {r['rrf_score']:.4f}")
        logger.info(f"     向量排名: {r['vector_rank'] if r['vector_rank'] else '-'}")
        logger.info(f"     BM25排名: {r['bm25_rank'] if r['bm25_rank'] else '-'}")
        logger.info(f"     来源: {r['source_file']}")
        logger.info(
            f"     标题: {r['title_one']}{(' - ' + r['title_two']) if r['title_two'] else ''}{(' - ' + r['title_three']) if r['title_three'] else ''}")
        logger.info(f"     内容预览: {r['content'][:100]}...")

    logger.info("\n步骤7: 构建提示词...")
    prompt = build_prompt(query, final_results)

    logger.info("步骤8: 调用 qwen-plus 模型生成回答...")
    answer = generate_answer(client, prompt)

    logger.info(f"\n{'=' * 60}")
    logger.info("模型回答:")
    logger.info(f"{'=' * 60}")
    logger.info(answer)

    return answer


def main():
    parser = argparse.ArgumentParser(description="RAG 检索增强生成系统（BM25 + 向量混合检索）")
    parser.add_argument("--question", "-q", type=str, help="用户问题")
    parser.add_argument("--top_k", "-k", type=int, default=5, help="单个检索方法返回的条数，默认5条")
    parser.add_argument("--final_top_k", "-f", type=int, default=4, help="融合后最终保留的条数，默认4条")

    args = parser.parse_args()
    if not args.question:
        args.question = "马尔可夫假设"

    rag_pipeline(args.question, args.top_k, args.final_top_k)


if __name__ == "__main__":
    main()