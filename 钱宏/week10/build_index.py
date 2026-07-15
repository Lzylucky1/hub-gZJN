# -*- coding: utf-8 -*-
"""
向量索引构建脚本（支持 chunks 目录下的 Markdown 分块数据）

Embedding 方案：阿里云 DashScope text-embedding-v3
  - 无需下载本地模型，直接 API 调用
  - 维度：768（可设为 768 / 512 节省存储）
  - 每批最多 10 条（DashScope 限制）
  - 费用极低：约 0.0007 元 / 千 token

向量库：FAISS（IndexFlatIP，内积 = 归一化后的余弦相似度）

数据来源：d:\\code\\ai-study\\week10\\data\\chunks\\ 目录下的 JSON 文件
每个 JSON 文件包含多个 chunk，格式为：
{
    "content": "内容",
    "title_one": "一级标题",
    "title_two": "二级标题",
    "title_three": "三级标题",
    "source_file": "源文件名称",
    "source_path": "源文件绝对路径",
    "chunk_id": "唯一标识符"
}

依赖：
  pip install faiss-cpu openai numpy
  set DASHSCOPE_API_KEY="sk-xxx"
"""

import os
import json
import time
import logging
import numpy as np
from pathlib import Path
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
CHUNKS_DIR      = BASE_DIR / "data" / "chunks"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL     = "text-embedding-v3"
EMBED_DIM       = 768                # 可选 768 / 512 节省存储 1024
BATCH_SIZE      = 10                  # DashScope text-embedding-v3 单次最多 10 条
DASHSCOPE_URL   = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def get_client() -> OpenAI:
    """
    获取 DashScope OpenAI 兼容客户端
    
    Returns:
        OpenAI: 配置好的客户端实例
    """
    DASHSCOPE_API_KEY = "DASHSCOPE_API_KEY"

    # api_key = os.getenv("DASHSCOPE_API_KEY")
    # if not api_key:
    #     raise EnvironmentError(
    #         "请设置环境变量 DASHSCOPE_API_KEY\n"
    #         "  Windows: set DASHSCOPE_API_KEY=sk-xxx\n"
    #         "  Linux/Mac: export DASHSCOPE_API_KEY=sk-xxx"
    #     )
    return OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_URL)


def load_chunks():
    """
    加载 chunks 目录下所有 JSON 文件中的 chunk 数据
    
    Returns:
        list[dict]: 所有 chunk 的列表
    """
    chunks = []
    
    if not CHUNKS_DIR.exists():
        logger.error(f"chunks 目录不存在: {CHUNKS_DIR}")
        return chunks
    
    json_files = sorted([f for f in CHUNKS_DIR.iterdir() if f.suffix == ".json"])
    
    if not json_files:
        logger.error(f"chunks 目录下没有 JSON 文件: {CHUNKS_DIR}")
        return chunks
    
    for json_file in json_files:
        try:
            with open(json_file, encoding="utf-8") as f:
                file_chunks = json.load(f)
            
            if isinstance(file_chunks, list):
                chunks.extend(file_chunks)
                logger.info(f"加载 {json_file.name}: {len(file_chunks)} 个 chunks")
            else:
                logger.warning(f"{json_file.name} 格式不正确，不是列表")
        
        except Exception as e:
            logger.error(f"加载 {json_file.name} 失败: {e}")
    
    logger.info(f"共加载 {len(chunks)} 个 chunks")
    return chunks


def embed_texts(client: OpenAI, texts: list[str], show_progress: bool = True) -> np.ndarray:
    """
    批量计算 embedding，每批最多 10 条。
    返回 shape=(N, EMBED_DIM) 的 float32 数组，已 L2 归一化。
    
    Args:
        client: OpenAI 客户端实例
        texts: 待编码的文本列表
        show_progress: 是否显示进度
    
    Returns:
        np.ndarray: embedding 数组，shape=(N, EMBED_DIM)
    """
    all_embeddings = []
    total_batches  = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(texts), BATCH_SIZE):
        batch     = texts[i : i + BATCH_SIZE]
        batch_idx = i // BATCH_SIZE + 1

        if show_progress and batch_idx % 10 == 0:
            logger.info(f"  Embedding 进度: {batch_idx}/{total_batches} 批")

        for attempt in range(3):
            try:
                resp = client.embeddings.create(
                    model=EMBED_MODEL,
                    input=batch,
                    dimensions=EMBED_DIM,
                )
                vecs = [e.embedding for e in resp.data]
                all_embeddings.extend(vecs)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning(f"  第{attempt+1}次失败，重试: {e}")
                time.sleep(2 ** attempt)

    embeddings = np.array(all_embeddings, dtype="float32")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    embeddings = embeddings / norms

    return embeddings


def build_faiss_index(chunks: list[dict], client: OpenAI):
    """
    构建 FAISS 向量索引。

    FAISS 说明：
      IndexFlatIP = 暴力内积检索，精确但不近似。
      数据量 < 10 万时速度完全够用，是教学的首选。
      数据量更大时可换 IndexIVFFlat（需要 train）或 IndexHNSW。
    
    Args:
        chunks: chunk 列表
        client: OpenAI 客户端实例
    
    Returns:
        tuple: (faiss_index, meta_list)
    """
    import faiss

    logger.info(f"开始计算 {len(chunks)} 条 chunk 的 embedding...")
    texts      = [c["content"] for c in chunks]
    embeddings = embed_texts(client, texts)

    logger.info(f"构建 FAISS 索引，维度={EMBED_DIM}...")
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    logger.info(f"索引构建完成，共 {index.ntotal} 条向量")

    index_path = VECTORSTORE_DIR / "faiss_index.bin"
    meta_path  = VECTORSTORE_DIR / "faiss_meta.json"

    faiss.write_index(index, str(index_path))
    logger.info(f"FAISS 索引已保存 → {index_path}  ({index_path.stat().st_size//1024} KB)")

    meta_list = [
        {
            "chunk_id":     c["chunk_id"],
            "content":      c["content"],
            "title_one":    c.get("title_one", ""),
            "title_two":    c.get("title_two", ""),
            "title_three":  c.get("title_three", ""),
            "source_file":  c.get("source_file", ""),
            "source_path":  c.get("source_path", ""),
        }
        for c in chunks
    ]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)
    logger.info(f"元数据已保存 → {meta_path}")

    return index, meta_list


def main():
    """
    主流程：加载 chunks → 计算 embedding → 构建 FAISS 索引
    """
    chunks = load_chunks()
    
    if not chunks:
        logger.error("没有加载到任何 chunks，退出")
        return

    client = get_client()
    """存入 向量数据库"""
    build_faiss_index(chunks, client)

    logger.info("\n索引构建完成！")
    logger.info(f"  FAISS 索引: {VECTORSTORE_DIR / 'faiss_index.bin'}")
    logger.info(f"  元数据:     {VECTORSTORE_DIR / 'faiss_meta.json'}")


if __name__ == "__main__":
    main()