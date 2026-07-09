import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import logging
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
CHUNKS_FILE = BASE_DIR / "data" / "chunks" / "all_semantic.json"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# 本地模型配置（免费）
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # 轻量，384维
# 或 "BAAI/bge-base-en-v1.5" 效果更好，但需要下载较大模型

def embed_texts(texts, model, batch_size=32):
    """批量计算 embedding，返回归一化向量"""
    embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
    return np.array(embeddings, dtype="float32")

def build_index():
    if not CHUNKS_FILE.exists():
        logger.error(f"找不到 {CHUNKS_FILE}，请先运行 chunk_documents.py")
        return

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"加载 {len(chunks)} 个 chunks")

    # 加载 embedding 模型（首次运行会自动下载）
    logger.info(f"加载 embedding 模型: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # 计算向量
    texts = [c["content"] for c in chunks]
    embeddings = embed_texts(texts, model)

    # 构建 FAISS 索引
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # 内积，由于已经归一化，等价于余弦相似度
    index.add(embeddings)
    logger.info(f"索引构建完成，共 {index.ntotal} 条")

    # 保存索引和元数据
    faiss.write_index(index, str(VECTORSTORE_DIR / "faiss_index.bin"))
    meta_path = VECTORSTORE_DIR / "faiss_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"元数据已保存至 {meta_path}")

if __name__ == "__main__":
    build_index()
