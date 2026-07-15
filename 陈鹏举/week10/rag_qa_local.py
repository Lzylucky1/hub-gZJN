import json
import numpy as np
import faiss
from pathlib import Path
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

BASE_DIR = Path(__file__).parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
INDEX_PATH = VECTORSTORE_DIR / "faiss_index.bin"
META_PATH = VECTORSTORE_DIR / "faiss_meta.json"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 约1.1B参数，CPU可跑
# 如果有GPU，可换 "Qwen/Qwen2-1.5B-Instruct" 效果更好

TOP_K = 3

# 加载模型（全局只加载一次）
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# 加载生成模型（使用pipeline）
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else "cpu"
)
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    do_sample=True,
)

# 加载索引
index = faiss.read_index(str(INDEX_PATH))
with open(META_PATH, encoding="utf-8") as f:
    meta_list = json.load(f)

def embed_query(query):
    vec = embed_model.encode([query], normalize_embeddings=True)
    return np.array(vec, dtype="float32")

def retrieve(query_vec, k=TOP_K):
    distances, indices = index.search(query_vec, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        meta = meta_list[idx]
        results.append({
            "content": meta["content"],
            "score": float(dist),
            "source": meta.get("source_file", ""),
        })
    return results

def generate_answer(query, retrieved_chunks):
    context = "\n\n".join([f"【片段{i+1}】{c['content']}" for i, c in enumerate(retrieved_chunks)])
    prompt = f"根据以下信息回答问题。如果信息不足，请说明未找到。\n\n{context}\n\n问题：{query}\n回答："
    output = generator(prompt)[0]["generated_text"]
    # 截取回答部分（简单处理）
    answer = output.split("回答：")[-1].strip()
    return answer

def main():
    print("本地 RAG 问答系统（免费）启动，输入 q 退出")
    while True:
        query = input("\n问题：").strip()
        if query.lower() in ("q", "exit"):
            break
        if not query:
            continue

        q_vec = embed_query(query)
        retrieved = retrieve(q_vec)
        print("\n检索到的片段：")
        for i, item in enumerate(retrieved):
            print(f"  [{i+1}] 相似度 {item['score']:.4f} | {item['source']}")
            print(f"      {item['content'][:80]}...")

        print("\n生成回答中...")
        answer = generate_answer(query, retrieved)
        print(f"\n回答：{answer}")

if __name__ == "__main__":
    main()
