import os
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── 路径配置 ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
CUSTOM_DOCS_DIR = BASE_DIR / "data" / "custom_docs"
EXISTING_CHUNKS_FILE = BASE_DIR / "data" / "chunks" / "all_semantic.json"
VECTOR_DIR = BASE_DIR / "vectorstore" / "custom_api_rag"
INDEX_PATH = VECTOR_DIR / "faiss_index.bin"
META_PATH = VECTOR_DIR / "faiss_meta.json"


# ── API ─────────────────────────

API_KEY = os.getenv("RAG_API_KEY")
BASE_URL = os.getenv("RAG_BASE_URL", "https://api.openai.com/v1")

# 支持两种生成接口：OpenAI-compatible chat 或 Anthropic Messages API。
# 本地 Claude 配置是 Anthropic 接口；embedding 接口不可用时可用 BM25-only 跑通。
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or API_KEY
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "http://model.mify.ai.srv/anthropic")
USE_ANTHROPIC_CHAT = os.getenv("RAG_USE_ANTHROPIC_CHAT", "1") == "1"

EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", os.getenv("ANTHROPIC_DEFAULT_SONNET_MODEL", "ppio/pa/gpt-5.5"))
EMBED_DIM = int(os.getenv("RAG_EMBED_DIM", "1536"))

BATCH_SIZE = int(os.getenv("RAG_EMBED_BATCH_SIZE", "10"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))

TOP_K_VECTOR = 10
TOP_K_BM25 = 10
TOP_K_FINAL = 4
SCORE_THRESHOLD = 0.20


SYSTEM_PROMPT = """你是一个严谨的 RAG 问答助手。

回答规则：
1. 只能根据【参考资料】回答，不要编造资料外信息。
2. 如果资料不足以回答，直接说“根据提供的资料无法回答此问题”。
3. 引用具体信息时，在句末标注来源编号，例如：[1]。
4. 回答要简洁、准确、有条理。"""


# ── API 客户端 ──────────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    if not API_KEY:
        raise EnvironmentError(
            "请先设置环境变量 RAG_API_KEY。\n"
            "例如：set RAG_API_KEY=你的key  或  export RAG_API_KEY=你的key"
        )
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def get_anthropic_headers() -> dict:
    if not ANTHROPIC_API_KEY:
        raise EnvironmentError("请先设置环境变量 ANTHROPIC_API_KEY 或 RAG_API_KEY。")
    return {
        "Authorization": f"Bearer {ANTHROPIC_API_KEY}",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }


def anthropic_messages_url() -> str:
    return ANTHROPIC_BASE_URL.rstrip("/") + "/v1/messages"


# ── 文档读取与分块 ───────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """固定大小分块，简单可靠，适合作业演示。"""
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def load_existing_project_chunks() -> Optional[list[dict]]:
    """如果项目已有 data/chunks/all_semantic.json，直接复用已有分块。"""
    if not EXISTING_CHUNKS_FILE.exists():
        return None

    logger.info(f"发现已有分块文件，优先使用: {EXISTING_CHUNKS_FILE}")
    with open(EXISTING_CHUNKS_FILE, encoding="utf-8") as f:
        raw_chunks = json.load(f)

    chunks = []
    for idx, item in enumerate(raw_chunks):
        content = item.get("content", "").strip()
        if not content:
            continue
        meta = item.get("metadata", {}) or {}
        chunks.append({
            "chunk_id": item.get("chunk_id") or f"existing_{idx:05d}",
            "content": content,
            "source": meta.get("source_file", "all_semantic.json"),
            "page_num": meta.get("page_num", -1),
            "section": meta.get("section", ""),
        })
    return chunks


def load_custom_docs() -> list[dict]:
    """读取 data/custom_docs/ 下的 txt、md、json 文件并分块。"""
    CUSTOM_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    files = []
    for pattern in ("*.txt", "*.md", "*.json"):
        files.extend(CUSTOM_DOCS_DIR.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"没有找到文档。请把 .txt/.md/.json 文件放到: {CUSTOM_DOCS_DIR}\n"
            f"或者先运行项目原有流程生成: {EXISTING_CHUNKS_FILE}"
        )

    chunks = []
    for path in files:
        logger.info(f"读取文档: {path.name}")
        text = path.read_text(encoding="utf-8", errors="ignore")

        # 如果是 JSON，尽量提取可读文本；这里为了单文件作业简洁，直接整体转字符串处理
        if path.suffix.lower() == ".json":
            try:
                obj = json.loads(text)
                text = json.dumps(obj, ensure_ascii=False, indent=2)
            except Exception:
                pass

        for i, content in enumerate(chunk_text(text)):
            chunks.append({
                "chunk_id": f"{path.stem}_{i:05d}",
                "content": content,
                "source": path.name,
                "page_num": -1,
                "section": "",
            })

    return chunks


def load_documents() -> list[dict]:
    chunks = load_existing_project_chunks()
    if chunks is None:
        chunks = load_custom_docs()
    logger.info(f"共加载 {len(chunks)} 个 chunk")
    return chunks


# ── Embedding 与索引构建 ────────────────────────────────────────────────────────

def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    """调用自己的 embedding API，返回 L2 归一化后的向量。"""
    all_vectors = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        batch_idx = start // BATCH_SIZE + 1
        logger.info(f"Embedding: {batch_idx}/{total_batches}")

        for attempt in range(3):
            try:
                kwargs = {
                    "model": EMBED_MODEL,
                    "input": batch,
                }
                # 有些 OpenAI 兼容服务支持 dimensions，有些不支持。
                # 如果你的服务不支持 dimensions，可以加环境变量 RAG_DISABLE_DIMENSIONS=1。
                if os.getenv("RAG_DISABLE_DIMENSIONS", "0") != "1":
                    kwargs["dimensions"] = EMBED_DIM

                resp = client.embeddings.create(**kwargs)
                vectors = [item.embedding for item in resp.data]
                all_vectors.extend(vectors)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning(f"Embedding 第 {attempt + 1} 次失败，准备重试: {e}")
                time.sleep(2 ** attempt)

    embeddings = np.array(all_vectors, dtype="float32")

    if embeddings.shape[1] != EMBED_DIM:
        raise ValueError(
            f"向量维度不匹配：API 返回 {embeddings.shape[1]} 维，但 RAG_EMBED_DIM={EMBED_DIM}。\n"
            f"请把 RAG_EMBED_DIM 改成 {embeddings.shape[1]} 后重新运行。"
        )

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-9)
    return embeddings


def build_index():
    """构建 FAISS 索引，并保存索引文件和元数据。"""
    import faiss

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    client = get_client()
    chunks = load_documents()
    texts = [c["content"] for c in chunks]

    logger.info(f"开始构建索引，模型={EMBED_MODEL}，维度={EMBED_DIM}")
    embeddings = embed_texts(client, texts)

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"索引构建完成：{index.ntotal} 条向量")
    logger.info(f"FAISS 索引: {INDEX_PATH}")
    logger.info(f"元数据: {META_PATH}")


# ── 检索模块 ────────────────────────────────────────────────────────────────────

class VectorStore:
    def __init__(self, client: OpenAI):
        import faiss

        if not INDEX_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError("索引不存在，请先运行：python src/homework_custom_api_rag.py --build")

        self.client = client
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, encoding="utf-8") as f:
            self.meta = json.load(f)
        logger.info(f"FAISS 加载完成，共 {self.index.ntotal} 条向量")

    def embed_query(self, query: str) -> np.ndarray:
        kwargs = {
            "model": EMBED_MODEL,
            "input": [query],
        }
        if os.getenv("RAG_DISABLE_DIMENSIONS", "0") != "1":
            kwargs["dimensions"] = EMBED_DIM

        resp = self.client.embeddings.create(**kwargs)
        vec = np.array([resp.data[0].embedding], dtype="float32")
        if vec.shape[1] != EMBED_DIM:
            raise ValueError(f"查询向量维度为 {vec.shape[1]}，但 RAG_EMBED_DIM={EMBED_DIM}")
        vec = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
        return vec

    def search(self, query: str, top_k: int = TOP_K_VECTOR) -> list[dict]:
        query_vec = self.embed_query(query)
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = dict(self.meta[idx])
            item["vec_score"] = float(score)
            results.append(item)
        return results


class BM25Store:
    def __init__(self, meta: list[dict]):
        from rank_bm25 import BM25Okapi
        import jieba

        self.meta = meta
        self.jieba = jieba
        tokenized = [list(jieba.cut(item["content"])) for item in meta]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 初始化完成")

    def search(self, query: str, top_k: int = TOP_K_BM25) -> list[dict]:
        tokens = list(self.jieba.cut(query))
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            item = dict(self.meta[idx])
            item["bm25_score"] = float(scores[idx])
            results.append(item)
        return results


def reciprocal_rank_fusion(vec_results: list[dict], bm25_results: list[dict], k: int = 60) -> list[dict]:
    """RRF 融合：只看排名，不直接比较不同检索器的分数。"""
    scores = {}
    item_map = {}

    for rank, item in enumerate(vec_results, 1):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        item_map[cid] = item

    for rank, item in enumerate(bm25_results, 1):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        item_map[cid] = {**item_map.get(cid, {}), **item}

    merged = []
    for cid in sorted(scores, key=lambda x: -scores[x]):
        item = dict(item_map[cid])
        item["rrf_score"] = scores[cid]
        merged.append(item)
    return merged


# ── 生成模块 ────────────────────────────────────────────────────────────────────

def build_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    parts = []
    citations = []

    for i, item in enumerate(chunks, 1):
        source = item.get("source", "unknown")
        page = item.get("page_num", -1)
        section = item.get("section", "")

        label = f"[{i}] {source}"
        if section:
            label += f" · {section}"
        if page and page != -1:
            label += f" · 第{page}页"

        parts.append(f"{label}\n{item['content']}")
        citations.append({"index": i, "source": label, "chunk_id": item.get("chunk_id", "")})

    return "\n\n---\n\n".join(parts), citations


def call_chat_model(client: OpenAI, question: str, context: str) -> str:
    user_prompt = f"""【参考资料】
{context}

【问题】
{question}

请严格根据参考资料回答，并标注来源编号。"""

    if USE_ANTHROPIC_CHAT:
        from urllib import request, error

        payload = json.dumps({
            "model": CHAT_MODEL,
            "max_tokens": 1024,
            "temperature": 0.1,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_prompt}],
        }).encode("utf-8")
        req = request.Request(anthropic_messages_url(), data=payload, headers=get_anthropic_headers(), method="POST")
        try:
            with request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Anthropic API 调用失败: HTTP {e.code} {detail}") from e
        return "".join(part.get("text", "") for part in data.get("content", []) if part.get("type") == "text")

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content


class NativeRAG:
    def __init__(self, use_bm25: bool = True, bm25_only: bool = False):
        self.client = None if USE_ANTHROPIC_CHAT else get_client()
        self.use_bm25 = use_bm25
        self.bm25_only = bm25_only
        self.vector_store = None

        if bm25_only:
            meta = load_documents()
            logger.info("启用 BM25-only 模式：跳过向量索引和 embedding API")
        else:
            embed_client = get_client()
            self.vector_store = VectorStore(embed_client)
            meta = self.vector_store.meta

        self.bm25_store = BM25Store(meta) if use_bm25 else None

    def query(self, question: str, verbose: bool = False) -> dict:
        if self.vector_store:
            vec_results = self.vector_store.search(question, TOP_K_VECTOR)
            if verbose:
                top_score = vec_results[0]["vec_score"] if vec_results else 0
                logger.info(f"向量检索结果: {len(vec_results)} 条，最高分={top_score:.3f}")
        else:
            vec_results = []

        if self.use_bm25 and self.bm25_store:
            bm25_results = self.bm25_store.search(question, TOP_K_BM25)
            candidates = reciprocal_rank_fusion(vec_results, bm25_results) if vec_results else bm25_results
            if verbose:
                logger.info(f"BM25 结果: {len(bm25_results)} 条，候选: {len(candidates)} 条")
        else:
            bm25_results = []
            candidates = vec_results

        final = candidates[:TOP_K_FINAL]

        if not final:
            return {
                "answer": "未检索到相关资料，无法回答此问题。",
                "citations": [],
                "retrieved": [],
            }

        top_vec_score = max((item.get("vec_score", 0.0) for item in final), default=0.0)
        if not self.bm25_only and top_vec_score < SCORE_THRESHOLD:
            return {
                "answer": "根据知识库未能找到足够相关的内容，无法可靠回答此问题。",
                "citations": [],
                "retrieved": final,
            }

        context, citations = build_context(final)
        answer = call_chat_model(self.client, question, context)
        return {
            "answer": answer,
            "citations": citations,
            "retrieved": final,
            "vec_results": vec_results,
            "bm25_results": bm25_results,
        }


# ── 命令行入口 ──────────────────────────────────────────────────────────────────

def print_result(question: str, result: dict, show_sources: bool = True):
    print("\n" + "=" * 70)
    print(f"问题：{question}")
    print("=" * 70)
    print("\n" + result["answer"])

    if show_sources and result.get("citations"):
        print("\n── 来源 ──")
        for c in result["citations"]:
            print(f"  {c['source']}")


def main():
    parser = argparse.ArgumentParser(description="单文件原生 RAG：支持自定义 OpenAI 兼容 API")
    parser.add_argument("--build", action="store_true", help="构建 / 重建 FAISS 索引")
    parser.add_argument("--query", type=str, default=None, help="单次提问")
    parser.add_argument("--no-bm25", action="store_true", help="关闭 BM25，只使用向量检索")
    parser.add_argument("--bm25-only", action="store_true", help="只使用 BM25，跳过 embedding 和 FAISS")
    parser.add_argument("--show-config", action="store_true", help="显示当前 API 和模型配置")
    args = parser.parse_args()

    if args.show_config:
        print("当前配置：")
        print(f"  RAG_BASE_URL    = {BASE_URL}")
        print(f"  ANTHROPIC_BASE_URL = {ANTHROPIC_BASE_URL}")
        print(f"  RAG_USE_ANTHROPIC_CHAT = {USE_ANTHROPIC_CHAT}")
        print(f"  RAG_EMBED_MODEL = {EMBED_MODEL}")
        print(f"  RAG_CHAT_MODEL  = {CHAT_MODEL}")
        print(f"  RAG_EMBED_DIM   = {EMBED_DIM}")
        print(f"  INDEX_PATH      = {INDEX_PATH}")
        print(f"  META_PATH       = {META_PATH}")
        return

    if args.build:
        build_index()
        return

    if args.no_bm25 and args.bm25_only:
        raise ValueError("--no-bm25 和 --bm25-only 不能同时使用")

    rag = NativeRAG(use_bm25=not args.no_bm25, bm25_only=args.bm25_only)

    if args.query:
        result = rag.query(args.query, verbose=True)
        print_result(args.query, result)
        return

    print("单文件原生 RAG 问答系统")
    print(f"Embedding: {EMBED_MODEL} ({EMBED_DIM}维)")
    print(f"Chat:      {CHAT_MODEL}")
    print("输入 exit 退出\n")

    while True:
        try:
            question = input("问题：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break
        result = rag.query(question, verbose=True)
        print_result(question, result)


if __name__ == "__main__":
    main()
