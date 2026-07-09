r"""
问答系统核心：加载知识库 + 检索 + （可选）LLM 生成 = 完整 RAG

定位：检索增强生成（RAG, Retrieval-Augmented Generation）。
  - 知识库 = 一批 (question, answer) 对，answer 里直接含可运行代码
  - 用户提问 → 检索 top-k 相似问题（词法 or 语义）→ 喂给 LLM 生成最终答案
  - 检索方法可选：overlap / tfidf / bm25（词法）或 embedding（语义向量）
  - 生成可选：调 LLM（默认本地 Ollama）或降级为抽取式（直接返回 top-1 原答案）

两种使用模式：
  1. 检索模式（retrieval-only）：answer() 直接返回命中的知识库答案，零生成依赖。
  2. RAG 模式：generate() 把 top-k 作为 context 喂 LLM，生成改写/组合后的答案。

QASystem 对外能力：
  .answer(query, topk=1)            单条最佳回答（检索即作答）
  .answer_with_context(query, topk) 返回 top-k 结果（含相似度、命中问题、答案）
  .generate(query, topk=3)          RAG 生成：检索 top-k → LLM 生成
  .retriever_name                   当前检索方法名
"""

import json
import os
import sys
from dataclasses import dataclass

from retrieval import build_retriever
from tokenizer import tokenize

DEFAULT_KB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "qa_knowledge.jsonl"
)

# 词法检索方法（吃已分词 token）
LEXICAL_METHODS = {"overlap", "tfidf", "bm25"}


@dataclass
class QAResult:
    """单条检索结果。"""
    rank: int
    score: float
    question: str
    answer: str
    qa_id: str


def load_knowledge(path: str = DEFAULT_KB_PATH) -> list[dict]:
    """从 jsonl 读知识库，每行一个 {id, question, tags, answer}。"""
    path = os.path.abspath(path)
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _doc_text(rec: dict) -> str:
    """把一条知识库记录拼成检索用文本：问题 + 标签。"""
    text = rec["question"]
    if rec.get("tags"):
        text = text + " " + " ".join(rec["tags"])
    return text


class QASystem:
    """RAG 问答系统：检索 + （可选）生成。

    Args:
        method: 检索方法 "overlap" / "tfidf" / "bm25" / "embedding"
                默认 "embedding"（语义向量检索，RAG 主线）
        kb_path: 知识库 jsonl 路径
    """

    def __init__(self, method: str = "embedding", kb_path: str = DEFAULT_KB_PATH):
        self.method = method
        self.is_embedding = method == "embedding"
        if self.is_embedding:
            # 延迟导入：未装 sentence-transformers 时只有用到 embedding 才报错
            try:
                from embeddings import EmbeddingRetriever
                self.retriever = EmbeddingRetriever()
            except Exception as e:
                # 未装 sentence-transformers 或模型下载失败 → 降级 BM25，保证开箱即用
                print(f"[Embedding 不可用，降级为 BM25 词法检索] {e}")
                self.method = "bm25"
                self.is_embedding = False
                self.retriever = build_retriever("bm25")
        else:
            self.retriever = build_retriever(method)
        self.retriever_name = self.retriever.name
        self.kb: list[dict] = []
        self._questions_tokens: list[list[str]] = []  # 词法用
        self._questions_text: list[str] = []          # 语义用
        self._ollama_gen = None       # 懒加载：Ollama LLMGenerator
        self._local_hf_gen = None     # 懒加载：本地 HF 模型生成器
        self._ollama_dead = False     # Ollama 已确认不可用，后续直接跳过
        self._fit(kb_path)

    def _fit(self, kb_path: str):
        self.kb = load_knowledge(kb_path)
        self._questions_text = [_doc_text(rec) for rec in self.kb]
        if self.is_embedding:
            # 语义检索吃原始文本
            self.retriever.fit(self._questions_text)
        else:
            # 词法检索吃分词后的 token
            self._questions_tokens = [tokenize(t) for t in self._questions_text]
            self.retriever.fit(self._questions_tokens)

    def search(self, query: str, topk: int = 3) -> list[QAResult]:
        if self.is_embedding:
            hits = self.retriever.search(query, topk=topk)
        else:
            hits = self.retriever.search(tokenize(query), topk=topk)
        results: list[QAResult] = []
        for rank, (score, idx) in enumerate(hits, 1):
            rec = self.kb[idx]
            results.append(QAResult(
                rank=rank,
                score=score,
                question=rec["question"],
                answer=rec["answer"],
                qa_id=rec.get("id", ""),
            ))
        return results

    def answer(self, query: str, topk: int = 1) -> QAResult | None:
        """返回最相似的那条回答；命不中（空查询）返回 None。"""
        res = self.search(query, topk=topk)
        return res[0] if res else None

    def answer_with_context(self, query: str, topk: int = 3) -> list[QAResult]:
        return self.search(query, topk=topk)

    def _get_ollama_generator(self):
        if self._ollama_gen is None:
            from generator import LLMGenerator
            self._ollama_gen = LLMGenerator()
        return self._ollama_gen

    def _get_local_hf_generator(self):
        if self._local_hf_gen is None:
            from generator import LocalHFGenerator
            self._local_hf_gen = LocalHFGenerator()
        return self._local_hf_gen

    def generate(self, query: str, topk: int = 3, use_llm: bool = True) -> str:
        """RAG 生成：检索 top-k 作为上下文 → LLM 生成答案。

        三级降级，保证总能作答：
          1. Ollama（LLMGenerator）—— 默认，需 `ollama pull` 拉模型
          2. 本地 HF 模型（LocalHFGenerator）—— Ollama 不可用时离线生成
          3. 抽取式（ExtractiveGenerator）—— 都不可用时返回 top-1 原答案

        Args:
            query: 用户问题
            topk: 检索条数（喂给 LLM 的参考数）
            use_llm: True 走 LLM 链；False 直接抽取式
        Returns:
            生成的答案文本（含代码）
        """
        results = self.search(query, topk=topk)
        if not use_llm:
            from generator import ExtractiveGenerator
            return ExtractiveGenerator().generate(query, results)

        # 1) Ollama（未确认失败才试，避免每次都等 404）
        if not self._ollama_dead:
            try:
                return self._get_ollama_generator().generate(query, results)
            except Exception as e:
                self._ollama_dead = True
                print(f"[Ollama LLM 不可用，转本地模型生成] {e}")

        # 2) 本地 HF 模型（懒加载，首次调用加载几秒~十几秒，进程内复用）
        try:
            return self._get_local_hf_generator().generate(query, results)
        except Exception as e:
            print(f"[本地 LLM 不可用，降级抽取式] {e}")

        # 3) 抽取式兜底
        from generator import ExtractiveGenerator
        return ExtractiveGenerator().generate(query, results)


def format_answer(res: QAResult) -> str:
    """把单条结果格式化成可读文本，方便 CLI/打印。"""
    return (
        f"\n"
        f"答：{res.answer}"
    )


if __name__ == "__main__":
    # 自测：词法方法各答一个问题
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    for m in ["overlap", "tfidf", "bm25"]:
        qa = QASystem(method=m)
        r = qa.answer("怎么把列表里重复的元素去掉")
        print(f"\n=== {m} ===")
        print(format_answer(r)) if r else print("(未命中)")
