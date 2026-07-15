r"""
向量检索器：基于 sentence-transformers 的语义检索（RAG 的「检索」语义层）
"""

from __future__ import annotations

import numpy as np

from config import EMBEDDING_MODEL


class EmbeddingRetriever:
    """语义向量检索：把问题编码成向量，余弦相似度排序。

    Args:
        model_name: sentence-transformers 模型名，默认 BAAI/bge-small-zh-v1.5。
    """

    name = "Embedding"

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "向量检索需要 sentence-transformers，请先安装："
                "pip install sentence-transformers"
            ) from e
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._vecs: np.ndarray | None = None
        self._n: int = 0

    def fit(self, texts: list[str]) -> "EmbeddingRetriever":
        """把知识库问题编码成向量矩阵（行=文档，列=维度），并做 L2 归一。"""
        # normalize_embeddings=True 后直接点积即余弦相似度
        self._vecs = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        self._n = len(texts)
        return self

    def search(self, query: str, topk: int = 3) -> list[tuple[float, int]]:
        """查询编码 → 与库向量做内积（=归一后的余弦）→ 取 topk。"""
        if self._vecs is None or self._n == 0 or not query:
            return []
        qv = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )
        # (n,) 内积
        sims = self._vecs @ qv[0]
        k = min(topk, self._n)
        # 取最大的 k 个（argsort 默认升序，取尾部再翻转）
        idx = np.argsort(sims)[-k:][::-1]
        return [(float(sims[i]), int(i)) for i in idx]


if __name__ == "__main__":
    # 自测：直观感受语义检索 vs 字面重叠
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    r = EmbeddingRetriever()
    docs = ["如何对列表去重并保持原顺序？", "如何对列表排序？", "如何读写文件？"]
    r.fit(docs)
    for q in ["怎么去掉列表里重复的元素", "把数字从小到大排", "读取 txt 内容"]:
        print(f"\nQ: {q}")
        for score, i in r.search(q, topk=2):
            print(f"  {score:.4f}  {docs[i]}")
