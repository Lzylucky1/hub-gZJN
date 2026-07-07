# -*- coding: utf-8 -*-
"""
Week 10 作业：离线 RAG 问答系统

基于已有年报数据（已解析、已分块、已建索引），使用：
- 本地 BGE-small-zh-v1.5 做 Embedding 检索
- 本地 Qwen2-0.5B-Instruct 做答案生成
- 无需任何 API Key，完全离线运行

项目结构（相对本文件）：
    ../../week10 检索增强生成RAG/rag_annual_report/vectorstore/faiss_lc/
    ../../week10 检索增强生成RAG/rag_annual_report/models/bge-small-zh-v1.5/
    ../../pretrain_models/Qwen2-0.5B-Instruct/
"""

import json
import re
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict

import torch
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ── 路径配置 ─────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent  # homework 目录

# 支持两种目录布局：
# 1. 原始课程项目布局：homework 在项目根目录下，模型和数据按课程结构存放
# 2. WSL 工作副本布局：homework 与 rag_annual_report/llm_model 平级
PROJECT_ROOT = ROOT_DIR.parent.parent  # z:\LearningDocs\八斗AI

LAYOUT_1_VS = (
    PROJECT_ROOT
    / "week10检索增强生成"
    / "week10 检索增强生成RAG"
    / "rag_annual_report"
    / "vectorstore"
    / "faiss_lc"
)
LAYOUT_2_VS = ROOT_DIR.parent / "rag_annual_report" / "vectorstore" / "faiss_lc"

LAYOUT_1_EMB = (
    PROJECT_ROOT
    / "week10检索增强生成"
    / "week10 检索增强生成RAG"
    / "rag_annual_report"
    / "models"
    / "bge-small-zh-v1.5"
)
LAYOUT_2_EMB = ROOT_DIR.parent / "rag_annual_report" / "models" / "bge-small-zh-v1.5"

LAYOUT_1_LLM = PROJECT_ROOT / "pretrain_models" / "Qwen2-0.5B-Instruct"
LAYOUT_2_LLM = ROOT_DIR.parent / "llm_model" / "Qwen2-0.5B-Instruct"

VECTORSTORE_DIR = LAYOUT_2_VS if LAYOUT_2_VS.exists() else LAYOUT_1_VS
EMBEDDING_MODEL_DIR = LAYOUT_2_EMB if LAYOUT_2_EMB.exists() else LAYOUT_1_EMB
LLM_MODEL_DIR = LAYOUT_2_LLM if LAYOUT_2_LLM.exists() else LAYOUT_1_LLM

OUTPUTS_DIR = ROOT_DIR / "outputs"


# ── 配置参数 ─────────────────────────────────────────────────────────────────
TOP_K = 4               # 检索 top-k 个 chunk
RETRIEVE_CANDIDATES = 12  # 初检索候选数量，用于关键词重排
MAX_NEW_TOKENS = 256    # 生成答案最大长度
TEMPERATURE = 0.1       # 低温度，保证数字准确
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class QAResult:
    """单次问答结果"""
    question: str
    answer: str
    contexts: List[Dict]  # 检索到的上下文，含来源
    elapsed_seconds: float


class OfflineRAG:
    """离线 RAG 问答系统"""

    def __init__(self):
        logger.info(f"使用设备: {DEVICE}")
        self.embeddings = self._load_embeddings()
        self.vectorstore = self._load_vectorstore()
        self.llm = self._load_llm()

    def _load_embeddings(self) -> HuggingFaceBgeEmbeddings:
        """加载本地 BGE embedding 模型"""
        logger.info(f"加载 Embedding 模型: {EMBEDDING_MODEL_DIR}")
        return HuggingFaceBgeEmbeddings(
            model_name=str(EMBEDDING_MODEL_DIR),
            model_kwargs={"device": DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _load_vectorstore(self) -> FAISS:
        """加载本地 FAISS 索引"""
        logger.info(f"加载 FAISS 索引: {VECTORSTORE_DIR}")
        return FAISS.load_local(
            str(VECTORSTORE_DIR),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def _load_llm(self):
        """加载本地 Qwen2-0.5B-Instruct 模型"""
        logger.info(f"加载 LLM: {LLM_MODEL_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(LLM_MODEL_DIR),
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(LLM_MODEL_DIR),
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else "cpu",
            trust_remote_code=True,
        )
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

    def _extract_keywords(self, question: str) -> tuple:
        """从问题中提取年份和财务指标关键词"""
        # 年份
        year_match = re.search(r"20\d{2}", question)
        target_year = year_match.group(0) if year_match else None

        # 常见财务指标关键词（按优先级排序）
        metric_keywords = [
            "毛利率", "净利率", "营业收入", "营业总收入", "净利润",
            "归属于母公司股东的净利润", "总资产", "净资产", "研发投入",
            "研发费用", "销售费用", "管理费用", "资产负债率", "每股收益",
            "分红", "股息", "现金流", "研发投入金额",
        ]
        found_metrics = [kw for kw in metric_keywords if kw in question]
        return target_year, found_metrics

    def _score_chunk(self, content: str, year: str, metrics: List[str]) -> int:
        """基于关键词命中情况给 chunk 打分，用于重排"""
        score = 0
        if year and year in content:
            score += 2
        for metric in metrics:
            if metric in content:
                score += 3
        return score

    def retrieve(self, question: str, k: int = TOP_K) -> List[Dict]:
        """检索相关文档片段（向量检索 + 关键词重排）"""
        target_year, metrics = self._extract_keywords(question)

        # 先检索更多候选
        docs = self.vectorstore.similarity_search(question, k=RETRIEVE_CANDIDATES)

        # 按关键词加分重排
        scored_docs = []
        for doc in docs:
            meta = doc.metadata
            source = meta.get("source", "unknown")
            score = self._score_chunk(doc.page_content, target_year, metrics)
            scored_docs.append((score, doc))

        # 按得分降序，得分相同保持原向量顺序
        scored_docs.sort(key=lambda x: (-x[0], docs.index(x[1])))
        selected_docs = [doc for _, doc in scored_docs[:k]]

        contexts = []
        for idx, doc in enumerate(selected_docs, 1):
            meta = doc.metadata
            source = meta.get("source", "unknown")
            m = re.search(r"(\d{6})_(\d{4})_([^_]+)", Path(source).name)
            if m:
                stock_code, year, company = m.groups()
            else:
                stock_code, year, company = "-", "-", "-"

            contexts.append({
                "index": idx,
                "content": doc.page_content,
                "stock_code": stock_code,
                "company": company,
                "year": year,
                "source": source,
            })
        return contexts

    def generate(self, question: str, contexts: List[Dict]) -> str:
        """基于检索到的上下文生成答案"""
        context_text = "\n\n".join(
            f"[{c['index']}] {c['company']}({c['stock_code']}) {c['year']}年报：\n{c['content']}"
            for c in contexts
        )

        system_prompt = (
            "你是上市公司年报分析助手。请严格根据提供的参考资料回答问题，"
            "不要编造答案。如果参考资料不足，请明确回答'根据现有资料无法回答'。"
            "回答时请引用参考资料编号，如[1]、[2]。"
        )

        user_prompt = f"参考资料：\n{context_text}\n\n问题：{question}\n\n答案："

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Qwen2 使用 chat template
        prompt = self.llm.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        outputs = self.llm(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            return_full_text=False,
        )
        answer = outputs[0]["generated_text"].strip()
        return answer

    def query(self, question: str) -> QAResult:
        """完整问答流程"""
        t0 = time.time()
        contexts = self.retrieve(question)
        answer = self.generate(question, contexts)
        elapsed = time.time() - t0

        return QAResult(
            question=question,
            answer=answer,
            contexts=contexts,
            elapsed_seconds=elapsed,
        )


def save_results(results: List[QAResult], path: Path):
    """保存问答结果到 JSON"""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存: {path}")


def main():
    # 示例问题
    questions = [
        "贵州茅台2023年营业收入是多少？",
        "宁德时代2023年净利润是多少？",
        "中国平安2023年总资产是多少？",
        "五粮液2023年毛利率是多少？",
        "海康威视2023年研发投入是多少？",
    ]

    logger.info("=" * 60)
    logger.info("Week 10 离线 RAG 问答系统")
    logger.info("=" * 60)

    rag = OfflineRAG()
    results = []

    for q in questions:
        logger.info(f"\n问题: {q}")
        result = rag.query(q)
        results.append(result)
        logger.info(f"答案: {result.answer[:200]}...")
        logger.info(f"耗时: {result.elapsed_seconds:.2f}s")

    # 保存结果
    output_path = OUTPUTS_DIR / "qa_results.json"
    save_results(results, output_path)

    # 打印汇总
    logger.info("\n" + "=" * 60)
    logger.info("问答汇总")
    logger.info("=" * 60)
    for r in results:
        logger.info(f"Q: {r.question}")
        logger.info(f"A: {r.answer[:150]}...")
        logger.info("-" * 40)


if __name__ == "__main__":
    main()
