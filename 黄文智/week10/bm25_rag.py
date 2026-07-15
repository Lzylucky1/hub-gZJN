
import json
import os
import jieba
import numpy as np
from bm25 import BM25
from openai import OpenAI
import chromadb
from chromadb.config import Settings

'''
基于RAG来介绍Dota2英雄故事和技能
使用 BM25 + 向量数据库进行检索，RRF融合结果
使用 DeepSeek 大模型进行回答
DeepSeek API: https://platform.deepseek.com
'''

# DeepSeek 大模型调用
def call_large_model(prompt):
    client = OpenAI(
        api_key="api  key",
        base_url="https://api.deepseek.com/v1"
    )
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.7
    )
    response_text = response.choices[0].message.content
    return response_text

class RRF_RAG:
    def __init__(self, folder_path="Heroes"):
        self.load_hero_data(folder_path)
        self.init_bm25()
        self.init_vector_db()
    
    def load_hero_data(self, folder_path):
        self.hero_data = {}
        self.all_docs = []
        self.doc_ids = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):
                with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as file:
                    intro = file.read()
                    hero = file_name.split(".")[0]
                    self.hero_data[hero] = intro
                    self.all_docs.append(intro)
                    self.doc_ids.append(hero)
        return

    def init_bm25(self):
        """初始化 BM25 模型"""
        corpus = {}
        for hero, intro in self.hero_data.items():
            corpus[hero] = jieba.lcut(intro)
        self.bm25_model = BM25(corpus)

    def init_vector_db(self):
        """初始化 Chroma 向量数据库"""
        self.chroma_client = chromadb.Client(Settings(
            persist_directory="./chroma_db",
            is_persistent=True
        ))
        
        # 尝试获取或创建集合
        try:
            self.collection = self.chroma_client.get_collection(name="dota2_heroes")
        except:
            # 创建新集合并添加文档
            self.collection = self.chroma_client.create_collection(name="dota2_heroes")
            self.collection.add(
                documents=self.all_docs,
                ids=self.doc_ids
            )

    def bm25_retrieve(self, query, top_k=5):
        """使用 BM25 检索"""
        scores = self.bm25_model.get_scores(jieba.lcut(query))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        results = [(item[0], item[1]) for item in sorted_scores[:top_k]]
        return results

    def vector_db_retrieve(self, query, top_k=5):
        """使用向量数据库检索"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        docs = results["documents"][0]
        ids = results["ids"][0]
        distances = results["distances"][0] if "distances" in results else [0]*len(ids)
        
        # 将距离转换为相似度分数（距离越小越相似）
        scores = [(ids[i], 1.0 / (1.0 + distances[i])) for i in range(len(ids))]
        return scores

    def rrf_fusion(self, bm25_results, vector_results, k=60):
        """
        RRF (Reciprocal Rank Fusion) 融合算法
        k: RRF 参数，通常取 60
        """
        # 创建排名字典
        bm25_rank = {item[0]: rank + 1 for rank, item in enumerate(bm25_results)}
        vector_rank = {item[0]: rank + 1 for rank, item in enumerate(vector_results)}
        
        # 计算 RRF 分数
        rrf_scores = {}
        all_docs = set(bm25_rank.keys()).union(set(vector_rank.keys()))
        
        for doc_id in all_docs:
            bm25_r = bm25_rank.get(doc_id, float('inf'))
            vector_r = vector_rank.get(doc_id, float('inf'))
            
            score = 0
            if bm25_r != float('inf'):
                score += 1.0 / (k + bm25_r)
            if vector_r != float('inf'):
                score += 1.0 / (k + vector_r)
            
            rrf_scores[doc_id] = score
        
        # 按分数排序
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def retrieve(self, query, top_k=3):
        """综合检索：BM25 + 向量数据库 + RRF 融合"""
        bm25_results = self.bm25_retrieve(query, top_k=10)
        vector_results = self.vector_db_retrieve(query, top_k=10)
        
        print(f"BM25 检索结果 (Top 5): {[item[0] for item in bm25_results[:5]]}")
        print(f"向量检索结果 (Top 5): {[item[0] for item in vector_results[:5]]}")
        
        # RRF 融合
        fused_results = self.rrf_fusion(bm25_results, vector_results)
        print(f"RRF 融合结果 (Top 5): {[item[0] for item in fused_results[:5]]}")
        
        # 获取融合后的文档内容
        top_docs = []
        for doc_id, score in fused_results[:top_k]:
            if doc_id in self.hero_data:
                top_docs.append(self.hero_data[doc_id])
        
        return "\n\n---\n\n".join(top_docs)

    def query(self, user_query):    
        print("\n" + "="*60)
        print(f"用户问题: {user_query}")
        print("="*60)
        
        # 检索相关文档
        retrieved_text = self.retrieve(user_query)
        print("\n" + "-"*60)
        print("融合检索到的文档内容:")
        print(retrieved_text)
        print("-"*60)
        
        # 构建提示词
        prompt = f"请根据以下从数据库中获得的Dota2英雄故事和技能介绍，回答用户问题：\n\n" \
                 f"英雄故事及技能介绍：\n{retrieved_text}\n\n" \
                 f"用户问题：{user_query}\n\n" \
                 f"请用中文简洁明了地回答问题，不要编造信息。"
        
        # 调用大模型
        response_text = call_large_model(prompt)
        
        print("\n" + "="*60)
        print(f"DeepSeek 回答: {response_text}")
        print("="*60)

if __name__ == "__main__":
    rag = RRF_RAG()
    
    # 测试问题
    test_queries = [
        "高射火炮是谁的技能",
        "主宰有哪些技能",
        "幻影刺客的背景故事",
        "宙斯的大招是什么"
    ]
    
    for query in test_queries:
        rag.query(query)
        print("\n" + "#"*80 + "\n")
