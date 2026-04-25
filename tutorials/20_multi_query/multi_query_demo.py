#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第20小节：多查询（Multi-Query）RAG优化策略演示

本模块实现了多查询RAG系统，通过生成多个相关查询并行检索，
提升检索的全面性和准确性。

主要功能：
1. 多查询生成（基于原始查询生成多个相关查询）
2. 并行检索（同时检索多个查询）
3. 结果融合（合并和去重检索结果）
4. 与传统RAG和查询重写RAG的对比测试
5. 使用民法典PDF文档进行实际验证
"""

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("错误：未安装chromadb。请运行：pip install chromadb")
    sys.exit(1)

try:
    import json

    import requests
except ImportError:
    print("错误：未安装requests。请运行：pip install requests")
    sys.exit(1)

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from utils.config import get_config_dict, load_embedding_config, load_environment
    from utils.llm_factory import create_deepseek_llm, create_llm_from_config
except ImportError:
    print("错误：无法导入配置模块。请确保utils/config.py和utils/llm_factory.py存在。")
    sys.exit(1)

try:
    from chromadb.utils import embedding_functions
except ImportError:
    print("错误：无法导入chromadb embedding_functions。请确保chromadb已正确安装。")
    sys.exit(1)

# PDF处理
try:
    import PyPDF2
except ImportError:
    print("警告: PyPDF2未安装，无法处理PDF文档。请运行: pip install PyPDF2")
    PyPDF2 = None

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class MultiQueryResult:
    """多查询结果"""

    original_query: str
    generated_queries: List[str]
    documents: List[str]
    scores: List[float]
    response: str
    processing_time: float
    query_generation_time: float


@dataclass
class RAGResult:
    """RAG检索结果"""

    query: str
    documents: List[str]
    scores: List[float]
    response: str
    processing_time: float


@dataclass
class ComparisonResult:
    """对比测试结果"""

    query: str
    traditional: RAGResult
    multi_query: MultiQueryResult
    improvement_score: float


class MultiQueryGenerator:
    """多查询生成器"""

    def __init__(self, llm_model):
        """初始化多查询生成器

        Args:
            llm_model: LangChain的ChatOpenAI实例
        """
        self.llm_model = llm_model

    def generate_queries(self, original_query: str, num_queries: int = 3) -> List[str]:
        """生成多个相关查询"""
        prompt = f"""你是一个查询优化专家。基于用户的原始查询，生成{num_queries}个不同但相关的查询变体。

要求：
1. 保持与原始查询相同的核心意图
2. 使用不同的词汇和表达方式
3. 从不同角度探索相同的主题
4. 每个查询应该独立且完整

原始查询：{original_query}

请直接输出{num_queries}个查询，每行一个，不要包含编号或其他格式："""

        try:
            response = self.llm_model.invoke(prompt)
            generated_text = response.content.strip()

            # 解析生成的查询
            queries = [q.strip() for q in generated_text.split("\n") if q.strip()]

            # 确保包含原始查询
            if original_query not in queries:
                queries.insert(0, original_query)

            # 限制查询数量
            return queries[: num_queries + 1]  # +1 for original query

        except Exception as e:
            logger.error(f"查询生成失败: {e}")
            return [original_query]


class VectorDatabase:
    """向量数据库"""

    def __init__(
        self,
        collection_name: str = "multi_query_demo",
        persist_directory: str = "./demo_multi_query_db",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_function = None

        # 创建ChromaDB客户端
        self.client = chromadb.PersistentClient(
            path=persist_directory, settings=Settings(anonymized_telemetry=False)
        )

        # 设置embedding函数
        self._setup_embedding_function()

        # 获取或创建集合
        try:
            self.collection = self.client.get_collection(
                name=collection_name, embedding_function=self.embedding_function
            )
            logger.info(f"加载现有集合: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name, embedding_function=self.embedding_function
            )
            logger.info(f"创建新集合: {collection_name}")

    def _setup_embedding_function(self):
        """设置embedding函数"""
        try:
            # 加载embedding配置
            embedding_config = load_embedding_config()

            if embedding_config and embedding_config.get("embedding_api_key"):
                # 使用OpenAI兼容的embedding函数
                self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=embedding_config["embedding_api_key"],
                    api_base=embedding_config.get(
                        "embedding_base_url", "https://api.openai.com/v1"
                    ),
                    model_name=embedding_config.get(
                        "embedding_model_name", "text-embedding-ada-002"
                    ),
                )
                logger.info(
                    f"使用OpenAI兼容的embedding函数: {embedding_config.get('embedding_model_name')}"
                )
            else:
                # 使用默认的embedding函数
                self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
                logger.info("使用默认的embedding函数")

        except Exception as e:
            logger.warning(f"设置embedding函数失败: {e}，使用默认embedding函数")
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """添加文档到向量数据库"""
        try:
            ids = [f"doc_{i}" for i in range(len(documents))]

            if metadatas is None:
                metadatas = [{"source": f"document_{i}"} for i in range(len(documents))]

            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

            logger.info(f"成功添加 {len(documents)} 个文档到向量数据库")
        except Exception as e:
            logger.error(f"添加文档到向量数据库失败: {e}")

    def search(self, query_text: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
        """搜索相似文档"""
        try:
            results = self.collection.query(query_texts=[query_text], n_results=top_k)

            documents = results["documents"][0] if results["documents"] else []
            distances = results["distances"][0] if results["distances"] else []

            # 将距离转换为相似度分数（距离越小，相似度越高）
            scores = [1.0 / (1.0 + dist) for dist in distances]

            return documents, scores
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return [], []

    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "persist_directory": self.persist_directory,
            }
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {}


class MultiQueryRAG:
    """多查询RAG系统"""

    def __init__(self, llm_model, vector_db: VectorDatabase, debug_mode: bool = False):
        """初始化多查询RAG系统

        Args:
            llm_model: 大语言模型实例
            vector_db: 向量数据库实例
            debug_mode: 是否启用调试模式
        """
        self.llm_model = llm_model
        self.vector_db = vector_db
        self.query_generator = MultiQueryGenerator(llm_model)
        self.debug_mode = debug_mode

    def _merge_results_with_rrf(
        self, all_results: List[Tuple[List[str], List[float]]], queries: List[str]
    ) -> Tuple[List[str], List[float], Dict[str, Any]]:
        """使用RRF（Reciprocal Rank Fusion）合并多个查询的检索结果

        Args:
            all_results: 每个查询的检索结果
            queries: 对应的查询列表

        Returns:
            合并后的文档、分数和详细信息
        """
        # 记录详细信息用于调试
        debug_info = {"query_results": [], "rrf_scores": {}, "final_ranking": []}

        # 记录每个查询的结果
        for i, (docs, scores, query) in enumerate(
            zip(all_results, [r[1] for r in all_results], queries)
        ):
            query_info = {
                "query": query,
                "documents": docs,
                "scores": scores,
                "doc_count": len(docs),
            }
            debug_info["query_results"].append(query_info)

            if self.debug_mode:
                print(f"\n🔍 子查询 {i+1}: {query}")
                print(f"   召回文档数: {len(docs)}")
                for j, (doc, score) in enumerate(zip(docs[:3], scores[:3])):
                    preview = doc[:100] + "..." if len(doc) > 100 else doc
                    print(f"   文档 {j+1} (分数: {score:.3f}): {preview}")

        # RRF合并算法
        k = 60  # RRF参数
        rrf_scores = {}

        for docs, scores in all_results:
            for rank, doc in enumerate(docs):
                if doc not in rrf_scores:
                    rrf_scores[doc] = 0
                # RRF公式: 1 / (k + rank)
                rrf_scores[doc] += 1 / (k + rank + 1)

        debug_info["rrf_scores"] = rrf_scores.copy()

        # 按RRF分数排序
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        merged_docs = [doc for doc, _ in sorted_items]
        merged_scores = [score for _, score in sorted_items]

        debug_info["final_ranking"] = [
            (doc[:100] + "..." if len(doc) > 100 else doc, score)
            for doc, score in sorted_items[:5]
        ]

        if self.debug_mode:
            print(f"\n🔄 RRF融合结果:")
            print(f"   总文档数: {len(merged_docs)}")
            print(f"   前5个文档的RRF分数:")
            for i, (doc, score) in enumerate(debug_info["final_ranking"]):
                print(f"   {i+1}. (RRF分数: {score:.3f}) {doc}")

        return merged_docs, merged_scores, debug_info

    def _merge_results(
        self, all_results: List[Tuple[List[str], List[float]]]
    ) -> Tuple[List[str], List[float]]:
        """合并多个查询的检索结果（保持向后兼容）"""
        merged_docs, merged_scores, _ = self._merge_results_with_rrf(all_results, [])
        return merged_docs, merged_scores

    def query(
        self, question: str, top_k: int = 5, show_process: bool = False
    ) -> MultiQueryResult:
        """执行多查询RAG

        Args:
            question: 用户问题
            top_k: 返回的文档数量
            show_process: 是否显示详细过程
        """
        start_time = time.time()

        try:
            if show_process or self.debug_mode:
                print(f"\n🎯 原始查询: {question}")
                print("\n" + "=" * 60)
                print("📝 步骤1: 生成多个查询")
                print("=" * 60)

            # 1. 生成多个查询
            query_start = time.time()
            queries = self.query_generator.generate_queries(question)
            query_generation_time = time.time() - query_start

            if show_process or self.debug_mode:
                print(
                    f"✅ 生成了 {len(queries)} 个查询 (耗时: {query_generation_time:.3f}秒)"
                )
                for i, q in enumerate(queries, 1):
                    print(f"   {i}. {q}")

                print("\n" + "=" * 60)
                print("🔍 步骤2: 并行检索")
                print("=" * 60)

            # 2. 并行检索
            all_results = []
            for i, query in enumerate(queries):
                docs, scores = self.vector_db.search(query, top_k)
                all_results.append((docs, scores))

                if show_process or self.debug_mode:
                    print(f"\n查询 {i+1}: {query}")
                    print(f"   召回文档数: {len(docs)}")
                    if docs:
                        print(f"   最高分数: {max(scores):.3f}")
                        print(f"   平均分数: {sum(scores)/len(scores):.3f}")
                        # 显示前2个文档的预览
                        for j, (doc, score) in enumerate(zip(docs[:2], scores[:2])):
                            preview = doc[:80] + "..." if len(doc) > 80 else doc
                            print(f"   文档 {j+1} (分数: {score:.3f}): {preview}")

            if show_process or self.debug_mode:
                print("\n" + "=" * 60)
                print("🔄 步骤3: RRF结果融合")
                print("=" * 60)

            # 3. 使用RRF合并结果
            merged_docs, merged_scores, debug_info = self._merge_results_with_rrf(
                all_results, queries
            )

            # 4. 取前top_k个结果
            final_docs = merged_docs[:top_k]
            final_scores = merged_scores[:top_k]

            if show_process or self.debug_mode:
                print(f"\n✅ 融合完成，最终选择前 {top_k} 个文档")
                print("最终文档排序:")
                for i, (doc, score) in enumerate(zip(final_docs, final_scores)):
                    preview = doc[:80] + "..." if len(doc) > 80 else doc
                    print(f"   {i+1}. (RRF分数: {score:.3f}) {preview}")

                print("\n" + "=" * 60)
                print("💡 步骤4: 生成回答")
                print("=" * 60)

            # 5. 生成回答
            context = "\n\n".join(final_docs)

            prompt = f"""基于以下上下文信息回答问题：

上下文：
{context}

问题：{question}

请基于上下文信息提供准确、详细的回答。如果上下文中没有相关信息，请说明无法回答。"""

            response = self.llm_model.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)

            processing_time = time.time() - start_time

            if show_process or self.debug_mode:
                print(f"✅ 回答生成完成 (总耗时: {processing_time:.3f}秒)")

            # 创建增强的结果对象
            result = MultiQueryResult(
                original_query=question,
                generated_queries=queries,
                documents=final_docs,
                scores=final_scores,
                response=answer,
                processing_time=processing_time,
                query_generation_time=query_generation_time,
            )

            # 添加调试信息
            if hasattr(result, "__dict__"):
                result.__dict__["debug_info"] = debug_info

            return result

        except Exception as e:
            logger.error(f"多查询RAG处理失败: {e}")
            return MultiQueryResult(
                original_query=question,
                generated_queries=[],
                documents=[],
                scores=[],
                response=f"处理失败: {str(e)}",
                processing_time=time.time() - start_time,
                query_generation_time=0.0,
            )

    def traditional_query(self, question: str, top_k: int = 5) -> RAGResult:
        """传统RAG查询（用于对比）"""
        start_time = time.time()

        try:
            # 直接使用原始查询进行检索
            docs, scores = self.vector_db.search(question, top_k)

            # 生成回答
            context = "\n\n".join(docs)

            prompt = f"""基于以下上下文信息回答问题：

上下文：
{context}

问题：{question}

请基于上下文信息提供准确、详细的回答。如果上下文中没有相关信息，请说明无法回答。"""

            response = self.llm_model.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)

            processing_time = time.time() - start_time

            return RAGResult(
                query=question,
                documents=docs,
                scores=scores,
                response=answer,
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"传统RAG查询失败: {e}")
            return RAGResult(
                query=question,
                documents=[],
                scores=[],
                response=f"查询失败: {str(e)}",
                processing_time=time.time() - start_time,
            )

    def compare_strategies(self, query: str) -> ComparisonResult:
        """对比两种检索策略

        Args:
            query: 查询文本

        Returns:
            ComparisonResult: 对比结果
        """
        logger.info(f"开始对比测试，查询: {query}")

        # 传统RAG检索
        traditional_result = self.traditional_query(query)

        # 多查询RAG检索
        multi_query_result = self.query(query)

        # 计算改进分数
        improvement_score = 0.0
        if traditional_result.scores and multi_query_result.scores:
            traditional_avg = sum(traditional_result.scores) / len(
                traditional_result.scores
            )
            multi_query_avg = sum(multi_query_result.scores) / len(
                multi_query_result.scores
            )
            if traditional_avg > 0:
                improvement_score = (
                    multi_query_avg - traditional_avg
                ) / traditional_avg

        return ComparisonResult(
            query=query,
            traditional=traditional_result,
            multi_query=multi_query_result,
            improvement_score=improvement_score,
        )

    def generate_answer(self, query: str, documents: List[str]) -> str:
        """基于检索到的文档生成答案

        Args:
            query: 用户查询
            documents: 检索到的文档

        Returns:
            str: 生成的答案
        """
        if not documents:
            return "抱歉，没有找到相关信息来回答您的问题。"

        # 构建上下文
        context = "\n\n".join(documents[:3])

        # 构建提示词
        prompt = f"""
基于以下上下文信息，请回答用户的问题。如果上下文中没有相关信息，请诚实地说明。

上下文信息：
{context}

用户问题：{query}

请提供准确、有用的回答：
"""

        try:
            response = self.llm_model.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return "抱歉，生成答案时出现错误。"


class PerformanceEvaluator:
    """性能评估器"""

    def __init__(self, rag_system: MultiQueryRAG):
        self.rag_system = rag_system

    def evaluate_strategies(self, test_queries: List[str]) -> Dict[str, Any]:
        """评估不同策略的性能"""
        traditional_results = []
        multi_query_results = []

        logger.info("开始性能评估...")

        for i, query in enumerate(test_queries, 1):
            logger.info(f"评估查询 {i}/{len(test_queries)}: {query[:50]}...")

            # 传统RAG
            traditional_result = self.rag_system.traditional_query(query)
            traditional_results.append(traditional_result)

            # 多查询RAG
            multi_query_result = self.rag_system.query(query)
            multi_query_results.append(multi_query_result)

        # 计算平均指标
        traditional_avg_time = sum(
            r.processing_time for r in traditional_results
        ) / len(traditional_results)
        multi_query_avg_time = sum(
            r.processing_time for r in multi_query_results
        ) / len(multi_query_results)

        traditional_avg_docs = sum(len(r.documents) for r in traditional_results) / len(
            traditional_results
        )
        multi_query_avg_docs = sum(len(r.documents) for r in multi_query_results) / len(
            multi_query_results
        )

        traditional_avg_score = sum(
            sum(r.scores) / len(r.scores) if r.scores else 0
            for r in traditional_results
        ) / len(traditional_results)
        multi_query_avg_score = sum(
            sum(r.scores) / len(r.scores) if r.scores else 0
            for r in multi_query_results
        ) / len(multi_query_results)

        return {
            "traditional_avg_time": traditional_avg_time,
            "multi_query_avg_time": multi_query_avg_time,
            "traditional_avg_docs": traditional_avg_docs,
            "multi_query_avg_docs": multi_query_avg_docs,
            "traditional_avg_score": traditional_avg_score,
            "multi_query_avg_score": multi_query_avg_score,
            "improvement_ratio": (
                multi_query_avg_score / traditional_avg_score
                if traditional_avg_score > 0
                else 0
            ),
        }

    def print_comparison_results(self, results: Dict[str, Any]):
        """打印对比结果"""
        print("\n" + "=" * 60)
        print("性能对比结果")
        print("=" * 60)

        print(f"\n📊 处理时间对比:")
        print(f"   传统RAG平均时间: {results['traditional_avg_time']:.3f}秒")
        print(f"   多查询RAG平均时间: {results['multi_query_avg_time']:.3f}秒")

        print(f"\n📚 文档数量对比:")
        print(f"   传统RAG平均文档数: {results['traditional_avg_docs']:.1f}")
        print(f"   多查询RAG平均文档数: {results['multi_query_avg_docs']:.1f}")

        print(f"\n🔍 相似度分数对比:")
        print(f"   传统RAG平均分数: {results['traditional_avg_score']:.3f}")
        print(f"   多查询RAG平均分数: {results['multi_query_avg_score']:.3f}")
        print(f"   改进比率: {results['improvement_ratio']:.2f}x")

        print("\n" + "=" * 60)

    @staticmethod
    def print_query_result(
        result: MultiQueryResult, title: str, show_details: bool = False
    ):
        """打印查询结果

        Args:
            result: 查询结果
            title: 结果标题
            show_details: 是否显示详细信息
        """
        print(f"\n{'='*50}")
        print(f"{title}")
        print(f"{'='*50}")
        print(f"🎯 原始查询: {result.original_query}")
        print(f"⏱️  总处理时间: {result.processing_time:.3f}秒")
        print(f"⏱️  查询生成时间: {result.query_generation_time:.3f}秒")
        print(f"📝 生成的查询数: {len(result.generated_queries)}")
        print(f"📄 最终文档数: {len(result.documents)}")

        if show_details and result.generated_queries:
            print(f"\n📝 生成的查询列表:")
            for i, query in enumerate(result.generated_queries, 1):
                print(f"   {i}. {query}")

        if result.documents:
            print(f"\n📚 检索到的文档片段 (按RRF分数排序):")
            for i, (doc, score) in enumerate(
                zip(result.documents[:3], result.scores[:3]), 1
            ):
                content = doc[:200] + "..." if len(doc) > 200 else doc
                print(f"\n📄 文档 {i} (RRF分数: {score:.3f}):")
                print(f"   {content}")

        # 显示调试信息
        if (
            show_details
            and hasattr(result, "__dict__")
            and "debug_info" in result.__dict__
        ):
            debug_info = result.__dict__["debug_info"]
            if debug_info.get("query_results"):
                print(f"\n🔍 详细检索过程:")
                for i, query_result in enumerate(debug_info["query_results"], 1):
                    print(f"   查询 {i}: {query_result['query']}")
                    print(f"   召回文档数: {query_result['doc_count']}")
                    if query_result["scores"]:
                        avg_score = sum(query_result["scores"]) / len(
                            query_result["scores"]
                        )
                        print(f"   平均分数: {avg_score:.3f}")

    @staticmethod
    def print_detailed_process(result: MultiQueryResult):
        """打印详细的多查询过程

        Args:
            result: 查询结果
        """
        print(f"\n{'='*80}")
        print("🔍 多查询RAG详细过程分析")
        print(f"{'='*80}")

        print(f"\n🎯 原始问题: {result.original_query}")

        # 显示查询拆分
        print(f"\n📝 步骤1: 查询拆分 (耗时: {result.query_generation_time:.3f}秒)")
        print(f"   将原始问题拆分为 {len(result.generated_queries)} 个子查询:")
        for i, query in enumerate(result.generated_queries, 1):
            print(f"   {i}. {query}")

        # 显示调试信息
        if hasattr(result, "__dict__") and "debug_info" in result.__dict__:
            debug_info = result.__dict__["debug_info"]

            # 显示每个查询的检索结果
            if debug_info.get("query_results"):
                print(f"\n🔍 步骤2: 并行检索结果")
                for i, query_result in enumerate(debug_info["query_results"], 1):
                    print(f"\n   子查询 {i}: {query_result['query']}")
                    print(f"   召回文档数: {query_result['doc_count']}")

                    if query_result["documents"]:
                        print(f"   前3个文档:")
                        for j, (doc, score) in enumerate(
                            zip(
                                query_result["documents"][:3],
                                query_result["scores"][:3],
                            )
                        ):
                            preview = doc[:100] + "..." if len(doc) > 100 else doc
                            print(f"     {j+1}. (分数: {score:.3f}) {preview}")

            # 显示RRF融合结果
            if debug_info.get("final_ranking"):
                print(f"\n🔄 步骤3: RRF融合排序")
                print(f"   使用Reciprocal Rank Fusion算法融合多个查询结果")
                print(f"   最终排序前5个文档:")
                for i, (doc_preview, rrf_score) in enumerate(
                    debug_info["final_ranking"], 1
                ):
                    print(f"     {i}. (RRF分数: {rrf_score:.3f}) {doc_preview}")

        print(f"\n✅ 最终选择前 {len(result.documents)} 个文档用于生成回答")
        print(f"📊 总处理时间: {result.processing_time:.3f}秒")

    @staticmethod
    def print_comparison_result(result: ComparisonResult):
        """打印对比结果

        Args:
            result: 对比结果
        """
        print(f"\n{'='*60}")
        print("策略对比结果")
        print(f"{'='*60}")

        print(f"\n📊 性能指标对比:")
        print(
            f"传统RAG - 文档数: {len(result.traditional.documents)}, 时间: {result.traditional.processing_time:.3f}秒"
        )
        print(
            f"多查询RAG - 文档数: {len(result.multi_query.documents)}, 时间: {result.multi_query.processing_time:.3f}秒"
        )

        print(f"\n📈 改进效果:")
        print(f"改进分数: {result.improvement_score:.3f}")

        # 评估结果
        if result.improvement_score > 0.2:
            print("\n✅ 多查询策略显著提升了检索效果")
        elif result.improvement_score > 0:
            print("\n🔶 多查询策略略微提升了检索效果")
        else:
            print("\n❌ 多查询策略未显著改善检索效果")


def load_sample_documents() -> List[str]:
    """加载民法典文档"""
    try:
        # 尝试加载民法典PDF文档
        pdf_path = "../../docs/中华人民共和国民法典.pdf"

        if os.path.exists(pdf_path):
            logger.info(f"正在加载PDF文档: {pdf_path}")

            # 使用PyPDF2加载PDF
            documents = []
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num, page in enumerate(pdf_reader.pages[:50]):  # 只加载前50页
                    text = page.extract_text()
                    if text.strip():  # 只添加非空页面
                        # 清理文本
                        text = text.replace("\n", " ").replace("\r", " ")
                        text = " ".join(text.split())  # 去除多余空格

                        if len(text) > 100:  # 只保留有意义的文本
                            documents.append(text)

                logger.info(f"成功加载 {len(documents)} 页民法典内容")
                return documents

        else:
            logger.warning(f"未找到PDF文件: {pdf_path}，使用示例文档")

    except Exception as e:
        logger.error(f"加载PDF文档失败: {e}，使用示例文档")

    # 备用示例文档
    documents = [
        """第一条 为了保护民事主体的合法权益，调整民事关系，维护社会和经济秩序，适应中国特色社会主义发展要求，弘扬社会主义核心价值观，根据宪法，制定本法。

第二条 民法调整平等主体的自然人、法人和非法人组织之间的人身关系和财产关系。

第三条 民事主体的人身权利、财产权利以及其他合法权益受法律保护，任何组织或者个人不得侵犯。""",
        """第四条 民事主体在民事活动中的法律地位一律平等。

第五条 民事主体从事民事活动，应当遵循自愿原则，按照自己的意思设立、变更、终止民事法律关系。

第六条 民事主体从事民事活动，应当遵循公平原则，合理确定各方的权利和义务。""",
        """第七条 民事主体从事民事活动，应当遵循诚信原则，秉持诚实，恪守承诺。

第八条 民事主体从事民事活动，不得违反法律，不得违背公序良俗。

第九条 民事主体从事民事活动，应当有利于节约资源、保护生态环境。""",
        """第十条 处理民事纠纷，应当依照法律；法律没有规定的，可以适用习惯，但是不得违背公序良俗。

第十一条 其他法律对民事关系有特别规定的，依照其规定。

第十二条 中华人民共和国领域内的民事活动，适用中华人民共和国法律。法律另有规定的除外。""",
        """第十三条 自然人从出生时起到死亡时止，具有民事权利能力，依法享有民事权利，承担民事义务。

第十四条 自然人的民事权利能力一律平等。

第十五条 自然人的出生时间和死亡时间，以出生证明、死亡证明记载的时间为准；没有出生证明、死亡证明的，以户籍登记或者其他有效身份登记记载的时间为准。有其他证据足以推翻以上记载时间的，以该证据证明的时间为准。""",
    ]

    logger.info(f"使用示例文档，共 {len(documents)} 个文档")
    return documents


def interactive_demo(rag_system: MultiQueryRAG):
    """交互式演示

    Args:
        rag_system: RAG系统实例
    """
    print("\n" + "=" * 60)
    print("🚀 多查询RAG交互式演示")
    print("=" * 60)
    print("输入 'quit' 或 'exit' 退出程序")
    print("输入 'compare' 进行策略对比测试")
    print("输入 'detail' + 问题: 显示详细的多查询过程")
    print("输入 'debug' + 问题: 开启调试模式查询")
    print("输入 'help' 查看帮助信息")

    evaluator = PerformanceEvaluator(rag_system)

    while True:
        try:
            user_input = input("\n请输入您的问题: ").strip()

            if user_input.lower() in ["quit", "exit", "退出"]:
                print("感谢使用！再见！")
                break

            if user_input.lower() in ["help", "帮助"]:
                print("\n📖 帮助信息:")
                print("- 直接输入问题进行多查询RAG检索")
                print("- 输入 'compare' 进行传统RAG与多查询RAG的对比")
                print("- 输入 'detail' + 问题: 显示详细的多查询过程")
                print("- 输入 'debug' + 问题: 开启调试模式查询")
                print("- 输入 'quit' 或 'exit' 退出程序")
                continue

            if user_input.lower() == "compare":
                query = input("请输入要对比的查询: ").strip()
                if query:
                    comparison = rag_system.compare_strategies(query)
                    evaluator.print_comparison_result(comparison)
                continue

            # 检查是否是详细模式
            if user_input.lower().startswith("detail "):
                query = user_input[7:].strip()
                if query:
                    print("\n🔍 正在处理您的查询（详细模式）...")
                    result = rag_system.query(query, show_process=True)
                    evaluator.print_detailed_process(result)

                    if result.response:
                        print(f"\n💡 AI回答:\n{result.response}")
                continue

            # 检查是否是调试模式
            if user_input.lower().startswith("debug "):
                query = user_input[6:].strip()
                if query:
                    print("\n🔍 正在处理您的查询（调试模式）...")
                    # 临时开启调试模式
                    original_debug = rag_system.debug_mode
                    rag_system.debug_mode = True

                    result = rag_system.query(query, show_process=True)
                    evaluator.print_query_result(
                        result, "多查询RAG结果（调试模式）", show_details=True
                    )

                    # 恢复原始调试模式
                    rag_system.debug_mode = original_debug

                    if result.response:
                        print(f"\n💡 AI回答:\n{result.response}")
                continue

            if not user_input:
                print("请输入有效的问题")
                continue

            # 执行多查询检索
            print("\n🔍 正在执行多查询检索...")
            result = rag_system.query(user_input)

            # 显示检索结果
            evaluator.print_query_result(result, "多查询RAG检索结果")

            # 显示答案
            if result.response:
                print(f"\n💡 AI回答:\n{result.response}")

        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            logger.error(f"演示过程中出现错误: {e}")
            print(f"出现错误: {e}")


def run_batch_tests(rag_system: MultiQueryRAG):
    """运行批量测试

    Args:
        rag_system: RAG系统实例
    """
    test_queries = [
        "什么是民法典？",
        "合同的基本原则有哪些？",
        "物权包括哪些类型？",
        "人格权有哪些具体权利？",
        "继承的方式有哪些？",
        "侵权责任的构成要件是什么？",
    ]

    print("\n" + "=" * 60)
    print("🧪 批量测试开始")
    print("=" * 60)

    evaluator = PerformanceEvaluator(rag_system)
    total_improvement = 0

    for i, query in enumerate(test_queries, 1):
        print(f"\n测试 {i}/{len(test_queries)}: {query}")
        print("-" * 40)

        try:
            comparison = rag_system.compare_strategies(query)
            evaluator.print_comparison_result(comparison)

            total_improvement += comparison.improvement_score

        except Exception as e:
            logger.error(f"测试查询 '{query}' 失败: {e}")

    # 打印总体统计
    avg_improvement = total_improvement / len(test_queries)

    print(f"\n" + "=" * 60)
    print("📊 批量测试总结")
    print("=" * 60)
    print(f"平均改进分数: {avg_improvement:.3f}")

    if avg_improvement > 0.15:
        print("\n✅ 多查询策略总体表现优秀")
    elif avg_improvement > 0.05:
        print("\n🔶 多查询策略有一定改善效果")
    else:
        print("\n❌ 多查询策略改善效果有限")


def setup_environment_variables():
    """设置环境变量"""
    try:
        config = load_environment()
        if config:
            logger.info("环境变量设置成功")
        else:
            logger.warning("环境变量设置失败，使用默认配置")
    except Exception as e:
        logger.error(f"设置环境变量失败: {e}")


def main():
    """主函数"""
    try:
        # 设置环境变量
        setup_environment_variables()

        print("🚀 初始化多查询RAG系统...")

        # 创建LLM实例
        llm = create_deepseek_llm(
            model="deepseek-chat", temperature=0.3, max_tokens=2000
        )
        logger.info("LLM初始化成功")

        # 初始化向量数据库
        vector_db = VectorDatabase(persist_directory="./chroma_db_multi_query")

        # 检查数据库是否为空
        db_info = vector_db.get_collection_info()
        if db_info.get("count", 0) == 0:
            print("📚 加载示例文档...")
            documents = load_sample_documents()
            vector_db.add_documents(documents)
            print(f"✅ 成功添加 {len(documents)} 个文档")
        else:
            print(f"✅ 成功加载现有向量数据库，包含 {db_info.get('count', 0)} 个文档")

        # 创建RAG系统
        rag_system = MultiQueryRAG(llm, vector_db)
        print("✅ 多查询RAG系统初始化完成")

        # 显示菜单
        while True:
            print("\n" + "=" * 50)
            print("🎯 多查询RAG演示系统")
            print("=" * 50)
            print("1. 交互式演示")
            print("2. 批量测试")
            print("3. 退出")

            choice = input("\n请选择操作 (1-3): ").strip()

            if choice == "1":
                interactive_demo(rag_system)
            elif choice == "2":
                run_batch_tests(rag_system)
            elif choice == "3":
                print("感谢使用！")
                break
            else:
                print("无效选择，请重新输入")

    except Exception as e:
        logger.error(f"程序运行失败: {e}")
        print(f"❌ 程序运行失败: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
