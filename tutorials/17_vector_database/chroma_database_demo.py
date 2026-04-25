#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChromaDB向量数据库演示

本演示展示如何使用ChromaDB进行向量数据库操作，包括：
1. ChromaDB的初始化和配置
2. 数据的增删改查（CRUD）操作
3. 基本相似性搜索
4. MMR（最大边际相关性）搜索
5. 混合搜索（向量+关键词）
6. 批量操作和性能测试
7. 数据持久化和恢复

作者：jaguarliu
日期：2025年8月
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 添加项目根目录到Python路径
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError as e:
    print(f"请安装ChromaDB: pip install chromadb")
    print(f"错误详情: {e}")
    sys.exit(1)

try:
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError as e:
    print(f"请安装LangChain相关依赖: pip install langchain pypdf")
    print(f"错误详情: {e}")
    sys.exit(1)

# 导入配置模块
try:
    from utils.config import load_embedding_config
except ImportError as e:
    print(f"无法导入配置模块: {e}")
    print("请确保utils/config.py文件存在")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("chroma_database_demo.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ChromaDatabaseDemo:
    """ChromaDB向量数据库演示类"""

    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        初始化ChromaDB演示

        Args:
            persist_directory: 数据持久化目录
        """
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.embedding_function = None
        self.documents_cache = {}

        # 性能统计
        self.performance_stats = {
            "operations": [],
            "search_times": [],
            "insert_times": [],
        }

        logger.info(f"初始化ChromaDB演示，持久化目录: {persist_directory}")

    def setup_chromadb(self) -> bool:
        """
        设置ChromaDB客户端和嵌入函数

        Returns:
            bool: 设置是否成功
        """
        try:
            logger.info("正在设置ChromaDB客户端...")

            # 创建持久化客户端
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            # 设置嵌入函数
            self._setup_embedding_function()

            logger.info("ChromaDB客户端设置成功")
            return True

        except Exception as e:
            logger.error(f"设置ChromaDB客户端失败: {e}")
            return False

    def _setup_embedding_function(self):
        """
        设置嵌入函数
        """
        try:
            # 尝试加载配置的向量模型
            embedding_config = load_embedding_config()

            if embedding_config and embedding_config.get("api_key"):
                logger.info(
                    f"使用配置的向量模型: {embedding_config.get('model_name', 'default')}"
                )

                # 使用OpenAI兼容的嵌入函数
                self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=embedding_config["api_key"],
                    api_base=embedding_config.get(
                        "base_url", "https://api.openai.com/v1"
                    ),
                    model_name=embedding_config.get(
                        "model_name", "text-embedding-ada-002"
                    ),
                )
            else:
                logger.info("使用默认的SentenceTransformer嵌入函数")
                # 使用默认的嵌入函数
                self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

        except Exception as e:
            logger.warning(f"设置配置的嵌入函数失败，使用默认函数: {e}")
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

    def create_collection(self, collection_name: str = "civil_code_collection") -> bool:
        """
        创建或获取集合

        Args:
            collection_name: 集合名称

        Returns:
            bool: 操作是否成功
        """
        try:
            logger.info(f"创建或获取集合: {collection_name}")

            # 获取或创建集合
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "民法典向量数据库集合"},
            )

            # 获取集合信息
            count = self.collection.count()
            logger.info(f"集合 '{collection_name}' 创建成功，当前文档数量: {count}")

            return True

        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False

    def load_civil_code_documents(self, pdf_path: str = None) -> List[Dict[str, Any]]:
        """
        加载民法典PDF文档并进行分块处理

        Args:
            pdf_path: PDF文件路径，默认为项目根目录下的docs/中华人民共和国民法典.pdf

        Returns:
            分块后的文档列表
        """
        try:
            # 如果没有指定路径，使用默认的民法典PDF路径
            if pdf_path is None:
                # 获取项目根目录路径（从当前文件向上两级）
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(current_dir))
                pdf_path = os.path.join(
                    project_root, "docs", "中华人民共和国民法典.pdf"
                )

            logger.info(f"尝试加载PDF文件: {pdf_path}")

            # 检查文件是否存在
            if not os.path.exists(pdf_path):
                logger.warning(f"PDF文件不存在: {pdf_path}，使用示例文档")
                return self._create_sample_documents()

            # 加载PDF文档
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            # 文档分割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            )

            split_docs = text_splitter.split_documents(documents)

            # 转换为字典格式
            doc_list = []
            for i, doc in enumerate(split_docs[:100]):  # 限制前100个片段
                doc_dict = {
                    "id": f"civil_code_{i:04d}",
                    "content": doc.page_content.strip(),
                    "metadata": {
                        "source": pdf_path,
                        "page": doc.metadata.get("page", 0),
                        "chunk_index": i,
                        "timestamp": datetime.now().isoformat(),
                        "document_type": "civil_code",
                    },
                }
                doc_list.append(doc_dict)

            logger.info(f"成功加载 {len(doc_list)} 个文档片段")
            return doc_list

        except Exception as e:
            logger.error(f"加载民法典文档失败: {e}")
            return self._create_sample_documents()

    def _create_sample_documents(self) -> List[Dict[str, Any]]:
        """
        创建示例文档

        Returns:
            List[Dict]: 示例文档列表
        """
        sample_docs = [
            {
                "id": "sample_001",
                "content": "民法典是新中国第一部以法典命名的法律，是新时代我国社会主义法治建设的重大成果。",
                "metadata": {
                    "source": "sample",
                    "page": 1,
                    "chunk_index": 0,
                    "timestamp": datetime.now().isoformat(),
                    "document_type": "civil_code",
                    "category": "总则",
                },
            },
            {
                "id": "sample_002",
                "content": "自然人的人身自由、人格尊严受法律保护。任何组织或者个人不得侵害他人的人身自由、人格尊严。",
                "metadata": {
                    "source": "sample",
                    "page": 2,
                    "chunk_index": 1,
                    "timestamp": datetime.now().isoformat(),
                    "document_type": "civil_code",
                    "category": "人格权",
                },
            },
            {
                "id": "sample_003",
                "content": "物权是权利人依法对特定的物享有直接支配和排他的权利，包括所有权、用益物权和担保物权。",
                "metadata": {
                    "source": "sample",
                    "page": 3,
                    "chunk_index": 2,
                    "timestamp": datetime.now().isoformat(),
                    "document_type": "civil_code",
                    "category": "物权",
                },
            },
            {
                "id": "sample_004",
                "content": "合同是民事主体之间设立、变更、终止民事法律关系的协议。依法成立的合同，对当事人具有法律约束力。",
                "metadata": {
                    "source": "sample",
                    "page": 4,
                    "chunk_index": 3,
                    "timestamp": datetime.now().isoformat(),
                    "document_type": "civil_code",
                    "category": "合同",
                },
            },
            {
                "id": "sample_005",
                "content": "侵权责任是指行为人因其行为侵害他人民事权益应当承担的民事责任。侵权行为危害他人人身、财产安全。",
                "metadata": {
                    "source": "sample",
                    "page": 5,
                    "chunk_index": 4,
                    "timestamp": datetime.now().isoformat(),
                    "document_type": "civil_code",
                    "category": "侵权责任",
                },
            },
        ]

        logger.info(f"创建了 {len(sample_docs)} 个示例文档")
        return sample_docs

    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        添加文档到集合

        Args:
            documents: 文档列表

        Returns:
            bool: 操作是否成功
        """
        try:
            start_time = time.time()
            logger.info(f"正在添加 {len(documents)} 个文档到集合...")

            # 准备数据
            ids = [doc["id"] for doc in documents]
            contents = [doc["content"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]

            # 分批处理，每批最多10个文档（DashScope API限制）
            batch_size = 10
            total_added = 0

            for i in range(0, len(documents), batch_size):
                batch_ids = ids[i : i + batch_size]
                batch_contents = contents[i : i + batch_size]
                batch_metadatas = metadatas[i : i + batch_size]

                # 添加当前批次到ChromaDB
                self.collection.add(
                    documents=batch_contents, metadatas=batch_metadatas, ids=batch_ids
                )

                total_added += len(batch_ids)
                logger.info(f"已添加 {total_added}/{len(documents)} 个文档")

            # 缓存文档
            for doc in documents:
                self.documents_cache[doc["id"]] = doc

            end_time = time.time()
            insert_time = end_time - start_time
            self.performance_stats["insert_times"].append(insert_time)

            logger.info(
                f"成功添加所有 {len(documents)} 个文档，耗时: {insert_time:.2f}秒"
            )
            return True

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False

    def demo_basic_operations(self):
        """
        演示基本CRUD操作
        """
        logger.info("\n" + "=" * 50)
        logger.info("演示基本CRUD操作")
        logger.info("=" * 50)

        try:
            # 1. 查询集合信息
            count = self.collection.count()
            logger.info(f"当前集合中的文档数量: {count}")

            # 2. 获取指定文档
            if count > 0:
                # 获取前3个文档
                results = self.collection.get(
                    limit=3, include=["documents", "metadatas"]
                )
                logger.info(f"获取前3个文档:")
                for i, (doc_id, content, metadata) in enumerate(
                    zip(results["ids"], results["documents"], results["metadatas"])
                ):
                    logger.info(f"  文档{i+1}: ID={doc_id}")
                    logger.info(f"    内容: {content[:100]}...")
                    logger.info(f"    类别: {metadata.get('category', 'unknown')}")

            # 3. 更新文档示例
            if count > 0:
                first_id = results["ids"][0]
                logger.info(f"\n更新文档示例 (ID: {first_id})")

                self.collection.update(
                    ids=[first_id],
                    metadatas=[
                        {
                            **results["metadatas"][0],
                            "updated_at": datetime.now().isoformat(),
                            "status": "updated",
                        }
                    ],
                )
                logger.info("文档元数据更新成功")

            # 4. 条件查询示例
            logger.info("\n条件查询示例 - 查找人格权相关文档:")
            filtered_results = self.collection.get(
                where={"category": "人格权"}, include=["documents", "metadatas"]
            )

            if filtered_results["ids"]:
                for doc_id, content in zip(
                    filtered_results["ids"], filtered_results["documents"]
                ):
                    logger.info(f"  找到文档: {doc_id}")
                    logger.info(f"    内容: {content[:100]}...")
            else:
                logger.info("  未找到人格权相关文档")

        except Exception as e:
            logger.error(f"基本操作演示失败: {e}")

    def demo_similarity_search(self):
        """
        演示相似性搜索
        """
        logger.info("\n" + "=" * 50)
        logger.info("演示相似性搜索")
        logger.info("=" * 50)

        search_queries = [
            "什么是人格权？",
            "合同的法律效力如何？",
            "物权包括哪些类型？",
            "侵权责任的构成要件",
        ]

        for query in search_queries:
            try:
                start_time = time.time()
                logger.info(f"\n查询: {query}")

                # 执行相似性搜索
                results = self.collection.query(
                    query_texts=[query],
                    n_results=3,
                    include=["documents", "distances", "metadatas"],
                )

                end_time = time.time()
                search_time = end_time - start_time
                self.performance_stats["search_times"].append(search_time)

                logger.info(f"搜索耗时: {search_time:.3f}秒")
                logger.info("搜索结果:")

                for i, (doc_id, content, distance, metadata) in enumerate(
                    zip(
                        results["ids"][0],
                        results["documents"][0],
                        results["distances"][0],
                        results["metadatas"][0],
                    )
                ):
                    similarity = 1 - distance  # 转换为相似度
                    logger.info(f"  结果{i+1}: (相似度: {similarity:.3f})")
                    logger.info(f"    ID: {doc_id}")
                    logger.info(f"    类别: {metadata.get('category', 'unknown')}")
                    logger.info(f"    内容: {content[:150]}...")

            except Exception as e:
                logger.error(f"相似性搜索失败 (查询: {query}): {e}")

    def demo_mmr_search(self):
        """
        演示MMR（最大边际相关性）搜索
        """
        logger.info("\n" + "=" * 50)
        logger.info("演示MMR搜索")
        logger.info("=" * 50)

        query = "民法典的基本原则和规定"
        logger.info(f"查询: {query}")

        try:
            # 首先获取更多候选结果
            candidates = self.collection.query(
                query_texts=[query],
                n_results=10,
                include=["documents", "distances", "metadatas"],
            )

            if not candidates["ids"][0]:
                logger.warning("没有找到候选结果")
                return

            # 实现简化的MMR算法
            selected_indices = self._mmr_selection(
                candidates, k=5, lambda_param=0.7  # 0.7权重给相关性，0.3给多样性
            )

            logger.info("MMR搜索结果 (平衡相关性和多样性):")
            for i, idx in enumerate(selected_indices):
                doc_id = candidates["ids"][0][idx]
                content = candidates["documents"][0][idx]
                distance = candidates["distances"][0][idx]
                metadata = candidates["metadatas"][0][idx]

                similarity = 1 - distance
                logger.info(f"  结果{i+1}: (相似度: {similarity:.3f})")
                logger.info(f"    ID: {doc_id}")
                logger.info(f"    类别: {metadata.get('category', 'unknown')}")
                logger.info(f"    内容: {content[:150]}...")

        except Exception as e:
            logger.error(f"MMR搜索失败: {e}")

    def _mmr_selection(
        self, candidates: Dict, k: int = 5, lambda_param: float = 0.5
    ) -> List[int]:
        """
        实现MMR选择算法

        Args:
            candidates: 候选结果
            k: 选择的结果数量
            lambda_param: 相关性权重 (0-1)

        Returns:
            List[int]: 选中的结果索引
        """
        if not candidates["ids"][0]:
            return []

        selected = []
        remaining = list(range(len(candidates["ids"][0])))

        # 选择第一个最相关的结果
        if remaining:
            best_idx = 0  # 第一个结果通常是最相关的
            selected.append(best_idx)
            remaining.remove(best_idx)

        # 迭代选择剩余结果
        while len(selected) < k and remaining:
            best_score = -float("inf")
            best_idx = None

            for idx in remaining:
                # 相关性分数 (距离越小，相关性越高)
                relevance = 1 - candidates["distances"][0][idx]

                # 多样性分数 (与已选结果的最小相似度)
                if selected:
                    diversities = []
                    current_content = candidates["documents"][0][idx]

                    for sel_idx in selected:
                        selected_content = candidates["documents"][0][sel_idx]
                        # 简单的多样性计算：基于内容长度差异和类别差异
                        content_diversity = abs(
                            len(current_content) - len(selected_content)
                        ) / max(len(current_content), len(selected_content))

                        current_category = candidates["metadatas"][0][idx].get(
                            "category", ""
                        )
                        selected_category = candidates["metadatas"][0][sel_idx].get(
                            "category", ""
                        )
                        category_diversity = (
                            1.0 if current_category != selected_category else 0.0
                        )

                        diversity = (content_diversity + category_diversity) / 2
                        diversities.append(diversity)

                    min_diversity = min(diversities)
                else:
                    min_diversity = 1.0

                # MMR分数
                mmr_score = (
                    lambda_param * relevance + (1 - lambda_param) * min_diversity
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                break

        return selected

    def demo_hybrid_search(self):
        """
        演示混合搜索（向量搜索 + 关键词过滤）
        """
        logger.info("\n" + "=" * 50)
        logger.info("演示混合搜索")
        logger.info("=" * 50)

        # 混合搜索示例
        search_cases = [
            {
                "query": "法律责任和义务",
                "category_filter": "侵权责任",
                "description": "在侵权责任类别中搜索法律责任相关内容",
            },
            {
                "query": "权利保护",
                "category_filter": "人格权",
                "description": "在人格权类别中搜索权利保护相关内容",
            },
        ]

        for case in search_cases:
            try:
                logger.info(f"\n{case['description']}")
                logger.info(f"查询: {case['query']}")
                logger.info(f"类别过滤: {case['category_filter']}")

                # 执行混合搜索
                results = self.collection.query(
                    query_texts=[case["query"]],
                    n_results=5,
                    where={"category": case["category_filter"]},
                    include=["documents", "distances", "metadatas"],
                )

                if results["ids"][0]:
                    logger.info("混合搜索结果:")
                    for i, (doc_id, content, distance, metadata) in enumerate(
                        zip(
                            results["ids"][0],
                            results["documents"][0],
                            results["distances"][0],
                            results["metadatas"][0],
                        )
                    ):
                        similarity = 1 - distance
                        logger.info(f"  结果{i+1}: (相似度: {similarity:.3f})")
                        logger.info(f"    ID: {doc_id}")
                        logger.info(f"    类别: {metadata.get('category', 'unknown')}")
                        logger.info(f"    内容: {content[:150]}...")
                else:
                    logger.info("未找到匹配的结果")

                    # 尝试不带过滤的搜索
                    logger.info("尝试不带类别过滤的搜索:")
                    fallback_results = self.collection.query(
                        query_texts=[case["query"]],
                        n_results=3,
                        include=["documents", "distances", "metadatas"],
                    )

                    for i, (doc_id, content, distance, metadata) in enumerate(
                        zip(
                            fallback_results["ids"][0],
                            fallback_results["documents"][0],
                            fallback_results["distances"][0],
                            fallback_results["metadatas"][0],
                        )
                    ):
                        similarity = 1 - distance
                        logger.info(f"  结果{i+1}: (相似度: {similarity:.3f})")
                        logger.info(f"    ID: {doc_id}")
                        logger.info(f"    类别: {metadata.get('category', 'unknown')}")
                        logger.info(f"    内容: {content[:100]}...")

            except Exception as e:
                logger.error(f"混合搜索失败: {e}")

    def demo_batch_operations(self):
        """
        演示批量操作和性能测试
        """
        logger.info("\n" + "=" * 50)
        logger.info("演示批量操作和性能测试")
        logger.info("=" * 50)

        try:
            # 批量查询测试
            batch_queries = [
                "民事权利能力",
                "法人的设立条件",
                "代理权的行使",
                "诉讼时效期间",
                "不当得利返还",
            ]

            logger.info(f"执行批量查询测试 ({len(batch_queries)} 个查询)")
            start_time = time.time()

            batch_results = []
            for query in batch_queries:
                result = self.collection.query(
                    query_texts=[query], n_results=3, include=["documents", "distances"]
                )
                batch_results.append(result)

            end_time = time.time()
            total_time = end_time - start_time
            avg_time = total_time / len(batch_queries)

            logger.info(f"批量查询完成:")
            logger.info(f"  总耗时: {total_time:.3f}秒")
            logger.info(f"  平均每查询: {avg_time:.3f}秒")
            logger.info(f"  查询吞吐量: {len(batch_queries)/total_time:.2f} 查询/秒")

            # 显示部分结果
            logger.info("\n部分批量查询结果:")
            for i, (query, result) in enumerate(
                zip(batch_queries[:2], batch_results[:2])
            ):
                logger.info(f"  查询{i+1}: {query}")
                if result["ids"][0]:
                    best_match = result["documents"][0][0]
                    similarity = 1 - result["distances"][0][0]
                    logger.info(
                        f"    最佳匹配 (相似度: {similarity:.3f}): {best_match[:100]}..."
                    )

        except Exception as e:
            logger.error(f"批量操作测试失败: {e}")

    def demo_data_management(self):
        """
        演示数据管理操作
        """
        logger.info("\n" + "=" * 50)
        logger.info("演示数据管理操作")
        logger.info("=" * 50)

        try:
            # 获取集合统计信息
            count = self.collection.count()
            logger.info(f"当前集合统计:")
            logger.info(f"  文档总数: {count}")

            # 按类别统计
            all_docs = self.collection.get(include=["metadatas"])
            category_stats = {}

            for metadata in all_docs["metadatas"]:
                category = metadata.get("category", "unknown")
                category_stats[category] = category_stats.get(category, 0) + 1

            logger.info("  按类别统计:")
            for category, count in category_stats.items():
                logger.info(f"    {category}: {count} 个文档")

            # 演示删除操作
            if count > 0:
                logger.info("\n演示删除操作:")

                # 查找要删除的文档
                docs_to_delete = self.collection.get(
                    where={"status": "updated"}, include=["metadatas"]
                )

                if docs_to_delete["ids"]:
                    delete_id = docs_to_delete["ids"][0]
                    logger.info(f"删除文档: {delete_id}")

                    self.collection.delete(ids=[delete_id])

                    # 验证删除
                    new_count = self.collection.count()
                    logger.info(f"删除后文档数量: {new_count}")
                else:
                    logger.info("没有找到可删除的文档")

        except Exception as e:
            logger.error(f"数据管理操作失败: {e}")

    def print_performance_summary(self):
        """
        打印性能统计摘要
        """
        logger.info("\n" + "=" * 50)
        logger.info("性能统计摘要")
        logger.info("=" * 50)

        try:
            # 搜索性能统计
            if self.performance_stats["search_times"]:
                search_times = self.performance_stats["search_times"]
                logger.info(f"搜索性能统计:")
                logger.info(f"  搜索次数: {len(search_times)}")
                logger.info(f"  平均搜索时间: {np.mean(search_times):.3f}秒")
                logger.info(f"  最快搜索时间: {np.min(search_times):.3f}秒")
                logger.info(f"  最慢搜索时间: {np.max(search_times):.3f}秒")

            # 插入性能统计
            if self.performance_stats["insert_times"]:
                insert_times = self.performance_stats["insert_times"]
                logger.info(f"\n插入性能统计:")
                logger.info(f"  插入操作次数: {len(insert_times)}")
                logger.info(f"  平均插入时间: {np.mean(insert_times):.3f}秒")

            # 集合信息
            if self.collection:
                count = self.collection.count()
                logger.info(f"\n集合信息:")
                logger.info(f"  最终文档数量: {count}")
                logger.info(f"  持久化目录: {self.persist_directory}")

        except Exception as e:
            logger.error(f"性能统计失败: {e}")

    def run_comprehensive_demo(self):
        """
        运行完整的ChromaDB演示
        """
        logger.info("开始ChromaDB向量数据库完整演示")
        logger.info("=" * 60)

        try:
            # 1. 设置ChromaDB
            if not self.setup_chromadb():
                logger.error("ChromaDB设置失败，演示终止")
                return False

            # 2. 创建集合
            if not self.create_collection():
                logger.error("创建集合失败，演示终止")
                return False

            # 3. 加载和添加文档
            documents = self.load_civil_code_documents()
            if not documents:
                logger.error("加载文档失败，演示终止")
                return False

            if not self.add_documents(documents):
                logger.error("添加文档失败，演示终止")
                return False

            # 4. 演示各种操作
            self.demo_basic_operations()
            self.demo_similarity_search()
            self.demo_mmr_search()
            self.demo_hybrid_search()
            self.demo_batch_operations()
            self.demo_data_management()

            # 5. 性能统计
            self.print_performance_summary()

            logger.info("\n" + "=" * 60)
            logger.info("ChromaDB向量数据库演示完成！")
            logger.info("=" * 60)

            return True

        except Exception as e:
            logger.error(f"演示过程中发生错误: {e}")
            return False


def main():
    """
    主函数
    """
    try:
        # 创建演示实例
        demo = ChromaDatabaseDemo(persist_directory="./chroma_db_demo")

        # 运行完整演示
        success = demo.run_comprehensive_demo()

        if success:
            print("\n演示成功完成！")
            print("\n你可以查看以下内容：")
            print("1. 日志文件: chroma_database_demo.log")
            print("2. 持久化数据: ./chroma_db_demo/")
            print("3. 尝试修改查询内容，体验不同的搜索效果")
        else:
            print("\n演示过程中遇到问题，请查看日志了解详情")

    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"\n演示失败: {e}")
        logger.error(f"主函数执行失败: {e}")


if __name__ == "__main__":
    main()
