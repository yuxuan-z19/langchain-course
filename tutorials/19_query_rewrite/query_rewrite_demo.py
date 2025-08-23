#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第19小节：查询重写（Re-Phrase）RAG优化策略演示

本模块实现了查询重写RAG系统，通过多种重写策略优化用户查询，
提升检索精度和相关性。

主要功能：
1. 多种查询重写策略（标准化、扩展、纠错等）
2. 与传统RAG的对比测试
3. 重写效果评估
4. 使用民法典文档进行实际验证
"""

import os
import sys
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

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
    import requests
    import json
except ImportError:
    print("错误：未安装requests。请运行：pip install requests")
    sys.exit(1)

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from utils.config import load_environment, load_embedding_config
    from utils.llm_factory import create_deepseek_llm, create_llm_from_config
except ImportError:
    print("错误：无法导入配置模块。请确保utils/config.py和utils/llm_factory.py存在。")
    sys.exit(1)

try:
    from chromadb.utils import embedding_functions
except ImportError:
    print("错误：无法导入chromadb embedding_functions。请确保chromadb已正确安装。")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('query_rewrite_demo.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class QueryRewriteResult:
    """查询重写结果"""
    original_query: str
    rewritten_query: str
    strategy: str
    confidence: float
    processing_time: float


@dataclass
class RAGResult:
    """RAG检索结果"""
    query: str
    documents: List[str]
    scores: List[float]
    response: str
    processing_time: float


# 移除自定义的embedding类，使用ChromaDB的标准embedding函数
# 移除自定义的LLM类，使用项目统一的LLM工厂类


class QueryRewriter:
    """查询重写器"""
    
    def __init__(self, llm_model):
        """初始化查询重写器
        
        Args:
            llm_model: LangChain的ChatOpenAI实例
        """
        self.llm_model = llm_model
        self.strategies = {
            "standardize": self._standardize_rewrite,
            "expand": self._expand_rewrite,
            "correct": self._correct_rewrite,
            "structure": self._structure_rewrite
        }
    
    def rewrite_query(self, query: str, strategy: str = "standardize") -> QueryRewriteResult:
        """重写查询"""
        start_time = time.time()
        
        if strategy not in self.strategies:
            logger.warning(f"未知的重写策略: {strategy}，使用默认策略")
            strategy = "standardize"
        
        try:
            rewritten_query = self.strategies[strategy](query)
            confidence = self._calculate_confidence(query, rewritten_query)
            
            processing_time = time.time() - start_time
            
            return QueryRewriteResult(
                original_query=query,
                rewritten_query=rewritten_query,
                strategy=strategy,
                confidence=confidence,
                processing_time=processing_time
            )
        except Exception as e:
            logger.error(f"查询重写失败: {e}")
            processing_time = time.time() - start_time
            return QueryRewriteResult(
                original_query=query,
                rewritten_query=query,  # 失败时返回原查询
                strategy=strategy,
                confidence=0.0,
                processing_time=processing_time
            )
    
    def _standardize_rewrite(self, query: str) -> str:
        """标准化重写策略"""
        prompt = f"""你是一个查询优化专家。请将用户的查询重写为更适合检索的标准化表达。

要求：
1. 保持原始查询的核心意图
2. 使用标准化的术语和表达
3. 去除口语化表达
4. 使查询更加精确和专业

原始查询：{query}

请直接输出重写后的查询，不要包含其他解释："""
        
        try:
            response = self.llm_model.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return query
    
    def _expand_rewrite(self, query: str) -> str:
        """扩展重写策略"""
        prompt = f"""你是一个查询优化专家。请将用户的查询重写为更全面的表达，添加相关的同义词和概念。

要求：
1. 保持原始查询的核心意图
2. 添加相关的同义词和概念
3. 补充可能的上下文信息
4. 使查询更加全面和丰富

原始查询：{query}

请直接输出重写后的查询，不要包含其他解释："""
        
        try:
            response = self.llm_model.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return query
    
    def _correct_rewrite(self, query: str) -> str:
        """纠错重写策略"""
        prompt = f"""你是一个查询优化专家。请修正用户查询中的错别字、语法错误和表达问题。

要求：
1. 修正可能的错别字
2. 改正语法错误
3. 优化表达方式
4. 保持原始意图不变

原始查询：{query}

请直接输出修正后的查询，不要包含其他解释："""
        
        try:
            response = self.llm_model.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return query
    
    def _structure_rewrite(self, query: str) -> str:
        """结构化重写策略"""
        prompt = f"""你是一个查询优化专家。请重新组织用户查询的结构，使其更加清晰和有逻辑。

要求：
1. 重新组织查询结构
2. 突出关键信息
3. 使逻辑更加清晰
4. 保持完整的语义

原始查询：{query}

请直接输出重新结构化后的查询，不要包含其他解释："""
        
        try:
            response = self.llm_model.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return query
    
    def _calculate_confidence(self, original: str, rewritten: str) -> float:
        """计算重写置信度"""
        # 简单的置信度计算：基于长度变化和内容相似性
        if not rewritten or rewritten.strip() == "":
            return 0.0
        
        # 如果重写后的查询与原查询相同，置信度较低
        if original.strip() == rewritten.strip():
            return 0.3
        
        # 基于长度变化计算置信度
        length_ratio = len(rewritten) / max(len(original), 1)
        if 0.5 <= length_ratio <= 2.0:
            return 0.8
        elif 0.3 <= length_ratio <= 3.0:
            return 0.6
        else:
            return 0.4


class VectorDatabase:
    """向量数据库"""
    
    def __init__(self, collection_name: str = "query_rewrite_demo", persist_directory: str = "./demo_query_rewrite_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_function = None
        
        # 创建ChromaDB客户端
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 设置embedding函数
        self._setup_embedding_function()
        
        # 获取或创建集合
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"加载现有集合: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"创建新集合: {collection_name}")
    
    def _setup_embedding_function(self):
        """设置embedding函数"""
        try:
            # 加载embedding配置
            embedding_config = load_embedding_config()
            
            if embedding_config and embedding_config.get('embedding_api_key'):
                # 使用OpenAI兼容的embedding函数
                self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=embedding_config['embedding_api_key'],
                    api_base=embedding_config.get('embedding_base_url', 'https://api.openai.com/v1'),
                    model_name=embedding_config.get('embedding_model_name', 'text-embedding-ada-002')
                )
                logger.info(f"使用OpenAI兼容的embedding函数: {embedding_config.get('embedding_model_name')}")
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
            
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"成功添加 {len(documents)} 个文档到向量数据库")
        except Exception as e:
            logger.error(f"添加文档到向量数据库失败: {e}")
    
    def search(self, query_text: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
        """搜索相似文档"""
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k
            )
            
            documents = results['documents'][0] if results['documents'] else []
            distances = results['distances'][0] if results['distances'] else []
            
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
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {}


class QueryRewriteRAG:
    """查询重写RAG系统"""
    
    def __init__(self, llm_model, vector_db: VectorDatabase, query_rewriter: QueryRewriter):
        """初始化查询重写RAG系统
        
        Args:
            llm_model: LangChain ChatOpenAI实例
            vector_db: 向量数据库实例
            query_rewriter: 查询重写器实例
        """
        self.llm_model = llm_model
        self.vector_db = vector_db
        self.query_rewriter = query_rewriter
    
    def add_documents(self, documents: List[str]):
        """添加文档到知识库"""
        logger.info(f"开始添加 {len(documents)} 个文档到知识库")
        
        # 添加到向量数据库（ChromaDB会自动生成embeddings）
        self.vector_db.add_documents(documents)
        
        logger.info("文档添加完成")
    
    def query_with_rewrite(self, query: str, strategy: str = "standardize", top_k: int = 3) -> Tuple[QueryRewriteResult, RAGResult]:
        """使用查询重写的RAG检索"""
        start_time = time.time()
        
        # 1. 查询重写
        rewrite_result = self.query_rewriter.rewrite_query(query, strategy)
        logger.info(f"查询重写完成: {query} -> {rewrite_result.rewritten_query}")
        
        # 2. 向量检索（ChromaDB会自动生成query embedding）
        documents, scores = self.vector_db.search(rewrite_result.rewritten_query, top_k)
        
        # 3. 生成回答
        response = self._generate_response(rewrite_result.rewritten_query, documents)
        
        processing_time = time.time() - start_time
        
        rag_result = RAGResult(
            query=rewrite_result.rewritten_query,
            documents=documents,
            scores=scores,
            response=response,
            processing_time=processing_time
        )
        
        return rewrite_result, rag_result
    
    def query_traditional(self, query: str, top_k: int = 3) -> RAGResult:
        """传统RAG检索（不使用查询重写）"""
        start_time = time.time()
        
        # 1. 向量检索（ChromaDB会自动生成query embedding）
        documents, scores = self.vector_db.search(query, top_k)
        
        # 2. 生成回答
        response = self._generate_response(query, documents)
        
        processing_time = time.time() - start_time
        
        return RAGResult(
            query=query,
            documents=documents,
            scores=scores,
            response=response,
            processing_time=processing_time
        )
    
    def _generate_response(self, query: str, documents: List[str]) -> str:
        """基于检索到的文档生成回答"""
        if not documents:
            return "抱歉，没有找到相关的文档来回答您的问题。"
        
        context = "\n\n".join([f"文档{i+1}: {doc}" for i, doc in enumerate(documents)])
        
        prompt = f"""基于以下文档内容，回答用户的问题。请确保回答准确、相关且有帮助。

相关文档：
{context}

用户问题：{query}

请提供详细的回答："""
        
        try:
            from langchain_core.messages import HumanMessage
            response = self.llm_model.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return "抱歉，生成回答时出现错误。"


class PerformanceEvaluator:
    """性能评估器"""
    
    def __init__(self):
        self.test_queries = [
            "什么是合同？",
            "合同咋签的？",
            "签合同要注意啥？",
            "合同违约了咋办？",
            "合同可以改吗？",
            "什么情况下合同无效？",
            "合同纠纷怎么解决？",
            "电子合同有效吗？",
            "口头协议算合同吗？",
            "合同期限有规定吗？"
        ]
    
    def evaluate_strategies(self, rag_system: QueryRewriteRAG) -> Dict[str, Any]:
        """评估不同重写策略的效果"""
        strategies = ["standardize", "expand", "correct", "structure"]
        results = {}
        
        for strategy in strategies:
            logger.info(f"评估策略: {strategy}")
            strategy_results = []
            
            for query in self.test_queries:
                try:
                    rewrite_result, rag_result = rag_system.query_with_rewrite(query, strategy)
                    strategy_results.append({
                        "query": query,
                        "rewrite_result": rewrite_result,
                        "rag_result": rag_result
                    })
                except Exception as e:
                    logger.error(f"评估查询 '{query}' 失败: {e}")
            
            results[strategy] = strategy_results
        
        return results
    
    def compare_with_traditional(self, rag_system: QueryRewriteRAG) -> Dict[str, Any]:
        """与传统RAG对比"""
        comparison_results = []
        
        for query in self.test_queries:
            try:
                # 传统RAG
                traditional_result = rag_system.query_traditional(query)
                
                # 查询重写RAG
                rewrite_result, rewrite_rag_result = rag_system.query_with_rewrite(query, "standardize")
                
                comparison_results.append({
                    "query": query,
                    "traditional": traditional_result,
                    "rewrite": {
                        "rewrite_result": rewrite_result,
                        "rag_result": rewrite_rag_result
                    }
                })
            except Exception as e:
                logger.error(f"对比查询 '{query}' 失败: {e}")
        
        return comparison_results
    
    def print_evaluation_report(self, strategy_results: Dict[str, Any], comparison_results: Dict[str, Any]):
        """打印评估报告"""
        print("\n" + "="*80)
        print("查询重写RAG系统评估报告")
        print("="*80)
        
        # 策略效果对比
        print("\n1. 不同重写策略效果对比:")
        print("-"*50)
        
        for strategy, results in strategy_results.items():
            if results:
                avg_confidence = sum(r["rewrite_result"].confidence for r in results) / len(results)
                avg_processing_time = sum(r["rewrite_result"].processing_time for r in results) / len(results)
                
                print(f"\n策略: {strategy}")
                print(f"  平均置信度: {avg_confidence:.3f}")
                print(f"  平均处理时间: {avg_processing_time:.3f}秒")
                
                # 显示几个示例
                print("  重写示例:")
                for i, result in enumerate(results[:3]):
                    print(f"    原查询: {result['query']}")
                    print(f"    重写后: {result['rewrite_result'].rewritten_query}")
                    print()
        
        # 传统RAG对比
        print("\n2. 与传统RAG对比:")
        print("-"*50)
        
        if comparison_results:
            traditional_times = [r["traditional"].processing_time for r in comparison_results]
            rewrite_times = [r["rewrite"]["rag_result"].processing_time for r in comparison_results]
            
            avg_traditional_time = sum(traditional_times) / len(traditional_times)
            avg_rewrite_time = sum(rewrite_times) / len(rewrite_times)
            
            print(f"传统RAG平均处理时间: {avg_traditional_time:.3f}秒")
            print(f"查询重写RAG平均处理时间: {avg_rewrite_time:.3f}秒")
            print(f"时间开销增加: {((avg_rewrite_time - avg_traditional_time) / avg_traditional_time * 100):.1f}%")
            
            # 显示对比示例
            print("\n对比示例:")
            for i, result in enumerate(comparison_results[:3]):
                print(f"\n查询 {i+1}: {result['query']}")
                print(f"重写后: {result['rewrite']['rewrite_result'].rewritten_query}")
                print(f"传统RAG检索到 {len(result['traditional'].documents)} 个文档")
                print(f"重写RAG检索到 {len(result['rewrite']['rag_result'].documents)} 个文档")


def load_sample_documents() -> List[str]:
    """加载示例文档（民法典相关内容）"""
    documents = [
        "合同是民事主体之间设立、变更、终止民事法律关系的协议。依法成立的合同，对当事人具有法律约束力。",
        "当事人订立合同，应当具有相应的民事行为能力。当事人依法可以委托代理人订立合同。",
        "当事人订立合同，有书面形式、口头形式和其他形式。法律、行政法规规定或者当事人约定采用书面形式的，应当采用书面形式。",
        "合同的内容由当事人约定，一般包括下列条款：当事人的姓名或者名称和住所；标的；数量；质量；价款或者报酬；履行期限、地点和方式；违约责任；解决争议的方法。",
        "当事人订立合同后，不得因姓名、名称的变更或者法定代表人、负责人、承办人的变动而影响合同效力。",
        "有下列情形之一的，合同无效：一方以欺诈、胁迫的手段订立合同，损害国家利益；恶意串通，损害国家、集体或者第三人利益；以合法形式掩盖非法目的；损害社会公共利益；违反法律、行政法规的强制性规定。",
        "合同无效或者被撤销后，因该合同取得的财产，应当予以返还；不能返还或者没有必要返还的，应当折价补偿。有过错的一方应当赔偿对方因此所受到的损失，双方都有过错的，应当各自承担相应的责任。",
        "当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。",
        "当事人一方明确表示或者以自己的行为表明不履行合同义务的，对方可以在履行期限届满之前要求其承担违约责任。",
        "当事人既约定违约金，又约定定金的，一方违约时，对方可以选择适用违约金或者定金条款。",
        "合同解除后，尚未履行的，终止履行；已经履行的，根据履行情况和合同性质，当事人可以要求恢复原状、采取其他补救措施，并有权要求赔偿损失。",
        "因不可抗力不能履行合同的，根据不可抗力的影响，部分或者全部免除责任，但法律另有规定的除外。当事人迟延履行后发生不可抗力的，不能免除责任。",
        "电子合同的订立和履行，适用本编和其他法律的规定。以电子数据交换、电子邮件等方式能够有形地表现所载内容，并可以随时调取查用的数据电文，视为书面形式。",
        "合同争议的解决方式包括：协商、调解、仲裁、诉讼。当事人可以通过和解或者调解解决合同争议。当事人不愿和解、调解或者和解、调解不成的，可以根据仲裁协议向仲裁机构申请仲裁。",
        "合同的权利义务终止后，当事人应当遵循诚实信用原则，根据交易习惯履行通知、协助、保密等义务。"
    ]
    
    return documents


def main():
    """主函数"""
    print("第19小节：查询重写（Re-Phrase）RAG优化策略演示")
    print("="*60)
    
    try:
        # 1. 加载配置
        logger.info("加载配置...")
        config = load_environment()
        
        # 获取API密钥
        api_key = config.deepseek_api_key
        base_url = config.deepseek_base_url
        
        if not api_key:
            raise ValueError("未找到有效的API密钥，请检查环境变量配置")
        
        # 2. 初始化LLM模型
        logger.info("初始化LLM模型...")
        llm_model = create_deepseek_llm()
        
        # 3. 初始化组件
        logger.info("初始化系统组件...")
        vector_db = VectorDatabase()
        query_rewriter = QueryRewriter(llm_model)
        rag_system = QueryRewriteRAG(llm_model, vector_db, query_rewriter)
        
        # 5. 加载示例文档
        logger.info("加载示例文档...")
        documents = load_sample_documents()
        
        # 检查是否需要添加文档
        db_info = vector_db.get_collection_info()
        if db_info.get("count", 0) == 0:
            logger.info("向量数据库为空，添加示例文档...")
            rag_system.add_documents(documents)
        else:
            logger.info(f"向量数据库已包含 {db_info.get('count', 0)} 个文档")
        
        # 6. 交互式查询演示
        print("\n系统初始化完成！")
        print("\n可用的重写策略:")
        print("1. standardize - 标准化重写")
        print("2. expand - 扩展重写")
        print("3. correct - 纠错重写")
        print("4. structure - 结构化重写")
        print("5. traditional - 传统RAG（不重写）")
        print("6. evaluate - 运行性能评估")
        print("7. quit - 退出")
        
        while True:
            print("\n" + "-"*50)
            choice = input("请选择操作 (1-7): ").strip()
            
            if choice == "7" or choice.lower() == "quit":
                break
            elif choice == "6" or choice.lower() == "evaluate":
                print("\n开始性能评估...")
                evaluator = PerformanceEvaluator()
                
                # 评估不同策略
                strategy_results = evaluator.evaluate_strategies(rag_system)
                
                # 与传统RAG对比
                comparison_results = evaluator.compare_with_traditional(rag_system)
                
                # 打印报告
                evaluator.print_evaluation_report(strategy_results, comparison_results)
                
            elif choice in ["1", "2", "3", "4", "5"]:
                query = input("请输入您的查询: ").strip()
                if not query:
                    continue
                
                strategy_map = {
                    "1": "standardize",
                    "2": "expand", 
                    "3": "correct",
                    "4": "structure",
                    "5": "traditional"
                }
                
                strategy = strategy_map[choice]
                
                print(f"\n使用策略: {strategy}")
                print("处理中...")
                
                try:
                    if strategy == "traditional":
                        # 传统RAG
                        result = rag_system.query_traditional(query)
                        
                        print(f"\n原始查询: {query}")
                        print(f"检索到 {len(result.documents)} 个相关文档")
                        print(f"处理时间: {result.processing_time:.3f}秒")
                        print(f"\n回答: {result.response}")
                        
                        if result.documents:
                            print("\n相关文档:")
                            for i, (doc, score) in enumerate(zip(result.documents, result.scores)):
                                print(f"{i+1}. (相似度: {score:.3f}) {doc[:100]}...")
                    else:
                        # 查询重写RAG
                        rewrite_result, rag_result = rag_system.query_with_rewrite(query, strategy)
                        
                        print(f"\n原始查询: {rewrite_result.original_query}")
                        print(f"重写查询: {rewrite_result.rewritten_query}")
                        print(f"重写策略: {rewrite_result.strategy}")
                        print(f"重写置信度: {rewrite_result.confidence:.3f}")
                        print(f"重写时间: {rewrite_result.processing_time:.3f}秒")
                        print(f"总处理时间: {rag_result.processing_time:.3f}秒")
                        print(f"检索到 {len(rag_result.documents)} 个相关文档")
                        print(f"\n回答: {rag_result.response}")
                        
                        if rag_result.documents:
                            print("\n相关文档:")
                            for i, (doc, score) in enumerate(zip(rag_result.documents, rag_result.scores)):
                                print(f"{i+1}. (相似度: {score:.3f}) {doc[:100]}...")
                
                except Exception as e:
                    logger.error(f"查询处理失败: {e}")
                    print(f"查询处理失败: {e}")
            
            else:
                print("无效的选择，请重新输入。")
        
        print("\n感谢使用查询重写RAG系统！")
        
    except Exception as e:
        logger.error(f"系统运行失败: {e}")
        print(f"系统运行失败: {e}")


if __name__ == "__main__":
    main()