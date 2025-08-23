#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第18小节：父子分段RAG技术演示

本演示展示如何实现父子分段RAG系统，解决传统RAG中文档碎片化和召回准确率不高的问题。

主要功能：
1. 父子文档分段策略
2. ChromaDB存储父子文档关系
3. 子文档召回+父文档扩展的检索机制
4. 多种分段策略对比
5. 与传统RAG的效果对比
6. 使用民法典文档进行实际测试

作者：LangChain课程
日期：2024年
"""

import os
import sys
import logging
import time
import json
import hashlib
import uuid
import requests
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入配置模块
try:
    from utils.config import load_embedding_config
except ImportError as e:
    print(f"无法导入配置模块: {e}")
    print("请确保utils/config.py文件存在")
    sys.exit(1)

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("警告：未安装python-dotenv，请运行: pip install python-dotenv")
except Exception as e:
    print(f"警告：加载.env文件失败: {e}")

# 第三方库
try:
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    print(f"缺少ChromaDB依赖: {e}")
    print("请运行: pip install chromadb")
    sys.exit(1)

try:
    import PyPDF2
except ImportError as e:
    print(f"缺少PyPDF2依赖: {e}")
    print("请运行: pip install PyPDF2")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parent_child_chunking.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 嵌入模型基类和实现
# ============================================================================

class Embeddings(ABC):
    """嵌入模型基类"""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文档"""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        pass

class DashScopeEmbeddings(Embeddings):
    """阿里云DashScope嵌入模型"""
    
    def __init__(self, 
                 api_key: str = None,
                 base_url: str = None,
                 model_name: str = "text-embedding-v4"):
        self.api_key = api_key
        self.base_url = base_url or "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
        self.model_name = model_name
        
        if not self.api_key:
            raise ValueError("DashScope API密钥不能为空")
    
    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """调用DashScope API获取嵌入向量"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.model_name,
            'input': {'texts': texts},
            'parameters': {'text_type': 'document'}
        }
        
        try:
            logger.debug(f"正在调用DashScope API，文本数量: {len(texts)}")
            # 使用正确的DashScope嵌入API端点
            api_url = 'https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding'
            response = requests.post(api_url, headers=headers, json=data, timeout=30)
            
            if response.status_code != 200:
                error_msg = f"DashScope API错误 - 状态码: {response.status_code}, 响应: {response.text}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            result = response.json()
            logger.debug(f"DashScope API调用成功，返回结果: {len(result.get('output', {}).get('embeddings', []))} 个嵌入向量")
            
            if 'output' in result and 'embeddings' in result['output']:
                embeddings = []
                for embedding_data in result['output']['embeddings']:
                    if 'embedding' in embedding_data:
                        embeddings.append(embedding_data['embedding'])
                return embeddings
            else:
                error_msg = f"DashScope API响应格式错误: {result}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
        except requests.exceptions.RequestException as e:
            error_msg = f"DashScope API网络请求失败: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"DashScope API调用失败: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文档"""
        if not texts:
            return []
        
        # DashScope API批处理大小限制为10
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._call_api(batch)
            all_embeddings.extend(batch_embeddings)
            
            # 避免API限流
            if i + batch_size < len(texts):
                time.sleep(0.1)
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        return self.embed_documents([text])[0]

class OpenAIEmbeddings(Embeddings):
    """OpenAI嵌入模型"""
    
    def __init__(self, 
                 api_key: str = None,
                 base_url: str = None,
                 model_name: str = "text-embedding-ada-002"):
        self.api_key = api_key
        self.base_url = base_url or 'https://api.openai.com/v1'
        self.model_name = model_name
        
        if not self.api_key:
            raise ValueError("OpenAI API密钥不能为空")
    
    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """调用OpenAI API"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.model_name,
            'input': texts
        }
        
        try:
            url = f"{self.base_url}/embeddings" if self.base_url else "https://api.openai.com/v1/embeddings"
            logger.debug(f"正在调用OpenAI API，文本数量: {len(texts)}, URL: {url}")
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code != 200:
                error_msg = f"OpenAI API错误 - 状态码: {response.status_code}, 响应: {response.text}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            result = response.json()
            logger.debug(f"OpenAI API调用成功，返回结果: {len(result.get('data', []))} 个嵌入向量")
            
            if 'data' in result:
                embeddings = [item['embedding'] for item in result['data']]
                return embeddings
            else:
                error_msg = f"OpenAI API响应格式错误: {result}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"OpenAI API网络请求失败: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"OpenAI API调用失败: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文档"""
        if not texts:
            return []
        
        # 批量处理，每次最多100个文本
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._call_api(batch)
            all_embeddings.extend(batch_embeddings)
            
            # 避免API限流
            if i + batch_size < len(texts):
                time.sleep(0.1)
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        return self.embed_documents([text])[0]



# ============================================================================
# 文本分割器
# ============================================================================

class RecursiveCharacterTextSplitter:
    """递归字符文本分割器"""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """分割文本"""
        if len(text) <= self.chunk_size:
            return [text]
        
        # 尝试使用分隔符分割
        for separator in self.separators:
            if separator in text:
                chunks = self._split_by_separator(text, separator)
                if chunks:
                    return chunks
        
        # 如果没有合适的分隔符，强制分割
        return self._force_split(text)
    
    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """使用指定分隔符分割"""
        parts = text.split(separator)
        chunks = []
        current_chunk = ""
        
        for part in parts:
            # 如果当前块加上新部分不超过限制
            if len(current_chunk) + len(part) + len(separator) <= self.chunk_size:
                if current_chunk:
                    current_chunk += separator + part
                else:
                    current_chunk = part
            else:
                # 保存当前块
                if current_chunk:
                    chunks.append(current_chunk)
                
                # 如果单个部分太长，递归分割
                if len(part) > self.chunk_size:
                    sub_chunks = self.split_text(part)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part
        
        # 添加最后一块
        if current_chunk:
            chunks.append(current_chunk)
        
        # 处理重叠
        return self._add_overlap(chunks)
    
    def _force_split(self, text: str) -> List[str]:
        """强制分割文本"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """添加重叠"""
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            # 从前一个块的末尾取重叠部分
            overlap = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
            overlapped_chunk = overlap + current_chunk
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks

# ============================================================================
# 配置和数据结构
# ============================================================================

@dataclass
class ChunkingConfig:
    """分段配置类"""
    parent_chunk_size: int = 1500
    parent_overlap: int = 150
    child_chunk_size: int = 300
    child_overlap: int = 30
    strategy: str = "fixed_size"  # fixed_size, semantic, hybrid

# ============================================================================
# 父子分段器
# ============================================================================

class ParentChildChunker:
    """父子分段器"""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.parent_chunk_size,
            chunk_overlap=config.parent_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.child_chunk_size,
            chunk_overlap=config.child_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
    
    def chunk_document(self, text: str, source: str = "") -> List[Dict[str, Any]]:
        """将文档分割为父子片段"""
        logger.info(f"开始分割文档，长度: {len(text)} 字符")
        
        # 首先分割为父文档
        parent_chunks = self.parent_splitter.split_text(text)
        logger.info(f"生成 {len(parent_chunks)} 个父文档片段")
        
        all_chunks = []
        
        for parent_idx, parent_text in enumerate(parent_chunks):
            parent_id = f"parent_{uuid.uuid4().hex[:8]}"
            
            # 将父文档进一步分割为子文档
            child_chunks = self.child_splitter.split_text(parent_text)
            
            for child_idx, child_text in enumerate(child_chunks):
                child_id = f"child_{uuid.uuid4().hex[:8]}"
                
                chunk_data = {
                    "child_id": child_id,
                    "child_content": child_text,
                    "parent_id": parent_id,
                    "parent_content": parent_text,
                    "metadata": {
                        "source": source,
                        "parent_index": parent_idx,
                        "child_index": child_idx,
                        "parent_length": len(parent_text),
                        "child_length": len(child_text)
                    }
                }
                all_chunks.append(chunk_data)
        
        logger.info(f"总共生成 {len(all_chunks)} 个子文档片段")
        return all_chunks

# ============================================================================
# 父子分段RAG系统
# ============================================================================

class ParentChildRAG:
    """父子分段RAG系统"""
    
    def __init__(self, 
                 collection_name: str = "parent_child_rag",
                 persist_directory: str = "./chroma_parent_child_db",
                 embedding_model: Optional[Embeddings] = None):
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # 初始化嵌入模型
        if embedding_model:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = self._setup_embedding_model()
        
        # 初始化ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 创建嵌入函数
        self.embedding_function = self._create_embedding_function()
        
        # 创建或获取集合
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"加载现有集合: {collection_name}")
        except:
            # 如果集合已存在但嵌入函数不匹配，先删除再创建
            try:
                self.client.delete_collection(collection_name)
                logger.info(f"删除现有集合: {collection_name}")
            except:
                pass
            
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "父子分段RAG集合"}
            )
            logger.info(f"创建新集合: {collection_name}")
    
    def _setup_embedding_model(self) -> Embeddings:
        """设置嵌入模型"""
        try:
            # 加载嵌入配置
            logger.info("正在加载嵌入模型配置...")
            config = load_embedding_config()
            
            if not config:
                raise ValueError("无法加载嵌入模型配置，请检查.env文件")
                
            if not config.get('api_key'):
                raise ValueError("未找到有效的API密钥，请检查.env文件中的EMBEDDING_API_KEY")
            
            # 根据base_url判断使用哪种嵌入模型
            base_url = config.get('base_url', '')
            api_key = config['api_key']
            model_name = config.get('model_name', 'text-embedding-v2')
            
            logger.info(f"配置信息 - API密钥: {'已设置' if api_key else '未设置'}, 模型: {model_name}, 基础URL: {base_url}")
            
            if 'dashscope' in base_url.lower():
                # 使用DashScope嵌入
                logger.info(f"初始化DashScope嵌入模型: {model_name}")
                embedding_model = DashScopeEmbeddings(
                    api_key=api_key,
                    model_name=model_name,
                    base_url=base_url
                )
                logger.info("DashScope嵌入模型初始化成功")
                return embedding_model
            else:
                # 使用OpenAI嵌入
                logger.info(f"初始化OpenAI嵌入模型: {model_name}")
                embedding_model = OpenAIEmbeddings(
                    api_key=api_key,
                    model_name=model_name or 'text-embedding-ada-002',
                    base_url=base_url
                )
                logger.info("OpenAI嵌入模型初始化成功")
                return embedding_model
                
        except Exception as e:
            logger.error(f"设置嵌入模型失败: {str(e)}")
            logger.error(f"错误类型: {type(e).__name__}")
            raise ValueError(f"无法设置嵌入模型: {str(e)}")
    
    def _create_embedding_function(self):
        """创建ChromaDB嵌入函数"""
        class ChromaEmbeddingFunction:
            def __init__(self, embedding_model):
                self.embedding_model = embedding_model
            
            def __call__(self, input: List[str]) -> List[List[float]]:
                return self.embedding_model.embed_documents(input)
            
            def name(self) -> str:
                return "custom_embedding_function"
        
        return ChromaEmbeddingFunction(self.embedding_model)
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """添加文档到向量数据库"""
        logger.info(f"开始添加 {len(chunks)} 个文档片段到数据库")
        
        # 准备数据
        child_ids = []
        child_contents = []
        metadatas = []
        
        for chunk in chunks:
            child_ids.append(chunk["child_id"])
            child_contents.append(chunk["child_content"])
            
            # 准备元数据（包含父文档信息）
            metadata = {
                "parent_id": chunk["parent_id"],
                "parent_content": chunk["parent_content"],
                "source": chunk["metadata"]["source"],
                "parent_index": chunk["metadata"]["parent_index"],
                "child_index": chunk["metadata"]["child_index"],
                "parent_length": chunk["metadata"]["parent_length"],
                "child_length": chunk["metadata"]["child_length"]
            }
            metadatas.append(metadata)
        
        # 批量添加到ChromaDB
        try:
            self.collection.add(
                ids=child_ids,
                documents=child_contents,
                metadatas=metadatas
            )
            logger.info(f"成功添加 {len(chunks)} 个文档片段")
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise
    
    def search(self, 
               query: str, 
               top_k: int = 5, 
               expand_parents: bool = True,
               merge_adjacent: bool = True) -> List[Dict[str, Any]]:
        """搜索相关文档"""
        logger.info(f"搜索查询: {query}")
        
        # 在子文档中搜索
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        if not results['documents'][0]:
            logger.warning("未找到相关文档")
            return []
        
        search_results = []
        parent_ids_seen = set()
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            result = {
                "child_content": doc,
                "child_id": results['ids'][0][i],
                "similarity_score": 1 - distance,  # 转换为相似度分数
                "metadata": metadata
            }
            
            # 如果启用父文档扩展
            if expand_parents:
                result["parent_content"] = metadata["parent_content"]
                result["parent_id"] = metadata["parent_id"]
                parent_ids_seen.add(metadata["parent_id"])
            
            search_results.append(result)
        
        # 如果启用相邻父文档合并
        if expand_parents and merge_adjacent:
            search_results = self._merge_adjacent_parents(search_results)
        
        logger.info(f"返回 {len(search_results)} 个搜索结果")
        return search_results
    
    def _merge_adjacent_parents(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并相邻的父文档"""
        # 按父文档索引排序
        results.sort(key=lambda x: x["metadata"]["parent_index"])
        
        merged_results = []
        current_group = []
        
        for result in results:
            if not current_group:
                current_group.append(result)
            else:
                # 检查是否为相邻的父文档
                last_parent_idx = current_group[-1]["metadata"]["parent_index"]
                current_parent_idx = result["metadata"]["parent_index"]
                
                if abs(current_parent_idx - last_parent_idx) <= 1:
                    current_group.append(result)
                else:
                    # 合并当前组并开始新组
                    merged_results.append(self._merge_group(current_group))
                    current_group = [result]
        
        # 处理最后一组
        if current_group:
            merged_results.append(self._merge_group(current_group))
        
        return merged_results
    
    def _merge_group(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并一组相关的搜索结果"""
        if len(group) == 1:
            return group[0]
        
        # 合并父文档内容
        parent_contents = []
        parent_ids = set()
        child_contents = []
        max_similarity = 0
        
        for result in group:
            if result["parent_content"] not in parent_contents:
                parent_contents.append(result["parent_content"])
            parent_ids.add(result["parent_id"])
            child_contents.append(result["child_content"])
            max_similarity = max(max_similarity, result["similarity_score"])
        
        merged_result = {
            "child_content": "\n\n".join(child_contents),
            "parent_content": "\n\n".join(parent_contents),
            "child_id": ",".join([r["child_id"] for r in group]),
            "parent_id": ",".join(parent_ids),
            "similarity_score": max_similarity,
            "metadata": {
                "merged": True,
                "group_size": len(group),
                "source": group[0]["metadata"]["source"]
            }
        }
        
        return merged_result
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory
        }
    
    def clear_collection(self) -> None:
        """清空集合"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "父子分段RAG集合"}
            )
            logger.info(f"已清空集合: {self.collection_name}")
        except Exception as e:
            logger.error(f"清空集合失败: {e}")

# ============================================================================
# 传统RAG系统（用于对比）
# ============================================================================

class TraditionalRAG:
    """传统RAG系统（用于对比）"""
    
    def __init__(self, 
                 collection_name: str = "traditional_rag",
                 persist_directory: str = "./chroma_traditional_db",
                 embedding_model: Optional[Embeddings] = None,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 初始化嵌入模型
        if embedding_model:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = self._setup_embedding_model()
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        
        # 初始化ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 创建嵌入函数
        self.embedding_function = self._create_embedding_function()
        
        # 创建或获取集合
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"加载现有传统RAG集合: {collection_name}")
        except:
            # 如果集合已存在但嵌入函数不匹配，先删除再创建
            try:
                self.client.delete_collection(collection_name)
                logger.info(f"删除现有传统RAG集合: {collection_name}")
            except:
                pass
            
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "传统RAG集合"}
            )
            logger.info(f"创建新传统RAG集合: {collection_name}")
    
    def _setup_embedding_model(self) -> Embeddings:
        """设置嵌入模型"""
        # 优先尝试DashScope
        dashscope_api_key = os.getenv('EMBEDDING_API_KEY')
        if dashscope_api_key:
            try:
                logger.info("使用DashScope嵌入模型")
                return DashScopeEmbeddings(api_key=dashscope_api_key)
            except Exception as e:
                logger.warning(f"DashScope嵌入模型初始化失败: {e}")
        
        # 尝试OpenAI
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai_api_key:
            try:
                logger.info("使用OpenAI嵌入模型")
                return OpenAIEmbeddings(api_key=openai_api_key)
            except Exception as e:
                logger.warning(f"OpenAI嵌入模型初始化失败: {e}")
        
        # 抛出异常，不再使用模拟嵌入
        raise ValueError(f"无法设置嵌入模型: {e}")
    
    def _create_embedding_function(self):
        """创建ChromaDB嵌入函数"""
        class ChromaEmbeddingFunction:
            def __init__(self, embedding_model):
                self.embedding_model = embedding_model
            
            def __call__(self, input: List[str]) -> List[List[float]]:
                return self.embedding_model.embed_documents(input)
            
            def name(self) -> str:
                return "traditional_embedding_function"
        
        return ChromaEmbeddingFunction(self.embedding_model)
    
    def add_document(self, text: str, source: str = "") -> None:
        """添加文档到传统RAG系统"""
        logger.info(f"添加文档到传统RAG，长度: {len(text)} 字符")
        
        # 分割文档
        chunks = self.text_splitter.split_text(text)
        logger.info(f"生成 {len(chunks)} 个文档片段")
        
        # 准备数据
        ids = [f"chunk_{uuid.uuid4().hex[:8]}" for _ in chunks]
        metadatas = [{"source": source, "chunk_index": i, "chunk_length": len(chunk)} 
                    for i, chunk in enumerate(chunks)]
        
        # 添加到ChromaDB
        try:
            self.collection.add(
                ids=ids,
                documents=chunks,
                metadatas=metadatas
            )
            logger.info(f"成功添加 {len(chunks)} 个文档片段到传统RAG")
        except Exception as e:
            logger.error(f"添加文档到传统RAG失败: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相关文档"""
        logger.info(f"传统RAG搜索查询: {query}")
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        if not results['documents'][0]:
            logger.warning("传统RAG未找到相关文档")
            return []
        
        search_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            result = {
                "content": doc,
                "id": results['ids'][0][i],
                "similarity_score": 1 - distance,
                "metadata": metadata
            }
            search_results.append(result)
        
        logger.info(f"传统RAG返回 {len(search_results)} 个搜索结果")
        return search_results
    
    def clear_collection(self) -> None:
        """清空集合"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "传统RAG集合"}
            )
            logger.info(f"已清空传统RAG集合: {self.collection_name}")
        except Exception as e:
            logger.error(f"清空传统RAG集合失败: {e}")

# ============================================================================
# PDF文档处理
# ============================================================================

class PDFProcessor:
    """PDF文档处理器"""
    
    @staticmethod
    def load_pdf(pdf_path: str, max_pages: int = 50) -> str:
        """加载PDF文档"""
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        try:
            logger.info(f"加载PDF文件: {pdf_path}")
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                # 读取指定页数
                max_pages = min(max_pages, len(pdf_reader.pages))
                for page_num in range(max_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                logger.info(f"成功加载PDF，共 {max_pages} 页，{len(text)} 字符")
                return text
        
        except Exception as e:
            logger.error(f"加载PDF失败: {e}")
            raise
    
    @staticmethod
    def get_sample_civil_code_text() -> str:
        """获取民法典示例文本"""
        return """
中华人民共和国民法典

第一编 总则

第一章 基本规定

第一条 为了保护民事主体的合法权益，调整民事关系，维护社会和经济秩序，适应中国特色社会主义发展要求，弘扬社会主义核心价值观，根据宪法，制定本法。

第二条 民法调整平等主体的自然人、法人和非法人组织之间的人身关系和财产关系。

第三条 民事主体的人身权利、财产权利以及其他合法权益受法律保护，任何组织或者个人不得侵犯。

第二章 自然人

第一节 民事权利能力和民事行为能力

第十三条 自然人从出生时起到死亡时止，具有民事权利能力，依法享有民事权利，承担民事义务。

第十四条 自然人的民事权利能力一律平等。

第十五条 自然人的出生时间和死亡时间，以出生证明、死亡证明记载的时间为准；没有出生证明、死亡证明的，以户籍登记或者其他有效身份登记记载的时间为准。有其他证据足以推翻以上记载时间的，以该证据证明的时间为准。

第三章 法人

第一节 一般规定

第五十七条 法人是具有民事权利能力和民事行为能力，依法独立享有民事权利和承担民事义务的组织。

第五十八条 法人应当依法成立。法人应当有自己的名称、组织机构、住所、财产或者经费。法人成立的具体条件和程序，依照法律、行政法规的规定。设立法人，法律、行政法规规定须经有关机关批准的，依照其规定。

第四编 人格权

第九百九十条 人格权是民事主体享有的生命权、身体权、健康权、姓名权、名称权、肖像权、名誉权、荣誉权、隐私权等权利。除前款规定的人格权外，自然人享有基于人身自由、人格尊严产生的其他人格权益。

第五编 婚姻家庭

第一千零四十一条 婚姻家庭受国家保护。实行婚姻自由、一夫一妻、男女平等的婚姻制度。保护妇女、未成年人、老年人、残疾人的合法权益。

第六编 继承

第一千一百二十一条 继承从被继承人死亡时开始。相互有继承关系的数人在同一事件中死亡，难以确定死亡时间的，推定没有其他继承人的人先死亡。都有其他继承人，辈份不同的，推定长辈先死亡；辈份相同的，推定同时死亡，相互不发生继承。

第七编 侵权责任

第一千一百六十五条 行为人因过错侵害他人民事权益造成损害的，应当承担侵权责任。依照法律规定推定行为人有过错，其不能证明自己没有过错的，应当承担侵权责任。
"""

# ============================================================================
# 演示类
# ============================================================================

class ParentChildDemo:
    """父子分段RAG演示类"""
    
    def __init__(self):
        self.chunker_configs = {
            "小粒度": ChunkingConfig(parent_chunk_size=800, parent_overlap=80, 
                                  child_chunk_size=200, child_overlap=20),
            "中粒度": ChunkingConfig(parent_chunk_size=1500, parent_overlap=150, 
                                  child_chunk_size=300, child_overlap=30),
            "大粒度": ChunkingConfig(parent_chunk_size=2500, parent_overlap=250, 
                                  child_chunk_size=500, child_overlap=50)
        }
        
        self.test_queries = [
            "什么是民事权利能力？",
            "法人的设立条件是什么？",
            "合同的成立要件有哪些？",
            "物权的种类包括哪些？",
            "侵权责任的构成要件是什么？"
        ]
    
    def load_civil_code_document(self, pdf_path: str = None) -> str:
        """加载民法典PDF文档"""
        if pdf_path is None:
            # 使用项目根目录的绝对路径
            project_root = Path(__file__).parent.parent.parent
            pdf_path = project_root / "docs" / "中华人民共和国民法典.pdf"
        
        try:
            return PDFProcessor.load_pdf(pdf_path)
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"无法加载PDF文件: {e}，使用示例文本")
            return PDFProcessor.get_sample_civil_code_text()
    
    def run_chunking_strategy_comparison(self, text: str) -> Dict[str, Any]:
        """运行分段策略对比"""
        logger.info("开始分段策略对比")
        
        comparison_results = {}
        
        for strategy_name, config in self.chunker_configs.items():
            logger.info(f"测试分段策略: {strategy_name}")
            
            chunker = ParentChildChunker(config)
            chunks = chunker.chunk_document(text, source="民法典")
            
            # 统计信息
            parent_ids = set(chunk["parent_id"] for chunk in chunks)
            avg_parent_length = sum(chunk["metadata"]["parent_length"] for chunk in chunks) / len(chunks)
            avg_child_length = sum(chunk["metadata"]["child_length"] for chunk in chunks) / len(chunks)
            
            comparison_results[strategy_name] = {
                "config": config,
                "total_chunks": len(chunks),
                "parent_chunks": len(parent_ids),
                "child_chunks": len(chunks),
                "avg_parent_length": avg_parent_length,
                "avg_child_length": avg_child_length,
                "chunks": chunks[:3]  # 保存前3个样本
            }
        
        return comparison_results
    
    def run_retrieval_comparison(self, text: str) -> Dict[str, Any]:
        """运行检索效果对比"""
        logger.info("开始检索效果对比")
        
        # 使用中粒度配置
        config = self.chunker_configs["中粒度"]
        chunker = ParentChildChunker(config)
        chunks = chunker.chunk_document(text, source="民法典")
        
        # 初始化父子分段RAG
        parent_child_rag = ParentChildRAG(
            collection_name="demo_parent_child",
            persist_directory="./demo_parent_child_db"
        )
        parent_child_rag.clear_collection()
        parent_child_rag.add_documents(chunks)
        
        # 初始化传统RAG
        traditional_rag = TraditionalRAG(
            collection_name="demo_traditional",
            persist_directory="./demo_traditional_db",
            chunk_size=500,
            chunk_overlap=50
        )
        traditional_rag.clear_collection()
        traditional_rag.add_document(text, source="民法典")
        
        # 对比测试
        comparison_results = {
            "queries": [],
            "summary": {
                "parent_child_avg_score": 0,
                "traditional_avg_score": 0,
                "parent_child_avg_length": 0,
                "traditional_avg_length": 0
            }
        }
        
        total_pc_score = 0
        total_trad_score = 0
        total_pc_length = 0
        total_trad_length = 0
        
        for query in self.test_queries:
            logger.info(f"测试查询: {query}")
            
            # 父子分段RAG搜索
            pc_results = parent_child_rag.search(query, top_k=3, expand_parents=True)
            
            # 传统RAG搜索
            trad_results = traditional_rag.search(query, top_k=3)
            
            # 计算统计信息
            pc_avg_score = sum(r["similarity_score"] for r in pc_results) / len(pc_results) if pc_results else 0
            trad_avg_score = sum(r["similarity_score"] for r in trad_results) / len(trad_results) if trad_results else 0
            
            pc_avg_length = sum(len(r.get("parent_content", r.get("child_content", ""))) for r in pc_results) / len(pc_results) if pc_results else 0
            trad_avg_length = sum(len(r["content"]) for r in trad_results) / len(trad_results) if trad_results else 0
            
            total_pc_score += pc_avg_score
            total_trad_score += trad_avg_score
            total_pc_length += pc_avg_length
            total_trad_length += trad_avg_length
            
            query_result = {
                "query": query,
                "parent_child_results": pc_results,
                "traditional_results": trad_results,
                "parent_child_avg_score": pc_avg_score,
                "traditional_avg_score": trad_avg_score,
                "parent_child_avg_length": pc_avg_length,
                "traditional_avg_length": trad_avg_length
            }
            
            comparison_results["queries"].append(query_result)
        
        # 计算总体统计
        num_queries = len(self.test_queries)
        comparison_results["summary"] = {
            "parent_child_avg_score": total_pc_score / num_queries,
            "traditional_avg_score": total_trad_score / num_queries,
            "parent_child_avg_length": total_pc_length / num_queries,
            "traditional_avg_length": total_trad_length / num_queries
        }
        
        return comparison_results
    
    def print_chunking_comparison(self, results: Dict[str, Any]) -> None:
        """打印分段策略对比结果"""
        print("\n" + "="*80)
        print("分段策略对比结果")
        print("="*80)
        
        for strategy_name, result in results.items():
            config = result["config"]
            print(f"\n【{strategy_name}策略】")
            print(f"  父文档配置: {config.parent_chunk_size}字符, 重叠{config.parent_overlap}字符")
            print(f"  子文档配置: {config.child_chunk_size}字符, 重叠{config.child_overlap}字符")
            print(f"  生成父文档: {result['parent_chunks']}个")
            print(f"  生成子文档: {result['child_chunks']}个")
            print(f"  平均父文档长度: {result['avg_parent_length']:.0f}字符")
            print(f"  平均子文档长度: {result['avg_child_length']:.0f}字符")
            
            # 显示样本
            print(f"\n  样本子文档:")
            for i, chunk in enumerate(result['chunks'][:2]):
                print(f"    [{i+1}] {chunk['child_content'][:100]}...")
    
    def print_retrieval_comparison(self, results: Dict[str, Any]) -> None:
        """打印检索效果对比结果"""
        print("\n" + "="*80)
        print("检索效果对比结果")
        print("="*80)
        
        summary = results["summary"]
        print(f"\n【总体对比】")
        print(f"  父子分段RAG平均相似度: {summary['parent_child_avg_score']:.3f}")
        print(f"  传统RAG平均相似度: {summary['traditional_avg_score']:.3f}")
        print(f"  父子分段RAG平均内容长度: {summary['parent_child_avg_length']:.0f}字符")
        print(f"  传统RAG平均内容长度: {summary['traditional_avg_length']:.0f}字符")
        
        improvement = ((summary['parent_child_avg_score'] - summary['traditional_avg_score']) / 
                      summary['traditional_avg_score'] * 100) if summary['traditional_avg_score'] > 0 else 0
        print(f"  相似度提升: {improvement:+.1f}%")
        
        print(f"\n【详细查询结果】")
        for i, query_result in enumerate(results["queries"][:3]):
            print(f"\n查询 {i+1}: {query_result['query']}")
            print(f"  父子分段RAG相似度: {query_result['parent_child_avg_score']:.3f}")
            print(f"  传统RAG相似度: {query_result['traditional_avg_score']:.3f}")
            
            # 显示最佳结果
            if query_result['parent_child_results']:
                best_pc = query_result['parent_child_results'][0]
                print(f"  父子分段最佳匹配: {best_pc.get('child_content', '')[:150]}...")
                if 'parent_content' in best_pc:
                    print(f"  对应父文档: {best_pc['parent_content'][:150]}...")
            
            if query_result['traditional_results']:
                best_trad = query_result['traditional_results'][0]
                print(f"  传统RAG最佳匹配: {best_trad['content'][:150]}...")
    
    def run_comprehensive_demo(self) -> None:
        """运行完整的父子分段RAG演示"""
        print("\n" + "="*80)
        print("第18小节：父子分段RAG技术演示")
        print("="*80)
        
        # 加载民法典文档
        print("\n1. 加载民法典文档...")
        civil_code_text = self.load_civil_code_document()
        print(f"   文档长度: {len(civil_code_text)} 字符")
        
        # 分段策略对比
        print("\n2. 分段策略对比...")
        chunking_results = self.run_chunking_strategy_comparison(civil_code_text)
        self.print_chunking_comparison(chunking_results)
        
        # 检索效果对比
        print("\n3. 检索效果对比...")
        retrieval_results = self.run_retrieval_comparison(civil_code_text)
        self.print_retrieval_comparison(retrieval_results)
        
        # 性能总结
        print("\n" + "="*80)
        print("演示总结")
        print("="*80)
        print("\n父子分段RAG技术的优势:")
        print("1. 提高了召回的准确性和相关性")
        print("2. 保持了文档的完整性和上下文")
        print("3. 减少了文档碎片化问题")
        print("4. 提供了更丰富的背景信息")
        print("\n适用场景:")
        print("- 法律文档分析")
        print("- 技术文档问答")
        print("- 学术论文检索")
        print("- 长篇内容理解")

def main():
    """主函数"""
    try:
        # 创建演示实例
        demo = ParentChildDemo()
        
        # 运行完整演示
        demo.run_comprehensive_demo()
        
    except KeyboardInterrupt:
        logger.info("演示被用户中断")
    except Exception as e:
        logger.error(f"演示运行出错: {e}")
        raise

if __name__ == "__main__":
    main()