#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第16小节：向量模型与嵌入技术演示

本演示展示如何在LangChain中使用不同类型的向量模型进行文本嵌入，
包括OpenAI、HuggingFace和本地模型的集成与使用。

主要功能：
1. 多种向量模型的配置和初始化
2. 文本嵌入生成和比较
3. 向量相似度计算
4. 向量维度分析和可视化
5. 性能基准测试
"""

import os
import sys
import time
import logging
import requests
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import seaborn as sns

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.config import load_embedding_config

# 加载环境变量
load_dotenv()

class DashScopeEmbeddings(Embeddings):
    """阿里云DashScope嵌入模型"""
    
    def __init__(self, api_key: str, base_url: str, model: str = "text-embedding-v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文档"""
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        url = f"{self.base_url}/embeddings"
        
        # DashScope API的正确参数格式
        payload = {
            "model": self.model,
            "input": text,  # 直接使用字符串，不是contents
            "encoding_format": "float"
        }
        
        try:
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            
            result = response.json()
            if "data" in result and len(result["data"]) > 0:
                return result["data"][0]["embedding"]
            else:
                raise ValueError(f"Invalid response format: {result}")
                
        except Exception as e:
            raise ValueError(f"DashScope API error: {e}")


@dataclass
class EmbeddingResult:
    """嵌入结果数据类"""
    model_name: str
    text: str
    embedding: List[float]
    dimension: int
    generation_time: float

class VectorModelsDemo:
    """向量模型演示类"""
    
    def __init__(self):
        """初始化演示类"""
        self.embedding_models = {}
        self.civil_code_documents = []  # 存储民法典文档片段
        self.test_texts = []  # 将从民法典中提取
        
        # 加载民法典PDF文档
        self.load_civil_code_pdf()
        
    def setup_embedding_models(self) -> Dict[str, Any]:
        """设置和初始化不同的向量模型"""
        print("🔧 正在初始化向量模型...")
        
        models = {}
        
        # 获取向量模型配置
        try:
            embedding_config = load_embedding_config()
            print(f"📋 向量模型配置加载成功")
            print(f"   🔑 API密钥: {'已配置' if embedding_config.get('api_key') else '未配置'}")
            print(f"   🌐 基础URL: {embedding_config.get('base_url', '未配置')}")
            print(f"   🤖 模型名称: {embedding_config.get('model_name', '未配置')}")
        except Exception as e:
            print(f"❌ 向量模型配置加载失败: {e}")
            embedding_config = {'api_key': None, 'base_url': None, 'model_name': None}
        
        # 1. 优先使用配置的向量模型
        if embedding_config.get('api_key') and embedding_config.get('base_url'):
            try:
                model_name = embedding_config.get('model_name', 'text-embedding-v1')
                print(f"🔧 配置信息:")
                print(f"   API Key: {embedding_config['api_key'][:10]}...")
                print(f"   Base URL: {embedding_config['base_url']}")
                print(f"   Model: {model_name}")
                
                # 检测是否为DashScope API
                if 'dashscope.aliyuncs.com' in embedding_config['base_url']:
                    print(f"🎯 检测到DashScope API，使用专用嵌入类")
                    models['configured_embedding'] = DashScopeEmbeddings(
                        api_key=embedding_config['api_key'],
                        base_url=embedding_config['base_url'],
                        model=model_name
                    )
                else:
                    print(f"🎯 使用OpenAI兼容API")
                    models['configured_embedding'] = OpenAIEmbeddings(
                        model=model_name,
                        api_key=embedding_config['api_key'],
                        base_url=embedding_config['base_url']
                    )
                
                # 测试模型是否可用
                test_result = models['configured_embedding'].embed_query("测试文本")
                if test_result and len(test_result) > 0:
                    print(f"✅ 配置的向量模型 ({model_name}) 初始化成功")
                    print(f"   📏 向量维度: {len(test_result)}")
                    print(f"🎯 使用配置的向量模型，跳过其他备用模型")
                    
                    # 如果配置的模型成功初始化，直接返回，不再初始化其他模型
                    self.embedding_models = models
                    print(f"\n📊 使用配置的向量模型: {model_name}\n")
                    return models
                else:
                    print(f"❌ 配置的向量模型测试失败")
            except Exception as e:
                print(f"❌ 配置的向量模型初始化失败: {e}")
        else:
            print("⚠️  向量模型配置不完整，使用备用模型")
            if not embedding_config.get('api_key'):
                print("   ❌ EMBEDDING_API_KEY 未配置")
            if not embedding_config.get('base_url'):
                print("   ❌ EMBEDDING_BASE_URL 未配置")
        
        # 2. 备用模型：HuggingFace本地模型（优先，无需API密钥）
        try:
            models['huggingface_local'] = HuggingFaceEmbeddings(
                model_name=os.getenv('LOCAL_EMBEDDING_MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2'),
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✅ HuggingFace本地模型 (all-MiniLM-L6-v2) 初始化成功")
        except Exception as e:
            print(f"❌ HuggingFace本地模型初始化失败: {e}")
        
        # 3. 备用模型：HuggingFace推理API
        try:
            if os.getenv('HUGGINGFACE_API_TOKEN'):
                models['huggingface_api'] = HuggingFaceInferenceAPIEmbeddings(
                    api_key=os.getenv('HUGGINGFACE_API_TOKEN'),
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                print("✅ HuggingFace推理API 初始化成功")
            else:
                print("⚠️  HuggingFace API Token未配置，跳过API模型")
        except Exception as e:
            print(f"❌ HuggingFace推理API初始化失败: {e}")
        
        # 4. 备用模型：OpenAI Embeddings（仅在有API密钥时）
        try:
            if os.getenv('OPENAI_API_KEY') and os.getenv('OPENAI_API_KEY') != 'your_openai_api_key_here':
                models['openai_ada'] = OpenAIEmbeddings(
                    model="text-embedding-ada-002",
                    openai_api_key=os.getenv('OPENAI_API_KEY'),
                    openai_api_base=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
                )
                print("✅ OpenAI Embeddings (ada-002) 初始化成功")
            else:
                print("⚠️  OpenAI API密钥未正确配置，跳过OpenAI模型")
        except Exception as e:
            print(f"❌ OpenAI Embeddings 初始化失败: {e}")
        
        self.embedding_models = models
        print(f"\n📊 总共初始化了 {len(models)} 个向量模型\n")
        
        if not models:
            print("❌ 没有可用的向量模型！请检查配置：")
            print("   1. 配置 EMBEDDING_API_KEY 和 EMBEDDING_BASE_URL")
            print("   2. 或确保 HuggingFace 模型可以正常下载")
            print("   3. 或配置 HUGGINGFACE_API_TOKEN")
        
        return models
    
    def load_civil_code_pdf(self):
        """加载民法典PDF文档并进行切分"""
        print("📚 正在加载民法典PDF文档...")
        
        # 民法典PDF文件路径
        pdf_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
            'docs', 
            '中华人民共和国民法典.pdf'
        )
        
        if not os.path.exists(pdf_path):
            print(f"❌ 民法典PDF文件未找到: {pdf_path}")
            print("💡 使用默认测试文本")
            self.test_texts = [
                "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
                "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。",
                "深度学习是机器学习的一个分支，使用神经网络来模拟人脑的学习过程。",
                "自然语言处理是人工智能的一个领域，专注于计算机与人类语言之间的交互。",
                "今天天气很好，阳光明媚，适合户外活动。",
                "我喜欢在周末看电影，特别是科幻和动作片。",
                "Python是一种高级编程语言，广泛用于数据科学和机器学习。",
                "向量数据库是专门用于存储和检索高维向量数据的数据库系统。"
            ]
            return
        
        try:
            # 使用PyPDFLoader加载PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            print(f"✅ 成功加载PDF文档，共 {len(documents)} 页")
            
            # 合并所有页面的文本
            full_text = "\n".join([doc.page_content for doc in documents])
            print(f"📄 文档总字符数: {len(full_text)}")
            
            # 使用文本切分器进行切分
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # 每个片段500字符
                chunk_overlap=50,  # 片段间重叠50字符
                length_function=len,
                separators=["\n\n", "\n", "。", "；", "，", " ", ""]
            )
            
            # 切分文档
            split_docs = text_splitter.split_documents(documents)
            
            # 过滤掉过短的文档片段
            filtered_docs = [doc for doc in split_docs if len(doc.page_content.strip()) > 50]
            
            print(f"📝 文档切分完成，共 {len(filtered_docs)} 个片段")
            
            # 存储文档片段
            self.civil_code_documents = filtered_docs
            
            # 选择前10个片段作为测试文本
            self.test_texts = [doc.page_content.strip() for doc in filtered_docs[:10]]
            
            print("📋 民法典文档片段示例:")
            for i, text in enumerate(self.test_texts[:3], 1):
                preview = text[:100] + "..." if len(text) > 100 else text
                print(f"  {i}. {preview}")
            
            print(f"\n🎯 将使用 {len(self.test_texts)} 个民法典文档片段进行向量化演示\n")
            
        except Exception as e:
            print(f"❌ 加载民法典PDF时出错: {e}")
            print("💡 使用默认测试文本")
            self.test_texts = [
                "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
                "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。",
                "深度学习是机器学习的一个分支，使用神经网络来模拟人脑的学习过程。",
                "自然语言处理是人工智能的一个领域，专注于计算机与人类语言之间的交互。"
            ]
    
    def generate_embeddings(self, text: str, model_name: str) -> Optional[EmbeddingResult]:
        """生成文本嵌入"""
        if model_name not in self.embedding_models:
            print(f"❌ 模型 {model_name} 未找到")
            return None
        
        try:
            start_time = time.time()
            embedding = self.embedding_models[model_name].embed_query(text)
            generation_time = time.time() - start_time
            
            return EmbeddingResult(
                model_name=model_name,
                text=text,
                embedding=embedding,
                dimension=len(embedding),
                generation_time=generation_time
            )
        except Exception as e:
            print(f"❌ 生成嵌入失败 ({model_name}): {e}")
            return None
    
    def demo_basic_embeddings(self):
        """演示基本的文本嵌入功能"""
        print("=" * 60)
        print("📝 基本文本嵌入演示")
        print("=" * 60)
        
        if self.test_texts:
            sample_text = self.test_texts[0]  # 使用第一个民法典片段
            print(f"测试文本（民法典片段）: {sample_text[:100]}...\n")
        else:
            sample_text = "人工智能正在改变我们的世界，为各行各业带来创新和效率提升。"
            print(f"测试文本: {sample_text}\n")
        
        results = []
        for model_name in self.embedding_models.keys():
            print(f"🔄 使用 {model_name} 生成嵌入...")
            result = self.generate_embeddings(sample_text, model_name)
            
            if result:
                results.append(result)
                print(f"   ✅ 维度: {result.dimension}")
                print(f"   ⏱️  生成时间: {result.generation_time:.3f}秒")
                print(f"   📊 向量前5个值: {result.embedding[:5]}")
                print()
            else:
                print(f"   ❌ 生成失败\n")
        
        return results
    
    def demo_similarity_calculation(self):
        """演示向量相似度计算"""
        print("=" * 60)
        print("🔍 向量相似度计算演示")
        print("=" * 60)
        
        # 如果有民法典文档，使用民法典片段进行相似度计算
        if len(self.test_texts) >= 3:
            text_pairs = [
                (self.test_texts[0], self.test_texts[1]),  # 相邻的民法典片段
                (self.test_texts[0], self.test_texts[2]),  # 间隔的民法典片段
                (self.test_texts[1], self.test_texts[2])   # 另一对民法典片段
            ]
            print("📚 使用民法典文档片段进行相似度分析\n")
        else:
            # 备用文本对
            text_pairs = [
                ("人工智能是计算机科学的一个分支", "机器学习是人工智能的一个子集"),
                ("今天天气很好，阳光明媚", "我喜欢在周末看电影"),
                ("Python是一种编程语言", "向量数据库用于存储高维数据")
            ]
        
        for model_name in self.embedding_models.keys():
            print(f"\n🤖 使用模型: {model_name}")
            print("-" * 40)
            
            for i, (text1, text2) in enumerate(text_pairs, 1):
                result1 = self.generate_embeddings(text1, model_name)
                result2 = self.generate_embeddings(text2, model_name)
                
                if result1 and result2:
                    # 计算余弦相似度
                    similarity = cosine_similarity(
                        [result1.embedding], 
                        [result2.embedding]
                    )[0][0]
                    
                    print(f"文本对 {i}:")
                    print(f"  📄 文本1: {text1[:50]}...")
                    print(f"  📄 文本2: {text2[:50]}...")
                    print(f"  🎯 相似度: {similarity:.4f}")
                    print()
    
    def demo_batch_processing(self):
        """演示批量处理和性能对比"""
        print("=" * 60)
        print("⚡ 批量处理性能对比")
        print("=" * 60)
        
        performance_results = {}
        
        for model_name in self.embedding_models.keys():
            print(f"\n🔄 测试模型: {model_name}")
            
            start_time = time.time()
            successful_embeddings = 0
            total_dimensions = 0
            
            for text in self.test_texts:
                result = self.generate_embeddings(text, model_name)
                if result:
                    successful_embeddings += 1
                    total_dimensions = result.dimension
            
            total_time = time.time() - start_time
            avg_time = total_time / len(self.test_texts) if self.test_texts else 0
            
            performance_results[model_name] = {
                'total_time': total_time,
                'avg_time': avg_time,
                'successful_count': successful_embeddings,
                'dimension': total_dimensions
            }
            
            print(f"  📊 成功处理: {successful_embeddings}/{len(self.test_texts)}")
            print(f"  ⏱️  总时间: {total_time:.3f}秒")
            print(f"  📈 平均时间: {avg_time:.3f}秒/文本")
            print(f"  📏 向量维度: {total_dimensions}")
        
        return performance_results
    
    def demo_vector_analysis(self):
        """演示向量分析和可视化"""
        print("=" * 60)
        print("📊 向量分析和可视化")
        print("=" * 60)
        
        # 收集所有模型的嵌入结果
        all_embeddings = {}
        
        for model_name in self.embedding_models.keys():
            embeddings = []
            labels = []
            
            for i, text in enumerate(self.test_texts[:6]):  # 限制数量以便可视化
                result = self.generate_embeddings(text, model_name)
                if result:
                    embeddings.append(result.embedding)
                    labels.append(f"文本{i+1}")
            
            if embeddings:
                all_embeddings[model_name] = {
                    'embeddings': np.array(embeddings),
                    'labels': labels
                }
        
        # 创建可视化
        self.create_embedding_visualization(all_embeddings)
        
        return all_embeddings
    
    def create_embedding_visualization(self, all_embeddings: Dict[str, Dict]):
        """创建嵌入向量的可视化图表"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('向量模型分析可视化', fontsize=16, fontweight='bold')
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            model_names = list(all_embeddings.keys())
            
            # 1. 向量维度对比
            if model_names:
                dimensions = [all_embeddings[name]['embeddings'].shape[1] for name in model_names]
                axes[0, 0].bar(model_names, dimensions, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                axes[0, 0].set_title('向量维度对比')
                axes[0, 0].set_ylabel('维度')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. PCA降维可视化（如果有足够的数据）
            if model_names and len(all_embeddings[model_names[0]]['embeddings']) > 2:
                model_name = model_names[0]  # 使用第一个可用模型
                embeddings = all_embeddings[model_name]['embeddings']
                labels = all_embeddings[model_name]['labels']
                
                if embeddings.shape[1] > 2:  # 只有当维度大于2时才进行PCA
                    pca = PCA(n_components=2)
                    embeddings_2d = pca.fit_transform(embeddings)
                    
                    scatter = axes[0, 1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                               c=range(len(labels)), cmap='viridis', s=100)
                    axes[0, 1].set_title(f'PCA降维可视化 ({model_name})')
                    axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                    axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                    
                    # 添加标签
                    for i, label in enumerate(labels):
                        axes[0, 1].annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                                          xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # 3. 相似度热力图
            if model_names:
                model_name = model_names[0]
                embeddings = all_embeddings[model_name]['embeddings']
                labels = all_embeddings[model_name]['labels']
                
                similarity_matrix = cosine_similarity(embeddings)
                
                im = axes[1, 0].imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                axes[1, 0].set_title(f'相似度热力图 ({model_name})')
                axes[1, 0].set_xticks(range(len(labels)))
                axes[1, 0].set_yticks(range(len(labels)))
                axes[1, 0].set_xticklabels(labels, rotation=45, ha='right')
                axes[1, 0].set_yticklabels(labels)
                
                # 添加颜色条
                cbar = plt.colorbar(im, ax=axes[1, 0])
                cbar.set_label('余弦相似度')
            
            # 4. 向量分布统计
            if model_names:
                model_name = model_names[0]
                embeddings = all_embeddings[model_name]['embeddings']
                
                # 计算每个维度的统计信息
                mean_values = np.mean(embeddings, axis=0)
                std_values = np.std(embeddings, axis=0)
                
                axes[1, 1].plot(mean_values[:50], label='均值', alpha=0.7)  # 只显示前50个维度
                axes[1, 1].fill_between(range(50), 
                                       (mean_values - std_values)[:50], 
                                       (mean_values + std_values)[:50], 
                                       alpha=0.3, label='±1标准差')
                axes[1, 1].set_title(f'向量分布统计 ({model_name})')
                axes[1, 1].set_xlabel('维度索引')
                axes[1, 1].set_ylabel('值')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            output_path = os.path.join(os.path.dirname(__file__), 'vector_analysis.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n📊 可视化图表已保存到: {output_path}")
            
            # 显示图表（如果在支持的环境中）
            try:
                plt.show()
            except:
                print("💡 提示: 在支持GUI的环境中运行可以直接显示图表")
                
        except Exception as e:
            print(f"❌ 创建可视化时出错: {e}")
            print("💡 可能需要安装matplotlib和seaborn: pip install matplotlib seaborn")
    
    def demo_embedding_quality_assessment(self):
        """演示嵌入质量评估"""
        print("=" * 60)
        print("🎯 嵌入质量评估")
        print("=" * 60)
        
        # 定义语义相似的文本组
        semantic_groups = [
            {
                'topic': 'AI/ML',
                'texts': [
                    "人工智能是计算机科学的一个分支",
                    "机器学习是人工智能的一个子集",
                    "深度学习使用神经网络模拟人脑"
                ]
            },
            {
                'topic': '日常生活',
                'texts': [
                    "今天天气很好，阳光明媚",
                    "我喜欢在周末看电影",
                    "晚餐时间到了，该吃饭了"
                ]
            },
            {
                'topic': '编程技术',
                'texts': [
                    "Python是一种高级编程语言",
                    "向量数据库用于存储高维数据",
                    "API接口设计需要考虑用户体验"
                ]
            }
        ]
        
        for model_name in self.embedding_models.keys():
            print(f"\n🤖 评估模型: {model_name}")
            print("-" * 40)
            
            group_coherence_scores = []
            
            for group in semantic_groups:
                # 生成组内所有文本的嵌入
                embeddings = []
                for text in group['texts']:
                    result = self.generate_embeddings(text, model_name)
                    if result:
                        embeddings.append(result.embedding)
                
                if len(embeddings) >= 2:
                    # 计算组内平均相似度
                    similarities = []
                    for i in range(len(embeddings)):
                        for j in range(i + 1, len(embeddings)):
                            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                            similarities.append(sim)
                    
                    avg_similarity = np.mean(similarities)
                    group_coherence_scores.append(avg_similarity)
                    
                    print(f"  📂 {group['topic']} 组内平均相似度: {avg_similarity:.4f}")
            
            if group_coherence_scores:
                overall_coherence = np.mean(group_coherence_scores)
                print(f"  🎯 整体语义一致性得分: {overall_coherence:.4f}")
    
    def demo_civil_code_vectorization(self):
        """专门演示民法典文档的向量化"""
        print("=" * 60)
        print("⚖️  民法典文档向量化专项演示")
        print("=" * 60)
        
        if not self.test_texts:
            print("❌ 未找到民法典文档，跳过此演示")
            return
        
        print(f"📚 已加载民法典文档，共 {len(self.test_texts)} 个片段")
        print(f"📄 文档片段示例: {self.test_texts[0][:100]}...\n")
        
        # 分析不同类型的法律条文
        legal_categories = {
            "总则条文": [text for text in self.test_texts if "总则" in text or "第一编" in text],
            "物权条文": [text for text in self.test_texts if "物权" in text or "第二编" in text],
            "合同条文": [text for text in self.test_texts if "合同" in text or "第三编" in text],
            "人格权条文": [text for text in self.test_texts if "人格权" in text or "第四编" in text],
            "婚姻家庭条文": [text for text in self.test_texts if "婚姻" in text or "家庭" in text or "第五编" in text],
            "继承条文": [text for text in self.test_texts if "继承" in text or "第六编" in text],
            "侵权责任条文": [text for text in self.test_texts if "侵权" in text or "责任" in text or "第七编" in text]
        }
        
        print("📊 法律条文分类统计:")
        for category, texts in legal_categories.items():
            if texts:
                print(f"   {category}: {len(texts)} 个片段")
        print()
        
        # 对不同类型条文进行向量化并分析相似度
        for model_name in self.embedding_models.keys():
            print(f"🔄 使用 {model_name} 分析法律条文语义相似性...")
            
            # 选择几个代表性片段进行分析
            sample_texts = self.test_texts[:3]
            embeddings = []
            
            for i, text in enumerate(sample_texts):
                result = self.generate_embeddings(text, model_name)
                if result:
                    embeddings.append(result.embedding)
                    print(f"   ✅ 片段 {i+1} 向量化成功 (维度: {result.dimension})")
                else:
                    print(f"   ❌ 片段 {i+1} 向量化失败")
            
            # 计算法律条文间的语义相似度
            if len(embeddings) >= 2:
                from sklearn.metrics.pairwise import cosine_similarity
                similarity_matrix = cosine_similarity(embeddings)
                
                print(f"   📈 法律条文语义相似度矩阵:")
                for i in range(len(similarity_matrix)):
                    for j in range(len(similarity_matrix[i])):
                        if i != j:
                            print(f"      片段{i+1} vs 片段{j+1}: {similarity_matrix[i][j]:.4f}")
            print()
    
    def run_comprehensive_demo(self):
        """运行完整的向量模型演示"""
        print("🚀 向量模型与嵌入技术综合演示")
        print("=" * 80)
        
        # 1. 初始化模型
        models = self.setup_embedding_models()
        
        if not models:
            print("❌ 没有可用的向量模型，请检查配置")
            return
        
        # 2. 基本嵌入演示
        self.demo_basic_embeddings()
        
        # 3. 民法典专项向量化演示
        self.demo_civil_code_vectorization()
        
        # 4. 相似度计算演示
        self.demo_similarity_calculation()
        
        # 5. 批量处理性能测试
        performance_results = self.demo_batch_processing()
        
        # 6. 向量分析和可视化
        self.demo_vector_analysis()
        
        # 7. 嵌入质量评估
        self.demo_embedding_quality_assessment()
        
        # 8. 总结报告
        self.print_summary_report(performance_results)
    
    def print_summary_report(self, performance_results: Dict[str, Dict]):
        """打印总结报告"""
        print("\n" + "=" * 60)
        print("📋 演示总结报告")
        print("=" * 60)
        
        print(f"✅ 成功初始化 {len(self.embedding_models)} 个向量模型")
        print(f"📝 测试了 {len(self.test_texts)} 个样本文本")
        
        if performance_results:
            print("\n🏆 性能排行榜:")
            sorted_models = sorted(performance_results.items(), 
                                 key=lambda x: x[1]['avg_time'])
            
            for i, (model_name, stats) in enumerate(sorted_models, 1):
                print(f"  {i}. {model_name}:")
                print(f"     ⏱️  平均处理时间: {stats['avg_time']:.3f}秒")
                print(f"     📏 向量维度: {stats['dimension']}")
                print(f"     ✅ 成功率: {stats['successful_count']}/{len(self.test_texts)}")
        
        print("\n💡 使用建议:")
        print("  • 对于生产环境，建议使用OpenAI或商业API以获得最佳性能")
        print("  • 对于隐私敏感场景，推荐使用本地HuggingFace模型")
        print("  • 根据应用需求选择合适的向量维度")
        print("  • 定期评估和优化嵌入质量")
        
        print("\n🎉 演示完成！")

def main():
    """主函数"""
    try:
        demo = VectorModelsDemo()
        demo.run_comprehensive_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️  演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()