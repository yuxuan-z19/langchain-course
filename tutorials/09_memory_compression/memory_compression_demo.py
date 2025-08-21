#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain 记忆压缩演示

本脚本演示了如何实现智能记忆压缩功能，包括：
1. MemoryCompressor 核心类
2. 多种压缩策略（摘要、关键词、重要性评分、渐进式）
3. 阈值监控和自动压缩
4. 存储后端集成（InMemory和Redis）
5. 压缩效果评估和优化
"""

import os
import sys
import json
import time
import re
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import tiktoken

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# 项目配置
from utils.config import load_deepseek_config

# Redis支持（可选）
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("警告: Redis未安装，将使用内存存储")


@dataclass
class CompressionConfig:
    """压缩配置类"""
    max_messages: int = 50          # 最大消息数量
    max_tokens: int = 4000          # 最大token数量
    compression_ratio: float = 0.3   # 压缩比例
    min_importance: float = 0.5      # 最小重要性阈值
    preserve_recent: int = 10        # 保留最近消息数量
    summary_length: int = 200        # 摘要长度
    enable_progressive: bool = True   # 启用渐进式压缩


@dataclass
class MessageMetadata:
    """消息元数据"""
    timestamp: datetime
    importance_score: float = 0.0
    keywords: List[str] = None
    compressed: bool = False
    compression_method: str = None

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []


class CompressionStrategy(ABC):
    """压缩策略抽象基类"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    @abstractmethod
    def compress(self, messages: List[BaseMessage], config: CompressionConfig) -> List[BaseMessage]:
        """执行压缩"""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """获取策略名称"""
        pass


class SummaryCompressor(CompressionStrategy):
    """摘要压缩器"""
    
    def get_strategy_name(self) -> str:
        return "summary"
    
    def compress(self, messages: List[BaseMessage], config: CompressionConfig) -> List[BaseMessage]:
        """使用LLM生成对话摘要"""
        if len(messages) <= config.preserve_recent:
            return messages
        
        # 分离要压缩的消息和要保留的消息
        to_compress = messages[:-config.preserve_recent]
        to_preserve = messages[-config.preserve_recent:]
        
        # 构建摘要提示
        conversation_text = self._format_messages_for_summary(to_compress)
        
        summary_prompt = PromptTemplate(
            input_variables=["conversation", "max_length"],
            template="""
请对以下对话进行智能摘要，保留关键信息和重要决策：

对话内容：
{conversation}

要求：
1. 摘要长度不超过{max_length}字
2. 保留重要的事实信息和决策
3. 保持对话的逻辑脉络
4. 突出用户的核心需求和问题

摘要：
"""
        )
        
        try:
            # 生成摘要
            chain = summary_prompt | self.llm
            summary_response = chain.invoke({
                "conversation": conversation_text,
                "max_length": config.summary_length
            })
            
            summary_text = summary_response.content
            
            # 创建摘要消息
            summary_message = AIMessage(
                content=f"[对话摘要] {summary_text}",
                additional_kwargs={"compressed": True, "method": "summary"}
            )
            
            return [summary_message] + to_preserve
            
        except Exception as e:
            print(f"摘要压缩失败: {e}")
            return messages
    
    def _format_messages_for_summary(self, messages: List[BaseMessage]) -> str:
        """格式化消息用于摘要"""
        formatted = []
        for msg in messages:
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            formatted.append(f"{role}: {msg.content}")
        return "\n".join(formatted)


class KeywordCompressor(CompressionStrategy):
    """关键词压缩器"""
    
    def get_strategy_name(self) -> str:
        return "keyword"
    
    def compress(self, messages: List[BaseMessage], config: CompressionConfig) -> List[BaseMessage]:
        """提取关键词并压缩"""
        if len(messages) <= config.preserve_recent:
            return messages
        
        to_compress = messages[:-config.preserve_recent]
        to_preserve = messages[-config.preserve_recent:]
        
        # 提取关键词
        keywords = self._extract_keywords(to_compress)
        
        # 创建关键词消息
        keyword_message = AIMessage(
            content=f"[关键信息] {', '.join(keywords)}",
            additional_kwargs={"compressed": True, "method": "keyword"}
        )
        
        return [keyword_message] + to_preserve
    
    def _extract_keywords(self, messages: List[BaseMessage]) -> List[str]:
        """提取关键词"""
        conversation_text = " ".join([msg.content for msg in messages])
        
        keyword_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
从以下对话中提取最重要的关键词和概念，用逗号分隔：

对话内容：
{text}

要求：
1. 提取10-20个最重要的关键词
2. 包括人名、地名、专业术语、重要概念
3. 只返回关键词，用逗号分隔

关键词：
"""
        )
        
        try:
            chain = keyword_prompt | self.llm
            response = chain.invoke({"text": conversation_text})
            keywords = [kw.strip() for kw in response.content.split(",")]
            return keywords[:20]  # 限制数量
        except Exception as e:
            print(f"关键词提取失败: {e}")
            return ["对话记录"]


class ImportanceCompressor(CompressionStrategy):
    """重要性评分压缩器"""
    
    def get_strategy_name(self) -> str:
        return "importance"
    
    def compress(self, messages: List[BaseMessage], config: CompressionConfig) -> List[BaseMessage]:
        """基于重要性评分进行压缩"""
        if len(messages) <= config.preserve_recent:
            return messages
        
        # 为消息评分
        scored_messages = self._score_messages(messages)
        
        # 保留最近的消息
        recent_messages = messages[-config.preserve_recent:]
        
        # 从历史消息中选择重要的
        historical_messages = messages[:-config.preserve_recent]
        important_messages = []
        
        for i, msg in enumerate(historical_messages):
            score = scored_messages.get(i, 0.0)
            if score >= config.min_importance:
                important_messages.append(msg)
        
        return important_messages + recent_messages
    
    def _score_messages(self, messages: List[BaseMessage]) -> Dict[int, float]:
        """为消息评分"""
        scores = {}
        
        scoring_prompt = PromptTemplate(
            input_variables=["message"],
            template="""
请为以下消息的重要性打分（0-1之间的小数）：

消息：{message}

评分标准：
- 0.9-1.0: 极其重要（关键决策、重要信息）
- 0.7-0.8: 很重要（有价值的内容）
- 0.5-0.6: 一般重要（普通对话）
- 0.3-0.4: 不太重要（闲聊）
- 0.0-0.2: 不重要（无意义内容）

只返回数字分数：
"""
        )
        
        for i, msg in enumerate(messages):
            try:
                chain = scoring_prompt | self.llm
                response = chain.invoke({"message": msg.content})
                score_text = response.content.strip()
                score = float(re.findall(r'\d+\.\d+|\d+', score_text)[0])
                scores[i] = min(max(score, 0.0), 1.0)  # 确保在0-1范围内
            except Exception as e:
                print(f"评分失败: {e}")
                scores[i] = 0.5  # 默认分数
        
        return scores


class ProgressiveCompressor(CompressionStrategy):
    """渐进式压缩器"""
    
    def get_strategy_name(self) -> str:
        return "progressive"
    
    def compress(self, messages: List[BaseMessage], config: CompressionConfig) -> List[BaseMessage]:
        """渐进式压缩：近期详细，远期摘要"""
        if len(messages) <= config.preserve_recent:
            return messages
        
        # 分层处理
        recent = messages[-config.preserve_recent:]  # 最近的，保持原样
        middle = messages[-config.preserve_recent*2:-config.preserve_recent]  # 中期的，轻度压缩
        old = messages[:-config.preserve_recent*2]  # 早期的，重度压缩
        
        compressed_messages = []
        
        # 处理早期消息（重度压缩）
        if old:
            summary_compressor = SummaryCompressor(self.llm)
            old_compressed = summary_compressor.compress(old, config)
            compressed_messages.extend(old_compressed)
        
        # 处理中期消息（轻度压缩）
        if middle:
            # 每3条消息压缩为1条
            for i in range(0, len(middle), 3):
                chunk = middle[i:i+3]
                if len(chunk) >= 2:
                    summary = self._create_mini_summary(chunk)
                    compressed_messages.append(summary)
                else:
                    compressed_messages.extend(chunk)
        
        # 添加最近消息
        compressed_messages.extend(recent)
        
        return compressed_messages
    
    def _create_mini_summary(self, messages: List[BaseMessage]) -> AIMessage:
        """创建小段摘要"""
        content_parts = []
        for msg in messages:
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            content_parts.append(f"{role}: {msg.content[:100]}...")
        
        summary_content = f"[压缩记录] {' | '.join(content_parts)}"
        return AIMessage(
            content=summary_content,
            additional_kwargs={"compressed": True, "method": "progressive"}
        )


class MemoryCompressor:
    """记忆压缩器核心类"""
    
    def __init__(self, 
                 config: CompressionConfig = None,
                 storage_backend: str = "memory",
                 redis_url: str = "redis://localhost:6379/0"):
        
        self.config = config or CompressionConfig()
        self.storage_backend = storage_backend
        self.redis_url = redis_url
        
        # 初始化LLM
        self.llm = self._create_llm()
        
        # 初始化压缩策略
        self.strategies = {
            "summary": SummaryCompressor(self.llm),
            "keyword": KeywordCompressor(self.llm),
            "importance": ImportanceCompressor(self.llm),
            "progressive": ProgressiveCompressor(self.llm)
        }
        
        # 初始化存储
        self.messages = []
        self.compression_history = []
        self._init_storage()
        
        # Token计数器
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def _create_llm(self) -> ChatOpenAI:
        """创建LLM实例"""
        config = load_deepseek_config()
        
        return ChatOpenAI(
            model="deepseek-chat",
            api_key=config['api_key'],
            base_url=config['base_url'],
            temperature=0.1
        )
    
    def _init_storage(self):
        """初始化存储后端"""
        if self.storage_backend == "redis" and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(self.redis_url)
                self.redis_client.ping()
                print("Redis连接成功")
            except Exception as e:
                print(f"Redis连接失败，使用内存存储: {e}")
                self.storage_backend = "memory"
        else:
            self.storage_backend = "memory"
    
    def add_message(self, role: str, content: str) -> None:
        """添加消息"""
        if role.lower() == "user":
            message = HumanMessage(content=content)
        else:
            message = AIMessage(content=content)
        
        self.messages.append(message)
        
        # 检查是否需要压缩
        if self.should_compress():
            print("\n🔄 达到压缩阈值，开始自动压缩...")
            self.compress()
        
        # 保存到存储
        self._save_to_storage()
    
    def should_compress(self) -> bool:
        """检查是否需要压缩"""
        # 检查消息数量
        if len(self.messages) > self.config.max_messages:
            return True
        
        # 检查token数量
        total_tokens = self._count_tokens()
        if total_tokens > self.config.max_tokens:
            return True
        
        return False
    
    def _count_tokens(self) -> int:
        """计算总token数"""
        total = 0
        for msg in self.messages:
            total += len(self.tokenizer.encode(msg.content))
        return total
    
    def compress(self, strategy: str = "progressive") -> List[BaseMessage]:
        """执行压缩"""
        if strategy not in self.strategies:
            raise ValueError(f"未知的压缩策略: {strategy}")
        
        original_count = len(self.messages)
        original_tokens = self._count_tokens()
        
        # 执行压缩
        compressor = self.strategies[strategy]
        compressed_messages = compressor.compress(self.messages, self.config)
        
        # 记录压缩历史
        compression_record = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy,
            "original_count": original_count,
            "compressed_count": len(compressed_messages),
            "original_tokens": original_tokens,
            "compressed_tokens": self._count_tokens_for_messages(compressed_messages),
            "compression_ratio": len(compressed_messages) / original_count if original_count > 0 else 0
        }
        
        self.compression_history.append(compression_record)
        self.messages = compressed_messages
        
        print(f"✅ 压缩完成: {original_count} -> {len(compressed_messages)} 条消息")
        print(f"📊 Token减少: {original_tokens} -> {compression_record['compressed_tokens']}")
        
        return compressed_messages
    
    def _count_tokens_for_messages(self, messages: List[BaseMessage]) -> int:
        """计算指定消息的token数"""
        total = 0
        for msg in messages:
            total += len(self.tokenizer.encode(msg.content))
        return total
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        return {
            "message_count": len(self.messages),
            "token_count": self._count_tokens(),
            "compression_count": len(self.compression_history),
            "storage_backend": self.storage_backend,
            "last_compression": self.compression_history[-1] if self.compression_history else None
        }
    
    def get_compression_history(self) -> List[Dict[str, Any]]:
        """获取压缩历史"""
        return self.compression_history
    
    def clear_memory(self) -> None:
        """清空记忆"""
        self.messages = []
        self.compression_history = []
        if self.storage_backend == "redis":
            try:
                self.redis_client.delete("chat_messages")
                self.redis_client.delete("compression_history")
            except Exception as e:
                print(f"清空Redis失败: {e}")
    
    def _save_to_storage(self):
        """保存到存储"""
        if self.storage_backend == "redis":
            try:
                # 序列化消息
                messages_data = []
                for msg in self.messages:
                    msg_data = {
                        "type": "human" if isinstance(msg, HumanMessage) else "ai",
                        "content": msg.content,
                        "additional_kwargs": getattr(msg, 'additional_kwargs', {})
                    }
                    messages_data.append(msg_data)
                
                self.redis_client.set("chat_messages", json.dumps(messages_data))
                self.redis_client.set("compression_history", json.dumps(self.compression_history))
            except Exception as e:
                print(f"保存到Redis失败: {e}")
    
    def _load_from_storage(self):
        """从存储加载"""
        if self.storage_backend == "redis":
            try:
                # 加载消息
                messages_data = self.redis_client.get("chat_messages")
                if messages_data:
                    messages_list = json.loads(messages_data)
                    self.messages = []
                    for msg_data in messages_list:
                        if msg_data["type"] == "human":
                            msg = HumanMessage(
                                content=msg_data["content"],
                                additional_kwargs=msg_data.get("additional_kwargs", {})
                            )
                        else:
                            msg = AIMessage(
                                content=msg_data["content"],
                                additional_kwargs=msg_data.get("additional_kwargs", {})
                            )
                        self.messages.append(msg)
                
                # 加载压缩历史
                history_data = self.redis_client.get("compression_history")
                if history_data:
                    self.compression_history = json.loads(history_data)
                    
            except Exception as e:
                print(f"从Redis加载失败: {e}")


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_subsection(title: str):
    """打印子章节标题"""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")


def demo_basic_compression():
    """演示基础压缩功能"""
    print_section("基础压缩功能演示")
    
    # 创建压缩器
    config = CompressionConfig(
        max_messages=10,
        max_tokens=1000,
        compression_ratio=0.5,
        preserve_recent=3
    )
    
    compressor = MemoryCompressor(config=config, storage_backend="memory")
    
    # 模拟对话
    conversations = [
        ("user", "你好，我想学习Python编程"),
        ("assistant", "你好！我很乐意帮助你学习Python。Python是一门非常适合初学者的编程语言。"),
        ("user", "我应该从哪里开始？"),
        ("assistant", "建议你从基础语法开始，包括变量、数据类型、控制结构等。"),
        ("user", "能推荐一些学习资源吗？"),
        ("assistant", "当然！我推荐官方文档、《Python编程：从入门到实践》这本书。"),
        ("user", "今天天气真不错"),
        ("assistant", "是的，好天气总是让人心情愉快。"),
        ("user", "我想吃冰淇淋"),
        ("assistant", "冰淇淋确实很美味，不过要适量哦。"),
        ("user", "回到Python学习，我需要安装什么软件？"),
        ("assistant", "你需要安装Python解释器和一个代码编辑器，比如VS Code或PyCharm。"),
    ]
    
    print("\n📝 添加对话消息...")
    for role, content in conversations:
        compressor.add_message(role, content)
        print(f"{role}: {content[:50]}...")
    
    # 显示统计信息
    stats = compressor.get_memory_stats()
    print(f"\n📊 记忆统计:")
    print(f"   消息数量: {stats['message_count']}")
    print(f"   Token数量: {stats['token_count']}")
    print(f"   压缩次数: {stats['compression_count']}")
    
    # 显示压缩历史
    if compressor.compression_history:
        print("\n📈 压缩历史:")
        for record in compressor.compression_history:
            print(f"   {record['timestamp'][:19]} | {record['strategy']} | "
                  f"{record['original_count']} -> {record['compressed_count']} 条消息")


def demo_compression_strategies():
    """演示不同压缩策略"""
    print_section("压缩策略对比演示")
    
    # 准备测试数据
    test_messages = [
        HumanMessage("你好，我想了解机器学习"),
        AIMessage("你好！机器学习是人工智能的一个重要分支，它让计算机能够从数据中学习。"),
        HumanMessage("有哪些主要的机器学习算法？"),
        AIMessage("主要包括监督学习（如线性回归、决策树）、无监督学习（如聚类、降维）和强化学习。"),
        HumanMessage("今天天气很好"),
        AIMessage("是的，好天气确实让人心情愉快。"),
        HumanMessage("我想喝咖啡"),
        AIMessage("咖啡可以提神，但不要喝太多哦。"),
        HumanMessage("回到机器学习话题，我该如何开始学习？"),
        AIMessage("建议先学习Python和数学基础，然后从简单的算法开始实践。")
    ]
    
    config = CompressionConfig(preserve_recent=2)
    
    strategies = ["summary", "keyword", "importance", "progressive"]
    
    for strategy in strategies:
        print_subsection(f"{strategy.upper()} 压缩策略")
        
        compressor = MemoryCompressor(config=config)
        compressor.messages = test_messages.copy()
        
        print(f"原始消息数: {len(compressor.messages)}")
        print(f"原始Token数: {compressor._count_tokens()}")
        
        # 执行压缩
        compressed = compressor.compress(strategy=strategy)
        
        print(f"压缩后消息数: {len(compressed)}")
        print(f"压缩后Token数: {compressor._count_tokens()}")
        
        print("\n压缩结果:")
        for i, msg in enumerate(compressed):
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            compressed_flag = "[压缩]" if msg.additional_kwargs.get("compressed") else ""
            print(f"  {i+1}. {role}{compressed_flag}: {content}")


def demo_redis_integration():
    """演示Redis集成"""
    print_section("Redis存储集成演示")
    
    if not REDIS_AVAILABLE:
        print("❌ Redis未安装，跳过此演示")
        return
    
    try:
        # 测试Redis连接
        redis_client = redis.from_url("redis://localhost:6379/0")
        redis_client.ping()
        print("✅ Redis连接成功")
        
        # 创建Redis压缩器
        compressor = MemoryCompressor(
            storage_backend="redis",
            redis_url="redis://localhost:6379/0"
        )
        
        # 清空之前的数据
        compressor.clear_memory()
        
        # 添加一些对话
        conversations = [
            ("user", "我想学习数据科学"),
            ("assistant", "数据科学是一个很有前景的领域，涉及统计学、编程和领域知识。"),
            ("user", "需要掌握哪些技能？"),
            ("assistant", "主要需要Python/R编程、统计学、机器学习和数据可视化技能。")
        ]
        
        for role, content in conversations:
            compressor.add_message(role, content)
        
        print(f"\n💾 已保存 {len(compressor.messages)} 条消息到Redis")
        
        # 创建新的压缩器实例，测试数据持久化
        new_compressor = MemoryCompressor(
            storage_backend="redis",
            redis_url="redis://localhost:6379/0"
        )
        new_compressor._load_from_storage()
        
        print(f"🔄 从Redis加载了 {len(new_compressor.messages)} 条消息")
        
        # 显示加载的消息
        for i, msg in enumerate(new_compressor.messages):
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            print(f"  {i+1}. {role}: {msg.content}")
            
    except Exception as e:
        print(f"❌ Redis演示失败: {e}")
        print("请确保Redis服务正在运行")


def demo_compression_evaluation():
    """演示压缩效果评估"""
    print_section("压缩效果评估")
    
    # 创建包含不同类型对话的测试数据
    mixed_conversations = [
        ("user", "我需要帮助解决一个重要的技术问题"),
        ("assistant", "我很乐意帮助你解决技术问题。请详细描述你遇到的问题。"),
        ("user", "我的Python程序出现了内存泄漏"),
        ("assistant", "内存泄漏是个严重问题。请检查是否有循环引用或未释放的资源。"),
        ("user", "今天天气真好"),
        ("assistant", "是的，阳光明媚的日子总是让人心情愉快。"),
        ("user", "我想吃披萨"),
        ("assistant", "披萨确实很美味，你喜欢什么口味的？"),
        ("user", "回到技术问题，你能提供具体的调试方法吗？"),
        ("assistant", "当然！你可以使用memory_profiler库来监控内存使用，或者使用gc模块检查垃圾回收。"),
        ("user", "这个解决方案很有用，谢谢！"),
        ("assistant", "不客气！如果还有其他技术问题，随时可以问我。")
    ]
    
    config = CompressionConfig(
        max_messages=8,
        preserve_recent=3,
        min_importance=0.6
    )
    
    compressor = MemoryCompressor(config=config)
    
    # 添加所有对话
    for role, content in mixed_conversations:
        compressor.add_message(role, content)
    
    # 评估不同策略的效果
    strategies = ["summary", "importance", "progressive"]
    evaluation_results = {}
    
    for strategy in strategies:
        # 重置消息
        test_compressor = MemoryCompressor(config=config)
        for role, content in mixed_conversations:
            test_compressor.messages.append(
                HumanMessage(content) if role == "user" else AIMessage(content)
            )
        
        original_count = len(test_compressor.messages)
        original_tokens = test_compressor._count_tokens()
        
        # 执行压缩
        compressed = test_compressor.compress(strategy=strategy)
        
        # 计算评估指标
        compression_ratio = len(compressed) / original_count
        token_reduction = (original_tokens - test_compressor._count_tokens()) / original_tokens
        
        evaluation_results[strategy] = {
            "compression_ratio": compression_ratio,
            "token_reduction": token_reduction,
            "final_count": len(compressed),
            "final_tokens": test_compressor._count_tokens()
        }
    
    # 显示评估结果
    print("\n📊 压缩策略评估结果:")
    print(f"{'策略':<12} {'压缩比':<8} {'Token减少':<10} {'最终消息数':<10} {'最终Token数':<10}")
    print("-" * 60)
    
    for strategy, results in evaluation_results.items():
        print(f"{strategy:<12} {results['compression_ratio']:<8.2f} "
              f"{results['token_reduction']:<10.2%} {results['final_count']:<10} "
              f"{results['final_tokens']:<10}")
    
    # 推荐最佳策略
    best_strategy = min(evaluation_results.keys(), 
                       key=lambda x: evaluation_results[x]['final_tokens'])
    print(f"\n🏆 推荐策略: {best_strategy} (Token数最少)")


def demo_real_world_scenario():
    """演示真实世界应用场景"""
    print_section("真实应用场景演示")
    
    print("🎯 场景：智能客服机器人")
    print("用户咨询技术问题，期间穿插闲聊，需要保持技术上下文")
    
    # 模拟长对话
    customer_service_chat = [
        ("user", "你好，我的网站打不开了"),
        ("assistant", "你好！我来帮你解决网站问题。请问你的网站域名是什么？"),
        ("user", "www.example.com"),
        ("assistant", "我来检查一下。请问你最近有修改过DNS设置吗？"),
        ("user", "没有，昨天还好好的"),
        ("assistant", "明白了。让我检查服务器状态和域名解析。"),
        ("user", "好的，谢谢。顺便问一下，你们公司在哪里？"),
        ("assistant", "我们是在线技术支持服务。现在专注解决你的网站问题。"),
        ("user", "今天天气不错呢"),
        ("assistant", "是的。回到你的问题，我发现你的DNS解析有问题。"),
        ("user", "怎么解决？"),
        ("assistant", "你需要联系域名提供商更新DNS记录，指向正确的服务器IP。"),
        ("user", "具体步骤是什么？"),
        ("assistant", "1. 登录域名管理后台 2. 找到DNS设置 3. 修改A记录指向你的服务器IP"),
        ("user", "我不知道服务器IP"),
        ("assistant", "你可以联系你的主机提供商获取服务器IP地址。"),
        ("user", "好的，我去联系他们。如果还有问题可以再找你吗？"),
        ("assistant", "当然可以！我会记住你的问题背景，随时为你提供帮助。")
    ]
    
    # 配置压缩器
    config = CompressionConfig(
        max_messages=12,
        preserve_recent=4,
        min_importance=0.7
    )
    
    compressor = MemoryCompressor(config=config)
    
    print("\n💬 客服对话进行中...")
    for i, (role, content) in enumerate(customer_service_chat):
        compressor.add_message(role, content)
        
        # 显示关键节点
        if i == 5:
            print(f"\n📍 对话进行到第{i+1}轮，开始出现闲聊")
        elif i == 10:
            print(f"\n📍 对话进行到第{i+1}轮，回到技术问题")
        elif i == len(customer_service_chat) - 1:
            print(f"\n📍 对话结束，共{i+1}轮")
    
    # 显示最终状态
    print("\n🔍 最终记忆状态:")
    for i, msg in enumerate(compressor.messages):
        role = "客户" if isinstance(msg, HumanMessage) else "客服"
        compressed_flag = "[已压缩]" if msg.additional_kwargs.get("compressed") else ""
        content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        print(f"  {i+1}. {role}{compressed_flag}: {content}")
    
    # 分析压缩效果
    stats = compressor.get_memory_stats()
    print(f"\n📈 压缩效果分析:")
    print(f"   保留消息数: {stats['message_count']}")
    print(f"   总Token数: {stats['token_count']}")
    print(f"   压缩次数: {stats['compression_count']}")
    
    if compressor.compression_history:
        last_compression = compressor.compression_history[-1]
        print(f"   最后压缩: {last_compression['original_count']} -> {last_compression['compressed_count']} 条")
        print(f"   压缩策略: {last_compression['strategy']}")
    
    print("\n✅ 关键技术信息得到保留，闲聊内容被有效压缩")


def main():
    """主函数"""
    print("🚀 LangChain 记忆压缩功能演示")
    print("本演示将展示智能记忆压缩的各种功能和策略")
    
    try:
        # 1. 基础压缩功能
        demo_basic_compression()
        
        # 2. 压缩策略对比
        demo_compression_strategies()
        
        # 3. Redis集成
        demo_redis_integration()
        
        # 4. 压缩效果评估
        demo_compression_evaluation()
        
        # 5. 真实应用场景
        demo_real_world_scenario()
        
        print_section("演示完成")
        print("🎉 恭喜！你已经掌握了LangChain记忆压缩的核心功能")
        print("\n📚 关键要点回顾:")
        print("   1. 记忆压缩解决上下文窗口限制问题")
        print("   2. 多种压缩策略适应不同场景")
        print("   3. 智能保留重要信息，过滤无关内容")
        print("   4. 支持多种存储后端")
        print("   5. 提供压缩效果评估和优化")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        print("请检查网络连接和API配置")


if __name__ == "__main__":
    main()