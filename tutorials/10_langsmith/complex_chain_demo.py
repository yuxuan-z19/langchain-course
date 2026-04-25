#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangSmith 复杂链式应用接入示例

本示例演示如何将 LangSmith 集成到复杂的 LangChain 应用中，实现：
1. 多步骤处理链
2. 工具调用和函数执行
3. 条件分支和路由
4. 并行处理
5. 错误恢复和重试机制
6. 自定义追踪和标签
7. 与 DeepSeek API 集成

运行前请确保：
1. 已在 https://smith.langchain.com 注册账号并创建项目
2. 环境变量已正确配置（参考 .env.example）
3. 已安装必要的依赖包
"""

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.tools import tool

# LangChain 相关导入
from langchain_openai import ChatOpenAI

# LangSmith 相关导入
from langsmith import Client
from langsmith.run_helpers import traceable
from langsmith.wrappers import wrap_openai

# 导入项目统一配置管理
from utils.config import (
    load_deepseek_config,
    load_environment,
    setup_environment_variables,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("complex_langsmith_demo.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class QueryType(Enum):
    """查询类型枚举"""

    SIMPLE_QA = "simple_qa"
    CALCULATION = "calculation"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    MULTI_STEP = "multi_step"


@dataclass
class ProcessingResult:
    """处理结果数据类"""

    query_type: QueryType
    result: str
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]


class LangSmithComplexConfig:
    """LangSmith 复杂应用配置管理"""

    def __init__(self):
        # 加载项目统一配置
        self.env_config = load_environment()

        self.api_url = self.env_config.langchain_endpoint
        self.api_key = self.env_config.langchain_api_key
        self.project = self.env_config.langchain_project
        self.tracing_enabled = self.env_config.langchain_tracing

        # 验证必要的配置
        if not self.api_key:
            raise ValueError(
                "LANGCHAIN_API_KEY 环境变量未设置，请在项目根目录的 .env 文件中配置"
            )
        if not (self.api_key.startswith("ls__") or self.api_key.startswith("lsv2_")):
            raise ValueError(
                "LANGCHAIN_API_KEY 格式不正确，应该以 'ls__' 或 'lsv2_' 开头"
            )

    def setup_environment(self):
        """设置 LangSmith 环境变量"""
        # 使用项目统一的环境变量设置
        setup_environment_variables()

        logger.info(f"LangSmith 复杂应用配置完成:")
        logger.info(f"  - API URL: {self.api_url}")
        logger.info(f"  - Project: {self.project}")
        logger.info(f"  - Tracing: {self.tracing_enabled}")

    def get_client(self) -> Client:
        """获取 LangSmith 客户端"""
        return Client(api_url=self.api_url, api_key=self.api_key)


# 定义工具函数
@tool
def calculate_math(expression: str) -> str:
    """安全地计算数学表达式"""
    try:
        # 简单的数学表达式计算（生产环境中应使用更安全的方法）
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不允许的字符"

        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def get_current_time() -> str:
    """获取当前时间"""
    return f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def analyze_sentiment(text: str) -> str:
    """简单的情感分析（模拟）"""
    positive_words = ["好", "棒", "优秀", "喜欢", "满意", "开心", "高兴"]
    negative_words = ["坏", "差", "糟糕", "讨厌", "不满", "难过", "生气"]

    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)

    if positive_count > negative_count:
        return "情感分析结果: 积极"
    elif negative_count > positive_count:
        return "情感分析结果: 消极"
    else:
        return "情感分析结果: 中性"


class ComplexChainProcessor:
    """复杂链式处理器"""

    def __init__(self, langsmith_config: LangSmithComplexConfig):
        self.langsmith_config = langsmith_config
        self.langsmith_client = langsmith_config.get_client()

        # 获取 DeepSeek 配置
        deepseek_config = load_deepseek_config()

        # 初始化 LLM
        self.llm = ChatOpenAI(
            api_key=deepseek_config["api_key"],
            base_url=deepseek_config["base_url"],
            model="deepseek-chat",  # 使用默认的 DeepSeek 模型
            temperature=0.7,
            max_tokens=1500,
        )

        # 初始化工具
        self.tools = [calculate_math, get_current_time, analyze_sentiment]

        # 创建查询分类器
        self.query_classifier = self._create_query_classifier()

        # 创建处理链
        self.processing_chain = self._create_processing_chain()

        logger.info("ComplexChainProcessor 初始化完成")

    @traceable(name="classify_query")
    def _create_query_classifier(self):
        """创建查询分类器"""
        classification_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            你是一个查询分类器。请将用户的查询分类为以下类型之一：
            - simple_qa: 简单的问答
            - calculation: 数学计算
            - analysis: 文本分析（如情感分析）
            - creative: 创意写作
            - multi_step: 需要多步骤处理的复杂任务
            
            请只返回分类结果，格式为JSON：{{"type": "分类类型", "confidence": 0.9}}
            """,
                ),
                ("human", "请分类这个查询：{query}"),
            ]
        )

        return classification_prompt | self.llm | JsonOutputParser()

    @traceable(name="create_processing_chain")
    def _create_processing_chain(self):
        """创建主处理链"""

        # 定义不同类型的处理分支
        def route_query(inputs):
            query_type = inputs.get("classification", {}).get("type", "simple_qa")
            query = inputs["query"]

            if query_type == "calculation":
                return self._process_calculation(query)
            elif query_type == "analysis":
                return self._process_analysis(query)
            elif query_type == "creative":
                return self._process_creative(query)
            elif query_type == "multi_step":
                return self._process_multi_step(query)
            else:
                return self._process_simple_qa(query)

        # 创建处理链
        chain = RunnablePassthrough.assign(
            classification=lambda x: self.query_classifier.invoke({"query": x["query"]})
        ) | RunnableLambda(route_query)

        return chain

    @traceable(name="process_multi_step")
    def _process_multi_step(self, query: str) -> str:
        """处理多步骤查询"""
        # 步骤1: 分析查询需求
        analysis_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "分析用户查询，识别需要执行的步骤。将复杂任务分解为具体的执行步骤。",
                ),
                ("human", "查询: {query}\n\n请分析这个查询需要哪些步骤来完成："),
            ]
        )

        analysis_chain = analysis_prompt | self.llm | StrOutputParser()
        steps_analysis = analysis_chain.invoke({"query": query})

        # 步骤2: 执行具体任务
        execution_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个任务执行助手。根据分析结果，逐步执行任务并提供详细的执行过程。",
                ),
                (
                    "human",
                    "原始查询: {query}\n\n步骤分析: {steps}\n\n请按步骤执行任务：",
                ),
            ]
        )

        execution_chain = execution_prompt | self.llm | StrOutputParser()
        result = execution_chain.invoke({"query": query, "steps": steps_analysis})

        # 步骤3: 结果整合和验证
        validation_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "检查执行结果的完整性和准确性，如有需要进行补充或修正。"),
                (
                    "human",
                    "原始查询: {query}\n\n执行结果: {result}\n\n请验证结果并提供最终答案：",
                ),
            ]
        )

        validation_chain = validation_prompt | self.llm | StrOutputParser()
        final_result = validation_chain.invoke({"query": query, "result": result})

        return f"[多步骤处理] {final_result}"

    @traceable(name="process_calculation")
    def _process_calculation(self, query: str) -> str:
        """处理计算查询"""
        # 提取数学表达式
        extraction_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "从用户的查询中提取数学表达式。只返回可以直接计算的表达式，如：2+3*4",
                ),
                ("human", "{query}"),
            ]
        )

        extraction_chain = extraction_prompt | self.llm | StrOutputParser()
        expression = extraction_chain.invoke({"query": query})

        # 使用工具计算
        result = calculate_math.invoke({"expression": expression.strip()})

        return f"[计算结果] {result}"

    @traceable(name="process_analysis")
    def _process_analysis(self, query: str) -> str:
        """处理分析查询"""
        # 提取要分析的文本
        extraction_prompt = ChatPromptTemplate.from_messages(
            [("system", "从用户的查询中提取需要分析的文本内容"), ("human", "{query}")]
        )

        extraction_chain = extraction_prompt | self.llm | StrOutputParser()
        text_to_analyze = extraction_chain.invoke({"query": query})

        # 进行情感分析
        sentiment_result = analyze_sentiment.invoke({"text": text_to_analyze})

        return f"[分析结果] {sentiment_result}"

    @traceable(name="process_creative")
    def _process_creative(self, query: str) -> str:
        """处理创意查询"""
        creative_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个创意写作助手。请根据用户的要求进行创意写作，保持内容积极向上。",
                ),
                ("human", "{query}"),
            ]
        )

        creative_chain = creative_prompt | self.llm | StrOutputParser()
        result = creative_chain.invoke({"query": query})

        return f"[创意内容] {result}"

    @traceable(name="process_simple_qa")
    def _process_simple_qa(self, query: str) -> str:
        """处理简单问答"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个友好的AI助手。请简洁、准确地回答用户的问题。"),
                ("human", "{query}"),
            ]
        )

        qa_chain = qa_prompt | self.llm | StrOutputParser()
        result = qa_chain.invoke({"query": query})

        return f"[简单回答] {result}"

    @traceable(name="process_query")
    def process_query(self, query: str) -> ProcessingResult:
        """处理用户查询"""
        start_time = datetime.now()

        try:
            # 分类查询
            classification = self.query_classifier.invoke({"query": query})
            query_type = QueryType(classification.get("type", "simple_qa"))
            confidence = classification.get("confidence", 0.5)

            # 处理查询
            result = self.processing_chain.invoke({"query": query})

            # 计算处理时间
            processing_time = (datetime.now() - start_time).total_seconds()

            return ProcessingResult(
                query_type=query_type,
                result=result,
                confidence=confidence,
                processing_time=processing_time,
                metadata={
                    "classification": classification,
                    "timestamp": start_time.isoformat(),
                },
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"查询处理失败: {e}")

            return ProcessingResult(
                query_type=QueryType.SIMPLE_QA,
                result=f"处理查询时发生错误: {str(e)}",
                confidence=0.0,
                processing_time=processing_time,
                metadata={"error": str(e)},
            )

    @traceable(name="batch_process")
    def batch_process(self, queries: List[str]) -> List[ProcessingResult]:
        """批量处理查询"""
        results = []
        for i, query in enumerate(queries):
            logger.info(f"处理批量查询 {i+1}/{len(queries)}: {query}")
            result = self.process_query(query)
            results.append(result)
        return results

    @traceable(name="parallel_process")
    async def parallel_process(self, queries: List[str]) -> List[ProcessingResult]:
        """并行处理查询"""

        async def process_single(query):
            return self.process_query(query)

        tasks = [process_single(query) for query in queries]
        results = await asyncio.gather(*tasks)
        return results


class ComplexChainDemo:
    """复杂链式应用演示"""

    def __init__(self):
        self.config = LangSmithComplexConfig()
        self.config.setup_environment()
        self.processor = ComplexChainProcessor(self.config)

    @traceable(name="demo_query_classification")
    def demo_query_classification(self):
        """演示查询分类"""
        print("\n=== 查询分类演示 ===")

        test_queries = [
            "帮我制定一个学习计划，包括时间安排和进度跟踪",  # multi_step
            "计算 15 * 8 + 32",  # calculation
            "分析这句话的情感：我今天很开心",  # analysis
            "写一首关于春天的诗",  # creative
            "今天天气怎么样？",  # simple_qa
        ]

        for query in test_queries:
            print(f"\n查询: {query}")
            result = self.processor.process_query(query)
            print(f"类型: {result.query_type.value}")
            print(f"置信度: {result.confidence:.2f}")
            print(f"结果: {result.result[:100]}...")
            print(f"处理时间: {result.processing_time:.2f}秒")

    @traceable(name="demo_multi_step_processing")
    def demo_multi_step_processing(self):
        """演示多步骤处理"""
        print("\n=== 多步骤处理演示 ===")

        multi_step_queries = [
            "帮我规划一次北京三日游，包括景点推荐、路线安排和预算估算",
            "设计一个简单的网站架构，包括前端、后端和数据库的选择",
            "制定一个月的健身计划，包括运动项目、时间安排和饮食建议",
        ]

        for query in multi_step_queries:
            print(f"\n查询: {query}")
            result = self.processor.process_query(query)
            print(f"处理结果: {result.result[:200]}...")

    @traceable(name="demo_tool_usage")
    def demo_tool_usage(self):
        """演示工具使用"""
        print("\n=== 工具使用演示 ===")

        tool_queries = [
            "现在几点了？",
            "计算 123 + 456 * 2",
            "分析这段文字的情感：这个产品真的很棒，我非常满意！",
        ]

        for query in tool_queries:
            print(f"\n查询: {query}")
            result = self.processor.process_query(query)
            print(f"结果: {result.result}")

    @traceable(name="demo_batch_processing")
    def demo_batch_processing(self):
        """演示批量处理"""
        print("\n=== 批量处理演示 ===")

        batch_queries = [
            "帮我分析一下学习编程的最佳路径",
            "计算 50 / 5 + 10",
            "写一个关于友谊的短故事",
        ]

        print(f"\n批量处理 {len(batch_queries)} 个查询...")
        results = self.processor.batch_process(batch_queries)

        for i, (query, result) in enumerate(zip(batch_queries, results), 1):
            print(f"\n查询 {i}: {query}")
            print(f"类型: {result.query_type.value}")
            print(f"结果: {result.result[:100]}...")

    @traceable(name="demo_error_handling")
    def demo_error_handling(self):
        """演示错误处理"""
        print("\n=== 错误处理演示 ===")

        error_queries = [
            "计算 1/0",  # 数学错误
            "",  # 空查询
            "a" * 5000,  # 超长查询
        ]

        for query in error_queries:
            print(f"\n测试查询: {query[:50]}{'...' if len(query) > 50 else ''}")
            try:
                result = self.processor.process_query(query)
                print(f"结果: {result.result[:100]}...")
                print(f"错误信息: {result.metadata.get('error', '无')}")
            except Exception as e:
                print(f"捕获异常: {e}")

    def demo_langsmith_features(self):
        """演示LangSmith高级特性"""
        print("\n=== LangSmith高级特性演示 ===")

        try:
            client = self.config.get_client()

            # 创建带标签的运行
            with client.trace(
                name="complex_chain_demo",
                project_name=self.config.project,
                inputs={"demo_type": "advanced_features"},
                tags=["demo", "complex", "rag", "tools"],
            ) as run:
                # 执行复杂查询
                query = "帮我制定一个完整的项目开发流程，包括需求分析、技术选型、开发阶段和测试部署，并计算如果团队有5个人，每人每天工作8小时，30天能完成多少工时"
                result = self.processor.process_query(query)

                run.end(
                    outputs={
                        "query": query,
                        "result": result.result,
                        "query_type": result.query_type.value,
                        "confidence": result.confidence,
                        "processing_time": result.processing_time,
                    }
                )

                print(f"\n复杂查询: {query}")
                print(f"结果: {result.result[:300]}...")
                print(f"追踪运行 ID: {run.id}")

        except Exception as e:
            logger.error(f"LangSmith高级特性演示失败: {e}")
            print(f"LangSmith连接失败: {e}")

    def run_all_demos(self):
        """运行所有演示"""
        print("\n🚀 开始 LangSmith 复杂链式应用演示")
        print("=" * 50)

        try:
            # 运行各种演示
            self.demo_query_classification()
            self.demo_multi_step_processing()
            self.demo_tool_usage()
            self.demo_batch_processing()
            self.demo_error_handling()
            self.demo_langsmith_features()

            print("\n✅ 所有演示完成！")
            print(f"\n📊 请查看 LangSmith 界面来查看详细的追踪信息:")
            print(f"URL: https://smith.langchain.com/o/{self.config.project}")

        except Exception as e:
            logger.error(f"演示运行失败: {e}")
            print(f"❌ 演示失败: {e}")


def main():
    """主函数"""
    print("LangSmith 复杂链式应用接入示例")
    print("=" * 40)

    try:
        # 创建并运行演示
        demo = ComplexChainDemo()
        demo.run_all_demos()

    except KeyboardInterrupt:
        print("\n👋 演示被用户中断")
    except Exception as e:
        logger.error(f"程序运行失败: {e}")
        print(f"❌ 程序运行失败: {e}")


if __name__ == "__main__":
    main()
