#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangSmith 简单应用接入示例

本示例演示如何将 LangSmith 集成到简单的 LangChain 应用中，实现：
1. 基础的追踪功能
2. 简单的聊天机器人
3. 提示词模板使用
4. 错误处理和日志记录
5. 与 DeepSeek API 集成

运行前请确保：
1. 已在 https://smith.langchain.com 注册账号并创建项目
2. 环境变量已正确配置（参考 .env.example）
3. 已安装必要的依赖包
"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

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
    handlers=[logging.FileHandler("langsmith_demo.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class LangSmithConfig:
    """LangSmith 配置管理"""

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

        logger.info(f"LangSmith 配置完成:")
        logger.info(f"  - API URL: {self.api_url}")
        logger.info(f"  - Project: {self.project}")
        logger.info(f"  - Tracing: {self.tracing_enabled}")

    def get_client(self) -> Client:
        """获取 LangSmith 客户端"""
        return Client(api_url=self.api_url, api_key=self.api_key)


class SimpleChatBot:
    """简单的聊天机器人，集成 LangSmith 追踪"""

    def __init__(self, langsmith_config: LangSmithConfig):
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
            max_tokens=1000,
        )

        # 创建提示词模板
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个友好的AI助手。请用简洁、有用的方式回答用户的问题。",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        # 创建处理链
        self.chain = (
            RunnablePassthrough.assign(chat_history=lambda x: x.get("chat_history", []))
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

        # 聊天历史
        self.chat_history = []

        logger.info("SimpleChatBot 初始化完成")

    @traceable(name="simple_chat")
    def chat(self, user_input: str) -> str:
        """处理用户输入并返回回复"""
        try:
            # 记录输入
            logger.info(f"用户输入: {user_input}")

            # 调用链处理
            response = self.chain.invoke(
                {"input": user_input, "chat_history": self.chat_history}
            )

            # 更新聊天历史
            self.chat_history.extend(
                [HumanMessage(content=user_input), AIMessage(content=response)]
            )

            # 限制历史长度
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]

            logger.info(f"AI回复: {response}")
            return response

        except Exception as e:
            error_msg = f"处理聊天时发生错误: {str(e)}"
            logger.error(error_msg)
            return f"抱歉，我遇到了一些问题: {str(e)}"

    @traceable(name="batch_chat")
    def batch_chat(self, inputs: list[str]) -> list[str]:
        """批量处理多个输入"""
        results = []
        for i, user_input in enumerate(inputs):
            logger.info(f"处理批量输入 {i+1}/{len(inputs)}: {user_input}")
            response = self.chat(user_input)
            results.append(response)
        return results

    def get_chat_history(self) -> list:
        """获取聊天历史"""
        return self.chat_history

    def clear_history(self):
        """清空聊天历史"""
        self.chat_history = []
        logger.info("聊天历史已清空")


class LangSmithDemo:
    """LangSmith 演示类"""

    def __init__(self):
        self.config = LangSmithConfig()
        self.config.setup_environment()
        self.chatbot = SimpleChatBot(self.config)

    @traceable(name="demo_single_chat")
    def demo_single_chat(self):
        """演示单次对话"""
        print("\n=== 单次对话演示 ===")

        questions = [
            "你好，请介绍一下自己",
            "什么是人工智能？",
            "请推荐一本关于机器学习的书",
        ]

        for question in questions:
            print(f"\n用户: {question}")
            response = self.chatbot.chat(question)
            print(f"AI: {response}")

    @traceable(name="demo_conversation")
    def demo_conversation(self):
        """演示连续对话"""
        print("\n=== 连续对话演示 ===")

        conversation = [
            "我想学习编程，有什么建议吗？",
            "我对Python比较感兴趣",
            "能推荐一些Python学习资源吗？",
            "谢谢你的建议！",
        ]

        for message in conversation:
            print(f"\n用户: {message}")
            response = self.chatbot.chat(message)
            print(f"AI: {response}")

    @traceable(name="demo_batch_processing")
    def demo_batch_processing(self):
        """演示批量处理"""
        print("\n=== 批量处理演示 ===")

        # 清空历史，开始新的批量处理
        self.chatbot.clear_history()

        batch_inputs = ["什么是区块链？", "解释一下云计算的概念", "人工智能的发展历史"]

        print(f"\n批量处理 {len(batch_inputs)} 个问题...")
        responses = self.chatbot.batch_chat(batch_inputs)

        for i, (question, answer) in enumerate(zip(batch_inputs, responses), 1):
            print(f"\n问题 {i}: {question}")
            print(f"回答 {i}: {answer}")

    @traceable(name="demo_error_handling")
    def demo_error_handling(self):
        """演示错误处理"""
        print("\n=== 错误处理演示 ===")

        # 模拟一些可能导致错误的输入
        test_inputs = [
            "正常问题：今天天气怎么样？",
            "",  # 空输入
            "a" * 10000,  # 超长输入
        ]

        for test_input in test_inputs:
            print(
                f"\n测试输入: {test_input[:50]}{'...' if len(test_input) > 50 else ''}"
            )
            try:
                response = self.chatbot.chat(test_input)
                print(f"响应: {response[:100]}{'...' if len(response) > 100 else ''}")
            except Exception as e:
                print(f"捕获错误: {e}")

    def demo_langsmith_features(self):
        """演示 LangSmith 特性"""
        print("\n=== LangSmith 特性演示 ===")

        try:
            client = self.config.get_client()

            # 获取项目信息
            print(f"\n当前项目: {self.config.project}")
            print(f"LangSmith API: {self.config.api_url}")

            # 创建一个简单的运行记录
            from langsmith import traceable

            @traceable(name="langsmith_feature_demo", project_name=self.config.project)
            def demo_chat():
                return self.chatbot.chat("请简单介绍一下 LangSmith 的作用")

            # 在追踪中执行一些操作
            result = demo_chat()

            print(f"\n响应: {result}")
            print("\n追踪已记录到 LangSmith，请访问 Web UI 查看详细信息")

        except Exception as e:
            logger.error(f"LangSmith 特性演示失败: {e}")
            print(f"LangSmith 连接失败: {e}")

    def run_all_demos(self):
        """运行所有演示"""
        print("开始 LangSmith 简单应用演示...")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            self.demo_single_chat()
            self.demo_conversation()
            self.demo_batch_processing()
            self.demo_error_handling()
            self.demo_langsmith_features()

            print("\n=== 演示完成 ===")
            print("请访问 LangSmith Web UI 查看追踪结果:")
            print(f"URL: https://smith.langchain.com/o/{self.config.project}")

        except Exception as e:
            logger.error(f"演示过程中发生错误: {e}")
            print(f"演示失败: {e}")


def main():
    """主函数"""
    print("LangSmith 简单应用接入演示")
    print("=" * 50)

    try:
        # 运行演示（环境检查在LangSmithConfig中进行）
        demo = LangSmithDemo()
        demo.run_all_demos()
    except ValueError as e:
        print(f"配置错误: {e}")
        print("请检查项目根目录的 .env 文件配置")
    except Exception as e:
        print(f"运行错误: {e}")
        logger.error(f"演示运行失败: {e}")


if __name__ == "__main__":
    main()
