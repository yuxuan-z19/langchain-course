#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础工具调用演示

本示例演示如何使用LangChain的基础工具调用功能，包括：
1. 数学计算工具
2. 时间查询工具
3. 文件操作工具
4. 字符串处理工具

作者: jaguarliu
日期: 2025年8月
"""

import datetime
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytz

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage
from langchain.tools import tool

# 导入LangChain相关模块
from langchain_openai import ChatOpenAI
from utils.config import load_deepseek_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("basic_tools_demo.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ==================== 基础工具定义 ====================


@tool
def calculator(expression: str) -> str:
    """
    执行数学计算

    Args:
        expression: 要计算的数学表达式，如 "2 + 3 * 4" 或 "sqrt(16)"

    Returns:
        计算结果的字符串表示
    """
    try:
        # 安全的数学表达式计算
        # 只允许基本的数学运算和函数
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        allowed_names.update({"abs": abs, "round": round})

        # 替换常用函数名
        expression = expression.replace("sqrt", "math.sqrt")
        expression = expression.replace("sin", "math.sin")
        expression = expression.replace("cos", "math.cos")
        expression = expression.replace("tan", "math.tan")
        expression = expression.replace("log", "math.log")
        expression = expression.replace("exp", "math.exp")

        result = eval(expression, {"__builtins__": {}, "math": math}, allowed_names)
        logger.info(f"计算表达式: {expression} = {result}")
        return str(result)
    except Exception as e:
        error_msg = f"计算错误: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def get_current_time(timezone: str = "UTC") -> str:
    """
    获取当前时间

    Args:
        timezone: 时区，默认为UTC。支持 "UTC", "Beijing", "local", 或标准时区名称如"Asia/Shanghai"

    Returns:
        格式化的当前时间字符串
    """
    try:
        # 获取当前UTC时间
        utc_now = datetime.datetime.now(pytz.UTC)

        if timezone.lower() == "beijing" or timezone == "Asia/Shanghai":
            # 北京时间 (Asia/Shanghai)
            beijing_tz = pytz.timezone("Asia/Shanghai")
            beijing_time = utc_now.astimezone(beijing_tz)
            result = f"北京时间: {beijing_time.strftime('%Y年%m月%d日 %H:%M:%S %Z')}"
        elif timezone.lower() == "utc":
            result = f"UTC时间: {utc_now.strftime('%Y年%m月%d日 %H:%M:%S %Z')}"
        elif timezone.lower() == "local":
            # 本地时间
            local_time = utc_now.astimezone()
            result = f"本地时间: {local_time.strftime('%Y年%m月%d日 %H:%M:%S %Z')}"
        else:
            # 尝试作为标准时区名称处理
            try:
                tz = pytz.timezone(timezone)
                local_time = utc_now.astimezone(tz)
                result = (
                    f"{timezone}时间: {local_time.strftime('%Y年%m月%d日 %H:%M:%S %Z')}"
                )
            except:
                # 如果时区名称无效，返回UTC时间
                result = f"UTC时间: {utc_now.strftime('%Y年%m月%d日 %H:%M:%S %Z')} (无效时区: {timezone})"

        logger.info(f"查询时间: {timezone} -> {result}")
        return result
    except Exception as e:
        error_msg = f"时间查询错误: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def read_file(file_path: str) -> str:
    """
    读取文件内容

    Args:
        file_path: 要读取的文件路径

    Returns:
        文件内容或错误信息
    """
    try:
        # 安全检查：只允许读取当前目录及子目录的文件
        abs_path = os.path.abspath(file_path)
        current_dir = os.path.abspath(".")

        if not abs_path.startswith(current_dir):
            return "错误: 只能读取当前目录及子目录的文件"

        if not os.path.exists(abs_path):
            return f"错误: 文件 {file_path} 不存在"

        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()

        logger.info(f"读取文件: {file_path}")
        return f"文件内容:\n{content}"
    except Exception as e:
        error_msg = f"文件读取错误: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def write_file(file_path: str, content: str) -> str:
    """
    写入文件内容

    Args:
        file_path: 要写入的文件路径
        content: 要写入的内容

    Returns:
        操作结果信息
    """
    try:
        # 安全检查：只允许写入当前目录及子目录的文件
        abs_path = os.path.abspath(file_path)
        current_dir = os.path.abspath(".")

        if not abs_path.startswith(current_dir):
            return "错误: 只能写入当前目录及子目录的文件"

        # 确保目录存在
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"写入文件: {file_path}")
        return f"成功写入文件: {file_path}"
    except Exception as e:
        error_msg = f"文件写入错误: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def string_processor(text: str, operation: str) -> str:
    """
    字符串处理工具

    Args:
        text: 要处理的文本
        operation: 操作类型，支持 "upper", "lower", "reverse", "length", "words"

    Returns:
        处理结果
    """
    try:
        if operation == "upper":
            result = text.upper()
        elif operation == "lower":
            result = text.lower()
        elif operation == "reverse":
            result = text[::-1]
        elif operation == "length":
            result = f"文本长度: {len(text)} 个字符"
        elif operation == "words":
            words = text.split()
            result = f"单词数量: {len(words)} 个单词\n单词列表: {words}"
        else:
            result = f"不支持的操作: {operation}。支持的操作: upper, lower, reverse, length, words"

        logger.info(f"字符串处理: {operation} -> {text[:50]}...")
        return result
    except Exception as e:
        error_msg = f"字符串处理错误: {str(e)}"
        logger.error(error_msg)
        return error_msg


class BasicToolsDemo:
    """基础工具调用演示类"""

    def __init__(self):
        """初始化演示类"""
        self.llm = None
        self.agent = None
        self.tools = [
            calculator,
            get_current_time,
            read_file,
            write_file,
            string_processor,
        ]
        self._setup_llm()
        self._setup_agent()

    def _setup_llm(self):
        """设置大语言模型"""
        try:
            config = load_deepseek_config()

            self.llm = ChatOpenAI(
                model="deepseek-chat",
                openai_api_key=config["api_key"],
                openai_api_base=config["base_url"],
                temperature=0.1,
                max_tokens=2000,
            )
            logger.info("DeepSeek模型初始化成功")
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise

    def _setup_agent(self):
        """设置智能代理"""
        try:
            # 创建提示模板
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
你是一个智能助手，可以使用各种工具来帮助用户完成任务。

可用工具:
1. calculator - 执行数学计算
2. get_current_time - 获取当前时间
3. read_file - 读取文件内容
4. write_file - 写入文件内容
5. string_processor - 处理字符串

重要规则：
- 当用户询问时间相关问题时，你必须使用get_current_time工具来获取准确的当前时间，不要凭记忆回答
- 当用户需要数学计算时，必须使用calculator工具
- 当用户需要文件操作时，必须使用相应的文件工具
- 当用户需要字符串处理时，必须使用string_processor工具

请根据用户的需求选择合适的工具，并提供清晰的解释。
如果需要多个步骤，请逐步执行。
                """,
                    ),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("user", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )

            # 创建代理
            agent = create_openai_functions_agent(
                llm=self.llm, tools=self.tools, prompt=prompt
            )

            # 创建代理执行器
            self.agent = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                max_iterations=5,
                return_intermediate_steps=True,
            )

            logger.info("智能代理初始化成功")
        except Exception as e:
            logger.error(f"代理初始化失败: {e}")
            raise

    def demo_calculator(self):
        """演示计算器工具"""
        print("\n" + "=" * 50)
        print("演示1: 数学计算工具")
        print("=" * 50)

        queries = ["计算 15 + 27 * 3", "求平方根 sqrt(144)", "计算圆的面积，半径为5"]

        for query in queries:
            print(f"\n用户: {query}")
            try:
                result = self.agent.invoke({"input": query, "chat_history": []})
                print(f"助手: {result['output']}")
            except Exception as e:
                print(f"错误: {e}")

    def demo_time_query(self):
        """演示时间查询工具"""
        print("\n" + "=" * 50)
        print("演示2: 时间查询工具")
        print("=" * 50)

        queries = ["现在几点了？", "北京时间是多少？", "UTC时间是多少？"]

        for query in queries:
            print(f"\n用户: {query}")
            try:
                result = self.agent.invoke({"input": query, "chat_history": []})
                print(f"助手: {result['output']}")
            except Exception as e:
                print(f"错误: {e}")

    def demo_file_operations(self):
        """演示文件操作工具"""
        print("\n" + "=" * 50)
        print("演示3: 文件操作工具")
        print("=" * 50)

        queries = [
            "创建一个名为test.txt的文件，内容是'Hello, Function Calling!'",
            "读取test.txt文件的内容",
            "在test.txt文件中添加一行'这是新增的内容'",
        ]

        for query in queries:
            print(f"\n用户: {query}")
            try:
                result = self.agent.invoke({"input": query, "chat_history": []})
                print(f"助手: {result['output']}")
            except Exception as e:
                print(f"错误: {e}")

    def demo_string_processing(self):
        """演示字符串处理工具"""
        print("\n" + "=" * 50)
        print("演示4: 字符串处理工具")
        print("=" * 50)

        queries = [
            "将'Hello World'转换为大写",
            "计算'LangChain Function Calling'的字符数和单词数",
            "将'Python'这个单词反转",
        ]

        for query in queries:
            print(f"\n用户: {query}")
            try:
                result = self.agent.invoke({"input": query, "chat_history": []})
                print(f"助手: {result['output']}")
            except Exception as e:
                print(f"错误: {e}")

    def demo_complex_task(self):
        """演示复杂任务（多工具组合使用）"""
        print("\n" + "=" * 50)
        print("演示5: 复杂任务（多工具组合）")
        print("=" * 50)

        query = """
        请帮我完成以下任务：
        1. 计算圆周率乘以10的结果
        2. 获取当前时间
        3. 将计算结果和时间信息写入一个名为result.txt的文件
        4. 然后读取这个文件确认内容
        """

        print(f"\n用户: {query}")
        try:
            result = self.agent.invoke({"input": query, "chat_history": []})
            print(f"助手: {result['output']}")
        except Exception as e:
            print(f"错误: {e}")

    def interactive_demo(self):
        """交互式演示"""
        print("\n" + "=" * 50)
        print("交互式演示 - 输入'quit'退出")
        print("=" * 50)

        chat_history = []

        while True:
            try:
                user_input = input("\n用户: ").strip()
                if user_input.lower() in ["quit", "exit", "退出"]:
                    break

                if not user_input:
                    continue

                result = self.agent.invoke(
                    {"input": user_input, "chat_history": chat_history}
                )

                print(f"助手: {result['output']}")

                # 更新对话历史
                chat_history.extend(
                    [
                        HumanMessage(content=user_input),
                        AIMessage(content=result["output"]),
                    ]
                )

            except KeyboardInterrupt:
                print("\n演示结束")
                break
            except Exception as e:
                print(f"错误: {e}")

    def run_all_demos(self):
        """运行所有演示"""
        print("开始基础工具调用演示...")

        try:
            self.demo_calculator()
            self.demo_time_query()
            self.demo_file_operations()
            self.demo_string_processing()
            self.demo_complex_task()

            print("\n" + "=" * 50)
            print("所有预设演示完成！")
            print("=" * 50)

            # 询问是否进入交互模式
            choice = input("\n是否进入交互模式？(y/n): ").strip().lower()
            if choice in ["y", "yes", "是"]:
                self.interactive_demo()

        except Exception as e:
            logger.error(f"演示运行失败: {e}")
            print(f"演示运行失败: {e}")


def main():
    """主函数"""
    try:
        print("基础工具调用演示")
        print("本演示将展示如何使用LangChain的基础工具调用功能")

        demo = BasicToolsDemo()
        demo.run_all_demos()

    except Exception as e:
        logger.error(f"程序运行失败: {e}")
        print(f"程序运行失败: {e}")


if __name__ == "__main__":
    main()
