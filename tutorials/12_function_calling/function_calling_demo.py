#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function Calling 综合演示

本文件演示了Function Calling的完整功能，包括：
1. 基础工具定义（计算器、时间查询、文件操作）
2. LangChain方式的Function Calling演示
3. DeepSeek原生API的Function Calling演示
4. 多工具组合使用示例
"""

import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from utils.config import load_qwen_config
except ImportError:
    print("警告：无法导入配置模块，将使用环境变量")

    def load_deepseek_config():
        return {
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
            "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        }


try:
    from openai import OpenAI
except ImportError:
    print("请安装OpenAI库：pip install openai")
    sys.exit(1)

try:
    from typing import Type

    from langchain.agents import AgentExecutor, create_openai_functions_agent
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.tools import BaseTool
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field
except ImportError:
    print("请安装LangChain库：pip install langchain langchain-openai")
    sys.exit(1)


class CalculatorTool(BaseTool):
    """计算器工具"""

    name: str = "calculator"
    description: str = "执行基本的数学计算，支持加减乘除和基本函数"

    class CalculatorInput(BaseModel):
        expression: str = Field(description="要计算的数学表达式")

    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        try:
            # 安全地计算数学表达式
            result = eval(expression)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"

    async def _arun(self, expression: str) -> str:
        return self._run(expression)


class TimeTool(BaseTool):
    """时间查询工具"""

    name: str = "time_query"
    description: str = "获取当前时间信息"

    class TimeInput(BaseModel):
        format: str = Field(
            description="时间格式：'current'获取当前时间，'date'获取日期，'time'获取时间"
        )

    args_schema: Type[BaseModel] = TimeInput

    def _run(self, format: str = "datetime") -> str:
        from datetime import datetime

        now = datetime.now()

        if format == "time":
            return f"当前时间: {now.strftime('%H:%M:%S')}"
        elif format == "date":
            return f"当前日期: {now.strftime('%Y-%m-%d')}"
        else:
            return f"当前日期时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"

    async def _arun(self, format: str = "current") -> str:
        return self._run(format)


class FileOperationTool(BaseTool):
    """文件操作工具"""

    name: str = "file_operation"
    description: str = "执行文件读写操作"

    class FileInput(BaseModel):
        operation: str = Field(description="操作类型：'read'读取文件，'write'写入文件")
        filename: str = Field(description="文件名")
        content: str = Field(
            default="", description="写入的内容（仅在write操作时需要）"
        )

    args_schema: Type[BaseModel] = FileInput

    def _run(self, operation: str, filename: str, content: str = "") -> str:
        try:
            # 获取当前工作目录并构建文件路径
            current_dir = os.getcwd()
            file_path = os.path.join(current_dir, filename)
            print(f"文件操作 - 操作类型: {operation}, 文件路径: {file_path}")

            # 安全检查：防止访问上级目录或绝对路径
            if ".." in filename or os.path.isabs(filename):
                return "错误：不允许访问上级目录或使用绝对路径"

            if operation == "read":
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    return f"文件内容：\n{content}"
                else:
                    return f"错误：文件 {filename} 不存在"

            elif operation == "write":
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return f"成功写入文件 {filename}"

            else:
                return f"错误：不支持的操作类型 {operation}"

        except Exception as e:
            return f"文件操作错误：{str(e)}"

    async def _arun(self, operation: str, filename: str, content: str = "") -> str:
        return self._run(operation, filename, content)


class FunctionCallingDemo:
    """Function Calling 演示类"""

    def __init__(self):
        """初始化演示"""
        self.config = load_qwen_config()
        if not self.config.get("api_key"):
            raise ValueError("请设置QWEN_API_KEY环境变量")

        # 初始化OpenAI客户端（用于Qwen API）
        self.client = OpenAI(
            api_key=self.config["api_key"], base_url=self.config["base_url"]
        )

        # 初始化LangChain工具
        self.tools = [CalculatorTool(), TimeTool(), FileOperationTool()]

        # 初始化LangChain模型
        self.llm = ChatOpenAI(
            model="qwen-plus",
            openai_api_key=self.config["api_key"],
            openai_api_base=self.config["base_url"],
            temperature=0.0,  # 降低temperature以获得更稳定的输出
            model_kwargs={"top_p": 0.1},
        )

        print("✅ Function Calling 演示初始化完成")
        print(f"🔧 已加载 {len(self.tools)} 个工具")
        print("📋 可用工具：计算器、时间查询、文件操作")

    def _custom_error_handler(self, error):
        """自定义错误处理函数"""
        error_msg = str(error)
        if "not valid JSON" in error_msg:
            return "抱歉，模型返回的格式有误，请重新尝试。"
        elif "timeout" in error_msg.lower():
            return "请求超时，请稍后重试。"
        else:
            return f"发生错误：{error_msg}"

    def demo_langchain_agent(self):
        """演示LangChain Agent方式的Function Calling"""
        print("\n" + "=" * 60)
        print("🚀 LangChain Agent 方式演示")
        print("=" * 60)

        # 创建系统提示
        system_prompt = """你是一个有用的AI助手，可以使用提供的工具来帮助用户。
        当用户提出问题时，请分析是否需要使用工具，如果需要，请调用相应的工具。
        请确保工具调用的参数格式正确，使用标准的JSON格式。
        请用中文回答用户的问题。"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # 创建Agent
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=self._custom_error_handler,
            max_iterations=3,  # 限制最大迭代次数
            early_stopping_method="generate",  # 早期停止策略
        )

        # 测试用例
        test_cases = [
            "计算 15 + 27 * 3 的结果",
            "现在几点了？",
            "帮我创建一个名为test.txt的文件，内容是'Hello Function Calling!'",
            "读取刚才创建的test.txt文件内容",
        ]

        for i, query in enumerate(test_cases, 1):
            print(f"\n📝 测试 {i}: {query}")
            print("-" * 40)
            try:
                result = agent_executor.invoke({"input": query})
                print(f"✅ 结果: {result['output']}")
            except Exception as e:
                print(f"❌ 错误: {str(e)}")

            time.sleep(1)  # 避免请求过快

    def demo_native_api(self):
        """演示DeepSeek原生API方式的Function Calling"""
        print("\n" + "=" * 60)
        print("🔥 DeepSeek 原生 API 方式演示")
        print("=" * 60)

        # 定义工具函数
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "执行基本数学计算",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "数学表达式，如 '2+3' 或 '15+27*3'",
                            }
                        },
                        "required": ["expression"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "获取当前时间信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "format": {
                                "type": "string",
                                "description": "时间格式：datetime, date, time",
                                "enum": ["datetime", "date", "time"],
                                "default": "datetime",
                            }
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            },
        ]

        # 测试用例
        test_queries = ["请计算 sqrt(144) + 10 * 2", "现在是什么时间？"]

        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 测试 {i}: {query}")
            print("-" * 40)

            try:
                # 第一次调用：获取工具调用请求
                response = self.client.chat.completions.create(
                    model="qwen-plus",
                    messages=[{"role": "user", "content": query}],
                    tools=tools,
                    tool_choice="auto",
                )

                message = response.choices[0].message

                if message.tool_calls:
                    print(f"🔧 模型选择调用工具: {message.tool_calls[0].function.name}")

                    # 执行工具调用
                    tool_call = message.tool_calls[0]
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    print(f"📋 工具参数: {function_args}")

                    # 执行相应的工具函数
                    if function_name == "calculator":
                        tool_result = self._execute_calculator(
                            function_args["expression"]
                        )
                    elif function_name == "get_current_time":
                        tool_result = self._execute_time_query(
                            function_args.get("format", "datetime")
                        )
                    else:
                        tool_result = "未知的工具函数"

                    print(f"⚙️ 工具执行结果: {tool_result}")

                    # 第二次调用：整合工具结果
                    messages = [
                        {"role": "user", "content": query},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tool_call.dict()],
                        },
                        {
                            "role": "tool",
                            "content": tool_result,
                            "tool_call_id": tool_call.id,
                        },
                    ]

                    final_response = self.client.chat.completions.create(
                        model="qwen-plus", messages=messages
                    )

                    print(f"✅ 最终回答: {final_response.choices[0].message.content}")
                else:
                    print(f"💬 直接回答: {message.content}")

            except Exception as e:
                print(f"❌ 错误: {str(e)}")

            time.sleep(1)

    def _execute_calculator(self, expression: str) -> str:
        """执行计算器工具"""
        calc_tool = CalculatorTool()
        return calc_tool._run(expression)

    def _execute_time_query(self, format_type: str) -> str:
        """执行时间查询工具"""
        time_tool = TimeTool()
        return time_tool._run(format_type)

    def demo_multi_tool_combination(self):
        """演示多工具组合使用"""
        print("\n" + "=" * 60)
        print("🎯 多工具组合使用演示")
        print("=" * 60)

        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个有用的助手，可以使用提供的工具来帮助用户。\n"
                    "重要：在调用工具时，必须确保JSON格式完全正确，所有字符串都要用双引号包围，不能有语法错误。\n"
                    '例如：{"expression": "15 + 27 * 3"}\n'
                    "如果遇到JSON解析错误，请检查格式并重新生成完整正确的JSON。",
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # 创建Agent
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True
        )

        # 简化的任务列表
        tasks = [
            "计算 (25 + 15) * 3 的结果",
            "创建一个名为report.txt的文件，内容是'计算结果：120，当前时间：2025-08-21'",
            "读取report.txt文件的内容",
        ]

        for i, task in enumerate(tasks, 1):
            print(f"\n📋 任务 {i}: {task}")
            print("-" * 40)
            try:
                result = agent_executor.invoke({"input": task})
                print(f"✅ 结果: {result['output']}")
            except Exception as e:
                print(f"❌ 错误: {str(e)}")

            time.sleep(1)  # 避免请求过快

    def run_all_demos(self):
        """运行所有演示"""
        print("🎉 Function Calling 综合演示开始")
        print("本演示将展示LangChain和DeepSeek原生API两种方式的Function Calling")

        try:
            # 1. LangChain Agent演示
            self.demo_langchain_agent()

            # 2. DeepSeek原生API演示
            self.demo_native_api()

            # 3. 多工具组合演示
            self.demo_multi_tool_combination()

            print("\n" + "=" * 60)
            print("🎊 所有演示完成！")
            print("=" * 60)
            print("\n📚 总结：")
            print("1. LangChain Agent 提供了高级的工具管理和对话流程")
            print("2. DeepSeek 原生 API 提供了更直接的控制和自定义能力")
            print("3. 两种方式都支持复杂的多工具组合使用")
            print("4. Function Calling 大大扩展了AI助手的实际应用能力")

        except Exception as e:
            print(f"❌ 演示过程中出现错误: {str(e)}")


def main():
    """主函数"""
    try:
        demo = FunctionCallingDemo()
        demo.run_all_demos()
    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")
        print("请检查：")
        print("1. DEEPSEEK_API_KEY 环境变量是否设置")
        print("2. 网络连接是否正常")
        print("3. 依赖库是否正确安装")


if __name__ == "__main__":
    main()
