#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph工具集成演示

本脚本演示如何在LangGraph中集成各种工具，包括：
1. Tavily搜索引擎工具
2. 计算器工具
3. 时间查询工具
4. 工具绑定和调用流程
5. 条件边和状态管理
6. 错误处理和可视化

作者: Jaguarliu
日期: 2025年 8 月
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from typing import Annotated, Literal

from typing_extensions import TypedDict

# 添加项目根目录到Python路径
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

try:
    from dotenv import load_dotenv
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.tools import tool
    from langchain_tavily import TavilySearch
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode

    # 导入自定义的LLM工厂
    from utils.llm_factory import create_llm_from_config
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保安装了所需的依赖包：")
    print("pip install langchain langgraph langchain-tavily python-dotenv")
    sys.exit(1)

# 加载环境变量
load_dotenv()


class State(TypedDict):
    """定义图的状态结构"""

    messages: Annotated[list, add_messages]


@tool
def calculator(expression: str) -> str:
    """
    计算数学表达式的结果

    Args:
        expression: 要计算的数学表达式，如 "2 + 3 * 4"

    Returns:
        计算结果的字符串表示
    """
    try:
        # 安全的数学表达式计算
        # 只允许基本的数学运算和函数
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})

        # 计算表达式
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def get_current_time(timezone_name: str = "UTC") -> str:
    """
    获取当前时间

    Args:
        timezone_name: 时区名称，默认为UTC

    Returns:
        当前时间的字符串表示
    """
    try:
        if timezone_name.upper() == "UTC":
            current_time = datetime.now(timezone.utc)
            return f"当前UTC时间: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        elif timezone_name.upper() in ["CST", "CHINA", "BEIJING"]:
            # 中国标准时间 (UTC+8)
            import pytz

            china_tz = pytz.timezone("Asia/Shanghai")
            current_time = datetime.now(china_tz)
            return f"当前北京时间: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        else:
            current_time = datetime.now()
            return f"当前本地时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
    except Exception as e:
        current_time = datetime.now()
        return f"当前时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')} (时区解析失败: {str(e)})"


@tool
def weather_info(location: str) -> str:
    """
    获取天气信息（模拟工具）

    Args:
        location: 地点名称

    Returns:
        天气信息字符串
    """
    # 这是一个模拟的天气工具
    # 在实际应用中，你可以集成真实的天气API
    import random

    weather_conditions = ["晴朗", "多云", "阴天", "小雨", "大雨", "雪"]
    temperature = random.randint(-10, 35)
    condition = random.choice(weather_conditions)

    return f"{location}的天气: {condition}, 温度: {temperature}°C (注意: 这是模拟数据)"


class ToolsChatbot:
    """
    集成工具的LangGraph聊天机器人
    """

    def __init__(self, provider: str = "deepseek"):
        """
        初始化聊天机器人

        Args:
            provider: LLM提供商，默认为deepseek
        """
        self.provider = provider
        self.llm = None
        self.tools = []
        self.graph = None
        self.memory = MemorySaver()

        # 初始化组件
        self._initialize_llm()
        self._initialize_tools()
        self._build_graph()

    def _initialize_llm(self):
        """
        初始化大语言模型
        """
        try:
            print(f"正在初始化{self.provider}模型...")
            self.llm = create_llm_from_config(self.provider)
            print(f"✅ {self.provider}模型初始化成功")
        except Exception as e:
            print(f"❌ LLM初始化失败: {e}")
            raise

    def _initialize_tools(self):
        """
        初始化工具集合
        """
        try:
            print("正在初始化工具...")

            # 1. Tavily搜索工具
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            if tavily_api_key:
                try:
                    tavily_search = TavilySearch(
                        max_results=3,
                        search_depth="advanced",
                        include_answer=True,
                        include_raw_content=False,
                    )
                    self.tools.append(tavily_search)
                    print("✅ Tavily搜索工具初始化成功")
                except Exception as e:
                    print(f"⚠️ Tavily搜索工具初始化失败: {e}")
            else:
                print("⚠️ 未找到TAVILY_API_KEY，跳过Tavily搜索工具")

            # 2. 计算器工具
            self.tools.append(calculator)
            print("✅ 计算器工具初始化成功")

            # 3. 时间查询工具
            self.tools.append(get_current_time)
            print("✅ 时间查询工具初始化成功")

            # 4. 天气工具（模拟）
            self.tools.append(weather_info)
            print("✅ 天气查询工具初始化成功")

            print(f"总共初始化了 {len(self.tools)} 个工具")

        except Exception as e:
            print(f"❌ 工具初始化失败: {e}")
            raise

    def _build_graph(self):
        """
        构建LangGraph图
        """
        try:
            print("正在构建LangGraph图...")

            # 创建状态图
            graph_builder = StateGraph(State)

            # 绑定工具到LLM
            llm_with_tools = self.llm.bind_tools(self.tools)

            # 定义聊天机器人节点
            def chatbot_node(state: State):
                """
                聊天机器人节点：处理用户消息并决定是否需要调用工具
                """
                try:
                    response = llm_with_tools.invoke(state["messages"])
                    return {"messages": [response]}
                except Exception as e:
                    error_msg = AIMessage(
                        content=f"抱歉，处理您的请求时出现错误: {str(e)}"
                    )
                    return {"messages": [error_msg]}

            # 创建工具节点
            tool_node = ToolNode(self.tools)

            # 定义条件边函数
            def should_continue(state: State) -> Literal["tools", "__end__"]:
                """
                判断是否需要继续到工具节点

                Returns:
                    "tools": 需要调用工具
                    "__end__": 结束对话
                """
                messages = state["messages"]
                last_message = messages[-1]

                # 检查最后一条消息是否包含工具调用
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    return "tools"
                return "__end__"

            # 添加节点
            graph_builder.add_node("chatbot", chatbot_node)
            graph_builder.add_node("tools", tool_node)

            # 添加边
            graph_builder.add_edge(START, "chatbot")
            graph_builder.add_conditional_edges(
                "chatbot", should_continue, {"tools": "tools", "__end__": END}
            )
            graph_builder.add_edge("tools", "chatbot")

            # 编译图
            self.graph = graph_builder.compile(checkpointer=self.memory)
            print("✅ LangGraph图构建成功")

        except Exception as e:
            print(f"❌ 图构建失败: {e}")
            raise

    def visualize_graph(self, output_path: str = "tools_chatbot_graph.png"):
        """
        可视化图结构

        Args:
            output_path: 输出图片路径
        """
        try:
            # 尝试生成图的可视化
            graph_image = self.graph.get_graph().draw_mermaid_png()

            # 保存图片
            with open(output_path, "wb") as f:
                f.write(graph_image)

            print(f"✅ 图结构已保存到: {output_path}")

        except Exception as e:
            print(f"⚠️ 图可视化失败: {e}")
            print("提示: 可能需要安装graphviz: pip install graphviz")

            # 输出文本版本的图结构
            print("\n📊 图结构 (文本版本):")
            print("节点:")
            print("  - chatbot: 处理用户消息，决定是否需要工具")
            print("  - tools: 执行工具调用")
            print("边:")
            print("  - START → chatbot")
            print("  - chatbot → tools (条件: 需要工具调用)")
            print("  - chatbot → END (条件: 不需要工具调用)")
            print("  - tools → chatbot")

    def chat(self, message: str, thread_id: str = "default") -> str:
        """
        处理单条消息

        Args:
            message: 用户消息
            thread_id: 对话线程ID

        Returns:
            助手回复
        """
        try:
            # 创建配置
            config = {"configurable": {"thread_id": thread_id}}

            # 调用图
            result = self.graph.invoke(
                {"messages": [HumanMessage(content=message)]}, config=config
            )

            # 获取最后一条AI消息
            messages = result["messages"]
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    return msg.content

            return "抱歉，我无法处理您的请求。"

        except Exception as e:
            return f"处理消息时出现错误: {str(e)}"

    def interactive_chat(self):
        """
        启动交互式聊天界面
        """
        print("\n🤖 LangGraph工具集成聊天机器人")
        print("=" * 50)
        print("可用功能:")
        print("  🔍 网页搜索 (如果配置了Tavily API)")
        print("  🧮 数学计算")
        print("  ⏰ 时间查询")
        print("  🌤️ 天气查询 (模拟数据)")
        print("\n输入 'quit' 或 'exit' 退出")
        print("输入 'help' 查看帮助")
        print("输入 'tools' 查看可用工具")
        print("=" * 50)

        thread_id = "interactive_session"

        while True:
            try:
                user_input = input("\n👤 您: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit", "退出"]:
                    print("\n👋 再见！")
                    break

                if user_input.lower() == "help":
                    self._show_help()
                    continue

                if user_input.lower() == "tools":
                    self._show_tools()
                    continue

                print("\n🤖 助手: ", end="")
                response = self.chat(user_input, thread_id)
                print(response)

            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")

    def _show_help(self):
        """
        显示帮助信息
        """
        print("\n📖 帮助信息:")
        print("\n🔍 搜索示例:")
        print("  - 今天的新闻")
        print("  - Python编程教程")
        print("  - 人工智能最新发展")

        print("\n🧮 计算示例:")
        print("  - 计算 2 + 3 * 4")
        print("  - sqrt(16) + log(10)")
        print("  - sin(pi/2)")

        print("\n⏰ 时间查询示例:")
        print("  - 现在几点了？")
        print("  - 北京时间")
        print("  - UTC时间")

        print("\n🌤️ 天气查询示例:")
        print("  - 北京的天气")
        print("  - 上海天气如何")

    def _show_tools(self):
        """
        显示可用工具
        """
        print("\n🛠️ 可用工具:")
        for i, tool in enumerate(self.tools, 1):
            tool_name = getattr(tool, "name", str(tool))
            tool_desc = getattr(tool, "description", "无描述")
            print(f"  {i}. {tool_name}: {tool_desc}")


def demo_tool_calls():
    """
    演示工具调用功能
    """
    print("\n🚀 LangGraph工具集成演示")
    print("=" * 50)

    try:
        # 创建聊天机器人
        chatbot = ToolsChatbot()

        # 可视化图结构
        chatbot.visualize_graph()

        # 演示不同类型的查询
        test_queries = [
            "计算 15 * 23 + 7 的结果",
            "现在几点了？",
            "北京的天气怎么样？",
        ]

        # 如果有Tavily API，添加搜索查询
        if os.getenv("TAVILY_API_KEY"):
            test_queries.append("Python编程的最新发展")

        print("\n📝 自动测试查询:")
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. 测试查询: {query}")
            print("-" * 30)
            response = chatbot.chat(query, f"demo_thread_{i}")
            print(f"回复: {response}")

        print("\n" + "=" * 50)
        print("🎯 演示完成！现在可以开始交互式聊天...")

        # 启动交互式聊天
        chatbot.interactive_chat()

    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    demo_tool_calls()
