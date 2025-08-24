#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph基础聊天机器人演示

本演示展示了如何使用LangGraph构建一个基础聊天机器人，包括：
1. StateGraph的创建和配置
2. 状态管理和消息处理
3. 节点和边的定义
4. 图的编译和运行
5. 图的可视化
6. 交互式聊天界面

作者：Jaguarliu
日期：2025年8月
"""

import os
import sys
from typing import Annotated
from typing_extensions import TypedDict

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError as e:
    print(f"❌ 导入LangGraph失败: {e}")
    print("请安装LangGraph: pip install langgraph")
    sys.exit(1)

from utils.llm_factory import create_llm_from_config
from utils.config import load_environment


class State(TypedDict):
    """
    聊天机器人的状态定义
    
    messages: 对话消息列表，使用add_messages函数处理状态更新
              新消息会追加到列表中，而不是覆盖现有消息
    """
    messages: Annotated[list, add_messages]


class BasicChatbot:
    """
    基础聊天机器人类
    
    使用LangGraph构建的简单聊天机器人，展示StateGraph的基本用法
    """
    
    def __init__(self, llm_provider="deepseek"):
        """
        初始化聊天机器人
        
        Args:
            llm_provider: LLM提供商，默认为deepseek
        """
        self.llm_provider = llm_provider
        self.llm = None
        self.graph = None
        self._initialize_llm()
        self._build_graph()
    
    def _initialize_llm(self):
        """
        初始化LLM模型
        复用utils.llm_factory中的LLM初始化逻辑
        """
        try:
            # 加载配置
            config = load_environment()
            
            # 创建LLM实例
            self.llm = create_llm_from_config(self.llm_provider)
            print(f"✅ 成功初始化{self.llm_provider} LLM")
            
        except Exception as e:
            print(f"❌ LLM初始化失败: {e}")
            raise
    
    def _chatbot_node(self, state: State):
        """
        聊天机器人节点函数
        
        这是图中的核心节点，负责处理用户消息并生成回复
        
        Args:
            state: 当前状态，包含消息历史
            
        Returns:
            dict: 包含新消息的状态更新
        """
        try:
            # 获取当前消息列表
            messages = state["messages"]
            
            # 调用LLM生成回复
            response = self.llm.invoke(messages)
            
            # 返回状态更新，新消息会被add_messages函数追加到列表中
            return {"messages": [response]}
            
        except Exception as e:
            print(f"❌ 聊天机器人节点处理失败: {e}")
            # 返回错误消息
            error_message = AIMessage(content=f"抱歉，处理您的消息时出现错误: {str(e)}")
            return {"messages": [error_message]}
    
    def _build_graph(self):
        """
        构建StateGraph
        
        创建状态图，添加节点和边，然后编译图
        """
        try:
            # 1. 创建StateGraph实例
            graph_builder = StateGraph(State)
            
            # 2. 添加聊天机器人节点
            # 第一个参数是节点名称，第二个参数是节点函数
            graph_builder.add_node("chatbot", self._chatbot_node)
            
            # 3. 添加入口点
            # 指定图从哪个节点开始执行
            graph_builder.add_edge(START, "chatbot")
            
            # 4. 添加结束边
            # 聊天机器人处理完消息后结束
            graph_builder.add_edge("chatbot", END)
            
            # 5. 编译图
            # 编译后的图可以被调用执行
            self.graph = graph_builder.compile()
            
            print("✅ StateGraph构建成功")
            
        except Exception as e:
            print(f"❌ StateGraph构建失败: {e}")
            raise
    
    def chat(self, message: str) -> str:
        """
        单次对话
        
        Args:
            message: 用户输入的消息
            
        Returns:
            str: 机器人的回复
        """
        try:
            # 创建用户消息
            user_message = HumanMessage(content=message)
            
            # 调用图执行
            result = self.graph.invoke({"messages": [user_message]})
            
            # 获取最后一条消息（机器人的回复）
            last_message = result["messages"][-1]
            
            return last_message.content
            
        except Exception as e:
            return f"处理消息时出现错误: {str(e)}"
    
    def chat_with_history(self, messages: list) -> dict:
        """
        带历史记录的对话
        
        Args:
            messages: 消息历史列表
            
        Returns:
            dict: 包含完整消息历史的状态
        """
        try:
            # 调用图执行，传入完整的消息历史
            result = self.graph.invoke({"messages": messages})
            return result
            
        except Exception as e:
            print(f"❌ 对话处理失败: {e}")
            return {"messages": messages}
    
    def visualize_graph(self, output_path: str = None):
        """
        可视化图结构
        
        Args:
            output_path: 输出文件路径，如果为None则只打印ASCII图
        """
        try:
            if self.graph is None:
                print("❌ 图尚未构建")
                return
            
            # 获取图对象
            graph_obj = self.graph.get_graph()
            
            # 打印ASCII图
            print("\n📊 图结构（ASCII）:")
            print(graph_obj.draw_ascii())
            
            # 如果指定了输出路径，尝试生成PNG图像
            if output_path:
                try:
                    # 尝试生成PNG图像（需要安装graphviz）
                    png_data = graph_obj.draw_png()
                    with open(output_path, "wb") as f:
                        f.write(png_data)
                    print(f"✅ 图像已保存到: {output_path}")
                except Exception as e:
                    print(f"⚠️  PNG图像生成失败: {e}")
                    print("提示：安装graphviz可以生成PNG图像: pip install graphviz")
            
        except Exception as e:
            print(f"❌ 图可视化失败: {e}")
    
    def interactive_chat(self):
        """
        交互式聊天界面
        
        提供命令行交互式聊天体验
        """
        print("\n🤖 LangGraph基础聊天机器人")
        print("=" * 50)
        print("输入消息开始对话，输入 'quit' 或 'exit' 退出")
        print("输入 'clear' 清空对话历史")
        print("输入 'history' 查看对话历史")
        print("输入 'graph' 查看图结构")
        print("=" * 50)
        
        # 初始化消息历史
        messages = []
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n👤 您: ").strip()
                
                # 处理特殊命令
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 再见！")
                    break
                elif user_input.lower() == 'clear':
                    messages = []
                    print("🧹 对话历史已清空")
                    continue
                elif user_input.lower() == 'history':
                    print("\n📜 对话历史:")
                    for i, msg in enumerate(messages, 1):
                        role = "👤" if isinstance(msg, HumanMessage) else "🤖"
                        print(f"{i}. {role} {msg.content}")
                    continue
                elif user_input.lower() == 'graph':
                    self.visualize_graph()
                    continue
                elif not user_input:
                    continue
                
                # 创建用户消息
                user_message = HumanMessage(content=user_input)
                messages.append(user_message)
                
                # 获取机器人回复
                result = self.chat_with_history(messages)
                messages = result["messages"]
                
                # 显示机器人回复
                bot_reply = messages[-1].content
                print(f"🤖 机器人: {bot_reply}")
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 处理输入时出现错误: {e}")


def demo_basic_usage():
    """
    基础用法演示
    """
    print("\n🚀 LangGraph基础聊天机器人演示")
    print("=" * 60)
    
    try:
        # 创建聊天机器人实例
        print("\n1. 初始化聊天机器人...")
        chatbot = BasicChatbot()
        
        # 可视化图结构
        print("\n2. 图结构可视化:")
        chatbot.visualize_graph()
        
        # 单次对话演示
        print("\n3. 单次对话演示:")
        response = chatbot.chat("你好，请介绍一下LangGraph")
        print(f"👤 用户: 你好，请介绍一下LangGraph")
        print(f"🤖 机器人: {response}")
        
        # 带历史记录的对话演示
        print("\n4. 带历史记录的对话演示:")
        messages = [
            HumanMessage(content="我想学习Python编程"),
        ]
        
        result = chatbot.chat_with_history(messages)
        messages = result["messages"]
        
        print(f"👤 用户: 我想学习Python编程")
        print(f"🤖 机器人: {messages[-1].content}")
        
        # 继续对话
        messages.append(HumanMessage(content="有什么好的学习资源推荐吗？"))
        result = chatbot.chat_with_history(messages)
        messages = result["messages"]
        
        print(f"👤 用户: 有什么好的学习资源推荐吗？")
        print(f"🤖 机器人: {messages[-1].content}")
        
        return chatbot
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        return None


def main():
    """
    主函数
    """
    try:
        # 运行基础演示
        chatbot = demo_basic_usage()
        
        if chatbot:
            # 询问是否进入交互模式
            print("\n" + "=" * 60)
            choice = input("是否进入交互式聊天模式？(y/n): ").strip().lower()
            
            if choice in ['y', 'yes', '是']:
                chatbot.interactive_chat()
            else:
                print("👋 演示结束！")
        
    except Exception as e:
        print(f"❌ 程序运行失败: {e}")


if __name__ == "__main__":
    main()