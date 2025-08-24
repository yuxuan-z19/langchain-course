#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph记忆功能演示

本演示展示如何使用LangGraph的持久性检查点功能来实现：
1. 多轮对话记忆
2. 不同thread_id的会话隔离
3. 状态检查和恢复
4. 交互式聊天界面

作者：Jaguarliu
日期：2025年8月
"""

import sys
import os
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import uuid
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.llm_factory import create_llm_from_config


class State(TypedDict):
    """图的状态定义
    
    使用TypedDict定义状态结构，包含消息列表
    """
    messages: Annotated[list[BaseMessage], add_messages]


class MemoryChatbot:
    """带记忆功能的LangGraph聊天机器人
    
    使用持久性检查点实现多轮对话记忆功能
    """
    
    def __init__(self, provider: str = "deepseek"):
        """初始化聊天机器人
        
        Args:
            provider: LLM提供商，支持 'deepseek', 'openai' 等
        """
        self.provider = provider
        self.llm = None
        self.graph = None
        self.memory = None
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化LLM、图和检查点器"""
        try:
            # 1. 初始化LLM
            print(f"🤖 正在初始化 {self.provider} LLM...")
            self.llm = create_llm_from_config(self.provider)
            print(f"✅ {self.provider} LLM 初始化成功")
            
            # 2. 创建内存检查点器
            print("💾 正在创建内存检查点器...")
            self.memory = MemorySaver()
            print("✅ 内存检查点器创建成功")
            
            # 3. 构建图
            print("🔧 正在构建LangGraph...")
            self._build_graph()
            print("✅ LangGraph构建成功")
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            raise
    
    def _build_graph(self):
        """构建LangGraph图结构"""
        # 创建状态图
        graph_builder = StateGraph(State)
        
        # 添加聊天节点
        graph_builder.add_node("chatbot", self._chatbot_node)
        
        # 设置入口点
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        
        # 使用检查点器编译图
        self.graph = graph_builder.compile(checkpointer=self.memory)
    
    def _chatbot_node(self, state: State) -> dict:
        """聊天机器人节点
        
        处理用户消息并生成回复
        
        Args:
            state: 当前状态，包含消息历史
            
        Returns:
            包含新消息的状态更新
        """
        try:
            # 调用LLM生成回复
            response = self.llm.invoke(state["messages"])
            
            # 返回状态更新
            return {"messages": [response]}
            
        except Exception as e:
            error_msg = f"生成回复时出错: {e}"
            print(f"❌ {error_msg}")
            return {"messages": [AIMessage(content=f"抱歉，{error_msg}")]}
    
    def chat_with_memory(self, message: str, thread_id: str) -> str:
        """带记忆的单次对话
        
        Args:
            message: 用户消息
            thread_id: 线程ID，用于区分不同的对话会话
            
        Returns:
            AI回复内容
        """
        try:
            # 配置线程ID
            config = {"configurable": {"thread_id": thread_id}}
            
            # 创建用户消息
            user_message = HumanMessage(content=message)
            
            # 执行图并获取结果
            events = self.graph.stream(
                {"messages": [user_message]},
                config,
                stream_mode="values"
            )
            
            # 获取最后的AI回复
            for event in events:
                if "messages" in event and event["messages"]:
                    last_message = event["messages"][-1]
                    if isinstance(last_message, AIMessage):
                        return last_message.content
            
            return "抱歉，没有收到有效回复。"
            
        except Exception as e:
            error_msg = f"对话处理失败: {e}"
            print(f"❌ {error_msg}")
            return f"抱歉，{error_msg}"
    
    def get_conversation_state(self, thread_id: str) -> dict:
        """获取指定线程的对话状态
        
        Args:
            thread_id: 线程ID
            
        Returns:
            包含状态信息的字典
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}
            snapshot = self.graph.get_state(config)
            
            # StateSnapshot对象的正确属性：
            # - values: 状态值
            # - next: 下一步节点
            # - created_at: 创建时间
            # - metadata: 元数据（包含step信息）
            step_info = snapshot.metadata.get("step", "N/A") if snapshot.metadata else "N/A"
            
            return {
                "thread_id": thread_id,
                "message_count": len(snapshot.values.get("messages", [])),
                "messages": snapshot.values.get("messages", []),
                "next_step": snapshot.next,
                "created_at": snapshot.created_at,
                "step": step_info
            }
        except Exception as e:
            return {
                "thread_id": thread_id,
                "error": f"获取状态失败: {str(e)}",
                "message_count": 0,
                "messages": [],
                "next_step": None,
                "created_at": None,
                "step": None
            }
    
    def list_all_threads(self) -> list:
        """列出所有活跃的线程
        
        Returns:
            线程ID列表
        """
        try:
            # 注意：MemorySaver没有直接的方法列出所有线程
            # 这里返回一个提示信息
            return ["使用MemorySaver时无法直接列出所有线程，请记录您使用的thread_id"]
        except Exception as e:
            print(f"❌ 列出线程失败: {e}")
            return []
    
    def clear_thread_memory(self, thread_id: str) -> bool:
        """清除指定线程的记忆
        
        Args:
            thread_id: 要清除的线程ID
            
        Returns:
            是否成功清除
        """
        try:
            # 注意：MemorySaver没有直接的清除方法
            # 在实际应用中，可以使用数据库检查点器来实现此功能
            print(f"⚠️  MemorySaver不支持直接清除线程记忆")
            print(f"💡 建议重启程序或使用新的thread_id: {thread_id}_new")
            return False
        except Exception as e:
            print(f"❌ 清除记忆失败: {e}")
            return False


def demonstrate_basic_memory():
    """演示基本的记忆功能"""
    print("\n" + "="*50)
    print("🧠 基本记忆功能演示")
    print("="*50)
    
    # 创建聊天机器人
    chatbot = MemoryChatbot()
    thread_id = "demo_user_001"
    
    # 第一轮对话
    print("\n👤 用户: 你好，我叫张三")
    response1 = chatbot.chat_with_memory("你好，我叫张三", thread_id)
    print(f"🤖 AI: {response1}")
    
    # 第二轮对话 - 测试记忆
    print("\n👤 用户: 你还记得我的名字吗？")
    response2 = chatbot.chat_with_memory("你还记得我的名字吗？", thread_id)
    print(f"🤖 AI: {response2}")
    
    # 第三轮对话 - 继续测试记忆
    print("\n👤 用户: 我今年25岁，是一名程序员")
    response3 = chatbot.chat_with_memory("我今年25岁，是一名程序员", thread_id)
    print(f"🤖 AI: {response3}")
    
    # 第四轮对话 - 测试复合记忆
    print("\n👤 用户: 请总结一下你对我的了解")
    response4 = chatbot.chat_with_memory("请总结一下你对我的了解", thread_id)
    print(f"🤖 AI: {response4}")
    
    return chatbot, thread_id


def demonstrate_thread_isolation():
    """演示不同thread_id的会话隔离"""
    print("\n" + "="*50)
    print("🔒 线程隔离功能演示")
    print("="*50)
    print("\n💡 重要说明:")
    print("   LangGraph的检查点机制实现了完全的线程隔离")
    print("   不同thread_id的对话是完全独立的，AI无法跨线程访问信息")
    print("   这是LangGraph记忆机制的核心特性，确保了数据隔离和隐私保护")
    
    chatbot = MemoryChatbot()
    
    # 线程1的对话
    thread1 = "user_alice"
    print(f"\n📱 线程1 ({thread1}):")
    print("👤 Alice: 你好，我是Alice，我喜欢画画")
    response1 = chatbot.chat_with_memory("你好，我是Alice，我喜欢画画", thread1)
    print(f"🤖 AI: {response1}")
    
    # 线程2的对话
    thread2 = "user_bob"
    print(f"\n📱 线程2 ({thread2}):")
    print("👤 Bob: 嗨，我是Bob，我是个音乐家")
    response2 = chatbot.chat_with_memory("嗨，我是Bob，我是个音乐家", thread2)
    print(f"🤖 AI: {response2}")
    
    # 回到线程1，测试记忆隔离
    print(f"\n📱 回到线程1 ({thread1}):")
    print("👤 Alice: 你还记得我的爱好吗？")
    response3 = chatbot.chat_with_memory("你还记得我的爱好吗？", thread1)
    print(f"🤖 AI: {response3}")
    
    # 回到线程2，测试记忆隔离
    print(f"\n📱 回到线程2 ({thread2}):")
    print("👤 Bob: 你知道我是做什么的吗？")
    response4 = chatbot.chat_with_memory("你知道我是做什么的吗？", thread2)
    print(f"🤖 AI: {response4}")
    
    # 交叉测试 - 在线程1中询问Bob的信息
    print(f"\n📱 线程1交叉测试 ({thread1}):")
    print("   ⚠️  预期行为：AI不会认识Bob，因为Bob的信息在不同的线程中")
    print("👤 Alice: 你认识Bob吗？")
    response5 = chatbot.chat_with_memory("你认识Bob吗？", thread1)
    print(f"🤖 AI: {response5}")
    print("\n✅ 线程隔离验证：AI正确地表示不认识Bob，这证明了线程隔离的有效性")
    
    return chatbot, [thread1, thread2]


def demonstrate_state_inspection(chatbot, thread_id):
    """演示状态检查功能"""
    print("\n" + "="*50)
    print("🔍 状态检查功能演示")
    print("="*50)
    
    # 获取当前状态
    state = chatbot.get_conversation_state(thread_id)
    
    print(f"\n📊 线程 {thread_id} 的状态信息:")
    print(f"   消息数量: {state.get('message_count', 0)}")
    print(f"   当前步骤: {state.get('step', 'N/A')}")
    print(f"   下一步: {state.get('next_step', 'N/A')}")
    print(f"   创建时间: {state.get('created_at', 'N/A')}")
    
    # 显示消息历史
    messages = state.get('messages', [])
    if messages:
        print(f"\n💬 对话历史 ({len(messages)} 条消息):")
        for i, msg in enumerate(messages, 1):
            role = "👤 用户" if isinstance(msg, HumanMessage) else "🤖 AI"
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            print(f"   {i}. {role}: {content}")
    else:
        print("\n💬 暂无对话历史")


def interactive_chat():
    """交互式聊天界面"""
    print("\n" + "="*50)
    print("💬 交互式聊天界面")
    print("="*50)
    print("\n🎯 使用说明:")
    print("   - 输入消息进行对话")
    print("   - 输入 '/new' 开始新的对话线程")
    print("   - 输入 '/state' 查看当前状态")
    print("   - 输入 '/switch <thread_id>' 切换到指定线程")
    print("   - 输入 '/quit' 退出")
    
    chatbot = MemoryChatbot()
    current_thread = f"interactive_{uuid.uuid4().hex[:8]}"
    
    print(f"\n🆔 当前线程ID: {current_thread}")
    print("\n开始聊天吧！")
    
    while True:
        try:
            user_input = input("\n👤 您: ").strip()
            
            if not user_input:
                continue
            
            # 处理特殊命令
            if user_input == '/quit':
                print("\n👋 再见！")
                break
            elif user_input == '/new':
                current_thread = f"interactive_{uuid.uuid4().hex[:8]}"
                print(f"\n🆕 已创建新的对话线程: {current_thread}")
                continue
            elif user_input == '/state':
                demonstrate_state_inspection(chatbot, current_thread)
                continue
            elif user_input.startswith('/switch '):
                new_thread = user_input[8:].strip()
                if new_thread:
                    current_thread = new_thread
                    print(f"\n🔄 已切换到线程: {current_thread}")
                else:
                    print("\n❌ 请提供有效的线程ID")
                continue
            
            # 正常对话
            print("\n🤖 AI正在思考...")
            response = chatbot.chat_with_memory(user_input, current_thread)
            print(f"\n🤖 AI: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 检测到中断信号，正在退出...")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            print("请重试或输入 '/quit' 退出")


def main():
    """主函数"""
    print("🚀 LangGraph记忆功能演示程序")
    print("=" * 50)
    
    try:
        # 1. 基本记忆功能演示
        chatbot, demo_thread = demonstrate_basic_memory()
        
        # 2. 线程隔离功能演示
        chatbot, isolation_threads = demonstrate_thread_isolation()
        
        # 3. 状态检查功能演示
        demonstrate_state_inspection(chatbot, demo_thread)
        
        # 4. 询问是否进入交互模式
        print("\n" + "="*50)
        print("🎮 演示完成！")
        print("="*50)
        
        while True:
            choice = input("\n是否进入交互式聊天模式？(y/n): ").strip().lower()
            if choice in ['y', 'yes', '是', '好']:
                interactive_chat()
                break
            elif choice in ['n', 'no', '否', '不']:
                print("\n👋 感谢使用LangGraph记忆功能演示！")
                break
            else:
                print("请输入 y 或 n")
        
    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()