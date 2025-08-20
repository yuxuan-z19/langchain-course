#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain 记忆功能演示

本脚本演示了如何使用 LangChain 的 InMemoryHistory 实现基础的短时记忆功能，
包括基础记忆操作、带记忆的聊天机器人、多轮对话和记忆管理等功能。

主要功能：
1. 基础 InMemoryHistory 使用
2. 带记忆的聊天机器人实现
3. 对话历史管理（添加、查看、清除）
4. 记忆容量限制演示
5. 多轮对话演示
6. 上下文保持能力展示
"""

import sys
import os
from typing import List, Optional
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from langchain.memory import ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain.chat_models import ChatOpenAI
from utils.config import load_environment

# 加载环境配置
load_environment()

def print_section(title: str):
    """打印章节标题"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_subsection(title: str):
    """打印子章节标题"""
    print(f"\n--- {title} ---")

def create_llm() -> ChatOpenAI:
    """创建 DeepSeek LLM 实例"""
    try:
        llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            temperature=0.7,
            max_tokens=1000
        )
        return llm
    except Exception as e:
        print(f"创建LLM失败: {e}")
        return None

def demo_basic_memory_operations():
    """演示基础记忆操作"""
    print_section("1. 基础记忆操作演示")
    
    # 创建记忆实例
    history = ChatMessageHistory()
    
    print_subsection("1.1 创建和添加消息")
    
    # 添加系统消息
    history.add_message(SystemMessage(content="你是一个有用的AI助手，请用中文回答问题。"))
    print("✓ 添加系统消息")
    
    # 添加用户消息
    history.add_user_message("你好，我是小明")
    print("✓ 添加用户消息: 你好，我是小明")
    
    # 添加AI消息
    history.add_ai_message("你好小明！很高兴认识你。有什么我可以帮助你的吗？")
    print("✓ 添加AI消息: 你好小明！很高兴认识你。有什么我可以帮助你的吗？")
    
    # 添加更多对话
    history.add_user_message("我想了解Python编程")
    history.add_ai_message("Python是一门很棒的编程语言！它简单易学，功能强大。你想从哪个方面开始学习呢？")
    
    print_subsection("1.2 查看消息历史")
    print(f"当前历史消息数量: {len(history.messages)}")
    
    for i, message in enumerate(history.messages):
        msg_type = type(message).__name__
        content = message.content[:50] + "..." if len(message.content) > 50 else message.content
        print(f"{i+1}. [{msg_type}] {content}")
    
    print_subsection("1.3 消息类型检查")
    for message in history.messages:
        if isinstance(message, SystemMessage):
            print(f"系统消息: {message.content}")
        elif isinstance(message, HumanMessage):
            print(f"用户消息: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"AI消息: {message.content}")
    
    return history

def demo_memory_with_llm():
    """演示带记忆的LLM调用"""
    print_section("2. 带记忆的LLM调用演示")
    
    # 创建LLM和记忆
    llm = create_llm()
    if not llm:
        print("❌ LLM创建失败，跳过此演示")
        return None
    
    history = ChatMessageHistory()
    
    # 添加系统消息
    system_msg = "你是一个有用的AI助手。请记住用户告诉你的信息，并在后续对话中使用这些信息。"
    history.add_message(SystemMessage(content=system_msg))
    
    print_subsection("2.1 第一轮对话")
    
    # 第一轮对话
    user_input1 = "我叫张三，今年25岁，是一名软件工程师"
    history.add_user_message(user_input1)
    print(f"用户: {user_input1}")
    
    try:
        # 调用LLM
        response1 = llm.invoke(history.messages)
        ai_response1 = response1.content
        history.add_ai_message(ai_response1)
        print(f"AI: {ai_response1}")
        
        print_subsection("2.2 第二轮对话（测试记忆）")
        
        # 第二轮对话 - 测试是否记住了用户信息
        user_input2 = "我的职业是什么？"
        history.add_user_message(user_input2)
        print(f"用户: {user_input2}")
        
        response2 = llm.invoke(history.messages)
        ai_response2 = response2.content
        history.add_ai_message(ai_response2)
        print(f"AI: {ai_response2}")
        
        print_subsection("2.3 第三轮对话（测试上下文保持）")
        
        # 第三轮对话 - 测试上下文保持
        user_input3 = "根据我的年龄和职业，给我一些职业发展建议"
        history.add_user_message(user_input3)
        print(f"用户: {user_input3}")
        
        response3 = llm.invoke(history.messages)
        ai_response3 = response3.content
        history.add_ai_message(ai_response3)
        print(f"AI: {ai_response3}")
        
    except Exception as e:
        print(f"❌ LLM调用失败: {e}")
        return None
    
    return history

def demo_conversation_history_management():
    """演示对话历史管理"""
    print_section("3. 对话历史管理演示")
    
    history = ChatMessageHistory()
    
    # 添加一些测试消息
    messages_data = [
        ("system", "你是一个有用的助手"),
        ("user", "第一条用户消息"),
        ("ai", "第一条AI回复"),
        ("user", "第二条用户消息"),
        ("ai", "第二条AI回复"),
        ("user", "第三条用户消息"),
        ("ai", "第三条AI回复")
    ]
    
    for msg_type, content in messages_data:
        if msg_type == "system":
            history.add_message(SystemMessage(content=content))
        elif msg_type == "user":
            history.add_user_message(content)
        elif msg_type == "ai":
            history.add_ai_message(content)
    
    print_subsection("3.1 查看完整历史")
    print(f"总消息数: {len(history.messages)}")
    for i, msg in enumerate(history.messages):
        msg_type = type(msg).__name__.replace("Message", "")
        print(f"{i+1}. [{msg_type}] {msg.content}")
    
    print_subsection("3.2 获取最近的消息")
    recent_messages = history.messages[-3:]  # 最近3条消息
    print(f"最近3条消息:")
    for i, msg in enumerate(recent_messages):
        msg_type = type(msg).__name__.replace("Message", "")
        print(f"{i+1}. [{msg_type}] {msg.content}")
    
    print_subsection("3.3 按类型筛选消息")
    user_messages = [msg for msg in history.messages if isinstance(msg, HumanMessage)]
    ai_messages = [msg for msg in history.messages if isinstance(msg, AIMessage)]
    
    print(f"用户消息数: {len(user_messages)}")
    print(f"AI消息数: {len(ai_messages)}")
    
    print_subsection("3.4 清除历史")
    print(f"清除前消息数: {len(history.messages)}")
    history.clear()
    print(f"清除后消息数: {len(history.messages)}")
    
    return history

def demo_memory_capacity_limits():
    """演示记忆容量限制"""
    print_section("4. 记忆容量限制演示")
    
    history = ChatMessageHistory()
    max_messages = 10  # 设置最大消息数
    
    print_subsection("4.1 添加大量消息")
    
    # 添加系统消息
    history.add_message(SystemMessage(content="你是一个有用的助手"))
    
    # 添加大量对话消息
    for i in range(15):
        history.add_user_message(f"用户消息 {i+1}")
        history.add_ai_message(f"AI回复 {i+1}")
        
        # 检查并限制消息数量
        if len(history.messages) > max_messages:
            # 保留系统消息和最近的消息
            system_messages = [msg for msg in history.messages if isinstance(msg, SystemMessage)]
            recent_messages = [msg for msg in history.messages if not isinstance(msg, SystemMessage)][-max_messages+len(system_messages):]
            history.messages = system_messages + recent_messages
            print(f"  消息数量达到限制，保留最近 {max_messages} 条消息")
    
    print_subsection("4.2 最终消息状态")
    print(f"最终消息数: {len(history.messages)}")
    for i, msg in enumerate(history.messages):
        msg_type = type(msg).__name__.replace("Message", "")
        print(f"{i+1}. [{msg_type}] {msg.content}")
    
    return history

def demo_token_counting():
    """演示简单的Token计数"""
    print_section("5. Token计数演示")
    
    def simple_token_count(messages: List[BaseMessage]) -> int:
        """简单的token计数（基于空格分割）"""
        total_tokens = 0
        for msg in messages:
            # 简化的token计数：按空格分割单词
            tokens = len(msg.content.split())
            total_tokens += tokens
        return total_tokens
    
    history = ChatMessageHistory()
    max_tokens = 100  # 设置最大token数
    
    print_subsection("5.1 添加消息并监控Token数")
    
    messages_to_add = [
        "你好，我想了解人工智能的发展历史",
        "人工智能起源于20世纪50年代，当时科学家们开始探索让机器模拟人类智能的可能性",
        "那么机器学习和深度学习是什么时候发展起来的？",
        "机器学习在20世纪80年代开始兴起，而深度学习则在2010年代迎来了突破性发展",
        "现在AI技术主要应用在哪些领域？",
        "AI技术广泛应用于自然语言处理、计算机视觉、推荐系统、自动驾驶等多个领域"
    ]
    
    for i, content in enumerate(messages_to_add):
        if i % 2 == 0:
            history.add_user_message(content)
        else:
            history.add_ai_message(content)
        
        token_count = simple_token_count(history.messages)
        print(f"添加消息 {i+1}: Token数 = {token_count}")
        
        # 如果超过限制，移除最早的消息
        while token_count > max_tokens and len(history.messages) > 1:
            removed_msg = history.messages.pop(0)
            token_count = simple_token_count(history.messages)
            print(f"  移除消息以控制Token数: {token_count}")
    
    print_subsection("5.2 最终状态")
    print(f"最终消息数: {len(history.messages)}")
    print(f"最终Token数: {simple_token_count(history.messages)}")
    
    return history

def demo_interactive_chat():
    """演示交互式聊天（模拟）"""
    print_section("6. 交互式聊天演示")
    
    llm = create_llm()
    if not llm:
        print("❌ LLM创建失败，跳过此演示")
        return None
    
    history = ChatMessageHistory()
    
    # 添加系统消息
    system_msg = "你是一个友好的AI助手，名叫小助手。请记住用户的信息并提供有用的帮助。"
    history.add_message(SystemMessage(content=system_msg))
    
    # 模拟用户输入
    simulated_inputs = [
        "你好，我是李华",
        "我正在学习Python编程",
        "你能推荐一些Python学习资源吗？",
        "我之前提到我在学什么？",
        "谢谢你的帮助！"
    ]
    
    print_subsection("6.1 模拟对话过程")
    
    for i, user_input in enumerate(simulated_inputs):
        print(f"\n回合 {i+1}:")
        print(f"用户: {user_input}")
        
        # 添加用户消息
        history.add_user_message(user_input)
        
        try:
            # 调用LLM
            response = llm.invoke(history.messages)
            ai_response = response.content
            
            # 添加AI响应
            history.add_ai_message(ai_response)
            print(f"AI: {ai_response}")
            
            # 显示当前历史长度
            print(f"[当前历史消息数: {len(history.messages)}]")
            
        except Exception as e:
            print(f"❌ 调用失败: {e}")
            break
    
    print_subsection("6.2 对话总结")
    print(f"总对话轮数: {len([msg for msg in history.messages if isinstance(msg, HumanMessage)])}")
    print(f"总消息数: {len(history.messages)}")
    
    return history

def demo_memory_persistence():
    """演示记忆持久化（模拟）"""
    print_section("7. 记忆持久化演示")
    
    print_subsection("7.1 创建会话A")
    session_a = ChatMessageHistory()
    session_a.add_user_message("我喜欢喝咖啡")
    session_a.add_ai_message("知道了，你喜欢喝咖啡")
    
    print(f"会话A消息数: {len(session_a.messages)}")
    for msg in session_a.messages:
        msg_type = type(msg).__name__.replace("Message", "")
        print(f"[{msg_type}] {msg.content}")
    
    print_subsection("7.2 创建会话B")
    session_b = ChatMessageHistory()
    session_b.add_user_message("我是程序员")
    session_b.add_ai_message("很棒！编程是一个很有趣的职业")
    
    print(f"会话B消息数: {len(session_b.messages)}")
    for msg in session_b.messages:
        msg_type = type(msg).__name__.replace("Message", "")
        print(f"[{msg_type}] {msg.content}")
    
    print_subsection("7.3 会话隔离验证")
    print(f"会话A和会话B是独立的:")
    print(f"  会话A消息数: {len(session_a.messages)}")
    print(f"  会话B消息数: {len(session_b.messages)}")
    print(f"  会话A不包含会话B的内容: {all(msg.content != '我是程序员' for msg in session_a.messages)}")
    
    return session_a, session_b

def main():
    """主函数"""
    print("LangChain 记忆功能演示")
    print(f"演示时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 基础记忆操作
        demo_basic_memory_operations()
        
        # 2. 带记忆的LLM调用
        demo_memory_with_llm()
        
        # 3. 对话历史管理
        demo_conversation_history_management()
        
        # 4. 记忆容量限制
        demo_memory_capacity_limits()
        
        # 5. Token计数
        demo_token_counting()
        
        # 6. 交互式聊天
        demo_interactive_chat()
        
        # 7. 记忆持久化
        demo_memory_persistence()
        
        print_section("演示完成")
        print("✅ 所有记忆功能演示已完成！")
        print("\n主要学习点:")
        print("1. InMemoryHistory 基础使用")
        print("2. 消息类型和历史管理")
        print("3. 记忆容量和Token限制")
        print("4. 与LLM的集成调用")
        print("5. 多轮对话和上下文保持")
        print("6. 会话隔离和管理")
        
    except KeyboardInterrupt:
        print("\n用户中断演示")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()