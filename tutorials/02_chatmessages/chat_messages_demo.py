#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain ChatMessage 类型演示

本模块演示 LangChain 中四种核心 ChatMessage 类型的使用：
- HumanMessage: 用户消息
- AIMessage: AI助手消息
- SystemMessage: 系统消息
- FunctionMessage: 函数调用消息

同时演示这些消息类型与 DeepSeek API 的实际交互效果。

作者: Jaguarliu
日期: 2025-08
"""

import json
import os
import sys
from typing import List

from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.config import load_environment


def demonstrate_message_creation():
    """
    演示如何创建四种不同类型的 ChatMessage
    """
    print("=" * 60)
    print("📝 ChatMessage 创建演示")
    print("=" * 60)

    # 1. 创建 HumanMessage - 用户消息
    human_msg = HumanMessage(content="你好！我想学习 Python 编程，有什么建议吗？")
    print("\n1️⃣ HumanMessage (用户消息):")
    print(f"   类型: {type(human_msg).__name__}")
    print(f"   内容: {human_msg.content}")
    print(f"   消息类型: {human_msg.type}")

    # 2. 创建 AIMessage - AI助手消息
    ai_msg = AIMessage(
        content="很高兴帮助你学习Python！我建议从基础语法开始，然后逐步学习数据结构、函数和面向对象编程。"
    )
    print("\n2️⃣ AIMessage (AI助手消息):")
    print(f"   类型: {type(ai_msg).__name__}")
    print(f"   内容: {ai_msg.content}")
    print(f"   消息类型: {ai_msg.type}")

    # 3. 创建 SystemMessage - 系统消息
    system_msg = SystemMessage(
        content="你是一个专业的Python编程导师，擅长用简单易懂的方式解释复杂概念。请保持友好和耐心的态度。"
    )
    print("\n3️⃣ SystemMessage (系统消息):")
    print(f"   类型: {type(system_msg).__name__}")
    print(f"   内容: {system_msg.content}")
    print(f"   消息类型: {system_msg.type}")

    # 4. 创建 FunctionMessage - 函数调用消息
    function_msg = FunctionMessage(
        content='{"result": "Python是一种高级编程语言，语法简洁，适合初学者"}',
        name="get_language_info",
    )
    print("\n4️⃣ FunctionMessage (函数调用消息):")
    print(f"   类型: {type(function_msg).__name__}")
    print(f"   内容: {function_msg.content}")
    print(f"   函数名: {function_msg.name}")
    print(f"   消息类型: {function_msg.type}")

    return human_msg, ai_msg, system_msg, function_msg


def demonstrate_message_attributes():
    """
    演示消息的属性访问和操作
    """
    print("\n" + "=" * 60)
    print("🔍 消息属性详细演示")
    print("=" * 60)

    # 创建一个带有额外参数的消息
    enhanced_human_msg = HumanMessage(
        content="请帮我写一个计算斐波那契数列的函数",
        additional_kwargs={
            "user_id": "user_123",
            "timestamp": "2024-01-01 10:00:00",
            "priority": "high",
        },
    )

    print("\n📋 增强版 HumanMessage 属性:")
    print(f"   内容: {enhanced_human_msg.content}")
    print(f"   类型: {enhanced_human_msg.type}")
    print(f"   额外参数: {enhanced_human_msg.additional_kwargs}")

    # 访问额外参数
    if enhanced_human_msg.additional_kwargs:
        print("\n🔧 额外参数详情:")
        for key, value in enhanced_human_msg.additional_kwargs.items():
            print(f"   {key}: {value}")


def demonstrate_conversation_flow():
    """
    演示在实际对话流程中如何使用不同类型的消息
    """
    print("\n" + "=" * 60)
    print("💬 对话流程演示")
    print("=" * 60)

    # 构建一个完整的对话流程
    conversation: List = []

    # 1. 系统消息 - 设置AI角色
    system_message = SystemMessage(
        content="你是一个Python编程助手，专门帮助初学者解决编程问题。请用简洁明了的方式回答。"
    )
    conversation.append(system_message)

    # 2. 用户消息 - 提出问题
    user_question = HumanMessage(content="什么是Python中的列表推导式？能给个例子吗？")
    conversation.append(user_question)

    # 3. AI消息 - 回答问题
    ai_response = AIMessage(
        content="列表推导式是Python中创建列表的简洁方式。例如：[x**2 for x in range(5)] 会创建 [0, 1, 4, 9, 16]。"
    )
    conversation.append(ai_response)

    # 4. 函数调用消息 - 模拟函数调用结果
    function_result = FunctionMessage(
        content='{"code_example": "[x**2 for x in range(5)]", "output": "[0, 1, 4, 9, 16]"}',
        name="execute_python_code",
    )
    conversation.append(function_result)

    # 5. 用户后续问题
    follow_up = HumanMessage(content="太棒了！那如何在列表推导式中添加条件呢？")
    conversation.append(follow_up)

    # 显示完整对话
    print("\n📝 完整对话记录:")
    for i, message in enumerate(conversation, 1):
        role_emoji = {"system": "⚙️", "human": "👤", "ai": "🤖", "function": "🔧"}
        emoji = role_emoji.get(message.type, "❓")
        print(f"\n{i}. {emoji} {message.type.upper()}:")

        if hasattr(message, "name"):  # FunctionMessage 有 name 属性
            print(f"   函数名: {message.name}")

        # 格式化显示内容
        content = message.content
        if len(content) > 80:
            content = content[:77] + "..."
        print(f"   内容: {content}")

    return conversation


def demonstrate_message_comparison():
    """
    演示消息类型的比较和识别
    """
    print("\n" + "=" * 60)
    print("🔄 消息类型比较演示")
    print("=" * 60)

    # 创建不同类型的消息
    messages = [
        HumanMessage(content="用户消息示例"),
        AIMessage(content="AI回复示例"),
        SystemMessage(content="系统指令示例"),
        FunctionMessage(content="函数结果示例", name="test_function"),
    ]

    print("\n🏷️ 消息类型识别:")
    for i, msg in enumerate(messages, 1):
        print(f"\n{i}. 消息内容: '{msg.content[:20]}...'")
        print(f"   类型: {msg.type}")
        print(f"   是否为用户消息: {isinstance(msg, HumanMessage)}")
        print(f"   是否为AI消息: {isinstance(msg, AIMessage)}")
        print(f"   是否为系统消息: {isinstance(msg, SystemMessage)}")
        print(f"   是否为函数消息: {isinstance(msg, FunctionMessage)}")


def initialize_deepseek_model():
    """
    初始化 DeepSeek 模型

    Returns:
        ChatOpenAI: 配置好的 DeepSeek 模型实例
    """
    try:
        # 加载环境配置
        config = load_environment()

        # 初始化 DeepSeek 模型
        model = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=config.deepseek_api_key,
            openai_api_base=config.deepseek_base_url,
            temperature=0.7,
            max_tokens=500,
        )

        print("✅ DeepSeek 模型初始化成功")
        return model

    except Exception as e:
        print(f"❌ DeepSeek 模型初始化失败: {e}")
        print("请检查 .env 文件中的 DEEPSEEK_API_KEY 配置")
        return None


def demonstrate_system_message_effect():
    """
    演示 SystemMessage 对 AI 行为的影响
    通过对比有无 SystemMessage 的回复差异
    """
    print("\n" + "=" * 60)
    print("🎭 SystemMessage 效果对比演示")
    print("=" * 60)

    model = initialize_deepseek_model()
    if not model:
        print("⚠️ 无法初始化模型，跳过实际API调用演示")
        return

    user_question = "请介绍一下你自己"

    try:
        # 1. 没有 SystemMessage 的对话
        print("\n1️⃣ 没有 SystemMessage 的回复:")
        print(f"👤 用户: {user_question}")

        messages_without_system = [HumanMessage(content=user_question)]
        response_without_system = model.invoke(messages_without_system)
        print(f"🤖 AI: {response_without_system.content}")

        # 2. 有 SystemMessage 的对话
        print("\n2️⃣ 有 SystemMessage 的回复:")
        system_prompt = "你是一个专业的Python编程导师，请用简洁、专业的语言介绍自己，重点强调你在Python教学方面的能力。"
        print(f"⚙️ 系统设定: {system_prompt}")
        print(f"👤 用户: {user_question}")

        messages_with_system = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_question),
        ]
        response_with_system = model.invoke(messages_with_system)
        print(f"🤖 AI: {response_with_system.content}")

        print("\n📊 对比分析:")
        print("• 没有SystemMessage: AI使用默认行为模式回复")
        print("• 有SystemMessage: AI按照指定角色和风格回复")
        print("• SystemMessage可以有效控制AI的回复风格和内容重点")

    except Exception as e:
        print(f"❌ API调用失败: {e}")
        print("请检查网络连接和API密钥配置")


def demonstrate_conversation_with_api():
    """
    演示 HumanMessage 和 AIMessage 在实际对话中的作用
    """
    print("\n" + "=" * 60)
    print("💬 实际对话演示 (HumanMessage & AIMessage)")
    print("=" * 60)

    model = initialize_deepseek_model()
    if not model:
        print("⚠️ 无法初始化模型，跳过实际API调用演示")
        return

    try:
        # 构建对话历史
        conversation_history = []

        # 系统消息设定角色
        system_msg = SystemMessage(
            content="你是一个Python编程助手，请用简洁明了的方式回答编程问题。"
        )
        conversation_history.append(system_msg)

        # 第一轮对话
        print("\n🔄 第一轮对话:")
        user_msg1 = HumanMessage(content="什么是Python中的列表？")
        conversation_history.append(user_msg1)
        print(f"👤 用户: {user_msg1.content}")

        response1 = model.invoke(conversation_history)
        ai_msg1 = AIMessage(content=response1.content)
        conversation_history.append(ai_msg1)
        print(f"🤖 AI: {ai_msg1.content}")

        # 第二轮对话（基于上下文）
        print("\n🔄 第二轮对话 (基于上下文):")
        user_msg2 = HumanMessage(content="能给个具体的例子吗？")
        conversation_history.append(user_msg2)
        print(f"👤 用户: {user_msg2.content}")

        response2 = model.invoke(conversation_history)
        ai_msg2 = AIMessage(content=response2.content)
        conversation_history.append(ai_msg2)
        print(f"🤖 AI: {ai_msg2.content}")

        print("\n📋 完整对话历史:")
        for i, msg in enumerate(conversation_history):
            role_map = {"system": "⚙️ 系统", "human": "👤 用户", "ai": "🤖 AI"}
            role = role_map.get(msg.type, "❓ 未知")
            content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            print(f"  {i+1}. {role}: {content}")

        print("\n💡 关键点:")
        print("• HumanMessage: 承载用户的问题和需求")
        print("• AIMessage: 保存AI的回复，用于维护对话上下文")
        print("• 对话历史让AI能够理解上下文，提供更准确的回复")

    except Exception as e:
        print(f"❌ 对话演示失败: {e}")
        print("请检查网络连接和API密钥配置")


def demonstrate_function_message_simulation():
    """
    演示 FunctionMessage 的使用场景（概念演示）
    注意：由于不同API对函数调用的支持方式不同，这里主要演示概念
    """
    print("\n" + "=" * 60)
    print("🔧 FunctionMessage 使用场景演示")
    print("=" * 60)

    try:
        # 演示 FunctionMessage 的概念和结构
        print("\n📋 场景: 用户询问当前时间，需要调用时间函数")

        # 模拟函数调用结果
        import datetime

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        function_result = {
            "function_name": "get_current_time",
            "result": current_time,
            "timezone": "Asia/Shanghai",
        }

        # 创建 FunctionMessage
        function_msg = FunctionMessage(
            content=json.dumps(function_result, ensure_ascii=False),
            name="get_current_time",
        )

        print(f"👤 用户询问: 现在几点了？")
        print("\n🔧 函数调用过程演示:")
        print("1. AI识别需要调用时间函数")
        print(f"2. 调用函数: {function_msg.name}")
        print(f"3. 函数返回结果: {function_msg.content}")

        # 展示 FunctionMessage 的属性
        print("\n📊 FunctionMessage 详细信息:")
        print(f"   消息类型: {function_msg.type}")
        print(f"   函数名称: {function_msg.name}")
        print(f"   返回内容: {function_msg.content}")
        print(f"   内容长度: {len(function_msg.content)} 字符")

        # 模拟基于函数结果的AI回复
        print("\n🤖 基于函数结果的AI回复示例:")
        print(
            f"根据系统时间，现在是 {function_result['result']} ({function_result['timezone']} 时区)"
        )

        print("\n📚 FunctionMessage 要点:")
        print("• FunctionMessage 用于传递函数调用的结果")
        print("• 包含函数名称(name)和返回的数据(content)")
        print("• 让AI能够基于外部数据源提供准确信息")
        print("• 是构建AI Agent和工具调用的基础")
        print("• 不同的API提供商对函数调用有不同的实现方式")

        # 展示在对话流程中的使用
        print("\n💬 在对话流程中的使用示例:")
        conversation_example = [
            SystemMessage(content="你是一个助手，可以调用函数获取实时信息"),
            HumanMessage(content="现在几点了？"),
            function_msg,  # 函数调用结果
            AIMessage(content=f"现在是 {function_result['result']}"),
        ]

        for i, msg in enumerate(conversation_example, 1):
            role_emoji = {"system": "⚙️", "human": "👤", "ai": "🤖", "function": "🔧"}
            emoji = role_emoji.get(msg.type, "❓")
            content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            if hasattr(msg, "name"):
                print(f"   {i}. {emoji} {msg.type.upper()} [{msg.name}]: {content}")
            else:
                print(f"   {i}. {emoji} {msg.type.upper()}: {content}")

    except Exception as e:
        print(f"❌ FunctionMessage演示失败: {e}")
        print("这是一个概念演示，不涉及实际API调用")


def main():
    """
    主函数 - 运行所有演示
    """
    print("🚀 LangChain ChatMessage 完整演示")
    print("本演示将展示四种核心消息类型的创建、使用和实际API交互效果")

    try:
        # 1. 基本消息创建演示
        demonstrate_message_creation()

        # 2. 消息属性演示
        demonstrate_message_attributes()

        # 3. 对话流程演示
        demonstrate_conversation_flow()

        # 4. 消息比较演示
        demonstrate_message_comparison()

        # 5. SystemMessage 实际效果演示
        demonstrate_system_message_effect()

        # 6. 实际对话演示
        demonstrate_conversation_with_api()

        # 7. FunctionMessage 使用场景演示
        demonstrate_function_message_simulation()

        print("\n" + "=" * 60)
        print("✅ 演示完成！")
        print("=" * 60)
        print("\n📚 学习总结:")
        print("• HumanMessage: 用于表示用户输入")
        print("• AIMessage: 用于表示AI回复，维护对话上下文")
        print("• SystemMessage: 用于设置AI行为和角色，影响回复风格")
        print("• FunctionMessage: 用于处理函数调用结果，扩展AI能力")
        print("\n💡 提示: 这些消息类型是构建LangChain应用的基础！")
        print("\n🎯 实际应用: 通过API交互，你可以看到不同消息类型的真实效果")

    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        print("请检查是否正确安装了相关依赖包:")
        print("• pip install langchain-core")
        print("• pip install langchain-openai")
        print("• 确保 .env 文件中配置了 DEEPSEEK_API_KEY")


if __name__ == "__main__":
    main()
