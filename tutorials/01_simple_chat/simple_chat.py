#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LangChain 初体验 - 简单聊天示例

这个脚本演示了如何使用 LangChain 与 DeepSeek 模型进行简单的对话。
这是最基础的 LangChain 使用示例，帮助你快速上手。

运行方式：
    python tutorials/01_environment_setup/simple_chat.py

作者：Jaguarliu
日期：2025
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径，以便导入 utils 模块
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from utils.config import load_environment
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保已安装所需依赖：pip install -r requirements.txt")
    sys.exit(1)


def main():
    """
    主函数：演示 LangChain 的基本使用
    """
    print("🤖 LangChain 初体验 - 简单聊天示例")
    print("=" * 50)

    # 1. 加载配置和环境变量
    print("📋 正在加载配置...")
    try:
        config = load_environment()
        print("✅ 配置加载成功")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        print("请检查 .env 文件是否存在且配置正确")
        return

    # 2. 检查 DeepSeek API 密钥
    deepseek_api_key = config.deepseek_api_key
    deepseek_base_url = config.deepseek_base_url

    if not deepseek_api_key:
        print("❌ 未找到 DEEPSEEK_API_KEY")
        print("请在 .env 文件中配置你的 DeepSeek API 密钥")
        return

    print(f"🔑 DeepSeek API 密钥已配置 (前4位: {deepseek_api_key[:4]}...)")

    # 3. 初始化 LangChain 的 ChatOpenAI 模型（兼容 DeepSeek API）
    print("🚀 正在初始化 DeepSeek 模型...")
    try:
        # 创建 ChatOpenAI 实例（配置为使用 DeepSeek API）
        # model: 指定使用的模型版本
        # temperature: 控制输出的随机性 (0-1，0最确定，1最随机)
        # max_tokens: 限制输出的最大token数
        # base_url: DeepSeek API 的基础URL
        llm = ChatOpenAI(
            model="deepseek-chat",  # 使用 DeepSeek Chat 模型
            temperature=0.7,  # 适中的创造性
            max_tokens=500,  # 限制输出长度
            api_key=deepseek_api_key,  # DeepSeek API密钥
            base_url=deepseek_base_url,  # DeepSeek API基础URL
        )
        print("✅ 模型初始化成功")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return

    # 4. 准备对话消息
    print("\n💬 准备发送消息...")

    # 系统消息：定义AI的角色和行为
    system_message = SystemMessage(
        content="你是一个友好的AI助手，擅长用简洁明了的方式回答问题。请用中文回复。"
    )

    # 用户消息：实际的问题或对话内容
    user_question = "你好！请简单介绍一下什么是人工智能？"
    human_message = HumanMessage(content=user_question)

    print(f"👤 用户问题: {user_question}")

    # 5. 发送消息并获取响应
    print("\n🤔 AI 正在思考...")
    try:
        # 调用模型生成响应
        # messages 参数接受一个消息列表，可以包含系统消息、用户消息等
        response = llm.invoke([system_message, human_message])

        # 提取响应内容
        ai_response = response.content

        print("\n🤖 AI 回复:")
        print("-" * 30)
        print(ai_response)
        print("-" * 30)

    except Exception as e:
        print(f"❌ 获取响应失败: {e}")
        print("可能的原因：")
        print("1. DeepSeek API 密钥无效或已过期")
        print("2. 网络连接问题")
        print("3. DeepSeek API 配额不足")
        print("4. DeepSeek API 服务暂时不可用")
        return

    # 6. 显示一些有用的信息
    print("\n📊 调用信息:")
    print(f"模型: {llm.model_name}")
    print(f"温度参数: {llm.temperature}")
    print(f"最大tokens: {llm.max_tokens}")

    print("\n🎉 简单聊天示例完成！")
    print("\n💡 接下来你可以：")
    print("1. 修改这个脚本中的问题，尝试不同的对话")
    print("2. 调整 temperature 参数，观察输出的变化")
    print("3. 尝试不同的 DeepSeek 模型（如 deepseek-coder）")
    print("4. 学习下一章节：LangChain 基础组件详解")


if __name__ == "__main__":
    """
    脚本入口点
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        print("请检查错误信息并重试")
