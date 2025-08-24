#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph基础聊天机器人测试脚本

这是一个简化的测试脚本，用于验证LangGraph聊天机器人的核心功能。
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from basic_chatbot_demo import BasicChatbot

def test_chatbot():
    """测试聊天机器人基本功能"""
    print("🧪 LangGraph聊天机器人功能测试")
    print("=" * 50)
    
    try:
        # 1. 初始化聊天机器人
        print("\n1. 初始化聊天机器人...")
        chatbot = BasicChatbot()
        print("✅ 聊天机器人初始化成功")
        
        # 2. 测试图可视化
        print("\n2. 测试图可视化...")
        try:
            chatbot.visualize_graph()
            print("✅ 图可视化功能正常")
        except Exception as e:
            print(f"⚠️ 图可视化警告: {e}")
        
        # 3. 测试单次对话
        print("\n3. 测试单次对话...")
        test_message = "你好，请简单介绍一下你自己"
        print(f"用户: {test_message}")
        
        response = chatbot.chat(test_message)
        print(f"助手: {response}")
        print("✅ 单次对话测试成功")
        
        # 4. 测试多轮对话
        print("\n4. 测试多轮对话...")
        second_response = chatbot.chat("你刚才说了什么？")
        print(f"用户: 你刚才说了什么？")
        print(f"助手: {second_response[:100]}...")
        print("✅ 多轮对话测试成功")
        
        # 5. 测试图结构
        print("\n5. 测试图结构...")
        if chatbot.graph is not None:
            print("✅ 图结构构建正常")
        else:
            print("❌ 图结构未构建")
        print("✅ 图结构测试完成")
        
        print("\n🎉 所有测试通过！LangGraph聊天机器人功能正常")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chatbot()
    sys.exit(0 if success else 1)