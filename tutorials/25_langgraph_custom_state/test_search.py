#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试第25小节搜索功能修复
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from custom_state_demo import CustomStateChatbot
from langchain_core.messages import HumanMessage

def test_search_functionality():
    """测试搜索功能"""
    print("🧪 测试搜索功能修复...")
    
    try:
        # 创建聊天机器人实例
        chatbot = CustomStateChatbot()
        
        if chatbot.graph is None:
            print("❌ LangGraph图未正确初始化")
            return False
            
        # 测试搜索功能 - 通过图执行
        print("\n🔍 测试搜索 'LangGraph'...")
        
        # 创建输入状态
        input_state = {
            "messages": [HumanMessage(content="请搜索LangGraph的信息")],
            "name": "",
            "birthday": "",
            "verification_status": "pending",
            "search_results": []
        }
        
        # 通过图执行搜索
        config = {"configurable": {"thread_id": "test"}}
        result = chatbot.graph.invoke(input_state, config)
        
        print(f"执行结果: {result}")
        
        if "❌" in str(result):
            print(f"❌ 搜索失败: {result}")
            return False
        else:
            print(f"✅ 搜索成功!")
            return True
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_search_functionality()
    if success:
        print("\n🎉 搜索功能修复成功！")
    else:
        print("\n💥 搜索功能仍有问题")