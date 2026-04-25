#!/usr/bin/env python3
"""
简单的LangServe API测试脚本
"""

import json

import requests


def test_api():
    """测试LangServe API"""
    base_url = "http://localhost:8002"

    # 测试健康检查
    print("🔍 测试健康检查...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"健康检查状态: {response.status_code}")
        print(f"响应: {response.json()}")
    except Exception as e:
        print(f"健康检查失败: {e}")
        return

    # 测试根路径
    print("\n🔍 测试根路径...")
    try:
        response = requests.get(base_url)
        print(f"根路径状态: {response.status_code}")
        print(f"响应: {response.json()}")
    except Exception as e:
        print(f"根路径测试失败: {e}")

    # 测试API文档
    print("\n🔍 测试API文档...")
    try:
        response = requests.get(f"{base_url}/docs")
        print(f"API文档状态: {response.status_code}")
        print(f"内容类型: {response.headers.get('content-type')}")
    except Exception as e:
        print(f"API文档测试失败: {e}")

    # 测试聊天API
    print("\n🔍 测试聊天API invoke...")
    try:
        chat_data = {"input": {"messages": [{"role": "user", "content": "你好"}]}}

        response = requests.post(
            f"{base_url}/chat/invoke",
            json=chat_data,
            headers={"Content-Type": "application/json"},
        )

        print(f"聊天API状态: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")

        if response.status_code == 200:
            print(f"响应内容: {response.json()}")
        else:
            print(f"错误响应: {response.text}")

    except Exception as e:
        print(f"聊天API测试失败: {e}")


if __name__ == "__main__":
    test_api()
