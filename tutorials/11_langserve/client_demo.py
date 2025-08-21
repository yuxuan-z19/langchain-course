#!/usr/bin/env python3
"""
LangServe客户端演示

这个脚本演示如何使用HTTP请求调用LangServe API，包括：
1. 单次调用 (invoke)
2. 流式调用 (stream) 
3. 批量调用 (batch)
4. 错误处理
5. 性能测试

使用方法:
    python client_demo.py

确保LangServe服务器正在运行:
    python chat_api_demo.py --port 8002
"""

import asyncio
import time
from typing import List, Dict, Any, Iterator
import requests
import json
from pydantic import BaseModel


class ChatMessage(BaseModel):
    """聊天消息模型"""
    role: str
    content: str


class ChatInput(BaseModel):
    """聊天输入模型"""
    messages: List[ChatMessage]
    
    def dict(self):
        return {"messages": [msg.dict() for msg in self.messages]}


class LangServeClientDemo:
    """LangServe客户端演示类"""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        """初始化客户端"""
        self.base_url = base_url
        self.chat_url = f"{base_url}/chat"
        
    def check_server_health(self) -> bool:
        """检查服务器健康状态"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ 服务器健康状态正常")
                return True
            else:
                print(f"❌ 服务器健康检查失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 无法连接到服务器: {e}")
            return False
    
    def invoke_chat(self, messages: List[Dict[str, str]]) -> str:
        """单次调用聊天API"""
        try:
            chat_input = ChatInput(messages=[ChatMessage(**msg) for msg in messages])
            response = requests.post(
                f"{self.chat_url}/invoke",
                json={"input": chat_input.dict()},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result.get("output", "")
        except Exception as e:
            raise Exception(f"调用失败: {e}")
    
    def stream_chat(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        """流式调用聊天API"""
        try:
            chat_input = ChatInput(messages=[ChatMessage(**msg) for msg in messages])
            response = requests.post(
                f"{self.chat_url}/stream",
                json={"input": chat_input.dict()},
                headers={"Content-Type": "application/json"},
                stream=True,
                timeout=30
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        # LangServe流式响应格式
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]  # 移除'data: '前缀
                            if data_str.strip() and data_str != '[DONE]':
                                data = json.loads(data_str)
                                yield data
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
        except Exception as e:
            raise Exception(f"流式调用失败: {e}")
    
    def batch_chat(self, batch_messages: List[List[Dict[str, str]]]) -> List[str]:
        """批量调用聊天API"""
        try:
            batch_inputs = []
            for messages in batch_messages:
                chat_input = ChatInput(messages=[ChatMessage(**msg) for msg in messages])
                batch_inputs.append(chat_input.dict())
            
            response = requests.post(
                f"{self.chat_url}/batch",
                json={"inputs": batch_inputs},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return [item.get("output", "") for item in result.get("outputs", [])]
        except Exception as e:
            raise Exception(f"批量调用失败: {e}")
    
    def demo_invoke(self):
        """演示单次调用"""
        print("\n" + "=" * 50)
        print("🔄 演示单次调用 (invoke)")
        print("=" * 50)
        
        try:
            messages = [{"role": "user", "content": "请简单介绍一下人工智能的发展历史"}]
            
            print(f"用户: {messages[0]['content']}")
            
            # 记录开始时间
            start_time = time.time()
            
            # 调用API
            response = self.invoke_chat(messages)
            
            # 计算耗时
            elapsed_time = time.time() - start_time
            
            print(f"AI: {response}")
            print(f"\n⏱️ 调用耗时: {elapsed_time:.2f}秒")
            
        except Exception as e:
            print(f"❌ 调用失败: {e}")
    
    def demo_stream(self):
        """演示流式调用"""
        print("\n" + "=" * 50)
        print("🌊 演示流式调用 (stream)")
        print("=" * 50)
        
        try:
            messages = [{"role": "user", "content": "请讲一个关于机器学习的有趣故事"}]
            
            print(f"用户: {messages[0]['content']}")
            print("AI: ", end="", flush=True)
            
            # 记录开始时间
            start_time = time.time()
            
            # 流式调用API
            full_response = ""
            for chunk in self.stream_chat(messages):
                chunk_text = str(chunk)
                print(chunk_text, end="", flush=True)
                full_response += chunk_text
            
            # 计算耗时
            elapsed_time = time.time() - start_time
            
            print(f"\n\n⏱️ 流式调用耗时: {elapsed_time:.2f}秒")
            
        except Exception as e:
            print(f"❌ 流式调用失败: {e}")
    
    def demo_batch(self):
        """演示批量调用"""
        print("\n" + "=" * 50)
        print("📦 演示批量调用 (batch)")
        print("=" * 50)
        
        try:
            # 准备批量输入数据
            batch_messages = [
                [{"role": "user", "content": "什么是深度学习？"}],
                [{"role": "user", "content": "什么是自然语言处理？"}],
                [{"role": "user", "content": "什么是计算机视觉？"}]
            ]
            
            print(f"批量处理 {len(batch_messages)} 个问题...")
            
            # 记录开始时间
            start_time = time.time()
            
            # 批量调用API
            responses = self.batch_chat(batch_messages)
            
            # 计算耗时
            elapsed_time = time.time() - start_time
            
            # 显示结果
            for i, (messages, response) in enumerate(zip(batch_messages, responses), 1):
                question = messages[0]['content']
                print(f"\n问题 {i}: {question}")
                display_response = response[:100] + "..." if len(response) > 100 else response
                print(f"回答 {i}: {display_response}")
            
            print(f"\n⏱️ 批量调用耗时: {elapsed_time:.2f}秒")
            print(f"📊 平均每个请求: {elapsed_time/len(batch_messages):.2f}秒")
            
        except Exception as e:
            print(f"❌ 批量调用失败: {e}")
    
    async def demo_async_stream(self):
        """演示异步流式调用（模拟）"""
        print("\n" + "=" * 50)
        print("⚡ 演示异步流式调用 (astream)")
        print("=" * 50)
        
        try:
            messages = [{"role": "user", "content": "请解释一下什么是大语言模型，以及它们是如何工作的"}]
            
            print(f"用户: {messages[0]['content']}")
            print("AI: ", end="", flush=True)
            
            # 记录开始时间
            start_time = time.time()
            
            # 使用同步方法模拟异步（在实际应用中可以使用aiohttp）
            full_response = ""
            for chunk in self.stream_chat(messages):
                chunk_text = str(chunk)
                print(chunk_text, end="", flush=True)
                full_response += chunk_text
                await asyncio.sleep(0.01)  # 模拟异步处理
            
            # 计算耗时
            elapsed_time = time.time() - start_time
            
            print(f"\n\n⏱️ 异步流式调用耗时: {elapsed_time:.2f}秒")
            
        except Exception as e:
            print(f"❌ 异步流式调用失败: {e}")
    
    def demo_error_handling(self):
        """演示错误处理"""
        print("\n" + "=" * 50)
        print("🛡️ 演示错误处理")
        print("=" * 50)
        
        # 测试无效输入
        print("测试无效输入...")
        try:
            # 发送格式错误的请求
            response = requests.post(
                f"{self.chat_url}/invoke",
                json={"invalid_key": "invalid_value"},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            print(f"意外成功: {response.json()}")
        except Exception as e:
            print(f"✅ 正确捕获错误: {type(e).__name__}: {e}")
        
        # 测试空消息
        print("\n测试空消息...")
        try:
            empty_messages = []
            response = self.invoke_chat(empty_messages)
            print(f"响应: {response}")
        except Exception as e:
            print(f"✅ 正确捕获错误: {type(e).__name__}: {e}")
    
    def demo_performance_test(self):
        """演示性能测试"""
        print("\n" + "=" * 50)
        print("📈 演示性能测试")
        print("=" * 50)
        
        test_questions = [
            "什么是机器学习？",
            "解释一下神经网络",
            "什么是强化学习？",
            "深度学习的优势是什么？",
            "什么是迁移学习？"
        ]
        
        print(f"测试 {len(test_questions)} 个问题的响应时间...")
        
        times = []
        for i, question in enumerate(test_questions, 1):
            try:
                messages = [{"role": "user", "content": question}]
                
                start_time = time.time()
                response = self.invoke_chat(messages)
                elapsed_time = time.time() - start_time
                times.append(elapsed_time)
                
                print(f"问题 {i}: {elapsed_time:.2f}秒 - {question}")
                
            except Exception as e:
                print(f"问题 {i} 失败: {e}")
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"\n📊 性能统计:")
            print(f"   平均响应时间: {avg_time:.2f}秒")
            print(f"   最快响应时间: {min_time:.2f}秒")
            print(f"   最慢响应时间: {max_time:.2f}秒")
        else:
            print("\n❌ 没有成功的请求用于统计")


async def run_async_demos(client: LangServeClientDemo):
    """运行异步演示"""
    await client.demo_async_stream()


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 LangServe 客户端演示")
    print("=" * 60)
    
    # 创建客户端
    client = LangServeClientDemo()
    
    # 检查服务器连接
    print(f"✅ 连接到LangServe API: {client.base_url}/chat")
    if not client.check_server_health():
        print("\n❌ 无法连接到LangServe服务器")
        print("请确保服务器正在运行: python chat_api_demo.py --port 8002")
        return
    
    # 运行同步演示
    client.demo_invoke()
    client.demo_stream()
    client.demo_batch()
    
    # 运行异步演示
    print("\n运行异步演示...")
    asyncio.run(run_async_demos(client))
    
    # 演示错误处理和性能测试
    client.demo_error_handling()
    client.demo_performance_test()
    
    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()