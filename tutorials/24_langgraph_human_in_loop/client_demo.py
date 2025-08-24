#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人工在环 Web 界面客户端演示

本示例演示如何通过 HTTP 请求与人工在环 Web API 进行交互。
主要功能：
1. 发送聊天消息
2. 检查中断状态
3. 提供人工输入恢复执行
4. 演示不同类型的人工在环场景

使用方法:
    python client_demo.py

确保 Web 服务器正在运行:
    python web_interface.py --port 8000
"""

import requests
import json
import time
from typing import Dict, Any, Optional
from pydantic import BaseModel


class HumanInLoopClient:
    """人工在环客户端类"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """初始化客户端"""
        self.base_url = base_url
        self.session = requests.Session()
        
    def check_server_health(self) -> bool:
        """检查服务器健康状态"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ 服务器健康状态正常")
                return True
            else:
                print(f"❌ 服务器健康检查失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 无法连接到服务器: {e}")
            return False
    
    def send_chat_message(self, message: str, thread_id: str = "demo_session", 
                         demo_type: str = "basic") -> Dict[str, Any]:
        """发送聊天消息"""
        try:
            payload = {
                "message": message,
                "thread_id": thread_id,
                "demo_type": demo_type
            }
            
            response = self.session.post(
                f"{self.base_url}/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            raise Exception(f"发送消息失败: {e}")
    
    def resume_execution(self, thread_id: str, human_input: str) -> Dict[str, Any]:
        """恢复执行"""
        try:
            payload = {
                "thread_id": thread_id,
                "human_input": human_input
            }
            
            response = self.session.post(
                f"{self.base_url}/resume",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            raise Exception(f"恢复执行失败: {e}")
    
    def check_status(self, thread_id: str) -> Dict[str, Any]:
        """检查状态"""
        try:
            response = self.session.get(
                f"{self.base_url}/status/{thread_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            raise Exception(f"状态检查失败: {e}")
    
    def interactive_chat_session(self, thread_id: str = "interactive_session"):
        """交互式聊天会话"""
        print("\n" + "=" * 60)
        print("🎮 交互式人工在环聊天会话")
        print("=" * 60)
        print("输入 'quit' 退出，输入 'status' 检查状态")
        print("演示类型: basic (基础), search (搜索), confirmation (确认)")
        print("\n开始聊天...")
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n您: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 会话结束")
                    break
                
                if user_input.lower() == 'status':
                    # 检查状态
                    status = self.check_status(thread_id)
                    print(f"📊 状态: {'🔴 中断' if status['is_interrupted'] else '🟢 正常'}")
                    if status['is_interrupted']:
                        print("⏸️ 需要人工输入")
                    continue
                
                if not user_input:
                    continue
                
                # 选择演示类型
                demo_type = input("演示类型 (basic/search/confirmation) [basic]: ").strip() or "basic"
                
                # 发送消息
                print("\n🤖 AI正在处理...")
                result = self.send_chat_message(user_input, thread_id, demo_type)
                
                print(f"\nAI: {result['response']}")
                
                # 检查是否需要人工输入
                if result['status'] == 'interrupted' and result['needs_human_input']:
                    print("\n⏸️ AI正在等待您的输入...")
                    
                    # 获取人工输入
                    human_input = input("请提供所需信息: ").strip()
                    
                    if human_input:
                        print("\n🔄 恢复执行...")
                        resume_result = self.resume_execution(thread_id, human_input)
                        print(f"\nAI: {resume_result['response']}")
                    else:
                        print("❌ 未提供输入，会话保持中断状态")
                
            except KeyboardInterrupt:
                print("\n\n👋 会话被用户中断")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")
    
    def demo_basic_human_in_loop(self):
        """演示基础人工在环功能"""
        print("\n" + "=" * 50)
        print("🔄 演示基础人工在环功能")
        print("=" * 50)
        
        thread_id = "demo_basic"
        
        try:
            # 发送需要人工确认的消息
            message = "我想要删除一些重要文件，请帮我确认一下"
            print(f"用户: {message}")
            
            result = self.send_chat_message(message, thread_id, "basic")
            print(f"AI: {result['response']}")
            
            # 检查是否被中断
            if result['status'] == 'interrupted':
                print("\n⏸️ AI正在等待人工输入...")
                
                # 模拟人工输入
                human_input = "请不要删除，这些文件很重要"
                print(f"人工输入: {human_input}")
                
                # 恢复执行
                resume_result = self.resume_execution(thread_id, human_input)
                print(f"AI恢复后: {resume_result['response']}")
            
        except Exception as e:
            print(f"❌ 演示失败: {e}")
    
    def demo_search_human_in_loop(self):
        """演示搜索人工在环功能"""
        print("\n" + "=" * 50)
        print("🔍 演示搜索人工在环功能")
        print("=" * 50)
        
        thread_id = "demo_search"
        
        try:
            # 发送搜索请求
            message = "我想了解一些技术信息"
            print(f"用户: {message}")
            
            result = self.send_chat_message(message, thread_id, "search")
            print(f"AI: {result['response']}")
            
            # 检查是否被中断
            if result['status'] == 'interrupted':
                print("\n⏸️ AI正在等待您指定搜索主题...")
                
                # 模拟人工输入搜索主题
                search_topic = "Python异步编程最佳实践"
                print(f"搜索主题: {search_topic}")
                
                # 恢复执行
                resume_result = self.resume_execution(thread_id, search_topic)
                print(f"AI搜索结果: {resume_result['response']}")
            
        except Exception as e:
            print(f"❌ 演示失败: {e}")
    
    def demo_confirmation_human_in_loop(self):
        """演示确认人工在环功能"""
        print("\n" + "=" * 50)
        print("✅ 演示确认人工在环功能")
        print("=" * 50)
        
        thread_id = "demo_confirmation"
        
        try:
            # 发送需要确认的操作
            message = "帮我发送一封重要邮件给客户"
            print(f"用户: {message}")
            
            result = self.send_chat_message(message, thread_id, "confirmation")
            print(f"AI: {result['response']}")
            
            # 检查是否被中断
            if result['status'] == 'interrupted':
                print("\n⏸️ AI正在等待您的确认...")
                
                # 模拟人工确认
                confirmation = "是的，请发送邮件，内容是关于项目进度更新"
                print(f"用户确认: {confirmation}")
                
                # 恢复执行
                resume_result = self.resume_execution(thread_id, confirmation)
                print(f"AI执行结果: {resume_result['response']}")
            
        except Exception as e:
            print(f"❌ 演示失败: {e}")
    
    def demo_status_monitoring(self):
        """演示状态监控"""
        print("\n" + "=" * 50)
        print("📊 演示状态监控")
        print("=" * 50)
        
        thread_ids = ["demo_basic", "demo_search", "demo_confirmation"]
        
        for thread_id in thread_ids:
            try:
                status = self.check_status(thread_id)
                print(f"\n线程 {thread_id}:")
                print(f"  状态: {'🔴 中断' if status['is_interrupted'] else '🟢 正常'}")
                print(f"  需要输入: {'是' if status['needs_human_input'] else '否'}")
                
                if status['conversation_state']:
                    session_status = status['conversation_state'].get('session_status', 'unknown')
                    print(f"  会话状态: {session_status}")
                
            except Exception as e:
                print(f"❌ 检查线程 {thread_id} 状态失败: {e}")
    
    def run_all_demos(self):
        """运行所有演示"""
        print("🚀 开始运行所有人工在环演示...")
        
        # 运行各种演示
        self.demo_basic_human_in_loop()
        time.sleep(1)
        
        self.demo_search_human_in_loop()
        time.sleep(1)
        
        self.demo_confirmation_human_in_loop()
        time.sleep(1)
        
        self.demo_status_monitoring()
        
        print("\n✅ 所有演示完成！")


def main():
    """主函数"""
    print("=" * 60)
    print("🤖 人工在环 Web 界面客户端演示")
    print("=" * 60)
    
    # 创建客户端
    client = HumanInLoopClient()
    
    # 检查服务器连接
    print(f"🔗 连接到服务器: {client.base_url}")
    if not client.check_server_health():
        print("\n❌ 无法连接到服务器")
        print("请确保服务器正在运行: python web_interface.py --port 8000")
        return
    
    # 显示菜单
    while True:
        print("\n" + "=" * 40)
        print("请选择演示模式:")
        print("1. 🔄 基础人工在环演示")
        print("2. 🔍 搜索人工在环演示")
        print("3. ✅ 确认人工在环演示")
        print("4. 📊 状态监控演示")
        print("5. 🎮 交互式聊天会话")
        print("6. 🚀 运行所有演示")
        print("7. 🌐 打开Web界面")
        print("0. 退出")
        print("=" * 40)
        
        try:
            choice = input("请输入选择 (0-7): ").strip()
            
            if choice == "0":
                print("👋 再见！")
                break
            elif choice == "1":
                client.demo_basic_human_in_loop()
            elif choice == "2":
                client.demo_search_human_in_loop()
            elif choice == "3":
                client.demo_confirmation_human_in_loop()
            elif choice == "4":
                client.demo_status_monitoring()
            elif choice == "5":
                client.interactive_chat_session()
            elif choice == "6":
                client.run_all_demos()
            elif choice == "7":
                print(f"🌐 请在浏览器中打开: {client.base_url}/interface")
                print(f"📚 API文档: {client.base_url}/docs")
            else:
                print("❌ 无效选择，请重试")
                
        except KeyboardInterrupt:
            print("\n\n👋 程序被用户中断")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    main()