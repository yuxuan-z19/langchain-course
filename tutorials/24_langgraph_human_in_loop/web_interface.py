#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph 人工在环 Web 界面

本示例演示如何使用 LangServe 为人工在环功能提供 Web 界面。
主要功能：
1. 集成 HumanInLoopChatbot 类
2. 提供 Web API 接口
3. 支持人工在环交互
4. 提供可视化界面

运行方式：
    python web_interface.py

访问地址：
    - API文档：http://localhost:8000/docs
    - 交互界面：http://localhost:8000/chat/playground
    - 状态检查：http://localhost:8000/status
"""

import asyncio
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# 导入人工在环聊天机器人
from human_in_loop_demo import HumanInLoopChatbot
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langserve import add_routes
from pydantic import BaseModel, Field


class ChatInput(BaseModel):
    """聊天输入模型"""

    message: str = Field(description="用户消息内容")
    thread_id: str = Field(default="web_session", description="会话线程ID")
    demo_type: str = Field(
        default="basic", description="演示类型：basic, search, confirmation"
    )


class ChatOutput(BaseModel):
    """聊天输出模型"""

    response: str = Field(description="AI回复内容")
    status: str = Field(description="状态：completed, interrupted, error")
    thread_id: str = Field(description="会话线程ID")
    needs_human_input: bool = Field(description="是否需要人工输入")
    interrupt_reason: Optional[str] = Field(default=None, description="中断原因")


class ResumeInput(BaseModel):
    """恢复执行输入模型"""

    thread_id: str = Field(description="会话线程ID")
    human_input: str = Field(description="人工输入内容")


class StatusOutput(BaseModel):
    """状态输出模型"""

    thread_id: str = Field(description="会话线程ID")
    is_interrupted: bool = Field(description="是否处于中断状态")
    needs_human_input: bool = Field(description="是否需要人工输入")
    conversation_state: Optional[Dict[str, Any]] = Field(description="对话状态")


class HumanInLoopWebInterface:
    """人工在环 Web 界面类"""

    def __init__(self):
        """初始化 Web 界面"""
        self.app = None
        self.chatbot = None
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self._setup_chatbot()
        self._setup_app()

    def _setup_chatbot(self):
        """设置聊天机器人"""
        try:
            self.chatbot = HumanInLoopChatbot()
            print("✅ 人工在环聊天机器人初始化成功")
        except Exception as e:
            print(f"❌ 初始化聊天机器人失败: {e}")
            raise

    def _setup_app(self):
        """设置 FastAPI 应用"""
        # 创建 FastAPI 应用
        self.app = FastAPI(
            title="LangGraph 人工在环 Web 界面",
            description="使用 LangServe 为人工在环功能提供 Web 界面",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        # 添加 CORS 中间件
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 添加路由
        self._add_routes()

        print("✅ FastAPI 应用设置完成")

    def _add_routes(self):
        """添加路由"""

        @self.app.get("/")
        async def root():
            """根路径信息"""
            return {
                "message": "LangGraph 人工在环 Web 界面",
                "docs": "/docs",
                "chat_api": "/chat",
                "resume_api": "/resume",
                "status_api": "/status",
                "web_interface": "/interface",
            }

        @self.app.get("/health")
        async def health_check():
            """健康检查端点"""
            return {"status": "healthy", "service": "human-in-loop-web"}

        @self.app.post("/chat", response_model=ChatOutput)
        async def chat_endpoint(
            chat_input: ChatInput, background_tasks: BackgroundTasks
        ):
            """聊天端点"""
            try:
                # 初始化会话状态
                if chat_input.thread_id not in self.active_sessions:
                    self.active_sessions[chat_input.thread_id] = {
                        "status": "active",
                        "last_message": None,
                        "interrupt_reason": None,
                    }

                # 根据演示类型选择不同的处理方式
                if chat_input.demo_type == "basic":
                    result = await self._handle_basic_chat(chat_input)
                elif chat_input.demo_type == "search":
                    result = await self._handle_search_chat(chat_input)
                elif chat_input.demo_type == "confirmation":
                    result = await self._handle_confirmation_chat(chat_input)
                else:
                    result = await self._handle_basic_chat(chat_input)

                return result

            except Exception as e:
                print(f"❌ 聊天处理失败: {e}")
                return ChatOutput(
                    response=f"处理失败: {str(e)}",
                    status="error",
                    thread_id=chat_input.thread_id,
                    needs_human_input=False,
                    interrupt_reason=None,
                )

        @self.app.post("/resume", response_model=ChatOutput)
        async def resume_endpoint(resume_input: ResumeInput):
            """恢复执行端点"""
            try:
                # 检查会话是否存在
                if resume_input.thread_id not in self.active_sessions:
                    raise HTTPException(status_code=404, detail="会话不存在")

                # 检查是否处于中断状态
                if not self.chatbot.check_interrupt_status(resume_input.thread_id):
                    raise HTTPException(status_code=400, detail="会话未处于中断状态")

                # 恢复执行 - resume_execution返回事件列表
                events = self.chatbot.resume_execution(
                    resume_input.human_input, resume_input.thread_id
                )

                # 从事件列表中提取AI的最终响应
                ai_response = "执行已恢复"
                if events:
                    for event in reversed(events):  # 从最后一个事件开始查找
                        if "messages" in event and event["messages"]:
                            latest_message = event["messages"][-1]
                            if (
                                hasattr(latest_message, "content")
                                and latest_message.content
                            ):
                                from langchain_core.messages import AIMessage

                                if isinstance(latest_message, AIMessage):
                                    ai_response = latest_message.content
                                    break

                # 更新会话状态
                self.active_sessions[resume_input.thread_id]["status"] = "active"
                self.active_sessions[resume_input.thread_id]["interrupt_reason"] = None

                return ChatOutput(
                    response=ai_response,
                    status="completed",
                    thread_id=resume_input.thread_id,
                    needs_human_input=False,
                    interrupt_reason=None,
                )

            except Exception as e:
                print(f"❌ 恢复执行失败: {e}")
                return ChatOutput(
                    response=f"恢复执行失败: {str(e)}",
                    status="error",
                    thread_id=resume_input.thread_id,
                    needs_human_input=False,
                    interrupt_reason=None,
                )

        @self.app.get("/status/{thread_id}", response_model=StatusOutput)
        async def status_endpoint(thread_id: str):
            """状态检查端点"""
            try:
                # 检查中断状态
                is_interrupted = self.chatbot.check_interrupt_status(thread_id)

                # 获取对话状态
                conversation_state = self.chatbot.get_conversation_state(thread_id)

                # 获取会话信息
                session_info = self.active_sessions.get(thread_id, {})

                return StatusOutput(
                    thread_id=thread_id,
                    is_interrupted=is_interrupted,
                    needs_human_input=is_interrupted,
                    conversation_state={
                        "session_status": session_info.get("status", "unknown"),
                        "interrupt_reason": session_info.get("interrupt_reason"),
                        "conversation_state": conversation_state,
                    },
                )

            except Exception as e:
                print(f"❌ 状态检查失败: {e}")
                raise HTTPException(status_code=500, detail=f"状态检查失败: {str(e)}")

        @self.app.get("/interface", response_class=HTMLResponse)
        async def web_interface():
            """简单的 Web 界面"""
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>人工在环聊天界面</title>
                <meta charset="utf-8">
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .container { max-width: 800px; margin: 0 auto; }
                    .chat-box { border: 1px solid #ccc; height: 400px; overflow-y: scroll; padding: 10px; margin: 10px 0; }
                    .input-group { margin: 10px 0; }
                    input, select, button { padding: 8px; margin: 5px; }
                    .message { margin: 5px 0; padding: 5px; border-radius: 5px; }
                    .user-message { background-color: #e3f2fd; }
                    .ai-message { background-color: #f3e5f5; }
                    .status-message { background-color: #fff3e0; }
                    .error-message { background-color: #ffebee; }
                    .waiting { color: #ff9800; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🤖 人工在环聊天界面</h1>
                    
                    <div class="input-group">
                        <label>演示类型：</label>
                        <select id="demoType">
                            <option value="basic">基础演示</option>
                            <option value="search">搜索演示</option>
                            <option value="confirmation">确认演示</option>
                        </select>
                        <label>线程ID：</label>
                        <input type="text" id="threadId" value="web_session" />
                    </div>
                    
                    <div id="chatBox" class="chat-box"></div>
                    
                    <div class="input-group">
                        <input type="text" id="messageInput" placeholder="输入消息..." style="width: 60%;" />
                        <button onclick="sendMessage()">发送</button>
                        <button onclick="checkStatus()">检查状态</button>
                    </div>
                    
                    <div class="input-group" id="resumeGroup" style="display: none;">
                        <input type="text" id="resumeInput" placeholder="输入人工反馈..." style="width: 60%;" />
                        <button onclick="resumeExecution()">恢复执行</button>
                    </div>
                </div>
                
                <script>
                    let currentThreadId = 'web_session';
                    
                    function addMessage(content, type) {
                        const chatBox = document.getElementById('chatBox');
                        const message = document.createElement('div');
                        message.className = `message ${type}-message`;
                        message.innerHTML = `<strong>${type.toUpperCase()}:</strong> ${content}`;
                        chatBox.appendChild(message);
                        chatBox.scrollTop = chatBox.scrollHeight;
                    }
                    
                    async function sendMessage() {
                        const messageInput = document.getElementById('messageInput');
                        const demoType = document.getElementById('demoType').value;
                        const threadId = document.getElementById('threadId').value;
                        const message = messageInput.value.trim();
                        
                        if (!message) return;
                        
                        currentThreadId = threadId;
                        addMessage(message, 'user');
                        messageInput.value = '';
                        
                        try {
                            const response = await fetch('/chat', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    message: message,
                                    thread_id: threadId,
                                    demo_type: demoType
                                })
                            });
                            
                            const result = await response.json();
                            
                            if (result.status === 'interrupted') {
                                addMessage(result.response, 'status');
                                addMessage('⏸️ 等待人工输入...', 'status');
                                document.getElementById('resumeGroup').style.display = 'block';
                            } else if (result.status === 'completed') {
                                addMessage(result.response, 'ai');
                                document.getElementById('resumeGroup').style.display = 'none';
                            } else {
                                addMessage(result.response, 'error');
                            }
                        } catch (error) {
                            addMessage(`错误: ${error.message}`, 'error');
                        }
                    }
                    
                    async function resumeExecution() {
                        const resumeInput = document.getElementById('resumeInput');
                        const humanInput = resumeInput.value.trim();
                        
                        if (!humanInput) return;
                        
                        addMessage(humanInput, 'user');
                        resumeInput.value = '';
                        
                        try {
                            const response = await fetch('/resume', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    thread_id: currentThreadId,
                                    human_input: humanInput
                                })
                            });
                            
                            const result = await response.json();
                            
                            if (result.status === 'completed') {
                                addMessage(result.response, 'ai');
                                document.getElementById('resumeGroup').style.display = 'none';
                            } else {
                                addMessage(result.response, 'error');
                            }
                        } catch (error) {
                            addMessage(`恢复执行错误: ${error.message}`, 'error');
                        }
                    }
                    
                    async function checkStatus() {
                        try {
                            const response = await fetch(`/status/${currentThreadId}`);
                            const result = await response.json();
                            
                            const statusText = result.is_interrupted ? 
                                '🔴 中断状态 - 需要人工输入' : '🟢 正常状态';
                            addMessage(statusText, 'status');
                            
                            if (result.is_interrupted) {
                                document.getElementById('resumeGroup').style.display = 'block';
                            }
                        } catch (error) {
                            addMessage(`状态检查错误: ${error.message}`, 'error');
                        }
                    }
                    
                    // 回车发送消息
                    document.getElementById('messageInput').addEventListener('keypress', function(e) {
                        if (e.key === 'Enter') {
                            sendMessage();
                        }
                    });
                    
                    document.getElementById('resumeInput').addEventListener('keypress', function(e) {
                        if (e.key === 'Enter') {
                            resumeExecution();
                        }
                    });
                </script>
            </body>
            </html>
            """
            return HTMLResponse(content=html_content)

    async def _handle_basic_chat(self, chat_input: ChatInput) -> ChatOutput:
        """处理基础聊天"""
        try:
            # 在后台线程中运行聊天
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._run_chat_with_interrupt_check,
                chat_input.message,
                chat_input.thread_id,
                "basic",
            )

            return result

        except Exception as e:
            print(f"❌ 基础聊天处理失败: {e}")
            raise

    async def _handle_search_chat(self, chat_input: ChatInput) -> ChatOutput:
        """处理搜索聊天"""
        try:
            # 在后台线程中运行聊天
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._run_chat_with_interrupt_check,
                chat_input.message,
                chat_input.thread_id,
                "search",
            )

            return result

        except Exception as e:
            print(f"❌ 搜索聊天处理失败: {e}")
            raise

    async def _handle_confirmation_chat(self, chat_input: ChatInput) -> ChatOutput:
        """处理确认聊天"""
        try:
            # 在后台线程中运行聊天
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._run_chat_with_interrupt_check,
                chat_input.message,
                chat_input.thread_id,
                "confirmation",
            )

            return result

        except Exception as e:
            print(f"❌ 确认聊天处理失败: {e}")
            raise

    def _run_chat_with_interrupt_check(
        self, message: str, thread_id: str, demo_type: str
    ) -> ChatOutput:
        """运行聊天并检查中断状态"""
        try:
            # 根据演示类型构造不同的提示
            if demo_type == "search":
                prompt = f"用户想要搜索相关信息：{message}。请使用human_assistance工具获取用户想要搜索的具体主题，然后使用search_web工具进行搜索并总结结果。"
            elif demo_type == "confirmation":
                prompt = f"用户请求：{message}。请使用human_assistance工具向用户确认是否要执行此操作，然后根据用户的确认结果进行相应处理。"
            else:
                prompt = f"用户消息：{message}。如果需要更多信息或确认，请使用human_assistance工具与用户交互。"

            # 发送消息到聊天机器人
            events = self.chatbot.chat_with_human_in_loop(prompt, thread_id)

            # 检查是否被中断
            is_interrupted = self.chatbot.check_interrupt_status(thread_id)

            if is_interrupted:
                # 更新会话状态
                self.active_sessions[thread_id]["status"] = "interrupted"
                self.active_sessions[thread_id]["interrupt_reason"] = "需要人工输入"

                return ChatOutput(
                    response="AI正在等待您的输入，请在下方提供所需信息。",
                    status="interrupted",
                    thread_id=thread_id,
                    needs_human_input=True,
                    interrupt_reason="需要人工输入",
                )
            else:
                # 从事件列表中提取AI消息内容
                ai_response = "处理完成"
                if events and isinstance(events, list):
                    for event in events:
                        if isinstance(event, dict) and "messages" in event:
                            messages = event["messages"]
                            if isinstance(messages, list):
                                for msg in messages:
                                    if isinstance(msg, AIMessage) and msg.content:
                                        ai_response = msg.content
                                        break
                            elif isinstance(messages, AIMessage) and messages.content:
                                ai_response = messages.content

                return ChatOutput(
                    response=ai_response,
                    status="completed",
                    thread_id=thread_id,
                    needs_human_input=False,
                    interrupt_reason=None,
                )

        except Exception as e:
            print(f"❌ 聊天执行失败: {e}")
            return ChatOutput(
                response=f"处理失败: {str(e)}",
                status="error",
                thread_id=thread_id,
                needs_human_input=False,
                interrupt_reason=None,
            )

    def run_server(self, host: str = "localhost", port: int = 8000):
        """启动 Web 服务器"""
        print("🚀 启动人工在环 Web 服务器...")
        print(f"📍 服务地址: http://{host}:{port}")
        print(f"📚 API文档: http://{host}:{port}/docs")
        print(f"🎮 Web界面: http://{host}:{port}/interface")
        print(f"🔗 聊天API: http://{host}:{port}/chat")
        print(f"▶️ 恢复API: http://{host}:{port}/resume")
        print(f"📊 状态API: http://{host}:{port}/status/{{thread_id}}")
        print("\n按 Ctrl+C 停止服务器")

        try:
            uvicorn.run(self.app, host="127.0.0.1", port=port, log_level="info")
        except KeyboardInterrupt:
            print("\n👋 服务器已停止")
        except Exception as e:
            print(f"❌ 服务器启动失败: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("🤖 LangGraph 人工在环 Web 界面")
    print("=" * 60)

    try:
        # 创建并运行 Web 界面
        web_interface = HumanInLoopWebInterface()
        web_interface.run_server()
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("\n请检查：")
        print("1. 环境变量是否正确设置")
        print("2. 依赖包是否正确安装")
        print("3. 端口8000是否被占用")


if __name__ == "__main__":
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="人工在环 Web 界面")
    parser.add_argument("--host", default="localhost", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    args = parser.parse_args()

    # 创建并运行 Web 界面
    web_interface = HumanInLoopWebInterface()
    web_interface.run_server(host=args.host, port=args.port)
