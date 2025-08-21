#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangServe 聊天API演示

本示例演示如何使用LangServe将LangChain聊天机器人快速转换为REST API服务。
主要功能：
1. 创建基于DeepSeek的聊天链
2. 使用LangServe包装为API服务
3. 提供交互式Web界面
4. 支持多种调用方式（invoke、stream、batch）

运行方式：
    python chat_api_demo.py
    
访问地址：
    - API文档：http://localhost:8000/docs
    - 交互界面：http://localhost:8000/chat/playground
    - API端点：http://localhost:8000/chat/
"""

import sys
import os
from typing import List, Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from pydantic import BaseModel, Field

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.config import load_deepseek_config
from utils.llm_factory import create_deepseek_llm


class ChatInput(BaseModel):
    """聊天输入模型"""
    messages: List[Dict[str, str]] = Field(
        description="聊天消息列表，每个消息包含role和content字段",
        example=[
            {"role": "user", "content": "你好，请介绍一下你自己"}
        ]
    )
    
    def to_langchain_messages(self) -> List[BaseMessage]:
        """转换为LangChain消息格式"""
        lc_messages = []
        for msg in self.messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
        return lc_messages


class ChatOutput(BaseModel):
    """聊天输出模型"""
    content: str = Field(description="AI回复内容")
    

class ChatAPIDemo:
    """LangServe聊天API演示类"""
    
    def __init__(self):
        """初始化聊天API演示"""
        self.app = None
        self.chat_chain = None
        self._setup_chain()
        self._setup_app()
    
    def _setup_chain(self):
        """设置聊天链"""
        try:
            # 加载DeepSeek配置
            config = load_deepseek_config()
            if not config["api_key"]:
                raise ValueError("DEEPSEEK_API_KEY环境变量未设置")
            
            # 创建DeepSeek LLM
            llm = create_deepseek_llm(
                model="deepseek-chat",
                temperature=0.7,
                max_tokens=2000
            )
            
            # 创建聊天提示模板
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", 
                 "你是一个友好、专业的AI助手。请用中文回答用户的问题，"
                 "提供准确、有用的信息。如果不确定答案，请诚实地说明。"),
                MessagesPlaceholder(variable_name="messages")
            ])
            
            # 创建聊天链
            self.chat_chain = (
                chat_prompt 
                | llm 
                | StrOutputParser()
            )
            
            print("✅ 聊天链创建成功")
            
        except Exception as e:
            print(f"❌ 创建聊天链失败: {e}")
            raise
    
    def _setup_app(self):
        """设置FastAPI应用"""
        # 创建FastAPI应用
        self.app = FastAPI(
            title="LangServe 聊天API演示",
            description="使用LangServe将LangChain聊天机器人转换为REST API",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # 添加CORS中间件
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # 生产环境中应该限制具体域名
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 添加健康检查端点
        @self.app.get("/health")
        async def health_check():
            """健康检查端点"""
            return {"status": "healthy", "service": "langserve-chat-api"}
        
        # 添加根路径信息
        @self.app.get("/")
        async def root():
            """根路径信息"""
            return {
                "message": "LangServe 聊天API演示",
                "docs": "/docs",
                "chat_playground": "/chat/playground",
                "chat_api": "/chat/"
            }
        
        # 使用LangServe添加聊天路由
        # 这会自动创建以下端点：
        # - POST /chat/invoke - 单次调用
        # - POST /chat/stream - 流式调用  
        # - POST /chat/batch - 批量调用
        # - GET /chat/playground - 交互式界面
        add_routes(
            self.app,
            self.chat_chain,
            path="/chat",
            input_type=ChatInput,
            playground_type="chat",  # 启用聊天界面
        )
        
        print("✅ FastAPI应用设置完成")
    
    def run_server(self, host: str = "localhost", port: int = 8000):
        """启动LangServe服务器
        
        Args:
            host: 服务器主机地址，默认localhost
            port: 服务器端口，默认8000
        """
        print("🚀 启动LangServe聊天API服务器...")
        print(f"📍 服务地址: http://{host}:{port}")
        print(f"📚 API文档: http://{host}:{port}/docs")
        print(f"🎮 交互界面: http://{host}:{port}/chat/playground")
        print(f"🔗 API端点: http://{host}:{port}/chat/")
        print("\n按 Ctrl+C 停止服务器")
        
        try:
            uvicorn.run(
                self.app,
                host="127.0.0.1",
                port=port,
                log_level="info"
            )
        except KeyboardInterrupt:
            print("\n👋 服务器已停止")
        except Exception as e:
            print(f"❌ 服务器启动失败: {e}")


def test_chain_locally():
    """本地测试聊天链"""
    print("🧪 本地测试聊天链...")
    
    try:
        demo = ChatAPIDemo()
        
        # 测试消息
        test_input = ChatInput(messages=[
            {"role": "user", "content": "你好，请简单介绍一下你自己"}
        ])
        
        # 调用聊天链
        messages = test_input.to_langchain_messages()
        response = demo.chat_chain.invoke({"messages": messages})
        
        print(f"✅ 测试成功")
        print(f"用户: {test_input.messages[0]['content']}")
        print(f"AI: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("🤖 LangServe 聊天API演示")
    print("=" * 60)
    
    # 首先进行本地测试
    if not test_chain_locally():
        print("❌ 本地测试失败，请检查配置")
        return
    
    print("\n" + "=" * 60)
    
    # 创建并运行API服务
    try:
        demo = ChatAPIDemo()
        demo.run_server()
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("\n请检查：")
        print("1. DEEPSEEK_API_KEY环境变量是否正确设置")
        print("2. 网络连接是否正常")
        print("3. 端口8000是否被占用")


if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="LangServe聊天API演示")
    parser.add_argument("--host", default="localhost", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    args = parser.parse_args()
    
    # 创建演示实例
    demo = ChatAPIDemo()
    
    # 运行服务器
    demo.run_server(host=args.host, port=args.port)