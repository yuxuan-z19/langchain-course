#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen Function Calling 演示

本文件演示如何使用 Qwen 模型进行 Function Calling，
严格按照 Qwen 的 JSON Schema 规范定义工具。

Qwen 工具定义规范：
- type 字段固定为 "function"
- function 字段包含 name、description 和 parameters
- parameters 使用标准 JSON Schema 格式
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openai import OpenAI
from utils.config import load_environment


class QwenFunctionCallingDemo:
    """Qwen Function Calling 演示类"""
    
    def __init__(self):
        """初始化 Qwen 客户端"""
        config = load_environment()
        
        # 使用 Qwen 配置
        self.client = OpenAI(
            api_key=config.qwen_api_key,
            base_url=config.qwen_base_url
        )
        
        self.model = "qwen-plus"  # 使用 Qwen Plus 模型
        
        # 定义工具（严格按照 Qwen 规范）
        self.tools = self._define_tools()
        
        print(f"✅ Qwen Function Calling 演示初始化完成")
        print(f"📝 模型: {self.model}")
        print(f"🔧 工具数量: {len(self.tools)}")
    
    def _define_tools(self) -> List[Dict[str, Any]]:
        """定义工具，严格按照 Qwen JSON Schema 规范"""
        
        tools = [
            # 计算器工具
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "执行基本的数学计算，支持加减乘除运算。当你需要进行数学计算时非常有用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "要计算的数学表达式，例如：'2+3*4'、'10/2'、'(5+3)*2'等。"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            },
            
            # 时间查询工具
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "获取当前的日期和时间信息。当你需要知道现在的时间时非常有用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "format": {
                                "type": "string",
                                "description": "时间格式，可选值：'datetime'（完整日期时间）、'date'（仅日期）、'time'（仅时间）",
                                "enum": ["datetime", "date", "time"]
                            }
                        },
                        "required": ["format"]
                    }
                }
            },
            
            # 文件操作工具
            {
                "type": "function",
                "function": {
                    "name": "file_operation",
                    "description": "执行文件操作，包括读取和写入文件。当你需要操作文件时非常有用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "description": "要执行的操作类型",
                                "enum": ["read", "write"]
                            },
                            "filename": {
                                "type": "string",
                                "description": "文件名或文件路径"
                            },
                            "content": {
                                "type": "string",
                                "description": "要写入的内容（仅在写入操作时需要）"
                            }
                        },
                        "required": ["operation", "filename"]
                    }
                }
            }
        ]
        
        return tools
    
    def _execute_function(self, function_name: str, arguments: Dict[str, Any]) -> str:
        """执行函数调用"""
        
        try:
            if function_name == "calculator":
                return self._calculator(arguments)
            elif function_name == "get_current_time":
                return self._get_current_time(arguments)
            elif function_name == "file_operation":
                return self._file_operation(arguments)
            else:
                return f"❌ 未知的函数: {function_name}"
                
        except Exception as e:
            return f"❌ 执行函数 {function_name} 时出错: {str(e)}"
    
    def _calculator(self, arguments: Dict[str, Any]) -> str:
        """计算器工具实现"""
        expression = arguments.get("expression", "")
        
        if not expression:
            return "❌ 缺少计算表达式"
        
        try:
            # 安全的数学表达式计算
            allowed_chars = set("0123456789+-*/().")
            if not all(c in allowed_chars or c.isspace() for c in expression):
                return "❌ 表达式包含不允许的字符"
            
            result = eval(expression)
            return f"🧮 计算结果: {expression} = {result}"
            
        except Exception as e:
            return f"❌ 计算错误: {str(e)}"
    
    def _get_current_time(self, arguments: Dict[str, Any]) -> str:
        """时间查询工具实现"""
        format_type = arguments.get("format", "datetime")
        
        now = datetime.now()
        
        if format_type == "datetime":
            return f"🕐 当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"
        elif format_type == "date":
            return f"📅 当前日期: {now.strftime('%Y-%m-%d')}"
        elif format_type == "time":
            return f"⏰ 当前时间: {now.strftime('%H:%M:%S')}"
        else:
            return f"❌ 不支持的时间格式: {format_type}"
    
    def _file_operation(self, arguments: Dict[str, Any]) -> str:
        """文件操作工具实现"""
        operation = arguments.get("operation")
        filename = arguments.get("filename")
        content = arguments.get("content", "")
        
        if not operation or not filename:
            return "❌ 缺少必要的参数"
        
        try:
            if operation == "read":
                if os.path.exists(filename):
                    with open(filename, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    return f"📖 文件内容:\n{file_content}"
                else:
                    return f"❌ 文件不存在: {filename}"
                    
            elif operation == "write":
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f"✅ 成功写入文件: {filename}"
                
            else:
                return f"❌ 不支持的操作: {operation}"
                
        except Exception as e:
            return f"❌ 文件操作错误: {str(e)}"
    
    def chat_with_tools(self, user_message: str) -> str:
        """与 Qwen 模型进行工具调用对话"""
        
        print(f"\n👤 用户: {user_message}")
        
        try:
            # 调用 Qwen API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个有用的助手，可以使用提供的工具来帮助用户。请根据用户的需求选择合适的工具。"
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                tools=self.tools,
                tool_choice="auto",
                temperature=0.1
            )
            
            message = response.choices[0].message
            
            # 检查是否有工具调用
            if message.tool_calls:
                print(f"🔧 模型选择使用工具: {len(message.tool_calls)} 个")
                
                # 执行工具调用
                tool_results = []
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    print(f"📞 调用工具: {function_name}")
                    print(f"📋 参数: {arguments}")
                    
                    result = self._execute_function(function_name, arguments)
                    tool_results.append(result)
                    print(f"📤 结果: {result}")
                
                # 获取最终回复
                final_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个有用的助手，可以使用提供的工具来帮助用户。请根据工具执行结果给出最终回复。"
                        },
                        {
                            "role": "user",
                            "content": user_message
                        },
                        {
                            "role": "assistant",
                            "content": f"我使用了工具并得到以下结果：{'; '.join(tool_results)}"
                        }
                    ],
                    temperature=0.1
                )
                
                final_answer = final_response.choices[0].message.content
                print(f"🤖 Qwen: {final_answer}")
                return final_answer
                
            else:
                # 没有工具调用，直接返回回复
                answer = message.content
                print(f"🤖 Qwen: {answer}")
                return answer
                
        except Exception as e:
            error_msg = f"❌ 调用 Qwen API 时出错: {str(e)}"
            print(error_msg)
            return error_msg
    
    def run_demo(self):
        """运行演示"""
        
        print("\n" + "="*60)
        print("🚀 Qwen Function Calling 演示开始")
        print("="*60)
        
        # 测试用例
        test_cases = [
            "帮我计算 15 * 8 + 32 的结果",
            "现在几点了？",
            "请创建一个名为 test_qwen.txt 的文件，内容是 'Hello from Qwen!'",
            "读取刚才创建的 test_qwen.txt 文件内容",
            "计算 (100 - 25) / 5 的值，然后告诉我现在的日期"
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 测试用例 {i}: {test_case}")
            print("-" * 50)
            
            try:
                result = self.chat_with_tools(test_case)
                print(f"✅ 测试用例 {i} 完成")
                
            except Exception as e:
                print(f"❌ 测试用例 {i} 失败: {str(e)}")
            
            # 添加延迟避免 API 限制
            if i < len(test_cases):
                time.sleep(1)
        
        print("\n" + "="*60)
        print("🎉 Qwen Function Calling 演示完成")
        print("="*60)


def main():
    """主函数"""
    try:
        demo = QwenFunctionCallingDemo()
        demo.run_demo()
        
    except Exception as e:
        print(f"❌ 演示运行失败: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())