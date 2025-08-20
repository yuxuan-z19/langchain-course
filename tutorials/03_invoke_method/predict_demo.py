#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain 第三部分：invoke 接口演示（现代最佳实践）

本脚本演示：
1. ChatModel 的现代化使用方式
2. invoke 接口的使用（替代已弃用的 predict 方法）
3. 字符串和消息两种输入方式
4. 参数传递和配置覆盖
5. 实际的 DeepSeek API 调用示例

运行前请确保：
- 已安装依赖: pip install -r requirements.txt
- 已配置 DeepSeek API 密钥
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.config import load_environment
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

def print_section(title):
    """打印章节标题"""
    print("\n" + "="*50)
    print(f" {title} ")
    print("="*50)

def print_subsection(title):
    """打印子章节标题"""
    print(f"\n--- {title} ---")

def demonstrate_model_initialization():
    """演示模型初始化"""
    print_section("1. 模型初始化演示")
    
    # 加载配置
    config = load_environment()
    
    print("正在初始化 ChatModel...")
    
    # 初始化 ChatModel（使用 DeepSeek API）
    chat_model = ChatOpenAI(
        api_key=config.deepseek_api_key,
        base_url=config.deepseek_base_url,
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=200
    )
    
    print("✅ 模型初始化完成！")
    print(f"ChatModel 类型: {type(chat_model).__name__}")
    print(f"模型名称: {chat_model.model_name}")
    print(f"API 基础URL: {chat_model.openai_api_base}")
    
    return chat_model

def demonstrate_string_invoke(chat_model):
    """演示字符串输入的 invoke 方法"""
    print_section("2. 字符串输入 invoke 演示")
    
    # 测试文本
    text = "制造多彩袜子的公司的好名字是什么？"
    print(f"输入文本: {text}")
    
    try:
        print_subsection("ChatModel.invoke(string) 调用")
        result = chat_model.invoke(text)
        print(f"输出类型: {type(result).__name__}")
        print(f"输出内容: {result.content}")
        
        print_subsection("分析")
        print("✅ invoke 方法可以直接接受字符串输入")
        print("✅ 返回 AIMessage 对象，包含 content 属性")
        print("✅ 这是现代 LangChain 的推荐方式")
        
    except Exception as e:
        print(f"❌ 字符串 invoke 调用失败: {e}")

def demonstrate_messages_invoke(chat_model):
    """演示消息列表输入的 invoke 方法"""
    print_section("3. 消息列表输入 invoke 演示")
    
    # 创建消息列表
    text = "制造多彩袜子的公司的好名字是什么？"
    messages = [HumanMessage(content=text)]
    
    print(f"输入消息: {[msg.content for msg in messages]}")
    print(f"消息类型: {[type(msg).__name__ for msg in messages]}")
    
    try:
        print_subsection("ChatModel.invoke(messages) 调用")
        result = chat_model.invoke(messages)
        print(f"输出类型: {type(result).__name__}")
        print(f"输出内容: {result.content}")
        
        print_subsection("分析")
        print("✅ invoke 方法也可以接受消息列表")
        print("✅ 返回相同的 AIMessage 对象")
        print("✅ 消息列表方式更适合复杂对话")
        
    except Exception as e:
        print(f"❌ 消息列表 invoke 调用失败: {e}")

def demonstrate_complex_messages(chat_model):
    """演示复杂消息结构"""
    print_section("4. 复杂消息结构演示")
    
    # 创建包含系统消息的对话
    messages = [
        SystemMessage(content="你是一个创意营销专家，专门为公司起名字。请提供简洁、有创意的建议。"),
        HumanMessage(content="我需要为一家制造多彩袜子的公司起个名字")
    ]
    
    print("消息结构:")
    for i, msg in enumerate(messages):
        print(f"  {i+1}. {type(msg).__name__}: {msg.content}")
    
    try:
        result = chat_model.invoke(messages)
        print(f"\n输出: {result.content}")
        print(f"输出类型: {type(result).__name__}")
        
        print_subsection("添加 AI 回复到对话历史")
        # 将 AI 回复添加到对话历史
        messages.append(result)
        messages.append(HumanMessage(content="请再提供3个更有趣的名字"))
        
        print("更新后的消息结构:")
        for i, msg in enumerate(messages):
            content_preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            print(f"  {i+1}. {type(msg).__name__}: {content_preview}")
        
        # 继续对话
        final_result = chat_model.invoke(messages)
        print(f"\n最终输出: {final_result.content}")
        
    except Exception as e:
        print(f"❌ 复杂消息演示失败: {e}")

def demonstrate_parameter_passing(chat_model):
    """演示参数传递"""
    print_section("5. 参数传递演示")
    
    message = [HumanMessage(content="讲一个关于程序员的笑话")]
    
    print("测试不同的 temperature 值对输出的影响:")
    
    temperatures = [0.1, 0.7, 1.0]
    
    for temp in temperatures:
        try:
            print_subsection(f"Temperature = {temp}")
            # 创建临时模型实例以测试不同参数
            config = load_environment()
            temp_model = ChatOpenAI(
                api_key=config.deepseek_api_key,
                base_url=config.deepseek_base_url,
                model="deepseek-chat",
                temperature=temp,
                max_tokens=150
            )
            result = temp_model.invoke(message)
            print(f"输出: {result.content}")
            
        except Exception as e:
            print(f"❌ Temperature {temp} 调用失败: {e}")
    
    print_subsection("参数说明")
    print("- temperature=0.1: 输出更确定性，重复性高")
    print("- temperature=0.7: 平衡创造性和一致性")
    print("- temperature=1.0: 输出更有创造性，随机性高")
    print("- max_tokens: 限制输出长度")
    print("\n注意：现代 LangChain 推荐在模型初始化时设置参数")

def demonstrate_method_comparison():
    """演示新旧方法的对比"""
    print_section("6. 新旧方法对比")
    
    print("🔄 LangChain 方法演进:")
    print("\n❌ 已弃用的方法:")
    print("  - model.predict(text)")
    print("  - model.predict_messages(messages)")
    print("  - 这些方法在 langchain-core 0.1.7 中被弃用")
    
    print("\n✅ 现代推荐方法:")
    print("  - model.invoke(text)")
    print("  - model.invoke(messages)")
    print("  - 统一的接口，支持字符串和消息列表")
    
    print("\n📝 invoke 方法优势:")
    print("  - 统一接口: 字符串和消息都用 invoke")
    print("  - 类型安全: 返回明确的 AIMessage 对象")
    print("  - 更好的错误处理")
    print("  - 支持流式输出（stream=True）")
    print("  - 支持异步调用（ainvoke）")
    
    print("\n🎯 使用建议:")
    print("  - 简单文本生成 → model.invoke('你的问题')")
    print("  - 对话系统 → model.invoke([SystemMessage(...), HumanMessage(...)])")
    print("  - 多轮对话 → 维护消息历史列表")
    print("  - 流式输出 → model.stream(messages)")

def main():
    """主函数"""
    print("🚀 LangChain 现代化 invoke 接口演示")
    print("本演示展示 LangChain 的现代最佳实践")
    
    try:
        # 1. 模型初始化
        chat_model = demonstrate_model_initialization()
        
        # 2. 字符串输入演示
        demonstrate_string_invoke(chat_model)
        
        # 3. 消息列表输入演示
        demonstrate_messages_invoke(chat_model)
        
        # 4. 复杂消息结构演示
        demonstrate_complex_messages(chat_model)
        
        # 5. 参数传递演示
        demonstrate_parameter_passing(chat_model)
        
        # 6. 方法对比
        demonstrate_method_comparison()
        
        print_section("演示完成")
        print("🎉 恭喜！您已经掌握了 LangChain 的现代化接口")
        print("\n📚 学习要点:")
        print("1. ✅ 使用 invoke 替代已弃用的 predict 方法")
        print("2. ✅ ChatOpenAI 支持字符串和消息列表输入")
        print("3. ✅ 返回的 AIMessage 对象包含丰富信息")
        print("4. ✅ 参数在模型初始化时设置")
        print("5. ✅ 现代 LangChain 更加类型安全和易用")
        
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        print("\n🔧 故障排除建议:")
        print("1. 检查 DeepSeek API 密钥是否正确配置")
        print("2. 确认网络连接正常")
        print("3. 验证 .env 文件中的配置")
        print("4. 检查 API 配额是否充足")
        print("5. 确保已安装最新版本的依赖: pip install -r requirements.txt")

if __name__ == "__main__":
    main()