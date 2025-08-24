#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph 人工在环 (Human-in-the-Loop) 功能演示

本演示展示如何使用 LangGraph 的 interrupt 函数实现人工在环功能，
包括暂停执行、等待人工输入、恢复执行等完整流程。

主要功能：
1. human_assistance 工具 - 使用 interrupt 函数请求人工协助
2. 搜索工具集成 - 结合 Tavily 搜索和人工辅助
3. 暂停/恢复机制 - 演示完整的暂停恢复流程
4. 交互式界面 - 提供用户友好的交互体验
5. 状态管理 - 基于检查点的状态保存和恢复
"""

import os
import sys
import json
from typing import Annotated, Dict, Any, List
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_tavily import TavilySearch
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

# 导入 LLM 工厂函数
from utils.llm_factory import create_llm_from_config, get_available_models


class State(TypedDict):
    """图状态定义"""
    messages: Annotated[List, add_messages]
    human_input_history: List[Dict[str, Any]]  # 记录人工输入历史
    interrupt_count: int  # 中断次数统计


class HumanInLoopChatbot:
    """带人工在环功能的聊天机器人"""
    
    def __init__(self, provider: str = "deepseek"):
        """初始化聊天机器人
        
        Args:
            provider: LLM提供商，支持 'deepseek', 'openai' 等
        """
        self.provider = provider
        self.llm = None
        self.graph = None
        self.memory = MemorySaver()
        
        # 初始化工具
        self.tools = self._create_tools()
        
        # 初始化LLM
        self._initialize_llm()
        
        # 构建图
        self._build_graph()
    
    def _create_tools(self):
        """创建工具列表"""
        
        @tool
        def human_assistance(query: str) -> str:
            """请求人工协助的工具
            
            Args:
                query: 需要人工协助的问题或请求
                
            Returns:
                人工提供的回复
            """
            print(f"\n🤖 AI 请求人工协助：{query}")
            print("⏸️  执行已暂停，等待人工输入...")
            
            # 使用 interrupt 函数暂停执行并等待人工输入
            human_response = interrupt({
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "type": "human_assistance_request"
            })
            
            # 返回人工提供的数据
            return human_response.get("data", "未收到有效的人工输入")
        
        @tool
        def search_web(query: str) -> str:
            """搜索网络信息
            
            Args:
                query: 搜索查询
                
            Returns:
                搜索结果
            """
            try:
                # 检查是否配置了 Tavily API Key
                tavily_api_key = os.getenv("TAVILY_API_KEY")
                if not tavily_api_key:
                    return "❌ 未配置 Tavily API Key，无法进行网络搜索。请在 .env 文件中设置 TAVILY_API_KEY"
                
                # 使用与第22小节相同的配置方式
                search_tool = TavilySearch(
                    max_results=3,
                    search_depth="advanced",
                    include_answer=True,
                    include_raw_content=False
                )
                
                # 执行搜索
                results = search_tool.invoke(query)
                
                if not results:
                    return "❌ 未找到相关搜索结果"
                
                # 格式化搜索结果 - 处理不同格式
                formatted_output = f"🔍 搜索主题: {query}\n\n📋 搜索结果:\n"
                
                if isinstance(results, str):
                    # 如果结果是字符串，直接返回
                    formatted_output += results
                elif isinstance(results, list) and results:
                    # 如果是列表格式，格式化每个结果
                    for i, result in enumerate(results[:5], 1):  # 限制显示前5个结果
                        if isinstance(result, dict):
                            title = result.get('title', '无标题')
                            content = result.get('content', result.get('snippet', result.get('description', '无内容')))
                            url = result.get('url', '无链接')
                            
                            # 限制内容长度
                            if len(content) > 200:
                                content = content[:200] + "..."
                            
                            formatted_output += f"{i}. 📄 {title}\n"
                            formatted_output += f"   💬 {content}\n"
                            formatted_output += f"   🔗 {url}\n\n"
                        else:
                            formatted_output += f"{i}. {str(result)}\n\n"
                else:
                    # 处理其他格式
                    formatted_output += str(results)
                
                formatted_output += "\n✅ 搜索完成"
                return formatted_output
                
            except Exception as e:
                return f"搜索时发生错误: {str(e)}。请检查 TAVILY_API_KEY 是否正确配置。"
        
        @tool
        def get_current_time() -> str:
            """获取当前时间"""
            return f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return [human_assistance, search_web, get_current_time]
    
    def _build_graph(self):
        """构建 LangGraph"""
        # 创建图构建器
        graph_builder = StateGraph(State)
        
        # 定义聊天机器人节点
        def chatbot(state: State):
            """聊天机器人节点"""
            if self.llm is None:
                return {
                    "messages": [AIMessage(content="❌ LLM 未初始化，请先选择模型")],
                    "interrupt_count": state.get("interrupt_count", 0)
                }
            
            # 绑定工具到 LLM
            llm_with_tools = self.llm.bind_tools(self.tools)
            
            # 获取最近的消息来判断上下文
            messages = state["messages"]
            recent_messages = messages[-5:] if len(messages) > 5 else messages
            
            # 检查是否刚刚完成了工具调用
            has_recent_tool_result = any(
                hasattr(msg, 'type') and msg.type == 'tool' 
                for msg in recent_messages[-3:]
            )
            
            # 如果刚完成工具调用，添加指导性提示
            if has_recent_tool_result:
                # 检查是否是搜索工具的结果
                search_result_found = any(
                    hasattr(msg, 'content') and 
                    isinstance(msg.content, str) and 
                    ('搜索结果' in msg.content or '🔍' in msg.content)
                    for msg in recent_messages[-2:]
                )
                
                if search_result_found:
                    # 为搜索结果添加总结指导
                    guidance_msg = HumanMessage(content="请总结上述搜索结果，提取关键信息并给出简洁的回答。不要再次调用任何工具。")
                    messages_with_guidance = messages + [guidance_msg]
                else:
                    messages_with_guidance = messages
            else:
                messages_with_guidance = messages
            
            # 调用 LLM
            response = llm_with_tools.invoke(messages_with_guidance)
            
            # 检查是否有工具调用，如果有多个工具调用，确保不并行执行
            # 这对于包含 interrupt 的工具很重要
            if hasattr(response, 'tool_calls') and len(response.tool_calls) > 1:
                # 如果有多个工具调用，只保留第一个
                response.tool_calls = response.tool_calls[:1]
            
            return {
                "messages": [response],
                "interrupt_count": state.get("interrupt_count", 0)
            }
        
        # 添加节点
        graph_builder.add_node("chatbot", chatbot)
        
        # 创建工具节点
        tool_node = ToolNode(tools=self.tools)
        graph_builder.add_node("tools", tool_node)
        
        # 添加边
        graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
        )
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_edge(START, "chatbot")
        
        # 编译图
        self.graph = graph_builder.compile(checkpointer=self.memory)
    
    def _initialize_llm(self):
        """初始化LLM"""
        try:
            print(f"🤖 正在初始化 {self.provider} LLM...")
            self.llm = create_llm_from_config(self.provider)
            print(f"✅ {self.provider} LLM 初始化成功")
        except Exception as e:
            print(f"❌ LLM 初始化失败: {e}")
            self.llm = None
    
    def chat_with_human_in_loop(self, message: str, thread_id: str = "default") -> List[Dict]:
        """带人工在环的聊天
        
        Args:
            message: 用户消息
            thread_id: 线程 ID
            
        Returns:
            对话事件列表
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # 流式执行图
            events = []
            for event in self.graph.stream(
                {"messages": [HumanMessage(content=message)]},
                config,
                stream_mode="values"
            ):
                events.append(event)
                
                # 显示最新消息
                if "messages" in event and event["messages"]:
                    latest_message = event["messages"][-1]
                    if hasattr(latest_message, 'content') and latest_message.content:
                        if isinstance(latest_message, AIMessage):
                            print(f"\n🤖 AI: {latest_message.content}")
                        elif isinstance(latest_message, HumanMessage):
                            print(f"\n👤 用户: {latest_message.content}")
            
            return events
            
        except Exception as e:
            print(f"❌ 聊天过程中发生错误: {e}")
            return []
    
    def resume_execution(self, human_input: str, thread_id: str = "default") -> List[Dict]:
        """恢复执行并提供人工输入
        
        Args:
            human_input: 人工输入的数据
            thread_id: 线程 ID
            
        Returns:
            恢复执行后的事件列表
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            print(f"\n▶️  恢复执行，人工输入: {human_input}")
            
            # 使用 Command 对象恢复执行
            events = []
            for event in self.graph.stream(
                Command(resume={"data": human_input}),
                config,
                stream_mode="values"
            ):
                events.append(event)
                
                # 显示最新消息
                if "messages" in event and event["messages"]:
                    latest_message = event["messages"][-1]
                    if hasattr(latest_message, 'content') and latest_message.content:
                        if isinstance(latest_message, AIMessage):
                            print(f"\n🤖 AI: {latest_message.content}")
            
            return events
            
        except Exception as e:
            print(f"❌ 恢复执行时发生错误: {e}")
            return []
    
    def get_conversation_state(self, thread_id: str = "default") -> Dict:
        """获取对话状态
        
        Args:
            thread_id: 线程 ID
            
        Returns:
            对话状态信息
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            snapshot = self.graph.get_state(config)
            
            return {
                "values": snapshot.values,
                "next": snapshot.next,
                "metadata": {
                    "step": getattr(snapshot.metadata, 'step', 0),
                    "writes": getattr(snapshot.metadata, 'writes', {})
                },
                "created_at": getattr(snapshot, 'created_at', None),
                "parent_config": getattr(snapshot, 'parent_config', None)
            }
            
        except Exception as e:
            print(f"❌ 获取状态失败: {e}")
            return {}
    
    def check_interrupt_status(self, thread_id: str = "default") -> bool:
        """检查是否处于中断状态
        
        Args:
            thread_id: 线程 ID
            
        Returns:
            是否处于中断状态
        """
        state = self.get_conversation_state(thread_id)
        return bool(state.get("next"))


def demonstrate_basic_human_in_loop():
    """演示基本的人工在环功能 - 用户选择处理方式"""
    print("\n" + "="*60)
    print("🔄 基本人工在环功能演示")
    print("="*60)
    
    chatbot = HumanInLoopChatbot()
    
    # 检查 LLM 是否初始化成功
    if chatbot.llm is None:
        print("❌ LLM 初始化失败，跳过演示")
        return
    
    # 发送需要用户选择处理方式的请求
    print("\n📝 发送请求: 需要用户选择如何处理任务")
    events = chatbot.chat_with_human_in_loop(
        "我有一个重要任务需要处理，但有多种处理方式可选。请调用human_assistance工具来请求用户选择处理方式。获得用户选择后，请根据用户的选择给出相应的处理建议，不要再次调用human_assistance工具。",
        "demo_thread_1"
    )
    
    # 检查是否处于中断状态并等待真实用户输入
    while chatbot.check_interrupt_status("demo_thread_1"):
        print("\n✅ 触发了处理方式选择请求")
        print("⏸️  AI 正在等待您的选择...")
        print("💡 提示: 请选择您希望的处理方式")
        
        print("\n可选的处理方式:")
        print("1. 快速处理 - 使用默认设置快速完成")
        print("2. 详细处理 - 进行全面分析和处理")
        print("3. 自定义处理 - 根据特定需求定制")
        print("4. 延后处理 - 稍后再处理")
        
        try:
            # 真正等待用户选择处理方式
            choice = input("\n🎯 请选择处理方式 (1-4): ").strip()
            
            if choice in ['1', '2', '3', '4']:
                choice_map = {
                    '1': '快速处理',
                    '2': '详细处理', 
                    '3': '自定义处理',
                    '4': '延后处理'
                }
                selected_method = choice_map[choice]
                # 恢复执行
                chatbot.resume_execution(f"用户选择: {selected_method}", "demo_thread_1")
                break
            else:
                print("❌ 无效选择，请输入 1-4 之间的数字")
                
        except KeyboardInterrupt:
            print("\n\n👋 演示被中断")
            break
        except Exception as e:
            print(f"\n❌ 输入处理错误: {e}")
            break


def demonstrate_search_and_human_assistance():
    """演示搜索工具和人工协助的结合使用 - 用户提供搜索主题"""
    print("\n" + "="*60)
    print("🔍 搜索工具 + 人工协助演示")
    print("="*60)
    
    chatbot = HumanInLoopChatbot()
    
    # 检查 LLM 是否初始化成功
    if chatbot.llm is None:
        print("❌ LLM 初始化失败，跳过演示")
        return
    
    # 发送需要用户提供搜索主题的请求
    print("\n📝 发送请求: 请求用户提供搜索主题")
    events = chatbot.chat_with_human_in_loop(
        "我需要为用户搜索信息，但需要用户告诉我具体要搜索什么主题。请调用human_assistance工具来请求用户提供搜索主题。获得用户提供的搜索主题后，请立即使用search_web工具进行搜索，然后总结搜索结果给用户，不要再次调用human_assistance工具。",
        "demo_thread_2"
    )
    
    # 检查是否处于中断状态并等待真实用户输入
    while chatbot.check_interrupt_status("demo_thread_2"):
        print("\n✅ 触发了搜索主题请求")
        print("⏸️  AI 正在等待您提供搜索主题...")
        print("💡 提示: 请告诉我您想搜索什么内容")
        
        try:
            # 真正等待用户输入搜索主题
            search_topic = input("\n🔍 请输入您想搜索的主题 (例如: 人工智能最新发展): ").strip()
            
            if search_topic:
                # 恢复执行，AI将使用用户提供的主题进行搜索
                chatbot.resume_execution(f"用户想搜索: {search_topic}", "demo_thread_2")
                break
            else:
                print("❌ 搜索主题不能为空，请重新输入")
                
        except KeyboardInterrupt:
            print("\n\n👋 演示被中断")
            break
        except Exception as e:
            print(f"\n❌ 输入处理错误: {e}")
            break


def demonstrate_user_confirmation():
    """演示用户确认场景 - AI完成操作后请求用户确认"""
    print("\n" + "="*60)
    print("✅ 用户确认演示")
    print("="*60)
    
    chatbot = HumanInLoopChatbot()
    
    # 检查 LLM 是否初始化成功
    if chatbot.llm is None:
        print("❌ LLM 初始化失败，跳过演示")
        return
    
    # 发送需要用户确认的请求
    print("\n📝 发送请求: AI完成分析后请求用户确认")
    events = chatbot.chat_with_human_in_loop(
        "我已经完成了数据分析，得出了一些重要结论。在继续下一步操作之前，请调用human_assistance工具来请求用户确认这些结论是否正确。获得用户确认后，请根据用户的决定给出相应的后续建议，不要再次调用human_assistance工具。",
        "demo_thread_3"
    )
    
    # 检查是否处于中断状态并等待真实用户输入
    while chatbot.check_interrupt_status("demo_thread_3"):
        print("\n✅ 触发了用户确认请求")
        print("⏸️  AI 正在等待您的确认...")
        print("💡 提示: AI已完成分析，请确认是否继续")
        
        print("\n📊 分析结果摘要:")
        print("- 数据趋势显示上升态势")
        print("- 预测准确率达到85%")
        print("- 建议采取积极策略")
        
        print("\n请选择您的决定:")
        print("1. 确认结果，继续执行")
        print("2. 需要更多信息")
        print("3. 拒绝，重新分析")
        print("4. 暂停，稍后决定")
        
        try:
            # 真正等待用户确认
            choice = input("\n✅ 请选择您的决定 (1-4): ").strip()
            
            if choice in ['1', '2', '3', '4']:
                choice_map = {
                    '1': '确认结果，继续执行',
                    '2': '需要更多信息',
                    '3': '拒绝，重新分析',
                    '4': '暂停，稍后决定'
                }
                user_decision = choice_map[choice]
                # 恢复执行
                chatbot.resume_execution(f"用户决定: {user_decision}", "demo_thread_3")
                break
            else:
                print("❌ 无效选择，请输入 1-4 之间的数字")
                
        except KeyboardInterrupt:
            print("\n\n👋 演示被中断")
            break
        except Exception as e:
            print(f"\n❌ 输入处理错误: {e}")
            break


def interactive_human_in_loop_chat():
    """交互式人工在环聊天"""
    print("\n" + "="*60)
    print("💬 交互式人工在环聊天")
    print("="*60)
    print("输入 'quit' 退出，输入 'help' 查看帮助")
    
    # 选择 LLM 提供商
    print("\n请选择 LLM 提供商:")
    print("1. DeepSeek")
    print("2. OpenAI")
    
    choice = input("请输入选择 (1-2): ").strip()
    
    if choice == "1":
        chatbot = HumanInLoopChatbot("deepseek")
    elif choice == "2":
        chatbot = HumanInLoopChatbot("openai")
    else:
        print("❌ 无效选择")
        return
    
    # 检查 LLM 是否初始化成功
    if chatbot.llm is None:
        print("❌ LLM 初始化失败，请检查API密钥配置")
        return
    
    thread_id = f"interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n👤 您: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 再见！")
                break
            elif user_input.lower() == 'help':
                print("\n📖 帮助信息:")
                print("- 输入任何问题开始对话")
                print("- AI 可能会请求人工协助，此时需要您提供输入")
                print("- 输入 'quit' 退出聊天")
                print("- 输入 'status' 查看当前状态")
                continue
            elif user_input.lower() == 'status':
                state = chatbot.get_conversation_state(thread_id)
                print(f"\n📊 当前状态: {json.dumps(state, indent=2, ensure_ascii=False)}")
                continue
            
            if not user_input:
                continue
            
            # 发送消息
            events = chatbot.chat_with_human_in_loop(user_input, thread_id)
            
            # 检查是否需要人工输入
            while chatbot.check_interrupt_status(thread_id):
                print("\n⏸️  AI 正在等待您的输入...")
                human_input = input("🧑‍💼 请提供协助: ").strip()
                
                if human_input:
                    chatbot.resume_execution(human_input, thread_id)
                else:
                    print("❌ 输入不能为空")
                    
        except KeyboardInterrupt:
            print("\n\n👋 聊天被中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")


def main():
    """主函数"""
    print("🚀 LangGraph 人工在环功能演示")
    print("本演示展示如何使用 interrupt 函数实现人工在环功能")
    print("\n💡 演示场景说明:")
    print("- 用户选择处理方式: AI提供选项，用户选择")
    print("- 用户提供搜索主题: AI请求用户输入搜索内容")
    print("- 用户确认结果: AI完成分析后请求用户确认")
    
    while True:
        print("\n" + "="*50)
        print("请选择演示模式:")
        print("1. 用户选择处理方式演示")
        print("2. 用户提供搜索主题演示")
        print("3. 用户确认结果演示")
        print("4. 交互式人工在环聊天")
        print("5. 退出")
        
        choice = input("\n请输入选择 (1-5): ").strip()
        
        if choice == "1":
            demonstrate_basic_human_in_loop()
        elif choice == "2":
            demonstrate_search_and_human_assistance()
        elif choice == "3":
            demonstrate_user_confirmation()
        elif choice == "4":
            interactive_human_in_loop_chat()
        elif choice == "5":
            print("👋 再见！")
            break
        else:
            print("❌ 无效选择，请重新输入")


if __name__ == "__main__":
    main()