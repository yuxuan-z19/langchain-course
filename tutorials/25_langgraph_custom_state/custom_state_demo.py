#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第25小节：LangGraph自定义状态演示

本模块演示如何在LangGraph中使用自定义状态来构建复杂的聊天机器人。
主要功能包括：
1. 自定义State类定义
2. 实体信息查找和存储
3. 人工审查验证机制
4. 状态更新和管理

作者：LangChain课程
日期：2024年1月
"""

import os
import sys
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
import asyncio
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# LangChain和LangGraph相关导入
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from langgraph.errors import NodeInterrupt

# 工具导入
try:
    from langchain_tavily import TavilySearch
except ImportError:
    print("⚠️ 警告：未安装langchain-tavily，搜索功能将不可用")
    print("请运行：pip install langchain-tavily")
    TavilySearch = None

# 导入LLM工厂
try:
    from utils.llm_factory import create_deepseek_llm, create_openai_llm
except ImportError:
    print("⚠️ 警告：无法导入LLM工厂，请确保utils/llm_factory.py存在")
    create_deepseek_llm = None
    create_openai_llm = None


class State(TypedDict):
    """自定义状态类，包含消息和实体信息"""
    # 消息列表，使用add_messages进行自动合并
    messages: Annotated[List, add_messages]
    
    # 实体名称
    name: str
    
    # 实体生日/发布日期
    birthday: str
    
    # 验证状态：pending, verified, corrected
    verification_status: str
    
    # 搜索结果列表
    search_results: List[Dict[str, Any]]


class CustomStateChatbot:
    """自定义状态聊天机器人类"""
    
    def __init__(self, provider="deepseek"):
        """初始化聊天机器人"""
        self.provider = provider
        self.llm = None
        self.graph = None
        self.memory = MemorySaver()
        
        # 初始化组件
        self._initialize_llm()
        self._build_graph()
        
    def _initialize_llm(self):
        """初始化大语言模型"""
        if create_deepseek_llm is None:
            print("❌ LLM工厂不可用，请检查utils/llm_factory.py")
            self.llm = None
            return
            
        try:
            # 优先使用DeepSeek，如果失败则尝试OpenAI
            try:
                self.llm = create_deepseek_llm()
                print("✅ 成功初始化DeepSeek LLM")
            except Exception as e:
                print(f"⚠️ DeepSeek初始化失败: {e}")
                if create_openai_llm is not None:
                    self.llm = create_openai_llm()
                    print("✅ 成功初始化OpenAI LLM")
                else:
                    raise e
        except Exception as e:
            print(f"❌ LLM初始化失败: {e}")
            self.llm = None
    

    
    def _build_graph(self):
        """构建LangGraph图"""
        if self.llm is None:
            print("❌ 无法构建图：LLM未初始化")
            return None
            
        # 创建工具列表
        tools = self._create_tools()
        
        # 绑定工具到LLM
        llm_with_tools = self.llm.bind_tools(tools)
        
        # 创建状态图
        graph = StateGraph(State)
        
        # 添加节点
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", ToolNode(tools))
        
        # 添加边
        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "tools", "end": END}
        )
        graph.add_edge("tools", "agent")
        
        # 编译图
        self.graph = graph.compile(checkpointer=self.memory)
        
        print("✅ 成功构建LangGraph图")
    
    def _agent_node(self, state: State) -> Dict[str, Any]:
        """代理节点：处理用户消息并调用工具"""
        try:
            # 绑定工具到LLM
            tools = self._create_tools()
            llm_with_tools = self.llm.bind_tools(tools)
            
            # 构建系统提示
            system_prompt = self._build_system_prompt(state)
            messages = [HumanMessage(content=system_prompt)] + state["messages"]
            
            # 调用LLM
            response = llm_with_tools.invoke(messages)
            
            return {"messages": [response]}
            
        except Exception as e:
            error_msg = f"代理节点执行失败: {e}"
            print(f"❌ {error_msg}")
            return {"messages": [AIMessage(content=error_msg)]}
    
    def _build_system_prompt(self, state: State) -> str:
        """构建系统提示词"""
        prompt = """
你是一个专门用于实体信息查找和验证的AI助手。你的任务是：

1. 当用户询问某个实体（如软件、公司、人物等）的信息时，使用search_entity_info工具进行搜索
2. 找到相关信息后，使用human_assistance工具请求人工验证
3. 根据验证结果更新状态中的实体信息
4. 提供准确、有用的回复

当前状态信息：
- 实体名称: {name}
- 生日/发布日期: {birthday}
- 验证状态: {verification_status}
- 搜索结果数量: {search_count}

请根据用户的问题，合理使用可用的工具来提供帮助。
""".format(
            name=state.get("name", "未设置"),
            birthday=state.get("birthday", "未设置"),
            verification_status=state.get("verification_status", "pending"),
            search_count=len(state.get("search_results", []))
        )
        
        return prompt
    
    def _should_continue(self, state: State) -> str:
        """判断是否继续执行工具"""
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        return "end"
    
    def _create_tools(self):
        """创建工具列表"""
        
        @tool
        def human_assistance(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
            """请求人工协助验证信息
            
            Args:
                query: 需要人工验证的问题
                tool_call_id: 工具调用ID
            
            Returns:
                Command对象，用于更新状态
            """
            print(f"\n🤔 需要人工协助验证: {query}")
            print("请输入验证结果 (或输入 'quit' 退出):")
            
            # 先更新状态
            state_update = Command(
                update={
                    "verification_status": "pending",
                    "messages": [ToolMessage(content=f"等待人工验证: {query}", tool_call_id=tool_call_id)]
                }
            )
            
            # 直接调用interrupt函数来暂停执行
            interrupt(query)
            
            # 这行代码不会执行到，因为interrupt会抛出异常
            return state_update
        
        @tool
        def search_entity_info(
            query: str, 
            tool_call_id: Annotated[str, InjectedToolCallId]
        ) -> Command:
            """搜索实体信息
            
            Args:
                query: 搜索查询
                tool_call_id: 工具调用ID（自动注入）
                
            Returns:
                Command对象，包含状态更新和工具消息
            """
            try:
                # 检查是否配置了 Tavily API Key
                tavily_api_key = os.getenv("TAVILY_API_KEY")
                if not tavily_api_key:
                    error_msg = "❌ 未配置 Tavily API Key，无法进行网络搜索。请在 .env 文件中设置 TAVILY_API_KEY"
                    return Command(
                        update={"messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)]}
                    )
                
                # 使用 TavilySearch 工具
                search_tool = TavilySearch(
                    max_results=3,
                    search_depth="advanced",
                    include_answer=True,
                    include_raw_content=False
                )
                
                # 执行搜索
                results = search_tool.invoke(query)
                
                if not results:
                    error_msg = "❌ 未找到相关搜索结果"
                    return Command(
                        update={"messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)]}
                    )
                
                # 格式化搜索结果
                formatted_output = f"🔍 搜索主题: {query}\n\n📋 搜索结果:\n"
                search_results_list = []
                
                if isinstance(results, str):
                    formatted_output += results
                    search_results_list.append({"type": "text", "content": results})
                elif isinstance(results, list) and results:
                    for i, result in enumerate(results[:3], 1):
                        if isinstance(result, dict):
                            title = result.get('title', '无标题')
                            content = result.get('content', result.get('snippet', result.get('description', '无内容')))
                            url = result.get('url', '无链接')
                            
                            if len(content) > 200:
                                content = content[:200] + "..."
                            
                            formatted_output += f"{i}. 📄 {title}\n"
                            formatted_output += f"   💬 {content}\n"
                            formatted_output += f"   🔗 {url}\n\n"
                            
                            # 添加到搜索结果列表
                            search_results_list.append({
                                "title": title,
                                "content": content,
                                "url": url
                            })
                        else:
                            formatted_output += f"{i}. {str(result)}\n\n"
                            search_results_list.append({"type": "text", "content": str(result)})
                else:
                    formatted_output += str(results)
                    search_results_list.append({"type": "text", "content": str(results)})
                
                formatted_output += "\n✅ 搜索完成"
                
                # 尝试从查询中提取实体名称
                entity_name = ""
                if "LangGraph" in query:
                    entity_name = "LangGraph"
                elif "发布日期" in query or "release" in query.lower():
                    # 尝试提取实体名称
                    words = query.replace("的发布日期", "").replace("发布日期", "").replace("release date", "").strip()
                    if words:
                        entity_name = words
                
                # 使用Command对象更新状态
                state_update = {
                    "search_results": search_results_list,
                    "messages": [ToolMessage(content=formatted_output, tool_call_id=tool_call_id)]
                }
                
                # 如果提取到了实体名称，也更新name字段
                if entity_name:
                    state_update["name"] = entity_name
                
                # 添加调试信息
                print(f"🔧 调试: 搜索结果数量: {len(search_results_list)}")
                print(f"🔧 调试: 实体名称: {entity_name}")
                print(f"🔧 调试: 状态更新: {state_update}")
                
                return Command(update=state_update)
                
            except Exception as e:
                error_msg = f"❌ 搜索时发生错误: {str(e)}。请检查 TAVILY_API_KEY 是否正确配置。"
                return Command(
                    update={"messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)]}
                )
        
        @tool
        def get_current_time() -> str:
            """获取当前时间"""
            return f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return [human_assistance, search_entity_info, get_current_time]
    
    def chat_with_custom_state(
        self, 
        message: str, 
        thread_id: str = "default",
        stream: bool = True
    ) -> Optional[Dict[str, Any]]:
        """与自定义状态聊天机器人对话
        
        Args:
            message: 用户消息
            thread_id: 线程ID
            stream: 是否流式输出
            
        Returns:
            最终状态或None
        """
        if self.graph is None:
            print("❌ 图未初始化，无法进行对话")
            return None
        
        try:
            # 配置
            config = {"configurable": {"thread_id": thread_id}}
            
            # 检查是否已有状态，如果有则保持，如果没有则初始化
            existing_state = self.get_conversation_state(thread_id)
            if existing_state:
                # 只添加新消息，保持其他状态
                input_data = {
                    "messages": [HumanMessage(content=message)]
                }
            else:
                # 首次初始化完整状态
                input_data = {
                    "messages": [HumanMessage(content=message)],
                    "name": "",
                    "birthday": "",
                    "verification_status": "pending",
                    "search_results": []
                }
            
            print(f"\n🤖 用户: {message}")
            
            if stream:
                # 流式执行
                final_state = None
                for event in self.graph.stream(input_data, config):
                    for node_name, node_output in event.items():
                        if node_name == "agent" and "messages" in node_output:
                            latest_message = node_output["messages"][-1]
                            if hasattr(latest_message, 'content') and latest_message.content:
                                print(f"🤖 AI: {latest_message.content}")
                        
                        # 保存最终状态
                        if isinstance(node_output, dict):
                            final_state = node_output
                
                return final_state
            else:
                # 非流式执行
                result = self.graph.invoke(input_data, config)
                if result and "messages" in result:
                    latest_message = result["messages"][-1]
                    if hasattr(latest_message, 'content'):
                        print(f"🤖 AI: {latest_message.content}")
                return result
                
        except NodeInterrupt:
            # 让NodeInterrupt异常传播到调用者，以便演示函数可以处理人工在环交互
            raise
        except Exception as e:
            print(f"❌ 对话失败: {e}")
            return None
    
    def get_conversation_state(self, thread_id: str = "default") -> Optional[Dict[str, Any]]:
        """获取对话状态
        
        Args:
            thread_id: 线程ID
            
        Returns:
            当前状态或None
        """
        if self.graph is None:
            return None
            
        try:
            config = {"configurable": {"thread_id": thread_id}}
            snapshot = self.graph.get_state(config)
            return snapshot.values if snapshot else None
        except Exception as e:
            print(f"❌ 获取状态失败: {e}")
            return None
    
    def check_interrupt_status(self, thread_id: str = "default") -> bool:
        """检查是否处于中断状态
        
        Args:
            thread_id: 线程ID
            
        Returns:
            True if interrupted, False otherwise
        """
        if self.graph is None:
            return False
            
        try:
            config = {"configurable": {"thread_id": thread_id}}
            snapshot = self.graph.get_state(config)
            return snapshot.next if snapshot else False
        except Exception as e:
            print(f"❌ 检查中断状态失败: {e}")
            return False
    
    def resume_execution(self, user_input: str, thread_id: str = "default") -> Optional[Dict[str, Any]]:
        """恢复执行（用于人工在环场景）
        
        Args:
            user_input: 用户输入
            thread_id: 线程ID
            
        Returns:
            执行结果或None
        """
        if self.graph is None:
            print("❌ 图未初始化，无法恢复执行")
            return None
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            
            print(f"\n🔄 恢复执行，用户输入: {user_input}")
            
            # 使用Command(resume={"data": user_input})来恢复执行
            # 这是LangGraph推荐的人工在环恢复方式
            final_state = None
            for event in self.graph.stream(Command(resume={"data": user_input}), config):
                for node_name, node_output in event.items():
                    if node_name == "agent" and "messages" in node_output:
                        latest_message = node_output["messages"][-1]
                        if hasattr(latest_message, 'content') and latest_message.content:
                            print(f"🤖 AI: {latest_message.content}")
                    
                    # 保存最终状态
                    if isinstance(node_output, dict):
                        final_state = node_output
            
            return final_state
            
        except Exception as e:
            print(f"❌ 恢复执行失败: {e}")
            return None


# 全局chatbot实例，确保状态持久化
global_chatbot = None

def get_global_chatbot():
    """获取全局chatbot实例"""
    global global_chatbot
    if global_chatbot is None:
        global_chatbot = CustomStateChatbot()
    return global_chatbot

def demonstrate_entity_lookup():
    """演示实体信息查找功能"""
    print("\n" + "="*60)
    print("🔍 演示1: 实体信息查找功能")
    print("="*60)
    
    chatbot = get_global_chatbot()
    if chatbot.graph is None:
        print("❌ 聊天机器人初始化失败")
        return
    
    # 演示查找LangGraph发布日期
    thread_id = "entity_lookup_demo"
    message = "请帮我查找LangGraph的发布日期"
    
    print(f"\n📝 演示场景: 查找LangGraph发布日期")
    print(f"用户输入: {message}")
    
    # 开始对话
    result = chatbot.chat_with_custom_state(message, thread_id)
    
    # 使用while循环检查中断状态，参考第24小节的实现
    while chatbot.check_interrupt_status(thread_id):
        print("\n⏸️ 检测到中断，需要人工验证")
        print("💡 请验证搜索结果的准确性...")
        
        try:
            user_input = input("\n👤 请输入验证结果 (正确/错误): ").strip()
            if user_input.lower() in ['正确', '错误', 'correct', 'incorrect', 'yes', 'no']:
                print(f"\n🔄 收到验证结果: {user_input}")
                # 恢复执行
                result = chatbot.resume_execution(user_input, thread_id)
            else:
                print("❌ 请输入有效的验证结果 (正确/错误)")
                continue
        except KeyboardInterrupt:
            print("\n❌ 用户取消验证")
            break
        except Exception as ex:
            print(f"❌ 处理验证结果时出错: {ex}")
            break
    
    # 显示最终状态 - 从持久化存储中获取最新状态
    final_state = chatbot.get_conversation_state(thread_id)
    if final_state:
        print("\n📊 最终状态信息:")
        print(f"实体名称: {final_state.get('name', '未设置')}")
        print(f"生日/发布日期: {final_state.get('birthday', '未设置')}")
        print(f"验证状态: {final_state.get('verification_status', 'pending')}")
        print(f"搜索结果数量: {len(final_state.get('search_results', []))}")
    else:
        print("\n📊 最终状态信息: 无状态信息")
    
    return chatbot


def demonstrate_human_verification():
    """演示人工验证流程"""
    print("\n" + "="*60)
    print("👤 演示2: 人工验证流程")
    print("="*60)
    
    chatbot = get_global_chatbot()
    if chatbot.graph is None:
        print("❌ 聊天机器人初始化失败")
        return
    
    thread_id = "human_verification_demo"
    
    print("\n📝 演示场景: 人工验证实体信息")
    print("说明: 当AI找到信息后，会请求人工验证")
    print("您可以回答 'yes' 确认信息正确，或提供正确的信息")
    
    try:
        # 开始对话
        message = "请查找并验证Python编程语言的发布日期"
        print(f"\n用户输入: {message}")
        
        result = chatbot.chat_with_custom_state(message, thread_id)
        
        # 显示最终状态
        if result:
            print("\n📊 验证后的状态信息:")
            print(f"实体名称: {result.get('name', '未设置')}")
            print(f"生日/发布日期: {result.get('birthday', '未设置')}")
            print(f"验证状态: {result.get('verification_status', 'pending')}")
            
    except NodeInterrupt as e:
        print(f"\n⏸️ 流程被中断，需要人工验证: {e}")
        print("💡 请验证AI找到的信息是否正确...")
        
        # 显示当前状态供用户参考
        current_state = chatbot.get_conversation_state(thread_id)
        if current_state:
            print("\n📋 当前找到的信息:")
            print(f"实体名称: {current_state.get('name', '未设置')}")
            print(f"生日/发布日期: {current_state.get('birthday', '未设置')}")
        
        # 等待用户验证
        while True:
            try:
                user_input = input("\n👤 请验证信息是否正确 (yes/no 或提供正确信息): ").strip()
                if user_input:
                    print(f"\n🔄 收到用户验证: {user_input}")
                    # 恢复执行
                    result = chatbot.resume_execution(user_input, thread_id)
                    
                    # 显示最终状态
                    final_state = chatbot.get_conversation_state(thread_id)
                    if final_state:
                        print("\n📊 验证后的状态信息:")
                        print(f"实体名称: {final_state.get('name', '未设置')}")
                        print(f"生日/发布日期: {final_state.get('birthday', '未设置')}")
                        print(f"验证状态: {final_state.get('verification_status', 'pending')}")
                    break
                else:
                    print("❌ 请输入有效的验证信息")
            except KeyboardInterrupt:
                print("\n❌ 用户取消验证")
                break
            except Exception as ex:
                print(f"❌ 处理用户验证时出错: {ex}")
                break
    
    return chatbot


def demonstrate_state_display():
    """演示状态显示功能"""
    print("\n" + "="*60)
    print("📊 演示3: 状态显示功能")
    print("="*60)
    
    chatbot = get_global_chatbot()
    if chatbot.graph is None:
        print("❌ 聊天机器人初始化失败")
        return
    
    # 检查不同线程的状态
    thread_ids = ["entity_lookup_demo", "human_verification_demo", "default"]
    
    for thread_id in thread_ids:
        print(f"\n🔍 检查线程 '{thread_id}' 的状态:")
        state = chatbot.get_conversation_state(thread_id)
        
        if state:
            print(f"  实体名称: {state.get('name', '未设置')}")
            print(f"  生日/发布日期: {state.get('birthday', '未设置')}")
            print(f"  验证状态: {state.get('verification_status', 'pending')}")
            print(f"  消息数量: {len(state.get('messages', []))}")
            print(f"  搜索结果数量: {len(state.get('search_results', []))}")
        else:
            print("  ❌ 无状态信息")


def interactive_chat():
    """交互式聊天模式"""
    print("\n" + "="*60)
    print("💬 交互式聊天模式")
    print("="*60)
    print("说明: 您可以与自定义状态聊天机器人进行自由对话")
    print("输入 'quit' 或 'exit' 退出聊天")
    print("输入 'state' 查看当前状态")
    print("输入 'clear' 清除对话历史")
    
    chatbot = CustomStateChatbot()
    if chatbot.graph is None:
        print("❌ 聊天机器人初始化失败")
        return
    
    thread_id = "interactive_chat"
    
    while True:
        try:
            user_input = input("\n👤 您: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("👋 再见！")
                break
            elif user_input.lower() in ['state', '状态']:
                state = chatbot.get_conversation_state(thread_id)
                if state:
                    print("\n📊 当前状态:")
                    print(f"  实体名称: {state.get('name', '未设置')}")
                    print(f"  生日/发布日期: {state.get('birthday', '未设置')}")
                    print(f"  验证状态: {state.get('verification_status', 'pending')}")
                    print(f"  消息数量: {len(state.get('messages', []))}")
                else:
                    print("❌ 无状态信息")
                continue
            elif user_input.lower() in ['clear', '清除']:
                thread_id = f"interactive_chat_{datetime.now().strftime('%H%M%S')}"
                print("🗑️ 对话历史已清除")
                continue
            elif not user_input:
                continue
            
            # 进行对话
            chatbot.chat_with_custom_state(user_input, thread_id)
            
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")


def main():
    """主函数"""
    print("\n🚀 第25小节：LangGraph自定义状态演示")
    print("="*60)
    
    # 直接运行实体信息查找演示来测试状态更新
    print("\n🔧 自动运行实体信息查找演示来测试状态更新功能")
    demonstrate_entity_lookup()
    
    # 显示状态
    print("\n🔧 显示状态信息")
    demonstrate_state_display()


if __name__ == "__main__":
    main()