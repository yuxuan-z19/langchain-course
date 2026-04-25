#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph 时间旅行功能演示

本模块演示了 LangGraph 的时间旅行功能，包括：
1. 状态历史管理 - 获取和查看执行历史
2. 检查点操作 - 查看和修改特定检查点
3. 执行控制 - 从检查点恢复执行和创建分支
4. 状态修改 - 编辑状态并重新执行

时间旅行功能允许用户在图执行过程中的任意时间点进行状态回溯、修改和重新执行，
为调试、优化和多路径探索提供了强大的支持。
"""

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, TypedDict

# 添加项目根目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.message import add_messages

# SQLite检查点器在当前版本中可能不可用，使用内存检查点器作为替代
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from utils.llm_factory import create_llm_from_config


class TimeTravelState(TypedDict):
    """时间旅行演示的状态定义"""

    messages: Annotated[List[BaseMessage], add_messages]
    step_count: int
    current_task: str


@dataclass
class CheckpointInfo:
    """检查点信息数据类"""

    checkpoint_id: str
    thread_id: str
    created_at: datetime
    step_count: int
    current_task: str
    message_count: int
    metadata: Dict[str, Any]


# 定义工具函数
@tool
def get_current_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate_math(expression: str) -> str:
    """计算数学表达式

    Args:
        expression: 要计算的数学表达式，如 "2 + 3 * 4"

    Returns:
        计算结果
    """
    try:
        # 安全的数学表达式计算
        allowed_chars = set("0123456789+-*/().")
        if not all(c in allowed_chars or c.isspace() for c in expression):
            return "错误：表达式包含不允许的字符"

        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


@tool
def search_information(query: str) -> str:
    """搜索信息（模拟）

    Args:
        query: 搜索查询

    Returns:
        搜索结果
    """
    # 模拟搜索结果
    search_results = {
        "天气": "今天天气晴朗，温度25°C，适合外出活动。",
        "新闻": "最新科技新闻：AI技术在各个领域都有新的突破。",
        "股票": "今日股市表现平稳，科技股略有上涨。",
        "美食": "推荐今日特色菜：宫保鸡丁、麻婆豆腐、糖醋里脊。",
    }

    for key, value in search_results.items():
        if key in query:
            return f"搜索结果：{value}"

    return f"搜索结果：关于'{query}'的信息，这是一个模拟的搜索结果。"


class TimeTravelChatbot:
    """时间旅行聊天机器人类

    这个类演示了 LangGraph 的时间旅行功能，包括：
    - 状态历史管理
    - 检查点操作
    - 执行控制
    - 状态修改
    """

    def __init__(self):
        """初始化时间旅行聊天机器人

        使用内存检查点器进行状态管理
        """
        self.llm = None
        self.graph = None
        self.tools = [get_current_time, calculate_math, search_information]
        self.tool_node = ToolNode(self.tools)

        # 初始化检查点管理器（使用内存存储）
        self.checkpointer = MemorySaver()
        print("使用内存检查点存储")

        self._initialize_llm()
        self._build_graph()

    def _initialize_llm(self):
        """初始化语言模型"""
        try:
            self.llm = create_llm_from_config("deepseek")
            print("LLM 初始化成功")
        except Exception as e:
            print(f"LLM 初始化失败：{e}")
            raise

    def _build_graph(self):
        """构建 LangGraph 图"""
        # 创建状态图
        graph_builder = StateGraph(TimeTravelState)

        # 添加节点
        graph_builder.add_node("chatbot", self._chatbot_node)
        graph_builder.add_node("tools", self.tool_node)

        # 设置入口点
        graph_builder.set_entry_point("chatbot")

        # 添加条件边
        graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
        )

        # 从工具节点返回到聊天机器人
        graph_builder.add_edge("tools", "chatbot")

        # 编译图并设置检查点管理器
        self.graph = graph_builder.compile(checkpointer=self.checkpointer)
        print("LangGraph 图构建完成")

    def _chatbot_node(self, state: TimeTravelState) -> dict:
        """聊天机器人节点

        处理用户消息并生成响应

        Args:
            state: 当前状态

        Returns:
            包含AI响应的状态更新
        """
        try:
            # 获取消息历史
            messages = state["messages"]

            # 绑定工具并调用LLM
            llm_with_tools = self.llm.bind_tools(self.tools)
            response = llm_with_tools.invoke(messages)

            # 更新状态 - 只返回新响应，不包含历史消息
            return {
                "messages": [response],
                "step_count": state["step_count"] + 1,
                "current_task": "处理用户请求",
            }

        except Exception as e:
            print(f"❌ 聊天机器人节点错误: {e}")
            error_response = AIMessage(content=f"抱歉，处理请求时发生错误：{str(e)}")
            return {
                "messages": [error_response],
                "step_count": state["step_count"] + 1,
                "current_task": "错误处理",
            }

    def chat_with_time_travel(
        self, user_input: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """与时间旅行聊天机器人对话"""
        try:
            # 创建用户消息
            user_message = HumanMessage(content=user_input)

            # 创建输入状态
            input_state = {
                "messages": [user_message],
                "step_count": 0,
                "current_task": "用户输入",
            }

            # 流式执行图
            final_state = None
            for chunk in self.graph.stream(input_state, config, stream_mode="values"):
                final_state = chunk
                print(
                    f"步骤 {chunk.get('step_count', 0)}: {chunk.get('current_task', '未知任务')}"
                )

            # 提取最终响应
            if final_state and "messages" in final_state:
                messages = final_state["messages"]
                if messages:
                    last_message = messages[-1]
                    if hasattr(last_message, "content"):
                        return {
                            "response": last_message.content,
                            "state": final_state,
                            "config": config,
                        }

            return {
                "response": "抱歉，无法生成响应。",
                "state": final_state,
                "config": config,
            }

        except Exception as e:
            print(f"对话过程中发生错误: {e}")
            return {"response": f"发生错误: {str(e)}", "state": None, "config": config}

    def get_state_history(
        self, config: Dict[str, Any], limit: int = 10
    ) -> List[CheckpointInfo]:
        """获取状态历史

        Args:
            config: 配置信息
            limit: 返回的历史记录数量限制

        Returns:
            检查点信息列表
        """
        try:
            history = []
            state_history = self.graph.get_state_history(config, limit=limit)

            for i, state in enumerate(state_history):
                checkpoint_id = state.config.get("configurable", {}).get(
                    "checkpoint_id", f"checkpoint_{i}"
                )
                thread_id = state.config.get("configurable", {}).get(
                    "thread_id", "unknown"
                )

                # 提取状态信息
                values = state.values or {}
                step_count = values.get("step_count", 0)
                current_task = values.get("current_task", "未知任务")
                messages = values.get("messages", [])
                metadata = values.get("metadata", {})

                # 创建检查点信息
                checkpoint_info = CheckpointInfo(
                    checkpoint_id=checkpoint_id,
                    thread_id=thread_id,
                    created_at=datetime.now(),  # 实际应该从检查点获取
                    step_count=step_count,
                    current_task=current_task,
                    message_count=len(messages),
                    metadata=metadata,
                )

                history.append(checkpoint_info)

            return history

        except Exception as e:
            print(f"获取状态历史失败：{e}")
            return []

    def get_checkpoint_state(
        self, config: Dict[str, Any], checkpoint_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """获取特定检查点的状态

        Args:
            config: 配置信息
            checkpoint_id: 检查点ID（可选，默认获取当前状态）

        Returns:
            状态信息字典
        """
        try:
            if checkpoint_id:
                # 获取特定检查点的状态
                config_with_checkpoint = {
                    **config,
                    "configurable": {
                        **config.get("configurable", {}),
                        "checkpoint_id": checkpoint_id,
                    },
                }
                state = self.graph.get_state(config_with_checkpoint)
            else:
                # 获取当前状态
                state = self.graph.get_state(config)

            # 安全处理时间戳
            created_at_str = None
            if hasattr(state, "created_at") and state.created_at:
                if hasattr(state.created_at, "isoformat"):
                    created_at_str = state.created_at.isoformat()
                else:
                    created_at_str = str(state.created_at)

            return {
                "values": state.values,
                "config": state.config,
                "metadata": state.metadata,
                "created_at": created_at_str,
                "parent_config": state.parent_config,
            }

        except Exception as e:
            print(f"获取检查点状态失败：{e}")
            return None

    def update_state(
        self,
        config: Dict[str, Any],
        state_update: Dict[str, Any],
        as_node: Optional[str] = None,
    ) -> bool:
        """更新图状态

        Args:
            config: 配置信息
            state_update: 要更新的状态
            as_node: 指定作为哪个节点的输出（可选）

        Returns:
            更新是否成功
        """
        try:
            self.graph.update_state(config, state_update, as_node=as_node)
            print(f"状态更新成功：{state_update}")
            return True

        except Exception as e:
            print(f"状态更新失败：{e}")
            return False

    def resume_from_checkpoint(
        self,
        config: Dict[str, Any],
        checkpoint_id: str,
        new_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """从特定检查点恢复执行

        Args:
            config: 配置信息
            checkpoint_id: 要恢复的检查点ID
            new_input: 新的输入数据（可选）

        Returns:
            执行结果
        """
        try:
            # 创建带检查点的配置
            config_with_checkpoint = {
                **config,
                "configurable": {
                    **config.get("configurable", {}),
                    "checkpoint_id": checkpoint_id,
                },
            }

            print(f"从检查点 {checkpoint_id} 恢复执行...")

            # 从检查点继续执行
            final_state = None
            for event in self.graph.stream(new_input, config_with_checkpoint):
                for node_name, node_output in event.items():
                    print(f"节点 '{node_name}' 输出：")
                    if "messages" in node_output:
                        latest_message = node_output["messages"][-1]
                        if hasattr(latest_message, "content"):
                            print(f"  消息：{latest_message.content}")
                    print(f"  步骤计数：{node_output.get('step_count', 'N/A')}")
                    print("---")
                    final_state = node_output

            return {
                "success": True,
                "final_state": final_state,
                "checkpoint_id": checkpoint_id,
            }

        except Exception as e:
            print(f"从检查点恢复执行失败：{e}")
            return {"success": False, "error": str(e), "checkpoint_id": checkpoint_id}

    def display_state_history(self, config: Dict[str, Any], limit: int = 5):
        """显示状态历史

        Args:
            config: 配置信息
            limit: 显示的历史记录数量
        """
        print("\n=== 状态历史 ===")
        history = self.get_state_history(config, limit)

        if not history:
            print("暂无历史记录")
            return

        for i, checkpoint in enumerate(history):
            print(f"\n[{i+1}] 检查点 ID: {checkpoint.checkpoint_id}")
            print(f"    线程 ID: {checkpoint.thread_id}")
            print(f"    步骤计数: {checkpoint.step_count}")
            print(f"    当前任务: {checkpoint.current_task}")
            print(f"    消息数量: {checkpoint.message_count}")
            if checkpoint.metadata:
                print(
                    f"    元数据: {json.dumps(checkpoint.metadata, ensure_ascii=False, indent=6)}"
                )

    def display_checkpoint_details(
        self, config: Dict[str, Any], checkpoint_id: Optional[str] = None
    ):
        """显示检查点详细信息

        Args:
            config: 配置信息
            checkpoint_id: 检查点ID（可选）
        """
        print(
            f"\n=== 检查点详情 {'(当前状态)' if not checkpoint_id else f'(ID: {checkpoint_id})'} ==="
        )

        state_info = self.get_checkpoint_state(config, checkpoint_id)
        if not state_info:
            print("无法获取检查点信息")
            return

        values = state_info.get("values", {})

        print(f"步骤计数: {values.get('step_count', 'N/A')}")
        print(f"当前任务: {values.get('current_task', 'N/A')}")
        print(f"执行路径: {values.get('execution_path', [])}")

        messages = values.get("messages", [])
        print(f"\n消息历史 ({len(messages)} 条):")
        for i, msg in enumerate(messages[-5:]):  # 只显示最近5条消息
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            content = (
                msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            )
            print(f"  [{i+1}] {role}: {content}")

        metadata = values.get("metadata", {})
        if metadata:
            print(f"\n元数据: {json.dumps(metadata, ensure_ascii=False, indent=2)}")


def demonstrate_basic_time_travel():
    """演示基本的时间旅行功能"""
    print("\n" + "=" * 50)
    print("演示1：基本时间旅行功能")
    print("=" * 50)

    # 创建聊天机器人实例
    chatbot = TimeTravelChatbot()

    # 配置
    config = {"configurable": {"thread_id": "demo_basic_time_travel"}}

    print("\n1. 开始对话...")

    # 第一轮对话
    result1 = chatbot.chat_with_time_travel("你好，请告诉我现在的时间", config)
    print(f"助手回复: {result1['response']}")

    # 第二轮对话
    result2 = chatbot.chat_with_time_travel("请帮我计算 15 * 8 + 7", config)
    print(f"助手回复: {result2['response']}")

    # 第三轮对话
    result3 = chatbot.chat_with_time_travel("搜索一下今天的天气信息", config)
    print(f"助手回复: {result3['response']}")

    print("\n2. 查看状态历史...")
    chatbot.display_state_history(config)

    print("\n3. 查看当前状态详情...")
    chatbot.display_checkpoint_details(config)


def demonstrate_checkpoint_operations():
    """演示检查点操作功能"""
    print("\n" + "=" * 50)
    print("演示2：检查点操作功能")
    print("=" * 50)

    # 创建聊天机器人实例
    chatbot = TimeTravelChatbot()

    # 配置
    config = {"configurable": {"thread_id": "demo_checkpoint_ops"}}

    print("\n1. 进行多轮对话以创建检查点...")

    # 多轮对话
    conversations = [
        "你好，我想了解一些数学知识",
        "请计算 25 * 4",
        "现在几点了？",
        "搜索一下最新的科技新闻",
    ]

    for i, user_input in enumerate(conversations):
        print(f"\n第{i+1}轮对话：{user_input}")
        result = chatbot.chat_with_time_travel(user_input, config)
        print(f"助手回复: {result['response']}")

    print("\n2. 获取状态历史...")
    history = chatbot.get_state_history(config)

    if len(history) >= 2:
        # 选择一个早期的检查点
        target_checkpoint = history[-2]  # 倒数第二个检查点
        print(f"\n3. 选择检查点进行详细查看: {target_checkpoint.checkpoint_id}")
        chatbot.display_checkpoint_details(config, target_checkpoint.checkpoint_id)

        print(f"\n4. 从检查点 {target_checkpoint.checkpoint_id} 恢复执行...")
        resume_result = chatbot.resume_from_checkpoint(
            config,
            target_checkpoint.checkpoint_id,
            {
                "messages": [
                    HumanMessage(content="从这个时间点开始，请告诉我一个有趣的事实")
                ]
            },
        )

        if resume_result["success"]:
            print("从检查点恢复执行成功！")
        else:
            print(f"从检查点恢复执行失败：{resume_result['error']}")
    else:
        print("历史记录不足，无法演示检查点操作")


def demonstrate_state_modification():
    """演示状态修改功能"""
    print("\n" + "=" * 50)
    print("演示3：状态修改功能")
    print("=" * 50)

    # 创建聊天机器人实例
    chatbot = TimeTravelChatbot()

    # 配置
    config = {"configurable": {"thread_id": "demo_state_modification"}}

    print("\n1. 初始对话...")
    result1 = chatbot.chat_with_time_travel("你好，我想学习编程", config)
    print(f"助手回复: {result1['response']}")

    print("\n2. 查看当前状态...")
    chatbot.display_checkpoint_details(config)

    print("\n3. 修改状态 - 添加自定义元数据...")
    state_update = {
        "metadata": {
            "user_preference": "编程学习",
            "skill_level": "初学者",
            "modified_at": datetime.now().isoformat(),
            "modification_reason": "添加用户偏好信息",
        },
        "current_task": "个性化编程学习辅导",
    }

    success = chatbot.update_state(config, state_update)
    if success:
        print("状态修改成功！")

        print("\n4. 查看修改后的状态...")
        chatbot.display_checkpoint_details(config)

        print("\n5. 基于修改后的状态继续对话...")
        result2 = chatbot.chat_with_time_travel(
            "我应该从哪种编程语言开始学习？", config
        )
        print(f"助手回复: {result2['response']}")
    else:
        print("状态修改失败！")


def demonstrate_branch_creation():
    """演示分支创建功能"""
    print("\n" + "=" * 50)
    print("演示4：分支创建功能")
    print("=" * 50)

    # 创建聊天机器人实例
    chatbot = TimeTravelChatbot()

    # 主分支配置
    main_config = {"configurable": {"thread_id": "demo_main_branch"}}

    print("\n1. 在主分支进行对话...")

    # 主分支对话
    main_conversations = ["你好，我想规划一次旅行", "我想去日本旅游，请给我一些建议"]

    for user_input in main_conversations:
        result = chatbot.chat_with_time_travel(user_input, main_config)
        print(f"用户: {user_input}")
        print(f"助手: {result['response']}")
        print()

    print("\n2. 获取主分支的历史...")
    main_history = chatbot.get_state_history(main_config)

    if len(main_history) >= 1:
        # 选择一个检查点作为分支起点
        branch_point = main_history[-1]  # 最后一个检查点

        print(f"\n3. 从检查点 {branch_point.checkpoint_id} 创建新分支...")

        # 创建分支配置（使用不同的 thread_id）
        branch_config = {"configurable": {"thread_id": "demo_branch_alternative"}}

        # 首先复制状态到新分支
        main_state = chatbot.get_checkpoint_state(main_config)
        if main_state and main_state["values"]:
            # 在新分支中设置相同的初始状态
            chatbot.update_state(branch_config, main_state["values"])

            print("\n4. 在新分支中探索不同的对话路径...")

            # 分支对话（不同的路径）
            branch_result = chatbot.chat_with_time_travel(
                "实际上，我改变主意了，我想去欧洲旅游，特别是法国", branch_config
            )
            print(f"分支助手回复: {branch_result['response']}")

            print("\n5. 比较主分支和新分支的状态...")
            print("\n主分支当前状态:")
            chatbot.display_checkpoint_details(main_config)

            print("\n新分支当前状态:")
            chatbot.display_checkpoint_details(branch_config)
        else:
            print("无法获取主分支状态，分支创建失败")
    else:
        print("主分支历史记录不足，无法演示分支创建")


def interactive_time_travel_demo():
    """交互式时间旅行演示"""
    print("\n" + "=" * 50)
    print("交互式时间旅行演示")
    print("=" * 50)
    print("输入 'quit' 退出，'history' 查看历史，'checkpoint <id>' 查看特定检查点")
    print("输入 'resume <checkpoint_id>' 从检查点恢复，'modify' 修改状态")

    # 创建聊天机器人实例
    chatbot = TimeTravelChatbot()

    # 配置
    config = {"configurable": {"thread_id": "interactive_demo"}}

    while True:
        try:
            user_input = input("\n用户: ").strip()

            if user_input.lower() == "quit":
                print("再见！")
                break
            elif user_input.lower() == "history":
                chatbot.display_state_history(config, limit=10)
            elif user_input.lower().startswith("checkpoint "):
                checkpoint_id = user_input[11:].strip()
                chatbot.display_checkpoint_details(config, checkpoint_id)
            elif user_input.lower().startswith("resume "):
                checkpoint_id = user_input[7:].strip()
                new_input_text = input("请输入新的消息（可选，直接回车跳过）: ").strip()
                new_input = None
                if new_input_text:
                    new_input = {"messages": [HumanMessage(content=new_input_text)]}

                result = chatbot.resume_from_checkpoint(
                    config, checkpoint_id, new_input
                )
                if result["success"]:
                    print("从检查点恢复执行成功！")
                else:
                    print(f"恢复失败：{result['error']}")
            elif user_input.lower() == "modify":
                print("当前支持的修改选项：")
                print("1. 修改当前任务描述")
                print("2. 添加元数据")
                choice = input("请选择 (1/2): ").strip()

                if choice == "1":
                    new_task = input("请输入新的任务描述: ").strip()
                    if new_task:
                        chatbot.update_state(config, {"current_task": new_task})
                elif choice == "2":
                    key = input("请输入元数据键: ").strip()
                    value = input("请输入元数据值: ").strip()
                    if key and value:
                        chatbot.update_state(
                            config,
                            {
                                "metadata": {
                                    key: value,
                                    "modified_at": datetime.now().isoformat(),
                                }
                            },
                        )
            else:
                # 正常对话
                result = chatbot.chat_with_time_travel(user_input, config)
                print(f"助手: {result['response']}")

        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"发生错误：{e}")


def main():
    """主函数"""
    print("LangGraph 时间旅行功能演示")
    print("=" * 50)

    try:
        # 运行各种演示
        demonstrate_basic_time_travel()
        demonstrate_checkpoint_operations()
        demonstrate_state_modification()
        demonstrate_branch_creation()

        # 询问是否运行交互式演示
        print("\n" + "=" * 50)
        choice = input("是否运行交互式演示？(y/n): ").strip().lower()
        if choice in ["y", "yes", "是"]:
            interactive_time_travel_demo()

    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"程序执行过程中发生错误：{e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
