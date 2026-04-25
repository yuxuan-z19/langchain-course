# 第22小节：LangGraph工具集成

## 概述

本教程将展示如何在LangGraph中集成各种工具，包括搜索引擎、计算器等，让聊天机器人能够处理需要外部信息或计算的查询。

## 核心概念

### 1. 工具绑定 (Tool Binding)

工具绑定是将外部工具与LLM连接的过程。通过`bind_tools()`方法，我们可以告诉LLM有哪些工具可用，以及如何调用这些工具。

```python
from langchain_tavily import TavilySearch

# 定义工具
tool = TavilySearch(max_results=2)
tools = [tool]

# 绑定工具到LLM
llm_with_tools = llm.bind_tools(tools)
```

### 2. 工具调用节点 (Tool Node)

工具调用节点负责执行LLM请求的工具调用。当LLM决定需要使用工具时，会生成包含工具调用信息的消息，工具节点会解析这些信息并执行相应的工具。

```python
from langgraph.prebuilt import ToolNode

# 创建工具节点
tool_node = ToolNode(tools)
```

### 3. 条件边 (Conditional Edges)

条件边用于根据当前状态决定下一步的执行路径。在工具集成中，我们需要判断LLM的响应是否包含工具调用：

- 如果包含工具调用 → 转到工具节点
- 如果不包含工具调用 → 结束对话

```python
def should_continue(state: State) -> Literal["tools", "__end__"]:
    messages = state['messages']
    last_message = messages[-1]
    # 如果LLM调用了工具，继续到工具节点
    if last_message.tool_calls:
        return "tools"
    # 否则结束
    return "__end__"
```

### 4. 状态管理

在工具集成的场景中，状态需要包含消息历史，以便跟踪对话流程和工具调用结果。

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

## 支持的工具类型

### 1. 搜索引擎工具
- **Tavily Search**: 网页搜索工具，用于获取实时信息
- 支持设置最大结果数量
- 返回相关网页的摘要信息

### 2. 计算工具
- **Calculator**: 数学计算工具
- 支持基本算术运算
- 支持复杂数学表达式

### 3. 时间工具
- **DateTime**: 时间查询工具
- 获取当前时间
- 时区转换

### 4. 自定义工具
- 可以根据需要创建自定义工具
- 实现特定的业务逻辑

## 工具调用流程

1. **用户输入**: 用户提出需要外部信息的问题
2. **LLM分析**: LLM分析问题并决定是否需要使用工具
3. **工具调用**: 如果需要，LLM生成工具调用请求
4. **工具执行**: 工具节点执行相应的工具并返回结果
5. **结果整合**: LLM将工具结果整合到最终回答中
6. **用户响应**: 向用户提供完整的回答

## 错误处理

- **工具调用失败**: 当工具调用失败时，系统会捕获错误并提供友好的错误信息
- **网络超时**: 对于网络相关的工具，实现超时机制
- **参数验证**: 验证工具调用的参数是否正确
- **降级策略**: 当工具不可用时，提供基于LLM知识的回答

## 可视化功能

本教程包含图结构的可视化功能，帮助理解工具调用的流程：

- 节点表示：聊天机器人节点、工具节点
- 边表示：条件边、普通边
- 状态流转：显示消息在不同节点间的传递

## 环境要求

- Python 3.8+
- LangChain
- LangGraph
- langchain-tavily (用于搜索功能)
- python-dotenv (用于环境变量管理)

## 配置要求

在`.env`文件中配置以下API密钥：

```bash
# Tavily搜索引擎API密钥
TAVILY_API_KEY=your_tavily_api_key_here

# LLM API密钥（DeepSeek或其他）
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

## 使用方法

1. 确保已安装所需依赖
2. 配置环境变量
3. 运行演示脚本：

```bash
python tutorials/22_langgraph_tools/tools_chatbot_demo.py
```

## 示例对话

```
用户: 今天的天气怎么样？
助手: [调用搜索工具获取天气信息]

用户: 计算 123 * 456 的结果
助手: [调用计算器工具] 123 * 456 = 56,088

用户: 现在几点了？
助手: [调用时间工具] 当前时间是...
```

## 扩展功能

- 添加更多工具类型
- 实现工具链调用
- 支持并行工具调用
- 添加工具调用的缓存机制
- 实现工具调用的权限控制

## 注意事项

1. **API配额**: 注意各种API的调用配额限制
2. **响应时间**: 工具调用可能增加响应时间
3. **错误处理**: 确保有完善的错误处理机制
4. **安全性**: 验证工具调用的参数，防止恶意输入
5. **成本控制**: 监控API调用成本，特别是搜索API

## 相关资源

- [LangGraph官方文档](https://langgraph.com.cn/)
- [Tavily搜索API文档](https://tavily.com/)
- [LangChain工具文档](https://python.langchain.com/docs/modules/agents/tools/)