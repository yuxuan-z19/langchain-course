# 第21小节：LangGraph基础聊天机器人

## 概述

本教程将带您学习LangGraph的基础知识，通过构建一个简单的聊天机器人来了解LangGraph的核心概念：StateGraph、节点(Node)、边(Edge)、状态管理和图的可视化。

## LangGraph核心概念

### 1. StateGraph（状态图）

StateGraph是LangGraph的核心组件，它将应用程序定义为一个"状态机"。状态图包含：
- **状态(State)**：图的数据模式和处理状态更新的reducer函数
- **节点(Node)**：表示工作单元的函数
- **边(Edge)**：指定节点之间的转换路径

### 2. 状态(State)

状态定义了图的数据结构和更新规则：
```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    # messages字段使用add_messages函数来处理状态更新
    # 这意味着新消息会追加到列表中，而不是覆盖现有消息
    messages: Annotated[list, add_messages]
```

### 3. 节点(Node)

节点是执行具体工作的函数，每个节点：
- 接收当前状态作为输入
- 返回状态的更新
- 可以是任何Python函数

### 4. 边(Edge)

边定义了节点之间的连接关系：
- **入口点**：图的起始位置
- **条件边**：根据状态决定下一个节点
- **普通边**：直接连接两个节点

### 5. 图的编译和运行

图需要编译后才能运行：
```python
graph = graph_builder.compile()
result = graph.invoke({"messages": ["Hello!"]})
```

## 文件说明

- `basic_chatbot_demo.py` - 基础聊天机器人演示代码
- `README.md` - 本教程文档

## 功能特性

1. **基础聊天功能**：实现简单的问答对话
2. **状态管理**：使用StateGraph管理对话历史
3. **图可视化**：生成图结构的可视化表示
4. **交互式界面**：提供命令行交互式聊天体验
5. **LLM集成**：复用项目中的LLM工厂模块

## 运行方式

```bash
# 运行基础聊天机器人演示
python tutorials/21_langgraph_basic_chatbot/basic_chatbot_demo.py
```

## 学习目标

通过本教程，您将学会：

1. 理解LangGraph的基本概念和架构
2. 创建和配置StateGraph
3. 定义状态类型和reducer函数
4. 添加节点和边
5. 编译和运行图
6. 可视化图结构
7. 构建交互式聊天应用

## 依赖要求

- langgraph
- langchain
- 项目中的utils.llm_factory模块
- 环境变量配置(.env文件)

## 注意事项

- 确保已正确配置.env文件中的API密钥
- 本教程复用了utils/llm_factory.py中的LLM初始化逻辑
- 图的可视化需要安装相应的绘图依赖

## 下一步

完成本教程后，您可以继续学习：
- 工具调用和函数集成
- 条件边和复杂路由
- 持久化和检查点
- 高级状态管理