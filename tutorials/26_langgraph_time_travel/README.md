# LangGraph 时间旅行功能演示

本教程演示了 LangGraph 的时间旅行功能，这是一个强大的特性，允许您在图执行过程中的任意时间点进行状态回溯、修改和重新执行。

## 功能概述

时间旅行功能提供以下核心能力：

1. **状态历史管理** - 查看和管理执行历史记录
2. **检查点操作** - 访问和操作特定的执行检查点
3. **状态修改** - 编辑图状态并创建新的执行分支
4. **执行控制** - 从任意检查点恢复和继续执行

## 核心概念

### 检查点 (Checkpoints)
检查点是图执行过程中的状态快照，包含：
- 当前状态值
- 执行配置
- 时间戳信息
- 元数据

### 状态历史 (State History)
状态历史是所有检查点的时间序列，允许您：
- 查看执行轨迹
- 比较不同时间点的状态
- 选择回溯点

### 分支执行 (Branching)
从任意检查点创建新的执行路径，实现：
- 多路径探索
- A/B 测试
- 错误恢复

## 安装和设置

### 依赖要求

```bash
pip install langchain langgraph langchain-openai
```

### 环境配置

确保设置了 OpenAI API 密钥：

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 使用指南

### 基本用法

```python
from time_travel_demo import TimeTravelChatbot

# 创建聊天机器人实例
chatbot = TimeTravelChatbot(use_sqlite=False)  # 使用内存存储

# 配置会话
config = {
    "configurable": {
        "thread_id": "my_session"
    }
}

# 进行对话
result = chatbot.chat_with_time_travel("你好，请告诉我现在的时间", config)
print(result['response'])
```

### 查看状态历史

```python
# 获取状态历史
history = chatbot.get_state_history(config, limit=10)

# 显示历史记录
chatbot.display_state_history(config)
```

### 检查点操作

```python
# 获取特定检查点的状态
checkpoint_state = chatbot.get_checkpoint_state(config, checkpoint_id="some_id")

# 显示检查点详情
chatbot.display_checkpoint_details(config, checkpoint_id="some_id")
```

### 状态修改

```python
# 更新状态
state_update = {
    "current_task": "新任务描述",
    "metadata": {
        "user_preference": "编程学习",
        "skill_level": "初学者"
    }
}

success = chatbot.update_state(config, state_update)
```

### 从检查点恢复执行

```python
# 从特定检查点恢复执行
result = chatbot.resume_from_checkpoint(
    config, 
    checkpoint_id="target_checkpoint",
    new_input={"messages": [HumanMessage(content="新的输入")]}
)
```

## 演示程序

运行完整的演示程序：

```bash
cd tutorials/26_langgraph_time_travel
python time_travel_demo.py
```

### 演示内容

1. **基本时间旅行功能**
   - 多轮对话
   - 状态历史查看
   - 当前状态详情

2. **检查点操作功能**
   - 检查点选择
   - 状态恢复
   - 执行继续

3. **状态修改功能**
   - 状态更新
   - 元数据修改
   - 修改后执行

4. **分支创建功能**
   - 多分支对话
   - 状态比较
   - 路径探索

5. **交互式演示**
   - 实时操作
   - 命令行界面
   - 持久化存储

## 高级特性

### 持久化存储

使用 SQLite 进行持久化存储：

```python
# 使用 SQLite 检查点存储
chatbot = TimeTravelChatbot(use_sqlite=True)
```

这将在当前目录创建 `time_travel_checkpoints.db` 文件，保存所有检查点数据。

### 自定义状态结构

```python
class CustomState(TypedDict):
    messages: List[BaseMessage]
    custom_field: str
    user_data: Dict[str, Any]
```

### 工具集成

演示程序包含三个示例工具：

1. **get_current_time** - 获取当前时间
2. **calculate_math** - 数学计算
3. **search_information** - 信息搜索（模拟）

## 交互式命令

在交互式演示中，您可以使用以下命令：

- `history` - 查看状态历史
- `checkpoint <id>` - 查看特定检查点
- `resume <checkpoint_id>` - 从检查点恢复
- `modify` - 修改当前状态
- `quit` - 退出程序

## 实际应用场景

### 1. 调试和开发
- 回溯到错误发生前的状态
- 测试不同的执行路径
- 分析状态变化过程

### 2. 用户体验优化
- 允许用户撤销操作
- 提供多种选择路径
- 保存会话进度

### 3. A/B 测试
- 从同一起点测试不同策略
- 比较不同路径的结果
- 收集性能数据

### 4. 错误恢复
- 从最后正确状态恢复
- 避免重新开始整个流程
- 提高系统可靠性

## 最佳实践

### 1. 检查点管理
- 定期清理旧检查点
- 为重要检查点添加标记
- 合理设置历史记录限制

### 2. 状态设计
- 保持状态结构简洁
- 避免存储大量数据
- 使用元数据存储辅助信息

### 3. 性能优化
- 选择合适的存储后端
- 控制检查点频率
- 及时清理无用数据

### 4. 错误处理
- 验证检查点有效性
- 处理状态不一致情况
- 提供回退机制

## 故障排除

### 常见问题

1. **检查点不存在**
   ```python
   # 检查检查点是否存在
   state = chatbot.get_checkpoint_state(config, checkpoint_id)
   if state is None:
       print("检查点不存在")
   ```

2. **状态更新失败**
   ```python
   # 验证状态格式
   try:
       success = chatbot.update_state(config, state_update)
   except Exception as e:
       print(f"状态更新失败：{e}")
   ```

3. **内存使用过高**
   - 使用 SQLite 存储替代内存存储
   - 定期清理历史记录
   - 限制状态大小

### 调试技巧

1. **启用详细日志**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **检查状态一致性**
   ```python
   # 比较不同检查点的状态
   state1 = chatbot.get_checkpoint_state(config, "checkpoint1")
   state2 = chatbot.get_checkpoint_state(config, "checkpoint2")
   ```

3. **监控执行路径**
   ```python
   # 查看执行路径
   state = chatbot.get_checkpoint_state(config)
   execution_path = state["values"].get("execution_path", [])
   print(f"执行路径：{execution_path}")
   ```

## 扩展开发

### 自定义检查点策略

```python
class CustomCheckpointer(MemorySaver):
    def put(self, config, checkpoint, metadata):
        # 自定义检查点保存逻辑
        super().put(config, checkpoint, metadata)
        
    def get(self, config):
        # 自定义检查点获取逻辑
        return super().get(config)
```

### 添加新工具

```python
@tool
def custom_tool(input_data: str) -> str:
    """自定义工具"""
    # 工具逻辑
    return "工具结果"

# 添加到工具列表
chatbot.tools.append(custom_tool)
chatbot.tool_node = ToolNode(chatbot.tools)
```

### 状态验证

```python
def validate_state(state: TimeTravelState) -> bool:
    """验证状态有效性"""
    required_fields = ["messages", "step_count", "current_task"]
    return all(field in state for field in required_fields)
```

## 参考资料

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [检查点和持久化](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [时间旅行功能](https://langchain-ai.github.io/langgraph/concepts/time_travel/)
- [状态管理最佳实践](https://langchain-ai.github.io/langgraph/concepts/state/)

## 许可证

本教程遵循 MIT 许可证。

## 贡献

欢迎提交问题和改进建议！

---

**注意**：本演示程序仅用于教学目的，生产环境使用时请根据实际需求进行适当的错误处理和安全检查。