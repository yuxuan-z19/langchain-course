# LangChain 结构化输出教程

## 概述

在许多应用场景中，我们需要AI模型以特定的结构化格式返回结果，而不是自由形式的自然语言文本。例如：
- 将模型输出存储到数据库中
- 确保输出符合特定的数据模式
- 提取结构化信息用于后续处理
- 构建API响应格式

LangChain 提供了强大的结构化输出功能，让我们能够轻松地指导模型按照预定义的结构返回结果。

## 学习目标

通过本教程，你将学会：

1. **Schema 定义**：使用字典和 Pydantic 定义输出结构
2. **with_structured_output() 方法**：LangChain 推荐的结构化输出方式
3. **Tool Calling**：通过工具调用实现结构化输出
4. **JSON Mode**：使用 JSON 模式强制结构化输出
5. **实际应用场景**：数据提取、格式化回复等实用案例

## 核心概念

### 1. Schema 定义

输出结构需要通过 Schema 来表示，主要有两种方式：

#### 字典 (Dict) Schema
```python
# 简单的字典结构
schema = {
    "answer": "用户问题的答案",
    "followup_question": "可以继续询问的问题"
}
```

#### Pydantic Schema
```python
from pydantic import BaseModel, Field

class ResponseFormatter(BaseModel):
    """响应格式化器"""
    answer: str = Field(description="用户问题的答案")
    followup_question: str = Field(description="可以继续询问的问题")
```

### 2. 结构化输出方法

#### with_structured_output() 方法（推荐）
```python
# 绑定 schema 到模型
model_with_structure = model.with_structured_output(schema)
# 调用模型生成结构化输出
structured_output = model_with_structure.invoke(user_input)
```

#### Tool Calling 方式
```python
# 将 schema 作为工具绑定到模型
model_with_tools = model.bind_tools([ResponseFormatter])
# 调用模型
ai_msg = model_with_tools.invoke(user_input)
# 提取工具调用参数
result = ai_msg.tool_calls[0]["args"]
```

#### JSON Mode
```python
# 使用 JSON 模式
model_json = model.with_structured_output(method="json_mode")
result = model_json.invoke("返回包含指定字段的JSON对象")
```

## 应用场景

### 1. 数据提取
从非结构化文本中提取结构化信息，如：
- 从新闻文章中提取关键信息
- 从用户评论中提取情感和关键词
- 从简历中提取个人信息

### 2. 格式化回复
确保AI回复符合特定格式，如：
- API 响应格式
- 聊天机器人回复结构
- 报告生成格式

### 3. 数据验证
利用 Pydantic 的验证功能：
- 类型检查
- 字段验证
- 数据清洗

## 最佳实践

### 1. 工具绑定顺序
当结合结构化输出和其他工具时，正确的顺序是：
```python
# 正确：先绑定工具，再应用结构化输出
model_with_tools = model.bind_tools([tool1, tool2])
structured_model = model_with_tools.with_structured_output(schema)

# 错误：会导致工具解析错误
structured_model = model.with_structured_output(schema)
broken_model = structured_model.bind_tools([tool1, tool2])
```

### 2. Schema 设计原则
- 使用清晰的字段名称
- 提供详细的字段描述
- 合理设置字段类型和约束
- 考虑可选字段和默认值

### 3. 错误处理
- 验证模型输出是否符合预期格式
- 处理解析失败的情况
- 提供回退机制

## 快速开始

1. 确保已安装必要的依赖：
```bash
pip install langchain langchain-openai pydantic
```

2. 配置环境变量：
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，添加你的 DeepSeek API 密钥
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

3. 运行示例代码：
```bash
python tutorials/04_structured_output/structured_output_demo.py
```

## 示例代码说明

`structured_output_demo.py` 文件包含了完整的示例代码，演示了：

1. **基础字典 Schema**：简单的键值对结构
2. **Pydantic Schema**：带类型验证的复杂结构
3. **with_structured_output() 实战**：推荐的使用方式
4. **Tool Calling 演示**：通过工具调用实现结构化输出
5. **JSON Mode 使用**：强制 JSON 格式输出
6. **实际应用案例**：新闻摘要、用户信息提取等

## 注意事项

1. **模型支持**：不是所有模型都支持结构化输出，请确认你使用的模型支持此功能
2. **API 限制**：某些 API 提供商可能对结构化输出有特定限制
3. **性能考虑**：结构化输出可能会增加响应时间，请根据实际需求权衡
4. **成本影响**：某些结构化输出方法可能会增加 token 消耗

## 下一步

完成本教程后，你可以：
- 在自己的项目中应用结构化输出
- 探索更复杂的 Schema 设计
- 结合其他 LangChain 功能使用结构化输出
- 学习下一个教程主题

---

**提示**：如果在运行示例时遇到问题，请检查：
1. API 密钥是否正确配置
2. 网络连接是否正常
3. 依赖包是否正确安装
4. Python 版本是否兼容（推荐 3.8+）