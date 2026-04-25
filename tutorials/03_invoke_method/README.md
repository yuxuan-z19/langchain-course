# LangChain 第三部分：invoke 接口（现代最佳实践）

## 📚 学习目标

本节将学习 LangChain 的现代化调用接口，掌握：

1. **invoke 方法的使用**：替代已弃用的 predict 方法
2. **统一接口设计**：字符串和消息列表的统一处理
3. **现代最佳实践**：类型安全和错误处理
4. **参数配置**：模型初始化时的参数设置
5. **实际应用**：DeepSeek API 的集成使用

## 🔄 LangChain 方法演进

### ❌ 已弃用的方法（不推荐使用）

```python
# 这些方法在 langchain-core 0.1.7 中被弃用
result = model.predict("你好")
result = model.predict_messages([HumanMessage(content="你好")])
```

**弃用原因：**
- 接口不统一（predict vs predict_messages）
- 类型安全性不足
- 错误处理机制较弱
- 不支持现代功能（流式输出、异步调用等）

### ✅ 现代推荐方法

```python
# 统一的 invoke 接口
result = model.invoke("你好")  # 字符串输入
result = model.invoke([HumanMessage(content="你好")])  # 消息列表输入
```

**优势：**
- 🎯 **统一接口**：字符串和消息都用 invoke
- 🛡️ **类型安全**：返回明确的 AIMessage 对象
- 🔧 **更好的错误处理**
- 🌊 **支持流式输出**：`model.stream(messages)`
- ⚡ **支持异步调用**：`await model.ainvoke(messages)`

## 🏗️ 核心概念

### ChatModel vs LLM

在现代 LangChain 中，推荐使用 **ChatModel**：

```python
from langchain_openai import ChatOpenAI

# ✅ 推荐：使用 ChatModel
chat_model = ChatOpenAI(
    api_key="your-api-key",
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.7
)

# ❌ 不推荐：传统的 LLM 类
# from langchain.llms import OpenAI  # 较老的方式
```

### invoke 方法的两种输入方式

#### 1. 字符串输入（简单场景）

```python
result = chat_model.invoke("制造多彩袜子的公司的好名字是什么？")
print(result.content)  # 输出：AI 的回复内容
print(type(result))    # 输出：<class 'langchain.schema.AIMessage'>
```

#### 2. 消息列表输入（复杂对话）

```python
from langchain.schema import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="你是一个创意营销专家"),
    HumanMessage(content="为袜子公司起名字")
]

result = chat_model.invoke(messages)
print(result.content)  # AI 的回复
```

## 🛠️ 环境要求

### 依赖安装

```bash
# 安装所有必要依赖
pip install -r requirements.txt
```

**关键依赖说明：**

- `langchain>=0.1.0`：核心框架
- `langchain-openai>=0.1.0`：OpenAI 兼容的模型接口
- `langchain-community>=0.1.0`：社区扩展包
- `python-dotenv>=1.0.0`：环境变量管理

### 为什么需要 langchain-community？

`langchain-community` 包含了许多第三方集成和扩展功能：

1. **第三方 API 集成**：各种 LLM 提供商的适配器
2. **工具和实用程序**：文档加载器、向量存储等
3. **社区贡献**：由社区维护的组件
4. **向后兼容**：一些旧版本的导入路径

即使你只使用 OpenAI 兼容的 API，某些内部依赖可能仍需要 community 包。

### API 密钥配置

在项目根目录创建 `.env` 文件：

```env
# DeepSeek API 配置
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

## 🚀 快速开始

### 1. 运行演示脚本

```bash
# 在项目根目录运行
python tutorials/03_predict_methods/predict_demo.py
```

### 2. 预期输出

脚本将演示以下内容：

1. **模型初始化**：ChatOpenAI 的正确配置
2. **字符串输入**：直接传递文本给 invoke
3. **消息列表输入**：使用结构化消息
4. **复杂对话**：多轮对话和角色设定
5. **参数配置**：temperature 等参数的影响
6. **方法对比**：新旧方法的差异说明

### 3. 成功运行的标志

- ✅ 没有弃用警告（DeprecationWarning）
- ✅ 正确的 API 调用和响应
- ✅ 清晰的输出格式和类型信息

## 📖 代码示例

### 基础使用

```python
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# 初始化模型
model = ChatOpenAI(
    api_key="your-api-key",
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.7
)

# 简单调用
response = model.invoke("你好，世界！")
print(response.content)

# 结构化对话
messages = [
    SystemMessage(content="你是一个有用的助手"),
    HumanMessage(content="解释什么是人工智能")
]
response = model.invoke(messages)
print(response.content)
```

### 多轮对话

```python
# 维护对话历史
conversation = [
    SystemMessage(content="你是一个编程助手")
]

# 第一轮
conversation.append(HumanMessage(content="什么是Python？"))
response = model.invoke(conversation)
conversation.append(response)

# 第二轮
conversation.append(HumanMessage(content="给我一个简单的例子"))
response = model.invoke(conversation)
print(response.content)
```

## 🔧 故障排除

### 常见问题

#### 1. DeprecationWarning 警告

**问题**：看到 `predict` 或 `predict_messages` 的弃用警告

**解决**：
```python
# ❌ 旧方法
result = model.predict("hello")

# ✅ 新方法
result = model.invoke("hello")
```

#### 2. 导入错误

**问题**：`ImportError: cannot import name 'ChatOpenAI'`

**解决**：
```python
# ❌ 旧导入
from langchain.chat_models import ChatOpenAI

# ✅ 新导入
from langchain_openai import ChatOpenAI
```

#### 3. API 调用失败

**问题**：`422 Unprocessable Entity` 错误

**解决**：
- 检查 API 密钥是否正确
- 确认 base_url 设置正确
- 验证模型名称（如 "deepseek-chat"）
- 检查网络连接

#### 4. 缺少依赖

**问题**：`ModuleNotFoundError: No module named 'langchain_community'`

**解决**：
```bash
pip install langchain-community
# 或者
pip install -r requirements.txt
```

## 🎯 最佳实践

### 1. 模型初始化

```python
# ✅ 推荐：在初始化时设置所有参数
model = ChatOpenAI(
    api_key=api_key,
    base_url=base_url,
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=1000,
    timeout=30
)
```

### 2. 错误处理

```python
try:
    response = model.invoke(messages)
    print(response.content)
except Exception as e:
    print(f"API 调用失败: {e}")
    # 处理错误逻辑
```

### 3. 类型检查

```python
from langchain.schema import AIMessage

response = model.invoke("hello")
assert isinstance(response, AIMessage)
print(f"回复内容: {response.content}")
print(f"消息类型: {type(response).__name__}")
```

### 4. 参数优化

```python
# 不同场景使用不同参数
creative_model = ChatOpenAI(temperature=0.9)  # 创意任务
factual_model = ChatOpenAI(temperature=0.1)   # 事实性任务
balanced_model = ChatOpenAI(temperature=0.7)  # 平衡模式
```

## 📚 学习建议

1. **实践为主**：多运行示例代码，观察输出差异
2. **参数实验**：尝试不同的 temperature 和 max_tokens 值
3. **消息结构**：练习构建复杂的对话结构
4. **错误处理**：学会处理 API 调用中的各种异常
5. **类型理解**：熟悉 AIMessage、HumanMessage 等类型

## 🔗 相关资源

- [LangChain 官方文档](https://python.langchain.com/)
- [ChatOpenAI API 参考](https://python.langchain.com/docs/integrations/chat/openai)
- [DeepSeek API 文档](https://platform.deepseek.com/api-docs/)
- [消息类型详解](https://python.langchain.com/docs/modules/model_io/chat/message_types)

## 🎉 下一步

完成本节学习后，你将掌握：

- ✅ 现代 LangChain 的调用方式
- ✅ invoke 方法的灵活使用
- ✅ 正确的错误处理和类型安全
- ✅ 参数配置和优化技巧

接下来可以学习：
- **第四部分**：LangChain 的链式调用（Chains）
- **第五部分**：提示词模板（Prompt Templates）
- **第六部分**：输出解析器（Output Parsers）