# Function Calling 教程

## 概述

Function Calling（函数调用）是大语言模型的一项重要能力，允许模型在对话过程中调用外部工具和函数来完成特定任务。这使得AI助手能够执行计算、查询数据、操作文件等实际操作，而不仅仅是生成文本。

## 核心概念

### 什么是 Function Calling？

Function Calling 是指大语言模型能够：
1. **识别需求**：理解用户请求中需要使用工具的部分
2. **选择工具**：从可用工具中选择合适的函数
3. **构造参数**：根据上下文生成正确的函数参数
4. **执行调用**：调用函数并获取结果
5. **整合回答**：将函数结果整合到最终回复中

### 主要优势

- **扩展能力**：突破纯文本生成的限制
- **实时数据**：获取最新信息而非训练时的静态知识
- **精确计算**：执行准确的数学运算和逻辑操作
- **系统集成**：与外部API和服务无缝对接

## DeepSeek API 支持

DeepSeek API 完全兼容 OpenAI 的 Function Calling 格式，支持：

- **工具定义**：使用 JSON Schema 定义函数参数
- **自动调用**：模型自动识别并调用合适的工具
- **结果处理**：将工具执行结果整合到对话中
- **多轮对话**：支持复杂的多步骤工具调用流程

### 支持的调用方式

1. **LangChain 集成**：通过 LangChain 框架使用 Function Calling
2. **原生 API**：直接使用 DeepSeek API 进行函数调用
3. **混合模式**：结合多种工具实现复杂任务

## 文件说明

### function_calling_demo.py

这是本教程的核心演示文件，包含：

1. **基础工具定义**
   - 计算器工具：执行数学运算
   - 时间查询工具：获取当前时间和日期
   - 文件操作工具：读写文件内容

2. **LangChain 方式演示**
   - 使用 LangChain 的 Agent 框架
   - 展示工具的自动选择和调用
   - 演示多工具协作场景

3. **DeepSeek 原生 API 演示**
   - 直接使用 DeepSeek API
   - 展示原生函数调用流程
   - 对比不同实现方式的特点

4. **实际应用场景**
   - 数据分析任务
   - 文件处理流程
   - 信息查询和整合

## 快速开始

1. **环境准备**
   ```bash
   # 安装依赖
   pip install langchain openai pytz
   
   # 配置环境变量
   export DEEPSEEK_API_KEY="your_api_key_here"
   ```

2. **运行演示**
   ```bash
   cd tutorials/12_function_calling
   python function_calling_demo.py
   ```

3. **交互体验**
   - 尝试数学计算："计算 15 + 27 * 3"
   - 查询时间："现在几点了？"
   - 文件操作："帮我创建一个文件"

## 最佳实践

1. **工具设计**
   - 保持函数功能单一且明确
   - 提供详细的参数描述
   - 实现适当的错误处理

2. **安全考虑**
   - 验证输入参数的合法性
   - 限制文件操作的范围
   - 避免执行危险的系统命令

3. **性能优化**
   - 缓存频繁查询的结果
   - 合理设置超时时间
   - 优化工具的响应速度

## 不同模型的 Function Calling 差异

### OpenAI vs Qwen&DeepSeek 系列对比

虽然各大模型厂商都支持 Function Calling，但在具体实现上存在一些重要差异。了解这些差异有助于选择合适的模型和正确配置工具。

#### 1. JSON Schema 格式差异

**OpenAI 格式**：
```json
{
  "type": "function",
  "function": {
    "name": "calculator",
    "description": "执行数学计算",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {
          "type": "string",
          "description": "要计算的数学表达式"
        }
      },
      "required": ["expression"],
      "additionalProperties": false
    }
  }
}
```

**Qwen 格式**：
```json
{
  "name": "calculator",
  "description": "执行数学计算",
  "parameters": {
    "type": "object",
    "properties": {
      "expression": {
        "type": "string",
        "description": "要计算的数学表达式"
      }
    },
    "required": ["expression"]
  }
}
```

**主要区别**：
- OpenAI 需要外层的 `type: "function"` 和 `function` 包装
- Qwen 使用更简洁的扁平结构
- DeepSeek 通常兼容 OpenAI 格式

#### 2. 工具参数定义的不同

**OpenAI/DeepSeek**：
- 支持 `additionalProperties: false` 严格模式
- 对 JSON Schema 验证更严格
- 支持更复杂的嵌套对象结构

**Qwen**：
- 参数定义相对宽松
- 不支持 `additionalProperties` 字段
- 更注重参数的实际可用性

#### 3. 响应格式的差异

**OpenAI 响应**：
```json
{
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "calculator",
        "arguments": "{\"expression\": \"15 + 27 * 3\"}"
      }
    }
  ]
}
```

**Qwen 响应**：
```json
{
  "function_call": {
    "name": "calculator",
    "arguments": "{\"expression\": \"15 + 27 * 3\"}"
  }
}
```

#### 4. 使用建议和注意事项

**选择 OpenAI/DeepSeek 的场景**：
- 需要严格的参数验证
- 使用 LangChain 等成熟框架
- 要求高度的 JSON Schema 兼容性
- 复杂的多工具协作场景

**选择 Qwen 的场景**：
- 追求简洁的工具定义
- 中文场景下的更好理解能力
- 对参数格式要求相对宽松
- 快速原型开发

**通用最佳实践**：
1. **参数验证**：无论使用哪种模型，都要在工具函数内部进行参数验证
2. **错误处理**：实现健壮的错误处理机制，应对不同模型的响应差异
3. **测试覆盖**：针对不同模型分别进行测试，确保兼容性
4. **文档维护**：清楚记录使用的模型类型和对应的工具定义格式

#### 5. 代码示例对比

本教程提供了两个演示文件：
- `function_calling_demo.py`：使用 OpenAI/DeepSeek 格式
- `qwen_function_calling_demo.py`：专门适配 Qwen 格式

通过对比这两个文件，你可以更好地理解不同模型间的差异，并根据实际需求选择合适的实现方式。

## 扩展学习

- 探索更多 LangChain 工具类型
- 学习自定义工具的开发
- 了解 Function Calling 的高级用法
- 研究多模态工具调用场景
- 比较不同模型厂商的 Function Calling 实现

通过本教程，你将掌握 Function Calling 的核心概念和实际应用，了解不同模型间的差异，为构建更强大的AI应用打下坚实基础。