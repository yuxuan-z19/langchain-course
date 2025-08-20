# 第06小节：LangChain 提示词模板

## 概述

提示词模板是 LangChain 中的核心组件之一，它们提供了一种结构化的方式来创建、管理和复用提示词。通过使用模板，我们可以将静态文本与动态变量结合，创建灵活且可重用的提示词。

## 学习目标

通过本小节的学习，你将掌握：

1. **基础模板概念**：理解 LangChain 中提示词模板的核心概念
2. **多种模板类型**：学习 PromptTemplate、ChatPromptTemplate、FewShotPromptTemplate 等
3. **Jinja2 集成**：使用 Jinja2 模板引擎创建复杂的提示词模板
4. **文件管理**：从外部文件加载和管理长提示词模板
5. **最佳实践**：模板设计、变量管理和错误处理的最佳实践

## 提示词模板的类型

### 1. PromptTemplate

最基础的提示词模板，支持简单的变量替换：

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["product", "language"],
    template="请为 {product} 写一个 {language} 的产品描述。"
)
```

### 2. ChatPromptTemplate

专为聊天模型设计的模板，支持系统消息、人类消息等：

```python
from langchain.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的 {role}。"),
    ("human", "请帮我 {task}。")
])
```

### 3. FewShotPromptTemplate

支持少样本学习的模板，可以包含示例：

```python
from langchain.prompts import FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"}
]

few_shot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=PromptTemplate(
        input_variables=["input", "output"],
        template="输入: {input}\n输出: {output}"
    ),
    prefix="给出下列词语的反义词：",
    suffix="输入: {adjective}\n输出:",
    input_variables=["adjective"]
)
```

### 4. PipelinePromptTemplate

组合多个模板的复合模板：

```python
from langchain.prompts import PipelinePromptTemplate

# 可以将多个子模板组合成一个完整的提示词
```

## Jinja2 模板集成

LangChain 支持 Jinja2 模板引擎，这为创建复杂的提示词模板提供了强大的功能：

### 优势

1. **条件渲染**：根据变量值显示不同内容
2. **循环结构**：处理列表和重复内容
3. **过滤器**：对变量进行格式化和转换
4. **模板继承**：创建可重用的模板基础
5. **宏定义**：定义可重用的模板片段

### 基本语法

```jinja2
{# 这是注释 #}

{# 变量输出 #}
{{ variable_name }}

{# 条件语句 #}
{% if condition %}
    内容
{% endif %}

{# 循环语句 #}
{% for item in items %}
    {{ item }}
{% endfor %}

{# 过滤器 #}
{{ name|upper }}
{{ price|round(2) }}
```

## 从文件加载模板

对于长的提示词模板，建议将它们保存在单独的文件中：

### 文件组织结构

```
templates/
├── basic/
│   ├── simple_prompt.txt
│   └── chat_prompt.txt
├── jinja2/
│   ├── complex_analysis.jinja2
│   ├── report_generator.jinja2
│   └── multi_language.jinja2
└── examples/
    ├── few_shot_examples.json
    └── conversation_examples.yaml
```

### 加载方式

```python
# 从文件加载普通模板
with open('templates/basic/simple_prompt.txt', 'r', encoding='utf-8') as f:
    template_content = f.read()

template = PromptTemplate.from_template(template_content)

# 从文件加载 Jinja2 模板
from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader('templates/jinja2'))
template = env.get_template('complex_analysis.jinja2')
```

## 模板管理最佳实践

### 1. 文件命名规范

- 使用描述性的文件名
- 普通模板使用 `.txt` 扩展名
- Jinja2 模板使用 `.jinja2` 扩展名
- 按功能分类组织目录

### 2. 变量命名规范

- 使用清晰、描述性的变量名
- 遵循 Python 命名约定（snake_case）
- 在模板顶部注释说明所需变量

### 3. 模板文档化

```jinja2
{#
模板名称: 产品分析报告生成器
描述: 根据产品数据生成详细的分析报告

必需变量:
- product_name: 产品名称 (string)
- sales_data: 销售数据 (list)
- analysis_period: 分析周期 (string)

可选变量:
- include_charts: 是否包含图表 (boolean, 默认: false)
- language: 报告语言 (string, 默认: 'zh')
#}
```

### 4. 错误处理

- 为必需变量提供默认值或验证
- 使用 try-catch 处理模板渲染错误
- 提供有意义的错误消息

### 5. 性能优化

- 缓存编译后的模板
- 避免在循环中重复编译模板
- 对于大型模板，考虑分块处理

## 实际应用场景

1. **内容生成**：博客文章、产品描述、营销文案
2. **数据分析**：报告生成、数据摘要、趋势分析
3. **客户服务**：自动回复、FAQ 生成、问题分类
4. **代码生成**：API 文档、测试用例、配置文件
5. **多语言支持**：国际化内容、本地化文档

## 运行示例

```bash
# 安装依赖
pip install jinja2

# 运行演示脚本
python prompt_templates_demo.py
```

## 文件说明

- `prompt_templates_demo.py`: 主要演示脚本，包含所有模板类型的示例
- `templates/`: 模板文件目录
  - `basic/`: 基础文本模板
  - `jinja2/`: Jinja2 模板文件
  - `examples/`: 示例数据文件

## 下一步

学习完提示词模板后，建议继续学习：

1. **输出解析器**：如何解析和验证模型输出
2. **记忆管理**：如何在对话中保持上下文
3. **代理系统**：如何构建智能代理
4. **工具集成**：如何集成外部工具和API

通过掌握提示词模板，你将能够创建更加灵活、可维护和可重用的 LangChain 应用程序。