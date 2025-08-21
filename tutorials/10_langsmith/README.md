# LangChain 第10节：LangSmith 接入与监控

## 概述

LangSmith 是 LangChain 官方提供的统一可观测性和评估平台，用于调试、测试和监控 AI 应用性能。无论您是否使用 LangChain 构建应用，LangSmith 都能为您的 LLM 应用提供全面的监控和分析能力。

## LangSmith 的核心功能

### 1. 全链路追踪 (Tracing)
- **完整的执行轨迹**：记录应用中每个步骤的输入和输出
- **性能监控**：追踪延迟、token 使用量、成本等关键指标
- **错误诊断**：快速定位和分析应用中的问题
- **可视化界面**：直观展示复杂链式调用的执行流程

### 2. 评估系统 (Evaluations)
- **自动化评估**：使用 LLM-as-Judge 进行自动评分
- **人工反馈**：支持人工标注和反馈收集
- **基准测试**：对比不同模型和配置的性能
- **持续监控**：生产环境中的实时质量监控

### 3. 数据集管理
- **测试数据集**：管理和版本化测试用例
- **生产数据**：收集和分析真实用户交互数据
- **数据标注**：支持团队协作进行数据标注

### 4. 提示词管理
- **版本控制**：管理提示词的不同版本
- **A/B 测试**：对比不同提示词的效果
- **协作编辑**：团队共同优化提示词

## 接入线上 LangSmith

### 注册和获取 API Key

1. **访问 LangSmith 官网**
   - 打开浏览器访问：https://smith.langchain.com
   - 使用 GitHub 或 Google 账号注册登录

2. **创建项目**
   - 在 LangSmith 控制台中创建新项目
   - 记录项目名称，后续配置中会用到

3. **获取 API Key**
   - 在设置页面生成 API Key
   - 妥善保存 API Key，不要泄露给他人

### 环境配置

本项目使用统一的配置管理系统。请在项目根目录的 `.env` 文件中配置以下环境变量：

```bash
# LangSmith 线上服务配置
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your-api-key  # 从 LangSmith 控制台获取
LANGCHAIN_PROJECT=your-project-name  # 您创建的项目名称
```

**注意**：本教程的示例代码会自动使用项目根目录的 `utils.config` 模块来加载这些配置，无需手动设置环境变量。

### 快速验证

配置完成后，可以通过以下代码验证连接：

```python
from langsmith import Client
from utils.config import load_environment

# 加载项目统一配置
config = load_environment()
client = Client()
print(f"连接成功！当前项目：{config.langchain_project}")
```

## 应用接入示例

### 简单应用接入

`simple_langsmith_demo.py` 演示了如何将基础的 LLM 应用接入 LangSmith：

- 使用项目统一配置管理
- 基础配置和初始化
- 简单的问答对话追踪
- 基本的性能监控
- 错误处理和日志记录

### 复杂链式应用接入

`complex_chain_demo.py` 展示了复杂应用的监控：

- 使用项目统一配置管理
- 多步骤链式调用追踪
- 自定义标签和元数据
- 分支逻辑监控
- 性能优化分析
- 成本追踪和分析

## 最佳实践

### 1. 项目组织
- 为不同的应用创建独立的项目
- 使用有意义的项目名称和描述
- 合理设置标签和元数据

### 2. 追踪策略
- 在开发环境中启用详细追踪
- 在生产环境中选择性追踪关键路径
- 定期清理历史数据

### 3. 评估设计
- 设计全面的评估指标
- 建立基准数据集
- 定期进行模型对比

### 4. 性能优化
- 监控 token 使用量和成本
- 分析延迟瓶颈
- 优化提示词效率

## 与其他章节的关系

- **第03节 (invoke方法)**：LangSmith 自动追踪 invoke 调用
- **第04节 (链式调用)**：完整追踪链式操作的每个步骤
- **第07-09节 (记忆系统)**：监控记忆使用和压缩效果
- **第06节 (提示词模板)**：追踪不同模板的性能表现

## 故障排除

### 常见问题

1. **连接失败**
   - 检查网络连接是否正常
   - 确认 LANGCHAIN_ENDPOINT 设置为 https://api.smith.langchain.com
   - 验证防火墙是否允许 HTTPS 连接

2. **追踪数据不显示**
   - 确认环境变量配置正确
   - 检查 API Key 是否有效且未过期
   - 验证项目名称是否存在于您的 LangSmith 账户中
   - 确认 LANGCHAIN_TRACING_V2=true

3. **API 配额问题**
   - 检查您的 LangSmith 账户配额使用情况
   - 考虑升级到付费计划以获得更多配额
   - 优化追踪频率以减少 API 调用

## 进阶功能

### 自定义评估器
```python
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run

def custom_evaluator(run: Run, example: Example) -> dict:
    # 自定义评估逻辑
    return {"score": 0.8, "reasoning": "评估原因"}
```

### 批量数据分析
```python
from langsmith import Client

client = Client()
runs = client.list_runs(project_name="your-project")
# 分析运行数据
```

## 总结

LangSmith 为 LLM 应用开发提供了强大的可观测性和评估能力。通过本节的学习，您将掌握：

1. LangSmith 的核心概念和功能
2. 本地部署和配置方法
3. 简单和复杂应用的接入方式
4. 监控和优化的最佳实践

这些技能将帮助您构建更可靠、更高效的 LLM 应用，并在生产环境中持续优化应用性能。