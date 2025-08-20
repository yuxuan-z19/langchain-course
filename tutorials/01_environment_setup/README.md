# LangChain 初体验

## 📚 本节目标

通过本节学习，你将：
- 了解 LangChain 的基本概念和核心组件
- 学会使用 LangChain 与大语言模型进行交互
- 掌握创建简单聊天应用的方法
- 体验 LangChain 的便捷性和强大功能

## 🤖 什么是 LangChain？

LangChain 是一个用于开发由语言模型驱动的应用程序的框架。它提供了：

- **模型抽象**：统一的接口来调用不同的语言模型（DeepSeek、OpenAI、Anthropic、本地模型等）
- **链式组合**：将多个组件链接在一起，创建复杂的应用程序
- **记忆管理**：为对话添加上下文记忆功能
- **工具集成**：轻松集成外部工具和API
- **模板系统**：管理和优化提示词模板

## 🚀 核心概念

### 1. LLM (Large Language Model)
语言模型是 LangChain 的核心，负责生成文本响应。

### 2. Prompt Templates
提示词模板帮助你构建结构化的输入，提高模型响应质量。

### 3. Chains
将多个组件连接起来，形成完整的应用程序流程。

### 4. Memory
为对话添加记忆功能，让AI能够记住之前的对话内容。

## 📁 本节内容

### simple_chat.py
- 最基础的 LangChain 使用示例
- 与 DeepSeek 模型进行单次对话
- 了解基本的模型调用方法
- 代码简洁易懂，适合初学者入门
- 使用国内友好的 DeepSeek API

## 🛠️ 环境要求

确保你已经：
1. ✅ 激活了虚拟环境：`source venv/bin/activate`
2. ✅ 安装了依赖：`pip install -r requirements.txt`
3. ✅ 配置了环境变量：复制 `.env.example` 为 `.env` 并填入你的 API 密钥

## 🔑 API 密钥配置

在项目根目录的 `.env` 文件中配置：

```bash
# DeepSeek API 密钥
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 可选：其他模型的API密钥
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 如何获取 DeepSeek API 密钥？

1. 访问 [DeepSeek 官网](https://www.deepseek.com/)
2. 注册账号并登录
3. 进入 API 管理页面
4. 创建新的 API 密钥
5. 将密钥复制到 `.env` 文件中

## 🏃‍♂️ 快速开始

### 运行简单聊天示例
```bash
python tutorials/01_environment_setup/simple_chat.py
```

## 💡 学习提示

1. **保持简单**：本节专注于最基础的 LangChain 使用，不涉及复杂功能
2. **阅读代码**：仔细阅读代码注释，理解每一步的作用
3. **动手实践**：尝试修改代码中的问题，观察不同参数的效果
4. **循序渐进**：掌握基础后再学习后续章节的高级功能

## 🔍 常见问题

### Q: 运行时提示 API 密钥错误？
A: 检查 `.env` 文件是否正确配置，确保 API 密钥有效且有足够的配额。

### Q: 模型响应很慢？
A: 这是正常现象，API 调用需要网络请求时间。可以尝试使用更快的模型或调整参数。

### Q: 想使用其他模型？
A: LangChain 支持多种模型，可以在后续章节学习如何切换和配置不同的模型。

## 📖 下一步

完成本节后，你可以继续学习：
- **02_langchain_basics**: LangChain 基础组件详解
- **03_advanced_chains**: 高级链式应用
- **04_langgraph_intro**: LangGraph 图形化工作流

---

🎉 **准备好了吗？让我们开始 LangChain 的奇妙之旅！**