# LangChain & LangGraph 教程项目

🚀 一个全面的LangChain和LangGraph学习教程，从基础概念到生产部署的完整学习路径。

## 📋 项目概述

本项目提供了一个渐进式的LangChain和LangGraph学习体验，包含：

- 📚 **6个核心教程模块**：从环境配置到生产部署
- 🛠️ **3个实战项目**：聊天机器人、文档问答、代码生成器
- 🔧 **完整的开发环境**：Python虚拟环境、依赖管理、配置系统
- 📖 **详细的文档**：每个概念都有清晰的解释和示例

## 🏗️ 项目结构

```
langchain-course/
├── tutorials/                 # 教程模块
│   ├── 01_environment_setup/  # 环境配置
│   ├── 02_langchain_basics/   # LangChain基础
│   ├── 03_advanced_chains/    # 高级链式操作
│   ├── 04_langgraph_intro/    # LangGraph入门
│   ├── 05_custom_tools/       # 自定义工具
│   └── 06_production_ready/   # 生产部署
├── examples/                  # 实战项目
│   ├── chatbot/              # 聊天机器人
│   ├── document_qa/          # 文档问答系统
│   └── code_generator/       # 代码生成器
├── utils/                    # 工具模块
│   ├── __init__.py
│   └── config.py            # 配置管理
├── tests/                   # 测试文件
├── docs/                    # 文档
├── requirements.txt         # 项目依赖
├── .env.example            # 环境变量模板
├── setup.py               # 安装配置
└── README.md              # 项目说明
```

## 🚀 快速开始

### 1. 环境要求

- Python 3.11 或更高版本
- pip 或 conda 包管理器
- Git（用于克隆项目）

### 2. 安装步骤

#### 步骤 1：克隆项目
```bash
git clone <repository-url>
cd langchain-course
```

#### 步骤 2：创建虚拟环境
```bash
# 使用 venv
python -m venv langchain_env

# 激活虚拟环境
# Windows
langchain_env\Scripts\activate
# macOS/Linux
source langchain_env/bin/activate
```

#### 步骤 3：安装依赖
```bash
pip install -r requirements.txt
```

#### 步骤 4：配置环境变量
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的API密钥
# 至少需要配置 OPENAI_API_KEY
```

#### 步骤 5：验证安装
```bash
python -m utils.config
```

如果看到 "✅ Configuration loaded successfully"，说明环境配置成功！

## 🔑 API密钥配置

### 必需的API密钥

1. **OpenAI API Key**（必需）
   - 访问 [OpenAI API](https://platform.openai.com/api-keys)
   - 创建新的API密钥
   - 在 `.env` 文件中设置 `OPENAI_API_KEY=your_key_here`

### 可选的API密钥

2. **LangChain API Key**（用于追踪和监控）
   - 访问 [LangSmith](https://smith.langchain.com/)
   - 获取API密钥
   - 设置 `LANGCHAIN_API_KEY=your_key_here`

3. **其他服务**（根据需要配置）
   - Anthropic Claude: `ANTHROPIC_API_KEY`
   - Google Gemini: `GOOGLE_API_KEY`
   - Hugging Face: `HUGGINGFACE_API_TOKEN`

## 📚 学习路径

### 模块 1：环境配置 (01_environment_setup)
- Python环境设置
- 依赖管理
- API密钥配置
- 基础测试

### 模块 2：LangChain基础 (02_langchain_basics)
- LLM基础概念
- 提示模板
- 链式操作
- 内存管理

### 模块 3：高级链式操作 (03_advanced_chains)
- 复杂链构建
- 条件分支
- 并行处理
- 错误处理

### 模块 4：LangGraph入门 (04_langgraph_intro)
- 图结构概念
- 节点和边
- 状态管理
- 工作流设计

### 模块 5：自定义工具 (05_custom_tools)
- 工具开发
- 外部API集成
- 函数调用
- 工具链组合

### 模块 6：生产部署 (06_production_ready)
- 性能优化
- 错误处理
- 监控和日志
- 部署策略

## 🛠️ 实战项目

### 1. 聊天机器人 (examples/chatbot)
- 基础对话系统
- 上下文记忆
- 个性化回复
- Web界面

### 2. 文档问答系统 (examples/document_qa)
- 文档解析
- 向量存储
- 语义搜索
- 答案生成

### 3. 代码生成器 (examples/code_generator)
- 代码理解
- 自动生成
- 代码优化
- 测试生成

## 🧪 测试

运行所有测试：
```bash
pytest tests/
```

运行特定测试：
```bash
pytest tests/test_config.py -v
```

## 📖 文档

- [API文档](docs/api.md)
- [配置指南](docs/configuration.md)
- [故障排除](docs/troubleshooting.md)
- [贡献指南](docs/contributing.md)

## 🤝 贡献

欢迎贡献代码、报告问题或提出改进建议！

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🆘 获取帮助

如果遇到问题：

1. 查看 [故障排除文档](docs/troubleshooting.md)
2. 搜索现有的 [Issues](../../issues)
3. 创建新的 Issue 描述你的问题
4. 加入我们的社区讨论

## 🎯 学习目标

完成本教程后，你将能够：

- ✅ 熟练使用LangChain构建AI应用
- ✅ 掌握LangGraph的图结构编程
- ✅ 开发自定义工具和集成
- ✅ 部署生产级AI应用
- ✅ 优化性能和处理错误
- ✅ 实现复杂的AI工作流

---

🌟 **开始你的LangChain学习之旅吧！** 🌟

从 `tutorials/01_environment_setup/` 开始，按顺序完成每个模块。每个教程都包含详细的说明、代码示例和练习。

祝学习愉快！ 🚀