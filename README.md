# LangChain & LangGraph 教程项目

> 本项目为 [jaguarliu.cool/course/langchain-course](https://cnb.cool/jaguarliu.cool/course/langchain-course) 的 GitHub 镜像，修复了旧接口不匹配、代码错误等问题。

🚀 一个全面的LangChain和LangGraph学习教程，从基础概念到高级应用的完整学习路径。

## 📋 项目概述

本项目提供了一个渐进式的LangChain和LangGraph学习体验，包含：

- 📚 **20个LangChain教程**：已完成，涵盖从基础聊天到高级RAG系统
- 🔄 **LangGraph教程**：即将开始，图结构AI应用开发
- 🛠️ **实战项目**：多模态RAG、文档问答、智能聊天等
- 🔧 **完整的开发环境**：Python虚拟环境、依赖管理、配置系统
- 📖 **详细的文档**：每个概念都有清晰的解释和示例

## 🏗️ 项目结构

```
langchain-course/
├── tutorials/                    # 教程模块
│   ├── 01_simple_chat/          # ✅ 简单聊天
│   ├── 02_message_types/        # ✅ 消息类型处理
│   ├── 03_invoke_methods/       # ✅ 调用方法
│   ├── 04_chain_operations/     # ✅ 链式操作
│   ├── 05_structured_output/    # ✅ 结构化输出
│   ├── 06_prompt_templates/     # ✅ 提示模板
│   ├── 07_memory_management/    # ✅ 内存管理
│   ├── 08_redis_memory/         # ✅ Redis内存
│   ├── 09_memory_compression/   # ✅ 内存压缩
│   ├── 10_langsmith/           # ✅ LangSmith监控
│   ├── 11_langserve/           # ✅ LangServe部署
│   ├── 12_function_calling/     # ✅ 函数调用
│   ├── 13_document_loaders/     # ✅ 文档加载器
│   ├── 14_multimodal_rag/      # ✅ 多模态RAG
│   ├── 15_text_splitters/      # ✅ 文本分割
│   ├── 16_vector_models/       # ✅ 向量模型
│   ├── 17_vector_databases/    # ✅ 向量数据库
│   ├── 18_parent_child_chunks/ # ✅ 父子分块
│   ├── 19_query_rewriting/     # ✅ 查询重写
│   ├── 20_multi_query/         # ✅ 多查询RAG
│   └── 21_langgraph_intro/     # 🔄 LangGraph入门（即将开始）
├── utils/                       # 工具模块
│   ├── __init__.py
│   ├── config.py               # 配置管理
│   ├── llm_factory.py          # LLM工厂
│   └── vector_store_factory.py # 向量存储工厂
├── tests/                      # 测试文件
├── docs/                       # 文档
├── requirements.txt            # 项目依赖
├── .env.example               # 环境变量模板
└── README.md                  # 项目说明
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

### 🎯 已完成：LangChain基础到高级 (01-20)

#### 基础入门 (01-06)
- **01_simple_chat** ✅ - LangChain基础聊天功能
- **02_message_types** ✅ - 消息类型和处理机制
- **03_invoke_methods** ✅ - 不同的调用方法
- **04_chain_operations** ✅ - 链式操作和组合
- **05_structured_output** ✅ - 结构化输出处理
- **06_prompt_templates** ✅ - 提示模板设计

#### 内存与状态管理 (07-09)
- **07_memory_management** ✅ - 对话内存管理
- **08_redis_memory** ✅ - Redis持久化内存
- **09_memory_compression** ✅ - 内存压缩技术

#### 工具与部署 (10-12)
- **10_langsmith** ✅ - LangSmith监控和追踪
- **11_langserve** ✅ - LangServe API部署
- **12_function_calling** ✅ - 函数调用和工具使用

#### 文档处理与RAG (13-20)
- **13_document_loaders** ✅ - 各种文档加载器
- **14_multimodal_rag** ✅ - 多模态RAG系统（支持PPT、图片等）
- **15_text_splitters** ✅ - 文本分割策略
- **16_vector_models** ✅ - 向量模型和嵌入
- **17_vector_databases** ✅ - 向量数据库集成
- **18_parent_child_chunks** ✅ - 父子分块技术
- **19_query_rewriting** ✅ - 查询重写优化
- **20_multi_query** ✅ - 多查询RAG系统

### 🔄 即将开始：LangGraph图结构编程 (21+)

#### LangGraph基础
- **21_langgraph_intro** 🔄 - LangGraph入门和概念
- **22_graph_nodes** 📋 - 图节点设计
- **23_graph_edges** 📋 - 图边和流程控制
- **24_state_management** 📋 - 状态管理
- **25_conditional_flows** 📋 - 条件分支流程

#### LangGraph高级应用
- **26_parallel_processing** 📋 - 并行处理
- **27_error_handling** 📋 - 错误处理和恢复
- **28_custom_tools** 📋 - 自定义工具集成
- **29_production_deployment** 📋 - 生产环境部署
- **30_monitoring_optimization** 📋 - 监控和性能优化

## 🛠️ 实战项目特色

### 🎯 已实现的核心功能

#### 1. 多模态RAG系统 (14_multimodal_rag)
- 📄 支持PDF、Word、Markdown、PPT文件解析
- 🖼️ 图片提取和云存储集成（腾讯云COS）
- 🔍 多模态内容检索和问答
- 🎨 可视化展示和调试功能

#### 2. 高级RAG技术栈
- 🔄 **多查询RAG** (20_multi_query) - 查询扩展和结果融合
- ✏️ **查询重写** (19_query_rewriting) - 智能查询优化
- 🏗️ **父子分块** (18_parent_child_chunks) - 层次化文档处理
- 🗄️ **向量数据库** (17_vector_databases) - 多种向量存储支持

#### 3. 生产级功能
- 📊 **LangSmith监控** (10_langsmith) - 完整的追踪和分析
- 🚀 **LangServe部署** (11_langserve) - API服务化部署
- 🧠 **智能内存管理** (07-09) - 多种内存策略
- 🔧 **函数调用** (12_function_calling) - 工具集成

### 🔄 即将推出：LangGraph应用

#### 1. 智能工作流系统
- 图结构任务编排
- 条件分支决策
- 并行任务处理
- 状态持久化

#### 2. 复杂AI代理
- 多步骤推理
- 工具链组合
- 自适应策略
- 错误恢复机制

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

### ✅ 已掌握技能（LangChain 01-20）

完成LangChain部分后，你已经能够：

- ✅ **基础应用开发** - 构建智能聊天系统和对话应用
- ✅ **高级RAG系统** - 实现多模态、多查询、父子分块等先进技术
- ✅ **生产级部署** - 使用LangServe部署API服务
- ✅ **监控和优化** - 集成LangSmith进行性能追踪
- ✅ **内存管理** - 实现多种内存策略和持久化
- ✅ **工具集成** - 开发和使用自定义函数工具
- ✅ **文档处理** - 处理PDF、Word、PPT等多种格式
- ✅ **向量技术** - 掌握嵌入模型和向量数据库

### 🔄 即将掌握技能（LangGraph 21+）

完成LangGraph部分后，你将能够：

- 🎯 **图结构编程** - 设计复杂的AI工作流
- 🎯 **状态管理** - 实现持久化和动态状态控制
- 🎯 **条件分支** - 构建智能决策流程
- 🎯 **并行处理** - 优化性能和资源利用
- 🎯 **错误恢复** - 构建健壮的AI系统
- 🎯 **复杂代理** - 开发多步骤推理AI代理

---

## 🚀 开始学习

### 📖 对于新学习者
从 `tutorials/01_simple_chat/` 开始，按顺序完成每个LangChain教程。每个教程都包含详细的说明、代码示例和实践练习。

### 🔄 对于已完成LangChain的学习者
准备开始LangGraph学习之旅！即将推出的 `tutorials/21_langgraph_intro/` 将带你进入图结构AI编程的新世界。

### 🎯 学习建议
1. **循序渐进** - 按照编号顺序学习，每个教程都建立在前面的基础上
2. **动手实践** - 运行每个示例代码，修改参数观察效果
3. **深入理解** - 阅读代码注释，理解每个组件的作用
4. **扩展应用** - 尝试将学到的技术应用到自己的项目中

🌟 **祝学习愉快！** 🌟