# LangChain 基础教学

配置 `.env` 文件：

```bash
cp .env.example .env
```

对于 01-langchain-basic 到 09-text2sql 可以使用 BDMI 课程提供的 API：

```bash
OPENAI_API_KEY=sk-xxxxx
OPENAI_BASE_URL=http://166.111.238.55:11800/v1
OPENAI_MODEL=Qwen/Qwen3-32B
```

对于 10-langchain-chunk 到 13-langgraph-basic 请使用学校平台提供的 API：

```bash
OPENAI_API_KEY=sk-xxxxx
OPENAI_BASE_URL=https://llmapi.paratera.com/v1
OPENAI_MODEL=DeepSeek-V4-Flash
EMBEDDING_MODEL=GLM-Embedding-3
```

15-langgraph-deepsearch 到 17-mem0 使用搜索，在平台API的基础上、需要到 [Tavily](https://www.tavily.com/) 注册账号并获取 API Key，每个月免费 1000 credits：

```bash
TAVILY_API_KEY=tvly-dev-xxxxxxxxxxxx
```

> 14-multiRAG 需要另外配置腾讯云 COS 的 API Key 属于选讲内容。
