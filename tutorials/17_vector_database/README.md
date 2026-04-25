# 第17小节：LangChain + 向量数据库（ChromaDB）

## 概述

本小节将深入探讨如何将LangChain与向量数据库结合使用，重点介绍ChromaDB的使用方法。我们将学习向量数据库的核心概念，以及如何实现数据的增删改查、相似性搜索、MMR搜索和混合搜索等高级功能。

## 什么是向量数据库？

### 向量数据库的定义

向量数据库是专门用于存储、索引和查询高维向量数据的数据库系统。它们针对向量相似性搜索进行了优化，能够高效处理大规模的向量数据。

### 向量数据库 vs 传统数据库

| 特性 | 传统数据库 | 向量数据库 |
|------|------------|------------|
| 数据类型 | 结构化数据（文本、数字） | 高维向量数据 |
| 查询方式 | 精确匹配、范围查询 | 相似性搜索、近似查询 |
| 索引方式 | B树、哈希索引 | 向量索引（HNSW、IVF等） |
| 应用场景 | 传统业务系统 | AI、机器学习、推荐系统 |
| 查询复杂度 | O(log n) | O(log n) 到 O(√n) |

### 向量数据库的优势

1. **高效相似性搜索**：专门优化的算法和索引结构
2. **可扩展性**：支持大规模向量数据存储和查询
3. **实时性**：支持实时数据插入和查询
4. **多样化搜索**：支持多种相似性度量和搜索策略

## ChromaDB 简介

### 什么是 ChromaDB？

ChromaDB是一个开源的向量数据库，专为AI应用设计。它提供了简单易用的API，支持多种向量相似性搜索算法，并且可以轻松集成到Python应用中。

### ChromaDB 的特点

- **轻量级**：易于安装和部署
- **Python原生**：完全用Python编写，API友好
- **持久化**：支持数据持久化存储
- **多模态**：支持文本、图像等多种数据类型
- **可扩展**：支持分布式部署

## 安装和配置

### 安装 ChromaDB

```bash
pip install chromadb
```

### 基本配置

```python
import chromadb
from chromadb.config import Settings

# 创建客户端
client = chromadb.Client()

# 持久化配置
client = chromadb.PersistentClient(path="./chroma_db")
```

## 核心概念

### Collection（集合）

Collection是ChromaDB中的基本存储单元，类似于传统数据库中的表。每个Collection包含：

- **Documents**：原始文档内容
- **Embeddings**：文档的向量表示
- **Metadata**：文档的元数据
- **IDs**：文档的唯一标识符

### 向量嵌入（Embeddings）

向量嵌入是将文本、图像等数据转换为高维向量的过程。ChromaDB支持：

- 自动嵌入生成
- 自定义嵌入函数
- 预计算嵌入导入

## 核心操作（CRUD）

### 1. 创建和获取Collection

```python
# 创建新集合
collection = client.create_collection(name="my_collection")

# 获取现有集合
collection = client.get_collection(name="my_collection")

# 获取或创建集合
collection = client.get_or_create_collection(name="my_collection")
```

### 2. 添加数据（Create）

```python
# 添加文档
collection.add(
    documents=["这是第一个文档", "这是第二个文档"],
    metadatas=[{"source": "doc1"}, {"source": "doc2"}],
    ids=["id1", "id2"]
)
```

### 3. 查询数据（Read）

```python
# 相似性搜索
results = collection.query(
    query_texts=["查询文本"],
    n_results=5
)

# 根据ID获取
results = collection.get(ids=["id1", "id2"])
```

### 4. 更新数据（Update）

```python
# 更新文档
collection.update(
    ids=["id1"],
    documents=["更新后的文档内容"],
    metadatas=[{"source": "updated_doc1"}]
)
```

### 5. 删除数据（Delete）

```python
# 删除指定文档
collection.delete(ids=["id1", "id2"])

# 根据条件删除
collection.delete(where={"source": "doc1"})
```

## 搜索方法详解

### 1. 相似性搜索（Similarity Search）

相似性搜索是向量数据库的核心功能，通过计算查询向量与数据库中向量的相似度来找到最相关的结果。

#### 相似度度量方法

- **余弦相似度**：计算向量夹角的余弦值
- **欧几里得距离**：计算向量间的直线距离
- **点积**：计算向量的内积

```python
# 基本相似性搜索
results = collection.query(
    query_texts=["民法典相关条款"],
    n_results=10,
    include=["documents", "distances", "metadatas"]
)
```

### 2. MMR搜索（Maximal Marginal Relevance）

MMR搜索是一种平衡相关性和多样性的搜索策略，避免返回过于相似的结果。

#### MMR算法原理

1. **相关性**：结果与查询的相似度
2. **多样性**：结果之间的差异性
3. **平衡参数λ**：控制相关性和多样性的权重

```python
# MMR搜索实现
def mmr_search(collection, query, k=10, lambda_param=0.5):
    # 获取更多候选结果
    candidates = collection.query(
        query_texts=[query],
        n_results=k * 2
    )
    
    # 实现MMR算法
    selected = []
    remaining = list(range(len(candidates['documents'][0])))
    
    while len(selected) < k and remaining:
        if not selected:
            # 选择最相关的第一个结果
            best_idx = 0
        else:
            # 计算MMR分数
            best_score = -float('inf')
            best_idx = None
            
            for idx in remaining:
                relevance = 1 - candidates['distances'][0][idx]
                diversity = min([calculate_diversity(idx, sel_idx, candidates) 
                               for sel_idx in selected])
                mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
        
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    return selected
```

### 3. 混合搜索（Hybrid Search）

混合搜索结合了向量搜索和传统的关键词搜索，提供更全面的搜索结果。

#### 混合搜索策略

1. **向量搜索**：基于语义相似性
2. **关键词搜索**：基于精确匹配
3. **结果融合**：合并和排序两种搜索结果

```python
# 混合搜索实现
def hybrid_search(collection, query, keywords=None, vector_weight=0.7):
    results = {}
    
    # 向量搜索
    vector_results = collection.query(
        query_texts=[query],
        n_results=20
    )
    
    # 关键词搜索（通过metadata过滤）
    if keywords:
        keyword_results = collection.query(
            query_texts=[query],
            where_document={"$contains": keywords},
            n_results=20
        )
    else:
        keyword_results = vector_results
    
    # 结果融合
    combined_results = combine_search_results(
        vector_results, keyword_results, vector_weight
    )
    
    return combined_results
```

## 性能优化

### 1. 索引优化

```python
# 配置索引参数
collection = client.create_collection(
    name="optimized_collection",
    metadata={"hnsw:space": "cosine", "hnsw:M": 16}
)
```

### 2. 批量操作

```python
# 批量添加数据
batch_size = 1000
for i in range(0, len(documents), batch_size):
    batch_docs = documents[i:i+batch_size]
    batch_ids = [f"doc_{j}" for j in range(i, min(i+batch_size, len(documents)))]
    
    collection.add(
        documents=batch_docs,
        ids=batch_ids
    )
```

### 3. 内存管理

```python
# 定期清理和压缩
collection.delete(where={"timestamp": {"$lt": old_timestamp}})
```

## 最佳实践

### 1. 数据预处理

- **文本清洗**：去除噪声和无关信息
- **分块策略**：合理分割长文档
- **元数据设计**：添加有用的元数据信息

### 2. 向量质量

- **选择合适的嵌入模型**：根据领域选择专业模型
- **向量维度平衡**：在性能和精度间找到平衡
- **定期更新**：保持向量的时效性

### 3. 查询优化

- **查询重写**：优化查询文本
- **结果过滤**：使用元数据过滤不相关结果
- **缓存策略**：缓存常用查询结果

### 4. 监控和维护

- **性能监控**：监控查询延迟和吞吐量
- **数据质量**：定期检查数据完整性
- **版本管理**：管理数据和模型版本

## 实际应用场景

### 1. 文档检索系统

- 法律文档检索
- 技术文档搜索
- 学术论文查找

### 2. 推荐系统

- 内容推荐
- 商品推荐
- 用户匹配

### 3. 问答系统

- 智能客服
- 知识库问答
- 教育辅助

### 4. 内容分析

- 文本分类
- 情感分析
- 主题发现

## 总结

ChromaDB作为一个轻量级但功能强大的向量数据库，为AI应用提供了优秀的向量存储和检索能力。通过本小节的学习，我们掌握了：

1. 向量数据库的基本概念和优势
2. ChromaDB的安装配置和基本使用
3. 数据的增删改查操作
4. 多种搜索策略的实现和应用
5. 性能优化和最佳实践

在下一个演示文件中，我们将通过实际代码来演示这些概念和技术的具体实现。