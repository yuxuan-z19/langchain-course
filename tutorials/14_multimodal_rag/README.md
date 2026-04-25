# 第14节：多模态RAG与腾讯云COS集成

本节介绍如何构建多模态RAG（检索增强生成）系统，支持从各种文档格式中提取图片，并将图片上传到腾讯云COS（对象存储）进行统一管理。

## 🎯 学习目标

- 理解多模态RAG的概念和应用场景
- 掌握从PDF、Word、Markdown等文档中提取图片的方法
- 学会集成腾讯云COS进行图片存储和管理
- 实现文档中图片引用的自动替换和URL更新

## 📋 功能特性

### 支持的文档格式
- **PDF文档**: 提取嵌入的图片资源
- **Word文档**: 提取.docx文件中的图片
- **Markdown文档**: 识别并处理图片引用

### 图片处理能力
- 自动提取文档中的图片
- 生成唯一的图片标识（MD5哈希）
- 获取图片尺寸和格式信息
- 支持多种图片格式（JPG、PNG、GIF、BMP等）

### 腾讯云COS集成
- 自动上传图片到COS存储桶
- 生成可访问的图片URL
- 支持自定义域名配置
- 失败时自动回退到模拟模式

## 🛠️ 环境配置

### 1. 安装依赖

```bash
pip install cos-python-sdk-v5 PyMuPDF python-docx Pillow
```

### 2. 配置腾讯云COS

在项目根目录的`.env`文件中添加以下配置：

```env
# 腾讯云COS配置
COS_SECRET_ID=your_secret_id_here
COS_SECRET_KEY=your_secret_key_here
COS_REGION=ap-shanghai
COS_BUCKET=your-bucket-name
COS_DOMAIN=your-custom-domain.com  # 可选，自定义域名
```

### 3. 获取腾讯云COS凭证

1. 登录[腾讯云控制台](https://console.cloud.tencent.com/)
2. 进入「访问管理」→「API密钥管理」
3. 创建或查看SecretId和SecretKey
4. 进入「对象存储」创建存储桶
5. 记录存储桶名称和所在地域

## 📖 使用指南

### 基础用法

```python
from multimodal_loader_demo import MultimodalDocumentLoader

# 初始化加载器（自动从.env读取配置）
loader = MultimodalDocumentLoader()

# 处理单个文档
result = loader.process_document("path/to/your/document.pdf")

if result['success']:
    print(f"提取图片: {len(result['images'])}张")
    print(f"上传COS: {len(result['cos_urls'])}个URL")
    
    # 查看上传的图片URL
    for url_info in result['cos_urls']:
        print(f"图片: {url_info['filename']} -> {url_info['cos_url']}")
else:
    print(f"处理失败: {result.get('error')}")
```

### 批量处理

```python
# 批量处理多个文档
documents = [
    "document1.pdf",
    "document2.docx", 
    "document3.md"
]

for doc_path in documents:
    result = loader.process_document(doc_path)
    # 处理结果...
```

### Markdown图片引用替换

对于Markdown文档，系统会自动将本地图片引用替换为COS URL：

```markdown
# 原始内容
![示例图片](./images/sample.png)

# 处理后内容
![示例图片](https://your-bucket.cos.ap-shanghai.myqcloud.com/images/md_sample_a1b2c3d4.png)
```

## 🔧 高级配置

### 自定义存储桶和区域

```python
# 覆盖环境变量中的配置
loader = MultimodalDocumentLoader(
    cos_bucket="custom-bucket",
    cos_region="ap-beijing"
)
```

### 处理结果结构

```python
result = {
    'file_path': str,           # 文档路径
    'file_type': str,           # 文件类型(.pdf, .docx, .md)
    'images': [                 # 提取的图片信息
        {
            'data': bytes,      # 图片二进制数据
            'format': str,      # 图片格式
            'filename': str,    # 生成的文件名
            'size': tuple,      # 图片尺寸(width, height)
            'hash': str,        # MD5哈希值
            'alt_text': str,    # 替代文本(仅Markdown)
        }
    ],
    'cos_urls': [              # COS上传结果
        {
            'filename': str,    # 文件名
            'cos_url': str,     # COS访问URL
            'original_info': dict  # 原始图片信息
        }
    ],
    'original_content': str,    # 原始内容(仅Markdown)
    'processed_content': str,   # 处理后内容(仅Markdown)
    'success': bool,           # 处理是否成功
    'error': str               # 错误信息(如果有)
}
```

## 🚀 运行演示

```bash
cd tutorials/14_multimodal_rag
python multimodal_loader_demo.py
```

演示程序会：
1. 测试PDF文档的图片提取
2. 测试Markdown文档的图片处理
3. 展示COS上传功能
4. 显示处理结果统计

## ⚠️ 注意事项

1. **权限配置**: 确保COS密钥具有存储桶的读写权限
2. **网络连接**: COS上传需要稳定的网络连接
3. **存储费用**: 上传到COS会产生存储和流量费用
4. **文件大小**: 大文件上传可能需要较长时间
5. **模拟模式**: 配置错误时会自动切换到模拟模式

## 🔍 故障排除

### 常见问题

**Q: 提示"COS配置加载失败"**
A: 检查`.env`文件中的COS配置是否正确，确保所有必需字段都已填写

**Q: 上传失败，显示权限错误**
A: 验证SecretId和SecretKey是否正确，检查存储桶权限设置

**Q: 无法提取PDF中的图片**
A: 确保安装了PyMuPDF库，某些PDF可能使用了特殊的图片编码

**Q: Markdown图片路径无法识别**
A: 检查图片路径是否正确，支持相对路径和绝对路径

## 📚 相关资源

- [腾讯云COS Python SDK文档](https://cloud.tencent.com/document/product/436/12269)
- [PyMuPDF文档](https://pymupdf.readthedocs.io/)
- [python-docx文档](https://python-docx.readthedocs.io/)
- [Pillow文档](https://pillow.readthedocs.io/)

## 🎉 下一步

完成本节学习后，你可以：
- 将多模态RAG集成到你的应用中
- 探索更多文档格式的支持
- 优化图片处理和存储策略
- 构建基于图片的检索和问答系统