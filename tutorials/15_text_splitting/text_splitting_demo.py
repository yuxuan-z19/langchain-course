#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档切分技术演示

本脚本演示了多种文档切分方法，包括：
1. 基于长度的切分（字符数、Token数）
2. 基于文本结构的切分（递归、段落、句子）
3. 基于文档格式的切分（Markdown、代码）
4. 基于语义的切分（语义相似度）
5. 切分效果对比和质量评估
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from langchain.schema import Document

# LangChain导入
from langchain.text_splitter import (
    CharacterTextSplitter,
    Language,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_community.document_loaders import PyPDFLoader

# 尝试导入语义切分器（可能需要额外安装）
try:
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai import OpenAIEmbeddings

    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("注意：语义切分器不可用，需要安装 langchain-experimental 和配置 OpenAI API")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("text_splitting_demo.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class TextSplittingDemo:
    """文档切分演示类"""

    def __init__(self):
        self.test_document = None
        self.results = {}

    def load_test_document(self, file_path: str) -> Optional[str]:
        """加载测试文档"""
        try:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                text = "\n\n".join([doc.page_content for doc in documents])
                logger.info(f"成功加载PDF文档，总长度: {len(text)} 字符")
                return text
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                logger.info(f"成功加载文本文档，总长度: {len(text)} 字符")
                return text
        except Exception as e:
            logger.error(f"加载文档失败: {e}")
            return None

    def create_sample_texts(self) -> Dict[str, str]:
        """创建示例文本用于演示"""
        return {
            "markdown_text": """
# 人工智能发展史

## 早期发展（1950-1980）

人工智能的概念最早可以追溯到1950年，当时艾伦·图灵提出了著名的"图灵测试"。
这个测试成为了判断机器是否具有智能的重要标准。

### 符号主义时期

在1960年代，研究者们主要关注符号推理和专家系统。
这一时期的代表性工作包括：
- ELIZA聊天机器人
- MYCIN医疗诊断系统
- DENDRAL化学分析系统

## 现代发展（1980-至今）

### 机器学习兴起

1980年代开始，机器学习方法逐渐兴起，特别是神经网络的发展。

### 深度学习革命

2010年代，深度学习技术取得了突破性进展：
- 2012年：AlexNet在ImageNet竞赛中获胜
- 2016年：AlphaGo击败世界围棋冠军
- 2017年：Transformer架构提出
- 2018年：BERT模型发布
- 2020年：GPT-3展示强大的语言能力
""",
            "python_code": '''def calculate_fibonacci(n):
    """计算斐波那契数列的第n项"""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

class DataProcessor:
    """数据处理器类"""
    
    def __init__(self, data_source):
        self.data_source = data_source
        self.processed_data = []
    
    def load_data(self):
        """加载数据"""
        try:
            with open(self.data_source, 'r') as f:
                raw_data = f.read()
            return raw_data
        except FileNotFoundError:
            print(f"文件 {self.data_source} 不存在")
            return None
    
    def process_data(self, raw_data):
        """处理数据"""
        if raw_data:
            lines = raw_data.split('\n')
            self.processed_data = [line.strip() for line in lines if line.strip()]
        return self.processed_data

if __name__ == "__main__":
    processor = DataProcessor("data.txt")
    data = processor.load_data()
    if data:
        result = processor.process_data(data)
        print(f"处理完成，共 {len(result)} 行数据")
''',
        }

    def demo_character_splitter(self, text: str) -> List[Document]:
        """演示字符切分器"""
        logger.info("=== 字符切分器演示 ===")

        splitter = CharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, separator="\n\n"
        )

        chunks = splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]

        logger.info(f"切分结果: {len(documents)} 个片段")
        for i, doc in enumerate(documents[:3]):  # 只显示前3个
            logger.info(
                f"片段 {i+1} (长度: {len(doc.page_content)}): {doc.page_content[:100]}..."
            )

        return documents

    def demo_recursive_splitter(self, text: str) -> List[Document]:
        """演示递归字符切分器"""
        logger.info("=== 递归字符切分器演示 ===")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", ";", ",", " ", ""],
        )

        chunks = splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]

        logger.info(f"切分结果: {len(documents)} 个片段")
        for i, doc in enumerate(documents[:3]):
            logger.info(
                f"片段 {i+1} (长度: {len(doc.page_content)}): {doc.page_content[:100]}..."
            )

        return documents

    def demo_token_splitter(self, text: str) -> List[Document]:
        """演示Token切分器"""
        logger.info("=== Token切分器演示 ===")

        try:
            splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=20)

            chunks = splitter.split_text(text)
            documents = [Document(page_content=chunk) for chunk in chunks]

            logger.info(f"切分结果: {len(documents)} 个片段")
            for i, doc in enumerate(documents[:3]):
                logger.info(
                    f"片段 {i+1} (长度: {len(doc.page_content)}): {doc.page_content[:100]}..."
                )

            return documents
        except Exception as e:
            logger.error(f"Token切分器演示失败: {e}")
            return []

    def demo_markdown_splitter(self, markdown_text: str) -> List[Document]:
        """演示Markdown标题切分器"""
        logger.info("=== Markdown标题切分器演示 ===")

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )

        documents = markdown_splitter.split_text(markdown_text)

        logger.info(f"切分结果: {len(documents)} 个片段")
        for i, doc in enumerate(documents):
            metadata = doc.metadata if hasattr(doc, "metadata") else {}
            logger.info(f"片段 {i+1} - 元数据: {metadata}")
            logger.info(
                f"内容 (长度: {len(doc.page_content)}): {doc.page_content[:100]}..."
            )

        return documents

    def demo_python_code_splitter(self, python_code: str) -> List[Document]:
        """演示Python代码切分器"""
        logger.info("=== Python代码切分器演示 ===")

        python_splitter = PythonCodeTextSplitter(chunk_size=300, chunk_overlap=30)

        chunks = python_splitter.split_text(python_code)
        documents = [Document(page_content=chunk) for chunk in chunks]

        logger.info(f"切分结果: {len(documents)} 个片段")
        for i, doc in enumerate(documents):
            logger.info(f"片段 {i+1} (长度: {len(doc.page_content)}):")
            logger.info(f"{doc.page_content[:200]}...")

        return documents

    def demo_semantic_splitter(self, text: str) -> List[Document]:
        """演示语义切分器"""
        logger.info("=== 语义切分器演示 ===")

        if not SEMANTIC_AVAILABLE:
            logger.warning("语义切分器不可用，跳过演示")
            return []

        try:
            # 需要配置OpenAI API Key
            embeddings = OpenAIEmbeddings()
            semantic_splitter = SemanticChunker(
                embeddings=embeddings, breakpoint_threshold_type="percentile"
            )

            documents = semantic_splitter.create_documents([text])

            logger.info(f"切分结果: {len(documents)} 个片段")
            for i, doc in enumerate(documents[:3]):
                logger.info(
                    f"片段 {i+1} (长度: {len(doc.page_content)}): {doc.page_content[:100]}..."
                )

            return documents
        except Exception as e:
            logger.error(f"语义切分器演示失败: {e}")
            return []

    def evaluate_splitting_quality(
        self, documents: List[Document], method_name: str
    ) -> Dict[str, Any]:
        """评估切分质量"""
        if not documents:
            return {"method": method_name, "error": "无文档"}

        # 基本统计
        chunk_lengths = [len(doc.page_content) for doc in documents]

        metrics = {
            "method": method_name,
            "total_chunks": len(documents),
            "avg_length": sum(chunk_lengths) / len(chunk_lengths),
            "min_length": min(chunk_lengths),
            "max_length": max(chunk_lengths),
            "length_std": self._calculate_std(chunk_lengths),
            "total_chars": sum(chunk_lengths),
        }

        # 计算长度分布
        length_ranges = {
            "0-100": sum(1 for l in chunk_lengths if l <= 100),
            "101-300": sum(1 for l in chunk_lengths if 100 < l <= 300),
            "301-500": sum(1 for l in chunk_lengths if 300 < l <= 500),
            "501-1000": sum(1 for l in chunk_lengths if 500 < l <= 1000),
            "1000+": sum(1 for l in chunk_lengths if l > 1000),
        }
        metrics["length_distribution"] = length_ranges

        return metrics

    def _calculate_std(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance**0.5

    def compare_splitting_methods(self, text: str, sample_texts: Dict[str, str]):
        """对比不同切分方法"""
        logger.info("\n" + "=" * 50)
        logger.info("开始对比不同切分方法")
        logger.info("=" * 50)

        methods_and_results = []

        # 1. 字符切分
        char_docs = self.demo_character_splitter(text)
        char_metrics = self.evaluate_splitting_quality(char_docs, "字符切分")
        methods_and_results.append(("字符切分", char_docs, char_metrics))

        # 2. 递归切分
        recursive_docs = self.demo_recursive_splitter(text)
        recursive_metrics = self.evaluate_splitting_quality(recursive_docs, "递归切分")
        methods_and_results.append(("递归切分", recursive_docs, recursive_metrics))

        # 3. Token切分
        token_docs = self.demo_token_splitter(text)
        token_metrics = self.evaluate_splitting_quality(token_docs, "Token切分")
        methods_and_results.append(("Token切分", token_docs, token_metrics))

        # 4. Markdown切分（使用示例文本）
        if "markdown_text" in sample_texts:
            md_docs = self.demo_markdown_splitter(sample_texts["markdown_text"])
            md_metrics = self.evaluate_splitting_quality(md_docs, "Markdown切分")
            methods_and_results.append(("Markdown切分", md_docs, md_metrics))

        # 5. Python代码切分（使用示例代码）
        if "python_code" in sample_texts:
            py_docs = self.demo_python_code_splitter(sample_texts["python_code"])
            py_metrics = self.evaluate_splitting_quality(py_docs, "Python代码切分")
            methods_and_results.append(("Python代码切分", py_docs, py_metrics))

        # 6. 语义切分
        semantic_docs = self.demo_semantic_splitter(text)
        semantic_metrics = self.evaluate_splitting_quality(semantic_docs, "语义切分")
        methods_and_results.append(("语义切分", semantic_docs, semantic_metrics))

        # 输出对比结果
        self.print_comparison_results(methods_and_results)

        return methods_and_results

    def print_comparison_results(self, methods_and_results: List):
        """打印对比结果"""
        logger.info("\n" + "=" * 50)
        logger.info("切分方法对比结果")
        logger.info("=" * 50)

        # 表头
        header = f"{'方法':<12} {'片段数':<8} {'平均长度':<10} {'最小长度':<10} {'最大长度':<10} {'标准差':<10}"
        logger.info(header)
        logger.info("-" * len(header))

        # 数据行
        for method_name, docs, metrics in methods_and_results:
            if "error" not in metrics:
                row = f"{metrics['method']:<12} {metrics['total_chunks']:<8} {metrics['avg_length']:<10.1f} {metrics['min_length']:<10} {metrics['max_length']:<10} {metrics['length_std']:<10.1f}"
                logger.info(row)
            else:
                logger.info(f"{method_name:<12} 错误: {metrics['error']}")

        # 详细分布信息
        logger.info("\n长度分布详情:")
        for method_name, docs, metrics in methods_and_results:
            if "error" not in metrics and "length_distribution" in metrics:
                logger.info(f"\n{metrics['method']}:")
                for range_name, count in metrics["length_distribution"].items():
                    percentage = (
                        (count / metrics["total_chunks"]) * 100
                        if metrics["total_chunks"] > 0
                        else 0
                    )
                    logger.info(f"  {range_name}: {count} 个片段 ({percentage:.1f}%)")

    def save_results_to_file(
        self, methods_and_results: List, output_file: str = "splitting_results.json"
    ):
        """保存结果到文件"""
        results = []
        for method_name, docs, metrics in methods_and_results:
            result = {
                "method": method_name,
                "metrics": metrics,
                "sample_chunks": [
                    doc.page_content[:200] + "..." for doc in docs[:3]
                ],  # 保存前3个片段的样本
            }
            results.append(result)

        try:
            with open("splitting_results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info("结果已保存到 splitting_results.json")
        except Exception as e:
            logger.error(f"保存结果失败: {e}")


def main():
    """主函数"""
    demo = TextSplittingDemo()

    # 测试文档路径 - 使用项目根目录的绝对路径
    project_root = Path(__file__).parent.parent.parent
    pdf_path = str(project_root / "docs" / "中华人民共和国民法典.pdf")

    # 检查PDF文件是否存在
    if os.path.exists(pdf_path):
        logger.info(f"使用PDF文档进行测试: {pdf_path}")
        text = demo.load_test_document(pdf_path)
        if not text:
            logger.error("无法加载PDF文档，使用示例文本")
            text = "这是一个测试文档。" * 100  # 创建简单的测试文本
    else:
        logger.warning(f"PDF文档不存在: {pdf_path}，使用示例文本")
        text = (
            """人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
        
        人工智能的研究历史有着一条从以"推理"为重点，到以"知识"为重点，再到以"学习"为重点的自然、清晰的脉络。显然，智能水平的提高是一个不断学习的过程。
        
        机器学习是人工智能的核心，是使计算机具有智能的根本途径。机器学习的应用遍及人工智能的各个分支，如专家系统、自动推理、自然语言理解、模式识别、计算机视觉、智能机器人等领域。
        
        深度学习是机器学习的一个分支，它基于人工神经网络的研究，特别是利用多层次的神经网络来进行学习和表示。深度学习的概念源于人工神经网络的研究，含多隐层的多层感知器就是一种深度学习结构。
        
        自然语言处理（Natural Language Processing，NLP）是人工智能和语言学领域的分支学科。此领域探讨如何处理及运用自然语言；自然语言处理包括多个方面和步骤，基本有认知、理解、生成等部分。
        
        计算机视觉是一门研究如何使机器"看"的科学，更进一步的说，就是是指用摄影机和电脑代替人眼对目标进行识别、跟踪和测量等机器视觉，并进一步做图形处理。"""
            * 5
        )

    # 创建示例文本
    sample_texts = demo.create_sample_texts()

    # 执行对比测试
    start_time = time.time()
    methods_and_results = demo.compare_splitting_methods(text, sample_texts)
    end_time = time.time()

    logger.info(f"\n总耗时: {end_time - start_time:.2f} 秒")

    # 保存结果
    output_file = "tutorials/15_text_splitting/splitting_results.json"
    demo.save_results_to_file(methods_and_results, output_file)

    logger.info("\n文档切分演示完成！")
    logger.info("\n建议:")
    logger.info("1. 根据文档类型选择合适的切分方法")
    logger.info("2. 调整chunk_size和chunk_overlap参数以优化效果")
    logger.info("3. 对于结构化文档，优先使用格式感知的切分器")
    logger.info("4. 在质量要求高的场景中考虑使用语义切分")
    logger.info("5. 通过A/B测试验证切分效果对RAG性能的影响")


if __name__ == "__main__":
    main()
