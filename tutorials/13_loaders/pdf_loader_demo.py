#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF文档加载器演示

本演示展示如何使用LangChain的文档加载器来处理PDF文件，
包括文档加载、文本提取、分块处理等核心功能。

作者: Jaguarliu
日期: 2025年8月
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


class PDFLoaderDemo:
    """PDF文档加载器演示类"""

    def __init__(self):
        """初始化演示类"""
        self.pdf_path = project_root / "docs" / "中华人民共和国民法典.pdf"
        self.documents: List[Document] = []
        self.chunks: List[Document] = []

        # 配置文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 每个文本块的大小
            chunk_overlap=200,  # 文本块之间的重叠
            length_function=len,  # 长度计算函数
            separators=[
                "\n\n",
                "\n",
                "。",
                "！",
                "？",
                "；",
                "，",
                " ",
                "",
            ],  # 分割符优先级
        )

    def check_pdf_file(self) -> bool:
        """检查PDF文件是否存在"""
        if not self.pdf_path.exists():
            print(f"❌ PDF文件不存在: {self.pdf_path}")
            print("请确保docs目录下有'中华人民共和国民法典.pdf'文件")
            return False

        file_size = self.pdf_path.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ 找到PDF文件: {self.pdf_path.name}")
        print(f"📁 文件大小: {file_size:.2f} MB")
        return True

    def load_pdf_document(self) -> bool:
        """加载PDF文档"""
        try:
            print("\n🔄 开始加载PDF文档...")

            # 使用PyPDFLoader加载PDF
            loader = PyPDFLoader(str(self.pdf_path))
            self.documents = loader.load()

            print(f"✅ 成功加载PDF文档")
            print(f"📄 总页数: {len(self.documents)}")

            # 显示文档基本信息
            if self.documents:
                first_page = self.documents[0]
                print(f"📝 第一页内容预览 (前200字符):")
                print(f"   {first_page.page_content[:200]}...")

                # 显示元数据
                if first_page.metadata:
                    print(f"📋 文档元数据:")
                    for key, value in first_page.metadata.items():
                        print(f"   {key}: {value}")

            return True

        except Exception as e:
            print(f"❌ 加载PDF文档失败: {str(e)}")
            return False

    def analyze_document_content(self):
        """分析文档内容"""
        if not self.documents:
            print("❌ 没有加载的文档可供分析")
            return

        print("\n📊 文档内容分析:")

        # 统计总字符数
        total_chars = sum(len(doc.page_content) for doc in self.documents)
        print(f"📝 总字符数: {total_chars:,}")

        # 统计平均每页字符数
        avg_chars_per_page = total_chars / len(self.documents)
        print(f"📄 平均每页字符数: {avg_chars_per_page:.0f}")

        # 找出最长和最短的页面
        page_lengths = [len(doc.page_content) for doc in self.documents]
        max_page_idx = page_lengths.index(max(page_lengths))
        min_page_idx = page_lengths.index(min(page_lengths))

        print(f"📏 最长页面: 第{max_page_idx + 1}页 ({page_lengths[max_page_idx]}字符)")
        print(f"📏 最短页面: 第{min_page_idx + 1}页 ({page_lengths[min_page_idx]}字符)")

        # 显示页面长度分布
        print(f"\n📈 页面长度分布:")
        length_ranges = [(0, 500), (500, 1000), (1000, 2000), (2000, float("inf"))]
        for min_len, max_len in length_ranges:
            count = sum(1 for length in page_lengths if min_len <= length < max_len)
            range_str = f"{min_len}-{max_len if max_len != float('inf') else '∞'}"
            print(f"   {range_str}字符: {count}页")

    def split_documents(self) -> bool:
        """分割文档为文本块"""
        if not self.documents:
            print("❌ 没有加载的文档可供分割")
            return False

        try:
            print("\n🔄 开始分割文档...")

            # 使用文本分割器分割文档
            self.chunks = self.text_splitter.split_documents(self.documents)

            print(f"✅ 文档分割完成")
            print(f"📦 生成文本块数量: {len(self.chunks)}")

            return True

        except Exception as e:
            print(f"❌ 文档分割失败: {str(e)}")
            return False

    def analyze_chunks(self):
        """分析文本块"""
        if not self.chunks:
            print("❌ 没有文本块可供分析")
            return

        print("\n📊 文本块分析:")

        # 统计文本块长度
        chunk_lengths = [len(chunk.page_content) for chunk in self.chunks]
        avg_chunk_length = sum(chunk_lengths) / len(chunk_lengths)

        print(f"📦 文本块数量: {len(self.chunks)}")
        print(f"📏 平均块长度: {avg_chunk_length:.0f}字符")
        print(f"📏 最长块: {max(chunk_lengths)}字符")
        print(f"📏 最短块: {min(chunk_lengths)}字符")

        # 显示文本块长度分布
        print(f"\n📈 文本块长度分布:")
        length_ranges = [
            (0, 500),
            (500, 800),
            (800, 1000),
            (1000, 1200),
            (1200, float("inf")),
        ]
        for min_len, max_len in length_ranges:
            count = sum(1 for length in chunk_lengths if min_len <= length < max_len)
            range_str = f"{min_len}-{max_len if max_len != float('inf') else '∞'}"
            print(f"   {range_str}字符: {count}块")

        # 显示几个示例文本块
        print(f"\n📝 文本块示例:")
        for i, chunk in enumerate(self.chunks[:3]):
            print(f"\n--- 文本块 {i+1} ---")
            print(f"长度: {len(chunk.page_content)}字符")
            print(f"来源: {chunk.metadata.get('source', 'Unknown')}")
            print(f"页码: {chunk.metadata.get('page', 'Unknown')}")
            print(f"内容预览: {chunk.page_content[:200]}...")

    def search_content(self, query: str) -> List[Document]:
        """在文档中搜索内容"""
        if not self.chunks:
            print("❌ 没有文本块可供搜索")
            return []

        print(f"\n🔍 搜索关键词: '{query}'")

        # 简单的文本搜索
        matching_chunks = []
        for chunk in self.chunks:
            if query.lower() in chunk.page_content.lower():
                matching_chunks.append(chunk)

        print(f"📋 找到 {len(matching_chunks)} 个相关文本块")

        # 显示前几个匹配结果
        for i, chunk in enumerate(matching_chunks[:3]):
            print(f"\n--- 搜索结果 {i+1} ---")
            print(f"页码: {chunk.metadata.get('page', 'Unknown')}")

            # 找到关键词在文本中的位置
            content = chunk.page_content
            query_pos = content.lower().find(query.lower())
            if query_pos != -1:
                start = max(0, query_pos - 100)
                end = min(len(content), query_pos + len(query) + 100)
                context = content[start:end]
                print(f"上下文: ...{context}...")

        return matching_chunks

    def run_demo(self):
        """运行完整演示"""
        print("🚀 PDF文档加载器演示开始")
        print("=" * 50)

        # 1. 检查PDF文件
        if not self.check_pdf_file():
            return

        # 2. 加载PDF文档
        if not self.load_pdf_document():
            return

        # 3. 分析文档内容
        self.analyze_document_content()

        # 4. 分割文档
        if not self.split_documents():
            return

        # 5. 分析文本块
        self.analyze_chunks()

        # 6. 演示内容搜索
        search_queries = ["合同", "财产", "责任"]
        for query in search_queries:
            self.search_content(query)

        print("\n✅ PDF文档加载器演示完成")
        print("=" * 50)

        # 7. 总结信息
        print(f"\n📋 演示总结:")
        print(f"   📄 处理文档: {self.pdf_path.name}")
        print(f"   📝 总页数: {len(self.documents)}")
        print(f"   📦 文本块数: {len(self.chunks)}")
        print(f"   🔧 分块策略: 递归字符分割 (块大小: 1000, 重叠: 200)")
        print(f"   💡 应用场景: RAG系统的文档预处理")


def main():
    """主函数"""
    try:
        # 创建演示实例
        demo = PDFLoaderDemo()

        # 运行演示
        demo.run_demo()

    except KeyboardInterrupt:
        print("\n\n⚠️ 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
