"""LangChain教程项目安装配置"""

from setuptools import setup, find_packages
import os

# 读取README文件作为长描述
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取requirements.txt文件
def read_requirements():
    """读取requirements.txt文件并返回依赖列表"""
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 跳过注释和空行
                if line and not line.startswith("#"):
                    requirements.append(line)
    return requirements

setup(
    name="langchain-tutorial",
    version="1.0.0",
    author="LangChain Tutorial Team",
    author_email="tutorial@example.com",
    description="一个全面的LangChain和LangGraph学习教程项目",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/langchain-tutorial",
    project_urls={
        "Bug Tracker": "https://github.com/example/langchain-tutorial/issues",
        "Documentation": "https://github.com/example/langchain-tutorial/docs",
        "Source Code": "https://github.com/example/langchain-tutorial",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
    ],
    packages=find_packages(exclude=["tests*", "docs*"]),
    python_requires=">=3.11",
    install_requires=[
        "langchain>=0.1.0",
        "langgraph>=0.0.40",
        "python-dotenv>=1.0.0",
        "openai>=1.0.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "web": [
            "streamlit>=1.28.0",
            "gradio>=4.0.0",
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
        ],
        "data": [
            "pandas>=2.0.0",
            "numpy>=1.24.0",
        ],
        "docs": [
            "pypdf2>=3.0.0",
            "docx2txt>=0.8",
        ],
        "vector": [
            "chromadb>=0.4.0",
            "faiss-cpu>=1.7.0",
        ],
        "all": [
            # 开发依赖
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            # Web界面
            "streamlit>=1.28.0",
            "gradio>=4.0.0",
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            # 数据处理
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            # 文档处理
            "pypdf2>=3.0.0",
            "docx2txt>=0.8",
            # 向量数据库
            "chromadb>=0.4.0",
            "faiss-cpu>=1.7.0",
            # 其他工具
            "requests>=2.31.0",
            "pyyaml>=6.0.0",
            "tqdm>=4.65.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "langchain-tutorial=utils.config:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml", "*.json"],
        "tutorials": ["**/*.py", "**/*.md", "**/*.ipynb"],
        "examples": ["**/*.py", "**/*.md", "**/*.ipynb"],
        "docs": ["**/*.md", "**/*.rst"],
    },
    zip_safe=False,
    keywords=[
        "langchain",
        "langgraph",
        "ai",
        "machine learning",
        "tutorial",
        "education",
        "llm",
        "large language model",
        "chatbot",
        "nlp",
    ],
    license="MIT",
)