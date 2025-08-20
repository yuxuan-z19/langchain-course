#!/bin/bash
# LangChain教程项目环境设置脚本

set -e  # 遇到错误时退出

echo "🚀 开始设置LangChain教程项目环境..."

# 检查Python版本
echo "📋 检查Python版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python版本检查通过: $python_version"
else
    echo "⚠️  Python版本较低: $python_version (推荐 >= $required_version)"
    echo "💡 虚拟环境将使用当前Python版本，某些功能可能需要更高版本"
    echo "🔧 继续创建虚拟环境..."
fi

# 创建虚拟环境
echo "🔧 创建Python虚拟环境..."
if [ -d "langchain_env" ]; then
    echo "⚠️  虚拟环境已存在，跳过创建"
else
    python3 -m venv langchain_env
    echo "✅ 虚拟环境创建成功"
fi

# 激活虚拟环境
echo "🔌 激活虚拟环境..."
source langchain_env/bin/activate
echo "✅ 虚拟环境已激活"

# 升级pip
echo "📦 升级pip..."
pip install --upgrade pip
echo "✅ pip升级完成"

# 安装依赖
echo "📚 安装项目依赖..."
pip install -r requirements.txt
echo "✅ 依赖安装完成"

# 检查环境变量文件
echo "🔑 检查环境变量配置..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "📋 已创建.env文件，请编辑并填入你的API密钥"
        echo "⚠️  重要：请在.env文件中设置OPENAI_API_KEY"
    else
        echo "❌ 未找到.env.example文件"
    fi
else
    echo "✅ .env文件已存在"
fi

# 验证配置
echo "🧪 验证环境配置..."
if python -m utils.config; then
    echo "✅ 环境配置验证成功"
else
    echo "⚠️  环境配置验证失败，请检查.env文件中的API密钥"
fi

echo ""
echo "🎉 环境设置完成！"
echo ""
echo "📝 下一步操作："
echo "1. 编辑 .env 文件，填入你的API密钥"
echo "2. 运行 'source langchain_env/bin/activate' 激活环境"
echo "3. 开始学习教程：cd tutorials/01_environment_setup"
echo ""
echo "💡 提示：每次使用项目时，请先激活虚拟环境"
echo "   source langchain_env/bin/activate"
echo ""
echo "🆘 如需帮助，请查看 README.md 文件"