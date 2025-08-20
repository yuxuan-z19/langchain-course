@echo off
REM LangChain教程项目环境设置脚本 (Windows)

echo 🚀 开始设置LangChain教程项目环境...

REM 检查Python版本
echo 📋 检查Python版本...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到Python，请先安装Python 3.11或更高版本
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo ✅ Python版本: %python_version%

REM 创建虚拟环境
echo 🔧 创建Python虚拟环境...
if exist "langchain_env" (
    echo ⚠️  虚拟环境已存在，跳过创建
) else (
    python -m venv langchain_env
    if errorlevel 1 (
        echo ❌ 虚拟环境创建失败
        pause
        exit /b 1
    )
    echo ✅ 虚拟环境创建成功
)

REM 激活虚拟环境
echo 🔌 激活虚拟环境...
call langchain_env\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ 虚拟环境激活失败
    pause
    exit /b 1
)
echo ✅ 虚拟环境已激活

REM 升级pip
echo 📦 升级pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ❌ pip升级失败
    pause
    exit /b 1
)
echo ✅ pip升级完成

REM 安装依赖
echo 📚 安装项目依赖...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ 依赖安装失败
    pause
    exit /b 1
)
echo ✅ 依赖安装完成

REM 检查环境变量文件
echo 🔑 检查环境变量配置...
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo 📋 已创建.env文件，请编辑并填入你的API密钥
        echo ⚠️  重要：请在.env文件中设置OPENAI_API_KEY
    ) else (
        echo ❌ 未找到.env.example文件
    )
) else (
    echo ✅ .env文件已存在
)

REM 验证配置
echo 🧪 验证环境配置...
python -m utils.config
if errorlevel 1 (
    echo ⚠️  环境配置验证失败，请检查.env文件中的API密钥
) else (
    echo ✅ 环境配置验证成功
)

echo.
echo 🎉 环境设置完成！
echo.
echo 📝 下一步操作：
echo 1. 编辑 .env 文件，填入你的API密钥
echo 2. 运行 'langchain_env\Scripts\activate.bat' 激活环境
echo 3. 开始学习教程：cd tutorials\01_environment_setup
echo.
echo 💡 提示：每次使用项目时，请先激活虚拟环境
echo    langchain_env\Scripts\activate.bat
echo.
echo 🆘 如需帮助，请查看 README.md 文件
echo.
pause