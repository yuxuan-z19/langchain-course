"""配置管理模块

提供环境变量加载、验证和配置管理功能。
"""

from dotenv import load_dotenv
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """环境配置数据模型"""
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    langchain_api_key: Optional[str] = None
    langchain_tracing: bool = False
    langchain_project: str = "langchain-tutorial"
    langchain_endpoint: str = "https://api.smith.langchain.com"
    log_level: str = "INFO"
    debug: bool = False
    
    # 可选API密钥
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    huggingface_api_token: Optional[str] = None
    
    # 数据库配置
    database_url: Optional[str] = None
    chroma_persist_directory: Optional[str] = None
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None


@dataclass
class TutorialConfig:
    """教程配置数据模型"""
    current_chapter: str = "01_environment_setup"
    completed_chapters: list = None
    user_progress: float = 0.0
    last_accessed: Optional[str] = None
    
    def __post_init__(self):
        if self.completed_chapters is None:
            self.completed_chapters = []


def load_environment() -> EnvironmentConfig:
    """加载环境变量配置
    
    Returns:
        EnvironmentConfig: 环境配置对象
        
    Raises:
        ValueError: 当必需的环境变量缺失时
    """
    # 加载.env文件
    load_dotenv()
    
    # 必需的环境变量
    required_vars = [
        'DEEPSEEK_API_KEY'
    ]
    
    # 检查必需的环境变量
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            f"Please copy .env.example to .env and fill in the required values."
        )
    
    # 创建配置对象
    config = EnvironmentConfig(
        deepseek_api_key=os.getenv('DEEPSEEK_API_KEY'),
        deepseek_base_url=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
        langchain_api_key=os.getenv('LANGCHAIN_API_KEY'),
        langchain_tracing=os.getenv('LANGCHAIN_TRACING_V2', 'false').lower() == 'true',
        langchain_project=os.getenv('LANGCHAIN_PROJECT', 'langchain-tutorial'),
        langchain_endpoint=os.getenv('LANGCHAIN_ENDPOINT', 'https://api.smith.langchain.com'),
        log_level=os.getenv('LOG_LEVEL', 'INFO'),
        debug=os.getenv('DEBUG', 'false').lower() == 'true',
        
        # 可选API密钥
        anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
        google_api_key=os.getenv('GOOGLE_API_KEY'),
        huggingface_api_token=os.getenv('HUGGINGFACE_API_TOKEN'),
        
        # 数据库配置
        database_url=os.getenv('DATABASE_URL'),
        chroma_persist_directory=os.getenv('CHROMA_PERSIST_DIRECTORY'),
        pinecone_api_key=os.getenv('PINECONE_API_KEY'),
        pinecone_environment=os.getenv('PINECONE_ENVIRONMENT')
    )
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, config.log_level.upper()))
    
    logger.info("Environment configuration loaded successfully")
    if config.langchain_tracing:
        logger.info("LangChain tracing is enabled")
    
    return config


def get_config_dict() -> Dict[str, Any]:
    """获取配置字典格式
    
    Returns:
        Dict[str, Any]: 配置字典
    """
    config = load_environment()
    return {
        'deepseek_api_key': config.deepseek_api_key,
        'deepseek_base_url': config.deepseek_base_url,
        'langchain_api_key': config.langchain_api_key,
        'langchain_tracing': config.langchain_tracing,
        'langchain_project': config.langchain_project,
        'log_level': config.log_level,
        'debug': config.debug
    }


def validate_api_keys() -> Dict[str, bool]:
    """验证API密钥的有效性
    
    Returns:
        Dict[str, bool]: 各API密钥的验证状态
    """
    config = load_environment()
    validation_results = {}
    
    # 验证DeepSeek API密钥
    if config.deepseek_api_key:
        validation_results['deepseek'] = len(config.deepseek_api_key) > 10
    else:
        validation_results['deepseek'] = False
    
    # 验证LangChain API密钥
    if config.langchain_api_key:
        validation_results['langchain'] = len(config.langchain_api_key) > 10
    else:
        validation_results['langchain'] = False
    
    # 验证其他API密钥
    validation_results['anthropic'] = bool(config.anthropic_api_key)
    validation_results['google'] = bool(config.google_api_key)
    validation_results['huggingface'] = bool(config.huggingface_api_token)
    
    return validation_results


def setup_environment_variables():
    """设置环境变量到系统环境中"""
    config = load_environment()
    
    # 设置DeepSeek API密钥
    os.environ['DEEPSEEK_API_KEY'] = config.deepseek_api_key
    os.environ['DEEPSEEK_BASE_URL'] = config.deepseek_base_url
    
    # 设置LangChain相关环境变量
    if config.langchain_api_key:
        os.environ['LANGCHAIN_API_KEY'] = config.langchain_api_key
    
    if config.langchain_tracing:
        os.environ['LANGCHAIN_TRACING_V2'] = 'true'
        os.environ['LANGCHAIN_PROJECT'] = config.langchain_project
        os.environ['LANGCHAIN_ENDPOINT'] = config.langchain_endpoint
    
    logger.info("Environment variables set successfully")


def load_deepseek_config() -> Dict[str, str]:
    """加载DeepSeek API配置
    
    Returns:
        Dict[str, str]: 包含api_key和base_url的配置字典
        
    Raises:
        ValueError: 当DeepSeek API密钥缺失时
    """
    config = load_environment()
    return {
        'api_key': config.deepseek_api_key,
        'base_url': config.deepseek_base_url
    }


if __name__ == "__main__":
    # 测试配置加载
    try:
        config = load_environment()
        print("✅ Configuration loaded successfully")
        print(f"📊 LangChain tracing: {config.langchain_tracing}")
        print(f"📝 Log level: {config.log_level}")
        
        # 验证API密钥
        validation = validate_api_keys()
        print("\n🔑 API Key Validation:")
        for service, is_valid in validation.items():
            status = "✅" if is_valid else "❌"
            print(f"  {service}: {status}")
            
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")