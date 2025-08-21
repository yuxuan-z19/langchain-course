"""LLM工厂模块

提供创建各种LLM实例的工厂函数。
"""

from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from .config import load_deepseek_config


def create_deepseek_llm(
    model: str = "deepseek-chat",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs
) -> ChatOpenAI:
    """创建DeepSeek LLM实例
    
    Args:
        model: 模型名称，默认为"deepseek-chat"
        temperature: 温度参数，控制输出随机性
        max_tokens: 最大token数
        **kwargs: 其他参数
        
    Returns:
        ChatOpenAI: DeepSeek LLM实例
        
    Raises:
        ValueError: 当API密钥缺失时
    """
    # 加载DeepSeek配置
    config = load_deepseek_config()
    
    if not config["api_key"]:
        raise ValueError("DEEPSEEK_API_KEY环境变量未设置")
    
    # 创建ChatOpenAI实例（DeepSeek兼容OpenAI API）
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=config["api_key"],
        openai_api_base=config["base_url"],
        **kwargs
    )
    
    return llm


def create_openai_llm(
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> ChatOpenAI:
    """创建OpenAI LLM实例
    
    Args:
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大token数
        api_key: API密钥，如果不提供则从环境变量读取
        **kwargs: 其他参数
        
    Returns:
        ChatOpenAI: OpenAI LLM实例
    """
    import os
    
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY环境变量未设置")
    
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=api_key,
        **kwargs
    )
    
    return llm


def get_available_models() -> Dict[str, list]:
    """获取可用的模型列表
    
    Returns:
        Dict[str, list]: 按提供商分组的模型列表
    """
    return {
        "deepseek": [
            "deepseek-chat",
            "deepseek-coder"
        ],
        "openai": [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o"
        ]
    }


def create_llm_from_config(provider: str, **kwargs) -> ChatOpenAI:
    """根据配置创建LLM实例
    
    Args:
        provider: 提供商名称（deepseek, openai）
        **kwargs: 其他参数
        
    Returns:
        ChatOpenAI: LLM实例
        
    Raises:
        ValueError: 当提供商不支持时
    """
    if provider.lower() == "deepseek":
        return create_deepseek_llm(**kwargs)
    elif provider.lower() == "openai":
        return create_openai_llm(**kwargs)
    else:
        raise ValueError(f"不支持的提供商: {provider}")