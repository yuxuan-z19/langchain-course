#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的Function Calling测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.tools import BaseTool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Type
import math
from utils.config import load_qwen_config

class CalculatorTool(BaseTool):
    """计算器工具"""
    name: str = "calculator"
    description: str = "执行基本的数学计算"
    
    class CalculatorInput(BaseModel):
        expression: str = Field(description="数学表达式")
    
    args_schema: Type[BaseModel] = CalculatorInput
    
    def _run(self, expression: str) -> str:
        try:
            # 安全的数学计算
            allowed_names = {
                k: v for k, v in math.__dict__.items() if not k.startswith("__")
            }
            allowed_names.update({"abs": abs, "round": round})
            result = eval(expression, {