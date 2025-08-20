#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain 链式调用演示

本脚本演示了 LangChain 中各种链的使用方法：
1. 基础链（LLMChain）
2. 顺序链（SequentialChain）
3. 路由链（RouterChain）
4. 转换链（TransformChain）
5. 简单顺序链（SimpleSequentialChain）
6. 映射-归约链（MapReduceChain）

作者：Jaguarliu
日期：2025年8月
"""

import os
import sys
import json
import re
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain, TransformChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.schema import BaseOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入配置管理
try:
    from utils.config import load_deepseek_config, setup_environment_variables
    
    # 加载配置
    setup_environment_variables()
    deepseek_config = load_deepseek_config()
    
    # 配置 DeepSeek API
    os.environ["OPENAI_API_KEY"] = deepseek_config['api_key']
    os.environ["OPENAI_API_BASE"] = deepseek_config['base_url']
    
    print("✅ DeepSeek API 配置加载成功")
    
except Exception as e:
    print(f"⚠️  配置加载失败: {e}")
    print("请确保已正确配置 .env 文件中的 DEEPSEEK_API_KEY")
    sys.exit(1)

# 初始化模型
model = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=1000
)

# ============================================================================
# 1. 基础链（LLMChain）演示
# ============================================================================

def demo_basic_llm_chain():
    """
    演示基础的 LLMChain 使用
    LLMChain 是最简单的链，包含一个提示模板和一个语言模型
    """
    print("\n" + "="*50)
    print("1. 基础链（LLMChain）演示")
    print("="*50)
    
    # 创建提示模板
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="请为以下主题写一首简短的诗：{topic}"
    )
    
    # 创建 LLMChain
    chain = LLMChain(
        llm=model,
        prompt=prompt,
        verbose=True  # 显示详细信息
    )
    
    try:
        # 运行链
        result = chain.run(topic="春天")
        print(f"\n生成的诗歌：\n{result}")
        
        # 也可以使用 invoke 方法
        result2 = chain.invoke({"topic": "科技"})
        print(f"\n使用 invoke 方法的结果：\n{result2['text']}")
        
    except Exception as e:
        print(f"基础链演示出错：{e}")

# ============================================================================
# 2. 输出解析器演示
# ============================================================================

class BookReview(BaseModel):
    """书评结构化输出模型"""
    title: str = Field(description="书名")
    author: str = Field(description="作者")
    rating: int = Field(description="评分（1-5分）")
    summary: str = Field(description="简短总结")
    pros: List[str] = Field(description="优点列表")
    cons: List[str] = Field(description="缺点列表")

def demo_output_parser_chain():
    """
    演示带输出解析器的链
    """
    print("\n" + "="*50)
    print("2. 输出解析器链演示")
    print("="*50)
    
    # 创建输出解析器
    parser = PydanticOutputParser(pydantic_object=BookReview)
    
    # 创建提示模板（包含格式说明）
    prompt = PromptTemplate(
        template="请对以下书籍进行评价：{book_info}\n\n{format_instructions}",
        input_variables=["book_info"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # 创建链
    chain = LLMChain(
        llm=model,
        prompt=prompt,
        output_parser=parser,
        verbose=True
    )
    
    try:
        # 运行链
        book_info = "《三体》是刘慈欣创作的科幻小说，讲述了地球文明与三体文明的故事。"
        result = chain.run(book_info=book_info)
        
        print(f"\n解析后的书评：")
        print(f"书名: {result.title}")
        print(f"作者: {result.author}")
        print(f"评分: {result.rating}/5")
        print(f"总结: {result.summary}")
        print(f"优点: {', '.join(result.pros)}")
        print(f"缺点: {', '.join(result.cons)}")
        
    except Exception as e:
        print(f"输出解析器链演示出错：{e}")
        # 提供降级方案
        try:
            simple_prompt = PromptTemplate(
                template="请对以下书籍进行简单评价：{book_info}",
                input_variables=["book_info"]
            )
            simple_chain = LLMChain(llm=model, prompt=simple_prompt)
            result = simple_chain.run(book_info=book_info)
            print(f"\n简单评价结果：\n{result}")
        except Exception as e2:
            print(f"降级方案也失败：{e2}")

# ============================================================================
# 3. 简单顺序链（SimpleSequentialChain）演示
# ============================================================================

def demo_simple_sequential_chain():
    """
    演示简单顺序链
    SimpleSequentialChain 将多个链按顺序连接，前一个链的输出作为后一个链的输入
    """
    print("\n" + "="*50)
    print("3. 简单顺序链演示")
    print("="*50)
    
    # 第一个链：生成故事大纲
    outline_prompt = PromptTemplate(
        input_variables=["theme"],
        template="请为以下主题创建一个简短的故事大纲：{theme}"
    )
    outline_chain = LLMChain(llm=model, prompt=outline_prompt)
    
    # 第二个链：根据大纲写故事
    story_prompt = PromptTemplate(
        input_variables=["outline"],
        template="根据以下大纲写一个简短的故事：\n{outline}"
    )
    story_chain = LLMChain(llm=model, prompt=story_prompt)
    
    # 创建顺序链
    sequential_chain = SimpleSequentialChain(
        chains=[outline_chain, story_chain],
        verbose=True
    )
    
    try:
        # 运行顺序链
        result = sequential_chain.run("人工智能与人类的友谊")
        print(f"\n最终故事：\n{result}")
        
    except Exception as e:
        print(f"简单顺序链演示出错：{e}")

# ============================================================================
# 4. 复杂顺序链（SequentialChain）演示
# ============================================================================

def demo_sequential_chain():
    """
    演示复杂顺序链
    SequentialChain 可以处理多个输入和输出变量
    """
    print("\n" + "="*50)
    print("4. 复杂顺序链演示")
    print("="*50)
    
    # 第一个链：分析产品需求
    analysis_prompt = PromptTemplate(
        input_variables=["product_idea"],
        template="分析以下产品想法的市场需求和可行性：{product_idea}\n\n请提供：市场分析、技术可行性、竞争分析"
    )
    analysis_chain = LLMChain(
        llm=model,
        prompt=analysis_prompt,
        output_key="analysis"  # 指定输出键名
    )
    
    # 第二个链：制定开发计划
    plan_prompt = PromptTemplate(
        input_variables=["product_idea", "analysis"],
        template="基于产品想法：{product_idea}\n\n和市场分析：{analysis}\n\n制定详细的开发计划，包括时间线和里程碑"
    )
    plan_chain = LLMChain(
        llm=model,
        prompt=plan_prompt,
        output_key="plan"
    )
    
    # 第三个链：估算成本
    cost_prompt = PromptTemplate(
        input_variables=["plan"],
        template="根据以下开发计划估算项目成本：\n{plan}\n\n请提供详细的成本分解"
    )
    cost_chain = LLMChain(
        llm=model,
        prompt=cost_prompt,
        output_key="cost"
    )
    
    # 创建复杂顺序链
    sequential_chain = SequentialChain(
        chains=[analysis_chain, plan_chain, cost_chain],
        input_variables=["product_idea"],
        output_variables=["analysis", "plan", "cost"],
        verbose=True
    )
    
    try:
        # 运行顺序链
        result = sequential_chain({
            "product_idea": "一个基于AI的个人健康管理应用"
        })
        
        print(f"\n市场分析：\n{result['analysis']}")
        print(f"\n开发计划：\n{result['plan']}")
        print(f"\n成本估算：\n{result['cost']}")
        
    except Exception as e:
        print(f"复杂顺序链演示出错：{e}")

# ============================================================================
# 5. 转换链（TransformChain）演示
# ============================================================================

def demo_transform_chain():
    """
    演示转换链
    TransformChain 用于数据转换，不调用语言模型
    """
    print("\n" + "="*50)
    print("5. 转换链演示")
    print("="*50)
    
    def clean_text(inputs: Dict[str, str]) -> Dict[str, str]:
        """
        文本清理函数
        """
        text = inputs["text"]
        
        # 移除多余的空格和换行
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # 移除特殊字符（保留基本标点）
        cleaned = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:]', '', cleaned)
        
        # 统计信息
        word_count = len(cleaned.split())
        char_count = len(cleaned)
        
        return {
            "cleaned_text": cleaned,
            "word_count": word_count,
            "char_count": char_count,
            "original_text": text
        }
    
    # 创建转换链
    transform_chain = TransformChain(
        input_variables=["text"],
        output_variables=["cleaned_text", "word_count", "char_count", "original_text"],
        transform=clean_text
    )
    
    # 创建分析链
    analysis_prompt = PromptTemplate(
        input_variables=["cleaned_text", "word_count", "char_count"],
        template="""请分析以下文本：
        
文本内容：{cleaned_text}
字数：{word_count}
字符数：{char_count}

请提供：
1. 文本主题
2. 情感倾向
3. 关键词
4. 简短总结"""
    )
    analysis_chain = LLMChain(llm=model, prompt=analysis_prompt)
    
    # 组合转换链和分析链
    try:
        combined_chain = SequentialChain(
            chains=[transform_chain, analysis_chain],
            input_variables=["text"],
            output_variables=["cleaned_text", "word_count", "char_count", "original_text"],
            verbose=True
        )
    except Exception as chain_error:
        print(f"演示过程中出现错误：{chain_error}")
        print("请检查网络连接和 API 配置")
        return
    
    try:
        # 测试文本（包含噪声）
        noisy_text = """
        这是一个    包含很多   
        
        噪声的文本！！！@#$%^&*()   
        
        我们需要清理它。。。
        """
        
        result = combined_chain({"text": noisy_text})
        
        print(f"\n原始文本：{result['original_text']}")
        print(f"\n清理后文本：{result['cleaned_text']}")
        print(f"字数：{result['word_count']}")
        print(f"字符数：{result['char_count']}")
        # 分析结果在LLMChain的默认输出键'text'中
        if 'text' in result:
            print(f"\n分析结果：{result['text']}")
        else:
            print("\n分析结果：未生成")
        
    except Exception as e:
        print(f"转换链演示出错：{e}")

# ============================================================================
# 6. 路由链（RouterChain）演示
# ============================================================================

def demo_router_chain():
    """
    演示路由链
    RouterChain 根据输入内容选择不同的处理路径
    """
    print("\n" + "="*50)
    print("6. 路由链演示")
    print("="*50)
    
    # 定义不同的专门链
    
    # 数学问题处理链
    math_template = """你是一个数学专家。请解决以下数学问题：

{input}

请提供详细的解题步骤。"""
    
    # 编程问题处理链
    programming_template = """你是一个编程专家。请帮助解决以下编程问题：

{input}

请提供代码示例和详细解释。"""
    
    # 通用问题处理链
    general_template = """请回答以下问题：

{input}

请提供准确和有用的信息。"""
    
    # 创建提示信息
    prompt_infos = [
        {
            "name": "math",
            "description": "适合回答数学相关的问题，包括算术、代数、几何、微积分等",
            "prompt_template": math_template
        },
        {
            "name": "programming",
            "description": "适合回答编程相关的问题，包括代码编写、调试、算法等",
            "prompt_template": programming_template
        },
        {
            "name": "general",
            "description": "适合回答一般性问题",
            "prompt_template": general_template
        }
    ]
    
    try:
        # 创建目标链
        destination_chains = {}
        for p_info in prompt_infos:
            name = p_info["name"]
            prompt_template = p_info["prompt_template"]
            prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
            chain = LLMChain(llm=model, prompt=prompt)
            destination_chains[name] = chain
        
        # 创建默认链
        default_prompt = PromptTemplate(template=general_template, input_variables=["input"])
        default_chain = LLMChain(llm=model, prompt=default_prompt)
        
        # 创建路由提示
        destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
        destinations_str = "\n".join(destinations)
        
        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
            destinations=destinations_str
        )
        
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"],
            output_parser=RouterOutputParser()
        )
        
        # 创建路由链
        router_chain = LLMRouterChain.from_llm(model, router_prompt)
        
        # 创建多提示链
        chain = MultiPromptChain(
            router_chain=router_chain,
            destination_chains=destination_chains,
            default_chain=default_chain,
            verbose=True
        )
        
        # 测试不同类型的问题
        test_questions = [
            "计算 2x + 3 = 7 中 x 的值",
            "如何在 Python 中实现快速排序算法？",
            "今天天气怎么样？"
        ]
        
        for question in test_questions:
            print(f"\n问题：{question}")
            try:
                result = chain.run(question)
                print(f"回答：{result}")
            except Exception as e:
                print(f"处理问题时出错：{e}")
                # 使用默认链作为降级方案
                result = default_chain.run(input=question)
                print(f"默认回答：{result}")
            print("-" * 30)
            
    except Exception as e:
        print(f"路由链演示出错：{e}")
        print("\n使用简化版路由演示：")
        
        # 简化版路由演示
        def simple_router(question: str) -> str:
            """简单的路由逻辑"""
            question_lower = question.lower()
            if any(word in question_lower for word in ['计算', '数学', '方程', '求解']):
                return 'math'
            elif any(word in question_lower for word in ['代码', '编程', '算法', 'python']):
                return 'programming'
            else:
                return 'general'
        
        # 测试简化路由
        for question in test_questions:
            route = simple_router(question)
            print(f"问题：{question} -> 路由到：{route}")

# ============================================================================
# 7. 错误处理和最佳实践演示
# ============================================================================

def demo_error_handling():
    """
    演示链中的错误处理和最佳实践
    """
    print("\n" + "="*50)
    print("7. 错误处理和最佳实践演示")
    print("="*50)
    
    # 创建一个可能失败的链
    risky_prompt = PromptTemplate(
        input_variables=["task"],
        template="执行以下任务：{task}\n\n如果任务不合理，请说明原因。"
    )
    
    risky_chain = LLMChain(
        llm=model,
        prompt=risky_prompt
    )
    
    # 测试不同的输入
    test_tasks = [
        "写一首关于春天的诗",  # 正常任务
        "",  # 空输入
        "a" * 10000,  # 过长输入
        "执行危险操作",  # 不当请求
    ]
    
    for i, task in enumerate(test_tasks, 1):
        print(f"\n测试 {i}: {task[:50]}{'...' if len(task) > 50 else ''}")
        
        try:
            # 输入验证
            if not task.strip():
                print("错误：输入为空")
                continue
                
            if len(task) > 1000:
                print("错误：输入过长，已截断")
                task = task[:1000] + "..."
            
            # 执行链
            result = risky_chain.run(task=task)
            print(f"结果：{result[:200]}{'...' if len(result) > 200 else ''}")
            
        except Exception as e:
            print(f"执行出错：{e}")
            
            # 提供降级方案
            try:
                simple_response = f"无法处理请求：{task[:50]}。请提供更清晰的指令。"
                print(f"降级响应：{simple_response}")
            except Exception as e2:
                print(f"降级方案也失败：{e2}")

# ============================================================================
# 8. 性能优化演示
# ============================================================================

def demo_performance_optimization():
    """
    演示链的性能优化技巧
    """
    print("\n" + "="*50)
    print("8. 性能优化演示")
    print("="*50)
    
    import time
    
    # 创建一个简单的链用于性能测试
    test_prompt = PromptTemplate(
        input_variables=["number"],
        template="请计算 {number} 的平方"
    )
    
    test_chain = LLMChain(
        llm=model,
        prompt=test_prompt
    )
    
    # 测试批量处理 vs 单个处理
    numbers = [str(i) for i in range(1, 6)]
    
    print("\n单个处理：")
    start_time = time.time()
    individual_results = []
    
    for number in numbers:
        try:
            result = test_chain.run(number=number)
            individual_results.append(result)
            print(f"{number} -> {result.strip()}")
        except Exception as e:
            print(f"处理 {number} 时出错：{e}")
    
    individual_time = time.time() - start_time
    print(f"单个处理总时间：{individual_time:.2f}秒")
    
    # 批量处理（如果支持）
    print("\n批量处理建议：")
    print("1. 使用缓存避免重复计算")
    print("2. 合并相似的请求")
    print("3. 使用异步处理提高并发")
    print("4. 设置合理的超时时间")
    print("5. 实现请求去重")

# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数：运行所有演示
    """
    print("LangChain 链式调用演示")
    print("=" * 60)
    
    # 检查 API 配置
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or "your-deepseek-api-key" in api_key:
        print("\n⚠️  请先配置你的 DeepSeek API Key！")
        print("请在 .env 文件中设置 DEEPSEEK_API_KEY")
        return
    
    print(f"\n✅ 使用 API 端点：{os.environ.get('OPENAI_API_BASE', 'default')}")
    print("开始运行链式调用演示...\n")
    
    try:
        # 运行所有演示
        demo_basic_llm_chain()
        demo_output_parser_chain()
        demo_simple_sequential_chain()
        demo_sequential_chain()
        demo_transform_chain()
        demo_router_chain()
        demo_error_handling()
        demo_performance_optimization()
        
        print("\n" + "="*60)
        print("所有演示完成！")
        print("\n总结：")
        print("1. LLMChain：最基础的链，适合简单任务")
        print("2. SequentialChain：适合多步骤处理")
        print("3. RouterChain：适合条件分支处理")
        print("4. TransformChain：适合数据转换")
        print("5. 错误处理：确保链的稳定性")
        print("6. 性能优化：提高链的执行效率")
        
    except KeyboardInterrupt:
        print("\n\n演示被用户中断")
    except Exception as e:
        print(f"\n演示过程中出现错误：{e}")
        print("请检查网络连接和 API 配置")

if __name__ == "__main__":
    main()