#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain 结构化输出示例

本示例演示了 LangChain 中结构化输出的各种实现方式：
1. 字典 Schema 定义
2. Pydantic Schema 定义
3. with_structured_output() 方法
4. Tool Calling 方式
5. JSON Mode 使用
6. 实际应用场景示例

运行前请确保已配置 DEEPSEEK_API_KEY 环境变量
"""

import os
import sys
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 添加项目根目录到路径，以便导入配置模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.config import load_environment


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_result(description: str, result):
    """打印结果"""
    print(f"\n{description}:")
    print(f"类型: {type(result)}")
    print(f"内容: {result}")
    print("-" * 40)


# ============================================================================
# 1. Pydantic Schema 定义示例
# ============================================================================

class PersonInfo(BaseModel):
    """个人信息提取器"""
    name: str = Field(description="人物姓名")
    age: Optional[int] = Field(description="年龄，如果未提及则为None")
    occupation: Optional[str] = Field(description="职业，如果未提及则为None")
    location: Optional[str] = Field(description="居住地，如果未提及则为None")
    skills: List[str] = Field(description="技能列表", default=[])


class NewsAnalysis(BaseModel):
    """新闻分析结果"""
    title: str = Field(description="新闻标题")
    summary: str = Field(description="新闻摘要，不超过100字")
    key_points: List[str] = Field(description="关键要点列表")
    sentiment: str = Field(description="情感倾向：positive/negative/neutral")
    category: str = Field(description="新闻分类")


class ResponseFormatter(BaseModel):
    """通用响应格式化器"""
    answer: str = Field(description="用户问题的答案")
    confidence: float = Field(description="答案的置信度，0-1之间")
    followup_question: str = Field(description="可以继续询问的问题")
    sources: List[str] = Field(description="信息来源", default=[])


# ============================================================================
# 主要演示函数
# ============================================================================

def demo_basic_dict_schema():
    """演示基础字典 Schema"""
    print_section("1. 基础字典 Schema 演示")
    
    try:
        # 加载配置
        config = load_environment()
        
        # 初始化模型
        model = ChatOpenAI(
            model="deepseek-chat",
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
            temperature=0.1
        )
        
        # 定义符合JSON Schema格式的字典Schema
        dict_schema = {
            "title": "QuestionAnswer",
            "description": "回答用户问题的结构化输出",
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "用户问题的答案"
                },
                "confidence": {
                    "type": "number",
                    "description": "答案的置信度(0-1)",
                    "minimum": 0,
                    "maximum": 1
                },
                "category": {
                    "type": "string",
                    "description": "问题的分类"
                }
            },
            "required": ["answer", "confidence", "category"]
        }
        
        print(f"JSON Schema格式: {dict_schema['title']}")
        
        # 由于DeepSeek API可能不支持某些response_format，使用普通提示方式
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        prompt = ChatPromptTemplate.from_template(
            "请回答以下问题，并按照指定的JSON格式返回结果：\n\n"
            "问题: {question}\n\n"
            "请返回包含以下字段的JSON格式数据：\n"
            "- answer: 问题的答案\n"
            "- confidence: 答案的置信度(0-1之间的数字)\n"
            "- category: 问题的分类\n\n"
            "请确保返回有效的JSON格式。"
        )
        
        # 创建链
        chain = prompt | model | StrOutputParser()
        
        # 测试问题
        question = "什么是人工智能？"
        print(f"\n测试问题: {question}")
        
        # 调用模型
        result = chain.invoke({"question": question})
        
        print(f"\n原始结果: {result}")
        
        # 尝试解析JSON
        try:
            import json
            parsed_result = json.loads(result)
            print(f"\n解析后的结果: {parsed_result}")
            print(f"结果类型: {type(parsed_result)}")
        except json.JSONDecodeError:
            print("\n⚠️ 返回结果不是有效的JSON格式，但内容如下:")
            print(result)
        
        # 验证结果结构
        try:
            import json
            if isinstance(result, str):
                parsed_result = json.loads(result)
                if isinstance(parsed_result, dict):
                    print("\n✓ 成功返回字典格式")
                    for key in dict_schema["properties"].keys():
                        if key in parsed_result:
                            print(f"  {key}: {parsed_result[key]}")
                        else:
                            print(f"  ❌ 缺少字段: {key}")
                else:
                    print(f"❌ 解析后的结果不是字典: {type(parsed_result)}")
            else:
                print(f"❌ 返回类型不是字符串: {type(result)}")
        except json.JSONDecodeError:
            print(f"❌ 无法解析为JSON格式: {result[:100]}...")
        
    except Exception as e:
        print(f"字典 Schema 演示出错: {e}")
        
        # 提供替代方案说明
        print("\n💡 解决方案:")
        print("1. 字典Schema需要符合JSON Schema格式，包含title和description")
        print("2. 推荐使用Pydantic BaseModel定义Schema")
        print("3. Tool Calling方式在DeepSeek API中支持更好")


def demo_pydantic_schema():
    """演示使用Pydantic定义Schema"""
    print("\n" + "=" * 60)
    print(" 2. Pydantic Schema 演示")
    print("=" * 60)
    
    try:
        # 加载配置
        config = load_environment()
        
        # 初始化模型
        model = ChatOpenAI(
            model="deepseek-chat",
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
            temperature=0.1
        )
        
        # 尝试使用with_structured_output，如果失败则使用普通提示
        try:
            structured_model = model.with_structured_output(PersonInfo)
            
            # 测试文本
            text = "我叫张三，今年25岁，是一名软件工程师，住在北京市朝阳区。"
            print(f"测试文本: {text}")
            
            # 调用模型
            result = structured_model.invoke(f"请从以下文本中提取人员信息：{text}")
            
            print_result("Pydantic Schema 结果", result)
            
            # 验证结果类型
            if isinstance(result, PersonInfo):
                print("\n✓ 成功返回PersonInfo对象")
                print(f"  姓名: {result.name}")
                print(f"  年龄: {result.age}")
                print(f"  职业: {result.occupation}")
                print(f"  地址: {result.location}")
            else:
                print(f"❌ 返回类型不正确: {type(result)}")
                
        except Exception as struct_error:
            print(f"with_structured_output失败: {struct_error}")
            print("\n尝试使用普通提示方式...")
            
            # 使用普通提示方式
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            
            prompt = ChatPromptTemplate.from_template(
                "请从以下文本中提取人员信息，并返回JSON格式：\n\n"
                "文本: {text}\n\n"
                "请返回包含以下字段的JSON数据：\n"
                "- name: 姓名\n"
                "- age: 年龄(数字)\n"
                "- occupation: 职业\n"
                "- location: 地址\n\n"
                "请确保返回有效的JSON格式。"
            )
            
            chain = prompt | model | StrOutputParser()
            
            text = "我叫张三，今年25岁，是一名软件工程师，住在北京市朝阳区。"
            print(f"测试文本: {text}")
            
            result = chain.invoke({"text": text})
            print(f"\n原始结果: {result}")
            
            # 尝试解析JSON并创建PersonInfo对象
            try:
                import json
                parsed_data = json.loads(result)
                person = PersonInfo(**parsed_data)
                print(f"\n✓ 成功创建PersonInfo对象")
                print(f"  姓名: {person.name}")
                print(f"  年龄: {person.age}")
                print(f"  职业: {person.occupation}")
                print(f"  地址: {person.location}")
            except (json.JSONDecodeError, TypeError, ValueError) as parse_error:
                print(f"\n⚠️ 解析失败: {parse_error}")
                print(f"原始结果: {result}")
            
    except Exception as e:
        print(f"Pydantic Schema 演示出错: {e}")
        print("\n💡 解决方案:")
        print("1. 如果with_structured_output不支持，使用普通提示要求JSON格式")
        print("2. 手动解析JSON并创建Pydantic对象")
        print("3. 确保模型支持结构化输出功能")


def demo_tool_calling():
    """演示 Tool Calling 方式"""
    print_section("3. Tool Calling 方式演示")
    
    try:
        # 加载配置
        config = load_environment()
        
        # 初始化模型
        model = ChatOpenAI(
            model="deepseek-chat",
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
            temperature=0.1
        )
        
        # 将 Pydantic Schema 作为工具绑定到模型
        model_with_tools = model.bind_tools([PersonInfo])
        
        # 测试文本
        text = "张三是一位30岁的软件工程师，住在北京，擅长Python、机器学习和数据分析。"
        print(f"测试文本: {text}")
        
        # 调用模型
        ai_msg = model_with_tools.invoke([
            SystemMessage(content="请从给定文本中提取人物信息，使用PersonInfo工具。"),
            HumanMessage(content=text)
        ])
        
        print_result("AI 消息", ai_msg)
        
        # 检查是否有工具调用
        if ai_msg.tool_calls:
            tool_call = ai_msg.tool_calls[0]
            print_result("工具调用参数", tool_call["args"])
            
            # 解析为 Pydantic 对象
            person_info = PersonInfo.model_validate(tool_call["args"])
            print_result("解析后的 PersonInfo 对象", person_info)
        else:
            print("没有检测到工具调用")
            
    except Exception as e:
        print(f"Tool Calling 演示出错: {e}")


def demo_json_mode():
    """演示 JSON Mode - 根据DeepSeek API要求"""
    print_section("4. JSON Mode 演示")
    
    try:
        # 加载配置
        config = load_environment()
        
        # 初始化模型 - 不使用response_format，因为DeepSeek可能不支持
        model = ChatOpenAI(
            model="deepseek-chat",
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
            temperature=0.1,
            max_tokens=1000  # 设置合理的max_tokens防止JSON被截断
        )
        
        # 构建包含'json'关键词和JSON格式样例的提示
        system_prompt = """你是一个数据分析助手。请严格按照JSON格式返回结果。
        
输出格式示例：
{
    "random_numbers": [12, 45, 78, 23, 56, 89, 34, 67, 91, 15],
    "sum": 510,
    "average": 51.0,
    "analysis": "数据分析结果"
}

请确保返回有效的JSON格式。"""
        
        user_prompt = "请生成10个0-99之间的随机整数，计算它们的总和和平均值，并提供简单的数据分析。请以JSON格式返回结果。"
        
        print(f"系统提示: {system_prompt}")
        print(f"用户请求: {user_prompt}")
        
        # 调用模型
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        result = model.invoke(messages)
        print(f"\n原始结果: {result.content}")
        
        # 处理可能返回空content的情况
        if not result.content or result.content.strip() == "":
            print("⚠️ 返回了空的content，这是DeepSeek API的已知问题")
            print("建议: 尝试修改prompt或重新调用")
            return
        
        # 尝试解析JSON
        try:
            import json
            parsed_result = json.loads(result.content)
            print_result("解析后的JSON结果", parsed_result)
            
            # 验证JSON结构
            if "random_numbers" in parsed_result and "sum" in parsed_result and "average" in parsed_result:
                print("\n✓ JSON结构验证成功")
                print(f"随机数列表: {parsed_result['random_numbers']}")
                print(f"总和: {parsed_result['sum']}")
                print(f"平均值: {parsed_result['average']}")
                if "analysis" in parsed_result:
                    print(f"分析: {parsed_result['analysis']}")
            else:
                print("⚠️ JSON结构不完整")
                
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON解析失败: {e}")
            print("原始内容可能不是有效的JSON格式")
        
    except Exception as e:
        print(f"JSON Mode 演示出错: {e}")
        
        # 提供替代方案说明
        print("\n💡 DeepSeek JSON Output 使用要点:")
        print("1. 设置 response_format 参数为 {'type': 'json_object'}")
        print("2. 在system或user prompt中必须包含'json'关键词")
        print("3. 提供JSON格式样例来指导模型输出")
        print("4. 设置合理的max_tokens参数防止JSON被截断")
        print("5. 处理可能返回空content的情况")
        print("6. 如果遇到问题，可以尝试修改prompt或使用with_structured_output()方法")


def demo_news_analysis():
    """演示新闻分析应用场景"""
    print_section("5. 实际应用场景：新闻分析")
    
    try:
        # 加载配置
        config = load_environment()
        
        # 初始化模型
        model = ChatOpenAI(
            model="deepseek-chat",
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
            temperature=0.1,
            max_tokens=1000
        )
        
        # 模拟新闻文本
        news_text = """
        科技巨头公司今日宣布推出革命性的人工智能芯片，该芯片采用全新的神经网络架构，
        能够将AI计算速度提升10倍，同时降低能耗50%。这一突破性技术将广泛应用于
        自动驾驶、智能手机、数据中心等领域。业界专家认为，这将推动整个AI行业
        进入新的发展阶段，预计将在未来两年内实现商业化应用。
        """
        
        print(f"新闻文本: {news_text.strip()}")
        
        # 首先尝试使用with_structured_output
        try:
            model_with_structure = model.with_structured_output(NewsAnalysis)
            result = model_with_structure.invoke(f"请分析以下新闻内容：\n\n{news_text}")
            print_result("新闻分析结果", result)
            
            # 展示结构化数据的使用
            print("\n结构化数据的使用示例:")
            print(f"标题: {result.title}")
            print(f"摘要: {result.summary}")
            print(f"情感: {result.sentiment}")
            print(f"分类: {result.category}")
            print("关键要点:")
            for i, point in enumerate(result.key_points, 1):
                print(f"  {i}. {point}")
                
        except Exception as struct_error:
            print(f"with_structured_output失败: {struct_error}")
            print("\n尝试使用普通提示方式...")
            
            # 使用普通提示方式
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            
            system_prompt = """你是一个专业的新闻分析师。请分析给定的新闻内容，并以JSON格式返回分析结果。
            
输出格式示例：
{{
    "title": "新闻标题",
    "summary": "新闻摘要",
    "sentiment": "positive/negative/neutral",
    "category": "新闻分类",
    "key_points": ["要点1", "要点2", "要点3"]
}}

请确保返回有效的JSON格式。"""
            
            user_prompt = f"请分析以下新闻内容：\n\n{news_text}"
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", user_prompt)
            ])
            
            chain = prompt | model | StrOutputParser()
            result = chain.invoke({})
            
            print(f"\n原始结果: {result}")
            
            # 尝试解析JSON并创建NewsAnalysis对象
            try:
                import json
                import re
                
                # 处理可能包含markdown代码块的结果
                cleaned_result = result.strip()
                
                # 检测并移除markdown代码块标记
                if cleaned_result.startswith('```json') and cleaned_result.endswith('```'):
                    # 移除开头的```json和结尾的```
                    cleaned_result = cleaned_result[7:-3].strip()
                    print(f"\n检测到markdown代码块，已清理: {cleaned_result[:100]}...")
                elif cleaned_result.startswith('```') and cleaned_result.endswith('```'):
                    # 移除开头和结尾的```
                    cleaned_result = cleaned_result[3:-3].strip()
                    print(f"\n检测到代码块，已清理: {cleaned_result[:100]}...")
                
                # 使用正则表达式进一步清理可能的多余标记
                json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
                match = re.search(json_pattern, result)
                if match:
                    cleaned_result = match.group(1).strip()
                    print(f"\n使用正则表达式提取JSON: {cleaned_result[:100]}...")
                
                parsed_data = json.loads(cleaned_result)
                news_analysis = NewsAnalysis(**parsed_data)
                
                print_result("解析后的新闻分析结果", news_analysis)
                
                # 展示结构化数据的使用
                print("\n结构化数据的使用示例:")
                print(f"标题: {news_analysis.title}")
                print(f"摘要: {news_analysis.summary}")
                print(f"情感: {news_analysis.sentiment}")
                print(f"分类: {news_analysis.category}")
                print("关键要点:")
                for i, point in enumerate(news_analysis.key_points, 1):
                    print(f"  {i}. {point}")
                    
            except (json.JSONDecodeError, TypeError, ValueError) as parse_error:
                print(f"\n⚠️ JSON解析失败: {parse_error}")
                print(f"原始结果: {result}")
                print(f"清理后结果: {cleaned_result if 'cleaned_result' in locals() else 'N/A'}")
                print("\n💡 建议: 调整提示词或重新尝试")
                print("💡 可能的问题: 结果包含markdown代码块标记或格式不正确")
            
    except Exception as e:
        print(f"新闻分析演示出错: {e}")
        print("\n💡 解决方案:")
        print("1. 使用普通提示要求JSON格式输出")
        print("2. 手动解析JSON并创建Pydantic对象")
        print("3. 确保提示中包含明确的格式要求")
        print("4. 处理可能的解析错误")


def demo_comparison():
    """演示不同方法的对比"""
    print_section("6. 不同方法对比")
    
    try:
        # 加载配置
        config = load_environment()
        
        # 初始化模型
        model = ChatOpenAI(
            model="deepseek-chat",
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
            temperature=0.1
        )
        
        question = "Python和Java的主要区别是什么？"
        print(f"测试问题: {question}")
        
        # 方法1: 普通调用（无结构化输出）
        print("\n方法1: 普通调用")
        normal_result = model.invoke(question)
        print(f"结果类型: {type(normal_result)}")
        print(f"内容: {normal_result.content[:200]}...")
        
        # 方法2: 字典 Schema
        print("\n方法2: 字典 Schema")
        dict_model = model.with_structured_output({
            "main_differences": "主要区别列表",
            "python_advantages": "Python的优势",
            "java_advantages": "Java的优势",
            "recommendation": "使用建议"
        })
        dict_result = dict_model.invoke(question)
        print(f"结果类型: {type(dict_result)}")
        print(f"内容: {dict_result}")
        
        # 方法3: Pydantic Schema
        print("\n方法3: Pydantic Schema")
        pydantic_model = model.with_structured_output(ResponseFormatter)
        pydantic_result = pydantic_model.invoke(question)
        print(f"结果类型: {type(pydantic_result)}")
        print(f"答案: {pydantic_result.answer[:200]}...")
        print(f"置信度: {pydantic_result.confidence}")
        
        print("\n对比总结:")
        print("- 普通调用: 返回自然语言，需要手动解析")
        print("- 字典 Schema: 返回字典，结构简单，易于使用")
        print("- Pydantic Schema: 返回类型化对象，支持验证，功能最强")
        
    except Exception as e:
        print(f"对比演示出错: {e}")


def demo_error_handling():
    """演示错误处理"""
    print_section("7. 错误处理演示")
    
    try:
        # 加载配置
        config = load_environment()
        
        # 初始化模型
        model = ChatOpenAI(
            model="deepseek-chat",
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
            temperature=0.1
        )
        
        # 定义一个严格的 Schema
        class StrictSchema(BaseModel):
            number: int = Field(description="一个整数")
            percentage: float = Field(description="一个0-100之间的百分比")
            category: str = Field(description="必须是 'A', 'B', 'C' 中的一个")
        
        model_with_structure = model.with_structured_output(StrictSchema)
        
        # 测试可能导致验证错误的输入
        test_cases = [
            "给我一个数字42，百分比85.5，类别A",  # 正常情况
            "给我一些随机的文本，不包含数字",      # 可能导致解析错误
        ]
        
        for i, test_input in enumerate(test_cases, 1):
            print(f"\n测试用例 {i}: {test_input}")
            try:
                result = model_with_structure.invoke(test_input)
                print(f"成功: {result}")
                print(f"验证通过: number={result.number}, percentage={result.percentage}, category={result.category}")
            except Exception as e:
                print(f"错误: {e}")
                print("建议: 检查输入格式或调整 Schema 定义")
                
    except Exception as e:
        print(f"错误处理演示出错: {e}")


def main():
    """主函数：运行所有结构化输出示例"""
    print("=" * 60)
    print("LangChain 结构化输出示例")
    print("=" * 60)
    
    # 加载配置
    config = load_environment()
    if not config.deepseek_api_key:
        print("❌ 无法加载DeepSeek配置，请检查环境变量")
        return
    
    # 初始化模型
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=config.deepseek_api_key,
        base_url=config.deepseek_base_url,
        temperature=0
    )
    
    # 运行所有演示
    demos = [
        demo_basic_dict_schema,
        demo_pydantic_schema,
        demo_tool_calling,
        demo_json_mode,
        demo_news_analysis,
        demo_comparison,
        demo_error_handling
    ]
    
    for demo in demos:
        try:
            demo()
        except KeyboardInterrupt:
            print("\n用户中断演示")
            break
        except Exception as e:
            print(f"\n演示 {demo.__name__} 出错: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("\n总结:")
    print("1. 结构化输出让AI返回可预测的数据格式")
    print("2. Pydantic Schema 提供类型安全和数据验证")
    print("3. with_structured_output() 是推荐的使用方式")
    print("4. 不同方法适用于不同的应用场景")
    print("5. 错误处理对于生产环境很重要")


if __name__ == "__main__":
    main()