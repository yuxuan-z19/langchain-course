#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain 提示词模板演示

本脚本演示了 LangChain 中各种提示词模板的使用方法，包括：
1. PromptTemplate - 基础提示词模板
2. ChatPromptTemplate - 聊天模板
3. FewShotPromptTemplate - 少样本学习模板
4. PipelinePromptTemplate - 管道模板
5. Jinja2 模板集成
6. 从文件加载模板
7. 与 DeepSeek API 的集成
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from jinja2 import Environment, FileSystemLoader, Template
    from langchain.prompts import (
        ChatPromptTemplate,
        FewShotPromptTemplate,
        PipelinePromptTemplate,
        PromptTemplate,
    )
    from langchain.prompts.example_selector import LengthBasedExampleSelector
    from langchain_openai import ChatOpenAI
    from utils.config import load_environment
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所需依赖: pip install langchain langchain-openai jinja2")
    sys.exit(1)

# 配置
TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_llm():
    """创建 LLM 实例"""
    try:
        # 加载配置
        config = load_environment()

        llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
            max_tokens=1000,
            temperature=0.7,
        )
        return llm
    except Exception as e:
        print(f"创建 LLM 失败: {e}")
        print("请确保已正确配置 .env 文件中的 DEEPSEEK_API_KEY")
        return None


def demo_basic_prompt_template():
    """演示基础 PromptTemplate"""
    print("\n" + "=" * 50)
    print("1. 基础 PromptTemplate 演示")
    print("=" * 50)

    # 创建基础模板
    template = PromptTemplate(
        input_variables=["product", "target_audience", "tone"],
        template="""请为 {product} 写一个面向 {target_audience} 的产品描述。
        
要求：
- 语调：{tone}
- 长度：100-200字
- 突出产品优势
- 包含行动号召

产品描述：""",
    )

    # 格式化提示词
    prompt = template.format(
        product="智能手表", target_audience="年轻专业人士", tone="专业且友好"
    )

    print("生成的提示词:")
    print(prompt)

    # 如果有 LLM，调用 API
    llm = create_llm()
    if llm:
        try:
            response = llm.invoke(prompt)
            print("\nLLM 响应:")
            print(response.content)
        except Exception as e:
            print(f"\nLLM 调用失败: {e}")

    return template


def demo_chat_prompt_template():
    """演示 ChatPromptTemplate"""
    print("\n" + "=" * 50)
    print("2. ChatPromptTemplate 演示")
    print("=" * 50)

    # 创建聊天模板
    chat_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个专业的 {role}，具有 {experience} 年的经验。你的回答应该准确、专业且易于理解。",
            ),
            ("human", "我有一个关于 {topic} 的问题：{question}"),
            ("assistant", "我理解您的问题。让我为您提供专业的解答。"),
            ("human", "{follow_up}"),
        ]
    )

    # 格式化消息
    messages = chat_template.format_messages(
        role="数据科学家",
        experience="5",
        topic="机器学习",
        question="什么是过拟合，如何避免？",
        follow_up="能否提供一些具体的防止过拟合的技术？",
    )

    print("生成的聊天消息:")
    for i, message in enumerate(messages):
        print(f"{i+1}. {message.type}: {message.content}")

    # 如果有 LLM，调用 API
    llm = create_llm()
    if llm:
        try:
            response = llm.invoke(messages)
            print("\nLLM 响应:")
            print(response.content)
        except Exception as e:
            print(f"\nLLM 调用失败: {e}")

    return chat_template


def demo_few_shot_prompt_template():
    """演示 FewShotPromptTemplate"""
    print("\n" + "=" * 50)
    print("3. FewShotPromptTemplate 演示")
    print("=" * 50)

    # 加载示例数据
    examples_file = TEMPLATES_DIR / "few_shot_examples.json"
    try:
        with open(examples_file, "r", encoding="utf-8") as f:
            all_examples = json.load(f)
        examples = all_examples["sentiment_analysis"]
    except Exception as e:
        print(f"加载示例文件失败: {e}")
        # 使用默认示例
        examples = [
            {"input": "这个产品真的很棒！", "output": "正面"},
            {"input": "质量太差了。", "output": "负面"},
            {"input": "还可以吧。", "output": "中性"},
        ]

    # 创建示例模板
    example_prompt = PromptTemplate(
        input_variables=["input", "output"], template="输入: {input}\n输出: {output}"
    )

    # 创建 Few-shot 模板
    few_shot_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="请根据以下示例，分析文本的情感倾向：",
        suffix="输入: {text}\n输出:",
        input_variables=["text"],
    )

    # 格式化提示词
    prompt = few_shot_template.format(text="这次购物体验超出了我的预期！")

    print("生成的 Few-shot 提示词:")
    print(prompt)

    # 如果有 LLM，调用 API
    llm = create_llm()
    if llm:
        try:
            response = llm.invoke(prompt)
            print("\nLLM 响应:")
            print(response.content)
        except Exception as e:
            print(f"\nLLM 调用失败: {e}")

    return few_shot_template


def demo_pipeline_prompt_template():
    """演示 PipelinePromptTemplate"""
    print("\n" + "=" * 50)
    print("4. PipelinePromptTemplate 演示")
    print("=" * 50)

    # 创建子模板
    introduction_template = PromptTemplate(
        input_variables=["name", "role"], template="我是 {name}，一名 {role}。"
    )

    task_template = PromptTemplate(
        input_variables=["task", "requirements"],
        template="我需要完成以下任务：{task}\n\n具体要求：\n{requirements}",
    )

    conclusion_template = PromptTemplate(
        input_variables=["deadline"], template="请在 {deadline} 之前完成，谢谢！"
    )

    # 创建管道模板
    pipeline_template = PipelinePromptTemplate(
        final_prompt=PromptTemplate(
            input_variables=["introduction", "task_description", "conclusion"],
            template="{introduction}\n\n{task_description}\n\n{conclusion}",
        ),
        pipeline_prompts=[
            ("introduction", introduction_template),
            ("task_description", task_template),
            ("conclusion", conclusion_template),
        ],
    )

    # 格式化提示词
    prompt = pipeline_template.format(
        name="张三",
        role="项目经理",
        task="制定下季度的营销计划",
        requirements="1. 分析市场趋势\n2. 制定推广策略\n3. 预算规划",
        deadline="下周五",
    )

    print("生成的管道提示词:")
    print(prompt)

    return pipeline_template


def demo_jinja2_templates():
    """演示 Jinja2 模板集成"""
    print("\n" + "=" * 50)
    print("5. Jinja2 模板集成演示")
    print("=" * 50)

    # 设置 Jinja2 环境
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

    # 演示 1: 分析报告模板
    print("\n5.1 复杂分析报告模板")
    print("-" * 30)

    try:
        template = env.get_template("analysis_report.jinja2")

        # 准备数据
        data = {
            "report_title": "第三季度销售数据分析",
            "author": "数据分析团队",
            "analysis_type": "performance",
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "include_summary": True,
            "data_points": [
                {"name": "产品A销量", "value": 150, "type": "销售", "trend": "up"},
                {"name": "产品B销量", "value": 80, "type": "销售", "trend": "down"},
                {
                    "name": "客户满意度",
                    "value": 95,
                    "type": "质量",
                    "description": "客户反馈评分",
                },
                {"name": "退货率", "value": 3, "type": "质量", "trend": "stable"},
            ],
        }

        report = template.render(**data)
        print(report)

    except Exception as e:
        print(f"渲染分析报告模板失败: {e}")

    # 演示 2: 多语言模板
    print("\n5.2 多语言模板")
    print("-" * 30)

    try:
        template = env.get_template("multilingual_template.jinja2")

        languages = ["zh", "en", "ja"]
        for lang in languages:
            print(f"\n{lang.upper()} 版本:")
            result = template.render(
                language=lang,
                task_type="文本翻译" if lang == "zh" else "Text Translation",
                content="Hello, how are you today?",
                style="formal",
            )
            print(result)

    except Exception as e:
        print(f"渲染多语言模板失败: {e}")


def demo_file_loading():
    """演示从文件加载模板"""
    print("\n" + "=" * 50)
    print("6. 从文件加载模板演示")
    print("=" * 50)

    # 加载简单文本模板
    simple_template_file = TEMPLATES_DIR / "simple_prompt.txt"
    try:
        with open(simple_template_file, "r", encoding="utf-8") as f:
            template_content = f.read()

        template = PromptTemplate.from_template(template_content)

        prompt = template.format(
            role="技术顾问",
            task_description="为客户设计一个微服务架构",
            requirements="1. 高可用性\n2. 可扩展性\n3. 安全性\n4. 成本效益",
        )

        print("从文件加载的简单模板:")
        print(prompt)

    except Exception as e:
        print(f"加载简单模板失败: {e}")

    # 加载聊天模板
    chat_template_file = TEMPLATES_DIR / "chat_template.txt"
    try:
        with open(chat_template_file, "r", encoding="utf-8") as f:
            chat_content = f.read()

        # 解析聊天模板格式
        lines = chat_content.strip().split("\n")
        messages = []

        for line in lines:
            if ":" in line:
                role, content = line.split(":", 1)
                messages.append((role.strip(), content.strip()))

        chat_template = ChatPromptTemplate.from_messages(messages)

        formatted_messages = chat_template.format_messages(
            expert_type="软件架构师",
            domain="云计算",
            user_question="如何设计一个高可用的微服务系统？",
            topic="微服务架构",
            follow_up_question="在选择服务网格时应该考虑哪些因素？",
        )

        print("\n从文件加载的聊天模板:")
        for i, msg in enumerate(formatted_messages):
            print(f"{i+1}. {msg.type}: {msg.content}")

    except Exception as e:
        print(f"加载聊天模板失败: {e}")


def demo_template_management_best_practices():
    """演示模板管理最佳实践"""
    print("\n" + "=" * 50)
    print("7. 模板管理最佳实践")
    print("=" * 50)

    print("\n7.1 模板缓存")
    print("-" * 20)

    # 模板缓存示例
    class TemplateManager:
        def __init__(self, templates_dir: Path):
            self.templates_dir = templates_dir
            self._cache = {}
            self.jinja_env = Environment(loader=FileSystemLoader(templates_dir))

        def get_prompt_template(self, filename: str) -> PromptTemplate:
            """获取并缓存 PromptTemplate"""
            if filename not in self._cache:
                try:
                    with open(
                        self.templates_dir / filename, "r", encoding="utf-8"
                    ) as f:
                        content = f.read()
                    self._cache[filename] = PromptTemplate.from_template(content)
                except Exception as e:
                    print(f"加载模板 {filename} 失败: {e}")
                    return None
            return self._cache[filename]

        def get_jinja_template(self, filename: str) -> Template:
            """获取并缓存 Jinja2 模板"""
            cache_key = f"jinja_{filename}"
            if cache_key not in self._cache:
                try:
                    self._cache[cache_key] = self.jinja_env.get_template(filename)
                except Exception as e:
                    print(f"加载 Jinja2 模板 {filename} 失败: {e}")
                    return None
            return self._cache[cache_key]

        def clear_cache(self):
            """清空缓存"""
            self._cache.clear()

        def get_cache_info(self):
            """获取缓存信息"""
            return {
                "cached_templates": len(self._cache),
                "template_names": list(self._cache.keys()),
            }

    # 使用模板管理器
    manager = TemplateManager(TEMPLATES_DIR)

    # 加载模板
    simple_template = manager.get_prompt_template("simple_prompt.txt")
    jinja_template = manager.get_jinja_template("analysis_report.jinja2")

    print(f"缓存信息: {manager.get_cache_info()}")

    print("\n7.2 错误处理和验证")
    print("-" * 20)

    def validate_template_variables(
        template: PromptTemplate, required_vars: List[str]
    ) -> bool:
        """验证模板变量"""
        template_vars = set(template.input_variables)
        required_vars_set = set(required_vars)

        missing_vars = required_vars_set - template_vars
        extra_vars = template_vars - required_vars_set

        if missing_vars:
            print(f"缺少必需变量: {missing_vars}")
            return False

        if extra_vars:
            print(f"额外变量: {extra_vars}")

        return True

    # 验证示例
    if simple_template:
        is_valid = validate_template_variables(
            simple_template, ["role", "task_description", "requirements"]
        )
        print(f"模板验证结果: {'通过' if is_valid else '失败'}")

    print("\n7.3 性能优化建议")
    print("-" * 20)
    print("1. 使用模板缓存避免重复加载")
    print("2. 对于大型模板，考虑分块处理")
    print("3. 预编译常用的 Jinja2 模板")
    print("4. 使用异步加载处理大量模板")
    print("5. 定期清理不再使用的模板缓存")


def main():
    """主函数"""
    print("LangChain 提示词模板演示")
    print("=" * 60)

    # 检查模板目录
    if not TEMPLATES_DIR.exists():
        print(f"错误: 模板目录不存在 {TEMPLATES_DIR}")
        return

    try:
        # 运行所有演示
        demo_basic_prompt_template()
        demo_chat_prompt_template()
        demo_few_shot_prompt_template()
        demo_pipeline_prompt_template()
        demo_jinja2_templates()
        demo_file_loading()
        demo_template_management_best_practices()

        print("\n" + "=" * 60)
        print("所有演示完成！")
        print("\n建议下一步学习:")
        print("1. 输出解析器 - 解析和验证模型输出")
        print("2. 记忆管理 - 在对话中保持上下文")
        print("3. 代理系统 - 构建智能代理")
        print("4. 工具集成 - 集成外部工具和API")

    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"\n演示过程中发生错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
