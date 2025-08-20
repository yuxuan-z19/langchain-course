#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain第08小节：Redis持久化记忆演示

本脚本演示如何使用Redis作为记忆存储后端，实现持久化的聊天机器人记忆功能。
包含以下主要功能：
1. Redis连接和配置
2. 基础记忆操作（CRUD）
3. 持久化聊天机器人
4. 跨会话记忆保持
5. 多用户记忆隔离
6. 记忆过期和清理机制
7. 从InMemoryHistory迁移

作者：Jaguarliu
日期：2025年
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# 添加项目根目录到Python路径
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    import redis
    from redis.exceptions import ConnectionError, TimeoutError
except ImportError:
    print("错误：请安装redis库")
    print("运行：pip install redis")
    sys.exit(1)

try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain.memory import ConversationBufferMemory
    from langchain_community.chat_message_histories import ChatMessageHistory
except ImportError:
    print("错误：请安装langchain相关库")
    print("运行：pip install langchain langchain-openai langchain-community")
    sys.exit(1)

# 导入配置模块
try:
    from utils.config import load_environment
except ImportError:
    print("错误：无法导入配置模块")
    print("请确保utils/config.py文件存在")
    sys.exit(1)


def print_section(title: str):
    """打印章节标题"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_subsection(title: str):
    """打印子章节标题"""
    print(f"\n--- {title} ---")


class RedisMemoryStore:
    """Redis记忆存储类"""
    
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None,
                 decode_responses: bool = True,
                 max_connections: int = 10):
        """初始化Redis连接"""
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        
        # 创建连接池
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=decode_responses,
            max_connections=max_connections,
            retry_on_timeout=True,
            socket_keepalive=True,
            socket_keepalive_options={}
        )
        
        # 创建Redis客户端
        self.redis_client = redis.Redis(connection_pool=self.pool)
        
        # 测试连接
        self._test_connection()
    
    def _test_connection(self):
        """测试Redis连接"""
        try:
            self.redis_client.ping()
            print(f"✅ Redis连接成功: {self.host}:{self.port}")
        except Exception as e:
            print(f"❌ Redis连接失败: {e}")
            print("请确保Redis服务正在运行")
            print("可以使用以下命令启动Redis:")
            print("docker-compose up -d")
            raise
    
    def _get_user_key(self, user_id: str) -> str:
        """获取用户记忆键"""
        return f"langchain:memory:user:{user_id}"
    
    def _get_session_key(self, session_id: str) -> str:
        """获取会话键"""
        return f"langchain:session:{session_id}"
    
    def _serialize_message(self, message: BaseMessage) -> str:
        """序列化消息"""
        return json.dumps({
            'content': message.content,
            'type': message.__class__.__name__,
            'timestamp': time.time()
        })
    
    def _deserialize_message(self, data: str) -> BaseMessage:
        """反序列化消息"""
        msg_data = json.loads(data)
        content = msg_data['content']
        msg_type = msg_data['type']
        
        if msg_type == 'HumanMessage':
            return HumanMessage(content=content)
        elif msg_type == 'AIMessage':
            return AIMessage(content=content)
        elif msg_type == 'SystemMessage':
            return SystemMessage(content=content)
        else:
            return HumanMessage(content=content)  # 默认为人类消息
    
    def save_message(self, user_id: str, message: BaseMessage, ttl: int = 86400):
        """保存消息到Redis"""
        try:
            key = self._get_user_key(user_id)
            serialized = self._serialize_message(message)
            
            # 使用列表存储消息历史
            self.redis_client.lpush(key, serialized)
            
            # 设置过期时间（秒）
            self.redis_client.expire(key, ttl)
            
            print(f"💾 消息已保存到Redis: {user_id}")
        except Exception as e:
            print(f"❌ 保存消息失败: {e}")
            raise
    
    def get_messages(self, user_id: str, limit: int = 50) -> List[BaseMessage]:
        """从Redis获取消息历史"""
        try:
            key = self._get_user_key(user_id)
            
            # 获取最近的消息（倒序）
            messages_data = self.redis_client.lrange(key, 0, limit - 1)
            
            # 反序列化并恢复正确顺序
            messages = []
            for data in reversed(messages_data):
                try:
                    message = self._deserialize_message(data)
                    messages.append(message)
                except Exception as e:
                    print(f"⚠️ 跳过无效消息: {e}")
                    continue
            
            print(f"📖 从Redis加载了 {len(messages)} 条消息: {user_id}")
            return messages
        except Exception as e:
            print(f"❌ 获取消息失败: {e}")
            return []
    
    def clear_user_memory(self, user_id: str):
        """清除用户记忆"""
        try:
            key = self._get_user_key(user_id)
            deleted = self.redis_client.delete(key)
            if deleted:
                print(f"🗑️ 已清除用户记忆: {user_id}")
            else:
                print(f"ℹ️ 用户记忆不存在: {user_id}")
        except Exception as e:
            print(f"❌ 清除记忆失败: {e}")
            raise
    
    def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """获取记忆统计信息"""
        try:
            key = self._get_user_key(user_id)
            
            # 获取消息数量
            message_count = self.redis_client.llen(key)
            
            # 获取过期时间
            ttl = self.redis_client.ttl(key)
            
            # 获取内存使用
            memory_usage = self.redis_client.memory_usage(key) if message_count > 0 else 0
            
            return {
                'user_id': user_id,
                'message_count': message_count,
                'ttl_seconds': ttl,
                'memory_bytes': memory_usage,
                'exists': message_count > 0
            }
        except Exception as e:
            print(f"❌ 获取统计信息失败: {e}")
            return {}
    
    def cleanup_expired_memories(self):
        """清理过期的记忆数据"""
        try:
            # 获取所有记忆键
            pattern = "langchain:memory:user:*"
            keys = self.redis_client.keys(pattern)
            
            cleaned_count = 0
            for key in keys:
                ttl = self.redis_client.ttl(key)
                if ttl == -2:  # 键已过期
                    self.redis_client.delete(key)
                    cleaned_count += 1
            
            print(f"🧹 清理了 {cleaned_count} 个过期记忆")
            return cleaned_count
        except Exception as e:
            print(f"❌ 清理过期记忆失败: {e}")
            return 0
    
    def list_all_users(self) -> List[str]:
        """列出所有有记忆的用户"""
        try:
            pattern = "langchain:memory:user:*"
            keys = self.redis_client.keys(pattern)
            
            # 提取用户ID
            users = []
            for key in keys:
                user_id = key.split(':')[-1]
                users.append(user_id)
            
            return users
        except Exception as e:
            print(f"❌ 获取用户列表失败: {e}")
            return []
    
    def close(self):
        """关闭Redis连接"""
        try:
            self.redis_client.close()
            print("🔌 Redis连接已关闭")
        except Exception as e:
            print(f"❌ 关闭连接失败: {e}")


class RedisChatMemory:
    """基于Redis的聊天记忆类"""
    
    def __init__(self, redis_store: RedisMemoryStore, user_id: str, max_messages: int = 20):
        self.redis_store = redis_store
        self.user_id = user_id
        self.max_messages = max_messages
        
        # 加载历史消息
        self.messages = self.redis_store.get_messages(user_id, max_messages)
    
    def add_message(self, message: BaseMessage):
        """添加消息"""
        # 添加到本地缓存
        self.messages.append(message)
        
        # 保存到Redis
        self.redis_store.save_message(self.user_id, message)
        
        # 限制消息数量
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_messages(self) -> List[BaseMessage]:
        """获取所有消息"""
        return self.messages
    
    def clear(self):
        """清除记忆"""
        self.messages = []
        self.redis_store.clear_user_memory(self.user_id)
    
    def get_context_string(self) -> str:
        """获取上下文字符串"""
        if not self.messages:
            return "这是我们的第一次对话。"
        
        context_parts = []
        for msg in self.messages[-10:]:  # 只使用最近10条消息作为上下文
            if isinstance(msg, HumanMessage):
                context_parts.append(f"用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                context_parts.append(f"助手: {msg.content}")
        
        return "\n".join(context_parts)


def create_llm() -> ChatOpenAI:
    """创建LLM实例"""
    try:
        config = load_environment()
        
        llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
            temperature=0.7,
            max_tokens=1000
        )
        
        print("✅ DeepSeek LLM初始化成功")
        return llm
    except Exception as e:
        print(f"❌ LLM初始化失败: {e}")
        raise


def demo_redis_connection():
    """演示Redis连接"""
    print_section("1. Redis连接演示")
    
    try:
        # 创建Redis存储
        redis_store = RedisMemoryStore(
            host='localhost',
            port=6379,
            db=0
        )
        
        # 测试基础操作
        print_subsection("基础Redis操作")
        test_key = "test:connection"
        test_value = "Hello Redis!"
        
        redis_store.redis_client.set(test_key, test_value, ex=60)
        retrieved_value = redis_store.redis_client.get(test_key)
        
        print(f"设置值: {test_value}")
        print(f"获取值: {retrieved_value}")
        print(f"连接测试: {'✅ 成功' if test_value == retrieved_value else '❌ 失败'}")
        
        # 清理测试数据
        redis_store.redis_client.delete(test_key)
        
        return redis_store
        
    except Exception as e:
        print(f"❌ Redis连接演示失败: {e}")
        print("\n请确保Redis服务正在运行:")
        print("1. 使用Docker: docker-compose up -d")
        print("2. 本地安装: redis-server")
        raise


def demo_basic_memory_operations(redis_store: RedisMemoryStore):
    """演示基础记忆操作"""
    print_section("2. 基础记忆操作演示")
    
    user_id = "demo_user_001"
    
    print_subsection("保存消息")
    
    # 创建测试消息
    messages = [
        HumanMessage(content="你好，我是张三"),
        AIMessage(content="你好张三！很高兴认识你。"),
        HumanMessage(content="我喜欢编程，特别是Python"),
        AIMessage(content="太好了！Python是一门很棒的编程语言。你在学习哪些方面？"),
        HumanMessage(content="我在学习机器学习和AI")
    ]
    
    # 保存消息
    for i, message in enumerate(messages, 1):
        redis_store.save_message(user_id, message)
        print(f"  {i}. 保存: {message.content[:30]}...")
        time.sleep(0.1)  # 确保时间戳不同
    
    print_subsection("检索消息")
    
    # 检索消息
    retrieved_messages = redis_store.get_messages(user_id)
    print(f"检索到 {len(retrieved_messages)} 条消息:")
    
    for i, message in enumerate(retrieved_messages, 1):
        msg_type = "👤" if isinstance(message, HumanMessage) else "🤖"
        print(f"  {i}. {msg_type} {message.content}")
    
    print_subsection("记忆统计")
    
    # 获取统计信息
    stats = redis_store.get_memory_stats(user_id)
    print(f"用户ID: {stats.get('user_id')}")
    print(f"消息数量: {stats.get('message_count')}")
    print(f"过期时间: {stats.get('ttl_seconds')} 秒")
    print(f"内存使用: {stats.get('memory_bytes')} 字节")
    print(f"记忆存在: {stats.get('exists')}")
    
    return user_id


def demo_persistent_chatbot(redis_store: RedisMemoryStore, llm: ChatOpenAI):
    """演示持久化聊天机器人"""
    print_section("3. 持久化聊天机器人演示")
    
    user_id = "chatbot_user_001"
    
    # 创建Redis聊天记忆
    chat_memory = RedisChatMemory(redis_store, user_id, max_messages=10)
    
    print_subsection("模拟对话会话1")
    
    # 第一轮对话
    conversations_1 = [
        "你好，我叫李明，是一名软件工程师",
        "我正在学习LangChain，你能帮我吗？",
        "我想了解如何实现记忆功能"
    ]
    
    for user_input in conversations_1:
        print(f"\n👤 用户: {user_input}")
        
        # 添加用户消息到记忆
        human_message = HumanMessage(content=user_input)
        chat_memory.add_message(human_message)
        
        # 构建提示，包含历史上下文
        context = chat_memory.get_context_string()
        prompt = f"""你是一个友好的AI助手。请根据对话历史回答用户的问题。

对话历史:
{context}

当前用户输入: {user_input}

请回答:"""
        
        try:
            # 调用LLM
            response = llm.invoke([HumanMessage(content=prompt)])
            ai_response = response.content
            
            print(f"🤖 助手: {ai_response}")
            
            # 添加AI回复到记忆
            ai_message = AIMessage(content=ai_response)
            chat_memory.add_message(ai_message)
            
        except Exception as e:
            print(f"❌ LLM调用失败: {e}")
            ai_response = "抱歉，我现在无法回答。"
            chat_memory.add_message(AIMessage(content=ai_response))
    
    print_subsection("模拟会话结束和重启")
    print("💤 模拟程序重启...")
    time.sleep(1)
    
    # 模拟新会话 - 重新创建聊天记忆（从Redis加载）
    print("🔄 重新启动聊天机器人...")
    new_chat_memory = RedisChatMemory(redis_store, user_id, max_messages=10)
    
    print_subsection("模拟对话会话2（跨会话记忆）")
    
    # 第二轮对话
    conversations_2 = [
        "你还记得我的名字吗？",
        "我们之前聊到了什么？"
    ]
    
    for user_input in conversations_2:
        print(f"\n👤 用户: {user_input}")
        
        # 添加用户消息到记忆
        human_message = HumanMessage(content=user_input)
        new_chat_memory.add_message(human_message)
        
        # 构建提示，包含历史上下文
        context = new_chat_memory.get_context_string()
        prompt = f"""你是一个友好的AI助手。请根据对话历史回答用户的问题。

对话历史:
{context}

当前用户输入: {user_input}

请回答:"""
        
        try:
            # 调用LLM
            response = llm.invoke([HumanMessage(content=prompt)])
            ai_response = response.content
            
            print(f"🤖 助手: {ai_response}")
            
            # 添加AI回复到记忆
            ai_message = AIMessage(content=ai_response)
            new_chat_memory.add_message(ai_message)
            
        except Exception as e:
            print(f"❌ LLM调用失败: {e}")
            ai_response = "抱歉，我现在无法回答。"
            new_chat_memory.add_message(AIMessage(content=ai_response))
    
    print_subsection("最终记忆状态")
    final_messages = new_chat_memory.get_messages()
    print(f"总共保存了 {len(final_messages)} 条消息")
    
    return user_id


def demo_multi_user_isolation(redis_store: RedisMemoryStore):
    """演示多用户记忆隔离"""
    print_section("4. 多用户记忆隔离演示")
    
    # 创建多个用户
    users = {
        "alice": "我是Alice，我喜欢音乐",
        "bob": "我是Bob，我是程序员",
        "charlie": "我是Charlie，我在学习AI"
    }
    
    print_subsection("为不同用户保存记忆")
    
    # 为每个用户保存不同的记忆
    for user_id, intro in users.items():
        print(f"\n👤 {user_id}: {intro}")
        
        # 保存用户介绍
        human_msg = HumanMessage(content=intro)
        redis_store.save_message(user_id, human_msg)
        
        # 保存AI回复
        ai_response = f"很高兴认识你，{user_id.title()}！"
        ai_msg = AIMessage(content=ai_response)
        redis_store.save_message(user_id, ai_msg)
        
        print(f"🤖 助手: {ai_response}")
    
    print_subsection("验证用户记忆隔离")
    
    # 验证每个用户的记忆是独立的
    for user_id in users.keys():
        messages = redis_store.get_messages(user_id)
        stats = redis_store.get_memory_stats(user_id)
        
        print(f"\n📊 {user_id} 的记忆:")
        print(f"  消息数量: {stats.get('message_count')}")
        print(f"  最新消息: {messages[-1].content if messages else '无'}")
    
    print_subsection("用户列表")
    all_users = redis_store.list_all_users()
    print(f"系统中的所有用户: {all_users}")
    
    return list(users.keys())


def demo_memory_management(redis_store: RedisMemoryStore, user_ids: List[str]):
    """演示记忆管理功能"""
    print_section("5. 记忆管理演示")
    
    print_subsection("记忆统计概览")
    
    # 显示所有用户的记忆统计
    total_messages = 0
    total_memory = 0
    
    for user_id in user_ids:
        stats = redis_store.get_memory_stats(user_id)
        if stats.get('exists'):
            print(f"👤 {user_id}:")
            print(f"  消息数: {stats.get('message_count', 0)}")
            print(f"  内存: {stats.get('memory_bytes', 0)} 字节")
            print(f"  TTL: {stats.get('ttl_seconds', 0)} 秒")
            
            total_messages += stats.get('message_count', 0)
            total_memory += stats.get('memory_bytes', 0)
    
    print(f"\n📊 总计:")
    print(f"  总消息数: {total_messages}")
    print(f"  总内存使用: {total_memory} 字节")
    
    print_subsection("记忆清理演示")
    
    # 清理特定用户的记忆
    if user_ids:
        test_user = user_ids[0]
        print(f"清理用户 {test_user} 的记忆...")
        redis_store.clear_user_memory(test_user)
        
        # 验证清理结果
        stats_after = redis_store.get_memory_stats(test_user)
        print(f"清理后状态: 存在={stats_after.get('exists')}")
    
    print_subsection("过期记忆清理")
    
    # 清理过期记忆
    cleaned_count = redis_store.cleanup_expired_memories()
    print(f"清理了 {cleaned_count} 个过期记忆")


def demo_migration_from_inmemory():
    """演示从InMemoryHistory迁移到Redis"""
    print_section("6. 从InMemoryHistory迁移演示")
    
    print_subsection("创建InMemoryHistory数据")
    
    # 创建传统的内存历史
    from langchain.memory.chat_message_histories import ChatMessageHistory
    
    inmemory_history = ChatMessageHistory()
    
    # 添加一些测试数据
    test_messages = [
        HumanMessage(content="这是内存中的第一条消息"),
        AIMessage(content="我收到了你的消息"),
        HumanMessage(content="我们需要迁移到Redis"),
        AIMessage(content="好的，让我们开始迁移过程")
    ]
    
    for msg in test_messages:
        inmemory_history.add_message(msg)
    
    print(f"InMemoryHistory中有 {len(inmemory_history.messages)} 条消息")
    
    print_subsection("迁移到Redis")
    
    # 创建Redis存储
    redis_store = RedisMemoryStore()
    migration_user_id = "migrated_user_001"
    
    # 迁移消息
    migrated_count = 0
    for message in inmemory_history.messages:
        try:
            redis_store.save_message(migration_user_id, message, ttl=7200)  # 2小时过期
            migrated_count += 1
        except Exception as e:
            print(f"❌ 迁移消息失败: {e}")
    
    print(f"✅ 成功迁移 {migrated_count} 条消息到Redis")
    
    print_subsection("验证迁移结果")
    
    # 验证迁移结果
    redis_messages = redis_store.get_messages(migration_user_id)
    print(f"Redis中有 {len(redis_messages)} 条消息")
    
    # 比较消息内容
    print("\n消息对比:")
    for i, (orig, migrated) in enumerate(zip(inmemory_history.messages, redis_messages)):
        match = "✅" if orig.content == migrated.content else "❌"
        print(f"  {i+1}. {match} 原始: {orig.content[:30]}...")
        print(f"      迁移: {migrated.content[:30]}...")
    
    # 清理迁移数据
    redis_store.clear_user_memory(migration_user_id)
    redis_store.close()
    
    return migrated_count


def main():
    """主函数"""
    print("🚀 LangChain第08小节：Redis持久化记忆演示")
    print("本演示将展示如何使用Redis实现持久化的聊天记忆功能")
    
    try:
        # 1. Redis连接演示
        redis_store = demo_redis_connection()
        
        # 2. 基础记忆操作
        demo_user = demo_basic_memory_operations(redis_store)
        
        # 3. 创建LLM
        llm = create_llm()
        
        # 4. 持久化聊天机器人
        chatbot_user = demo_persistent_chatbot(redis_store, llm)
        
        # 5. 多用户隔离
        multi_users = demo_multi_user_isolation(redis_store)
        
        # 6. 记忆管理
        all_users = [demo_user, chatbot_user] + multi_users
        demo_memory_management(redis_store, all_users)
        
        # 7. 迁移演示
        demo_migration_from_inmemory()
        
        print_section("演示完成")
        print("✅ 所有Redis持久化记忆功能演示完成！")
        print("\n🎯 关键要点:")
        print("1. Redis提供了真正的持久化记忆存储")
        print("2. 支持跨会话的对话历史保持")
        print("3. 实现了多用户记忆隔离")
        print("4. 提供了完整的记忆管理功能")
        print("5. 支持从InMemoryHistory的平滑迁移")
        
        print("\n📚 下一步学习建议:")
        print("- 探索Redis集群和高可用配置")
        print("- 学习更高级的记忆管理策略")
        print("- 集成其他存储后端（如PostgreSQL）")
        print("- 实现记忆数据的备份和恢复")
        
        # 关闭Redis连接
        redis_store.close()
        
    except KeyboardInterrupt:
        print("\n\n⏹️ 演示被用户中断")
    except Exception as e:
        print(f"\n\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 感谢使用LangChain Redis记忆演示！")


if __name__ == "__main__":
    main()