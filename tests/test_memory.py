"""
Codegnipy Phase 2 测试模块

测试记忆存储和反思功能。
"""

import pytest
import tempfile
import os
from pathlib import Path

from codegnipy.memory import (
    MemoryStore,
    InMemoryStore,
    FileStore,
    Message,
    MessageRole,
    ContextCompressor
)
from codegnipy.reflection import (
    Reflector,
    ReflectionStatus,
    ReflectionResult
)
from codegnipy import CognitiveContext


class TestMessage:
    """Message 类测试"""
    
    def test_message_creation(self):
        """测试消息创建"""
        msg = Message(MessageRole.USER, "Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.timestamp > 0
    
    def test_message_to_dict(self):
        """测试消息序列化"""
        msg = Message(MessageRole.ASSISTANT, "Hi there")
        data = msg.to_dict()
        assert data["role"] == "assistant"
        assert data["content"] == "Hi there"
    
    def test_message_from_dict(self):
        """测试消息反序列化"""
        data = {"role": "user", "content": "Test", "timestamp": 123.0}
        msg = Message.from_dict(data)
        assert msg.role == MessageRole.USER
        assert msg.content == "Test"
        assert msg.timestamp == 123.0
    
    def test_message_to_openai_format(self):
        """测试 OpenAI 格式转换"""
        msg = Message(MessageRole.USER, "Hello")
        openai_msg = msg.to_openai_format()
        assert openai_msg == {"role": "user", "content": "Hello"}
        
        # 反思消息应该转换为 system
        reflection = Message(MessageRole.REFLECTION, "Check this")
        openai_reflection = reflection.to_openai_format()
        assert openai_reflection["role"] == "system"


class TestInMemoryStore:
    """InMemoryStore 测试"""
    
    def test_add_and_get(self):
        """测试添加和获取消息"""
        store = InMemoryStore()
        msg_id = store.add_user_message("Hello")
        assert msg_id is not None
        
        msg = store.get(msg_id)
        assert msg is not None
        assert msg.content == "Hello"
    
    def test_get_all(self):
        """测试获取所有消息"""
        store = InMemoryStore()
        store.add_user_message("A")
        store.add_assistant_message("B")
        
        messages = store.get_all()
        assert len(messages) == 2
    
    def test_get_recent(self):
        """测试获取最近消息"""
        store = InMemoryStore()
        for i in range(5):
            store.add_user_message(f"Msg {i}")
        
        recent = store.get_recent(3)
        assert len(recent) == 3
        assert recent[0].content == "Msg 2"
    
    def test_clear(self):
        """测试清空"""
        store = InMemoryStore()
        store.add_user_message("Test")
        store.clear()
        assert store.count() == 0
    
    def test_to_openai_messages(self):
        """测试转换为 OpenAI 格式"""
        store = InMemoryStore()
        store.add_user_message("Q")
        store.add_assistant_message("A")
        
        messages = store.to_openai_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"


class TestFileStore:
    """FileStore 测试"""
    
    def test_persistence(self):
        """测试持久化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "memory.json")
            
            # 写入
            store = FileStore(filepath)
            store.add_user_message("Test message")
            assert store.count() == 1
            
            # 重新加载
            store2 = FileStore(filepath)
            assert store2.count() == 1
            assert store2.get_all()[0].content == "Test message"
    
    def test_clear_persists(self):
        """测试清空持久化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "memory.json")
            
            store = FileStore(filepath)
            store.add_user_message("Test")
            store.clear()
            
            store2 = FileStore(filepath)
            assert store2.count() == 0


class TestContextCompressor:
    """ContextCompressor 测试"""
    
    def test_estimate_tokens(self):
        """测试 token 估算"""
        compressor = ContextCompressor()
        # 每 4 字符约 1 token
        assert compressor.estimate_tokens("abcd") == 1
        assert compressor.estimate_tokens("abcdefgh") == 2
    
    def test_needs_compression(self):
        """测试压缩判断"""
        compressor = ContextCompressor(max_tokens=10)
        
        # 少量消息不需要压缩
        short_messages = [Message(MessageRole.USER, "short")]
        assert not compressor.needs_compression(short_messages)
        
        # 大量消息需要压缩
        long_messages = [Message(MessageRole.USER, "x" * 100) for _ in range(10)]
        assert compressor.needs_compression(long_messages)
    
    def test_compress(self):
        """测试压缩功能"""
        compressor = ContextCompressor(max_tokens=20, compression_ratio=0.5)
        
        messages = [
            Message(MessageRole.USER, f"Message {i} " + "x" * 50)
            for i in range(10)
        ]
        
        compressed = compressor.compress(messages)
        
        # 应该包含摘要消息
        assert any(m.metadata.get("compressed") for m in compressed)
        # 应该比原始消息少
        assert len(compressed) < len(messages)


class TestReflector:
    """Reflector 测试"""
    
    def test_reflection_result_creation(self):
        """测试反思结果创建"""
        result = ReflectionResult(
            status=ReflectionStatus.PASSED,
            original_response="Test response",
            iterations=1
        )
        assert result.status == ReflectionStatus.PASSED
        assert result.corrected_response is None
    
    def test_reflector_initialization(self):
        """测试反思器初始化"""
        reflector = Reflector(max_iterations=5)
        assert reflector.max_iterations == 5
    
    def test_extract_issues(self):
        """测试问题提取"""
        reflector = Reflector()
        
        critique = """
        1. 回答不够完整
        2. 有语法错误
        - 缺少示例
        """
        issues = reflector._extract_issues(critique)
        
        assert len(issues) == 3
        assert "回答不够完整" in issues[0]


class TestCognitiveContextIntegration:
    """CognitiveContext 集成测试"""
    
    def test_context_with_memory_store(self):
        """测试上下文与记忆存储集成"""
        with CognitiveContext() as ctx:
            assert ctx.memory_store is not None
            assert ctx.memory_store.count() == 0
    
    def test_context_memory_operations(self):
        """测试上下文记忆操作"""
        with CognitiveContext() as ctx:
            ctx.memory_store.add_user_message("Hello")
            ctx.memory_store.add_assistant_message("Hi")
            
            assert ctx.memory_store.count() == 2
            memory = ctx.get_memory()
            assert len(memory) == 2
