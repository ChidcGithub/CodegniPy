"""
Codegnipy 流式响应测试模块
"""

import pytest

from codegnipy.streaming import (
    StreamStatus,
    StreamChunk,
    StreamResult
)


class TestStreamChunk:
    """StreamChunk 测试"""
    
    def test_chunk_creation(self):
        """测试块创建"""
        chunk = StreamChunk(
            content="Hello",
            status=StreamStatus.STREAMING,
            accumulated="Hello"
        )
        assert chunk.content == "Hello"
        assert chunk.status == StreamStatus.STREAMING
        assert chunk.accumulated == "Hello"
    
    def test_chunk_str(self):
        """测试块字符串表示"""
        chunk = StreamChunk(content="World", status=StreamStatus.STREAMING)
        assert str(chunk) == "World"


class TestStreamResult:
    """StreamResult 测试"""
    
    def test_result_creation(self):
        """测试结果创建"""
        result = StreamResult(
            content="Full response",
            status=StreamStatus.COMPLETED,
            chunks=[
                StreamChunk("Full ", StreamStatus.STREAMING),
                StreamChunk("response", StreamStatus.STREAMING),
            ]
        )
        assert result.content == "Full response"
        assert result.status == StreamStatus.COMPLETED
        assert len(result.chunks) == 2


class TestStreamStatus:
    """StreamStatus 测试"""
    
    def test_status_values(self):
        """测试状态值"""
        assert StreamStatus.STARTED.value == "started"
        assert StreamStatus.STREAMING.value == "streaming"
        assert StreamStatus.COMPLETED.value == "completed"
        assert StreamStatus.ERROR.value == "error"


class TestStreamingFunctions:
    """流式函数测试"""
    
    def test_stream_call_signature(self):
        """测试 stream_call 函数签名"""
        from codegnipy.streaming import stream_call
        import inspect
        
        sig = inspect.signature(stream_call)
        params = list(sig.parameters.keys())
        
        assert "prompt" in params
        assert "context" in params
        assert "on_chunk" in params
    
    def test_stream_iter_signature(self):
        """测试 stream_iter 函数签名"""
        from codegnipy.streaming import stream_iter
        import inspect
        
        sig = inspect.signature(stream_iter)
        params = list(sig.parameters.keys())
        
        assert "prompt" in params
        assert "context" in params
