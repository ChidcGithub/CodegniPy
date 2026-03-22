"""
Codegnipy 记忆存储模块

提供会话记忆的存储、检索和管理功能。
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum


class MessageRole(Enum):
    """消息角色"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    REFLECTION = "reflection"  # 反思消息


@dataclass
class Message:
    """单条消息"""
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {})
        )
    
    def to_openai_format(self) -> dict:
        """转换为 OpenAI API 格式"""
        return {
            "role": self.role.value if self.role != MessageRole.REFLECTION else "system",
            "content": self.content
        }


class MemoryStore(ABC):
    """记忆存储抽象基类"""
    
    @abstractmethod
    def add(self, message: Message) -> str:
        """添加消息，返回消息 ID"""
        pass
    
    @abstractmethod
    def get(self, message_id: str) -> Optional[Message]:
        """获取单条消息"""
        pass
    
    @abstractmethod
    def get_all(self) -> List[Message]:
        """获取所有消息"""
        pass
    
    @abstractmethod
    def get_recent(self, n: int) -> List[Message]:
        """获取最近 n 条消息"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空记忆"""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """获取消息数量"""
        pass
    
    def add_user_message(self, content: str, **metadata) -> str:
        """添加用户消息"""
        return self.add(Message(MessageRole.USER, content, metadata=metadata))
    
    def add_assistant_message(self, content: str, **metadata) -> str:
        """添加助手消息"""
        return self.add(Message(MessageRole.ASSISTANT, content, metadata=metadata))
    
    def add_reflection(self, content: str, **metadata) -> str:
        """添加反思消息"""
        return self.add(Message(MessageRole.REFLECTION, content, metadata=metadata))
    
    def to_openai_messages(self, include_reflections: bool = True) -> List[dict]:
        """转换为 OpenAI API 消息格式"""
        messages = []
        for msg in self.get_all():
            if msg.role == MessageRole.REFLECTION and not include_reflections:
                continue
            messages.append(msg.to_openai_format())
        return messages


class InMemoryStore(MemoryStore):
    """内存存储实现"""
    
    def __init__(self):
        self._messages: List[Message] = []
        self._counter = 0
    
    def add(self, message: Message) -> str:
        self._counter += 1
        message.metadata["_id"] = str(self._counter)
        self._messages.append(message)
        return message.metadata["_id"]
    
    def get(self, message_id: str) -> Optional[Message]:
        for msg in self._messages:
            if msg.metadata.get("_id") == message_id:
                return msg
        return None
    
    def get_all(self) -> List[Message]:
        return self._messages.copy()
    
    def get_recent(self, n: int) -> List[Message]:
        return self._messages[-n:] if n > 0 else []
    
    def clear(self) -> None:
        self._messages.clear()
    
    def count(self) -> int:
        return len(self._messages)


class FileStore(MemoryStore):
    """文件持久化存储"""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self._messages: List[Message] = []
        self._counter = 0
        self._load()
    
    def _load(self) -> None:
        """从文件加载记忆"""
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._messages = [Message.from_dict(m) for m in data.get("messages", [])]
                    self._counter = data.get("counter", 0)
            except (json.JSONDecodeError, KeyError):
                self._messages = []
                self._counter = 0
    
    def _save(self) -> None:
        """保存记忆到文件"""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump({
                "messages": [m.to_dict() for m in self._messages],
                "counter": self._counter
            }, f, ensure_ascii=False, indent=2)
    
    def add(self, message: Message) -> str:
        self._counter += 1
        message.metadata["_id"] = str(self._counter)
        self._messages.append(message)
        self._save()
        return message.metadata["_id"]
    
    def get(self, message_id: str) -> Optional[Message]:
        for msg in self._messages:
            if msg.metadata.get("_id") == message_id:
                return msg
        return None
    
    def get_all(self) -> List[Message]:
        return self._messages.copy()
    
    def get_recent(self, n: int) -> List[Message]:
        return self._messages[-n:] if n > 0 else []
    
    def clear(self) -> None:
        self._messages.clear()
        self._counter = 0
        self._save()
    
    def count(self) -> int:
        return len(self._messages)


class ContextCompressor:
    """上下文压缩器"""
    
    def __init__(self, max_tokens: int = 4000, compression_ratio: float = 0.5):
        """
        参数:
            max_tokens: 最大 token 数（近似）
            compression_ratio: 压缩比例
        """
        self.max_tokens = max_tokens
        self.compression_ratio = compression_ratio
    
    def estimate_tokens(self, text: str) -> int:
        """估算文本 token 数（简单估算：4 字符 ≈ 1 token）"""
        return len(text) // 4
    
    def needs_compression(self, messages: List[Message]) -> bool:
        """检查是否需要压缩"""
        total = sum(self.estimate_tokens(m.content) for m in messages)
        return total > self.max_tokens
    
    def compress(self, messages: List[Message], summarizer=None) -> List[Message]:
        """
        压缩消息历史
        
        参数:
            messages: 消息列表
            summarizer: 可选的摘要函数，接收消息列表返回摘要字符串
        
        返回:
            压缩后的消息列表
        """
        if not self.needs_compression(messages):
            return messages
        
        # 保留最近的消息
        keep_recent = int(len(messages) * (1 - self.compression_ratio))
        recent_messages = messages[-keep_recent:]
        old_messages = messages[:-keep_recent]
        
        if not old_messages:
            return recent_messages
        
        # 生成摘要
        if summarizer:
            summary = summarizer(old_messages)
        else:
            # 简单摘要：提取关键信息
            summary = self._simple_summarize(old_messages)
        
        # 创建摘要消息
        summary_msg = Message(
            role=MessageRole.SYSTEM,
            content=f"[历史摘要] {summary}",
            metadata={"compressed": True, "original_count": len(old_messages)}
        )
        
        return [summary_msg] + recent_messages
    
    def _simple_summarize(self, messages: List[Message]) -> str:
        """简单摘要：提取用户和助手的主要交互"""
        interactions = []
        current_turn = []
        
        for msg in messages:
            if msg.role in (MessageRole.USER, MessageRole.ASSISTANT):
                current_turn.append(f"{msg.role.value}: {msg.content[:100]}...")
                if len(current_turn) == 2:
                    interactions.append(" | ".join(current_turn))
                    current_turn = []
        
        if current_turn:
            interactions.append(" | ".join(current_turn))
        
        return f"之前的 {len(messages)} 条消息已压缩。主要交互: {'; '.join(interactions[:5])}"
