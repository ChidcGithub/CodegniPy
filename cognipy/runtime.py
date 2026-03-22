"""
CogniPy 运行时核心模块

提供 cognitive_call 函数和 CognitiveContext 上下文管理器。
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Callable, Union, TYPE_CHECKING
import os

if TYPE_CHECKING:
    from .memory import MemoryStore, InMemoryStore, FileStore


@dataclass
class LLMConfig:
    """LLM 配置"""
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024


@dataclass
class CognitiveContext:
    """
    认知上下文管理器
    
    管理 LLM 配置、会话记忆和调用追踪。
    
    示例:
        with CognitiveContext(api_key="sk-...") as ctx:
            result = cognitive_call("翻译：你好")
    """
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    
    # 记忆存储（可以是 InMemoryStore 或 FileStore）
    memory_store: Optional["MemoryStore"] = field(default=None, repr=False)
    
    # 内部状态
    _is_active: bool = field(default=False, repr=False, init=False)
    
    # 全局上下文引用
    _current: Optional['CognitiveContext'] = field(default=None, repr=False, init=False)
    
    def __post_init__(self):
        """初始化记忆存储"""
        if self.memory_store is None:
            from .memory import InMemoryStore
            self.memory_store = InMemoryStore()
    
    def __enter__(self) -> 'CognitiveContext':
        self._is_active = True
        CognitiveContext._current = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._is_active = False
        CognitiveContext._current = None
        return False
    
    @classmethod
    def get_current(cls) -> Optional['CognitiveContext']:
        """获取当前活动上下文"""
        return cls._current
    
    def get_config(self) -> LLMConfig:
        """获取 LLM 配置"""
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        return LLMConfig(
            api_key=api_key,
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def add_to_memory(self, role: str, content: str):
        """添加到会话记忆"""
        from .memory import MessageRole
        role_enum = MessageRole(role) if isinstance(role, str) else role
        self.memory_store.add(
            self.memory_store.add.__self__.__class__.__module__  # 获取 store 类型
        )
        # 直接操作内存存储
        if hasattr(self.memory_store, '_messages'):
            from .memory import Message
            self.memory_store._messages.append(Message(role_enum, content))
    
    def get_memory(self) -> list:
        """获取会话记忆（OpenAI 格式）"""
        return self.memory_store.to_openai_messages()
    
    def get_memory_store(self) -> "MemoryStore":
        """获取记忆存储对象"""
        return self.memory_store
    
    def clear_memory(self) -> None:
        """清空会话记忆"""
        self.memory_store.clear()


def _call_openai(config: LLMConfig, prompt: str, memory: list = None) -> str:
    """调用 OpenAI API"""
    try:
        import openai
    except ImportError:
        raise ImportError(
            "需要安装 openai 包。运行: pip install openai"
        )
    
    client = openai.OpenAI(
        api_key=config.api_key,
        base_url=config.base_url
    )
    
    messages = []
    if memory:
        messages.extend(memory)
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens
    )
    
    return response.choices[0].message.content


def cognitive_call(
    prompt: str,
    context: Optional[CognitiveContext] = None,
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None
) -> str:
    """
    执行认知调用（调用 LLM）
    
    这是 `~"prompt"` 语法的运行时实现。
    
    参数:
        prompt: 发送给 LLM 的提示
        context: 认知上下文（可选，默认使用当前活动上下文）
        model: 覆盖上下文中的模型设置
        temperature: 覆盖上下文中的温度设置
    
    返回:
        LLM 的响应文本
    
    示例:
        # 直接调用
        result = cognitive_call("将这句话翻译成英文：你好世界")
        
        # 使用上下文
        with CognitiveContext(api_key="sk-...", model="gpt-4"):
            result = cognitive_call("解释量子纠缠")
    """
    # 获取上下文
    ctx = context or CognitiveContext.get_current()
    
    if ctx is None:
        # 无上下文时，创建临时配置
        config = LLMConfig()
    else:
        config = ctx.get_config()
        # 应用覆盖参数
        if model:
            config.model = model
        if temperature is not None:
            config.temperature = temperature
    
    # 检查 API 密钥
    if not config.api_key:
        raise ValueError(
            "未配置 API 密钥。请设置 OPENAI_API_KEY 环境变量，"
            "或在 CognitiveContext 中提供 api_key 参数。"
        )
    
    # 获取记忆
    memory = ctx.get_memory() if ctx else []
    
    # 调用 LLM
    response = _call_openai(config, prompt, memory)
    
    # 更新记忆
    if ctx:
        ctx.add_to_memory("user", prompt)
        ctx.add_to_memory("assistant", response)
    
    return response
