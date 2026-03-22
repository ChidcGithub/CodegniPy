# Codegnipy - AI 原生的 Python 语言扩展
"""
Codegnipy 让非确定性的 AI 能力成为 Python 的一等公民。

核心特性:
- `~"prompt"` 操作符：将自然语言提示直接嵌入代码
- `@cognitive` 装饰器：让函数由 LLM 实现
- 记忆存储：会话级别的记忆管理
- 反思循环：LLM 自我检查与修正
- 异步调度：高性能并发调用
- 确定性保证：类型约束、幻觉检测
- 流式响应：实时输出支持
- 工具调用：Function Calling 支持
- 多提供商：OpenAI、Anthropic 等
- 混合执行模型：确定性逻辑与模糊意图的无缝协同
"""

__version__ = "0.2.0"

from .runtime import cognitive_call, CognitiveContext
from .decorator import cognitive
from .memory import (
    MemoryStore,
    InMemoryStore,
    FileStore,
    Message,
    MessageRole,
    ContextCompressor
)
from .reflection import (
    Reflector,
    ReflectionResult,
    ReflectionStatus,
    with_reflection,
    ReflectiveCognitiveCall
)
from .scheduler import (
    CognitiveScheduler,
    ScheduledTask,
    TaskStatus,
    Priority,
    SchedulerConfig,
    RetryPolicy,
    async_cognitive_call,
    batch_call,
    run_async
)
from .determinism import (
    TypeConstraint,
    PrimitiveConstraint,
    EnumConstraint,
    SchemaConstraint,
    ListConstraint,
    ValidationStatus,
    ValidationResult,
    SimulationMode,
    Simulator,
    HallucinationDetector,
    HallucinationCheck,
    deterministic_call
)
from .streaming import (
    StreamStatus,
    StreamChunk,
    StreamResult,
    stream_call,
    stream_call_async,
    stream_iter,
    stream_iter_async
)
from .tools import (
    ToolType,
    ToolParameter,
    ToolDefinition,
    ToolCall,
    ToolResult,
    ToolRegistry,
    tool,
    call_with_tools,
    register_tool,
    get_global_registry
)
from .providers import (
    ProviderType,
    ProviderConfig,
    BaseProvider,
    OpenAIProvider,
    AnthropicProvider,
    ProviderFactory,
    create_provider
)

__all__ = [
    # Core
    "cognitive_call",
    "CognitiveContext",
    "cognitive",
    # Memory
    "MemoryStore",
    "InMemoryStore",
    "FileStore",
    "Message",
    "MessageRole",
    "ContextCompressor",
    # Reflection
    "Reflector",
    "ReflectionResult",
    "ReflectionStatus",
    "with_reflection",
    "ReflectiveCognitiveCall",
    # Scheduler
    "CognitiveScheduler",
    "ScheduledTask",
    "TaskStatus",
    "Priority",
    "SchedulerConfig",
    "RetryPolicy",
    "async_cognitive_call",
    "batch_call",
    "run_async",
    # Determinism
    "TypeConstraint",
    "PrimitiveConstraint",
    "EnumConstraint",
    "SchemaConstraint",
    "ListConstraint",
    "ValidationStatus",
    "ValidationResult",
    "SimulationMode",
    "Simulator",
    "HallucinationDetector",
    "HallucinationCheck",
    "deterministic_call",
    # Streaming
    "StreamStatus",
    "StreamChunk",
    "StreamResult",
    "stream_call",
    "stream_call_async",
    "stream_iter",
    "stream_iter_async",
    # Tools
    "ToolType",
    "ToolParameter",
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    "ToolRegistry",
    "tool",
    "call_with_tools",
    "register_tool",
    "get_global_registry",
    # Providers
    "ProviderType",
    "ProviderConfig",
    "BaseProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "ProviderFactory",
    "create_provider",
]
