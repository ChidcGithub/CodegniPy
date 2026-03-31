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

__version__ = "0.0.3"

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
    OllamaProvider,
    TransformersProvider,
    LlamaCppProvider,
    QuantizationConfig,
    ProviderFactory,
    create_provider
)
from .validation import (
    ExternalValidationStatus,
    Evidence,
    ExternalValidationResult,
    BaseValidator,
    WebSearchValidator,
    KnowledgeGraphValidator,
    FactCheckValidator,
    CompositeValidator,
    create_default_validator,
    verify_claim,
    verify_claim_async
)
from .observability import (
    LogLevel,
    MetricType,
    SpanContext,
    Metric,
    CognitiveLogger,
    MetricsCollector,
    Tracer,
    OpenTelemetryExporter,
    ObservabilityManager,
    traced,
    logged,
    metered,
    get_default_manager,
    configure_observability,
)
from .distributed import (
    QueueBackendType,
    TaskPriority,
    TaskState,
    LoadBalanceStrategy,
    DistributedTask,
    WorkerInfo,
    QueueBackend,
    InMemoryQueueBackend,
    RedisQueueBackend,
    RabbitMQQueueBackend,
    LoadBalancer,
    DistributedScheduler,
    create_queue_backend,
    submit_distributed_task,
)
from .cache import (
    CacheBackendType,
    CachePolicy,
    CacheEntry,
    CacheStats,
    CacheBackend,
    LRUCacheBackend,
    RedisCacheBackend,
    ResponseCache,
    EmbeddingCache,
    SemanticCache,
    CacheInvalidator,
    CostOptimizer,
    cached,
    create_cache_backend,
    create_response_cache,
)
from .security import (
    PIIType,
    FilterAction,
    AuditEventType,
    SeverityLevel,
    PIIMatch,
    FilterResult,
    AuditEvent,
    PIIPatterns,
    PIIDetector,
    DataMasker,
    ContentFilter,
    PIIFilter,
    KeywordFilter,
    CompositeFilter,
    AuditLogger,
    RateLimiter,
    SecurityManager,
    secure,
    create_default_security_manager,
    detect_pii,
    mask_pii,
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
    "OllamaProvider",
    "TransformersProvider",
    "LlamaCppProvider",
    "QuantizationConfig",
    "ProviderFactory",
    "create_provider",
    # Validation
    "ExternalValidationStatus",
    "Evidence",
    "ExternalValidationResult",
    "BaseValidator",
    "WebSearchValidator",
    "KnowledgeGraphValidator",
    "FactCheckValidator",
    "CompositeValidator",
    "create_default_validator",
    "verify_claim",
    "verify_claim_async",
    # Observability
    "LogLevel",
    "MetricType",
    "SpanContext",
    "Metric",
    "CognitiveLogger",
    "MetricsCollector",
    "Tracer",
    "OpenTelemetryExporter",
    "ObservabilityManager",
    "traced",
    "logged",
    "metered",
    "get_default_manager",
    "configure_observability",
    # Distributed (Phase 7)
    "QueueBackendType",
    "TaskPriority",
    "TaskState",
    "LoadBalanceStrategy",
    "DistributedTask",
    "WorkerInfo",
    "QueueBackend",
    "InMemoryQueueBackend",
    "RedisQueueBackend",
    "RabbitMQQueueBackend",
    "LoadBalancer",
    "DistributedScheduler",
    "create_queue_backend",
    "submit_distributed_task",
    # Cache (Phase 7)
    "CacheBackendType",
    "CachePolicy",
    "CacheEntry",
    "CacheStats",
    "CacheBackend",
    "LRUCacheBackend",
    "RedisCacheBackend",
    "ResponseCache",
    "EmbeddingCache",
    "SemanticCache",
    "CacheInvalidator",
    "CostOptimizer",
    "cached",
    "create_cache_backend",
    "create_response_cache",
    # Security (Phase 7)
    "PIIType",
    "FilterAction",
    "AuditEventType",
    "SeverityLevel",
    "PIIMatch",
    "FilterResult",
    "AuditEvent",
    "PIIPatterns",
    "PIIDetector",
    "DataMasker",
    "ContentFilter",
    "PIIFilter",
    "KeywordFilter",
    "CompositeFilter",
    "AuditLogger",
    "RateLimiter",
    "SecurityManager",
    "secure",
    "create_default_security_manager",
    "detect_pii",
    "mask_pii",
]
