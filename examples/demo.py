"""
Codegnipy 功能演示

运行方式:
    python -m codegnipy run examples/demo.py

注意: 
    - 需要设置 OPENAI_API_KEY 环境变量
    - 或使用 -k 参数: python -m codegnipy run examples/demo.py -k sk-xxx
"""

import asyncio
from pydantic import BaseModel
from codegnipy import (
    # Core
    cognitive_call,
    CognitiveContext,
    cognitive,
    # Memory
    InMemoryStore,
    Message,
    MessageRole,
    ContextCompressor,
    # Reflection
    Reflector,
    with_reflection,
    # Scheduler
    CognitiveScheduler,
    SchedulerConfig,
    Priority,
    async_cognitive_call,
    batch_call,
    # Determinism
    PrimitiveConstraint,
    EnumConstraint,
    SchemaConstraint,
    Simulator,
    SimulationMode,
    HallucinationDetector,
    deterministic_call,
    # Streaming
    stream_iter,
    # Tools
    tool,
    ToolRegistry,
    call_with_tools,
    get_global_registry,
    # Providers
    create_provider,
    ProviderType,
    # Validation
    WebSearchValidator,
    verify_claim,
)


def demo_basic():
    """基础功能演示"""
    print("\n" + "=" * 50)
    print("1. 基础功能 - cognitive_call 和 ~ 操作符")
    print("=" * 50)

    # 方式 1: 使用 cognitive_call (需要 API key)
    try:
        result = cognitive_call("用一句话解释什么是递归")
        print(f"cognitive_call 结果: {result}")
    except ValueError as e:
        print("cognitive_call 需要配置 API key")
        print(f"  错误: {e}")

    # 方式 2: 使用 ~ 操作符 (需要通过 codegnipy run 执行)
    print("\n~ 操作符示例 (需要通过 codegnipy run 执行):")
    print('  result = ~"用一句话解释什么是递归"')
    print('  print(f"结果: {result}")')


def demo_decorator():
    """装饰器功能演示"""
    print("\n" + "=" * 50)
    print("2. @cognitive 装饰器 - 让函数由 LLM 实现")
    print("=" * 50)

    @cognitive
    def summarize(text: str) -> str:
        """总结这段文字的核心观点，不超过两句话。"""
        pass

    @cognitive
    def translate(text: str, target_lang: str = "英文") -> str:
        """将文本翻译成目标语言。"""
        pass

    # 使用模拟模式演示 (实际使用需要 API key)
    print("定义了两个 cognitive 函数:")
    print(f"  - summarize(text: str) -> str")
    print(f"  - translate(text: str, target_lang: str) -> str")


def demo_memory():
    """记忆存储演示"""
    print("\n" + "=" * 50)
    print("3. 记忆存储 - 会话上下文管理")
    print("=" * 50)

    # 创建内存存储
    store = InMemoryStore()

    # 添加消息
    store.add(Message(role=MessageRole.SYSTEM, content="你是一个有帮助的助手"))
    store.add(Message(role=MessageRole.USER, content="你好！"))
    store.add(Message(role=MessageRole.ASSISTANT, content="你好！有什么可以帮助你的？"))

    # 获取最近的对话
    recent = store.get_recent(3)
    print(f"存储的消息数量: {len(store.get_all())}")
    print("最近的消息:")
    for msg in recent:
        print(f"  [{msg.role.value}]: {msg.content}")

    # 上下文压缩器
    compressor = ContextCompressor(max_tokens=100)
    messages = store.get_all()
    print(f"\n是否需要压缩: {compressor.needs_compression(messages)}")


def demo_reflection():
    """反思循环演示"""
    print("\n" + "=" * 50)
    print("4. 反思循环 - LLM 自我检查与修正")
    print("=" * 50)

    # 创建反思器
    reflector = Reflector()

    # 反思示例
    print("反思器配置:")
    print(f"  - 最大迭代次数: {reflector.max_iterations}")

    # 使用 with_reflection 函数
    print("\nwith_reflection 使用示例:")
    print("""
    result = with_reflection(
        "解释量子纠缠",
        max_iterations=2
    )
    print(result.corrected_response or result.original_response)
    """)


def demo_scheduler():
    """异步调度演示"""
    print("\n" + "=" * 50)
    print("5. 异步调度 - 高性能并发调用")
    print("=" * 50)

    # 创建调度器
    config = SchedulerConfig(max_concurrent=5)
    scheduler = CognitiveScheduler(config)

    print("调度器配置:")
    print(f"  - 最大并发数: {config.max_concurrent}")
    print(f"  - 默认超时: {config.default_timeout}s")

    # 异步批量调用示例
    async def demo_async():
        prompts = [
            "什么是机器学习？",
            "什么是深度学习？",
            "什么是自然语言处理？",
        ]
        # results = await batch_call(prompts)
        # for prompt, result in zip(prompts, results):
        #     print(f"  Q: {prompt} -> A: {result[:50]}...")
        print("  使用 batch_call() 可以并发执行多个 LLM 调用")
        print(f"  批量调用数量: {len(prompts)}")

    asyncio.run(demo_async())


def demo_determinism():
    """确定性保证演示"""
    print("\n" + "=" * 50)
    print("6. 确定性保证 - 类型约束与幻觉检测")
    print("=" * 50)

    # 原始类型约束
    int_constraint = PrimitiveConstraint(
        expected_type=int,
        min_value=0,
        max_value=100
    )
    print(f"整数约束: min={int_constraint.min_value}, max={int_constraint.max_value}")

    # 枚举约束
    color_constraint = EnumConstraint(allowed_values=["red", "green", "blue"])
    print(f"枚举约束: {color_constraint.allowed_values}")

    # Schema 约束 (Pydantic)
    class Person(BaseModel):
        name: str
        age: int
        email: str

    schema_constraint = SchemaConstraint(model_class=Person)
    print(f"Schema 约束: Person(name, age, email)")

    # 模拟器
    simulator = Simulator(mode=SimulationMode.MOCK)
    print(f"\n模拟器模式: {simulator.mode.value}")

    # 幻觉检测器
    detector = HallucinationDetector()
    sample_text = "Python 是由 Guido van Rossum 在 1991 年创建的。"

    print(f"\n幻觉检测示例:")
    print(f"  文本: '{sample_text}'")
    check = detector.check(sample_text)
    print(f"  可信度: {check.confidence:.2f}")
    print(f"  原因: {check.reasons}")


def demo_streaming():
    """流式响应演示"""
    print("\n" + "=" * 50)
    print("7. 流式响应 - 实时输出支持")
    print("=" * 50)

    print("流式输出示例 (需要 API key 才能运行):")
    print("""
    for chunk in stream_iter("写一首关于编程的短诗"):
        print(chunk.content, end="", flush=True)
    """)
    print("\n流式响应适合需要实时反馈的场景")


def demo_tools():
    """工具调用演示"""
    print("\n" + "=" * 50)
    print("8. 工具调用 - Function Calling 支持")
    print("=" * 50)

    # 获取全局工具注册表
    registry = get_global_registry()

    # 定义工具并注册到全局注册表
    @registry.register(description="获取指定城市的天气信息")
    def get_weather(city: str) -> str:
        """获取指定城市的天气信息"""
        # 模拟返回
        return f"{city} 今天晴天，温度 25°C"

    @registry.register(description="计算数学表达式")
    def calculate(expression: str) -> float:
        """计算数学表达式"""
        try:
            return eval(expression)
        except Exception:
            return 0.0

    # 获取工具列表
    tools = registry.get_openai_tools()
    print(f"已注册的工具数量: {len(tools)}")

    # 显示工具信息
    for tool_def in tools:
        func = tool_def["function"]
        print(f"  - {func['name']}: {func.get('description', '')}")

    print("\n工具会被 LLM 自动调用以完成任务")


def demo_providers():
    """多提供商演示"""
    print("\n" + "=" * 50)
    print("9. 多提供商支持 - OpenAI、Anthropic、本地模型")
    print("=" * 50)

    # OpenAI
    print("OpenAI:")
    print("  provider = create_provider('openai', api_key='sk-...', model='gpt-4')")

    # Anthropic
    print("Anthropic:")
    print("  provider = create_provider('anthropic', api_key='sk-ant-...', model='claude-3-opus')")

    # Ollama (本地模型)
    print("Ollama (本地):")
    print("  provider = create_provider('ollama', model='llama2', base_url='http://localhost:11434')")

    # HuggingFace (本地模型)
    print("HuggingFace Transformers (本地):")
    print("  provider = create_provider('huggingface', model='microsoft/DialoGPT-medium')")

    print("\n支持的提供商类型:")
    for pt in ProviderType:
        print(f"  - {pt.value}")


def demo_validation():
    """外部验证演示"""
    print("\n" + "=" * 50)
    print("10. 外部验证 - 事实核查与验证")
    print("=" * 50)

    # 创建验证器
    validator = WebSearchValidator()
    print(f"WebSearchValidator: 使用 DuckDuckGo 进行搜索验证")

    # 验证声明
    print("\n验证示例 (需要网络才能运行):")
    print("""
    result = verify_claim("地球是圆的")
    print(f"状态: {result.status}")
    print(f"证据数量: {len(result.evidences)}")
    for evidence in result.evidences:
        print(f"  - {evidence.source}: {evidence.snippet[:50]}...")
    """)

    print("\n支持的验证器:")
    print("  - WebSearchValidator: 网络搜索验证")
    print("  - KnowledgeGraphValidator: 知识图谱查询")
    print("  - FactCheckValidator: 事实核查 API")
    print("  - CompositeValidator: 组合多个验证器")


def demo_context_manager():
    """上下文管理器演示"""
    print("\n" + "=" * 50)
    print("11. 上下文管理器 - 全局配置")
    print("=" * 50)

    # 使用上下文管理器设置全局配置
    print("使用 CognitiveContext 设置全局配置:")
    print("""
    with CognitiveContext(
        model="gpt-4",
        api_key="sk-...",
        temperature=0.7,
        max_tokens=2048
    ):
        # 所有 cognitive 调用都会使用这些配置
        result = cognitive_call("你好")
    """)

    # 嵌套上下文
    print("\n支持嵌套上下文，内层会覆盖外层配置")


def main():
    """运行所有演示"""
    print("=" * 50)
    print("Codegnipy 功能演示")
    print("=" * 50)

    # 基础功能
    demo_basic()

    # 装饰器
    demo_decorator()

    # 记忆存储
    demo_memory()

    # 反思循环
    demo_reflection()

    # 异步调度
    demo_scheduler()

    # 确定性保证
    demo_determinism()

    # 流式响应
    demo_streaming()

    # 工具调用
    demo_tools()

    # 多提供商
    demo_providers()

    # 外部验证
    demo_validation()

    # 上下文管理器
    demo_context_manager()

    print("\n" + "=" * 50)
    print("演示完成！")
    print("=" * 50)
    print("\n提示: 部分功能需要 API key 才能正常运行")
    print("设置环境变量: export OPENAI_API_KEY=sk-xxx")
    print("或使用参数: python -m codegnipy run demo.py -k sk-xxx")


if __name__ == "__main__":
    main()
