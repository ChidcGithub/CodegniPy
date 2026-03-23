"""
Codegnipy 功能演示

运行方式:
    python -m codegnipy run examples/demo.py

注意: 
    - 需要设置 OPENAI_API_KEY 环境变量
    - 或使用 -k 参数: python -m codegnipy run examples/demo.py -k sk-xxx
"""

import os
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


def get_api_key() -> str | None:
    """获取 API key"""
    return os.environ.get("OPENAI_API_KEY")


def demo_basic():
    """基础功能演示"""
    print("\n" + "=" * 50)
    print("1. 基础功能 - cognitive_call")
    print("=" * 50)

    api_key = get_api_key()
    if not api_key:
        print("需要设置 OPENAI_API_KEY 环境变量")
        return

    with CognitiveContext(api_key=api_key, model="gpt-4o-mini"):
        # 方式 1: 使用 cognitive_call
        result = cognitive_call("用一句话解释什么是递归")
        print(f"cognitive_call 结果: {result}")


def demo_decorator():
    """装饰器功能演示"""
    print("\n" + "=" * 50)
    print("2. @cognitive 装饰器 - 让函数由 LLM 实现")
    print("=" * 50)

    api_key = get_api_key()
    if not api_key:
        print("需要设置 OPENAI_API_KEY 环境变量")
        return

    @cognitive
    def summarize(text: str) -> str:
        """总结这段文字的核心观点，不超过两句话。"""
        pass

    @cognitive
    def translate(text: str, target_lang: str = "英文") -> str:
        """将文本翻译成目标语言。"""
        pass

    with CognitiveContext(api_key=api_key, model="gpt-4o-mini"):
        # 调用 LLM 实现的函数
        text = "Python 是一种广泛使用的高级编程语言，由 Guido van Rossum 于 1991 年创建。它以简洁清晰的语法著称，被广泛用于 Web 开发、数据科学、人工智能等领域。"
        
        summary = summarize(text)
        print(f"原文: {text[:50]}...")
        print(f"摘要: {summary}")

        translated = translate("Hello, World!", target_lang="中文")
        print(f"翻译: Hello, World! -> {translated}")


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

    # 在上下文中使用记忆
    api_key = get_api_key()
    if api_key:
        print("\n在 CognitiveContext 中使用记忆:")
        with CognitiveContext(api_key=api_key, model="gpt-4o-mini") as ctx:
            ctx.memory.add(Message(role=MessageRole.USER, content="我喜欢编程"))
            ctx.memory.add(Message(role=MessageRole.ASSISTANT, content="太好了！编程是一项很有趣的技能。"))
            
            result = cognitive_call("根据之前的对话，我有什么爱好？")
            print(f"LLM 回答: {result}")


def demo_reflection():
    """反思循环演示"""
    print("\n" + "=" * 50)
    print("4. 反思循环 - LLM 自我检查与修正")
    print("=" * 50)

    api_key = get_api_key()
    if not api_key:
        print("需要设置 OPENAI_API_KEY 环境变量")
        return

    with CognitiveContext(api_key=api_key, model="gpt-4o-mini"):
        result = with_reflection(
            "请解释量子纠缠现象，要求包含具体例子和日常应用",
            max_iterations=2
        )
        
        print(f"原始回答: {result.original_response[:200]}...")
        if result.corrected_response:
            print(f"修正后回答: {result.corrected_response[:200]}...")
        print(f"状态: {result.status.value}")
        if result.issues:
            print(f"发现的问题: {result.issues}")


def demo_scheduler():
    """异步调度演示"""
    print("\n" + "=" * 50)
    print("5. 异步调度 - 高性能并发调用")
    print("=" * 50)

    api_key = get_api_key()
    if not api_key:
        print("需要设置 OPENAI_API_KEY 环境变量")
        return

    async def run_batch():
        prompts = [
            "用一句话解释什么是机器学习",
            "用一句话解释什么是深度学习",
            "用一句话解释什么是自然语言处理",
        ]
        
        with CognitiveContext(api_key=api_key, model="gpt-4o-mini"):
            results = await batch_call(prompts)
            
            for prompt, result in zip(prompts, results):
                print(f"  Q: {prompt}")
                print(f"  A: {result}")
                print()

    asyncio.run(run_batch())


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
    
    # 验证约束
    result = int_constraint.validate(50)
    print(f"整数约束验证 50: {result.status.value}")
    
    result = int_constraint.validate(150)
    print(f"整数约束验证 150: {result.status.value} - {result.errors}")

    # 枚举约束
    color_constraint = EnumConstraint(allowed_values=["red", "green", "blue"])
    result = color_constraint.validate("red")
    print(f"枚举约束验证 'red': {result.status.value}")
    
    result = color_constraint.validate("yellow")
    print(f"枚举约束验证 'yellow': {result.status.value} - {result.errors}")

    # Schema 约束
    class Person(BaseModel):
        name: str
        age: int
        email: str

    schema_constraint = SchemaConstraint(model_class=Person)
    
    valid_data = '{"name": "张三", "age": 25, "email": "zhangsan@example.com"}'
    result = schema_constraint.validate(valid_data)
    print(f"Schema 验证有效数据: {result.status.value} -> {result.value}")

    invalid_data = '{"name": "张三", "age": "不是数字"}'
    result = schema_constraint.validate(invalid_data)
    print(f"Schema 验证无效数据: {result.status.value} - {result.errors}")

    # 模拟器
    print("\n模拟器演示:")
    simulator = Simulator(mode=SimulationMode.MOCK)
    simulator.set_mock_response("Python", "Python 是一种高级编程语言，以简洁易读著称。")
    simulated = simulator.get_response("解释一下 Python")
    print(f"模拟结果: {simulated}")

    # 幻觉检测器
    print("\n幻觉检测:")
    detector = HallucinationDetector()
    
    sample_texts = [
        "Python 是由 Guido van Rossum 在 1991 年创建的。",
        "太阳从西边升起。",
        "地球是平的，所有的卫星照片都是伪造的。",
    ]
    
    for text in sample_texts:
        check = detector.check(text)
        print(f"  '{text[:30]}...'")
        print(f"    可信度: {check.confidence:.2f}, 是否幻觉: {check.is_hallucination}")


def demo_streaming():
    """流式响应演示"""
    print("\n" + "=" * 50)
    print("7. 流式响应 - 实时输出支持")
    print("=" * 50)

    api_key = get_api_key()
    if not api_key:
        print("需要设置 OPENAI_API_KEY 环境变量")
        return

    print("流式输出:")
    with CognitiveContext(api_key=api_key, model="gpt-4o-mini"):
        for chunk in stream_iter("写一首关于编程的四行短诗"):
            print(chunk.content, end="", flush=True)
        print()  # 换行


def demo_tools():
    """工具调用演示"""
    print("\n" + "=" * 50)
    print("8. 工具调用 - Function Calling 支持")
    print("=" * 50)

    # 获取全局工具注册表
    registry = get_global_registry()

    # 定义工具并注册
    @registry.register(description="获取指定城市的天气信息")
    def get_weather(city: str) -> str:
        """获取指定城市的天气信息"""
        # 模拟返回
        weather_data = {
            "北京": "晴天，温度 15°C",
            "上海": "多云，温度 18°C",
            "深圳": "小雨，温度 22°C",
        }
        return weather_data.get(city, f"{city} 天气信息不可用")

    @registry.register(description="计算数学表达式，返回计算结果")
    def calculate(expression: str) -> float:
        """计算数学表达式"""
        try:
            # 安全计算
            allowed = set("0123456789+-*/.() ")
            if all(c in allowed for c in expression):
                return eval(expression)
            return 0.0
        except Exception:
            return 0.0

    # 获取工具列表
    tools = registry.get_openai_tools()
    print(f"已注册的工具数量: {len(tools)}")

    # 测试工具执行
    print("\n直接执行工具:")
    print(f"  get_weather('北京'): {get_weather('北京')}")
    print(f"  calculate('2 + 3 * 4'): {calculate('2 + 3 * 4')}")

    # LLM 调用工具
    api_key = get_api_key()
    if api_key:
        print("\n让 LLM 调用工具:")
        with CognitiveContext(api_key=api_key, model="gpt-4o-mini"):
            result = call_with_tools(
                "北京和上海今天的天气怎么样？",
                tools=tools
            )
            print(f"  回答: {result}")


def demo_providers():
    """多提供商演示"""
    print("\n" + "=" * 50)
    print("9. 多提供商支持")
    print("=" * 50)

    print("支持的提供商类型:")
    for pt in ProviderType:
        print(f"  - {pt.value}")

    # OpenAI Provider
    api_key = get_api_key()
    if api_key:
        print("\n使用 OpenAI Provider:")
        provider = create_provider("openai", api_key=api_key, model="gpt-4o-mini")
        
        messages = [{"role": "user", "content": "用5个字回答：什么是AI？"}]
        result = provider.call(messages)
        print(f"  OpenAI 回答: {result}")

    # Ollama Provider (检查是否运行)
    print("\n检查 Ollama 是否可用:")
    try:
        ollama = create_provider("ollama", model="llama2", base_url="http://localhost:11434")
        models = ollama.list_models()
        if models:
            print(f"  可用模型: {models}")
        else:
            print("  Ollama 服务未运行或没有模型")
    except Exception as e:
        print(f"  Ollama 不可用: {e}")


def demo_validation():
    """外部验证演示"""
    print("\n" + "=" * 50)
    print("10. 外部验证 - 事实核查")
    print("=" * 50)

    # 创建验证器
    validator = WebSearchValidator()
    print(f"WebSearchValidator 状态: {'可用' if validator.is_available() else '不可用'}")

    # 验证声明
    claims = [
        "Python 是在 1991 年发布的",
        "地球是平的",
    ]

    for claim in claims:
        print(f"\n验证: '{claim}'")
        try:
            result = verify_claim(claim, validators=["web_search"])
            print(f"  状态: {result.status.value}")
            print(f"  可信度: {result.confidence:.2f}")
            if result.evidences:
                print(f"  证据数量: {len(result.evidences)}")
                for ev in result.evidences[:2]:  # 只显示前两个
                    print(f"    - {ev.source}: {ev.snippet[:50]}...")
        except Exception as e:
            print(f"  验证失败: {e}")


def demo_context_manager():
    """上下文管理器演示"""
    print("\n" + "=" * 50)
    print("11. 上下文管理器 - 全局配置")
    print("=" * 50)

    api_key = get_api_key()
    if not api_key:
        print("需要设置 OPENAI_API_KEY 环境变量")
        return

    # 外层上下文
    with CognitiveContext(api_key=api_key, model="gpt-4o-mini", temperature=0.7):
        print("外层上下文: model=gpt-4o-mini, temperature=0.7")
        
        result1 = cognitive_call("说一个形容词")
        print(f"  回答 1: {result1}")

        # 嵌套内层上下文
        with CognitiveContext(model="gpt-4o-mini", temperature=1.5):
            print("\n内层上下文: temperature=1.5 (更随机)")
            
            result2 = cognitive_call("说一个形容词")
            print(f"  回答 2: {result2}")

        print("\n回到外层上下文")
        result3 = cognitive_call("再说一个形容词")
        print(f"  回答 3: {result3}")


def main():
    """运行所有演示"""
    print("=" * 50)
    print("Codegnipy 功能演示")
    print("=" * 50)

    api_key = get_api_key()
    if api_key:
        print(f"API Key 已设置: {api_key[:10]}...")
    else:
        print("警告: 未设置 OPENAI_API_KEY 环境变量")
        print("部分演示将跳过")

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


if __name__ == "__main__":
    main()