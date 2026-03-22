"""
CogniPy 流式响应模块

提供 LLM 流式输出支持，实现实时响应。
"""

from dataclasses import dataclass, field
from typing import AsyncIterator, Iterator, Optional, Callable, List
from enum import Enum
import asyncio

from .runtime import LLMConfig, CognitiveContext


class StreamStatus(Enum):
    """流状态"""
    STARTED = "started"
    STREAMING = "streaming"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class StreamChunk:
    """流式响应块"""
    content: str
    status: StreamStatus
    accumulated: str = ""
    metadata: dict = field(default_factory=dict)
    
    def __str__(self) -> str:
        return self.content


@dataclass
class StreamResult:
    """流式响应结果"""
    content: str
    status: StreamStatus
    chunks: List[StreamChunk] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _stream_openai(
    config: LLMConfig,
    prompt: str,
    memory: Optional[list] = None,
    **kwargs
) -> Iterator[StreamChunk]:
    """使用 OpenAI API 进行流式调用"""
    try:
        import openai
    except ImportError:
        raise ImportError("需要安装 openai 包。运行: pip install openai")

    client = openai.OpenAI(
        api_key=config.api_key,
        base_url=config.base_url
    )

    messages: List = []
    if memory:
        messages.extend(memory)
    messages.append({"role": "user", "content": prompt})
    
    accumulated = ""
    
    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stream=True,
            **kwargs
        )
        
        # 开始
        yield StreamChunk(
            content="",
            status=StreamStatus.STARTED,
            accumulated=""
        )
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                accumulated += content
                
                yield StreamChunk(
                    content=content,
                    status=StreamStatus.STREAMING,
                    accumulated=accumulated
                )
        
        # 完成
        yield StreamChunk(
            content="",
            status=StreamStatus.COMPLETED,
            accumulated=accumulated
        )
        
    except Exception as e:
        yield StreamChunk(
            content=str(e),
            status=StreamStatus.ERROR,
            accumulated=accumulated,
            metadata={"error": str(e)}
        )


async def _stream_openai_async(
    config: LLMConfig,
    prompt: str,
    memory: Optional[list] = None,
    **kwargs
) -> AsyncIterator[StreamChunk]:
    """使用 OpenAI API 进行异步流式调用"""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError("需要安装 openai 包。运行: pip install openai")

    client = AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url
    )

    messages: List = []
    if memory:
        messages.extend(memory)
    messages.append({"role": "user", "content": prompt})
    
    accumulated = ""
    
    try:
        response = await client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stream=True,
            **kwargs
        )
        
        yield StreamChunk(
            content="",
            status=StreamStatus.STARTED,
            accumulated=""
        )
        
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                accumulated += content
                
                yield StreamChunk(
                    content=content,
                    status=StreamStatus.STREAMING,
                    accumulated=accumulated
                )
        
        yield StreamChunk(
            content="",
            status=StreamStatus.COMPLETED,
            accumulated=accumulated
        )
        
    except Exception as e:
        yield StreamChunk(
            content=str(e),
            status=StreamStatus.ERROR,
            accumulated=accumulated,
            metadata={"error": str(e)}
        )


def stream_call(
    prompt: str,
    context: Optional[CognitiveContext] = None,
    *,
    on_chunk: Optional[Callable[[StreamChunk], None]] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None
) -> StreamResult:
    """
    执行流式认知调用
    
    参数:
        prompt: 发送给 LLM 的提示
        context: 认知上下文
        on_chunk: 每个块的回调函数
        model: 覆盖模型设置
        temperature: 覆盖温度设置
    
    返回:
        StreamResult 包含完整响应
    
    示例:
        result = stream_call("解释量子计算", on_chunk=lambda c: print(c.content, end=""))
        print(result.content)
    """
    ctx = context or CognitiveContext.get_current()
    
    if ctx is None:
        config = LLMConfig()
    else:
        config = ctx.get_config()
        if model:
            config.model = model
        if temperature is not None:
            config.temperature = temperature
    
    if not config.api_key:
        raise ValueError(
            "未配置 API 密钥。请设置 OPENAI_API_KEY 环境变量，"
            "或在 CognitiveContext 中提供 api_key 参数。"
        )
    
    memory = ctx.get_memory() if ctx else []
    
    chunks = []
    accumulated = ""
    
    for chunk in _stream_openai(config, prompt, memory):
        chunks.append(chunk)
        accumulated = chunk.accumulated
        
        if on_chunk:
            on_chunk(chunk)
    
    # 更新记忆
    if ctx:
        ctx.add_to_memory("user", prompt)
        ctx.add_to_memory("assistant", accumulated)
    
    return StreamResult(
        content=accumulated,
        status=chunks[-1].status if chunks else StreamStatus.ERROR,
        chunks=chunks
    )


async def stream_call_async(
    prompt: str,
    context: Optional[CognitiveContext] = None,
    *,
    on_chunk: Optional[Callable[[StreamChunk], None]] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None
) -> StreamResult:
    """
    执行异步流式认知调用
    
    参数:
        prompt: 发送给 LLM 的提示
        context: 认知上下文
        on_chunk: 每个块的回调函数
        model: 覆盖模型设置
        temperature: 覆盖温度设置
    
    返回:
        StreamResult 包含完整响应
    
    示例:
        result = await stream_call_async("解释量子计算")
        print(result.content)
    """
    ctx = context or CognitiveContext.get_current()
    
    if ctx is None:
        config = LLMConfig()
    else:
        config = ctx.get_config()
        if model:
            config.model = model
        if temperature is not None:
            config.temperature = temperature
    
    if not config.api_key:
        raise ValueError(
            "未配置 API 密钥。请设置 OPENAI_API_KEY 环境变量，"
            "或在 CognitiveContext 中提供 api_key 参数。"
        )
    
    memory = ctx.get_memory() if ctx else []
    
    chunks = []
    accumulated = ""
    
    async for chunk in _stream_openai_async(config, prompt, memory):
        chunks.append(chunk)
        accumulated = chunk.accumulated
        
        if on_chunk:
            if asyncio.iscoroutinefunction(on_chunk):
                await on_chunk(chunk)
            else:
                on_chunk(chunk)
    
    if ctx:
        ctx.add_to_memory("user", prompt)
        ctx.add_to_memory("assistant", accumulated)
    
    return StreamResult(
        content=accumulated,
        status=chunks[-1].status if chunks else StreamStatus.ERROR,
        chunks=chunks
    )


def stream_iter(
    prompt: str,
    context: Optional[CognitiveContext] = None,
    **kwargs
) -> Iterator[StreamChunk]:
    """
    流式迭代器，逐块返回响应
    
    示例:
        for chunk in stream_iter("解释量子计算"):
            print(chunk.content, end="", flush=True)
    """
    ctx = context or CognitiveContext.get_current()
    
    if ctx is None:
        config = LLMConfig()
    else:
        config = ctx.get_config()
        if kwargs.get("model"):
            config.model = kwargs["model"]
        if kwargs.get("temperature") is not None:
            config.temperature = kwargs["temperature"]
    
    if not config.api_key:
        raise ValueError("未配置 API 密钥。")
    
    memory = ctx.get_memory() if ctx else []
    
    accumulated = ""
    
    for chunk in _stream_openai(config, prompt, memory):
        yield chunk
        accumulated = chunk.accumulated
    
    if ctx:
        ctx.add_to_memory("user", prompt)
        ctx.add_to_memory("assistant", accumulated)


async def stream_iter_async(
    prompt: str,
    context: Optional[CognitiveContext] = None,
    **kwargs
) -> AsyncIterator[StreamChunk]:
    """
    异步流式迭代器
    
    示例:
        async for chunk in stream_iter_async("解释量子计算"):
            print(chunk.content, end="", flush=True)
    """
    ctx = context or CognitiveContext.get_current()
    
    if ctx is None:
        config = LLMConfig()
    else:
        config = ctx.get_config()
        if kwargs.get("model"):
            config.model = kwargs["model"]
        if kwargs.get("temperature") is not None:
            config.temperature = kwargs["temperature"]
    
    if not config.api_key:
        raise ValueError("未配置 API 密钥。")
    
    memory = ctx.get_memory() if ctx else []
    
    accumulated = ""
    
    async for chunk in _stream_openai_async(config, prompt, memory):
        yield chunk
        accumulated = chunk.accumulated
    
    if ctx:
        ctx.add_to_memory("user", prompt)
        ctx.add_to_memory("assistant", accumulated)
