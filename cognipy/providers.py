"""
CogniPy 多提供商支持模块

支持多种 LLM 提供商：OpenAI、Anthropic、本地模型等。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, AsyncIterator, Iterator
from enum import Enum
import json

from .streaming import StreamChunk, StreamStatus


class ProviderType(Enum):
    """提供商类型"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


@dataclass
class ProviderConfig:
    """提供商配置"""
    provider_type: ProviderType = ProviderType.OPENAI
    api_key: Optional[str] = None
    model: str = ""
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    extra_params: dict = field(default_factory=dict)


class BaseProvider(ABC):
    """提供商基类"""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
    
    @abstractmethod
    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """执行 LLM 调用"""
        pass
    
    @abstractmethod
    def stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Iterator[StreamChunk]:
        """执行流式调用"""
        pass
    
    @abstractmethod
    async def call_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """执行异步调用"""
        pass
    
    @abstractmethod
    async def stream_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """执行异步流式调用"""
        pass
    
    @abstractmethod
    def call_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[dict],
        **kwargs
    ) -> Dict[str, Any]:
        """执行带工具的调用"""
        pass


class OpenAIProvider(BaseProvider):
    """OpenAI 提供商"""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None
        self._async_client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url
                )
            except ImportError:
                raise ImportError("需要安装 openai 包。运行: pip install openai")
        return self._client
    
    def _get_async_client(self):
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI
                self._async_client = AsyncOpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url
                )
            except ImportError:
                raise ImportError("需要安装 openai 包。运行: pip install openai")
        return self._async_client
    
    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        client = self._get_client()
        
        response = client.chat.completions.create(
            model=kwargs.get("model", self.config.model),
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            **self.config.extra_params
        )
        
        return response.choices[0].message.content
    
    def stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Iterator[StreamChunk]:
        client = self._get_client()
        
        response = client.chat.completions.create(
            model=kwargs.get("model", self.config.model),
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            stream=True,
            **self.config.extra_params
        )
        
        accumulated = ""
        yield StreamChunk(content="", status=StreamStatus.STARTED, accumulated="")
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                accumulated += content
                yield StreamChunk(
                    content=content,
                    status=StreamStatus.STREAMING,
                    accumulated=accumulated
                )
        
        yield StreamChunk(content="", status=StreamStatus.COMPLETED, accumulated=accumulated)
    
    async def call_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        client = self._get_async_client()
        
        response = await client.chat.completions.create(
            model=kwargs.get("model", self.config.model),
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            **self.config.extra_params
        )
        
        return response.choices[0].message.content
    
    async def stream_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        client = self._get_async_client()
        
        response = await client.chat.completions.create(
            model=kwargs.get("model", self.config.model),
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            stream=True,
            **self.config.extra_params
        )
        
        accumulated = ""
        yield StreamChunk(content="", status=StreamStatus.STARTED, accumulated="")
        
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                accumulated += content
                yield StreamChunk(
                    content=content,
                    status=StreamStatus.STREAMING,
                    accumulated=accumulated
                )
        
        yield StreamChunk(content="", status=StreamStatus.COMPLETED, accumulated=accumulated)
    
    def call_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[dict],
        **kwargs
    ) -> Dict[str, Any]:
        client = self._get_client()
        
        response = client.chat.completions.create(
            model=kwargs.get("model", self.config.model),
            messages=messages,
            tools=tools,
            tool_choice=kwargs.get("tool_choice", "auto"),
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            **self.config.extra_params
        )
        
        message = response.choices[0].message
        
        return {
            "content": message.content,
            "tool_calls": message.tool_calls,
            "message": message
        }


class AnthropicProvider(BaseProvider):
    """Anthropic 提供商"""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None
        self._async_client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(
                    api_key=self.config.api_key
                )
            except ImportError:
                raise ImportError(
                    "需要安装 anthropic 包。运行: pip install anthropic"
                )
        return self._client
    
    def _get_async_client(self):
        if self._async_client is None:
            try:
                import anthropic
                self._async_client = anthropic.AsyncAnthropic(
                    api_key=self.config.api_key
                )
            except ImportError:
                raise ImportError(
                    "需要安装 anthropic 包。运行: pip install anthropic"
                )
        return self._async_client
    
    def _convert_messages(
        self,
        messages: List[Dict[str, str]]
    ) -> tuple[str, List[Dict[str, str]]]:
        """转换消息格式为 Anthropic 格式"""
        system = ""
        converted = []
        
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            elif msg["role"] in ("user", "assistant"):
                converted.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        return system, converted
    
    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        client = self._get_client()
        system, converted = self._convert_messages(messages)
        
        params = {
            "model": kwargs.get("model", self.config.model),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "messages": converted,
        }
        
        if system:
            params["system"] = system
        if "temperature" in kwargs or self.config.temperature:
            params["temperature"] = kwargs.get("temperature", self.config.temperature)
        
        params.update(self.config.extra_params)
        
        response = client.messages.create(**params)
        
        # 提取文本内容
        for block in response.content:
            if hasattr(block, 'text'):
                return block.text
        
        return ""
    
    def stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Iterator[StreamChunk]:
        client = self._get_client()
        system, converted = self._convert_messages(messages)
        
        params = {
            "model": kwargs.get("model", self.config.model),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "messages": converted,
        }
        
        if system:
            params["system"] = system
        if "temperature" in kwargs or self.config.temperature:
            params["temperature"] = kwargs.get("temperature", self.config.temperature)
        
        params.update(self.config.extra_params)
        
        accumulated = ""
        yield StreamChunk(content="", status=StreamStatus.STARTED, accumulated="")
        
        with client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                accumulated += text
                yield StreamChunk(
                    content=text,
                    status=StreamStatus.STREAMING,
                    accumulated=accumulated
                )
        
        yield StreamChunk(content="", status=StreamStatus.COMPLETED, accumulated=accumulated)
    
    async def call_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        client = self._get_async_client()
        system, converted = self._convert_messages(messages)
        
        params = {
            "model": kwargs.get("model", self.config.model),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "messages": converted,
        }
        
        if system:
            params["system"] = system
        if "temperature" in kwargs or self.config.temperature:
            params["temperature"] = kwargs.get("temperature", self.config.temperature)
        
        params.update(self.config.extra_params)
        
        response = await client.messages.create(**params)
        
        for block in response.content:
            if hasattr(block, 'text'):
                return block.text
        
        return ""
    
    async def stream_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        client = self._get_async_client()
        system, converted = self._convert_messages(messages)
        
        params = {
            "model": kwargs.get("model", self.config.model),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "messages": converted,
        }
        
        if system:
            params["system"] = system
        if "temperature" in kwargs or self.config.temperature:
            params["temperature"] = kwargs.get("temperature", self.config.temperature)
        
        params.update(self.config.extra_params)
        
        accumulated = ""
        yield StreamChunk(content="", status=StreamStatus.STARTED, accumulated="")
        
        async with client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                accumulated += text
                yield StreamChunk(
                    content=text,
                    status=StreamStatus.STREAMING,
                    accumulated=accumulated
                )
        
        yield StreamChunk(content="", status=StreamStatus.COMPLETED, accumulated=accumulated)
    
    def call_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[dict],
        **kwargs
    ) -> Dict[str, Any]:
        client = self._get_client()
        system, converted = self._convert_messages(messages)
        
        # 转换工具格式
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {})
                })
        
        params = {
            "model": kwargs.get("model", self.config.model),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "messages": converted,
            "tools": anthropic_tools,
        }
        
        if system:
            params["system"] = system
        
        params.update(self.config.extra_params)
        
        response = client.messages.create(**params)
        
        # 解析工具调用
        tool_calls = []
        content = ""
        
        for block in response.content:
            if hasattr(block, 'text'):
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input)
                    }
                })
        
        return {
            "content": content,
            "tool_calls": tool_calls if tool_calls else None,
            "message": response
        }


class ProviderFactory:
    """提供商工厂"""
    
    _providers: Dict[ProviderType, type] = {
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.ANTHROPIC: AnthropicProvider,
    }
    
    @classmethod
    def create(cls, config: ProviderConfig) -> BaseProvider:
        """创建提供商实例"""
        provider_class = cls._providers.get(config.provider_type)
        
        if provider_class is None:
            raise ValueError(f"Unknown provider type: {config.provider_type}")
        
        return provider_class(config)
    
    @classmethod
    def register(cls, provider_type: ProviderType, provider_class: type) -> None:
        """注册自定义提供商"""
        cls._providers[provider_type] = provider_class


def create_provider(
    provider_type: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> BaseProvider:
    """
    创建提供商实例
    
    参数:
        provider_type: 提供商类型 ("openai", "anthropic", "custom")
        api_key: API 密钥
        model: 模型名称
        **kwargs: 其他配置
    
    返回:
        提供商实例
    
    示例:
        # OpenAI
        provider = create_provider("openai", api_key="sk-...", model="gpt-4")
        
        # Anthropic
        provider = create_provider("anthropic", api_key="sk-ant-...", model="claude-3-opus")
    """
    type_map = {
        "openai": ProviderType.OPENAI,
        "anthropic": ProviderType.ANTHROPIC,
        "custom": ProviderType.CUSTOM,
    }
    
    pt = type_map.get(provider_type.lower())
    if pt is None:
        raise ValueError(f"Unknown provider type: {provider_type}")
    
    config = ProviderConfig(
        provider_type=pt,
        api_key=api_key,
        model=model or "",
        **kwargs
    )
    
    return ProviderFactory.create(config)
