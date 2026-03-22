"""
CogniPy 工具调用模块

提供 Function Calling / Tool Use 支持，让 LLM 能够调用外部函数。
"""

import json
import inspect
from dataclasses import dataclass
from typing import (
    Any, Callable, Dict, List, Optional, Type, Union,
    get_type_hints, get_origin, get_args
)
from enum import Enum
from functools import wraps

from .runtime import LLMConfig, CognitiveContext


class ToolType(Enum):
    """工具类型"""
    FUNCTION = "function"


@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    param_type: str
    description: str = ""
    required: bool = True
    enum: Optional[List[str]] = None
    default: Any = None

    def to_json_schema(self) -> dict:
        """转换为 JSON Schema"""
        schema: Dict[str, Any] = {
            "type": self.param_type,
            "description": self.description
        }
        if self.enum:
            schema["enum"] = self.enum
        return schema


@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str
    parameters: List[ToolParameter]
    type: ToolType = ToolType.FUNCTION
    handler: Optional[Callable] = None
    
    def to_openai_tool(self) -> dict:
        """转换为 OpenAI 工具格式"""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "type": self.type.value,
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }


@dataclass
class ToolCall:
    """工具调用请求"""
    id: str
    name: str
    arguments: dict
    
    def execute(self, handler: Callable) -> Any:
        """执行工具调用"""
        return handler(**self.arguments)


@dataclass
class ToolResult:
    """工具调用结果"""
    tool_call_id: str
    name: str
    arguments: dict
    result: Any
    error: Optional[str] = None
    
    def to_openai_format(self) -> dict:
        """转换为 OpenAI 格式"""
        content = json.dumps(self.result) if self.result else self.error
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": content
        }


class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
    
    def register(
        self,
        func: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        description: str = ""
    ) -> Callable:
        """
        装饰器：注册工具函数
        
        支持 @registry.register 和 @registry.register() 两种用法。
        
        示例:
            @registry.register
            def get_weather(city: str) -> str:
                return f"{city}的天气晴朗"
            
            @registry.register(description="获取当前天气")
            def get_weather2(city: str) -> str:
                return f"{city}的天气晴朗"
        """
        def decorator(fn: Callable) -> Callable:
            tool_name = name or fn.__name__
            tool_desc = description or fn.__doc__ or f"执行 {tool_name}"
            
            # 从函数签名推断参数
            sig = inspect.signature(fn)
            hints = get_type_hints(fn) if hasattr(fn, '__annotations__') else {}
            
            parameters = []
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                    
                param_type = self._python_type_to_json(hints.get(param_name, str))
                required = param.default is inspect.Parameter.empty
                default = None if required else param.default
                
                parameters.append(ToolParameter(
                    name=param_name,
                    param_type=param_type,
                    description=f"参数 {param_name}",
                    required=required,
                    default=default
                ))
            
            tool = ToolDefinition(
                name=tool_name,
                description=tool_desc,
                parameters=parameters,
                handler=fn
            )
            
            self._tools[tool_name] = tool
            
            @wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            wrapper._tool_definition = tool  # type: ignore[attr-defined]

            return wrapper
        
        # 支持 @registry.register 和 @registry.register() 两种用法
        if func is not None:
            return decorator(func)
        return decorator
    
    def add_tool(self, tool: ToolDefinition) -> None:
        """添加工具定义"""
        self._tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """获取工具定义"""
        return self._tools.get(name)
    
    def get_all_tools(self) -> List[ToolDefinition]:
        """获取所有工具"""
        return list(self._tools.values())
    
    def get_openai_tools(self) -> List[dict]:
        """获取 OpenAI 格式的工具列表"""
        return [tool.to_openai_tool() for tool in self._tools.values()]
    
    def execute(self, tool_call: ToolCall) -> ToolResult:
        """执行工具调用"""
        tool = self.get_tool(tool_call.name)
        
        if tool is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                arguments=tool_call.arguments,
                result=None,
                error=f"Unknown tool: {tool_call.name}"
            )
        
        if tool.handler is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                arguments=tool_call.arguments,
                result=None,
                error=f"No handler for tool: {tool_call.name}"
            )
        
        try:
            result = tool.handler(**tool_call.arguments)
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                arguments=tool_call.arguments,
                result=result
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                arguments=tool_call.arguments,
                result=None,
                error=str(e)
            )
    
    @staticmethod
    def _python_type_to_json(python_type: Type) -> str:
        """将 Python 类型转换为 JSON Schema 类型"""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }
        
        # 处理 Optional 类型
        origin = get_origin(python_type)
        if origin is Union:
            args = get_args(python_type)
            # Optional[X] 实际上是 Union[X, None]
            non_none_args = [a for a in args if a is not type(None)]
            if non_none_args:
                return ToolRegistry._python_type_to_json(non_none_args[0])
        
        if origin is list:
            return "array"
        if origin is dict:
            return "object"
        
        return type_map.get(python_type, "string")


def tool(
    name: Optional[str] = None,
    description: str = ""
) -> Callable:
    """
    装饰器：定义工具函数
    
    示例:
        @tool(description="获取指定城市的天气信息")
        def get_weather(city: str, unit: str = "celsius") -> str:
            return f"{city}当前温度25{unit}"
    """
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or f"执行 {tool_name}"
        
        sig = inspect.signature(func)
        hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}
        
        parameters = []
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            param_type = ToolRegistry._python_type_to_json(hints.get(param_name, str))
            required = param.default is inspect.Parameter.empty
            default = None if required else param.default

            parameters.append(ToolParameter(
                name=param_name,
                param_type=param_type,
                description=f"参数 {param_name}",
                required=required,
                default=default
            ))
        
        tool_def = ToolDefinition(
            name=tool_name,
            description=tool_desc,
            parameters=parameters,
            handler=func
        )

        func._tool_definition = tool_def  # type: ignore[attr-defined]

        return func
    
    return decorator


def call_with_tools(
    prompt: str,
    tools: List[Union[ToolDefinition, Callable]],
    context: Optional[CognitiveContext] = None,
    *,
    max_iterations: int = 5,
    auto_execute: bool = True,
    **kwargs
) -> str:
    """
    执行带工具调用的认知请求
    
    参数:
        prompt: 用户提示
        tools: 工具列表（ToolDefinition 或带 @tool 装饰的函数）
        context: 认知上下文
        max_iterations: 最大工具调用迭代次数
        auto_execute: 是否自动执行工具调用
        **kwargs: 其他参数传递给 cognitive_call
    
    返回:
        最终响应文本
    
    示例:
        @tool
        def search(query: str) -> str:
            return f"搜索结果: {query}"
        
        result = call_with_tools(
            "搜索 Python 教程",
            tools=[search]
        )
    """
    try:
        import openai
    except ImportError:
        raise ImportError("需要安装 openai 包。运行: pip install openai")
    
    ctx = context or CognitiveContext.get_current()
    
    if ctx is None:
        config = LLMConfig()
    else:
        config = ctx.get_config()
    
    if not config.api_key:
        raise ValueError("未配置 API 密钥。")
    
    # 构建工具列表
    tool_defs = []
    tool_handlers = {}
    
    for t in tools:
        if isinstance(t, ToolDefinition):
            tool_defs.append(t)
            if t.handler:
                tool_handlers[t.name] = t.handler
        elif hasattr(t, '_tool_definition'):
            td = t._tool_definition  # type: ignore[attr-defined]
            tool_defs.append(td)
            tool_handlers[td.name] = t
        elif callable(t):
            # 从函数创建工具定义
            td = ToolDefinition(
                name=t.__name__,
                description=t.__doc__ or f"执行 {t.__name__}",
                parameters=[],
                handler=t
            )
            tool_defs.append(td)
            tool_handlers[td.name] = t
    
    client = openai.OpenAI(
        api_key=config.api_key,
        base_url=config.base_url
    )
    
    messages: List[Dict[str, Any]] = []
    if ctx:
        messages.extend(ctx.get_memory())
    messages.append({"role": "user", "content": prompt})

    openai_tools = [td.to_openai_tool() for td in tool_defs]

    iteration = 0
    while iteration < max_iterations:
        response = client.chat.completions.create(  # type: ignore[call-overload]
            model=config.model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        message = response.choices[0].message
        
        # 没有工具调用，返回结果
        if not message.tool_calls:
            result = message.content or ""
            if ctx:
                ctx.add_to_memory("user", prompt)
                ctx.add_to_memory("assistant", result)
            return result
        
        # 有工具调用
        messages.append(message)
        
        for tool_call in message.tool_calls:
            tc = ToolCall(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments)
            )
            
            if auto_execute and tc.name in tool_handlers:
                try:
                    result = tool_handlers[tc.name](**tc.arguments)
                    tool_result = ToolResult(
                        tool_call_id=tc.id,
                        name=tc.name,
                        arguments=tc.arguments,
                        result=result
                    )
                except Exception as e:
                    tool_result = ToolResult(
                        tool_call_id=tc.id,
                        name=tc.name,
                        arguments=tc.arguments,
                        result=None,
                        error=str(e)
                    )
                
                messages.append(tool_result.to_openai_format())
        
        iteration += 1
    
    # 达到最大迭代次数，返回最后响应
    final_response = client.chat.completions.create(
        model=config.model,
        messages=messages,  # type: ignore[arg-type]
        temperature=config.temperature,
        max_tokens=config.max_tokens
    )

    return final_response.choices[0].message.content or ""


# 全局工具注册表
_global_registry = ToolRegistry()


def register_tool(
    name: Optional[str] = None,
    description: str = ""
) -> Callable:
    """注册工具到全局注册表"""
    return _global_registry.register(name=name, description=description)


def get_global_registry() -> ToolRegistry:
    """获取全局工具注册表"""
    return _global_registry
