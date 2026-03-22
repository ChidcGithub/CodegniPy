"""
工具调用模块测试
"""

import pytest
from cognipy.tools import (
    ToolType,
    ToolParameter,
    ToolDefinition,
    ToolCall,
    ToolResult,
    ToolRegistry,
    tool
)


class TestToolParameter:
    """ToolParameter 测试"""
    
    def test_parameter_creation(self):
        """测试参数创建"""
        param = ToolParameter(
            name="city",
            type="string",
            description="城市名称",
            required=True
        )
        assert param.name == "city"
        assert param.type == "string"
        assert param.required is True
    
    def test_parameter_to_json_schema(self):
        """测试转换为 JSON Schema"""
        param = ToolParameter(
            name="count",
            type="integer",
            description="数量",
            enum=[1, 2, 3]
        )
        schema = param.to_json_schema()
        
        assert schema["type"] == "integer"
        assert schema["description"] == "数量"
        assert schema["enum"] == [1, 2, 3]


class TestToolDefinition:
    """ToolDefinition 测试"""
    
    def test_definition_creation(self):
        """测试工具定义创建"""
        tool_def = ToolDefinition(
            name="get_weather",
            description="获取天气信息",
            parameters=[
                ToolParameter("city", "string", "城市"),
                ToolParameter("unit", "string", "单位", required=False)
            ]
        )
        
        assert tool_def.name == "get_weather"
        assert len(tool_def.parameters) == 2
    
    def test_to_openai_tool(self):
        """测试转换为 OpenAI 工具格式"""
        tool_def = ToolDefinition(
            name="search",
            description="搜索",
            parameters=[
                ToolParameter("query", "string", "搜索词", required=True)
            ]
        )
        
        openai_tool = tool_def.to_openai_tool()
        
        assert openai_tool["type"] == "function"
        assert openai_tool["function"]["name"] == "search"
        assert "parameters" in openai_tool["function"]
        assert "query" in openai_tool["function"]["parameters"]["properties"]


class TestToolCall:
    """ToolCall 测试"""
    
    def test_tool_call_creation(self):
        """测试工具调用创建"""
        call = ToolCall(
            id="call_123",
            name="get_weather",
            arguments={"city": "Beijing"}
        )
        
        assert call.id == "call_123"
        assert call.name == "get_weather"
        assert call.arguments == {"city": "Beijing"}
    
    def test_tool_call_execute(self):
        """测试工具调用执行"""
        def handler(city: str) -> str:
            return f"Weather in {city}: Sunny"
        
        call = ToolCall(
            id="call_456",
            name="get_weather",
            arguments={"city": "Tokyo"}
        )
        
        result = call.execute(handler)
        assert result == "Weather in Tokyo: Sunny"


class TestToolResult:
    """ToolResult 测试"""
    
    def test_result_creation(self):
        """测试结果创建"""
        result = ToolResult(
            tool_call_id="call_789",
            name="search",
            arguments={"query": "Python"},
            result="Found 10 results"
        )
        
        assert result.tool_call_id == "call_789"
        assert result.result == "Found 10 results"
        assert result.error is None
    
    def test_result_to_openai_format(self):
        """测试转换为 OpenAI 格式"""
        result = ToolResult(
            tool_call_id="call_abc",
            name="calc",
            arguments={"a": 1, "b": 2},
            result=3
        )
        
        formatted = result.to_openai_format()
        
        assert formatted["role"] == "tool"
        assert formatted["tool_call_id"] == "call_abc"
        assert formatted["name"] == "calc"
        assert formatted["content"] == "3"
    
    def test_error_result(self):
        """测试错误结果"""
        result = ToolResult(
            tool_call_id="call_err",
            name="fail",
            arguments={},
            result=None,
            error="Something went wrong"
        )
        
        assert result.error == "Something went wrong"
        
        formatted = result.to_openai_format()
        assert formatted["content"] == "Something went wrong"


class TestToolRegistry:
    """ToolRegistry 测试"""
    
    def test_register_decorator(self):
        """测试注册装饰器"""
        registry = ToolRegistry()
        
        @registry.register(description="测试函数")
        def test_func(name: str) -> str:
            return f"Hello, {name}"
        
        assert "test_func" in registry._tools
        tool = registry.get_tool("test_func")
        assert tool.description == "测试函数"
    
    def test_register_decorator_no_parens(self):
        """测试无括号的装饰器"""
        registry = ToolRegistry()
        
        @registry.register
        def another_func(x: int) -> int:
            return x * 2
        
        assert "another_func" in registry._tools
    
    def test_add_tool(self):
        """测试添加工具"""
        registry = ToolRegistry()
        
        tool_def = ToolDefinition(
            name="my_tool",
            description="My tool",
            parameters=[]
        )
        
        registry.add_tool(tool_def)
        
        assert registry.get_tool("my_tool") == tool_def
    
    def test_get_openai_tools(self):
        """测试获取 OpenAI 工具列表"""
        registry = ToolRegistry()
        
        @registry.register
        def tool_a(x: int) -> int:
            return x
        
        @registry.register
        def tool_b(y: str) -> str:
            return y
        
        tools = registry.get_openai_tools()
        
        assert len(tools) == 2
        assert tools[0]["type"] == "function"
    
    def test_execute(self):
        """测试执行工具"""
        registry = ToolRegistry()
        
        @registry.register
        def add(a: int, b: int) -> int:
            return a + b
        
        call = ToolCall(
            id="call_1",
            name="add",
            arguments={"a": 3, "b": 5}
        )
        
        result = registry.execute(call)
        assert result.result == 8
        assert result.error is None
    
    def test_execute_unknown_tool(self):
        """测试执行未知工具"""
        registry = ToolRegistry()
        
        call = ToolCall(
            id="call_2",
            name="unknown",
            arguments={}
        )
        
        result = registry.execute(call)
        assert result.error is not None
        assert "Unknown tool" in result.error


class TestToolDecorator:
    """@tool 装饰器测试"""
    
    def test_tool_decorator(self):
        """测试 tool 装饰器"""
        @tool(description="计算两数之和")
        def add_numbers(a: int, b: int) -> int:
            return a + b
        
        assert hasattr(add_numbers, '_tool_definition')
        td = add_numbers._tool_definition
        assert td.name == "add_numbers"
        assert td.description == "计算两数之和"
    
    def test_tool_decorator_with_name(self):
        """测试带名称的 tool 装饰器"""
        @tool(name="custom_name", description="Custom tool")
        def my_func() -> str:
            return "result"
        
        assert my_func._tool_definition.name == "custom_name"


class TestTypeConversion:
    """类型转换测试"""
    
    def test_python_type_to_json(self):
        """测试 Python 类型转 JSON Schema 类型"""
        assert ToolRegistry._python_type_to_json(str) == "string"
        assert ToolRegistry._python_type_to_json(int) == "integer"
        assert ToolRegistry._python_type_to_json(float) == "number"
        assert ToolRegistry._python_type_to_json(bool) == "boolean"
        assert ToolRegistry._python_type_to_json(list) == "array"
        assert ToolRegistry._python_type_to_json(dict) == "object"
