"""
@cognitive 装饰器模块

让函数由 LLM 实现，而非手写逻辑。
"""

from functools import wraps
from typing import Callable, get_type_hints, Any, Optional
import inspect

from .runtime import cognitive_call, CognitiveContext


def cognitive(func: Callable = None, *, model: Optional[str] = None) -> Callable:
    """
    Decorator: Mark a function as cognitive, implemented by LLM.
    
    The function's docstring serves as prompt template,
    and the signature is used for input validation and output parsing.
    
    Args:
        func: The function to decorate
        model: Specify the model to use (optional)
    
    Examples:
        @cognitive
        def summarize(text: str) -> str:
            '''Summarize the key points of this text in no more than 3 sentences.'''
            pass
        
        @cognitive(model="gpt-4")
        def translate(text: str, target_lang: str = "English") -> str:
            '''Translate the text to {target_lang}.'''
            pass
    """
    def decorator(fn: Callable) -> Callable:
        # 获取函数信息
        sig = inspect.signature(fn)
        hints = get_type_hints(fn) if hasattr(fn, '__annotations__') else {}
        docstring = fn.__doc__ or f"执行函数 {fn.__name__}"
        
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # 绑定参数
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # 构建提示
            prompt = _build_prompt(docstring, bound.arguments)
            
            # 调用 LLM
            result = cognitive_call(prompt, model=model)
            
            # 类型转换（如果指定了返回类型）
            return_type = hints.get('return')
            if return_type and return_type is not str:
                result = _convert_result(result, return_type)
            
            return result
        
        # 标记为认知函数
        wrapper._is_cognitive = True
        wrapper._original_func = fn
        
        return wrapper
    
    # 支持 @cognitive 和 @cognitive(...) 两种用法
    if func is not None:
        return decorator(func)
    return decorator


def _build_prompt(docstring: str, arguments: dict) -> str:
    """
    根据文档字符串和参数构建提示。
    
    支持 {param} 占位符语法。
    """
    prompt = docstring
    
    # 替换占位符
    for key, value in arguments.items():
        placeholder = "{" + key + "}"
        if placeholder in prompt:
            prompt = prompt.replace(placeholder, str(value))
    
    # 如果没有占位符，将参数附加到提示后
    if "{" not in docstring and arguments:
        args_str = "\n".join(f"- {k}: {v}" for k, v in arguments.items())
        prompt = f"{docstring}\n\n参数:\n{args_str}"
    
    return prompt


def _convert_result(result: str, target_type: type) -> Any:
    """
    将 LLM 结果转换为目标类型。
    
    目前支持：
    - str: 直接返回
    - int/float: 尝试解析数字
    - bool: 解析布尔值
    - list: 尝试解析 JSON 数组
    - dict: 尝试解析 JSON 对象
    """
    if target_type is str:
        return result
    
    if target_type is int:
        try:
            return int(result.strip())
        except ValueError:
            # 尝试提取数字
            import re
            match = re.search(r'-?\d+', result)
            if match:
                return int(match.group())
            raise ValueError(f"无法将结果转换为整数: {result}")
    
    if target_type is float:
        try:
            return float(result.strip())
        except ValueError:
            import re
            match = re.search(r'-?\d+\.?\d*', result)
            if match:
                return float(match.group())
            raise ValueError(f"无法将结果转换为浮点数: {result}")
    
    if target_type is bool:
        lower = result.strip().lower()
        if lower in ('true', 'yes', '是', '1', '真'):
            return True
        if lower in ('false', 'no', '否', '0', '假'):
            return False
        raise ValueError(f"无法将结果转换为布尔值: {result}")
    
    if target_type in (list, dict):
        import json
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            # 尝试提取 JSON
            import re
            json_match = re.search(r'[\[{].*[\]}]', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"无法将结果解析为 JSON: {result}")
    
    # 未知类型，返回原字符串
    return result
