"""
Codegnipy 测试模块
"""

import pytest
import ast
from codegnipy.transformer import CognitiveTransformer, transform_code


class TestTransformer:
    """AST 转换器测试"""
    
    def test_simple_string_prompt(self):
        """测试简单字符串提示转换"""
        source = 'result = ~"你好"'
        tree = transform_code(source)
        
        # 检查转换后的结构
        assign = tree.body[0]
        assert isinstance(assign, ast.Assign)
        assert assign.targets[0].id == "result"
        
        # 应该是函数调用，不是一元操作
        call = assign.value
        assert isinstance(call, ast.Call)
        
        # 检查调用的函数名
        func = call.func
        assert isinstance(func, ast.Attribute)
        assert func.attr == "cognitive_call"
    
    def test_variable_prompt(self):
        """测试变量作为提示"""
        source = '''
prompt = "翻译这段话"
result = ~prompt
'''
        tree = transform_code(source)
        
        # 第二条语句应该是赋值
        assign = tree.body[1]
        call = assign.value
        assert isinstance(call, ast.Call)
        
        # 参数应该是变量名
        arg = call.args[0]
        assert isinstance(arg, ast.Name)
        assert arg.id == "prompt"
    
    def test_multiple_prompts(self):
        """测试多个认知调用"""
        source = '''
a = ~"第一个提示"
b = ~"第二个提示"
'''
        tree = transform_code(source)
        
        # 两个赋值都应该被转换
        for stmt in tree.body:
            assert isinstance(stmt.value, ast.Call)
    
    def test_nested_expression(self):
        """测试嵌套表达式"""
        source = 'result = ~("前缀 " + text + " 后缀")'
        tree = transform_code(source)
        
        assign = tree.body[0]
        call = assign.value
        assert isinstance(call, ast.Call)
        
        # 参数应该是 BinOp
        arg = call.args[0]
        assert isinstance(arg, ast.BinOp)
    
    def test_non_cognitive_code_unchanged(self):
        """测试非认知代码保持不变"""
        source = '''
x = 1 + 2
y = -x
z = not True
'''
        tree = transform_code(source)
        
        # 第一个赋值
        assert isinstance(tree.body[0].value, ast.BinOp)
        
        # 第二个赋值应该是 USub（负号）
        assert isinstance(tree.body[1].value.op, ast.USub)
        
        # 第三个赋值应该是 Not
        assert isinstance(tree.body[2].value.op, ast.Not)


class TestIntegration:
    """集成测试"""
    
    def test_import_codegnipy(self):
        """测试导入模块"""
        import codegnipy
        assert hasattr(codegnipy, "cognitive_call")
        assert hasattr(codegnipy, "CognitiveContext")
        assert hasattr(codegnipy, "cognitive")
    
    def test_decorator_import(self):
        """测试装饰器导入"""
        from codegnipy import cognitive
        assert callable(cognitive)
    
    def test_context_manager(self):
        """测试上下文管理器"""
        from codegnipy import CognitiveContext
        
        with CognitiveContext(model="test-model") as ctx:
            assert ctx.model == "test-model"
            assert CognitiveContext.get_current() is ctx
        
        # 退出后应该没有当前上下文
        assert CognitiveContext.get_current() is None