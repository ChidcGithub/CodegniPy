"""
AST 转换器模块

将 `~"prompt"` 语法转换为 `cognitive_call("prompt")` 调用。
"""

import ast
import types
from typing import Optional


class CognitiveTransformer(ast.NodeTransformer):
    """
    AST 转换器：将认知操作符转换为运行时调用。
    
    转换规则:
        ~"prompt"  →  cognitive_call("prompt")
        ~variable  →  cognitive_call(variable)
        ~(expr)    →  cognitive_call(expr)
    """
    
    def __init__(self, context_name: str = "__cognitive_context__"):
        super().__init__()
        self.context_name = context_name

    def _transform_invert(self, node: ast.UnaryOp) -> ast.Call:
        """
        处理 `~` 操作符（一元取反操作符被借用为认知操作符）

        将 ~expr 转换为 cognitive_call(expr)
        """
        # 递归处理嵌套的节点
        operand = self.visit(node.operand)

        # 构建 cognitive_call(...) 调用
        call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="codegnipy", ctx=ast.Load()),
                attr="cognitive_call",
                ctx=ast.Load()
            ),
            args=[operand],
            keywords=[
                ast.keyword(
                    arg="context",
                    value=ast.Name(id=self.context_name, ctx=ast.Load())
                )
            ]
        )

        # 设置行号信息用于调试
        ast.copy_location(call, node)
        ast.fix_missing_locations(call)

        return call

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        """
        处理一元操作

        只有 `~` 操作符需要特殊处理，其他保持原样。
        """
        if isinstance(node.op, ast.Invert):
            return self._transform_invert(node)

        # 其他一元操作符保持原样
        return self.generic_visit(node)


def transform_code(source: str, filename: str = "<codegnipy>") -> ast.Module:
    """
    转换 Codegnipy 源代码，返回转换后的 AST。
    
    参数:
        source: Python 源代码字符串
        filename: 文件名（用于错误信息）
    
    返回:
        转换后的 AST 模块
    
    示例:
        source = 'result = ~"你好"'
        tree = transform_code(source)
        # tree 现在包含: result = codegnipy.cognitive_call("你好", context=__cognitive_context__)
    """
    # 解析源代码
    tree = ast.parse(source, filename=filename)
    
    # 应用转换
    transformer = CognitiveTransformer()
    new_tree = transformer.visit(tree)
    
    # 修复位置信息
    ast.fix_missing_locations(new_tree)
    
    return new_tree


def compile_codegnipy(
    source: str,
    filename: str = "<codegnipy>",
    mode: str = "exec"
) -> types.CodeType:
    """
    编译 Codegnipy 源代码，返回代码对象。
    
    参数:
        source: Python 源代码字符串
        filename: 文件名
        mode: 编译模式 ('exec', 'eval', 'single')
    
    返回:
        编译后的代码对象
    """
    tree = transform_code(source, filename)
    return compile(tree, filename, mode)


def exec_codegnipy(
    source: str,
    globals_: Optional[dict] = None,
    locals_: Optional[dict] = None,
    filename: str = "<codegnipy>"
) -> dict:
    """
    执行 Codegnipy 源代码。

    参数:
        source: Python 源代码字符串
        globals_: 全局命名空间
        locals_: 局部命名空间
        filename: 文件名

    返回:
        执行后的局部命名空间
    """
    from .runtime import CognitiveContext

    if globals_ is None:
        globals_ = {}
    if locals_ is None:
        locals_ = {}

    # 确保 codegnipy 模块可用
    import codegnipy
    globals_['codegnipy'] = codegnipy

    # 创建上下文变量
    globals_['__cognitive_context__'] = CognitiveContext.get_current()

    # 编译并执行
    code = compile_codegnipy(source, filename)
    exec(code, globals_, locals_)

    return locals_