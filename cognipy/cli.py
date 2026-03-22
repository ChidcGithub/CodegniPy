"""
CogniPy 命令行接口

提供 `cognipy run` 和 `cognipy repl` 命令。
"""

import argparse
import sys
import os
from pathlib import Path

from .transformer import transform_code
from .runtime import CognitiveContext
import cognipy


def create_parser() -> argparse.ArgumentParser:
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        prog="cognipy",
        description="CogniPy - AI 原生的 Python 语言扩展"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # run 命令
    run_parser = subparsers.add_parser("run", help="运行 .py 文件")
    run_parser.add_argument("file", help="要运行的 Python 文件")
    run_parser.add_argument(
        "--model", "-m",
        default="gpt-4o-mini",
        help="使用的 LLM 模型"
    )
    run_parser.add_argument(
        "--api-key", "-k",
        help="API 密钥（也可通过环境变量 OPENAI_API_KEY 设置）"
    )
    
    # repl 命令
    repl_parser = subparsers.add_parser("repl", help="启动交互式 REPL")
    repl_parser.add_argument(
        "--model", "-m",
        default="gpt-4o-mini",
        help="使用的 LLM 模型"
    )
    
    # version 命令
    subparsers.add_parser("version", help="显示版本信息")
    
    return parser


def run_file(filepath: str, model: str, api_key: str = None):
    """运行 CogniPy 文件"""
    path = Path(filepath)
    
    if not path.exists():
        print(f"错误: 文件不存在: {filepath}", file=sys.stderr)
        sys.exit(1)
    
    if not path.suffix == ".py":
        print(f"警告: 文件扩展名不是 .py: {filepath}", file=sys.stderr)
    
    # 读取源代码
    source = path.read_text(encoding="utf-8")
    
    # 创建上下文并执行
    with CognitiveContext(api_key=api_key, model=model):
        # 准备执行环境
        globals_ = {
            "__name__": "__main__",
            "__file__": str(path.absolute()),
            "cognipy": cognipy,
            "__cognitive_context__": CognitiveContext.get_current()
        }
        
        # 转换并编译
        tree = transform_code(source, str(path))
        code = compile(tree, str(path), "exec")
        
        # 执行
        try:
            exec(code, globals_)
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)


def start_repl(model: str):
    """启动交互式 REPL"""
    import code
    
    print("CogniPy REPL")
    print(f"模型: {model}")
    print("输入 Python 代码，~\"prompt\" 语法将调用 LLM")
    print("输入 exit() 或 Ctrl+D 退出\n")
    
    # 创建上下文
    ctx = CognitiveContext(model=model)
    ctx.__enter__()
    
    # 准备 REPL 环境
    local_vars = {
        "cognipy": cognipy,
        "__cognitive_context__": ctx
    }
    
    # 自定义编译函数
    class CognitiveConsole(code.InteractiveConsole):
        def runsource(self, source, filename="<input>", symbol="single"):
            try:
                # 尝试转换
                tree = transform_code(source, filename)
                code_obj = compile(tree, filename, symbol)
            except (SyntaxError, OverflowError) as e:
                self.showsyntaxerror(e)
                return False
            
            # 执行
            try:
                exec(code_obj, self.locals)
            except SystemExit:
                raise
            except:
                self.showtraceback()
            
            return False
    
    console = CognitiveConsole(local_vars)
    console.interact(banner="", exitmsg="再见！")
    
    # 清理
    ctx.__exit__(None, None, None)


def main():
    """主入口"""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command == "run":
        run_file(args.file, args.model, args.api_key)
    elif args.command == "repl":
        start_repl(args.model)
    elif args.command == "version":
        print(f"CogniPy v{cognipy.__version__}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
