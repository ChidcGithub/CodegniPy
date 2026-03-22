"""
CogniPy 最小示例

运行方式:
    cognipy run examples/demo.py
    
或:
    python -m cognipy run examples/demo.py

注意: 需要设置 OPENAI_API_KEY 环境变量
"""

import cognipy
from cognipy import cognitive, CognitiveContext


def main():
    # 方式 1: 使用 ~ 操作符（需要通过 cognipy run 执行）
    # result = ~"用一句话解释什么是递归"
    # print(f"LLM 回答: {result}")
    
    # 方式 2: 使用 cognitive_call 直接调用
    with CognitiveContext(model="gpt-4o-mini"):
        result = cognipy.cognitive_call("用一句话解释什么是递归")
        print(f"LLM 回答: {result}")
    
    # 方式 3: 使用 @cognitive 装饰器
    @cognitive
    def summarize(text: str) -> str:
        """总结这段文字的核心观点，不超过两句话。"""
        pass
    
    # 这需要 API key 才能运行
    # summary = summarize("Python 是一种广泛使用的高级编程语言...")
    # print(f"摘要: {summary}")


if __name__ == "__main__":
    main()
