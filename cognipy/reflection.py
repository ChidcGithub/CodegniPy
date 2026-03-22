"""
CogniPy 反思模块

实现反思循环，让 LLM 能够检查和修正自己的输出。
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, List
from enum import Enum

from .runtime import cognitive_call, CognitiveContext


class ReflectionStatus(Enum):
    """反思状态"""
    PASSED = "passed"           # 通过验证
    NEEDS_FIX = "needs_fix"     # 需要修正
    FAILED = "failed"           # 修正失败


@dataclass
class ReflectionResult:
    """反思结果"""
    status: ReflectionStatus
    original_response: str
    corrected_response: Optional[str] = None
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    iterations: int = 1


class Reflector:
    """
    反思器：让 LLM 检查和修正自己的输出

    工作流程:
    1. 生成初始响应
    2. 自我审查，识别问题
    3. 根据问题进行修正
    4. 重复直到通过或达到最大迭代次数
    """

    DEFAULT_CRITIQUE_PROMPT = """请审查以下回答，检查是否存在问题：

原始问题: {prompt}
回答: {response}

请从以下角度检查：
1. 事实准确性：是否有错误或误导性信息
2. 完整性：是否完整回答了问题
3. 清晰度：表达是否清晰易懂
4. 格式：格式是否符合要求

如果存在问题，请列出具体问题。如果回答良好，请回复 "PASSED"。
"""

    DEFAULT_FIX_PROMPT = """请修正以下回答中的问题：

原始问题: {prompt}
原回答: {response}
发现的问题: {issues}

请提供修正后的回答。"""

    def __init__(
        self,
        max_iterations: int = 3,
        critique_prompt: Optional[str] = None,
        fix_prompt: Optional[str] = None,
        validator: Optional[Callable[[str], bool]] = None
    ):
        """
        参数:
            max_iterations: 最大反思迭代次数
            critique_prompt: 审查提示模板
            fix_prompt: 修正提示模板
            validator: 自定义验证函数
        """
        self.max_iterations = max_iterations
        self.critique_prompt = critique_prompt or self.DEFAULT_CRITIQUE_PROMPT
        self.fix_prompt = fix_prompt or self.DEFAULT_FIX_PROMPT
        self.validator = validator

    def reflect(
        self,
        prompt: str,
        initial_response: str,
        context: Optional[CognitiveContext] = None
    ) -> ReflectionResult:
        """
        执行反思循环

        参数:
            prompt: 原始用户提示
            initial_response: 初始响应
            context: 认知上下文
        返回:
            ReflectionResult 对象
        """
        current_response = initial_response
        issues: List[str] = []

        for iteration in range(self.max_iterations):
            # 审查当前响应
            critique = self._critique(prompt, current_response, context)

            if critique.strip().upper().startswith("PASSED"):
                # 通过审查
                return ReflectionResult(
                    status=ReflectionStatus.PASSED,
                    original_response=initial_response,
                    corrected_response=current_response if iteration > 0 else None,
                    iterations=iteration + 1
                )

            # 提取问题
            found_issues = self._extract_issues(critique)
            issues.extend(found_issues)

            # 修正响应
            current_response = self._fix(
                prompt, current_response, found_issues, context
            )

            # 自定义验证
            if self.validator is not None and self.validator(current_response):
                return ReflectionResult(
                    status=ReflectionStatus.PASSED,
                    original_response=initial_response,
                    corrected_response=current_response,
                    issues=issues,
                    iterations=iteration + 1
                )

        # 达到最大迭代次数
        return ReflectionResult(
            status=ReflectionStatus.NEEDS_FIX if current_response != initial_response else ReflectionStatus.FAILED,
            original_response=initial_response,
            corrected_response=current_response,
            issues=issues,
            iterations=self.max_iterations
        )

    def _critique(
        self,
        prompt: str,
        response: str,
        context: Optional[CognitiveContext]
    ) -> str:
        """审查响应"""
        critique_prompt = self.critique_prompt.format(
            prompt=prompt,
            response=response
        )
        return cognitive_call(critique_prompt, context)

    def _fix(
        self,
        prompt: str,
        response: str,
        issues: List[str],
        context: Optional[CognitiveContext]
    ) -> str:
        """修正响应"""
        fix_prompt = self.fix_prompt.format(
            prompt=prompt,
            response=response,
            issues="\n".join(f"- {i}" for i in issues)
        )
        return cognitive_call(fix_prompt, context)

    def _extract_issues(self, critique: str) -> List[str]:
        """从审查结果中提取问题"""
        issues = []
        lines = critique.strip().split("\n")

        for line in lines:
            line = line.strip()
            # 跳过 "PASSED" 标记和空行
            if not line or line.upper().startswith("PASSED"):
                continue

            # 提取以数字或破折号开头的问题
            if line[0].isdigit() or line.startswith("-"):
                # 移除序号前缀
                issue = line.lstrip("0123456789.-) ").strip()
                if issue:
                    issues.append(issue)

        return issues if issues else [critique]


def with_reflection(
    prompt: str,
    context: Optional[CognitiveContext] = None,
    max_iterations: int = 3,
    validator: Optional[Callable[[str], bool]] = None
) -> ReflectionResult:
    """
    带反思的便捷函数

    示例:
        with CognitiveContext(api_key="sk-...") as ctx:
            result = with_reflection(
                "解释量子纠缠",
                context=ctx,
                max_iterations=2
            )
            print(result.corrected_response or result.original_response)
    """
    # 生成初始响应
    initial = cognitive_call(prompt, context)

    # 创建反思器
    reflector = Reflector(max_iterations=max_iterations, validator=validator)

    # 执行反思
    return reflector.reflect(prompt, initial, context)


class ReflectiveCognitiveCall:
    """
    带反思的认知调用类

    使用方式:
        call = ReflectiveCognitiveCall(max_iterations=2)
        with CognitiveContext() as ctx:
            response = call("解释量子纠缠", context=ctx)
    """

    def __init__(self, max_iterations: int = 3, validator: Optional[Callable[[str], bool]] = None):
        self.max_iterations = max_iterations
        self.validator = validator
        self.reflector = Reflector(max_iterations, validator=validator)

    def __call__(self, prompt: str, context: Optional[CognitiveContext] = None) -> str:
        """执行带反思的调用"""
        initial = cognitive_call(prompt, context)
        result = self.reflector.reflect(prompt, initial, context)

        # 返回最佳响应
        if result.status == ReflectionStatus.FAILED:
            return initial  # 失败时返回原始响应
        return result.corrected_response or result.original_response