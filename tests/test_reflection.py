"""
Codegnipy 反思模块测试
"""

import pytest
from unittest.mock import patch, MagicMock

from codegnipy.reflection import (
    Reflector,
    ReflectionStatus,
    ReflectionResult,
    with_reflection,
    ReflectiveCognitiveCall,
)


class TestReflector:
    """反思器测试"""

    def test_reflector_init(self):
        """测试反思器初始化"""
        reflector = Reflector()
        assert reflector.max_iterations == 3
        assert reflector.validator is None

    def test_reflector_custom_config(self):
        """测试自定义配置"""
        def custom_validator(response: str) -> bool:
            return "correct" in response.lower()
        
        reflector = Reflector(
            max_iterations=5,
            critique_prompt="Custom critique: {prompt} {response}",
            fix_prompt="Custom fix: {prompt} {response} {issues}",
            validator=custom_validator
        )
        
        assert reflector.max_iterations == 5
        assert reflector.validator is custom_validator

    @patch('codegnipy.reflection.cognitive_call')
    def test_reflect_passed_immediately(self, mock_call):
        """测试立即通过的反思"""
        # 批评立即返回 PASSED
        mock_call.side_effect = ["PASSED"]
        
        reflector = Reflector()
        result = reflector.reflect("Test prompt", "Initial response")
        
        assert result.status == ReflectionStatus.PASSED
        assert result.original_response == "Initial response"
        assert result.corrected_response is None
        assert result.iterations == 1

    @patch('codegnipy.reflection.cognitive_call')
    def test_reflect_needs_fix(self, mock_call):
        """测试需要修正的反思"""
        # 批评 -> 问题列表 -> 修正 -> 批评（通过）
        mock_call.side_effect = [
            "1. Error in response\n2. Missing details",  # 批评
            "Corrected response with all details",  # 修正
            "PASSED"  # 再次批评，通过
        ]
        
        reflector = Reflector()
        result = reflector.reflect("Test prompt", "Initial response")
        
        assert result.status == ReflectionStatus.PASSED
        assert result.corrected_response == "Corrected response with all details"
        assert result.iterations == 2

    @patch('codegnipy.reflection.cognitive_call')
    def test_reflect_max_iterations(self, mock_call):
        """测试达到最大迭代次数"""
        # 所有批评都返回问题
        mock_call.side_effect = [
            "1. Problem found",  # 批评
            "Attempt 1",  # 修正
            "1. Still problems",  # 批评
            "Attempt 2",  # 修正
            "1. More problems",  # 批评
            "Attempt 3",  # 修正
        ]
        
        reflector = Reflector(max_iterations=3)
        result = reflector.reflect("Test prompt", "Initial response")
        
        assert result.status == ReflectionStatus.NEEDS_FIX
        assert result.iterations == 3

    @patch('codegnipy.reflection.cognitive_call')
    def test_reflect_with_custom_validator(self, mock_call):
        """测试自定义验证器"""
        mock_call.side_effect = [
            "1. Some issues",  # 批评
            "Corrected with keyword",  # 修正
        ]
        
        def validator(response: str) -> bool:
            return "keyword" in response
        
        reflector = Reflector(validator=validator)
        result = reflector.reflect("Test prompt", "Initial response")
        
        # 验证器通过了修正后的响应
        assert result.status == ReflectionStatus.PASSED
        assert result.iterations == 1

    def test_extract_issues(self):
        """测试问题提取"""
        reflector = Reflector()
        
        critique = """1. 第一个问题
2. 第二个问题
- 第三个问题
PASSED"""
        
        issues = reflector._extract_issues(critique)
        
        assert len(issues) == 3
        assert "第一个问题" in issues
        assert "第二个问题" in issues
        assert "第三个问题" in issues


class TestReflectionResult:
    """反思结果测试"""

    def test_result_creation(self):
        """测试结果创建"""
        result = ReflectionResult(
            status=ReflectionStatus.PASSED,
            original_response="Original",
            corrected_response="Corrected",
            issues=["Issue 1"],
            suggestions=["Fix 1"],
            iterations=2
        )
        
        assert result.status == ReflectionStatus.PASSED
        assert result.original_response == "Original"
        assert result.corrected_response == "Corrected"
        assert len(result.issues) == 1
        assert len(result.suggestions) == 1
        assert result.iterations == 2

    def test_result_default_values(self):
        """测试默认值"""
        result = ReflectionResult(
            status=ReflectionStatus.FAILED,
            original_response="Original"
        )
        
        assert result.issues == []
        assert result.suggestions == []
        assert result.corrected_response is None


class TestWithReflection:
    """with_reflection 便捷函数测试"""

    @patch('codegnipy.reflection.cognitive_call')
    def test_with_reflection_passed(self, mock_call):
        """测试通过的反思"""
        mock_call.side_effect = [
            "Initial response",  # 初始调用
            "PASSED"  # 批评
        ]
        
        result = with_reflection("Test prompt", max_iterations=2)
        
        assert result.status == ReflectionStatus.PASSED
        assert result.original_response == "Initial response"

    @patch('codegnipy.reflection.cognitive_call')
    def test_with_reflection_with_context(self, mock_call):
        """测试带上下文的反思"""
        from codegnipy import CognitiveContext
        
        mock_call.side_effect = [
            "Response",
            "PASSED"
        ]
        
        with CognitiveContext(api_key="test-key") as ctx:
            result = with_reflection("Test prompt", context=ctx, max_iterations=1)
            
        assert result.status == ReflectionStatus.PASSED

    @patch('codegnipy.reflection.cognitive_call')
    def test_with_reflection_with_validator(self, mock_call):
        """测试带验证器的反思"""
        mock_call.side_effect = [
            "Response with magic",  # 初始调用
            "1. Not good enough",  # 批评
            "Better response with magic word",  # 修正
        ]
        
        def validator(response: str) -> bool:
            return "magic word" in response
        
        result = with_reflection(
            "Test prompt",
            max_iterations=2,
            validator=validator
        )
        
        assert result.status == ReflectionStatus.PASSED


class TestReflectiveCognitiveCall:
    """反思认知调用类测试"""

    @patch('codegnipy.reflection.cognitive_call')
    def test_call_passed(self, mock_call):
        """测试通过的调用"""
        mock_call.side_effect = [
            "Initial response",
            "PASSED"
        ]
        
        caller = ReflectiveCognitiveCall(max_iterations=2)
        result = caller("Test prompt")
        
        assert result == "Initial response"

    @patch('codegnipy.reflection.cognitive_call')
    def test_call_corrected(self, mock_call):
        """测试修正后的调用"""
        mock_call.side_effect = [
            "Initial response",
            "1. Issues found",
            "Corrected response",
            "PASSED"
        ]
        
        caller = ReflectiveCognitiveCall(max_iterations=3)
        result = caller("Test prompt")
        
        assert result == "Corrected response"

    @patch('codegnipy.reflection.cognitive_call')
    def test_call_failed_returns_original(self, mock_call):
        """测试失败时返回原始响应（当修正无法改变响应时）"""
        # ReflectiveCognitiveCall.__call__ 会：
        # 1. 调用 cognitive_call 获取初始响应
        # 2. 然后调用 reflect，其中会：
        #    - 调用 _critique (cognitive_call)
        #    - 调用 _fix (cognitive_call)
        #    - 重复...
        # 3. 如果 FAILED（current_response == initial_response），返回原始响应
        
        # 模拟修正后响应不变（触发 FAILED 状态）
        # 初始响应 + 每次迭代（批评 + 修正）* max_iterations
        # max_iterations = 2: 初始 + 批评 + 修正 + 批评 + 修正 = 5 次
        mock_call.side_effect = [
            "Initial response",  # 初始调用
            "1. Always failing",  # 批评 1
            "Initial response",  # 修正 1 - 返回与初始相同的响应
            "1. Still failing",  # 批评 2
            "Initial response",  # 修正 2 - 返回与初始相同的响应
        ]
        
        caller = ReflectiveCognitiveCall(max_iterations=2)
        result = caller("Test prompt")
        
        # 失败时返回原始响应（因为 current_response == initial_response）
        assert result == "Initial response"

    def test_call_with_validator(self):
        """测试带验证器的调用"""
        def validator(response: str) -> bool:
            return len(response) > 10
        
        caller = ReflectiveCognitiveCall(validator=validator)
        assert caller.validator == validator


class TestReflectionStatus:
    """反思状态枚举测试"""

    def test_status_values(self):
        """测试状态值"""
        assert ReflectionStatus.PASSED.value == "passed"
        assert ReflectionStatus.NEEDS_FIX.value == "needs_fix"
        assert ReflectionStatus.FAILED.value == "failed"
