"""
Codegnipy Phase 4 测试模块

测试确定性保证功能。
"""

import pytest
import json
from pydantic import BaseModel

from codegnipy.determinism import (
    PrimitiveConstraint,
    EnumConstraint,
    SchemaConstraint,
    ListConstraint,
    ValidationStatus,
    ValidationResult,
    SimulationMode,
    Simulator,
    HallucinationDetector,
    HallucinationCheck
)


class TestPrimitiveConstraint:
    """PrimitiveConstraint 测试"""
    
    def test_string_validation(self):
        """测试字符串验证"""
        constraint = PrimitiveConstraint(str, min_length=2, max_length=10)
        
        # 有效字符串
        result = constraint.validate("hello")
        assert result.status == ValidationStatus.VALID
        assert result.value == "hello"
        
        # 太短
        result = constraint.validate("a")
        assert result.status == ValidationStatus.INVALID
        assert "小于最小长度" in result.errors[0]
        
        # 太长
        result = constraint.validate("a" * 15)
        assert result.status == ValidationStatus.INVALID
    
    def test_integer_validation(self):
        """测试整数验证"""
        constraint = PrimitiveConstraint(int, min_value=0, max_value=100)
        
        # 有效整数
        result = constraint.validate(50)
        assert result.status == ValidationStatus.VALID
        
        # 太小
        result = constraint.validate(-1)
        assert result.status == ValidationStatus.INVALID
        
        # 太大
        result = constraint.validate(101)
        assert result.status == ValidationStatus.INVALID
    
    def test_type_coercion(self):
        """测试类型转换"""
        constraint = PrimitiveConstraint(int)
        
        # 字符串转整数
        result = constraint.validate("42")
        assert result.status == ValidationStatus.VALID
        assert result.value == 42
    
    def test_pattern_validation(self):
        """测试正则模式验证"""
        constraint = PrimitiveConstraint(str, pattern=r'^\d{3}-\d{4}$')
        
        # 匹配模式
        result = constraint.validate("123-4567")
        assert result.status == ValidationStatus.VALID
        
        # 不匹配模式
        result = constraint.validate("abc-defg")
        assert result.status == ValidationStatus.INVALID
    
    def test_to_prompt(self):
        """测试提示生成"""
        constraint = PrimitiveConstraint(int, min_value=0, max_value=100)
        prompt = constraint.to_prompt()
        
        assert "整数" in prompt
        assert "最小值 0" in prompt
        assert "最大值 100" in prompt


class TestEnumConstraint:
    """EnumConstraint 测试"""
    
    def test_valid_value(self):
        """测试有效值"""
        constraint = EnumConstraint(["red", "green", "blue"])
        
        result = constraint.validate("red")
        assert result.status == ValidationStatus.VALID
    
    def test_invalid_value(self):
        """测试无效值"""
        constraint = EnumConstraint(["red", "green", "blue"])
        
        result = constraint.validate("yellow")
        assert result.status == ValidationStatus.INVALID
        assert "不在允许的值中" in result.errors[0]
    
    def test_case_insensitive(self):
        """测试大小写不敏感"""
        constraint = EnumConstraint(["Red", "Green", "Blue"], case_sensitive=False)
        
        result = constraint.validate("RED")
        assert result.status == ValidationStatus.VALID
        assert result.value == "Red"  # 返回原始大小写


class TestSchemaConstraint:
    """SchemaConstraint 测试"""
    
    def test_pydantic_validation(self):
        """测试 Pydantic 模型验证"""
        class Person(BaseModel):
            name: str
            age: int
        
        constraint = SchemaConstraint(Person)
        
        # 有效数据
        result = constraint.validate({"name": "Alice", "age": 30})
        assert result.status == ValidationStatus.VALID
        assert result.value["name"] == "Alice"
        
        # 无效数据
        result = constraint.validate({"name": "Bob"})  # 缺少 age
        assert result.status == ValidationStatus.INVALID
    
    def test_json_string_input(self):
        """测试 JSON 字符串输入"""
        class Item(BaseModel):
            id: int
            title: str
        
        constraint = SchemaConstraint(Item)
        
        result = constraint.validate('{"id": 1, "title": "Test"}')
        assert result.status == ValidationStatus.VALID


class TestListConstraint:
    """ListConstraint 测试"""
    
    def test_list_validation(self):
        """测试列表验证"""
        constraint = ListConstraint(min_length=1, max_length=3)
        
        # 有效列表
        result = constraint.validate([1, 2, 3])
        assert result.status == ValidationStatus.VALID
        
        # 空列表
        result = constraint.validate([])
        assert result.status == ValidationStatus.INVALID
    
    def test_item_constraint(self):
        """测试元素约束"""
        item_constraint = PrimitiveConstraint(int, min_value=0)
        constraint = ListConstraint(item_constraint=item_constraint)
        
        # 所有元素有效
        result = constraint.validate([1, 2, 3])
        assert result.status == ValidationStatus.VALID
        
        # 有无效元素
        result = constraint.validate([1, -1, 3])
        assert result.status == ValidationStatus.INVALID


class TestSimulator:
    """Simulator 测试"""
    
    def test_mock_mode(self):
        """测试模拟模式"""
        simulator = Simulator(mode=SimulationMode.MOCK)
        simulator.set_default_response("Mock response")
        
        response = simulator.get_response("Any prompt")
        assert response == "Mock response"
    
    def test_pattern_mock(self):
        """测试模式匹配模拟"""
        simulator = Simulator(mode=SimulationMode.MOCK)
        simulator.set_mock_response(r"翻译", "Translated text")
        simulator.set_default_response("Default")
        
        response = simulator.get_response("请翻译这段话")
        assert response == "Translated text"
        
        response = simulator.get_response("其他问题")
        assert response == "Default"
    
    def test_record_and_replay(self):
        """测试记录和回放"""
        simulator = Simulator(mode=SimulationMode.REPLAY)
        simulator.record("What is Python?", "Python is a programming language.")
        
        response = simulator.get_response("What is Python?")
        assert "programming language" in response
    
    def test_save_and_load_recordings(self, tmp_path):
        """测试保存和加载记录"""
        import os
        
        simulator1 = Simulator(mode=SimulationMode.RECORD)
        simulator1.record("Q1", "A1")
        simulator1.record("Q2", "A2")
        
        filepath = str(tmp_path / "recordings.json")
        simulator1.save_recordings(filepath)
        
        simulator2 = Simulator(mode=SimulationMode.REPLAY)
        simulator2.load_recordings(filepath)
        
        assert simulator2.get_response("Q1") == "A1"
        assert simulator2.get_response("Q2") == "A2"


class TestHallucinationDetector:
    """HallucinationDetector 测试"""
    
    def test_detection_returns_check_object(self):
        """测试检测返回检查对象"""
        detector = HallucinationDetector()
        
        check = detector.check("这是一段普通文本。")
        assert isinstance(check, HallucinationCheck)
        assert hasattr(check, 'is_hallucination')
        assert hasattr(check, 'confidence')
        assert hasattr(check, 'reasons')
    
    def test_url_detection(self):
        """测试 URL 检测"""
        detector = HallucinationDetector()
        
        check = detector.check("参考链接: https://example.com/fake-article")
        assert any("URL" in r for r in check.reasons)
    
    def test_citation_check(self):
        """测试引用检查"""
        detector = HallucinationDetector()
        
        check = detector.check("研究表明，这种方法很有效。")
        assert any("来源" in r or "研究" in r for r in check.reasons)
    
    def test_safe_content(self):
        """测试安全内容"""
        detector = HallucinationDetector()
        
        check = detector.check("Python 是一种流行的编程语言。")
        # 简单陈述应该不被标记为严重幻觉
        assert isinstance(check.confidence, float)
    
    def test_add_custom_pattern(self):
        """测试添加自定义模式"""
        detector = HallucinationDetector()
        detector.add_pattern(r'\b\d{16}\b', "可能是虚构的信用卡号")
        
        check = detector.check("卡号是 1234567890123456")
        assert any("信用卡号" in r for r in check.reasons)
    
    def test_confidence_calculation(self):
        """测试置信度计算"""
        detector = HallucinationDetector()
        
        # 包含多个可疑内容的文本
        check = detector.check("""
        根据2025年研究报告显示，请访问 https://fake-url.com，
        联系邮箱 test@fake.com，众所周知这个结果毫无疑问。
        """)
        
        # 应该检测到多个问题
        assert check.confidence > 0


class TestValidationResult:
    """ValidationResult 测试"""
    
    def test_result_creation(self):
        """测试结果创建"""
        result = ValidationResult(
            status=ValidationStatus.VALID,
            value="test",
            warnings=["minor warning"]
        )
        
        assert result.status == ValidationStatus.VALID
        assert result.value == "test"
        assert len(result.warnings) == 1


class TestValidationStatus:
    """ValidationStatus 枚举测试"""
    
    def test_status_values(self):
        """测试状态值"""
        assert ValidationStatus.VALID.value == "valid"
        assert ValidationStatus.INVALID.value == "invalid"
        assert ValidationStatus.UNCERTAIN.value == "uncertain"


class TestSimulationMode:
    """SimulationMode 枚举测试"""
    
    def test_mode_values(self):
        """测试模式值"""
        assert SimulationMode.OFF.value == "off"
        assert SimulationMode.MOCK.value == "mock"
        assert SimulationMode.RECORD.value == "record"
        assert SimulationMode.REPLAY.value == "replay"
