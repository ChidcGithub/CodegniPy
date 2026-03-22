"""
CogniPy 确定性保证模块

提供类型约束、模拟执行和幻觉检测功能，确保 LLM 输出的可靠性。
"""

import re
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Optional, List, Dict, Type, Callable, 
    Generic, TypeVar, get_origin, get_args, Union, TYPE_CHECKING
)
from pydantic import BaseModel, ValidationError as PydanticValidationError

if TYPE_CHECKING:
    from .runtime import CognitiveContext


T = TypeVar('T')


class ValidationStatus(Enum):
    """验证状态"""
    VALID = "valid"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"


@dataclass
class ValidationResult:
    """验证结果"""
    status: ValidationStatus
    value: Any
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    confidence: float = 1.0


class TypeConstraint(ABC):
    """类型约束抽象基类"""
    
    @abstractmethod
    def validate(self, value: Any) -> ValidationResult:
        """验证值是否符合约束"""
        pass
    
    @abstractmethod
    def to_prompt(self) -> str:
        """生成用于 LLM 提示的约束描述"""
        pass


class PrimitiveConstraint(TypeConstraint):
    """基础类型约束"""
    
    TYPE_MAP = {
        str: "字符串",
        int: "整数",
        float: "浮点数",
        bool: "布尔值",
        list: "列表",
        dict: "字典"
    }
    
    def __init__(self, expected_type: Type, min_value=None, max_value=None, 
                 min_length=None, max_length=None, pattern=None):
        self.expected_type = expected_type
        self.min_value = min_value
        self.max_value = max_value
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = pattern
    
    def validate(self, value: Any) -> ValidationResult:
        errors = []
        warnings = []
        
        # 类型检查
        if not isinstance(value, self.expected_type):
            # 尝试类型转换
            try:
                if self.expected_type == bool:
                    if isinstance(value, str):
                        if value.lower() in ('true', 'yes', '1'):
                            value = True
                        elif value.lower() in ('false', 'no', '0'):
                            value = False
                        else:
                            raise ValueError()
                    else:
                        value = bool(value)
                elif self.expected_type in (int, float):
                    value = self.expected_type(value)
                else:
                    errors.append(f"类型错误: 期望 {self.TYPE_MAP.get(self.expected_type, self.expected_type.__name__)}, 实际 {type(value).__name__}")
            except (ValueError, TypeError):
                errors.append(f"类型错误: 期望 {self.TYPE_MAP.get(self.expected_type, self.expected_type.__name__)}, 实际 {type(value).__name__}")
        
        # 数值范围检查
        if self.expected_type in (int, float) and isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                errors.append(f"值 {value} 小于最小值 {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                errors.append(f"值 {value} 大于最大值 {self.max_value}")
        
        # 长度检查
        if hasattr(value, '__len__'):
            length = len(value)
            if self.min_length is not None and length < self.min_length:
                errors.append(f"长度 {length} 小于最小长度 {self.min_length}")
            if self.max_length is not None and length > self.max_length:
                errors.append(f"长度 {length} 大于最大长度 {self.max_length}")
        
        # 正则模式检查
        if self.pattern and isinstance(value, str):
            if not re.match(self.pattern, value):
                errors.append(f"字符串不匹配模式: {self.pattern}")
        
        status = ValidationStatus.VALID if not errors else ValidationStatus.INVALID
        return ValidationResult(status=status, value=value, errors=errors, warnings=warnings)
    
    def to_prompt(self) -> str:
        desc = f"类型: {self.TYPE_MAP.get(self.expected_type, self.expected_type.__name__)}"
        
        if self.min_value is not None or self.max_value is not None:
            range_desc = []
            if self.min_value is not None:
                range_desc.append(f"最小值 {self.min_value}")
            if self.max_value is not None:
                range_desc.append(f"最大值 {self.max_value}")
            desc += f", {' '.join(range_desc)}"
        
        if self.min_length is not None or self.max_length is not None:
            len_desc = []
            if self.min_length is not None:
                len_desc.append(f"最小长度 {self.min_length}")
            if self.max_length is not None:
                len_desc.append(f"最大长度 {self.max_length}")
            desc += f", {' '.join(len_desc)}"
        
        if self.pattern:
            desc += f", 必须匹配模式: {self.pattern}"
        
        return desc


class EnumConstraint(TypeConstraint):
    """枚举约束"""
    
    def __init__(self, allowed_values: List[Any], case_sensitive: bool = True):
        self.allowed_values = allowed_values
        self.case_sensitive = case_sensitive
    
    def validate(self, value: Any) -> ValidationResult:
        errors = []
        
        check_value = value if self.case_sensitive else str(value).lower()
        check_allowed = self.allowed_values if self.case_sensitive else [str(v).lower() for v in self.allowed_values]
        
        if check_value not in check_allowed:
            errors.append(f"值 '{value}' 不在允许的值中: {self.allowed_values}")
        
        # 返回原始值（匹配正确的大小写）
        if not self.case_sensitive and isinstance(value, str):
            for av in self.allowed_values:
                if str(av).lower() == value.lower():
                    value = av
                    break
        
        status = ValidationStatus.VALID if not errors else ValidationStatus.INVALID
        return ValidationResult(status=status, value=value, errors=errors)
    
    def to_prompt(self) -> str:
        return f"必须是以下值之一: {', '.join(str(v) for v in self.allowed_values)}"


class SchemaConstraint(TypeConstraint):
    """Schema 约束 (使用 Pydantic)"""
    
    def __init__(self, model_class: Type[BaseModel]):
        self.model_class = model_class
    
    def validate(self, value: Any) -> ValidationResult:
        errors = []
        
        try:
            if isinstance(value, str):
                # 尝试解析 JSON
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    errors.append("无法解析 JSON 字符串")
                    return ValidationResult(
                        status=ValidationStatus.INVALID,
                        value=value,
                        errors=errors
                    )
            
            # 使用 Pydantic 验证
            validated = self.model_class.model_validate(value)
            return ValidationResult(
                status=ValidationStatus.VALID,
                value=validated.model_dump()
            )
            
        except PydanticValidationError as e:
            for error in e.errors():
                errors.append(f"{'.'.join(str(loc) for loc in error['loc'])}: {error['msg']}")
            
            return ValidationResult(
                status=ValidationStatus.INVALID,
                value=value,
                errors=errors
            )
    
    def to_prompt(self) -> str:
        schema = self.model_class.model_json_schema()
        return f"必须符合以下 JSON Schema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"


class ListConstraint(TypeConstraint):
    """列表约束"""
    
    def __init__(self, item_constraint: TypeConstraint = None, min_length=None, max_length=None):
        self.item_constraint = item_constraint
        self.min_length = min_length
        self.max_length = max_length
    
    def validate(self, value: Any) -> ValidationResult:
        errors = []
        warnings = []
        
        # 解析 JSON 字符串
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                errors.append("无法解析 JSON 字符串")
                return ValidationResult(status=ValidationStatus.INVALID, value=value, errors=errors)
        
        if not isinstance(value, list):
            errors.append(f"类型错误: 期望列表, 实际 {type(value).__name__}")
            return ValidationResult(status=ValidationStatus.INVALID, value=value, errors=errors)
        
        # 长度检查
        if self.min_length is not None and len(value) < self.min_length:
            errors.append(f"列表长度 {len(value)} 小于最小长度 {self.min_length}")
        if self.max_length is not None and len(value) > self.max_length:
            errors.append(f"列表长度 {len(value)} 大于最大长度 {self.max_length}")
        
        # 元素验证
        if self.item_constraint:
            validated_items = []
            for i, item in enumerate(value):
                result = self.item_constraint.validate(item)
                if result.status == ValidationStatus.INVALID:
                    errors.append(f"索引 {i}: {'; '.join(result.errors)}")
                else:
                    validated_items.append(result.value)
            value = validated_items
        
        status = ValidationStatus.VALID if not errors else ValidationStatus.INVALID
        return ValidationResult(status=status, value=value, errors=errors, warnings=warnings)
    
    def to_prompt(self) -> str:
        desc = "类型: 列表"
        if self.min_length is not None:
            desc += f", 最小长度 {self.min_length}"
        if self.max_length is not None:
            desc += f", 最大长度 {self.max_length}"
        if self.item_constraint:
            desc += f"\n元素约束: {self.item_constraint.to_prompt()}"
        return desc


# ============ 模拟执行模式 ============

class SimulationMode(Enum):
    """模拟模式"""
    OFF = "off"              # 不模拟，真实调用
    MOCK = "mock"            # 使用模拟响应
    RECORD = "record"        # 记录真实响应
    REPLAY = "replay"        # 回放记录的响应


@dataclass
class MockResponse:
    """模拟响应"""
    prompt: str
    response: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class Simulator:
    """
    模拟执行器
    
    用于测试和开发，无需真实调用 LLM。
    """
    
    def __init__(self, mode: SimulationMode = SimulationMode.MOCK):
        self.mode = mode
        self._recordings: List[MockResponse] = []
        self._mock_responses: Dict[str, str] = {}
        self._default_response = "This is a mock response."
    
    def set_mock_response(self, prompt_pattern: str, response: str):
        """设置模拟响应"""
        self._mock_responses[prompt_pattern] = response
    
    def set_default_response(self, response: str):
        """设置默认响应"""
        self._default_response = response
    
    def get_response(self, prompt: str) -> str:
        """获取响应（根据模式）"""
        if self.mode == SimulationMode.OFF:
            raise RuntimeError("模拟器处于关闭状态，不应调用此方法")
        
        if self.mode == SimulationMode.MOCK:
            return self._get_mock_response(prompt)
        
        if self.mode == SimulationMode.REPLAY:
            return self._get_replay_response(prompt)
        
        raise RuntimeError(f"未知模式: {self.mode}")
    
    def _get_mock_response(self, prompt: str) -> str:
        """获取模拟响应"""
        # 查找匹配的模式
        for pattern, response in self._mock_responses.items():
            if re.search(pattern, prompt, re.IGNORECASE):
                return response
        
        # 返回默认响应
        return self._default_response
    
    def _get_replay_response(self, prompt: str) -> str:
        """获取回放响应"""
        for recording in self._recordings:
            if recording.prompt == prompt:
                return recording.response
        
        raise ValueError(f"未找到匹配的回放响应: {prompt[:50]}...")
    
    def record(self, prompt: str, response: str, metadata: Dict[str, Any] = None):
        """记录响应"""
        self._recordings.append(MockResponse(
            prompt=prompt,
            response=response,
            metadata=metadata or {}
        ))
    
    def load_recordings(self, filepath: str):
        """从文件加载记录"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self._recordings = [MockResponse(**r) for r in data]
    
    def save_recordings(self, filepath: str):
        """保存记录到文件"""
        data = [
            {"prompt": r.prompt, "response": r.response, "metadata": r.metadata}
            for r in self._recordings
        ]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def clear_recordings(self):
        """清空记录"""
        self._recordings.clear()


# ============ 幻觉检测 ============

@dataclass
class HallucinationCheck:
    """幻觉检查结果"""
    is_hallucination: bool
    confidence: float
    reasons: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class HallucinationDetector:
    """
    幻觉检测器
    
    检测 LLM 输出中可能存在的幻觉内容。
    """
    
    def __init__(self):
        self._patterns = [
            # 常见幻觉模式
            (r'\b\d{4}年\d{1,2}月\d{1,2}日\b', "可能是虚构的日期"),
            (r'\bhttps?://[^\s]+\b', "可能是虚构的 URL"),
            (r'\b[\w\.-]+@[\w\.-]+\.\w+\b', "可能是虚构的邮箱"),
            (r'研究表明|研究显示|据统计', "未引用具体来源的声明"),
            (r'众所周知|显然|毫无疑问', "可能缺乏证据支持的断言"),
        ]
    
    def check(self, response: str, context: Dict[str, Any] = None) -> HallucinationCheck:
        """
        检查响应中的幻觉
        
        参数:
            response: LLM 响应文本
            context: 可选的上下文信息
        
        返回:
            HallucinationCheck 对象
        """
        reasons = []
        suggestions = []
        hallucination_score = 0.0
        
        # 模式检查
        for pattern, description in self._patterns:
            matches = re.findall(pattern, response)
            if matches:
                reasons.append(f"{description}: 发现 {len(matches)} 处")
                hallucination_score += 0.2 * len(matches)
        
        # 数值一致性检查
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
        if len(numbers) > 5:
            reasons.append("包含大量数字，请验证准确性")
            hallucination_score += 0.1
        
        # 引用检查
        if '引用' in response or '参考' in response:
            if not re.search(r'\[\d+\]|\(\d{4}\)|"([^"]+)"', response):
                reasons.append("提到引用但未提供具体引用格式")
                hallucination_score += 0.15
        
        # 计算置信度
        confidence = min(hallucination_score, 1.0)
        is_hallucination = confidence > 0.3
        
        if is_hallucination:
            suggestions.append("建议验证响应中的具体细节")
            suggestions.append("考虑使用反思循环进行二次确认")
        
        return HallucinationCheck(
            is_hallucination=is_hallucination,
            confidence=confidence,
            reasons=reasons,
            suggestions=suggestions
        )
    
    def add_pattern(self, pattern: str, description: str):
        """添加自定义幻觉检测模式"""
        self._patterns.append((pattern, description))


# ============ 确定性认知调用 ============

def deterministic_call(
    prompt: str,
    constraint: TypeConstraint,
    context: "CognitiveContext" = None,
    *,
    max_attempts: int = 3,
    use_reflection: bool = False,
    simulator: Simulator = None
) -> ValidationResult:
    """
    带类型约束的确定性认知调用
    
    参数:
        prompt: 提示文本
        constraint: 类型约束
        context: 认知上下文
        max_attempts: 最大尝试次数
        use_reflection: 是否使用反思
        simulator: 模拟器（用于测试）
    
    返回:
        ValidationResult 对象
    """
    from .runtime import cognitive_call
    from .reflection import with_reflection, ReflectionStatus
    
    # 构建带约束的提示
    constrained_prompt = f"{prompt}\n\n约束: {constraint.to_prompt()}\n\n请严格按照约束要求回答。"
    
    for attempt in range(max_attempts):
        # 获取响应
        if simulator and simulator.mode != SimulationMode.OFF:
            response = simulator.get_response(prompt)
        else:
            if use_reflection:
                result = with_reflection(constrained_prompt, context)
                response = result.corrected_response or result.original_response
            else:
                response = cognitive_call(constrained_prompt, context)
        
        # 验证响应
        validation = constraint.validate(response)
        
        if validation.status == ValidationStatus.VALID:
            return validation
        
        # 如果验证失败，添加反馈并重试
        if attempt < max_attempts - 1:
            error_feedback = "; ".join(validation.errors)
            constrained_prompt = f"{prompt}\n\n约束: {constraint.to_prompt()}\n\n上次的回答不符合要求，错误: {error_feedback}\n\n请修正后重新回答。"
    
    return validation
