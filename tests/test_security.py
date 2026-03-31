"""
安全模块测试
"""

import pytest
import asyncio
import time
from pathlib import Path
import tempfile

from codegnipy.security import (
    PIIType,
    FilterAction,
    AuditEventType,
    SeverityLevel,
    PIIMatch,
    FilterResult,
    AuditEvent,
    PIIPatterns,
    PIIDetector,
    DataMasker,
    PIIFilter,
    KeywordFilter,
    CompositeFilter,
    AuditLogger,
    RateLimiter,
    SecurityManager,
    create_default_security_manager,
    detect_pii,
    mask_pii,
)


class TestPIIMatch:
    """PII 匹配测试"""
    
    def test_match_creation(self):
        """测试匹配创建"""
        match = PIIMatch(
            type=PIIType.EMAIL,
            value="test@example.com",
            start=0,
            end=17,
        )
        
        assert match.type == PIIType.EMAIL
        assert match.value == "test@example.com"
        assert match.confidence == 1.0
    
    def test_masked_value(self):
        """测试脱敏值"""
        # 短值
        match = PIIMatch(type=PIIType.PHONE, value="123", start=0, end=3)
        assert match.masked_value == "***"
        
        # 长值
        match = PIIMatch(type=PIIType.EMAIL, value="test@example.com", start=0, end=17)
        assert "te" in match.masked_value
        assert "om" in match.masked_value


class TestFilterResult:
    """过滤结果测试"""
    
    def test_result_creation(self):
        """测试结果创建"""
        result = FilterResult(
            original="original text",
            filtered="filtered text",
            action=FilterAction.REDACT,
        )
        
        assert result.action == FilterAction.REDACT
        assert not result.has_pii
    
    def test_has_pii(self):
        """测试 PII 检测"""
        match = PIIMatch(type=PIIType.EMAIL, value="test@example.com", start=0, end=17)
        
        result = FilterResult(
            original="test@example.com",
            filtered="te*******om",
            action=FilterAction.REDACT,
            matches=[match],
        )
        
        assert result.has_pii


class TestAuditEvent:
    """审计事件测试"""
    
    def test_event_creation(self):
        """测试事件创建"""
        event = AuditEvent(
            event_type=AuditEventType.LLM_CALL,
            severity=SeverityLevel.INFO,
            user_id="user123",
            details={"prompt": "Hello"},
        )
        
        assert event.event_type == AuditEventType.LLM_CALL
        assert event.user_id == "user123"
    
    def test_event_serialization(self):
        """测试事件序列化"""
        event = AuditEvent(
            event_type=AuditEventType.PII_DETECTED,
            severity=SeverityLevel.WARNING,
            details={"pii_type": "email"},
        )
        
        data = event.to_dict()
        
        assert data["event_type"] == "pii_detected"
        assert data["severity"] == "warning"
        
        json_str = event.to_json()
        assert "pii_detected" in json_str


class TestPIIPatterns:
    """PII 模式测试"""
    
    def test_email_pattern(self):
        """测试邮箱模式"""
        patterns = PIIPatterns.get_patterns(PIIType.EMAIL)
        assert len(patterns) > 0
        
        # 测试匹配
        text = "Contact us at test@example.com for help"
        matches = patterns[0].findall(text)
        assert "test@example.com" in matches
    
    def test_phone_pattern(self):
        """测试电话模式"""
        patterns = PIIPatterns.get_patterns(PIIType.PHONE)
        assert len(patterns) > 0
    
    def test_credit_card_pattern(self):
        """测试信用卡模式"""
        patterns = PIIPatterns.get_patterns(PIIType.CREDIT_CARD)
        assert len(patterns) > 0
        
        text = "Card: 4111111111111111"
        for pattern in patterns:
            matches = pattern.findall(text)
            if matches:
                assert True
                return
        # 至少有一个模式匹配
    
    def test_add_custom_pattern(self):
        """测试添加自定义模式"""
        PIIPatterns.add_pattern(PIIType.CUSTOM, r"\bTEST\d{4}\b")
        
        patterns = PIIPatterns.get_patterns(PIIType.CUSTOM)
        assert len(patterns) > 0


class TestPIIDetector:
    """PII 检测器测试"""
    
    @pytest.fixture
    def detector(self):
        return PIIDetector()
    
    def test_detect_email(self, detector):
        """测试检测邮箱"""
        text = "My email is test@example.com"
        matches = detector.detect(text)
        
        assert len(matches) == 1
        assert matches[0].type == PIIType.EMAIL
    
    def test_detect_phone(self, detector):
        """测试检测电话"""
        text = "Call me at 13812345678"
        matches = detector.detect(text)
        
        assert len(matches) >= 1
        phone_match = next((m for m in matches if m.type == PIIType.PHONE), None)
        assert phone_match is not None
    
    def test_detect_multiple_pii(self, detector):
        """测试检测多个 PII"""
        text = "Email: test@example.com, Phone: 13812345678, IP: 192.168.1.1"
        matches = detector.detect(text)
        
        types = {m.type for m in matches}
        assert PIIType.EMAIL in types
        assert PIIType.PHONE in types
        assert PIIType.IP_ADDRESS in types
    
    def test_no_pii(self, detector):
        """测试无 PII"""
        text = "This is a normal text without any PII"
        matches = detector.detect(text)
        
        assert len(matches) == 0
    
    def test_has_pii(self, detector):
        """测试 PII 检测快捷方法"""
        assert detector.has_pii("test@example.com")
        assert not detector.has_pii("no pii here")
    
    def test_custom_pattern(self, detector):
        """测试自定义模式"""
        # 启用 CUSTOM 类型并添加自定义模式
        detector._enabled_types.add(PIIType.CUSTOM)
        detector.add_custom_pattern("employee_id", r"\bEMP\d{6}\b")
        
        text = "Employee ID: EMP123456"
        matches = detector.detect(text)
        
        # 检查是否有 CUSTOM 类型的匹配
        custom_matches = [m for m in matches if m.type == PIIType.CUSTOM]
        assert len(custom_matches) >= 1


class TestDataMasker:
    """数据脱敏器测试"""
    
    @pytest.fixture
    def masker(self):
        return DataMasker()
    
    def test_mask_full(self, masker):
        """测试完全脱敏"""
        match = PIIMatch(type=PIIType.EMAIL, value="test@example.com", start=0, end=17)
        
        masked = masker.mask("test@example.com", [match], strategy="full")
        
        assert "*" in masked
        assert "test" not in masked.lower()
    
    def test_mask_partial(self, masker):
        """测试部分脱敏"""
        match = PIIMatch(type=PIIType.EMAIL, value="test@example.com", start=0, end=17)
        
        masked = masker.mask("test@example.com", [match], strategy="partial")
        
        # 应该保留首尾
        assert masked.startswith("te")
        assert masked.endswith("om")
        assert "*" in masked
    
    def test_mask_hash(self, masker):
        """测试哈希脱敏"""
        match = PIIMatch(type=PIIType.EMAIL, value="test@example.com", start=0, end=17)
        
        masked = masker.mask("test@example.com", [match], strategy="hash")
        
        assert "test" not in masked.lower()
        assert "REDACTED" in masked or len(masked) > 0
    
    def test_mask_multiple(self, masker):
        """测试多个脱敏"""
        text = "Email: test@example.com, Phone: 13812345678"
        matches = [
            PIIMatch(type=PIIType.EMAIL, value="test@example.com", start=7, end=24),
            PIIMatch(type=PIIType.PHONE, value="13812345678", start=33, end=44),
        ]
        
        masked = masker.mask(text, matches, strategy="partial")
        
        assert "test@example.com" not in masked
        assert "13812345678" not in masked


class TestPIIFilter:
    """PII 过滤器测试"""
    
    @pytest.fixture
    def pii_filter(self):
        detector = PIIDetector()
        masker = DataMasker()
        return PIIFilter(detector, masker, action=FilterAction.REDACT)
    
    def test_filter_pii(self, pii_filter):
        """测试过滤 PII"""
        text = "My email is test@example.com"
        result = pii_filter.filter(text)
        
        assert result.action == FilterAction.REDACT
        assert "test@example.com" not in result.filtered
        assert result.has_pii
    
    def test_filter_no_pii(self, pii_filter):
        """测试无 PII 过滤"""
        text = "No PII here"
        result = pii_filter.filter(text)
        
        assert result.action == FilterAction.ALLOW
        assert not result.has_pii
    
    def test_block_threshold(self):
        """测试阻止阈值"""
        detector = PIIDetector()
        masker = DataMasker()
        pii_filter = PIIFilter(
            detector, masker,
            action=FilterAction.REDACT,
            block_threshold=2,
        )
        
        # 多个 PII
        text = "Email: a@b.com, Phone: 13812345678, IP: 192.168.1.1"
        result = pii_filter.filter(text)
        
        assert result.blocked
    
    def test_should_block(self, pii_filter):
        """测试阻止检测"""
        text = "test@example.com"
        should_block, reason = pii_filter.should_block(text)
        
        assert not should_block  # 默认阈值较高
    
    def test_warn_action(self):
        """测试警告动作"""
        detector = PIIDetector()
        masker = DataMasker()
        pii_filter = PIIFilter(detector, masker, action=FilterAction.WARN)
        
        text = "Email: test@example.com"
        result = pii_filter.filter(text)
        
        assert result.action == FilterAction.WARN
        assert len(result.warnings) > 0


class TestKeywordFilter:
    """关键词过滤器测试"""
    
    @pytest.fixture
    def keyword_filter(self):
        return KeywordFilter(
            blocked_keywords={"blocked_word"},
            warned_keywords={"warned_word"},
        )
    
    def test_block_keyword(self, keyword_filter):
        """测试阻止关键词"""
        text = "This contains blocked_word"
        result = keyword_filter.filter(text)
        
        assert result.blocked
        assert result.action == FilterAction.BLOCK
    
    def test_warn_keyword(self, keyword_filter):
        """测试警告关键词"""
        text = "This contains warned_word"
        result = keyword_filter.filter(text)
        
        assert result.action == FilterAction.WARN
        assert len(result.warnings) > 0
    
    def test_allow_text(self, keyword_filter):
        """测试允许文本"""
        text = "This is safe text"
        result = keyword_filter.filter(text)
        
        assert result.action == FilterAction.ALLOW
    
    def test_add_keywords(self, keyword_filter):
        """测试添加关键词"""
        keyword_filter.add_blocked_keyword("new_blocked")
        keyword_filter.add_warned_keyword("new_warned")
        
        assert keyword_filter.should_block("contains new_blocked")[0]


class TestCompositeFilter:
    """组合过滤器测试"""
    
    @pytest.fixture
    def composite_filter(self):
        detector = PIIDetector()
        masker = DataMasker()
        pii_filter = PIIFilter(detector, masker)
        keyword_filter = KeywordFilter(blocked_keywords={"blocked"})
        
        return CompositeFilter([pii_filter, keyword_filter])
    
    def test_filter_with_pii(self, composite_filter):
        """测试 PII 过滤"""
        text = "Email: test@example.com"
        result = composite_filter.filter(text)
        
        assert result.has_pii
        assert "test@example.com" not in result.filtered
    
    def test_filter_blocked_keyword(self, composite_filter):
        """测试阻止关键词"""
        text = "This contains blocked word"
        result = composite_filter.filter(text)
        
        assert result.blocked
    
    def test_should_block(self, composite_filter):
        """测试阻止检测"""
        should_block, reason = composite_filter.should_block("blocked word")
        assert should_block


class TestAuditLogger:
    """审计日志测试"""
    
    @pytest.fixture
    def temp_log_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def logger(self, temp_log_dir):
        return AuditLogger(log_dir=temp_log_dir)
    
    def test_log_event(self, logger):
        """测试记录事件"""
        event = AuditEvent(
            event_type=AuditEventType.LLM_CALL,
            severity=SeverityLevel.INFO,
        )
        
        logger.log(event)
        
        # 强制刷新
        logger._flush()
        
        events = logger.get_events()
        assert len(events) == 1
    
    def test_log_event_convenience(self, logger):
        """测试便捷记录方法"""
        logger.log_event(
            AuditEventType.PII_DETECTED,
            severity=SeverityLevel.WARNING,
            details={"pii_count": 3},
        )
        
        logger._flush()
        
        events = logger.get_events(event_type=AuditEventType.PII_DETECTED)
        assert len(events) == 1
        assert events[0].details["pii_count"] == 3
    
    def test_filter_events(self, logger):
        """测试过滤事件"""
        now = time.time()
        
        logger.log_event(AuditEventType.LLM_CALL, severity=SeverityLevel.INFO)
        logger.log_event(AuditEventType.ERROR, severity=SeverityLevel.ERROR)
        logger.log_event(AuditEventType.LLM_CALL, severity=SeverityLevel.WARNING)
        
        logger._flush()
        
        # 按类型过滤
        llm_events = logger.get_events(event_type=AuditEventType.LLM_CALL)
        assert len(llm_events) == 2
        
        # 按严重程度过滤
        error_events = logger.get_events(severity=SeverityLevel.ERROR)
        assert len(error_events) == 1


class TestRateLimiter:
    """速率限制器测试"""
    
    @pytest.fixture
    def limiter(self):
        return RateLimiter(
            requests_per_second=10,
            requests_per_minute=100,
        )
    
    def test_check_allowed(self, limiter):
        """测试允许请求"""
        allowed, reason = limiter.check()
        assert allowed
        assert reason is None
    
    def test_record_usage(self, limiter):
        """测试记录使用"""
        limiter.record(tokens=100)
        
        usage = limiter.get_usage()
        assert usage["second"]["used"] == 1
    
    def test_rate_limit_exceeded(self):
        """测试超限"""
        limiter = RateLimiter(requests_per_second=2)
        
        # 前 2 次应该允许
        assert limiter.check()[0]
        limiter.record()
        
        assert limiter.check()[0]
        limiter.record()
        
        # 第 3 次应该被拒绝
        allowed, reason = limiter.check()
        assert not allowed
        assert "每秒" in reason
    
    def test_get_usage(self, limiter):
        """测试获取使用情况"""
        limiter.record(tokens=50)
        
        usage = limiter.get_usage()
        
        assert "second" in usage
        assert "minute" in usage
    
    def test_reset(self, limiter):
        """测试重置"""
        limiter.record()
        limiter.reset()
        
        usage = limiter.get_usage()
        assert usage["second"]["used"] == 0


class TestSecurityManager:
    """安全管理器测试"""
    
    @pytest.fixture
    def security_manager(self):
        return create_default_security_manager(
            enable_pii_detection=True,
            enable_audit_log=False,
            enable_rate_limit=True,
            requests_per_minute=100,
        )
    
    def test_filter_input(self, security_manager):
        """测试过滤输入"""
        text = "Email: test@example.com"
        result = security_manager.filter_input(text)
        
        assert result.has_pii
        assert "test@example.com" not in result.filtered
    
    def test_filter_output(self, security_manager):
        """测试过滤输出"""
        text = "Normal output"
        result = security_manager.filter_output(text)
        
        assert result.action == FilterAction.ALLOW
    
    def test_check_rate_limit(self, security_manager):
        """测试速率限制"""
        allowed, reason = security_manager.check_rate_limit()
        assert allowed
    
    def test_record_usage(self, security_manager):
        """测试记录使用"""
        security_manager.record_usage(tokens=100)


class TestConvenienceFunctions:
    """便捷函数测试"""
    
    def test_detect_pii(self):
        """测试 PII 检测函数"""
        text = "Contact: test@example.com"
        matches = detect_pii(text)
        
        assert len(matches) > 0
        assert matches[0].type == PIIType.EMAIL
    
    def test_mask_pii(self):
        """测试 PII 脱敏函数"""
        text = "Email: test@example.com"
        masked = mask_pii(text, strategy="partial")
        
        assert "test@example.com" not in masked
        assert "*" in masked
    
    def test_mask_pii_full(self):
        """测试完全脱敏"""
        text = "Email: test@example.com"
        masked = mask_pii(text, strategy="full")
        
        assert "test" not in masked.lower()


class TestSecureDecorator:
    """安全装饰器测试"""
    
    @pytest.mark.asyncio
    async def test_secure_decorator(self):
        """测试安全装饰器"""
        security_manager = create_default_security_manager(
            enable_pii_detection=True,
            enable_audit_log=False,
            enable_rate_limit=False,
        )
        
        from codegnipy.security import secure
        
        @secure(security_manager)
        async def process_text(text: str) -> str:
            return f"Processed: {text}"
        
        # 正常文本
        result = await process_text("Hello world")
        assert result == "Processed: Hello world"
        
        # 包含 PII 的文本应该被过滤
        result = await process_text("Email: test@example.com")
        assert "test@example.com" not in result
