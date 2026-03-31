"""
Codegnipy 安全模块

提供 PII 检测、数据脱敏、输入/输出过滤和审计日志功能。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Optional, List, Dict, Any, Callable, 
    Pattern, Tuple, Set, TYPE_CHECKING
)
import re
import json
import time
import hashlib
import threading
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import asyncio
from functools import wraps

if TYPE_CHECKING:
    from .runtime import CognitiveContext


class PIIType(Enum):
    """PII 类型"""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"  # 社会安全号码
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    BANK_ACCOUNT = "bank_account"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    CUSTOM = "custom"


class FilterAction(Enum):
    """过滤动作"""
    ALLOW = "allow"
    REDACT = "redact"
    REPLACE = "replace"
    BLOCK = "block"
    WARN = "warn"


class AuditEventType(Enum):
    """审计事件类型"""
    LLM_CALL = "llm_call"
    LLM_RESPONSE = "llm_response"
    PII_DETECTED = "pii_detected"
    FILTER_APPLIED = "filter_applied"
    RATE_LIMIT = "rate_limit"
    ACCESS_DENIED = "access_denied"
    CONFIG_CHANGE = "config_change"
    ERROR = "error"


class SeverityLevel(Enum):
    """严重程度"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PIIMatch:
    """PII 匹配结果"""
    type: PIIType
    value: str
    start: int
    end: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def masked_value(self) -> str:
        """获取脱敏后的值"""
        if len(self.value) <= 4:
            return "*" * len(self.value)
        return self.value[:2] + "*" * (len(self.value) - 4) + self.value[-2:]


@dataclass
class FilterResult:
    """过滤结果"""
    original: str
    filtered: str
    action: FilterAction
    matches: List[PIIMatch] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    blocked: bool = False
    reason: Optional[str] = None
    
    @property
    def has_pii(self) -> bool:
        return len(self.matches) > 0


@dataclass
class AuditEvent:
    """审计事件"""
    event_type: AuditEventType
    timestamp: float = field(default_factory=time.time)
    severity: SeverityLevel = SeverityLevel.INFO
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source_ip: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "severity": self.severity.value,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "source_ip": self.source_ip,
            "resource": self.resource,
            "action": self.action,
            "details": self.details,
            "metadata": self.metadata,
        }
    
    def to_json(self) -> str:
        """转换为 JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class PIIPatterns:
    """PII 模式库"""
    
    # 预定义的正则表达式模式
    PATTERNS: Dict[PIIType, List[Pattern]] = {
        PIIType.EMAIL: [
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        ],
        PIIType.PHONE: [
            # 国际格式
            re.compile(r'\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
            # 中国手机号
            re.compile(r'\b1[3-9]\d{9}\b'),
            # 固定电话
            re.compile(r'\b\d{3,4}[-.\s]?\d{7,8}\b'),
        ],
        PIIType.SSN: [
            # 美国社会安全号码
            re.compile(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b'),
            # 中国身份证号
            re.compile(r'\b\d{17}[\dXx]\b'),
        ],
        PIIType.CREDIT_CARD: [
            # Visa
            re.compile(r'\b4\d{12}(\d{3})?\b'),
            # MasterCard
            re.compile(r'\b5[1-5]\d{14}\b'),
            # Amex
            re.compile(r'\b3[47]\d{13}\b'),
            # 通用格式
            re.compile(r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b'),
        ],
        PIIType.IP_ADDRESS: [
            # IPv4
            re.compile(r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'),
            # IPv6 (简化版)
            re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'),
        ],
        PIIType.BANK_ACCOUNT: [
            # 银行账号 (8-20位数字)
            re.compile(r'\b\d{8,20}\b'),
        ],
        PIIType.PASSPORT: [
            # 美国护照
            re.compile(r'\b[A-Z]{1,2}\d{8,9}\b'),
            # 中国护照
            re.compile(r'\b[GE]\d{8}\b'),
        ],
        PIIType.DATE_OF_BIRTH: [
            # YYYY-MM-DD
            re.compile(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'),
            # MM/DD/YYYY
            re.compile(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b'),
        ],
    }
    
    @classmethod
    def get_patterns(cls, pii_type: PIIType) -> List[Pattern]:
        """获取指定类型的模式"""
        return cls.PATTERNS.get(pii_type, [])
    
    @classmethod
    def add_pattern(cls, pii_type: PIIType, pattern: str) -> None:
        """添加自定义模式"""
        if pii_type not in cls.PATTERNS:
            cls.PATTERNS[pii_type] = []
        cls.PATTERNS[pii_type].append(re.compile(pattern))


class PIIDetector:
    """PII 检测器"""
    
    def __init__(
        self,
        enabled_types: Optional[Set[PIIType]] = None,
        min_confidence: float = 0.5,
    ):
        self._enabled_types = enabled_types or set(PIIType) - {PIIType.CUSTOM}
        self._min_confidence = min_confidence
        self._custom_patterns: Dict[PIIType, List[Pattern]] = {}
    
    def add_custom_pattern(
        self,
        name: str,
        pattern: str,
        pii_type: PIIType = PIIType.CUSTOM,
    ) -> None:
        """添加自定义检测模式"""
        if pii_type not in self._custom_patterns:
            self._custom_patterns[pii_type] = []
        self._custom_patterns[pii_type].append(re.compile(pattern))
    
    def detect(self, text: str) -> List[PIIMatch]:
        """检测文本中的 PII"""
        matches = []
        
        for pii_type in self._enabled_types:
            # 检查预定义模式
            for pattern in PIIPatterns.get_patterns(pii_type):
                for match in pattern.finditer(text):
                    matches.append(PIIMatch(
                        type=pii_type,
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9,
                    ))
            
            # 检查自定义模式
            for pattern in self._custom_patterns.get(pii_type, []):
                for match in pattern.finditer(text):
                    matches.append(PIIMatch(
                        type=pii_type,
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95,
                    ))
        
        # 按位置排序并去重
        matches.sort(key=lambda m: m.start)
        
        return self._remove_overlaps(matches)
    
    def _remove_overlaps(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """移除重叠的匹配"""
        if not matches:
            return matches
        
        result = [matches[0]]
        for match in matches[1:]:
            if match.start >= result[-1].end:
                result.append(match)
            elif match.confidence > result[-1].confidence:
                result[-1] = match
        
        return result
    
    def has_pii(self, text: str) -> bool:
        """检查是否包含 PII"""
        return len(self.detect(text)) > 0


class DataMasker:
    """数据脱敏器"""
    
    def __init__(
        self,
        mask_char: str = "*",
        preserve_length: bool = True,
    ):
        self._mask_char = mask_char
        self._preserve_length = preserve_length
    
    def mask(
        self,
        text: str,
        matches: List[PIIMatch],
        strategy: str = "partial",
    ) -> str:
        """
        脱敏文本
        
        策略:
        - full: 完全脱敏
        - partial: 部分脱敏（保留首尾）
        - hash: 哈希脱敏
        """
        if not matches:
            return text
        
        result = list(text)
        
        for match in reversed(matches):  # 从后往前替换
            masked_value = self._mask_value(match.value, strategy)
            result[match.start:match.end] = list(masked_value)
        
        return "".join(result)
    
    def _mask_value(self, value: str, strategy: str) -> str:
        """脱敏单个值"""
        if strategy == "full":
            if self._preserve_length:
                return self._mask_char * len(value)
            return self._mask_char * 4
        
        elif strategy == "partial":
            if len(value) <= 2:
                return self._mask_char * len(value)
            elif len(value) <= 4:
                return value[0] + self._mask_char * (len(value) - 1)
            else:
                visible = min(2, len(value) // 4)
                return (
                    value[:visible] 
                    + self._mask_char * (len(value) - 2 * visible) 
                    + value[-visible:]
                )
        
        elif strategy == "hash":
            hash_value = hashlib.sha256(value.encode()).hexdigest()[:8]
            if self._preserve_length:
                return hash_value[:len(value)]
            return f"[REDACTED:{hash_value}]"
        
        return value


class ContentFilter(ABC):
    """内容过滤器抽象基类"""
    
    @abstractmethod
    def filter(self, text: str) -> FilterResult:
        """过滤文本"""
        pass
    
    @abstractmethod
    def should_block(self, text: str) -> Tuple[bool, Optional[str]]:
        """检查是否应该阻止"""
        pass


class PIIFilter(ContentFilter):
    """PII 过滤器"""
    
    def __init__(
        self,
        detector: PIIDetector,
        masker: DataMasker,
        action: FilterAction = FilterAction.REDACT,
        block_threshold: int = 5,  # 超过此数量的 PII 将阻止
    ):
        self._detector = detector
        self._masker = masker
        self._action = action
        self._block_threshold = block_threshold
    
    def filter(self, text: str) -> FilterResult:
        """过滤 PII"""
        matches = self._detector.detect(text)
        
        if not matches:
            return FilterResult(
                original=text,
                filtered=text,
                action=FilterAction.ALLOW,
            )
        
        # 检查是否应该阻止
        if len(matches) >= self._block_threshold:
            return FilterResult(
                original=text,
                filtered="",
                action=FilterAction.BLOCK,
                matches=matches,
                blocked=True,
                reason=f"检测到过多 PII ({len(matches)} 处)，已阻止",
            )
        
        if self._action == FilterAction.REDACT:
            filtered = self._masker.mask(text, matches)
            return FilterResult(
                original=text,
                filtered=filtered,
                action=FilterAction.REDACT,
                matches=matches,
            )
        
        elif self._action == FilterAction.WARN:
            warnings = [
                f"检测到 {match.type.value}: {match.masked_value}"
                for match in matches
            ]
            return FilterResult(
                original=text,
                filtered=text,
                action=FilterAction.WARN,
                matches=matches,
                warnings=warnings,
            )
        
        return FilterResult(
            original=text,
            filtered=text,
            action=self._action,
            matches=matches,
        )
    
    def should_block(self, text: str) -> Tuple[bool, Optional[str]]:
        matches = self._detector.detect(text)
        if len(matches) >= self._block_threshold:
            return True, f"检测到 {len(matches)} 处 PII"
        return False, None


class KeywordFilter(ContentFilter):
    """关键词过滤器"""
    
    def __init__(
        self,
        blocked_keywords: Optional[Set[str]] = None,
        warned_keywords: Optional[Set[str]] = None,
        case_sensitive: bool = False,
    ):
        self._blocked_keywords = blocked_keywords or set()
        self._warned_keywords = warned_keywords or set()
        self._case_sensitive = case_sensitive
    
    def add_blocked_keyword(self, keyword: str) -> None:
        """添加阻止关键词"""
        self._blocked_keywords.add(
            keyword if self._case_sensitive else keyword.lower()
        )
    
    def add_warned_keyword(self, keyword: str) -> None:
        """添加警告关键词"""
        self._warned_keywords.add(
            keyword if self._case_sensitive else keyword.lower()
        )
    
    def filter(self, text: str) -> FilterResult:
        """过滤关键词"""
        check_text = text if self._case_sensitive else text.lower()
        warnings = []
        
        # 检查警告关键词
        for keyword in self._warned_keywords:
            if keyword in check_text:
                warnings.append(f"检测到警告关键词: {keyword}")
        
        # 检查阻止关键词
        for keyword in self._blocked_keywords:
            if keyword in check_text:
                return FilterResult(
                    original=text,
                    filtered="",
                    action=FilterAction.BLOCK,
                    blocked=True,
                    reason=f"检测到阻止关键词: {keyword}",
                    warnings=warnings,
                )
        
        if warnings:
            return FilterResult(
                original=text,
                filtered=text,
                action=FilterAction.WARN,
                warnings=warnings,
            )
        
        return FilterResult(
            original=text,
            filtered=text,
            action=FilterAction.ALLOW,
        )
    
    def should_block(self, text: str) -> Tuple[bool, Optional[str]]:
        check_text = text if self._case_sensitive else text.lower()
        
        for keyword in self._blocked_keywords:
            if keyword in check_text:
                return True, f"检测到阻止关键词: {keyword}"
        
        return False, None


class CompositeFilter(ContentFilter):
    """组合过滤器"""
    
    def __init__(self, filters: Optional[List[ContentFilter]] = None):
        self._filters = filters or []
    
    def add_filter(self, filter_: ContentFilter) -> None:
        """添加过滤器"""
        self._filters.append(filter_)
    
    def filter(self, text: str) -> FilterResult:
        """依次应用所有过滤器"""
        current_text = text
        all_matches = []
        all_warnings = []
        
        for filter_ in self._filters:
            result = filter_.filter(current_text)
            
            all_matches.extend(result.matches)
            all_warnings.extend(result.warnings)
            
            if result.blocked:
                return FilterResult(
                    original=text,
                    filtered="",
                    action=FilterAction.BLOCK,
                    matches=all_matches,
                    warnings=all_warnings,
                    blocked=True,
                    reason=result.reason,
                )
            
            current_text = result.filtered
        
        action = FilterAction.WARN if all_warnings else FilterAction.ALLOW
        if all_matches:
            action = FilterAction.REDACT
        
        return FilterResult(
            original=text,
            filtered=current_text,
            action=action,
            matches=all_matches,
            warnings=all_warnings,
        )
    
    def should_block(self, text: str) -> Tuple[bool, Optional[str]]:
        for filter_ in self._filters:
            should_block, reason = filter_.should_block(text)
            if should_block:
                return True, reason
        return False, None


class AuditLogger:
    """审计日志记录器"""
    
    def __init__(
        self,
        log_dir: Optional[Path] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        max_files: int = 10,
        flush_interval: float = 5.0,
    ):
        self._log_dir = log_dir
        self._max_file_size = max_file_size
        self._max_files = max_files
        self._flush_interval = flush_interval
        
        self._buffer: List[AuditEvent] = []
        self._buffer_lock = threading.Lock()
        self._current_file: Optional[Path] = None
        self._current_size: int = 0
        
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
    
    def log(self, event: AuditEvent) -> None:
        """记录审计事件"""
        with self._buffer_lock:
            self._buffer.append(event)
            
            # 如果达到刷新阈值，写入文件
            if len(self._buffer) >= 100:
                self._flush()
    
    def log_event(
        self,
        event_type: AuditEventType,
        severity: SeverityLevel = SeverityLevel.INFO,
        **kwargs
    ) -> None:
        """记录审计事件（便捷方法）"""
        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            **kwargs
        )
        self.log(event)
    
    def _flush(self) -> None:
        """刷新缓冲区到文件"""
        if not self._log_dir or not self._buffer:
            return
        
        events = self._buffer
        self._buffer = []
        
        # 写入文件
        log_file = self._get_log_file()
        with open(log_file, "a", encoding="utf-8") as f:
            for event in events:
                f.write(event.to_json() + "\n")
        
        self._current_size = log_file.stat().st_size if log_file.exists() else 0
    
    def _get_log_file(self) -> Path:
        """获取当前日志文件"""
        if self._current_file and self._current_size < self._max_file_size:
            return self._current_file
        
        # 创建新文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_file = self._log_dir / f"audit_{timestamp}.jsonl"
        self._current_size = 0
        
        # 清理旧文件
        self._cleanup_old_files()
        
        return self._current_file
    
    def _cleanup_old_files(self) -> None:
        """清理旧日志文件"""
        if not self._log_dir:
            return
        
        log_files = sorted(
            self._log_dir.glob("audit_*.jsonl"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        
        for old_file in log_files[self._max_files:]:
            old_file.unlink()
    
    def get_events(
        self,
        event_type: Optional[AuditEventType] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        severity: Optional[SeverityLevel] = None,
    ) -> List[AuditEvent]:
        """获取审计事件"""
        events = []
        
        if not self._log_dir:
            return events
        
        for log_file in self._log_dir.glob("audit_*.jsonl"):
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        event = AuditEvent(
                            event_type=AuditEventType(data["event_type"]),
                            timestamp=data["timestamp"],
                            severity=SeverityLevel(data["severity"]),
                            user_id=data.get("user_id"),
                            session_id=data.get("session_id"),
                            source_ip=data.get("source_ip"),
                            resource=data.get("resource"),
                            action=data.get("action"),
                            details=data.get("details", {}),
                            metadata=data.get("metadata", {}),
                        )
                        
                        # 应用过滤器
                        if event_type and event.event_type != event_type:
                            continue
                        if start_time and event.timestamp < start_time:
                            continue
                        if end_time and event.timestamp > end_time:
                            continue
                        if severity and event.severity != severity:
                            continue
                        
                        events.append(event)
                        
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        return sorted(events, key=lambda e: e.timestamp)
    
    def close(self) -> None:
        """关闭日志记录器"""
        with self._buffer_lock:
            self._flush()


class RateLimiter:
    """增强版速率限制器"""
    
    def __init__(
        self,
        requests_per_second: Optional[float] = None,
        requests_per_minute: Optional[int] = None,
        requests_per_hour: Optional[int] = None,
        requests_per_day: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
        tokens_per_day: Optional[int] = None,
    ):
        self._limits = {
            "second": requests_per_second,
            "minute": requests_per_minute,
            "hour": requests_per_hour,
            "day": requests_per_day,
            "tokens_minute": tokens_per_minute,
            "tokens_day": tokens_per_day,
        }
        
        self._usage: Dict[str, Dict[str, Any]] = {
            "second": {"count": 0, "reset_time": time.time() + 1},
            "minute": {"count": 0, "tokens": 0, "reset_time": time.time() + 60},
            "hour": {"count": 0, "tokens": 0, "reset_time": time.time() + 3600},
            "day": {"count": 0, "tokens": 0, "reset_time": time.time() + 86400},
        }
        
        self._lock = threading.Lock()
    
    def check(self, tokens: int = 0) -> Tuple[bool, Optional[str]]:
        """
        检查是否允许请求
        
        返回: (allowed, reason_if_blocked)
        """
        with self._lock:
            now = time.time()
            
            # 检查并重置过期的计数器
            for period, usage in self._usage.items():
                if now >= usage["reset_time"]:
                    usage["count"] = 0
                    usage["tokens"] = 0
                    
                    if period == "second":
                        usage["reset_time"] = now + 1
                    elif period == "minute":
                        usage["reset_time"] = now + 60
                    elif period == "hour":
                        usage["reset_time"] = now + 3600
                    elif period == "day":
                        usage["reset_time"] = now + 86400
            
            # 检查请求数限制
            if self._limits["second"] and self._usage["second"]["count"] >= self._limits["second"]:
                return False, "已达到每秒请求限制"
            
            if self._limits["minute"] and self._usage["minute"]["count"] >= self._limits["minute"]:
                return False, "已达到每分钟请求限制"
            
            if self._limits["hour"] and self._usage["hour"]["count"] >= self._limits["hour"]:
                return False, "已达到每小时请求限制"
            
            if self._limits["day"] and self._usage["day"]["count"] >= self._limits["day"]:
                return False, "已达到每日请求限制"
            
            # 检查 token 限制
            if self._limits["tokens_minute"] and self._usage["minute"]["tokens"] + tokens > self._limits["tokens_minute"]:
                return False, "已达到每分钟 token 限制"
            
            if self._limits["tokens_day"] and self._usage["day"]["tokens"] + tokens > self._limits["tokens_day"]:
                return False, "已达到每日 token 限制"
            
            return True, None
    
    def record(self, tokens: int = 0) -> None:
        """记录使用量"""
        with self._lock:
            for usage in self._usage.values():
                usage["count"] += 1
                usage["tokens"] = usage.get("tokens", 0) + tokens
    
    def get_usage(self) -> Dict[str, Dict[str, Any]]:
        """获取使用情况"""
        with self._lock:
            result = {}
            for period, usage in self._usage.items():
                limit = self._limits.get(period)
                if limit:
                    result[period] = {
                        "used": usage.get("count", 0),
                        "limit": limit,
                        "remaining": max(0, limit - usage.get("count", 0)),
                        "reset_time": usage["reset_time"],
                    }
            return result
    
    def reset(self) -> None:
        """重置所有计数器"""
        with self._lock:
            now = time.time()
            for period, usage in self._usage.items():
                usage["count"] = 0
                usage["tokens"] = 0
                if period == "second":
                    usage["reset_time"] = now + 1
                elif period == "minute":
                    usage["reset_time"] = now + 60
                elif period == "hour":
                    usage["reset_time"] = now + 3600
                elif period == "day":
                    usage["reset_time"] = now + 86400


class SecurityManager:
    """安全管理器"""
    
    def __init__(
        self,
        content_filter: Optional[ContentFilter] = None,
        audit_logger: Optional[AuditLogger] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self._content_filter = content_filter
        self._audit_logger = audit_logger
        self._rate_limiter = rate_limiter
    
    def filter_input(self, text: str) -> FilterResult:
        """过滤输入"""
        if self._content_filter:
            result = self._content_filter.filter(text)
            
            if self._audit_logger:
                self._audit_logger.log_event(
                    AuditEventType.FILTER_APPLIED,
                    details={
                        "direction": "input",
                        "action": result.action.value,
                        "has_pii": result.has_pii,
                        "blocked": result.blocked,
                    }
                )
            
            return result
        
        return FilterResult(original=text, filtered=text, action=FilterAction.ALLOW)
    
    def filter_output(self, text: str) -> FilterResult:
        """过滤输出"""
        if self._content_filter:
            result = self._content_filter.filter(text)
            
            if self._audit_logger:
                self._audit_logger.log_event(
                    AuditEventType.FILTER_APPLIED,
                    details={
                        "direction": "output",
                        "action": result.action.value,
                        "has_pii": result.has_pii,
                        "blocked": result.blocked,
                    }
                )
            
            return result
        
        return FilterResult(original=text, filtered=text, action=FilterAction.ALLOW)
    
    def check_rate_limit(self, tokens: int = 0) -> Tuple[bool, Optional[str]]:
        """检查速率限制"""
        if self._rate_limiter:
            allowed, reason = self._rate_limiter.check(tokens)
            
            if not allowed and self._audit_logger:
                self._audit_logger.log_event(
                    AuditEventType.RATE_LIMIT,
                    severity=SeverityLevel.WARNING,
                    details={"reason": reason}
                )
            
            return allowed, reason
        
        return True, None
    
    def record_usage(self, tokens: int = 0) -> None:
        """记录使用量"""
        if self._rate_limiter:
            self._rate_limiter.record(tokens)
    
    def log_audit(self, event: AuditEvent) -> None:
        """记录审计事件"""
        if self._audit_logger:
            self._audit_logger.log(event)
    
    def close(self) -> None:
        """关闭安全管理器"""
        if self._audit_logger:
            self._audit_logger.close()


# 装饰器
def secure(
    security_manager: SecurityManager,
    filter_input: bool = True,
    filter_output: bool = True,
    check_rate_limit: bool = True,
):
    """
    安全装饰器
    
    示例:
        @secure(security_manager)
        async def llm_call(prompt: str) -> str:
            return await call_llm(prompt)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 过滤输入
            if filter_input and args:
                filtered_args = []
                for arg in args:
                    if isinstance(arg, str):
                        result = security_manager.filter_input(arg)
                        if result.blocked:
                            raise ValueError(f"输入被阻止: {result.reason}")
                        filtered_args.append(result.filtered)
                    else:
                        filtered_args.append(arg)
                args = tuple(filtered_args)
            
            # 检查速率限制
            if check_rate_limit:
                allowed, reason = security_manager.check_rate_limit()
                if not allowed:
                    raise RuntimeError(f"速率限制: {reason}")
            
            # 执行函数
            result = await func(*args, **kwargs)
            
            # 过滤输出
            if filter_output and isinstance(result, str):
                filter_result = security_manager.filter_output(result)
                if filter_result.blocked:
                    raise ValueError(f"输出被阻止: {filter_result.reason}")
                return filter_result.filtered
            
            return result
        
        return wrapper
    return decorator


# 便捷函数
def create_default_security_manager(
    enable_pii_detection: bool = True,
    enable_audit_log: bool = True,
    enable_rate_limit: bool = True,
    log_dir: Optional[Path] = None,
    requests_per_minute: int = 60,
) -> SecurityManager:
    """创建默认安全管理器"""
    content_filter = None
    audit_logger = None
    rate_limiter = None
    
    if enable_pii_detection:
        detector = PIIDetector()
        masker = DataMasker()
        content_filter = PIIFilter(detector, masker)
    
    if enable_audit_log and log_dir:
        audit_logger = AuditLogger(log_dir=log_dir)
    
    if enable_rate_limit:
        rate_limiter = RateLimiter(requests_per_minute=requests_per_minute)
    
    return SecurityManager(
        content_filter=content_filter,
        audit_logger=audit_logger,
        rate_limiter=rate_limiter,
    )


def detect_pii(text: str) -> List[PIIMatch]:
    """
    检测文本中的 PII
    
    参数:
        text: 要检测的文本
        
    返回:
        PII 匹配列表
    """
    detector = PIIDetector()
    return detector.detect(text)


def mask_pii(text: str, strategy: str = "partial") -> str:
    """
    脱敏文本中的 PII
    
    参数:
        text: 要脱敏的文本
        strategy: 脱敏策略 (full/partial/hash)
        
    返回:
        脱敏后的文本
    """
    detector = PIIDetector()
    masker = DataMasker()
    matches = detector.detect(text)
    return masker.mask(text, matches, strategy)
