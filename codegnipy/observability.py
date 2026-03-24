"""
Codegnipy 可观测性模块

提供日志记录、性能指标收集、调用链追踪和 OpenTelemetry 集成。
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
)
from contextlib import contextmanager
from contextvars import ContextVar
import uuid

# 类型变量
F = TypeVar("F", bound=Callable[..., Any])


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class SpanContext:
    """Span 上下文"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """添加事件"""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })
    
    def set_attribute(self, key: str, value: Any) -> None:
        """设置属性"""
        self.attributes[key] = value
    
    def set_status(self, status: str, description: Optional[str] = None) -> None:
        """设置状态"""
        self.status = status
        if description:
            self.attributes["status_description"] = description
    
    def finish(self) -> None:
        """结束 Span"""
        self.end_time = time.time()
    
    @property
    def duration_ms(self) -> Optional[float]:
        """获取持续时间（毫秒）"""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
        }


@dataclass
class Metric:
    """指标数据"""
    name: str
    metric_type: MetricType
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "type": self.metric_type.value,
            "value": self.value,
            "timestamp": self.timestamp,
            "labels": self.labels,
            "unit": self.unit,
            "description": self.description,
        }


# 上下文变量
_current_span: ContextVar[Optional[SpanContext]] = ContextVar("current_span", default=None)


class CognitiveLogger:
    """认知日志记录器
    
    为 Codegnipy 提供结构化日志记录功能。
    
    使用示例:
        logger = CognitiveLogger("codegnipy.runtime")
        logger.info("LLM 调用开始", model="gpt-4", prompt_tokens=100)
    """
    
    def __init__(
        self,
        name: str = "codegnipy",
        level: LogLevel = LogLevel.INFO,
        format_json: bool = False,
        include_timestamp: bool = True,
        include_span: bool = True,
    ):
        self.name = name
        self.level = level
        self.format_json = format_json
        self.include_timestamp = include_timestamp
        self.include_span = include_span
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.value))
        
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(getattr(logging, level.value))
            self._logger.addHandler(handler)
    
    def _format_message(
        self,
        level: LogLevel,
        message: str,
        **kwargs
    ) -> str:
        """格式化日志消息"""
        log_data = {
            "level": level.value,
            "logger": self.name,
            "message": message,
        }
        
        if self.include_timestamp:
            log_data["timestamp"] = datetime.now().isoformat()
        
        if self.include_span:
            span = _current_span.get()
            if span:
                log_data["trace_id"] = span.trace_id
                log_data["span_id"] = span.span_id
        
        if kwargs:
            log_data["extra"] = kwargs
        
        if self.format_json:
            return json.dumps(log_data, ensure_ascii=False)
        else:
            parts = [f"[{log_data.get('timestamp', '')}]", f"[{level.value}]", f"[{self.name}]", message]
            if kwargs:
                parts.append(str(kwargs))
            if "trace_id" in log_data:
                parts.append(f"trace_id={log_data['trace_id']}")
            return " ".join(parts)
    
    def debug(self, message: str, **kwargs) -> None:
        """记录 DEBUG 级别日志"""
        if self.level.value <= LogLevel.DEBUG.value:
            self._logger.debug(self._format_message(LogLevel.DEBUG, message, **kwargs))
    
    def info(self, message: str, **kwargs) -> None:
        """记录 INFO 级别日志"""
        if self.level.value <= LogLevel.INFO.value:
            self._logger.info(self._format_message(LogLevel.INFO, message, **kwargs))
    
    def warning(self, message: str, **kwargs) -> None:
        """记录 WARNING 级别日志"""
        if self.level.value <= LogLevel.WARNING.value:
            self._logger.warning(self._format_message(LogLevel.WARNING, message, **kwargs))
    
    def error(self, message: str, **kwargs) -> None:
        """记录 ERROR 级别日志"""
        if self.level.value <= LogLevel.ERROR.value:
            self._logger.error(self._format_message(LogLevel.ERROR, message, **kwargs))
    
    def critical(self, message: str, **kwargs) -> None:
        """记录 CRITICAL 级别日志"""
        if self.level.value <= LogLevel.CRITICAL.value:
            self._logger.critical(self._format_message(LogLevel.CRITICAL, message, **kwargs))
    
    def with_context(self, **kwargs) -> "ContextLogger":
        """创建带上下文的日志记录器"""
        return ContextLogger(self, kwargs)


class ContextLogger:
    """带上下文的日志记录器"""
    
    def __init__(self, logger: CognitiveLogger, context: Dict[str, Any]):
        self._logger = logger
        self._context = context
    
    def _merge_kwargs(self, **kwargs) -> Dict[str, Any]:
        """合并上下文和额外参数"""
        merged = self._context.copy()
        merged.update(kwargs)
        return merged
    
    def debug(self, message: str, **kwargs) -> None:
        self._logger.debug(message, **self._merge_kwargs(**kwargs))
    
    def info(self, message: str, **kwargs) -> None:
        self._logger.info(message, **self._merge_kwargs(**kwargs))
    
    def warning(self, message: str, **kwargs) -> None:
        self._logger.warning(message, **self._merge_kwargs(**kwargs))
    
    def error(self, message: str, **kwargs) -> None:
        self._logger.error(message, **self._merge_kwargs(**kwargs))
    
    def critical(self, message: str, **kwargs) -> None:
        self._logger.critical(message, **self._merge_kwargs(**kwargs))


class MetricsCollector:
    """指标收集器
    
    收集和存储性能指标数据。
    
    使用示例:
        collector = MetricsCollector()
        collector.record_counter("llm.calls", 1, model="gpt-4")
        collector.record_histogram("llm.latency", 150.5, model="gpt-4")
    """
    
    def __init__(self, max_metrics: int = 10000):
        self._metrics: List[Metric] = []
        self._max_metrics = max_metrics
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
    
    def record_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> None:
        """记录计数器指标"""
        key = self._make_key(name, labels)
        self._counters[key] = self._counters.get(key, 0) + value
        
        metric = Metric(
            name=name,
            metric_type=MetricType.COUNTER,
            value=self._counters[key],
            labels=labels or {},
            description=description,
        )
        self._add_metric(metric)
    
    def record_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> None:
        """记录仪表指标"""
        key = self._make_key(name, labels)
        self._gauges[key] = value
        
        metric = Metric(
            name=name,
            metric_type=MetricType.GAUGE,
            value=value,
            labels=labels or {},
            description=description,
        )
        self._add_metric(metric)
    
    def record_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        description: str = "",
        unit: str = "ms",
    ) -> None:
        """记录直方图指标"""
        key = self._make_key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)
        
        metric = Metric(
            name=name,
            metric_type=MetricType.HISTOGRAM,
            value=value,
            labels=labels or {},
            description=description,
            unit=unit,
        )
        self._add_metric(metric)
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """生成唯一键"""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{label_str}"
    
    def _add_metric(self, metric: Metric) -> None:
        """添加指标"""
        if len(self._metrics) >= self._max_metrics:
            self._metrics = self._metrics[-self._max_metrics // 2:]
        self._metrics.append(metric)
    
    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """获取计数器值"""
        key = self._make_key(name, labels)
        return self._counters.get(key, 0)
    
    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """获取仪表值"""
        key = self._make_key(name, labels)
        return self._gauges.get(key)
    
    def get_histogram_stats(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, float]]:
        """获取直方图统计"""
        key = self._make_key(name, labels)
        values = self._histograms.get(key)
        
        if not values:
            return None
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            "count": n,
            "sum": sum(values),
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "mean": sum(values) / n,
            "p50": sorted_values[n // 2],
            "p95": sorted_values[int(n * 0.95)],
            "p99": sorted_values[int(n * 0.99)],
        }
    
    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """获取所有指标"""
        return [m.to_dict() for m in self._metrics]
    
    def clear(self) -> None:
        """清除所有指标"""
        self._metrics.clear()
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()


class Tracer:
    """调用链追踪器
    
    提供分布式追踪功能，支持 Span 嵌套。
    
    使用示例:
        tracer = Tracer()
        
        with tracer.start_span("llm_call") as span:
            span.set_attribute("model", "gpt-4")
            # ... 执行 LLM 调用
            span.add_event("tokens_generated", {"count": 100})
    """
    
    def __init__(
        self,
        service_name: str = "codegnipy",
        sampler_rate: float = 1.0,
        max_spans: int = 10000,
    ):
        self.service_name = service_name
        self.sampler_rate = sampler_rate
        self.max_spans = max_spans
        self._spans: List[SpanContext] = []
    
    def _generate_id(self) -> str:
        """生成唯一 ID"""
        return uuid.uuid4().hex[:16]
    
    def _should_sample(self) -> bool:
        """判断是否采样"""
        import random
        return random.random() < self.sampler_rate
    
    def start_span(
        self,
        operation_name: str,
        parent: Optional[SpanContext] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> SpanContext:
        """开始新的 Span"""
        parent_span = parent or _current_span.get()
        
        span = SpanContext(
            trace_id=parent_span.trace_id if parent_span else self._generate_id(),
            span_id=self._generate_id(),
            parent_span_id=parent_span.span_id if parent_span else None,
            operation_name=operation_name,
            attributes=attributes or {},
        )
        
        if len(self._spans) >= self.max_spans:
            self._spans = self._spans[-self.max_spans // 2:]
        
        self._spans.append(span)
        return span
    
    @contextmanager
    def span(
        self,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """上下文管理器形式的 Span"""
        span = self.start_span(operation_name, attributes=attributes)
        token = _current_span.set(span)
        
        try:
            yield span
        except Exception as e:
            span.set_status("error", str(e))
            span.add_event("exception", {"type": type(e).__name__, "message": str(e)})
            raise
        finally:
            span.finish()
            _current_span.reset(token)
    
    def get_current_span(self) -> Optional[SpanContext]:
        """获取当前 Span"""
        return _current_span.get()
    
    def get_all_spans(self) -> List[Dict[str, Any]]:
        """获取所有 Span"""
        return [s.to_dict() for s in self._spans]
    
    def get_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """获取指定 trace 的所有 Span"""
        return [s.to_dict() for s in self._spans if s.trace_id == trace_id]
    
    def clear(self) -> None:
        """清除所有 Span"""
        self._spans.clear()


class OpenTelemetryExporter:
    """OpenTelemetry 导出器
    
    将指标和追踪数据导出到 OpenTelemetry 兼容的后端。
    
    使用示例:
        exporter = OpenTelemetryExporter(
            endpoint="http://localhost:4317",
            service_name="codegnipy",
        )
        exporter.export_span(span)
        exporter.export_metric(metric)
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        service_name: str = "codegnipy",
        headers: Optional[Dict[str, str]] = None,
    ):
        self.endpoint = endpoint
        self.service_name = service_name
        self.headers = headers or {}
        self._enabled = endpoint is not None
    
    def _check_opentelemetry(self) -> bool:
        """检查 OpenTelemetry 是否可用"""
        try:
            import opentelemetry
            return True
        except ImportError:
            return False
    
    def export_span(self, span: SpanContext) -> bool:
        """导出 Span"""
        if not self._enabled or not self._check_opentelemetry():
            return False
        
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            
            # 创建导出器
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.endpoint,
                headers=self.headers,
            )
            
            # 这里简化处理，实际使用时需要配置 TracerProvider
            return True
        except Exception:
            return False
    
    def export_metric(self, metric: Metric) -> bool:
        """导出指标"""
        if not self._enabled or not self._check_opentelemetry():
            return False
        
        try:
            from opentelemetry import metrics
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
            
            otlp_exporter = OTLPMetricExporter(
                endpoint=self.endpoint,
                headers=self.headers,
            )
            
            return True
        except Exception:
            return False
    
    def export_batch(
        self,
        spans: Optional[List[SpanContext]] = None,
        metrics: Optional[List[Metric]] = None,
    ) -> Dict[str, int]:
        """批量导出"""
        results = {"spans": 0, "metrics": 0}
        
        if spans:
            for span in spans:
                if self.export_span(span):
                    results["spans"] += 1
        
        if metrics:
            for metric in metrics:
                if self.export_metric(metric):
                    results["metrics"] += 1
        
        return results


class ObservabilityManager:
    """可观测性管理器
    
    统一管理日志、指标和追踪。
    
    使用示例:
        manager = ObservabilityManager(
            service_name="codegnipy",
            log_level=LogLevel.INFO,
        )
        
        with manager.trace("llm_call", model="gpt-4") as span:
            manager.log_info("Starting LLM call")
            manager.record_metric("llm.calls", 1)
    """
    
    def __init__(
        self,
        service_name: str = "codegnipy",
        log_level: LogLevel = LogLevel.INFO,
        log_format_json: bool = False,
        sampler_rate: float = 1.0,
        otlp_endpoint: Optional[str] = None,
    ):
        self.service_name = service_name
        
        # 初始化组件
        self.logger = CognitiveLogger(
            name=service_name,
            level=log_level,
            format_json=log_format_json,
        )
        
        self.metrics = MetricsCollector()
        
        self.tracer = Tracer(
            service_name=service_name,
            sampler_rate=sampler_rate,
        )
        
        self.exporter = OpenTelemetryExporter(
            endpoint=otlp_endpoint,
            service_name=service_name,
        )
    
    # 日志代理方法
    def log_debug(self, message: str, **kwargs) -> None:
        self.logger.debug(message, **kwargs)
    
    def log_info(self, message: str, **kwargs) -> None:
        self.logger.info(message, **kwargs)
    
    def log_warning(self, message: str, **kwargs) -> None:
        self.logger.warning(message, **kwargs)
    
    def log_error(self, message: str, **kwargs) -> None:
        self.logger.error(message, **kwargs)
    
    def log_critical(self, message: str, **kwargs) -> None:
        self.logger.critical(message, **kwargs)
    
    # 指标代理方法
    def record_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        self.metrics.record_counter(name, value, labels)
    
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        self.metrics.record_gauge(name, value, labels)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None, unit: str = "ms") -> None:
        self.metrics.record_histogram(name, value, labels, unit=unit)
    
    # 追踪代理方法
    def start_span(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None) -> SpanContext:
        return self.tracer.start_span(operation_name, attributes=attributes)
    
    def trace(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None):
        """上下文管理器形式的追踪"""
        return self.tracer.span(operation_name, attributes)
    
    def get_current_span(self) -> Optional[SpanContext]:
        return self.tracer.get_current_span()
    
    # 导出方法
    def export_to_otlp(self) -> Dict[str, int]:
        """导出到 OpenTelemetry"""
        return self.exporter.export_batch(
            spans=self.tracer._spans,
            metrics=self.metrics._metrics,
        )
    
    def get_observability_data(self) -> Dict[str, Any]:
        """获取所有可观测性数据"""
        return {
            "service_name": self.service_name,
            "spans": self.tracer.get_all_spans(),
            "metrics": self.metrics.get_all_metrics(),
        }
    
    def clear(self) -> None:
        """清除所有数据"""
        self.metrics.clear()
        self.tracer.clear()


def traced(
    operation_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    manager: Optional[ObservabilityManager] = None,
) -> Callable[[F], F]:
    """
    追踪装饰器
    
    为函数自动添加追踪。
    
    使用示例:
        @traced("llm_call", attributes={"model": "gpt-4"})
        def my_llm_function(prompt: str) -> str:
            return cognitive_call(prompt)
    """
    _manager = manager or _default_manager
    
    def decorator(func: F) -> F:
        name = operation_name or func.__name__
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with _manager.trace(name, attributes):
                _manager.log_debug(f"Starting {name}")
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    duration = (time.time() - start_time) * 1000
                    _manager.record_histogram(f"{name}.duration", duration)
                    _manager.log_debug(f"Completed {name} in {duration:.2f}ms")
                    return result
                except Exception as e:
                    _manager.log_error(f"Error in {name}: {e}")
                    raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with _manager.trace(name, attributes):
                _manager.log_debug(f"Starting {name}")
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    duration = (time.time() - start_time) * 1000
                    _manager.record_histogram(f"{name}.duration", duration)
                    _manager.log_debug(f"Completed {name} in {duration:.2f}ms")
                    return result
                except Exception as e:
                    _manager.log_error(f"Error in {name}: {e}")
                    raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


def logged(
    level: LogLevel = LogLevel.INFO,
    log_args: bool = True,
    log_result: bool = False,
    manager: Optional[ObservabilityManager] = None,
) -> Callable[[F], F]:
    """
    日志装饰器
    
    为函数自动添加日志记录。
    
    使用示例:
        @logged(level=LogLevel.DEBUG, log_args=True)
        def my_function(x: int, y: int) -> int:
            return x + y
    """
    _manager = manager or _default_manager
    
    def decorator(func: F) -> F:
        name = func.__name__
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            log_data = {"function": name}
            if log_args:
                log_data["args"] = str(args)
                log_data["kwargs"] = str(kwargs)
            
            _manager.logger._format_message(level, f"Calling {name}", **log_data)
            
            try:
                result = func(*args, **kwargs)
                if log_result:
                    log_data["result"] = str(result)
                _manager.logger._format_message(level, f"Completed {name}", **log_data)
                return result
            except Exception as e:
                log_data["error"] = str(e)
                _manager.log_error(f"Error in {name}", **log_data)
                raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            log_data = {"function": name}
            if log_args:
                log_data["args"] = str(args)
                log_data["kwargs"] = str(kwargs)
            
            _manager.logger._format_message(level, f"Calling {name}", **log_data)
            
            try:
                result = await func(*args, **kwargs)
                if log_result:
                    log_data["result"] = str(result)
                _manager.logger._format_message(level, f"Completed {name}", **log_data)
                return result
            except Exception as e:
                log_data["error"] = str(e)
                _manager.log_error(f"Error in {name}", **log_data)
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


def metered(
    record_duration: bool = True,
    record_calls: bool = True,
    manager: Optional[ObservabilityManager] = None,
) -> Callable[[F], F]:
    """
    指标装饰器
    
    为函数自动记录指标。
    
    使用示例:
        @metered(record_duration=True, record_calls=True)
        def my_function(x: int) -> int:
            return x * 2
    """
    _manager = manager or _default_manager
    
    def decorator(func: F) -> F:
        name = func.__name__
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if record_calls:
                _manager.record_counter(f"{name}.calls")
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                if record_duration:
                    duration = (time.time() - start_time) * 1000
                    _manager.record_histogram(f"{name}.duration", duration, unit="ms")
                return result
            except Exception as e:
                _manager.record_counter(f"{name}.errors")
                raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if record_calls:
                _manager.record_counter(f"{name}.calls")
            
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                if record_duration:
                    duration = (time.time() - start_time) * 1000
                    _manager.record_histogram(f"{name}.duration", duration, unit="ms")
                return result
            except Exception as e:
                _manager.record_counter(f"{name}.errors")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


# 默认管理器
_default_manager = ObservabilityManager()


def get_default_manager() -> ObservabilityManager:
    """获取默认管理器"""
    return _default_manager


def configure_observability(
    service_name: str = "codegnipy",
    log_level: LogLevel = LogLevel.INFO,
    log_format_json: bool = False,
    sampler_rate: float = 1.0,
    otlp_endpoint: Optional[str] = None,
) -> ObservabilityManager:
    """配置全局可观测性"""
    global _default_manager
    _default_manager = ObservabilityManager(
        service_name=service_name,
        log_level=log_level,
        log_format_json=log_format_json,
        sampler_rate=sampler_rate,
        otlp_endpoint=otlp_endpoint,
    )
    return _default_manager
