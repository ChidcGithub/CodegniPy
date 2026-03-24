"""
测试 codegnipy.observability 模块
"""

import asyncio
import time
import pytest

from codegnipy.observability import (
    LogLevel,
    MetricType,
    SpanContext,
    Metric,
    CognitiveLogger,
    ContextLogger,
    MetricsCollector,
    Tracer,
    OpenTelemetryExporter,
    ObservabilityManager,
    traced,
    logged,
    metered,
    get_default_manager,
    configure_observability,
)


class TestLogLevel:
    """测试 LogLevel 枚举"""
    
    def test_log_level_values(self):
        """测试日志级别值"""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"


class TestMetricType:
    """测试 MetricType 枚举"""
    
    def test_metric_type_values(self):
        """测试指标类型值"""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"


class TestSpanContext:
    """测试 SpanContext"""
    
    def test_span_creation(self):
        """测试 Span 创建"""
        span = SpanContext(
            trace_id="abc123",
            span_id="def456",
            operation_name="test_operation",
        )
        
        assert span.trace_id == "abc123"
        assert span.span_id == "def456"
        assert span.operation_name == "test_operation"
        assert span.status == "ok"
        assert span.end_time is None
    
    def test_span_add_event(self):
        """测试添加事件"""
        span = SpanContext(
            trace_id="abc123",
            span_id="def456",
            operation_name="test",
        )
        
        span.add_event("test_event", {"key": "value"})
        
        assert len(span.events) == 1
        assert span.events[0]["name"] == "test_event"
        assert span.events[0]["attributes"]["key"] == "value"
    
    def test_span_set_attribute(self):
        """测试设置属性"""
        span = SpanContext(trace_id="abc", span_id="def")
        
        span.set_attribute("model", "gpt-4")
        span.set_attribute("tokens", 100)
        
        assert span.attributes["model"] == "gpt-4"
        assert span.attributes["tokens"] == 100
    
    def test_span_set_status(self):
        """测试设置状态"""
        span = SpanContext(trace_id="abc", span_id="def")
        
        span.set_status("error", "Something went wrong")
        
        assert span.status == "error"
        assert span.attributes["status_description"] == "Something went wrong"
    
    def test_span_finish(self):
        """测试结束 Span"""
        span = SpanContext(trace_id="abc", span_id="def")
        
        assert span.end_time is None
        assert span.duration_ms is None
        
        time.sleep(0.01)
        span.finish()
        
        assert span.end_time is not None
        assert span.duration_ms is not None
        assert span.duration_ms >= 10
    
    def test_span_to_dict(self):
        """测试转换为字典"""
        span = SpanContext(
            trace_id="abc",
            span_id="def",
            operation_name="test",
        )
        span.set_attribute("key", "value")
        span.finish()
        
        result = span.to_dict()
        
        assert result["trace_id"] == "abc"
        assert result["span_id"] == "def"
        assert result["operation_name"] == "test"
        assert result["attributes"]["key"] == "value"
        assert result["duration_ms"] is not None


class TestMetric:
    """测试 Metric"""
    
    def test_metric_creation(self):
        """测试指标创建"""
        metric = Metric(
            name="test_metric",
            metric_type=MetricType.COUNTER,
            value=42,
            labels={"model": "gpt-4"},
        )
        
        assert metric.name == "test_metric"
        assert metric.metric_type == MetricType.COUNTER
        assert metric.value == 42
        assert metric.labels["model"] == "gpt-4"
    
    def test_metric_to_dict(self):
        """测试转换为字典"""
        metric = Metric(
            name="latency",
            metric_type=MetricType.HISTOGRAM,
            value=150.5,
            unit="ms",
            description="Response latency",
        )
        
        result = metric.to_dict()
        
        assert result["name"] == "latency"
        assert result["type"] == "histogram"
        assert result["value"] == 150.5
        assert result["unit"] == "ms"
        assert result["description"] == "Response latency"


class TestCognitiveLogger:
    """测试 CognitiveLogger"""
    
    def test_logger_creation(self):
        """测试日志器创建"""
        logger = CognitiveLogger("test_logger")
        
        assert logger.name == "test_logger"
        assert logger.level == LogLevel.INFO
    
    def test_logger_custom_level(self):
        """测试自定义日志级别"""
        logger = CognitiveLogger("test", level=LogLevel.DEBUG)
        
        assert logger.level == LogLevel.DEBUG
    
    def test_logger_format_message(self):
        """测试格式化消息"""
        logger = CognitiveLogger("test", include_timestamp=False, include_span=False)
        
        msg = logger._format_message(LogLevel.INFO, "Test message", model="gpt-4")
        
        assert "[INFO]" in msg
        assert "[test]" in msg
        assert "Test message" in msg
        assert "model" in msg
    
    def test_logger_json_format(self):
        """测试 JSON 格式"""
        logger = CognitiveLogger("test", format_json=True, include_span=False)
        
        msg = logger._format_message(LogLevel.INFO, "Test message", key="value")
        
        import json
        parsed = json.loads(msg)
        
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert parsed["extra"]["key"] == "value"
    
    def test_with_context(self):
        """测试带上下文的日志器"""
        logger = CognitiveLogger("test", include_timestamp=False, include_span=False)
        ctx_logger = logger.with_context(service="codegnipy")
        
        assert isinstance(ctx_logger, ContextLogger)
        assert ctx_logger._context["service"] == "codegnipy"


class TestContextLogger:
    """测试 ContextLogger"""
    
    def test_context_logger_merges_kwargs(self):
        """测试上下文合并"""
        logger = CognitiveLogger("test", include_timestamp=False, include_span=False)
        ctx_logger = logger.with_context(service="codegnipy", version="1.0")
        
        merged = ctx_logger._merge_kwargs(extra="value")
        
        assert merged["service"] == "codegnipy"
        assert merged["version"] == "1.0"
        assert merged["extra"] == "value"


class TestMetricsCollector:
    """测试 MetricsCollector"""
    
    def test_record_counter(self):
        """测试记录计数器"""
        collector = MetricsCollector()
        
        collector.record_counter("requests", 1, labels={"endpoint": "/api"})
        collector.record_counter("requests", 2, labels={"endpoint": "/api"})
        
        assert collector.get_counter("requests", {"endpoint": "/api"}) == 3
    
    def test_record_gauge(self):
        """测试记录仪表"""
        collector = MetricsCollector()
        
        collector.record_gauge("temperature", 25.5)
        collector.record_gauge("temperature", 26.0)
        
        assert collector.get_gauge("temperature") == 26.0
    
    def test_record_histogram(self):
        """测试记录直方图"""
        collector = MetricsCollector()
        
        for val in [10, 20, 30, 40, 50]:
            collector.record_histogram("latency", val)
        
        stats = collector.get_histogram_stats("latency")
        
        assert stats is not None
        assert stats["count"] == 5
        assert stats["min"] == 10
        assert stats["max"] == 50
        assert stats["mean"] == 30
    
    def test_histogram_percentiles(self):
        """测试直方图百分位数"""
        collector = MetricsCollector()
        
        # 添加 100 个值
        for i in range(100):
            collector.record_histogram("latency", i)
        
        stats = collector.get_histogram_stats("latency")
        
        assert stats is not None
        # 由于索引计算方式，允许一定的误差
        assert 48 <= stats["p50"] <= 50
        assert 93 <= stats["p95"] <= 95
        assert 97 <= stats["p99"] <= 99
    
    def test_get_all_metrics(self):
        """测试获取所有指标"""
        collector = MetricsCollector()
        
        collector.record_counter("calls", 1)
        collector.record_gauge("memory", 1024)
        
        metrics = collector.get_all_metrics()
        
        assert len(metrics) == 2
        names = [m["name"] for m in metrics]
        assert "calls" in names
        assert "memory" in names
    
    def test_clear(self):
        """测试清除"""
        collector = MetricsCollector()
        
        collector.record_counter("test", 1)
        collector.record_gauge("temp", 25)
        
        collector.clear()
        
        assert len(collector._metrics) == 0
        assert len(collector._counters) == 0
        assert len(collector._gauges) == 0


class TestTracer:
    """测试 Tracer"""
    
    def test_start_span(self):
        """测试开始 Span"""
        tracer = Tracer()
        
        span = tracer.start_span("test_operation")
        
        assert span.operation_name == "test_operation"
        assert span.trace_id is not None
        assert span.span_id is not None
        assert span.parent_span_id is None
    
    def test_start_span_with_parent(self):
        """测试带父 Span 的情况"""
        tracer = Tracer()
        
        parent = tracer.start_span("parent")
        child = tracer.start_span("child", parent=parent)
        
        assert child.parent_span_id == parent.span_id
        assert child.trace_id == parent.trace_id
    
    def test_span_context_manager(self):
        """测试 Span 上下文管理器"""
        tracer = Tracer()
        
        with tracer.span("test_operation") as span:
            span.set_attribute("key", "value")
        
        assert span.end_time is not None
        assert span.attributes["key"] == "value"
    
    def test_span_exception(self):
        """测试 Span 异常处理"""
        tracer = Tracer()
        
        with pytest.raises(ValueError):
            with tracer.span("test") as span:
                raise ValueError("Test error")
        
        # Span 应该被标记为错误
        assert span.status == "error"
    
    def test_get_current_span(self):
        """测试获取当前 Span"""
        tracer = Tracer()
        
        assert tracer.get_current_span() is None
        
        with tracer.span("test") as span:
            current = tracer.get_current_span()
            assert current is span
    
    def test_get_all_spans(self):
        """测试获取所有 Span"""
        tracer = Tracer()
        
        with tracer.span("op1"):
            pass
        with tracer.span("op2"):
            pass
        
        spans = tracer.get_all_spans()
        
        assert len(spans) == 2
        operations = [s["operation_name"] for s in spans]
        assert "op1" in operations
        assert "op2" in operations
    
    def test_get_trace(self):
        """测试获取指定 trace"""
        tracer = Tracer()
        
        parent = tracer.start_span("parent")
        child = tracer.start_span("child", parent=parent)
        
        parent.finish()
        child.finish()
        
        trace = tracer.get_trace(parent.trace_id)
        
        assert len(trace) == 2


class TestOpenTelemetryExporter:
    """测试 OpenTelemetryExporter"""
    
    def test_exporter_creation(self):
        """测试导出器创建"""
        exporter = OpenTelemetryExporter()
        
        assert exporter.endpoint is None
        assert exporter._enabled is False
    
    def test_exporter_with_endpoint(self):
        """测试带端点的导出器"""
        exporter = OpenTelemetryExporter(endpoint="http://localhost:4317")
        
        assert exporter.endpoint == "http://localhost:4317"
        assert exporter._enabled is True
    
    def test_export_span_disabled(self):
        """测试禁用状态下导出 Span"""
        exporter = OpenTelemetryExporter()
        span = SpanContext(trace_id="abc", span_id="def", operation_name="test")
        
        result = exporter.export_span(span)
        
        assert result is False
    
    def test_export_metric_disabled(self):
        """测试禁用状态下导出指标"""
        exporter = OpenTelemetryExporter()
        metric = Metric(name="test", metric_type=MetricType.COUNTER, value=1)
        
        result = exporter.export_metric(metric)
        
        assert result is False
    
    def test_export_batch_disabled(self):
        """测试禁用状态下批量导出"""
        exporter = OpenTelemetryExporter()
        
        result = exporter.export_batch(
            spans=[SpanContext(trace_id="a", span_id="b")],
            metrics=[Metric(name="test", metric_type=MetricType.COUNTER, value=1)],
        )
        
        assert result["spans"] == 0
        assert result["metrics"] == 0


class TestObservabilityManager:
    """测试 ObservabilityManager"""
    
    def test_manager_creation(self):
        """测试管理器创建"""
        manager = ObservabilityManager(service_name="test_service")
        
        assert manager.service_name == "test_service"
        assert manager.logger is not None
        assert manager.metrics is not None
        assert manager.tracer is not None
    
    def test_log_methods(self):
        """测试日志方法"""
        manager = ObservabilityManager()
        
        # 不应该抛出异常
        manager.log_debug("Debug message")
        manager.log_info("Info message")
        manager.log_warning("Warning message")
        manager.log_error("Error message")
        manager.log_critical("Critical message")
    
    def test_metric_methods(self):
        """测试指标方法"""
        manager = ObservabilityManager()
        
        manager.record_counter("calls", 1)
        manager.record_gauge("memory", 1024)
        manager.record_histogram("latency", 50)
        
        assert manager.metrics.get_counter("calls") == 1
        assert manager.metrics.get_gauge("memory") == 1024
        assert manager.metrics.get_histogram_stats("latency")["count"] == 1
    
    def test_trace_methods(self):
        """测试追踪方法"""
        manager = ObservabilityManager()
        
        with manager.trace("test_operation") as span:
            span.set_attribute("key", "value")
        
        assert span.end_time is not None
        assert span.attributes["key"] == "value"
    
    def test_get_observability_data(self):
        """测试获取可观测性数据"""
        manager = ObservabilityManager()
        
        manager.record_counter("calls", 1)
        with manager.trace("test"):
            pass
        
        data = manager.get_observability_data()
        
        assert data["service_name"] == "codegnipy"
        assert len(data["spans"]) == 1
        assert len(data["metrics"]) == 1
    
    def test_clear(self):
        """测试清除"""
        manager = ObservabilityManager()
        
        manager.record_counter("calls", 1)
        manager.start_span("test")
        
        manager.clear()
        
        assert len(manager.metrics._metrics) == 0
        assert len(manager.tracer._spans) == 0


class TestTracedDecorator:
    """测试 traced 装饰器"""
    
    def test_traced_sync_function(self):
        """测试同步函数追踪"""
        manager = ObservabilityManager()
        
        @traced("custom_operation", manager=manager)
        def my_function(x: int) -> int:
            return x * 2
        
        result = my_function(5)
        
        assert result == 10
        
        # 检查指标
        stats = manager.metrics.get_histogram_stats("custom_operation.duration")
        assert stats is not None
        assert stats["count"] == 1
    
    def test_traced_async_function(self):
        """测试异步函数追踪"""
        manager = ObservabilityManager()
        
        @traced(manager=manager)
        async def async_function(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2
        
        result = asyncio.run(async_function(5))
        
        assert result == 10
        
        stats = manager.metrics.get_histogram_stats("async_function.duration")
        assert stats is not None
    
    def test_traced_with_attributes(self):
        """测试带属性的追踪"""
        manager = ObservabilityManager()
        
        @traced("op", attributes={"custom": "attr"}, manager=manager)
        def my_func():
            return "done"
        
        my_func()
        
        # 应该有 span 记录
        assert len(manager.tracer._spans) >= 1


class TestLoggedDecorator:
    """测试 logged 装饰器"""
    
    def test_logged_sync_function(self):
        """测试同步函数日志"""
        manager = ObservabilityManager()
        
        @logged(manager=manager)
        def my_function(x: int) -> int:
            return x + 1
        
        result = my_function(5)
        
        assert result == 6
    
    def test_logged_async_function(self):
        """测试异步函数日志"""
        manager = ObservabilityManager()
        
        @logged(manager=manager)
        async def async_function(x: int) -> int:
            return x + 1
        
        result = asyncio.run(async_function(5))
        
        assert result == 6
    
    def test_logged_with_args(self):
        """测试带参数日志"""
        manager = ObservabilityManager()
        
        @logged(log_args=True, log_result=True, manager=manager)
        def my_func(a: int, b: int) -> int:
            return a + b
        
        result = my_func(1, 2)
        
        assert result == 3


class TestMeteredDecorator:
    """测试 metered 装饰器"""
    
    def test_metered_sync_function(self):
        """测试同步函数指标"""
        manager = ObservabilityManager()
        
        @metered(manager=manager)
        def my_function(x: int) -> int:
            return x * 2
        
        result = my_function(5)
        
        assert result == 10
        assert manager.metrics.get_counter("my_function.calls") == 1
    
    def test_metered_async_function(self):
        """测试异步函数指标"""
        manager = ObservabilityManager()
        
        @metered(manager=manager)
        async def async_function(x: int) -> int:
            return x * 2
        
        result = asyncio.run(async_function(5))
        
        assert result == 10
        assert manager.metrics.get_counter("async_function.calls") == 1
    
    def test_metered_error_counting(self):
        """测试错误计数"""
        manager = ObservabilityManager()
        
        @metered(manager=manager)
        def error_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            error_function()
        
        assert manager.metrics.get_counter("error_function.errors") == 1


class TestGlobalFunctions:
    """测试全局函数"""
    
    def test_get_default_manager(self):
        """测试获取默认管理器"""
        manager = get_default_manager()
        
        assert manager is not None
        assert isinstance(manager, ObservabilityManager)
    
    def test_configure_observability(self):
        """测试配置可观测性"""
        manager = configure_observability(
            service_name="custom_service",
            log_level=LogLevel.DEBUG,
            log_format_json=True,
        )
        
        assert manager.service_name == "custom_service"
        assert manager.logger.level == LogLevel.DEBUG
        assert manager.logger.format_json is True
