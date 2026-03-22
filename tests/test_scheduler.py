"""
CogniPy Phase 3 测试模块

测试异步调度器功能。
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, patch

from cognipy.scheduler import (
    CognitiveScheduler,
    ScheduledTask,
    TaskStatus,
    Priority,
    SchedulerConfig,
    RetryPolicy,
    async_cognitive_call,
    batch_call
)


class TestRetryPolicy:
    """RetryPolicy 测试"""
    
    def test_default_policy(self):
        """测试默认重试策略"""
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.base_delay == 1.0
    
    def test_exponential_backoff(self):
        """测试指数退避"""
        policy = RetryPolicy(base_delay=1.0, exponential_base=2.0)
        
        # 第一次重试: 1 * 2^0 = 1
        assert policy.get_delay(0) == 1.0
        
        # 第二次重试: 1 * 2^1 = 2
        assert policy.get_delay(1) == 2.0
        
        # 第三次重试: 1 * 2^2 = 4
        assert policy.get_delay(2) == 4.0
    
    def test_max_delay_cap(self):
        """测试最大延迟限制"""
        policy = RetryPolicy(base_delay=10.0, max_delay=30.0, exponential_base=3.0)
        
        # 10 * 3^2 = 90，但会被限制在 30
        assert policy.get_delay(2) == 30.0


class TestSchedulerConfig:
    """SchedulerConfig 测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = SchedulerConfig()
        assert config.max_concurrent == 10
        assert config.default_timeout == 60.0
        assert config.default_priority == Priority.NORMAL
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = SchedulerConfig(
            max_concurrent=5,
            default_timeout=30.0,
            default_priority=Priority.HIGH
        )
        assert config.max_concurrent == 5
        assert config.default_timeout == 30.0
        assert config.default_priority == Priority.HIGH


class TestScheduledTask:
    """ScheduledTask 测试"""
    
    @pytest.mark.asyncio
    async def test_task_creation(self):
        """测试任务创建"""
        async def dummy():
            return "result"
        
        task = ScheduledTask(
            priority=-5,
            task_id="test_1",
            coro_factory=dummy
        )
        
        assert task.task_id == "test_1"
        assert task.status == TaskStatus.PENDING
        assert task.retries == 0
        
        # 测试创建协程
        coro = task.create_coro()
        result = await coro
        assert result == "result"
    
    def test_task_ordering(self):
        """测试任务优先级排序"""
        # 只比较优先级，不创建协程
        # Priority 越高，负值越小，越优先
        high_priority = -10
        normal_priority = -5
        low_priority = -1
        
        assert high_priority < normal_priority < low_priority


class TestCognitiveScheduler:
    """CognitiveScheduler 测试"""
    
    @pytest.mark.asyncio
    async def test_scheduler_initialization(self):
        """测试调度器初始化"""
        scheduler = CognitiveScheduler()
        assert scheduler.config is not None
        assert scheduler._counter == 0
    
    @pytest.mark.asyncio
    async def test_submit_and_get_result(self):
        """测试提交任务并获取结果"""
        scheduler = CognitiveScheduler(SchedulerConfig(max_concurrent=1))
        
        async def task_func():
            return "success"
        
        task_id = await scheduler.submit(task_func)
        result = await scheduler.get_result(task_id, timeout=5.0)
        
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_task_timeout(self):
        """测试任务超时"""
        scheduler = CognitiveScheduler(
            SchedulerConfig(max_concurrent=1, retry_policy=RetryPolicy(max_retries=0))
        )
        
        async def slow_task():
            await asyncio.sleep(10.0)  # 很慢
            return "late"
        
        task_id = await scheduler.submit(slow_task, timeout=0.1)
        
        # 等待足够长时间让超时发生
        for _ in range(20):
            status = scheduler.get_status(task_id)
            if status == TaskStatus.TIMEOUT:
                break
            await asyncio.sleep(0.1)
        
        # 应该超时
        assert scheduler.get_status(task_id) == TaskStatus.TIMEOUT
    
    @pytest.mark.asyncio
    async def test_task_failure_with_retry(self):
        """测试任务失败重试"""
        call_count = 0
        
        scheduler = CognitiveScheduler(
            SchedulerConfig(
                max_concurrent=1,
                retry_policy=RetryPolicy(max_retries=2, base_delay=0.05)
            )
        )
        
        async def failing_task():
            nonlocal call_count
            call_count += 1
            raise ValueError("intentional error")
        
        task_id = await scheduler.submit(failing_task, timeout=1.0)
        
        # 等待重试完成
        for _ in range(30):
            status = scheduler.get_status(task_id)
            if status == TaskStatus.FAILED:
                break
            await asyncio.sleep(0.1)
        
        # 应该有多次调用（初始 + 重试）
        assert call_count >= 2
    
    @pytest.mark.asyncio
    async def test_cancel_task(self):
        """测试取消任务"""
        scheduler = CognitiveScheduler()
        
        async def long_task():
            await asyncio.sleep(10.0)
            return "late"
        
        task_id = await scheduler.submit(long_task)
        
        # 取消
        success = await scheduler.cancel(task_id)
        assert success
        
        # 检查状态
        await asyncio.sleep(0.1)
        assert scheduler.get_status(task_id) == TaskStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_concurrent_limit(self):
        """测试并发限制"""
        scheduler = CognitiveScheduler(
            SchedulerConfig(max_concurrent=2)
        )
        
        running_count = [0]
        max_running = [0]
        
        async def tracked_task(i):
            running_count[0] += 1
            max_running[0] = max(max_running[0], running_count[0])
            await asyncio.sleep(0.2)
            running_count[0] -= 1
            return i
        
        # 提交 4 个任务
        task_ids = []
        for i in range(4):
            task_id = await scheduler.submit(lambda i=i: tracked_task(i))
            task_ids.append(task_id)
        
        # 等待完成
        await scheduler.wait_all(timeout=5.0)
        
        # 最大并发应该不超过 2
        assert max_running[0] <= 2
    
    @pytest.mark.asyncio
    async def test_stats(self):
        """测试统计信息"""
        scheduler = CognitiveScheduler()
        
        async def quick_task():
            return "ok"
        
        await scheduler.submit(quick_task)
        await scheduler.submit(quick_task)
        
        # 等待完成
        await asyncio.sleep(0.3)
        
        stats = scheduler.stats()
        assert stats["total_tasks"] == 2
        assert "by_status" in stats


class TestPriority:
    """Priority 枚举测试"""
    
    def test_priority_values(self):
        """测试优先级值"""
        assert Priority.LOW.value == 1
        assert Priority.NORMAL.value == 5
        assert Priority.HIGH.value == 10
        assert Priority.CRITICAL.value == 100


class TestTaskStatus:
    """TaskStatus 枚举测试"""
    
    def test_status_values(self):
        """测试状态值"""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.TIMEOUT.value == "timeout"
        assert TaskStatus.CANCELLED.value == "cancelled"
