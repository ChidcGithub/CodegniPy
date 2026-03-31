"""
分布式执行模块测试
"""

import pytest
import asyncio

from codegnipy.distributed import (
    QueueBackendType,
    TaskPriority,
    TaskState,
    LoadBalanceStrategy,
    DistributedTask,
    WorkerInfo,
    InMemoryQueueBackend,
    LoadBalancer,
    DistributedScheduler,
    create_queue_backend,
    submit_distributed_task,
)


class TestDistributedTask:
    """分布式任务测试"""
    
    def test_task_creation(self):
        """测试任务创建"""
        task = DistributedTask(
            name="test_task",
            payload={"key": "value"},
            priority=TaskPriority.HIGH,
        )
        
        assert task.name == "test_task"
        assert task.priority == TaskPriority.HIGH
        assert task.state == TaskState.PENDING
        assert task.id is not None
        assert task.retry_count == 0
    
    def test_task_to_dict(self):
        """测试任务序列化"""
        task = DistributedTask(
            name="test_task",
            payload={"key": "value"},
        )
        
        data = task.to_dict()
        
        assert data["name"] == "test_task"
        assert data["payload"] == {"key": "value"}
        assert data["state"] == "pending"
    
    def test_task_from_dict(self):
        """测试任务反序列化"""
        data = {
            "id": "test-id",
            "name": "test_task",
            "payload": {"key": "value"},
            "priority": 10,
            "state": "running",
            "created_at": 1000.0,
            "started_at": 1001.0,
            "completed_at": None,
            "result": None,
            "error": None,
            "retry_count": 0,
            "max_retries": 3,
            "timeout": 300.0,
            "worker_id": None,
            "metadata": {},
        }
        
        task = DistributedTask.from_dict(data)
        
        assert task.id == "test-id"
        assert task.name == "test_task"
        assert task.state == TaskState.RUNNING
        assert task.priority == TaskPriority.HIGH
    
    def test_task_duration(self):
        """测试任务时长计算"""
        task = DistributedTask(
            started_at=1000.0,
            completed_at=1005.0,
        )
        
        assert task.duration == 5.0
        
        # 未完成的任务
        task2 = DistributedTask()
        assert task2.duration is None


class TestWorkerInfo:
    """工作节点信息测试"""
    
    def test_worker_creation(self):
        """测试工作节点创建"""
        worker = WorkerInfo(
            name="worker-1",
            host="localhost",
            port=8080,
            max_tasks=10,
        )
        
        assert worker.name == "worker-1"
        assert worker.max_tasks == 10
        assert worker.current_tasks == 0
    
    def test_worker_load(self):
        """测试工作节点负载计算"""
        worker = WorkerInfo(
            max_tasks=10,
            current_tasks=5,
        )
        
        assert worker.load == 0.5
    
    def test_worker_availability(self):
        """测试工作节点可用性"""
        worker = WorkerInfo(max_tasks=10, current_tasks=5)
        assert worker.is_available
        
        # 超过最大任务数
        worker.current_tasks = 10
        assert not worker.is_available
        
        # 离线状态
        worker.status = "offline"
        worker.current_tasks = 0
        assert not worker.is_available
    
    def test_worker_serialization(self):
        """测试工作节点序列化"""
        worker = WorkerInfo(
            id="worker-1",
            name="test-worker",
            host="localhost",
            port=8080,
        )
        
        data = worker.to_dict()
        
        assert data["id"] == "worker-1"
        assert data["name"] == "test-worker"
        
        worker2 = WorkerInfo.from_dict(data)
        assert worker2.id == worker.id
        assert worker2.name == worker.name


class TestInMemoryQueueBackend:
    """内存队列后端测试"""
    
    @pytest.fixture
    def backend(self):
        return InMemoryQueueBackend()
    
    @pytest.mark.asyncio
    async def test_enqueue_dequeue(self, backend):
        """测试入队出队"""
        await backend.connect()
        
        task = DistributedTask(name="test_task", payload={"key": "value"})
        
        # 入队
        result = await backend.enqueue(task)
        assert result
        
        # 检查队列长度
        length = await backend.get_queue_length()
        assert length == 1
        
        # 出队
        dequeued = await backend.dequeue(timeout=1.0)
        assert dequeued is not None
        assert dequeued.name == "test_task"
        assert dequeued.state == TaskState.RUNNING
        
        await backend.disconnect()
    
    @pytest.mark.asyncio
    async def test_priority_queue(self, backend):
        """测试优先级队列"""
        await backend.connect()
        
        # 按不同优先级入队
        low_task = DistributedTask(name="low", priority=TaskPriority.LOW)
        high_task = DistributedTask(name="high", priority=TaskPriority.HIGH)
        normal_task = DistributedTask(name="normal", priority=TaskPriority.NORMAL)
        
        await backend.enqueue(low_task)
        await backend.enqueue(high_task)
        await backend.enqueue(normal_task)
        
        # 出队顺序应该是 high -> normal -> low
        first = await backend.dequeue(timeout=1.0)
        assert first.name == "high"
        
        second = await backend.dequeue(timeout=1.0)
        assert second.name == "normal"
        
        third = await backend.dequeue(timeout=1.0)
        assert third.name == "low"
        
        await backend.disconnect()
    
    @pytest.mark.asyncio
    async def test_ack_nack(self, backend):
        """测试确认和拒绝"""
        await backend.connect()
        
        task = DistributedTask(name="test_task")
        await backend.enqueue(task)
        
        dequeued = await backend.dequeue(timeout=1.0)
        
        # 确认任务
        result = await backend.ack(dequeued.id)
        assert result
        
        # 检查任务状态
        completed = await backend.get_task(dequeued.id)
        assert completed.state == TaskState.COMPLETED
        
        await backend.disconnect()
    
    @pytest.mark.asyncio
    async def test_purge_queue(self, backend):
        """测试清空队列"""
        await backend.connect()
        
        for i in range(5):
            task = DistributedTask(name=f"task_{i}")
            await backend.enqueue(task)
        
        count = await backend.purge_queue()
        assert count == 5
        
        length = await backend.get_queue_length()
        assert length == 0
        
        await backend.disconnect()


class TestLoadBalancer:
    """负载均衡器测试"""
    
    @pytest.fixture
    def balancer(self):
        return LoadBalancer(strategy=LoadBalanceStrategy.LEAST_CONNECTIONS)
    
    def test_register_worker(self, balancer):
        """测试注册工作节点"""
        worker = WorkerInfo(id="w1", name="worker-1", max_tasks=10)
        balancer.register_worker(worker)
        
        workers = balancer.get_available_workers()
        assert len(workers) == 1
    
    def test_unregister_worker(self, balancer):
        """测试注销工作节点"""
        worker = WorkerInfo(id="w1", name="worker-1")
        balancer.register_worker(worker)
        balancer.unregister_worker("w1")
        
        workers = balancer.get_available_workers()
        assert len(workers) == 0
    
    def test_select_worker_least_connections(self, balancer):
        """测试最少连接选择"""
        w1 = WorkerInfo(id="w1", max_tasks=10, current_tasks=5)
        w2 = WorkerInfo(id="w2", max_tasks=10, current_tasks=2)
        w3 = WorkerInfo(id="w3", max_tasks=10, current_tasks=8)
        
        balancer.register_worker(w1)
        balancer.register_worker(w2)
        balancer.register_worker(w3)
        
        selected = balancer.select_worker()
        assert selected.id == "w2"  # 最少连接
    
    def test_select_worker_round_robin(self):
        """测试轮询选择"""
        balancer = LoadBalancer(strategy=LoadBalanceStrategy.ROUND_ROBIN)
        
        w1 = WorkerInfo(id="w1", max_tasks=10)
        w2 = WorkerInfo(id="w2", max_tasks=10)
        
        balancer.register_worker(w1)
        balancer.register_worker(w2)
        
        # 轮询选择
        first = balancer.select_worker()
        second = balancer.select_worker()
        third = balancer.select_worker()
        
        # 应该按顺序轮询
        assert first.id == "w1"
        assert second.id == "w2"
        assert third.id == "w1"
    
    def test_select_worker_no_available(self, balancer):
        """测试没有可用工作节点"""
        selected = balancer.select_worker()
        assert selected is None
    
    def test_get_stats(self, balancer):
        """测试获取统计信息"""
        w1 = WorkerInfo(id="w1", max_tasks=10, current_tasks=5)
        w2 = WorkerInfo(id="w2", max_tasks=20, current_tasks=10)
        
        balancer.register_worker(w1)
        balancer.register_worker(w2)
        
        stats = balancer.get_stats()
        
        assert stats["strategy"] == "least_connections"
        assert stats["total_workers"] == 2
        assert stats["available_workers"] == 2
        assert stats["total_capacity"] == 30
        assert stats["current_load"] == 15


class TestDistributedScheduler:
    """分布式调度器测试"""
    
    @pytest.fixture
    def backend(self):
        return InMemoryQueueBackend()
    
    @pytest.mark.asyncio
    async def test_scheduler_submit(self, backend):
        """测试提交任务"""
        scheduler = DistributedScheduler(backend=backend, max_concurrent=5)
        
        task = DistributedTask(
            name="test_task",
            payload={"input": "test"},
        )
        
        task_id = await scheduler.submit(task)
        assert task_id == task.id
        
        # 检查队列
        length = await backend.get_queue_length()
        assert length == 1
    
    @pytest.mark.asyncio
    async def test_scheduler_get_status(self, backend):
        """测试获取任务状态"""
        scheduler = DistributedScheduler(backend=backend)
        
        task = DistributedTask(name="test_task")
        await scheduler.submit(task)
        
        status = await scheduler.get_task_status(task.id)
        assert status is not None
        assert status.name == "test_task"
    
    @pytest.mark.asyncio
    async def test_scheduler_process_task(self, backend):
        """测试处理任务"""
        scheduler = DistributedScheduler(backend=backend, max_concurrent=1)
        
        # 注册处理器
        async def handler(task: DistributedTask):
            return f"processed: {task.payload.get('input', '')}"
        
        scheduler.register_handler("test_task", handler)
        
        # 提交任务
        task = DistributedTask(
            name="test_task",
            payload={"input": "hello"},
        )
        
        await scheduler.submit(task)
        
        # 模拟处理
        await backend.connect()
        dequeued = await backend.dequeue(timeout=1.0)
        dequeued.worker_id = scheduler.worker_id
        
        # 执行处理器
        result = await handler(dequeued)
        assert result == "processed: hello"
        
        await backend.disconnect()
    
    def test_scheduler_stats(self, backend):
        """测试调度器统计"""
        scheduler = DistributedScheduler(backend=backend, max_concurrent=10)
        scheduler.register_handler("task1", lambda t: None)
        scheduler.register_handler("task2", lambda t: None)
        
        stats = scheduler.get_stats()
        
        assert stats["max_concurrent"] == 10
        assert "task1" in stats["registered_handlers"]
        assert "task2" in stats["registered_handlers"]


class TestConvenienceFunctions:
    """便捷函数测试"""
    
    def test_create_queue_backend(self):
        """测试创建队列后端"""
        memory_backend = create_queue_backend(QueueBackendType.MEMORY)
        assert isinstance(memory_backend, InMemoryQueueBackend)
    
    @pytest.mark.asyncio
    async def test_submit_distributed_task(self):
        """测试提交分布式任务"""
        backend = InMemoryQueueBackend()
        await backend.connect()
        
        task_id = await submit_distributed_task(
            task_name="test_task",
            payload={"key": "value"},
            backend=backend,
            priority=TaskPriority.HIGH,
        )
        
        assert task_id is not None
        
        # 检查任务在队列中
        length = await backend.get_queue_length()
        assert length == 1
        
        await backend.disconnect()
