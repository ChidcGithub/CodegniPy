"""
Codegnipy 分布式执行模块

提供分布式任务队列、负载均衡和多节点协调能力。
支持 Redis 和 RabbitMQ 作为消息队列后端。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Optional, List, Dict, Any, Callable, 
    Awaitable, Union, TYPE_CHECKING, Generic, TypeVar
)
import asyncio
import json
import time
import hashlib
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading

if TYPE_CHECKING:
    import redis.asyncio as aioredis
    import aio_pika

T = TypeVar('T')


class QueueBackendType(Enum):
    """队列后端类型"""
    MEMORY = "memory"
    REDIS = "redis"
    RABBITMQ = "rabbitmq"


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


class TaskState(Enum):
    """任务状态"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class LoadBalanceStrategy(Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    RANDOM = "random"
    CONSISTENT_HASH = "consistent_hash"


@dataclass
class DistributedTask(Generic[T]):
    """分布式任务"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    state: TaskState = TaskState.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[T] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 300.0
    worker_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "payload": self.payload,
            "priority": self.priority.value,
            "state": self.state.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "worker_id": self.worker_id,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistributedTask":
        """从字典创建"""
        return cls(
            id=data["id"],
            name=data["name"],
            payload=data["payload"],
            priority=TaskPriority(data["priority"]),
            state=TaskState(data["state"]),
            created_at=data["created_at"],
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            result=data.get("result"),
            error=data.get("error"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            timeout=data.get("timeout", 300.0),
            worker_id=data.get("worker_id"),
            metadata=data.get("metadata", {}),
        )
    
    @property
    def duration(self) -> Optional[float]:
        """任务执行时长"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


@dataclass
class WorkerInfo:
    """工作节点信息"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    host: str = "localhost"
    port: int = 8080
    status: str = "idle"  # idle, busy, offline
    current_tasks: int = 0
    max_tasks: int = 10
    weight: int = 1  # 用于加权负载均衡
    last_heartbeat: float = field(default_factory=time.time)
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def load(self) -> float:
        """当前负载率 (0-1)"""
        return self.current_tasks / self.max_tasks if self.max_tasks > 0 else 0.0
    
    @property
    def is_available(self) -> bool:
        """是否可用"""
        return (
            self.status != "offline" 
            and self.current_tasks < self.max_tasks
            and time.time() - self.last_heartbeat < 60  # 60秒心跳超时
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "status": self.status,
            "current_tasks": self.current_tasks,
            "max_tasks": self.max_tasks,
            "weight": self.weight,
            "last_heartbeat": self.last_heartbeat,
            "capabilities": self.capabilities,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkerInfo":
        """从字典创建"""
        return cls(
            id=data["id"],
            name=data["name"],
            host=data["host"],
            port=data["port"],
            status=data["status"],
            current_tasks=data["current_tasks"],
            max_tasks=data["max_tasks"],
            weight=data.get("weight", 1),
            last_heartbeat=data.get("last_heartbeat", time.time()),
            capabilities=data.get("capabilities", []),
            metadata=data.get("metadata", {}),
        )


class QueueBackend(ABC):
    """队列后端抽象基类"""
    
    @abstractmethod
    async def connect(self) -> None:
        """连接到队列"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """断开连接"""
        pass
    
    @abstractmethod
    async def enqueue(self, task: DistributedTask, queue_name: str = "default") -> bool:
        """入队任务"""
        pass
    
    @abstractmethod
    async def dequeue(self, queue_name: str = "default", timeout: float = 5.0) -> Optional[DistributedTask]:
        """出队任务"""
        pass
    
    @abstractmethod
    async def ack(self, task_id: str, queue_name: str = "default") -> bool:
        """确认任务完成"""
        pass
    
    @abstractmethod
    async def nack(self, task_id: str, queue_name: str = "default", requeue: bool = True) -> bool:
        """任务失败，重新入队"""
        pass
    
    @abstractmethod
    async def get_queue_length(self, queue_name: str = "default") -> int:
        """获取队列长度"""
        pass
    
    @abstractmethod
    async def purge_queue(self, queue_name: str = "default") -> int:
        """清空队列"""
        pass
    
    @abstractmethod
    async def update_task(self, task: DistributedTask) -> bool:
        """更新任务状态"""
        pass
    
    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[DistributedTask]:
        """获取任务"""
        pass


class InMemoryQueueBackend(QueueBackend):
    """内存队列后端（用于测试）"""
    
    def __init__(self):
        self._queues: Dict[str, List[DistributedTask]] = {}
        self._tasks: Dict[str, DistributedTask] = {}
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition()
    
    async def connect(self) -> None:
        pass
    
    async def disconnect(self) -> None:
        pass
    
    async def enqueue(self, task: DistributedTask, queue_name: str = "default") -> bool:
        async with self._lock:
            if queue_name not in self._queues:
                self._queues[queue_name] = []
            
            # 按优先级插入
            task.state = TaskState.QUEUED
            self._queues[queue_name].append(task)
            self._queues[queue_name].sort(key=lambda t: t.priority.value, reverse=True)
            self._tasks[task.id] = task
            
        async with self._condition:
            self._condition.notify_all()
        
        return True
    
    async def dequeue(self, queue_name: str = "default", timeout: float = 5.0) -> Optional[DistributedTask]:
        start_time = time.time()
        
        while True:
            async with self._lock:
                if queue_name in self._queues and self._queues[queue_name]:
                    task = self._queues[queue_name].pop(0)
                    task.state = TaskState.RUNNING
                    task.started_at = time.time()
                    self._tasks[task.id] = task
                    return task
            
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                return None
            
            async with self._condition:
                try:
                    await asyncio.wait_for(
                        self._condition.wait(),
                        timeout=min(1.0, timeout - elapsed)
                    )
                except asyncio.TimeoutError:
                    pass
    
    async def ack(self, task_id: str, queue_name: str = "default") -> bool:
        async with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].state = TaskState.COMPLETED
                self._tasks[task_id].completed_at = time.time()
                return True
        return False
    
    async def nack(self, task_id: str, queue_name: str = "default", requeue: bool = True) -> bool:
        async with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                if requeue:
                    task.state = TaskState.RETRYING
                    task.retry_count += 1
                    if queue_name not in self._queues:
                        self._queues[queue_name] = []
                    self._queues[queue_name].append(task)
                else:
                    task.state = TaskState.FAILED
                return True
        return False
    
    async def get_queue_length(self, queue_name: str = "default") -> int:
        async with self._lock:
            return len(self._queues.get(queue_name, []))
    
    async def purge_queue(self, queue_name: str = "default") -> int:
        async with self._lock:
            if queue_name in self._queues:
                count = len(self._queues[queue_name])
                self._queues[queue_name] = []
                return count
            return 0
    
    async def update_task(self, task: DistributedTask) -> bool:
        async with self._lock:
            if task.id in self._tasks:
                self._tasks[task.id] = task
                return True
        return False
    
    async def get_task(self, task_id: str) -> Optional[DistributedTask]:
        async with self._lock:
            return self._tasks.get(task_id)


class RedisQueueBackend(QueueBackend):
    """Redis 队列后端"""
    
    def __init__(
        self,
        url: str = "redis://localhost:6379",
        prefix: str = "codegnipy:",
    ):
        self._url = url
        self._prefix = prefix
        self._client: Optional["aioredis.Redis"] = None
        self._pubsub = None
    
    async def connect(self) -> None:
        try:
            import redis.asyncio as aioredis
        except ImportError:
            raise ImportError(
                "需要安装 redis 包。运行: pip install redis"
            )
        
        self._client = aioredis.from_url(
            self._url,
            encoding="utf-8",
            decode_responses=True
        )
    
    async def disconnect(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
    
    def _queue_key(self, queue_name: str) -> str:
        return f"{self._prefix}queue:{queue_name}"
    
    def _task_key(self, task_id: str) -> str:
        return f"{self._prefix}task:{task_id}"
    
    def _priority_score(self, priority: TaskPriority) -> float:
        """将优先级转换为分数（越高越优先）"""
        return priority.value + time.time() / 1e10
    
    async def enqueue(self, task: DistributedTask, queue_name: str = "default") -> bool:
        if not self._client:
            await self.connect()
        
        assert self._client is not None
        
        task.state = TaskState.QUEUED
        
        # 存储任务
        await self._client.set(
            self._task_key(task.id),
            json.dumps(task.to_dict())
        )
        
        # 添加到优先级队列
        score = self._priority_score(task.priority)
        await self._client.zadd(
            self._queue_key(queue_name),
            {task.id: score}
        )
        
        # 发布新任务通知
        await self._client.publish(
            f"{self._prefix}channel:{queue_name}",
            json.dumps({"event": "new_task", "task_id": task.id})
        )
        
        return True
    
    async def dequeue(self, queue_name: str = "default", timeout: float = 5.0) -> Optional[DistributedTask]:
        if not self._client:
            await self.connect()
        
        assert self._client is not None
        
        # 使用 BZPOPMIN 阻塞获取最高优先级任务
        result = await self._client.bzpopmin(
            self._queue_key(queue_name),
            timeout=int(timeout)
        )
        
        if not result:
            return None
        
        _, task_id, _ = result
        
        # 获取任务详情
        task_data = await self._client.get(self._task_key(task_id))
        if not task_data:
            return None
        
        task = DistributedTask.from_dict(json.loads(task_data))
        task.state = TaskState.RUNNING
        task.started_at = time.time()
        
        # 更新任务状态
        await self._client.set(
            self._task_key(task.id),
            json.dumps(task.to_dict())
        )
        
        return task
    
    async def ack(self, task_id: str, queue_name: str = "default") -> bool:
        if not self._client:
            await self.connect()
        
        assert self._client is not None
        
        task_data = await self._client.get(self._task_key(task_id))
        if not task_data:
            return False
        
        task = DistributedTask.from_dict(json.loads(task_data))
        task.state = TaskState.COMPLETED
        task.completed_at = time.time()
        
        await self._client.set(
            self._task_key(task.id),
            json.dumps(task.to_dict())
        )
        
        return True
    
    async def nack(self, task_id: str, queue_name: str = "default", requeue: bool = True) -> bool:
        if not self._client:
            await self.connect()
        
        assert self._client is not None
        
        task_data = await self._client.get(self._task_key(task_id))
        if not task_data:
            return False
        
        task = DistributedTask.from_dict(json.loads(task_data))
        
        if requeue:
            task.state = TaskState.RETRYING
            task.retry_count += 1
            score = self._priority_score(task.priority)
            await self._client.zadd(
                self._queue_key(queue_name),
                {task.id: score}
            )
        else:
            task.state = TaskState.FAILED
        
        await self._client.set(
            self._task_key(task.id),
            json.dumps(task.to_dict())
        )
        
        return True
    
    async def get_queue_length(self, queue_name: str = "default") -> int:
        if not self._client:
            await self.connect()
        
        assert self._client is not None
        return await self._client.zcard(self._queue_key(queue_name))
    
    async def purge_queue(self, queue_name: str = "default") -> int:
        if not self._client:
            await self.connect()
        
        assert self._client is not None
        
        # 获取所有任务ID
        task_ids = await self._client.zrange(
            self._queue_key(queue_name), 0, -1
        )
        
        # 删除队列
        await self._client.delete(self._queue_key(queue_name))
        
        # 删除任务数据
        if task_ids:
            keys = [self._task_key(tid) for tid in task_ids]
            await self._client.delete(*keys)
        
        return len(task_ids)
    
    async def update_task(self, task: DistributedTask) -> bool:
        if not self._client:
            await self.connect()
        
        assert self._client is not None
        
        await self._client.set(
            self._task_key(task.id),
            json.dumps(task.to_dict())
        )
        return True
    
    async def get_task(self, task_id: str) -> Optional[DistributedTask]:
        if not self._client:
            await self.connect()
        
        assert self._client is not None
        
        task_data = await self._client.get(self._task_key(task_id))
        if not task_data:
            return None
        
        return DistributedTask.from_dict(json.loads(task_data))


class RabbitMQQueueBackend(QueueBackend):
    """RabbitMQ 队列后端"""
    
    def __init__(
        self,
        url: str = "amqp://guest:guest@localhost:5672/",
        exchange: str = "codegnipy",
    ):
        self._url = url
        self._exchange = exchange
        self._connection = None
        self._channel = None
        self._queues: Dict[str, Any] = {}
        self._tasks: Dict[str, DistributedTask] = {}
    
    async def connect(self) -> None:
        try:
            import aio_pika
        except ImportError:
            raise ImportError(
                "需要安装 aio-pika 包。运行: pip install aio-pika"
            )
        
        self._connection = await aio_pika.connect_robust(self._url)
        self._channel = await self._connection.channel()
        
        # 声明交换机
        await self._channel.declare_exchange(
            self._exchange,
            aio_pika.ExchangeType.DIRECT,
            durable=True
        )
    
    async def disconnect(self) -> None:
        if self._connection:
            await self._connection.close()
            self._connection = None
            self._channel = None
    
    async def _ensure_queue(self, queue_name: str) -> None:
        """确保队列存在"""
        if self._channel is None:
            await self.connect()
        
        if queue_name not in self._queues and self._channel:
            import aio_pika
            
            queue = await self._channel.declare_queue(
                queue_name,
                durable=True,
                arguments={
                    "x-max-priority": 20,  # 支持优先级
                }
            )
            
            exchange = await self._channel.get_exchange(self._exchange)
            await queue.bind(exchange, routing_key=queue_name)
            
            self._queues[queue_name] = queue
    
    async def enqueue(self, task: DistributedTask, queue_name: str = "default") -> bool:
        import aio_pika
        
        await self._ensure_queue(queue_name)
        
        if self._channel is None:
            return False
        
        task.state = TaskState.QUEUED
        self._tasks[task.id] = task
        
        message = aio_pika.Message(
            body=json.dumps(task.to_dict()).encode(),
            priority=task.priority.value,
            message_id=task.id,
            content_type="application/json",
        )
        
        exchange = await self._channel.get_exchange(self._exchange)
        await exchange.publish(message, routing_key=queue_name)
        
        return True
    
    async def dequeue(self, queue_name: str = "default", timeout: float = 5.0) -> Optional[DistributedTask]:
        await self._ensure_queue(queue_name)
        
        if queue_name not in self._queues:
            return None
        
        queue = self._queues[queue_name]
        
        try:
            message = await asyncio.wait_for(
                queue.get(fail=False, no_ack=False),
                timeout=timeout
            )
            
            if message is None:
                return None
            
            task_data = json.loads(message.body.decode())
            task = DistributedTask.from_dict(task_data)
            task.state = TaskState.RUNNING
            task.started_at = time.time()
            
            # 存储消息引用以便后续 ack/nack
            task.metadata["_message"] = message
            self._tasks[task.id] = task
            
            return task
            
        except asyncio.TimeoutError:
            return None
    
    async def ack(self, task_id: str, queue_name: str = "default") -> bool:
        if task_id not in self._tasks:
            return False
        
        task = self._tasks[task_id]
        message = task.metadata.get("_message")
        
        if message:
            await message.ack()
            task.state = TaskState.COMPLETED
            task.completed_at = time.time()
            del task.metadata["_message"]
            return True
        
        return False
    
    async def nack(self, task_id: str, queue_name: str = "default", requeue: bool = True) -> bool:
        if task_id not in self._tasks:
            return False
        
        task = self._tasks[task_id]
        message = task.metadata.get("_message")
        
        if message:
            await message.nack(requeue=requeue)
            
            if requeue:
                task.state = TaskState.RETRYING
                task.retry_count += 1
            else:
                task.state = TaskState.FAILED
            
            del task.metadata["_message"]
            return True
        
        return False
    
    async def get_queue_length(self, queue_name: str = "default") -> int:
        await self._ensure_queue(queue_name)
        
        if queue_name not in self._queues:
            return 0
        
        queue = self._queues[queue_name]
        info = await queue.declare(passive=True)
        return info.message_count
    
    async def purge_queue(self, queue_name: str = "default") -> int:
        await self._ensure_queue(queue_name)
        
        if queue_name not in self._queues:
            return 0
        
        queue = self._queues[queue_name]
        result = await queue.purge()
        return result.message_count
    
    async def update_task(self, task: DistributedTask) -> bool:
        self._tasks[task.id] = task
        return True
    
    async def get_task(self, task_id: str) -> Optional[DistributedTask]:
        return self._tasks.get(task_id)


class LoadBalancer:
    """负载均衡器"""
    
    def __init__(
        self,
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.LEAST_CONNECTIONS
    ):
        self._strategy = strategy
        self._workers: Dict[str, WorkerInfo] = {}
        self._round_robin_index = 0
        self._lock = threading.Lock()
    
    def register_worker(self, worker: WorkerInfo) -> None:
        """注册工作节点"""
        with self._lock:
            self._workers[worker.id] = worker
    
    def unregister_worker(self, worker_id: str) -> None:
        """注销工作节点"""
        with self._lock:
            self._workers.pop(worker_id, None)
    
    def update_worker(self, worker: WorkerInfo) -> None:
        """更新工作节点状态"""
        with self._lock:
            if worker.id in self._workers:
                self._workers[worker.id] = worker
    
    def get_available_workers(self) -> List[WorkerInfo]:
        """获取可用的工作节点"""
        with self._lock:
            return [w for w in self._workers.values() if w.is_available]
    
    def select_worker(self, task: Optional[DistributedTask] = None) -> Optional[WorkerInfo]:
        """选择工作节点"""
        workers = self.get_available_workers()
        
        if not workers:
            return None
        
        if self._strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._select_round_robin(workers)
        elif self._strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return self._select_least_connections(workers)
        elif self._strategy == LoadBalanceStrategy.WEIGHTED:
            return self._select_weighted(workers)
        elif self._strategy == LoadBalanceStrategy.RANDOM:
            return self._select_random(workers)
        elif self._strategy == LoadBalanceStrategy.CONSISTENT_HASH:
            return self._select_consistent_hash(workers, task)
        else:
            return workers[0]
    
    def _select_round_robin(self, workers: List[WorkerInfo]) -> WorkerInfo:
        """轮询选择"""
        with self._lock:
            worker = workers[self._round_robin_index % len(workers)]
            self._round_robin_index += 1
            return worker
    
    def _select_least_connections(self, workers: List[WorkerInfo]) -> WorkerInfo:
        """最少连接选择"""
        return min(workers, key=lambda w: w.current_tasks)
    
    def _select_weighted(self, workers: List[WorkerInfo]) -> WorkerInfo:
        """加权随机选择"""
        import random
        
        total_weight = sum(w.weight for w in workers)
        r = random.uniform(0, total_weight)
        
        cumulative = 0
        for worker in workers:
            cumulative += worker.weight
            if r <= cumulative:
                return worker
        
        return workers[-1]
    
    def _select_random(self, workers: List[WorkerInfo]) -> WorkerInfo:
        """随机选择"""
        import random
        return random.choice(workers)
    
    def _select_consistent_hash(
        self, 
        workers: List[WorkerInfo], 
        task: Optional[DistributedTask]
    ) -> WorkerInfo:
        """一致性哈希选择"""
        if task is None:
            return workers[0]
        
        # 使用任务ID进行哈希
        hash_value = int(hashlib.md5(task.id.encode()).hexdigest(), 16)
        return workers[hash_value % len(workers)]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取负载均衡统计信息"""
        workers = self.get_available_workers()
        
        return {
            "strategy": self._strategy.value,
            "total_workers": len(self._workers),
            "available_workers": len(workers),
            "total_capacity": sum(w.max_tasks for w in workers),
            "current_load": sum(w.current_tasks for w in workers),
            "average_load": sum(w.load for w in workers) / len(workers) if workers else 0,
        }


class DistributedScheduler:
    """分布式任务调度器"""
    
    def __init__(
        self,
        backend: QueueBackend,
        queue_name: str = "default",
        worker_id: Optional[str] = None,
        max_concurrent: int = 10,
        heartbeat_interval: float = 10.0,
    ):
        self._backend = backend
        self._queue_name = queue_name
        self._worker_id = worker_id or str(uuid.uuid4())
        self._max_concurrent = max_concurrent
        self._heartbeat_interval = heartbeat_interval
        
        self._running = False
        self._current_tasks: Dict[str, asyncio.Task] = {}
        self._handlers: Dict[str, Callable] = {}
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._worker_info = WorkerInfo(
            id=self._worker_id,
            max_tasks=max_concurrent,
        )
    
    @property
    def worker_id(self) -> str:
        return self._worker_id
    
    def register_handler(
        self, 
        task_name: str, 
        handler: Callable[[DistributedTask], Awaitable[Any]]
    ) -> None:
        """注册任务处理器"""
        self._handlers[task_name] = handler
    
    async def start(self) -> None:
        """启动调度器"""
        await self._backend.connect()
        self._running = True
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        
        # 启动心跳任务
        asyncio.create_task(self._heartbeat_loop())
        
        # 启动任务处理循环
        await self._process_loop()
    
    async def stop(self) -> None:
        """停止调度器"""
        self._running = False
        
        # 等待当前任务完成
        if self._current_tasks:
            await asyncio.gather(*self._current_tasks.values(), return_exceptions=True)
        
        await self._backend.disconnect()
    
    async def submit(
        self,
        task: DistributedTask,
        queue_name: Optional[str] = None,
    ) -> str:
        """提交任务"""
        queue = queue_name or self._queue_name
        await self._backend.enqueue(task, queue)
        return task.id
    
    async def get_task_status(self, task_id: str) -> Optional[DistributedTask]:
        """获取任务状态"""
        return await self._backend.get_task(task_id)
    
    async def _heartbeat_loop(self) -> None:
        """心跳循环"""
        while self._running:
            self._worker_info.last_heartbeat = time.time()
            self._worker_info.current_tasks = len(self._current_tasks)
            await asyncio.sleep(self._heartbeat_interval)
    
    async def _process_loop(self) -> None:
        """任务处理循环"""
        while self._running:
            try:
                # 获取任务
                task = await self._backend.dequeue(
                    self._queue_name,
                    timeout=1.0
                )
                
                if task is None:
                    continue
                
                # 等待信号量
                if self._semaphore is None:
                    continue
                    
                async with self._semaphore:
                    task.worker_id = self._worker_id
                    await self._process_task(task)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                # 记录错误，继续处理
                print(f"Error in process loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_task(self, task: DistributedTask) -> None:
        """处理单个任务"""
        handler = self._handlers.get(task.name)
        
        if handler is None:
            task.error = f"No handler registered for task: {task.name}"
            task.state = TaskState.FAILED
            await self._backend.update_task(task)
            return
        
        try:
            # 执行处理器
            result = await asyncio.wait_for(
                handler(task),
                timeout=task.timeout
            )
            
            task.result = result
            await self._backend.ack(task.id, self._queue_name)
            
        except asyncio.TimeoutError:
            task.error = "Task timeout"
            await self._handle_failure(task)
            
        except Exception as e:
            task.error = str(e)
            await self._handle_failure(task)
    
    async def _handle_failure(self, task: DistributedTask) -> None:
        """处理任务失败"""
        if task.retry_count < task.max_retries:
            await self._backend.nack(task.id, self._queue_name, requeue=True)
        else:
            task.state = TaskState.FAILED
            await self._backend.update_task(task)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        return {
            "worker_id": self._worker_id,
            "running": self._running,
            "current_tasks": len(self._current_tasks),
            "max_concurrent": self._max_concurrent,
            "registered_handlers": list(self._handlers.keys()),
        }


# 便捷函数
def create_queue_backend(
    backend_type: QueueBackendType,
    **kwargs
) -> QueueBackend:
    """创建队列后端"""
    if backend_type == QueueBackendType.MEMORY:
        return InMemoryQueueBackend()
    elif backend_type == QueueBackendType.REDIS:
        return RedisQueueBackend(**kwargs)
    elif backend_type == QueueBackendType.RABBITMQ:
        return RabbitMQQueueBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


async def submit_distributed_task(
    task_name: str,
    payload: Dict[str, Any],
    backend: Optional[QueueBackend] = None,
    queue_name: str = "default",
    priority: TaskPriority = TaskPriority.NORMAL,
    **kwargs
) -> str:
    """
    提交分布式任务
    
    参数:
        task_name: 任务名称
        payload: 任务载荷
        backend: 队列后端（可选，默认使用内存队列）
        queue_name: 队列名称
        priority: 任务优先级
        **kwargs: 其他任务参数
        
    返回:
        任务ID
    """
    if backend is None:
        backend = InMemoryQueueBackend()
        await backend.connect()
    
    task = DistributedTask(
        name=task_name,
        payload=payload,
        priority=priority,
        **kwargs
    )
    
    await backend.enqueue(task, queue_name)
    return task.id
