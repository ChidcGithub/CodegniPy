"""
Codegnipy 异步调度器模块

提供高性能的异步 LLM 调用调度，包括：
- 异步执行
- 超时控制
- 重试机制
- 并发控制
- 优先级队列
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Optional, Callable, Coroutine, List, Dict, 
    TypeVar, Generic, Union
)

from .runtime import LLMConfig, CognitiveContext


T = TypeVar('T')


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class Priority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 100


@dataclass(order=True)
class ScheduledTask(Generic[T]):
    """调度任务"""
    priority: int
    task_id: str = field(compare=False)
    coro_factory: Callable[[], Coroutine] = field(compare=False, repr=False)  # 协程工厂
    created_at: float = field(default_factory=time.time, compare=False)
    started_at: Optional[float] = field(default=None, compare=False)
    completed_at: Optional[float] = field(default=None, compare=False)
    status: TaskStatus = field(default=TaskStatus.PENDING, compare=False)
    result: Optional[T] = field(default=None, compare=False)
    error: Optional[Exception] = field(default=None, compare=False)
    retries: int = field(default=0, compare=False)
    max_retries: int = field(default=3, compare=False)
    timeout: Optional[float] = field(default=None, compare=False)
    callback: Optional[Callable[[T], None]] = field(default=None, compare=False, repr=False)
    
    def __post_init__(self):
        self._async_task: Optional[asyncio.Task] = None
    
    def create_coro(self) -> Coroutine:
        """创建新的协程实例"""
        return self.coro_factory()


@dataclass
class RetryPolicy:
    """重试策略"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    
    def get_delay(self, attempt: int) -> float:
        """计算重试延迟（指数退避）"""
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)


@dataclass
class SchedulerConfig:
    """调度器配置"""
    max_concurrent: int = 10
    default_timeout: float = 60.0
    default_priority: Priority = Priority.NORMAL
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)


class CognitiveScheduler:
    """
    认知任务调度器
    
    管理异步 LLM 调用的执行，支持：
    - 并发控制
    - 优先级队列
    - 超时和重试
    - 任务回调
    
    示例:
        scheduler = CognitiveScheduler(max_concurrent=5)
        
        async def main():
            # 提交任务
            task_id = await scheduler.submit(
                async_cognitive_call("Hello"),
                priority=Priority.HIGH
            )
            
            # 等待结果
            result = await scheduler.get_result(task_id)
            print(result)
        
        asyncio.run(main())
    """
    
    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()
        self._tasks: Dict[str, ScheduledTask[Any]] = {}
        self._pending_queue: Optional[asyncio.PriorityQueue[ScheduledTask[Any]]] = None
        self._running_count: int = 0
        self._counter: int = 0
        self._lock: Optional[asyncio.Lock] = None
        self._started: bool = False
    
    async def _ensure_initialized(self):
        """确保异步资源初始化"""
        if self._pending_queue is None:
            self._pending_queue = asyncio.PriorityQueue()
        if self._lock is None:
            self._lock = asyncio.Lock()
    
    def _generate_task_id(self) -> str:
        """生成任务 ID"""
        self._counter += 1
        return f"task_{self._counter}_{int(time.time() * 1000)}"
    
    async def submit(
        self,
        coro_or_factory: Union[Coroutine[Any, Any, T], Callable[[], Coroutine[Any, Any, T]]],
        priority: Optional[Priority] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        callback: Optional[Callable[[T], None]] = None
    ) -> str:
        """
        提交异步任务
        
        参数:
            coro_or_factory: 协程对象或协程工厂函数（支持重试）
            priority: 优先级
            timeout: 超时时间（秒）
            max_retries: 最大重试次数
            callback: 完成回调
        
        返回:
            任务 ID
        
        注意:
            如果传入协程对象，重试将无法创建新实例。
            建议传入协程工厂函数以支持重试。
        """
        await self._ensure_initialized()
        
        # 确定协程工厂
        if asyncio.iscoroutine(coro_or_factory):
            # 直接传入协程，无法重试
            coro = coro_or_factory
            def coro_factory():
                return coro
            effective_max_retries = 0  # 无法重试
        else:
            # 传入工厂函数
            coro_factory = coro_or_factory
            coro = coro_factory()
            effective_max_retries = max_retries or self.config.retry_policy.max_retries
        
        task_id = self._generate_task_id()
        task = ScheduledTask(
            priority=(priority or self.config.default_priority).value * -1,  # 负数实现高优先级先出
            task_id=task_id,
            coro_factory=coro_factory,
            timeout=timeout or self.config.default_timeout,
            max_retries=effective_max_retries,
            callback=callback
        )

        # 存储初始协程
        task._current_coro = coro  # type: ignore[attr-defined]

        self._tasks[task_id] = task
        assert self._pending_queue is not None
        await self._pending_queue.put(task)

        # 尝试处理队列
        asyncio.create_task(self._process_queue())

        return task_id

    async def _process_queue(self):
        """处理任务队列"""
        await self._ensure_initialized()

        assert self._lock is not None
        async with self._lock:
            while self._running_count < self.config.max_concurrent:
                assert self._pending_queue is not None
                if self._pending_queue.empty():
                    break

                task = await self._pending_queue.get()
                if task.status == TaskStatus.CANCELLED:
                    continue

                asyncio.create_task(self._execute_task(task))
                self._running_count += 1
    
    async def _execute_task(self, task: ScheduledTask):
        """执行单个任务"""
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        
        try:
            # 获取当前协程
            coro = getattr(task, '_current_coro', None) or task.create_coro()
            
            # 执行带超时
            result = await asyncio.wait_for(
                coro,
                timeout=task.timeout
            )
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            
            # 调用回调
            if task.callback:
                try:
                    task.callback(result)
                except Exception:
                    # 回调错误不影响任务状态
                    pass
                    
        except asyncio.TimeoutError:
            task.status = TaskStatus.TIMEOUT
            task.error = TimeoutError(f"Task {task.task_id} timed out after {task.timeout}s")
            
            # 重试
            if task.retries < task.max_retries:
                await self._retry_task(task)
                
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = e

            # 重试
            if task.retries < task.max_retries:
                await self._retry_task(task)

        finally:
            assert self._lock is not None
            async with self._lock:
                self._running_count -= 1

            # 继续处理队列
            await self._process_queue()
    
    async def _retry_task(self, task: ScheduledTask):
        """重试任务"""
        task.retries += 1
        task.status = TaskStatus.PENDING

        # 创建新的协程实例
        task._current_coro = task.create_coro()  # type: ignore[attr-defined]

        # 等待延迟
        delay = self.config.retry_policy.get_delay(task.retries - 1)
        await asyncio.sleep(delay)

        # 重新加入队列
        assert self._pending_queue is not None
        await self._pending_queue.put(task)
    
    async def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        获取任务结果
        
        参数:
            task_id: 任务 ID
            timeout: 等待超时
        
        返回:
            任务结果
        
        抛出:
            KeyError: 任务不存在
            Exception: 任务失败
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task {task_id} not found")
        
        task = self._tasks[task_id]
        
        # 等待完成
        start_time = time.time()
        while task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Timeout waiting for task {task_id}")
            await asyncio.sleep(0.1)
        
        # 返回结果或抛出错误
        if task.status == TaskStatus.COMPLETED:
            return task.result
        elif task.error:
            raise task.error
        else:
            raise Exception(f"Task {task_id} failed with status {task.status}")
    
    async def cancel(self, task_id: str) -> bool:
        """取消任务"""
        if task_id not in self._tasks:
            return False
        
        task = self._tasks[task_id]
        task.status = TaskStatus.CANCELLED
        
        if task._async_task:
            task._async_task.cancel()
        
        return True
    
    def get_status(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        task = self._tasks.get(task_id)
        return task.status if task else None
    
    async def wait_all(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """等待所有任务完成"""
        start_time = time.time()
        
        while True:
            all_done = all(
                t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT)
                for t in self._tasks.values()
            )
            
            if all_done:
                break
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Timeout waiting for all tasks")
            
            await asyncio.sleep(0.1)
        
        return {tid: t.result for tid, t in self._tasks.items() if t.status == TaskStatus.COMPLETED}
    
    def stats(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        statuses: Dict[str, int] = {}
        for task in self._tasks.values():
            status = task.status.value
            statuses[status] = statuses.get(status, 0) + 1

        return {
            "total_tasks": len(self._tasks),
            "running": self._running_count,
            "max_concurrent": self.config.max_concurrent,
            "by_status": statuses
        }


# ============ 异步 LLM 调用 ============

async def async_cognitive_call(
    prompt: str,
    context: Optional[CognitiveContext] = None,
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    config: Optional[LLMConfig] = None
) -> str:
    """
    异步认知调用
    
    参数:
        prompt: 提示文本
        context: 认知上下文
        model: 模型名称
        temperature: 温度参数
        config: LLM 配置
    
    返回:
        LLM 响应文本
    """
    # 获取配置
    if config is None:
        ctx = context or CognitiveContext.get_current()
        if ctx:
            config = ctx.get_config()
            if model:
                config.model = model
            if temperature is not None:
                config.temperature = temperature
        else:
            config = LLMConfig()
    
    # 异步调用 OpenAI
    return await _call_openai_async(config, prompt)


async def _call_openai_async(config: LLMConfig, prompt: str) -> str:
    """异步调用 OpenAI API"""
    try:
        import openai
    except ImportError:
        raise ImportError("需要安装 openai 包。运行: pip install openai")
    
    client = openai.AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url
    )
    
    response = await client.chat.completions.create(
        model=config.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.temperature,
        max_tokens=config.max_tokens
    )

    return response.choices[0].message.content or ""


# ============ 便捷函数 ============

def run_async(coro: Coroutine) -> Any:
    """运行协程的便捷函数"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop and loop.is_running():
        # 已经在事件循环中，创建任务
        return asyncio.create_task(coro)
    else:
        # 新建事件循环
        return asyncio.run(coro)


async def batch_call(
    prompts: List[str],
    context: Optional[CognitiveContext] = None,
    max_concurrent: int = 5,
    timeout: float = 60.0
) -> List[Optional[str]]:
    """
    批量异步调用

    参数:
        prompts: 提示列表
        context: 认知上下文
        max_concurrent: 最大并发数
        timeout: 单个调用超时

    返回:
        响应列表（与输入顺序对应）
    """
    scheduler = CognitiveScheduler(SchedulerConfig(max_concurrent=max_concurrent))

    # 提交所有任务
    task_ids = []
    for prompt in prompts:
        task_id = await scheduler.submit(
            async_cognitive_call(prompt, context),
            timeout=timeout
        )
        task_ids.append(task_id)

    # 等待所有完成
    await scheduler.wait_all()

    # 收集结果
    results: List[Optional[str]] = []
    for task_id in task_ids:
        task = scheduler._tasks[task_id]
        if task.status == TaskStatus.COMPLETED:
            results.append(task.result)
        else:
            results.append(None)  # 或抛出异常

    return results
