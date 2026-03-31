"""
Codegnipy 缓存系统模块

提供 LLM 响应缓存、嵌入缓存和智能缓存失效功能。
支持内存缓存和 Redis 缓存后端。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Optional, List, Dict, Any, Generic, TypeVar, 
    Callable, TYPE_CHECKING, Tuple, Awaitable
)
import json
import time
import hashlib
import threading
from collections import OrderedDict
from datetime import datetime
import asyncio
from functools import wraps

if TYPE_CHECKING:
    import redis.asyncio as aioredis

T = TypeVar('T')
K = TypeVar('K')


class CacheBackendType(Enum):
    """缓存后端类型"""
    MEMORY = "memory"
    REDIS = "redis"


class CachePolicy(Enum):
    """缓存策略"""
    LRU = "lru"          # 最近最少使用
    LFU = "lfu"          # 最不常用
    FIFO = "fifo"        # 先进先出
    TTL = "ttl"          # 基于时间过期


@dataclass
class CacheEntry(Generic[T]):
    """缓存条目"""
    key: str
    value: T
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None  # 秒
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """是否过期"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    @property
    def age(self) -> float:
        """缓存年龄（秒）"""
        return time.time() - self.created_at
    
    def touch(self) -> None:
        """更新访问信息"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "ttl": self.ttl,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """从字典创建"""
        return cls(
            key=data["key"],
            value=data["value"],
            created_at=data["created_at"],
            last_accessed=data["last_accessed"],
            access_count=data["access_count"],
            ttl=data.get("ttl"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CacheStats:
    """缓存统计"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 1000
    
    @property
    def hit_rate(self) -> float:
        """命中率"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def usage_ratio(self) -> float:
        """使用率"""
        return self.size / self.max_size if self.max_size > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "size": self.size,
            "max_size": self.max_size,
            "hit_rate": self.hit_rate,
            "usage_ratio": self.usage_ratio,
        }


class CacheBackend(ABC, Generic[K, T]):
    """缓存后端抽象基类"""
    
    @abstractmethod
    async def get(self, key: K) -> Optional[CacheEntry[T]]:
        """获取缓存"""
        pass
    
    @abstractmethod
    async def set(self, key: K, value: T, ttl: Optional[float] = None, **metadata) -> bool:
        """设置缓存"""
        pass
    
    @abstractmethod
    async def delete(self, key: K) -> bool:
        """删除缓存"""
        pass
    
    @abstractmethod
    async def exists(self, key: K) -> bool:
        """检查是否存在"""
        pass
    
    @abstractmethod
    async def clear(self) -> int:
        """清空缓存"""
        pass
    
    @abstractmethod
    async def keys(self, pattern: Optional[str] = None) -> List[K]:
        """获取所有键"""
        pass
    
    @abstractmethod
    async def size(self) -> int:
        """获取缓存大小"""
        pass
    
    @abstractmethod
    def get_stats(self) -> CacheStats:
        """获取统计信息"""
        pass


class LRUCacheBackend(CacheBackend[str, T]):
    """LRU 内存缓存后端"""
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = None,
    ):
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = CacheStats(max_size=max_size)
    
    def _make_key(self, key: str) -> str:
        """生成缓存键"""
        return key
    
    async def get(self, key: str) -> Optional[CacheEntry[T]]:
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None
            
            entry = self._cache[key]
            
            # 检查过期
            if entry.is_expired:
                del self._cache[key]
                self._stats.misses += 1
                self._stats.size -= 1
                return None
            
            # 更新访问信息并移到末尾（最近使用）
            entry.touch()
            self._cache.move_to_end(key)
            self._stats.hits += 1
            
            return entry
    
    async def set(self, key: str, value: T, ttl: Optional[float] = None, **metadata) -> bool:
        with self._lock:
            # 如果键已存在，删除旧的
            if key in self._cache:
                del self._cache[key]
                self._stats.size -= 1
            
            # 如果已满，删除最旧的
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
                self._stats.evictions += 1
                self._stats.size -= 1
            
            # 添加新条目
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl or self._default_ttl,
                metadata=metadata,
            )
            self._cache[key] = entry
            self._stats.size += 1
            
            return True
    
    async def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.size -= 1
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        with self._lock:
            if key not in self._cache:
                return False
            entry = self._cache[key]
            return not entry.is_expired
    
    async def clear(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats.size = 0
            return count
    
    async def keys(self, pattern: Optional[str] = None) -> List[str]:
        with self._lock:
            if pattern is None:
                return list(self._cache.keys())
            
            # 简单的模式匹配
            import fnmatch
            return [k for k in self._cache.keys() if fnmatch.fnmatch(k, pattern)]
    
    async def size(self) -> int:
        with self._lock:
            return len(self._cache)
    
    def get_stats(self) -> CacheStats:
        return self._stats


class RedisCacheBackend(CacheBackend[str, T]):
    """Redis 缓存后端"""
    
    def __init__(
        self,
        url: str = "redis://localhost:6379",
        prefix: str = "codegnipy:cache:",
        default_ttl: Optional[float] = 3600,  # 默认 1 小时
        serializer: Optional[Callable[[T], str]] = None,
        deserializer: Optional[Callable[[str], T]] = None,
    ):
        self._url = url
        self._prefix = prefix
        self._default_ttl = default_ttl
        self._serializer = serializer or json.dumps
        self._deserializer = deserializer or json.loads
        self._client: Optional["aioredis.Redis"] = None
        self._stats = CacheStats()
    
    async def _connect(self) -> "aioredis.Redis":
        if self._client is None:
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
        
        return self._client
    
    def _make_key(self, key: str) -> str:
        return f"{self._prefix}{key}"
    
    async def get(self, key: str) -> Optional[CacheEntry[T]]:
        client = await self._connect()
        redis_key = self._make_key(key)
        
        data = await client.get(redis_key)
        
        if data is None:
            self._stats.misses += 1
            return None
        
        try:
            entry_dict = json.loads(data)
            entry = CacheEntry(
                key=entry_dict["key"],
                value=self._deserializer(entry_dict["value"]) if isinstance(entry_dict["value"], str) else entry_dict["value"],
                created_at=entry_dict["created_at"],
                last_accessed=time.time(),
                access_count=entry_dict.get("access_count", 0) + 1,
                ttl=entry_dict.get("ttl"),
                metadata=entry_dict.get("metadata", {}),
            )
            
            # 更新访问信息
            await client.set(
                redis_key,
                json.dumps(entry.to_dict()),
                ex=int(entry.ttl) if entry.ttl else None,
            )
            
            self._stats.hits += 1
            return entry
            
        except (json.JSONDecodeError, KeyError):
            self._stats.misses += 1
            return None
    
    async def set(self, key: str, value: T, ttl: Optional[float] = None, **metadata) -> bool:
        client = await self._connect()
        redis_key = self._make_key(key)
        
        entry = CacheEntry(
            key=key,
            value=value,
            ttl=ttl or self._default_ttl,
            metadata=metadata,
        )
        
        serialized_value = self._serializer(value)
        
        entry_dict = entry.to_dict()
        entry_dict["value"] = serialized_value
        
        await client.set(
            redis_key,
            json.dumps(entry_dict),
            ex=int(entry.ttl) if entry.ttl else None,
        )
        
        self._stats.size = await self.size()
        return True
    
    async def delete(self, key: str) -> bool:
        client = await self._connect()
        redis_key = self._make_key(key)
        
        result = await client.delete(redis_key)
        self._stats.size = await self.size()
        return result > 0
    
    async def exists(self, key: str) -> bool:
        client = await self._connect()
        redis_key = self._make_key(key)
        return await client.exists(redis_key) > 0
    
    async def clear(self) -> int:
        client = await self._connect()
        
        # 扫描并删除所有匹配的键
        keys = []
        async for key in client.scan_iter(match=f"{self._prefix}*"):
            keys.append(key)
        
        if keys:
            await client.delete(*keys)
        
        self._stats.size = 0
        return len(keys)
    
    async def keys(self, pattern: Optional[str] = None) -> List[str]:
        client = await self._connect()
        
        if pattern:
            search_pattern = f"{self._prefix}{pattern}"
        else:
            search_pattern = f"{self._prefix}*"
        
        keys = []
        async for key in client.scan_iter(match=search_pattern):
            # 移除前缀
            keys.append(key[len(self._prefix):])
        
        return keys
    
    async def size(self) -> int:
        client = await self._connect()
        count = 0
        async for _ in client.scan_iter(match=f"{self._prefix}*"):
            count += 1
        return count
    
    def get_stats(self) -> CacheStats:
        return self._stats
    
    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None


class ResponseCache:
    """LLM 响应缓存"""
    
    def __init__(
        self,
        backend: CacheBackend[str, str],
        key_prefix: str = "response:",
        include_model: bool = True,
        include_temperature: bool = False,
    ):
        self._backend = backend
        self._key_prefix = key_prefix
        self._include_model = include_model
        self._include_temperature = include_temperature
    
    def _generate_key(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """生成缓存键"""
        key_parts = [prompt]
        
        if self._include_model and model:
            key_parts.append(f"model:{model}")
        
        if self._include_temperature and temperature is not None:
            key_parts.append(f"temp:{temperature}")
        
        # 添加额外参数
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")
        
        key_string = "|".join(key_parts)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:32]
        
        return f"{self._key_prefix}{key_hash}"
    
    async def get(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Optional[str]:
        """获取缓存的响应"""
        key = self._generate_key(prompt, model, temperature, **kwargs)
        entry = await self._backend.get(key)
        
        return entry.value if entry else None
    
    async def set(
        self,
        prompt: str,
        response: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        ttl: Optional[float] = None,
        **kwargs
    ) -> bool:
        """缓存响应"""
        key = self._generate_key(prompt, model, temperature, **kwargs)
        return await self._backend.set(
            key, 
            response, 
            ttl=ttl,
            model=model,
            prompt_length=len(prompt),
            response_length=len(response),
        )
    
    async def invalidate(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> bool:
        """使缓存失效"""
        key = self._generate_key(prompt, model, **kwargs)
        return await self._backend.delete(key)
    
    async def clear(self) -> int:
        """清空所有缓存"""
        return await self._backend.clear()
    
    def get_stats(self) -> CacheStats:
        """获取缓存统计"""
        return self._backend.get_stats()


class EmbeddingCache:
    """嵌入向量缓存"""
    
    def __init__(
        self,
        backend: CacheBackend[str, List[float]],
        key_prefix: str = "embedding:",
        vector_size: int = 1536,  # OpenAI ada-002 维度
    ):
        self._backend = backend
        self._key_prefix = key_prefix
        self._vector_size = vector_size
    
    def _generate_key(self, text: str, model: str = "text-embedding-ada-002") -> str:
        """生成缓存键"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:32]
        return f"{self._key_prefix}{model}:{text_hash}"
    
    async def get(
        self,
        text: str,
        model: str = "text-embedding-ada-002"
    ) -> Optional[List[float]]:
        """获取缓存的嵌入"""
        key = self._generate_key(text, model)
        entry = await self._backend.get(key)
        
        return entry.value if entry else None
    
    async def set(
        self,
        text: str,
        embedding: List[float],
        model: str = "text-embedding-ada-002",
        ttl: Optional[float] = None,
    ) -> bool:
        """缓存嵌入"""
        key = self._generate_key(text, model)
        return await self._backend.set(
            key,
            embedding,
            ttl=ttl,
            model=model,
            text_length=len(text),
            vector_size=len(embedding),
        )
    
    async def get_batch(
        self,
        texts: List[str],
        model: str = "text-embedding-ada-002"
    ) -> List[Optional[List[float]]]:
        """批量获取嵌入"""
        results = []
        for text in texts:
            embedding = await self.get(text, model)
            results.append(embedding)
        return results
    
    async def set_batch(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        model: str = "text-embedding-ada-002",
        ttl: Optional[float] = None,
    ) -> List[bool]:
        """批量设置嵌入"""
        results = []
        for text, embedding in zip(texts, embeddings):
            result = await self.set(text, embedding, model, ttl)
            results.append(result)
        return results


class SemanticCache:
    """语义缓存 - 基于相似度的缓存"""
    
    def __init__(
        self,
        embedding_cache: EmbeddingCache,
        similarity_threshold: float = 0.95,
        ttl: Optional[float] = None,
    ):
        self._embedding_cache = embedding_cache
        self._similarity_threshold = similarity_threshold
        self._ttl = ttl
        self._cache: Dict[str, Tuple[List[float], str]] = {}  # key -> (embedding, response)
        self._lock = threading.Lock()
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        import math
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    async def get_similar(
        self,
        embedding: List[float],
    ) -> Optional[Tuple[str, float]]:
        """找到最相似的缓存条目"""
        with self._lock:
            best_match = None
            best_similarity = 0.0
            
            for key, (cached_embedding, response) in self._cache.items():
                similarity = self._cosine_similarity(embedding, cached_embedding)
                
                if similarity > best_similarity and similarity >= self._similarity_threshold:
                    best_similarity = similarity
                    best_match = (response, similarity)
            
            return best_match
    
    async def add(
        self,
        embedding: List[float],
        response: str,
        key: Optional[str] = None,
    ) -> str:
        """添加缓存条目"""
        import uuid
        
        cache_key = key or str(uuid.uuid4())
        
        with self._lock:
            self._cache[cache_key] = (embedding, response)
        
        return cache_key
    
    async def get(
        self,
        text: str,
        embedding_func: Callable[[str], Awaitable[List[float]]],
    ) -> Optional[Tuple[str, float]]:
        """
        获取语义相似的缓存响应
        
        参数:
            text: 输入文本
            embedding_func: 获取嵌入的异步函数
            
        返回:
            (response, similarity) 或 None
        """
        embedding = await embedding_func(text)
        return await self.get_similar(embedding)
    
    def clear(self) -> int:
        """清空缓存"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count
    
    def size(self) -> int:
        """获取缓存大小"""
        with self._lock:
            return len(self._cache)


class CacheInvalidator:
    """缓存失效管理器"""
    
    def __init__(self):
        self._invalidation_rules: Dict[str, List[Callable]] = {}
        self._dependencies: Dict[str, List[str]] = {}  # key -> dependent keys
    
    def register_rule(
        self,
        pattern: str,
        rule: Callable[[str, CacheEntry], bool]
    ) -> None:
        """注册失效规则"""
        if pattern not in self._invalidation_rules:
            self._invalidation_rules[pattern] = []
        self._invalidation_rules[pattern].append(rule)
    
    def add_dependency(self, key: str, depends_on: str) -> None:
        """添加依赖关系"""
        if depends_on not in self._dependencies:
            self._dependencies[depends_on] = []
        self._dependencies[depends_on].append(key)
    
    def should_invalidate(self, key: str, entry: CacheEntry) -> bool:
        """检查是否应该失效"""
        import fnmatch
        
        for pattern, rules in self._invalidation_rules.items():
            if fnmatch.fnmatch(key, pattern):
                for rule in rules:
                    if rule(key, entry):
                        return True
        
        return False
    
    def get_dependents(self, key: str) -> List[str]:
        """获取依赖此键的所有键"""
        return self._dependencies.get(key, [])
    
    async def invalidate_cascade(
        self,
        backend: CacheBackend,
        key: str
    ) -> List[str]:
        """级联失效"""
        invalidated = [key]
        
        # 获取依赖的键
        dependents = self.get_dependents(key)
        
        # 删除所有依赖的键
        for dependent_key in dependents:
            await backend.delete(dependent_key)
            invalidated.append(dependent_key)
        
        # 删除主键
        await backend.delete(key)
        
        return invalidated


class CostOptimizer:
    """成本优化器"""
    
    def __init__(
        self,
        cache: ResponseCache,
        cache_threshold: float = 0.8,  # 命中率阈值
    ):
        self._cache = cache
        self._cache_threshold = cache_threshold
        self._cost_tracking: Dict[str, Dict[str, float]] = {}
    
    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """估算成本（美元）"""
        # 简化的成本估算（实际应根据具体模型定价）
        pricing = {
            "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
            "gpt-4-turbo": {"input": 0.01 / 1000, "output": 0.03 / 1000},
            "gpt-3.5-turbo": {"input": 0.0005 / 1000, "output": 0.0015 / 1000},
            "gpt-4o": {"input": 0.005 / 1000, "output": 0.015 / 1000},
            "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.0006 / 1000},
        }
        
        if model not in pricing:
            # 默认价格
            return (input_tokens + output_tokens) * 0.001 / 1000
        
        rates = pricing[model]
        return input_tokens * rates["input"] + output_tokens * rates["output"]
    
    def track_cost(
        self,
        operation: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached: bool = False,
    ) -> float:
        """追踪成本"""
        cost = 0.0 if cached else self.estimate_cost(model, input_tokens, output_tokens)
        
        if operation not in self._cost_tracking:
            self._cost_tracking[operation] = {
                "total_cost": 0.0,
                "total_tokens": 0,
                "cache_savings": 0.0,
                "call_count": 0,
                "cached_count": 0,
            }
        
        stats = self._cost_tracking[operation]
        stats["total_cost"] += cost
        stats["total_tokens"] += input_tokens + output_tokens
        stats["call_count"] += 1
        
        if cached:
            # 计算节省的成本
            savings = self.estimate_cost(model, input_tokens, output_tokens)
            stats["cache_savings"] += savings
            stats["cached_count"] += 1
        
        return cost
    
    def get_cost_report(self) -> Dict[str, Dict[str, Any]]:
        """获取成本报告"""
        return {
            "operations": self._cost_tracking.copy(),
            "cache_stats": self._cache.get_stats().to_dict(),
            "recommendations": self._generate_recommendations(),
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        stats = self._cache.get_stats()
        
        if stats.hit_rate < self._cache_threshold:
            recommendations.append(
                f"缓存命中率 ({stats.hit_rate:.2%}) 低于阈值 ({self._cache_threshold:.2%})，"
                "考虑增加缓存大小或延长 TTL"
            )
        
        if stats.usage_ratio > 0.9:
            recommendations.append(
                f"缓存使用率 ({stats.usage_ratio:.2%}) 过高，"
                "考虑增加缓存容量"
            )
        
        for op, op_stats in self._cost_tracking.items():
            if op_stats["call_count"] > 0:
                cache_ratio = op_stats["cached_count"] / op_stats["call_count"]
                if cache_ratio < 0.5:
                    recommendations.append(
                        f"操作 '{op}' 的缓存利用率 ({cache_ratio:.2%}) 较低，"
                        "考虑优化缓存策略"
                    )
        
        return recommendations


# 装饰器
def cached(
    cache: ResponseCache,
    key_func: Optional[Callable] = None,
    ttl: Optional[float] = None,
):
    """
    缓存装饰器
    
    示例:
        @cached(cache, key_func=lambda prompt: prompt)
        async def generate(prompt: str) -> str:
            return await llm_call(prompt)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = str(args) + str(kwargs)
            
            # 尝试获取缓存
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 执行函数
            result = await func(*args, **kwargs)
            
            # 缓存结果
            await cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator


# 便捷函数
def create_cache_backend(
    backend_type: CacheBackendType,
    **kwargs
) -> CacheBackend:
    """创建缓存后端"""
    if backend_type == CacheBackendType.MEMORY:
        return LRUCacheBackend(**kwargs)
    elif backend_type == CacheBackendType.REDIS:
        return RedisCacheBackend(**kwargs)
    else:
        raise ValueError(f"Unknown cache backend type: {backend_type}")


def create_response_cache(
    backend_type: CacheBackendType = CacheBackendType.MEMORY,
    max_size: int = 1000,
    default_ttl: Optional[float] = 3600,
    **kwargs
) -> ResponseCache:
    """创建响应缓存"""
    if backend_type == CacheBackendType.MEMORY:
        backend = LRUCacheBackend(max_size=max_size, default_ttl=default_ttl)
    elif backend_type == CacheBackendType.REDIS:
        backend = RedisCacheBackend(default_ttl=default_ttl, **kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
    
    return ResponseCache(backend)
