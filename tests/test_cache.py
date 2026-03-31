"""
缓存系统模块测试
"""

import pytest
import asyncio
import time

from codegnipy.cache import (
    CacheBackendType,
    CachePolicy,
    CacheEntry,
    CacheStats,
    LRUCacheBackend,
    ResponseCache,
    EmbeddingCache,
    SemanticCache,
    CacheInvalidator,
    CostOptimizer,
    cached,
    create_cache_backend,
    create_response_cache,
)


class TestCacheEntry:
    """缓存条目测试"""
    
    def test_entry_creation(self):
        """测试条目创建"""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl=3600,
        )
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.ttl == 3600
        assert entry.access_count == 0
    
    def test_entry_expiration(self):
        """测试条目过期"""
        # 未过期
        entry = CacheEntry(key="test", value="value", ttl=10)
        assert not entry.is_expired
        
        # 已过期
        expired_entry = CacheEntry(
            key="test",
            value="value",
            ttl=0.001,
            created_at=time.time() - 1,
        )
        assert expired_entry.is_expired
    
    def test_entry_touch(self):
        """测试条目访问更新"""
        entry = CacheEntry(key="test", value="value")
        
        original_accessed = entry.last_accessed
        time.sleep(0.01)
        
        entry.touch()
        
        assert entry.last_accessed > original_accessed
        assert entry.access_count == 1
    
    def test_entry_serialization(self):
        """测试条目序列化"""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl=3600,
            metadata={"source": "test"},
        )
        
        data = entry.to_dict()
        
        assert data["key"] == "test_key"
        assert data["value"] == "test_value"
        
        entry2 = CacheEntry.from_dict(data)
        assert entry2.key == entry.key
        assert entry2.value == entry.value


class TestCacheStats:
    """缓存统计测试"""
    
    def test_stats_creation(self):
        """测试统计创建"""
        stats = CacheStats(hits=10, misses=5, size=50, max_size=100)
        
        assert stats.hits == 10
        assert stats.misses == 5
        assert stats.size == 50
    
    def test_hit_rate(self):
        """测试命中率计算"""
        stats = CacheStats(hits=10, misses=10)
        assert stats.hit_rate == 0.5
        
        stats2 = CacheStats(hits=0, misses=0)
        assert stats2.hit_rate == 0.0
    
    def test_usage_ratio(self):
        """测试使用率计算"""
        stats = CacheStats(size=50, max_size=100)
        assert stats.usage_ratio == 0.5


class TestLRUCacheBackend:
    """LRU 缓存后端测试"""
    
    @pytest.fixture
    def cache(self):
        return LRUCacheBackend(max_size=3)
    
    @pytest.mark.asyncio
    async def test_set_get(self, cache):
        """测试设置和获取"""
        await cache.set("key1", "value1")
        
        entry = await cache.get("key1")
        assert entry is not None
        assert entry.value == "value1"
    
    @pytest.mark.asyncio
    async def test_missing_key(self, cache):
        """测试不存在的键"""
        entry = await cache.get("nonexistent")
        assert entry is None
    
    @pytest.mark.asyncio
    async def test_lru_eviction(self, cache):
        """测试 LRU 淘汰"""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        
        # 访问 key1 使其变为最近使用
        await cache.get("key1")
        
        # 添加新键，应淘汰 key2（最久未使用）
        await cache.set("key4", "value4")
        
        # key1 应该还在
        assert await cache.exists("key1")
        # key2 应该被淘汰
        assert not await cache.exists("key2")
        # key4 应该存在
        assert await cache.exists("key4")
    
    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """测试 TTL 过期"""
        cache = LRUCacheBackend(default_ttl=0.1)  # 0.1秒
        
        await cache.set("key1", "value1")
        
        # 立即获取应该存在
        entry = await cache.get("key1")
        assert entry is not None
        
        # 等待过期
        await asyncio.sleep(0.2)
        
        # 过期后应该不存在
        entry = await cache.get("key1")
        assert entry is None
    
    @pytest.mark.asyncio
    async def test_delete(self, cache):
        """测试删除"""
        await cache.set("key1", "value1")
        
        result = await cache.delete("key1")
        assert result
        
        assert not await cache.exists("key1")
    
    @pytest.mark.asyncio
    async def test_clear(self, cache):
        """测试清空"""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        
        count = await cache.clear()
        assert count == 2
        
        assert await cache.size() == 0
    
    @pytest.mark.asyncio
    async def test_keys_pattern(self, cache):
        """测试键模式匹配"""
        await cache.set("user:1", "value1")
        await cache.set("user:2", "value2")
        await cache.set("admin:1", "value3")
        
        keys = await cache.keys("user:*")
        assert len(keys) == 2
        
        all_keys = await cache.keys()
        assert len(all_keys) == 3
    
    @pytest.mark.asyncio
    async def test_stats(self, cache):
        """测试统计"""
        await cache.set("key1", "value1")
        await cache.get("key1")  # hit
        await cache.get("nonexistent")  # miss
        
        stats = cache.get_stats()
        
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.size == 1


class TestResponseCache:
    """响应缓存测试"""
    
    @pytest.fixture
    def cache(self):
        backend = LRUCacheBackend(max_size=10)
        return ResponseCache(backend)
    
    @pytest.mark.asyncio
    async def test_cache_response(self, cache):
        """测试缓存响应"""
        prompt = "Translate hello to Chinese"
        response = "你好"
        
        # 缓存响应
        await cache.set(prompt, response, model="gpt-4")
        
        # 获取缓存
        cached = await cache.get(prompt, model="gpt-4")
        assert cached == response
    
    @pytest.mark.asyncio
    async def test_different_models(self, cache):
        """测试不同模型的缓存"""
        prompt = "Hello"
        
        await cache.set(prompt, "GPT-4 response", model="gpt-4")
        await cache.set(prompt, "GPT-3.5 response", model="gpt-3.5-turbo")
        
        response4 = await cache.get(prompt, model="gpt-4")
        response35 = await cache.get(prompt, model="gpt-3.5-turbo")
        
        assert response4 == "GPT-4 response"
        assert response35 == "GPT-3.5 response"
    
    @pytest.mark.asyncio
    async def test_invalidate(self, cache):
        """测试缓存失效"""
        prompt = "Hello"
        await cache.set(prompt, "response")
        
        await cache.invalidate(prompt)
        
        cached = await cache.get(prompt)
        assert cached is None
    
    @pytest.mark.asyncio
    async def test_stats(self, cache):
        """测试统计"""
        await cache.set("prompt1", "response1")
        await cache.get("prompt1")  # hit
        await cache.get("nonexistent")  # miss
        
        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1


class TestEmbeddingCache:
    """嵌入缓存测试"""
    
    @pytest.fixture
    def cache(self):
        backend = LRUCacheBackend(max_size=10)
        return EmbeddingCache(backend)
    
    @pytest.mark.asyncio
    async def test_cache_embedding(self, cache):
        """测试缓存嵌入"""
        text = "Hello world"
        embedding = [0.1, 0.2, 0.3]
        
        await cache.set(text, embedding)
        
        cached = await cache.get(text)
        assert cached == embedding
    
    @pytest.mark.asyncio
    async def test_batch_operations(self, cache):
        """测试批量操作"""
        texts = ["text1", "text2", "text3"]
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        
        # 批量设置
        results = await cache.set_batch(texts, embeddings)
        assert all(results)
        
        # 批量获取
        cached = await cache.get_batch(texts)
        
        for i, embedding in enumerate(embeddings):
            assert cached[i] == embedding


class TestSemanticCache:
    """语义缓存测试"""
    
    @pytest.fixture
    def cache(self):
        backend = LRUCacheBackend(max_size=10)
        embedding_cache = EmbeddingCache(backend)
        return SemanticCache(embedding_cache, similarity_threshold=0.9)
    
    def test_cosine_similarity(self, cache):
        """测试余弦相似度"""
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        c = [0.0, 1.0, 0.0]
        
        # 相同向量
        sim_ab = cache._cosine_similarity(a, b)
        assert abs(sim_ab - 1.0) < 0.001
        
        # 正交向量
        sim_ac = cache._cosine_similarity(a, c)
        assert abs(sim_ac - 0.0) < 0.001
    
    @pytest.mark.asyncio
    async def test_add_and_get_similar(self, cache):
        """测试添加和获取相似"""
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.99, 0.01, 0.0]  # 非常相似
        embedding3 = [0.0, 1.0, 0.0]  # 不相似
        
        await cache.add(embedding1, "response1")
        
        # 相似的应该匹配
        similar = await cache.get_similar(embedding2)
        assert similar is not None
        assert similar[0] == "response1"
        
        # 不相似的应该不匹配
        not_similar = await cache.get_similar(embedding3)
        assert not_similar is None
    
    @pytest.mark.asyncio
    async def test_size_and_clear(self, cache):
        """测试大小和清空"""
        assert cache.size() == 0
        
        await cache.add([1.0, 0.0], "response")
        assert cache.size() == 1
        
        count = cache.clear()
        assert count == 1
        assert cache.size() == 0


class TestCacheInvalidator:
    """缓存失效管理器测试"""
    
    @pytest.fixture
    def invalidator(self):
        return CacheInvalidator()
    
    def test_add_dependency(self, invalidator):
        """测试添加依赖"""
        invalidator.add_dependency("child_key", "parent_key")
        
        dependents = invalidator.get_dependents("parent_key")
        assert "child_key" in dependents
    
    def test_register_rule(self, invalidator):
        """测试注册规则"""
        # 规则：超过1小时的缓存应该失效
        def time_rule(key, entry):
            return entry.age > 3600
        
        invalidator.register_rule("*", time_rule)
        
        # 创建一个老条目
        old_entry = CacheEntry(
            key="test",
            value="value",
            created_at=time.time() - 4000,
        )
        
        assert invalidator.should_invalidate("test", old_entry)
        
        # 新条目不应该失效
        new_entry = CacheEntry(key="test", value="value")
        assert not invalidator.should_invalidate("test", new_entry)


class TestCostOptimizer:
    """成本优化器测试"""
    
    @pytest.fixture
    def optimizer(self):
        backend = LRUCacheBackend()
        cache = ResponseCache(backend)
        return CostOptimizer(cache)
    
    def test_estimate_cost(self, optimizer):
        """测试成本估算"""
        # GPT-4
        cost = optimizer.estimate_cost("gpt-4", 1000, 500)
        assert cost > 0
        
        # GPT-3.5
        cost_35 = optimizer.estimate_cost("gpt-3.5-turbo", 1000, 500)
        assert cost_35 < cost  # GPT-3.5 应该更便宜
    
    def test_track_cost(self, optimizer):
        """测试成本追踪"""
        # 追踪一次非缓存调用
        cost = optimizer.track_cost(
            operation="translate",
            model="gpt-3.5-turbo",
            input_tokens=100,
            output_tokens=50,
            cached=False,
        )
        assert cost > 0
        
        # 追踪一次缓存调用
        optimizer.track_cost(
            operation="translate",
            model="gpt-3.5-turbo",
            input_tokens=100,
            output_tokens=50,
            cached=True,
        )
        
        report = optimizer.get_cost_report()
        
        assert "translate" in report["operations"]
        assert report["operations"]["translate"]["call_count"] == 2
        assert report["operations"]["translate"]["cached_count"] == 1
        assert report["operations"]["translate"]["cache_savings"] > 0
    
    def test_get_cost_report(self, optimizer):
        """测试获取成本报告"""
        optimizer.track_cost("test", "gpt-3.5-turbo", 100, 50)
        
        report = optimizer.get_cost_report()
        
        assert "operations" in report
        assert "cache_stats" in report


class TestCachedDecorator:
    """缓存装饰器测试"""
    
    @pytest.mark.asyncio
    async def test_cached_decorator(self):
        """测试缓存装饰器"""
        call_count = 0
        backend = LRUCacheBackend()
        cache = ResponseCache(backend)
        
        @cached(cache, key_func=lambda x: f"key:{x}")
        async def expensive_function(x: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"result:{x}"
        
        # 第一次调用
        result1 = await expensive_function("test")
        assert result1 == "result:test"
        assert call_count == 1
        
        # 第二次调用（应该使用缓存）
        result2 = await expensive_function("test")
        assert result2 == "result:test"
        assert call_count == 1  # 没有增加
        
        # 不同参数
        result3 = await expensive_function("other")
        assert result3 == "result:other"
        assert call_count == 2


class TestConvenienceFunctions:
    """便捷函数测试"""
    
    def test_create_cache_backend(self):
        """测试创建缓存后端"""
        memory_backend = create_cache_backend(CacheBackendType.MEMORY, max_size=100)
        assert isinstance(memory_backend, LRUCacheBackend)
    
    def test_create_response_cache(self):
        """测试创建响应缓存"""
        cache = create_response_cache(
            backend_type=CacheBackendType.MEMORY,
            max_size=100,
            default_ttl=3600,
        )
        
        assert isinstance(cache, ResponseCache)
