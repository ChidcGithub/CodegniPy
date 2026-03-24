"""
Codegnipy 外部验证模块

提供外部验证集成，包括 Web 搜索验证、知识图谱查询、事实核查 API。
增强幻觉检测的准确率。
"""

import asyncio
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .runtime import CognitiveContext


class ExternalValidationStatus(Enum):
    """外部验证状态"""
    VERIFIED = "verified"        # 已验证为真
    REFUTED = "refuted"          # 已验证为假
    UNCERTAIN = "uncertain"      # 无法确定
    ERROR = "error"              # 验证出错
    UNAVAILABLE = "unavailable"  # 服务不可用


@dataclass
class Evidence:
    """验证证据"""
    source: str                  # 来源名称
    url: Optional[str] = None    # 来源 URL
    snippet: str = ""            # 相关片段
    relevance: float = 1.0       # 相关性评分 (0-1)
    supports_claim: Optional[bool] = None  # True=支持, False=反驳, None=不确定


@dataclass
class ExternalValidationResult:
    """外部验证结果"""
    claim: str
    status: ExternalValidationStatus
    confidence: float            # 0-1
    evidences: List[Evidence] = field(default_factory=list)
    summary: str = ""
    raw_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BaseValidator(ABC):
    """外部验证器抽象基类"""
    
    @abstractmethod
    async def validate_async(self, claim: str, context: Optional["CognitiveContext"] = None) -> ExternalValidationResult:
        """
        异步验证声明
        
        参数:
            claim: 要验证的声明文本
            context: 认知上下文（可选）
        返回:
            ExternalValidationResult 对象
        """
        pass
    
    def validate(self, claim: str, context: Optional["CognitiveContext"] = None) -> ExternalValidationResult:
        """
        同步验证声明
        
        参数:
            claim: 要验证的声明文本
            context: 认知上下文（可选）
        返回:
            ExternalValidationResult 对象
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            # 如果已有事件循环，创建新线程运行
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, 
                    self.validate_async(claim, context)
                )
                return future.result()
        else:
            return asyncio.run(self.validate_async(claim, context))
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查验证器是否可用"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """验证器名称"""
        pass


# ============ Web 搜索验证器 ============

class WebSearchValidator(BaseValidator):
    """
    Web 搜索验证器
    
    使用搜索引擎验证声明，支持 DuckDuckGo（免费）和 Bing API。
    """
    
    def __init__(
        self,
        engine: str = "duckduckgo",
        api_key: Optional[str] = None,
        max_results: int = 5,
        timeout: float = 10.0
    ):
        """
        初始化 Web 搜索验证器
        
        参数:
            engine: 搜索引擎 ("duckduckgo" 或 "bing")
            api_key: Bing API 密钥（仅 Bing 需要）
            max_results: 最大返回结果数
            timeout: 请求超时时间
        """
        self.engine = engine.lower()
        self.api_key = api_key
        self.max_results = max_results
        self.timeout = timeout
    
    @property
    def name(self) -> str:
        return f"web_search_{self.engine}"
    
    def is_available(self) -> bool:
        if self.engine == "duckduckgo":
            return True  # DuckDuckGo 无需 API 密钥
        elif self.engine == "bing":
            return self.api_key is not None
        return False
    
    async def validate_async(self, claim: str, context: Optional["CognitiveContext"] = None) -> ExternalValidationResult:
        """使用 Web 搜索验证声明"""
        if not self.is_available():
            return ExternalValidationResult(
                claim=claim,
                status=ExternalValidationStatus.UNAVAILABLE,
                confidence=0.0,
                error=f"验证器不可用: {self.name}"
            )
        
        try:
            # 提取搜索关键词
            keywords = self._extract_keywords(claim)
            search_query = " ".join(keywords)
            
            if self.engine == "duckduckgo":
                results = await self._search_duckduckgo(search_query)
            else:
                results = await self._search_bing(search_query)
            
            # 分析搜索结果
            evidences = self._analyze_results(claim, results)
            
            # 计算验证状态和置信度
            status, confidence, summary = self._compute_verdict(claim, evidences)
            
            return ExternalValidationResult(
                claim=claim,
                status=status,
                confidence=confidence,
                evidences=evidences,
                summary=summary,
                raw_response={"results": results}
            )
            
        except Exception as e:
            return ExternalValidationResult(
                claim=claim,
                status=ExternalValidationStatus.ERROR,
                confidence=0.0,
                error=str(e)
            )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 移除停用词（简单实现）
        stop_words = {"的", "是", "在", "和", "了", "有", "不", "这", "我", "他", "她", "它",
                     "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                     "have", "has", "had", "do", "does", "did", "will", "would", "could",
                     "should", "may", "might", "must", "can", "to", "of", "in", "for",
                     "on", "with", "at", "by", "from", "as", "into", "through"}
        
        # 分词（简单实现：按空格和标点分割）
        words = re.findall(r'[\w\u4e00-\u9fff]+', text.lower())
        
        # 过滤停用词并保留有意义的词
        keywords = [w for w in words if w not in stop_words and len(w) > 1]
        
        return keywords[:10]  # 最多返回 10 个关键词
    
    async def _search_duckduckgo(self, query: str) -> List[Dict[str, Any]]:
        """使用 DuckDuckGo 搜索"""
        import aiohttp
        
        # 使用 DuckDuckGo Instant Answer API
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=self.timeout) as response:
                data = await response.json()
        
        results = []
        
        # 解析相关主题
        if "RelatedTopics" in data:
            for topic in data["RelatedTopics"][:self.max_results]:
                if isinstance(topic, dict):
                    results.append({
                        "title": topic.get("Text", "")[:100],
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL", "")
                    })
        
        # 解析摘要
        if data.get("Abstract"):
            results.insert(0, {
                "title": data.get("Heading", "Summary"),
                "snippet": data.get("Abstract", ""),
                "url": data.get("AbstractURL", "")
            })
        
        return results
    
    async def _search_bing(self, query: str) -> List[Dict[str, Any]]:
        """使用 Bing 搜索 API"""
        import aiohttp
        
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {
            "q": query,
            "count": self.max_results,
            "responseFilter": "Webpages"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params, timeout=self.timeout) as response:
                data = await response.json()
        
        results = []
        if "webPages" in data and "value" in data["webPages"]:
            for item in data["webPages"]["value"]:
                results.append({
                    "title": item.get("name", ""),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("url", "")
                })
        
        return results
    
    def _analyze_results(self, claim: str, results: List[Dict[str, Any]]) -> List[Evidence]:
        """分析搜索结果"""
        evidences = []
        
        for result in results:
            snippet = result.get("snippet", "")
            title = result.get("title", "")
            
            # 简单的相关性判断
            relevance = self._compute_relevance(claim, title + " " + snippet)
            
            # 判断是否支持声明（简单实现）
            supports = self._check_support(claim, snippet)
            
            evidences.append(Evidence(
                source=result.get("title", "Unknown"),
                url=result.get("url"),
                snippet=snippet[:200] if snippet else "",
                relevance=relevance,
                supports_claim=supports
            ))
        
        return evidences
    
    def _compute_relevance(self, claim: str, text: str) -> float:
        """计算相关性"""
        claim_words = set(re.findall(r'[\w\u4e00-\u9fff]+', claim.lower()))
        text_words = set(re.findall(r'[\w\u4e00-\u9fff]+', text.lower()))
        
        if not claim_words:
            return 0.0
        
        intersection = claim_words & text_words
        return len(intersection) / len(claim_words)
    
    def _check_support(self, claim: str, text: str) -> Optional[bool]:
        """检查文本是否支持声明"""
        # 简单实现：检查否定词
        negation_patterns = [
            r'\b不\s*(正确|真实|存在|属实)\b',
            r'\b(false|fake|incorrect|wrong|not true)\b',
            r'\b谣言\b',
            r'\b辟谣\b'
        ]
        
        text_lower = text.lower()
        for pattern in negation_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return False
        
        # 检查确认词
        confirm_patterns = [
            r'\b(正确|真实|属实|确认)\b',
            r'\b(true|correct|confirmed|verified)\b'
        ]
        
        for pattern in confirm_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        return None  # 无法确定
    
    def _compute_verdict(
        self, 
        claim: str, 
        evidences: List[Evidence]
    ) -> tuple[ExternalValidationStatus, float, str]:
        """计算验证结论"""
        if not evidences:
            return ExternalValidationStatus.UNCERTAIN, 0.0, "未找到相关证据"
        
        # 加权计算
        support_score = 0.0
        refute_score = 0.0
        total_weight = 0.0
        
        for evidence in evidences:
            weight = evidence.relevance
            total_weight += weight
            
            if evidence.supports_claim is True:
                support_score += weight
            elif evidence.supports_claim is False:
                refute_score += weight
        
        if total_weight == 0:
            return ExternalValidationStatus.UNCERTAIN, 0.0, "无法确定相关性"
        
        support_ratio = support_score / total_weight
        refute_ratio = refute_score / total_weight
        
        # 生成摘要
        supporting = sum(1 for e in evidences if e.supports_claim is True)
        refuting = sum(1 for e in evidences if e.supports_claim is False)
        
        if support_ratio > 0.6:
            return (
                ExternalValidationStatus.VERIFIED,
                support_ratio,
                f"找到 {supporting} 个支持性证据，{refuting} 个反驳性证据"
            )
        elif refute_ratio > 0.4:
            return (
                ExternalValidationStatus.REFUTED,
                refute_ratio,
                f"找到 {refuting} 个反驳性证据，{supporting} 个支持性证据"
            )
        else:
            return (
                ExternalValidationStatus.UNCERTAIN,
                0.5,
                f"证据不足：{supporting} 个支持，{refuting} 个反驳，{len(evidences) - supporting - refuting} 个不确定"
            )


# ============ 知识图谱验证器 ============

class KnowledgeGraphValidator(BaseValidator):
    """
    知识图谱验证器
    
    使用 Wikidata SPARQL 查询验证实体和关系。
    """
    
    SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
    
    def __init__(self, timeout: float = 15.0, language: str = "zh"):
        """
        初始化知识图谱验证器
        
        参数:
            timeout: 请求超时时间
            language: 语言代码
        """
        self.timeout = timeout
        self.language = language
    
    @property
    def name(self) -> str:
        return "knowledge_graph_wikidata"
    
    def is_available(self) -> bool:
        return True  # Wikidata 是公开 API
    
    async def validate_async(self, claim: str, context: Optional["CognitiveContext"] = None) -> ExternalValidationResult:
        """使用知识图谱验证声明"""
        try:
            # 尝试提取实体
            entities = await self._extract_entities(claim)
            
            if not entities:
                return ExternalValidationResult(
                    claim=claim,
                    status=ExternalValidationStatus.UNCERTAIN,
                    confidence=0.3,
                    summary="无法从声明中提取已知实体"
                )
            
            # 查询实体信息
            evidences = []
            for entity_id, entity_label in entities[:3]:  # 最多查询 3 个实体
                entity_info = await self._query_entity(entity_id)
                if entity_info:
                    evidences.append(Evidence(
                        source=f"Wikidata: {entity_label}",
                        url=f"https://www.wikidata.org/wiki/{entity_id}",
                        snippet=entity_info.get("description", ""),
                        relevance=0.8,
                        supports_claim=None
                    ))
            
            if evidences:
                return ExternalValidationResult(
                    claim=claim,
                    status=ExternalValidationStatus.UNCERTAIN,
                    confidence=0.5,
                    evidences=evidences,
                    summary=f"找到 {len(evidences)} 个相关实体"
                )
            else:
                return ExternalValidationResult(
                    claim=claim,
                    status=ExternalValidationStatus.UNCERTAIN,
                    confidence=0.3,
                    summary="未在知识图谱中找到相关实体"
                )
                
        except Exception as e:
            return ExternalValidationResult(
                claim=claim,
                status=ExternalValidationStatus.ERROR,
                confidence=0.0,
                error=str(e)
            )
    
    async def _extract_entities(self, text: str) -> List[tuple[str, str]]:
        """从文本中提取 Wikidata 实体"""
        import aiohttp
        
        # 构建 SPARQL 查询搜索实体
        # 使用实体标签搜索
        query = f'''
        SELECT ?entity ?entityLabel WHERE {{
          ?entity ?label "{text}" .
          ?entity rdfs:label ?entityLabel .
          FILTER(LANG(?entityLabel) = "{self.language}")
        }}
        LIMIT 5
        '''
        
        # 如果文本较长，尝试搜索包含的实体
        words = re.findall(r'[\w\u4e00-\u9fff]+', text)
        if len(words) > 1:
            # 搜索最可能的实体词
            search_terms = words[:3]
            query = f'''
            SELECT ?entity ?entityLabel WHERE {{
              ?entity rdfs:label ?label .
              FILTER(LANG(?label) = "{self.language}")
              FILTER(CONTAINS(?label, "{search_terms[0]}") || 
                     ?label = "{search_terms[0]}")
            }}
            LIMIT 5
            '''
        
        headers = {"Accept": "application/sparql-results+json"}
        params = {"query": query, "format": "json"}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.SPARQL_ENDPOINT, 
                headers=headers, 
                params=params,
                timeout=self.timeout
            ) as response:
                data = await response.json()
        
        entities = []
        for binding in data.get("results", {}).get("bindings", []):
            entity_uri = binding.get("entity", {}).get("value", "")
            entity_label = binding.get("entityLabel", {}).get("value", "")
            if entity_uri:
                entity_id = entity_uri.split("/")[-1]
                entities.append((entity_id, entity_label))
        
        return entities
    
    async def _query_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """查询实体详细信息"""
        import aiohttp
        
        query = f'''
        SELECT ?description ?altLabel WHERE {{
          wd:{entity_id} schema:description ?description .
          OPTIONAL {{ wd:{entity_id} skos:altLabel ?altLabel . }}
          FILTER(LANG(?description) = "{self.language}")
          FILTER(LANG(?altLabel) = "{self.language}")
        }}
        LIMIT 1
        '''
        
        headers = {"Accept": "application/sparql-results+json"}
        params = {"query": query, "format": "json"}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.SPARQL_ENDPOINT,
                headers=headers,
                params=params,
                timeout=self.timeout
            ) as response:
                data = await response.json()
        
        bindings = data.get("results", {}).get("bindings", [])
        if bindings:
            return {
                "description": bindings[0].get("description", {}).get("value", ""),
                "alt_labels": [b.get("altLabel", {}).get("value", "") for b in bindings if b.get("altLabel")]
            }
        
        return None


# ============ 事实核查验证器 ============

class FactCheckValidator(BaseValidator):
    """
    事实核查验证器
    
    使用 Google Fact Check Tools API 验证声明。
    """
    
    API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    
    def __init__(self, api_key: Optional[str] = None, timeout: float = 10.0, max_results: int = 5):
        """
        初始化事实核查验证器
        
        参数:
            api_key: Google Fact Check API 密钥
            timeout: 请求超时时间
            max_results: 最大返回结果数
        """
        self.api_key = api_key
        self.timeout = timeout
        self.max_results = max_results
    
    @property
    def name(self) -> str:
        return "fact_check_google"
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    async def validate_async(self, claim: str, context: Optional["CognitiveContext"] = None) -> ExternalValidationResult:
        """使用事实核查 API 验证声明"""
        import aiohttp
        
        if not self.is_available():
            return ExternalValidationResult(
                claim=claim,
                status=ExternalValidationStatus.UNAVAILABLE,
                confidence=0.0,
                error="Google Fact Check API 密钥未配置"
            )
        
        try:
            params = {
                "query": claim,
                "key": self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.API_URL,
                    params=params,
                    timeout=self.timeout
                ) as response:
                    data = await response.json()
            
            # 解析事实核查结果
            evidences = []
            claims = data.get("claims", [])
            
            for claim_data in claims[:self.max_results]:
                claim_review = claim_data.get("claimReview", [])
                for review in claim_review:
                    rating = review.get("textualRating", "")
                    publisher = review.get("publisher", {}).get("name", "Unknown")
                    url = review.get("url", "")
                    
                    # 判断评级
                    supports = self._parse_rating(rating)
                    
                    evidences.append(Evidence(
                        source=f"Fact Check: {publisher}",
                        url=url,
                        snippet=f"Rating: {rating}",
                        relevance=0.9,
                        supports_claim=supports
                    ))
            
            if not evidences:
                return ExternalValidationResult(
                    claim=claim,
                    status=ExternalValidationStatus.UNCERTAIN,
                    confidence=0.3,
                    summary="未找到相关的事实核查报告"
                )
            
            # 计算结论
            status, confidence, summary = self._compute_verdict_from_checks(evidences)
            
            return ExternalValidationResult(
                claim=claim,
                status=status,
                confidence=confidence,
                evidences=evidences,
                summary=summary,
                raw_response=data
            )
            
        except Exception as e:
            return ExternalValidationResult(
                claim=claim,
                status=ExternalValidationStatus.ERROR,
                confidence=0.0,
                error=str(e)
            )
    
    def _parse_rating(self, rating: str) -> Optional[bool]:
        """解析事实核查评级"""
        rating_lower = rating.lower()
        
        # 真实
        true_patterns = ["true", "correct", "accurate", "真实", "正确"]
        for pattern in true_patterns:
            if pattern in rating_lower:
                return True
        
        # 虚假
        false_patterns = ["false", "fake", "incorrect", "pants on fire", "虚假", "错误", "谣言"]
        for pattern in false_patterns:
            if pattern in rating_lower:
                return False
        
        return None
    
    def _compute_verdict_from_checks(
        self, 
        evidences: List[Evidence]
    ) -> tuple[ExternalValidationStatus, float, str]:
        """从事实核查计算结论"""
        verified = sum(1 for e in evidences if e.supports_claim is True)
        refuted = sum(1 for e in evidences if e.supports_claim is False)
        uncertain = len(evidences) - verified - refuted
        
        if verified > refuted and verified > uncertain:
            return (
                ExternalValidationStatus.VERIFIED,
                0.7 + 0.1 * min(verified, 3),
                f"{verified} 个事实核查支持该声明"
            )
        elif refuted > verified:
            return (
                ExternalValidationStatus.REFUTED,
                0.7 + 0.1 * min(refuted, 3),
                f"{refuted} 个事实核查反驳该声明"
            )
        else:
            return (
                ExternalValidationStatus.UNCERTAIN,
                0.5,
                f"事实核查结果不一致：{verified} 支持，{refuted} 反驳，{uncertain} 不确定"
            )


# ============ 组合验证器 ============

class CompositeValidator(BaseValidator):
    """
    组合验证器
    
    组合多个验证器，综合评估结果。
    """
    
    def __init__(
        self, 
        validators: Optional[List[BaseValidator]] = None,
        strategy: str = "majority"
    ):
        """
        初始化组合验证器
        
        参数:
            validators: 验证器列表
            strategy: 组合策略 ("majority", "weighted", "any")
        """
        self.validators = validators or []
        self.strategy = strategy
    
    @property
    def name(self) -> str:
        return f"composite_{len(self.validators)}"
    
    def add_validator(self, validator: BaseValidator):
        """添加验证器"""
        self.validators.append(validator)
    
    def is_available(self) -> bool:
        return any(v.is_available() for v in self.validators)
    
    async def validate_async(self, claim: str, context: Optional["CognitiveContext"] = None) -> ExternalValidationResult:
        """使用所有验证器验证声明"""
        available_validators = [v for v in self.validators if v.is_available()]
        
        if not available_validators:
            return ExternalValidationResult(
                claim=claim,
                status=ExternalValidationStatus.UNAVAILABLE,
                confidence=0.0,
                error="没有可用的验证器"
            )
        
        # 并行执行所有验证器
        tasks = [v.validate_async(claim, context) for v in available_validators]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 收集结果
        valid_results: List[tuple[str, ExternalValidationResult]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                continue
            valid_results.append((available_validators[i].name, cast(ExternalValidationResult, result)))
        
        if not valid_results:
            return ExternalValidationResult(
                claim=claim,
                status=ExternalValidationStatus.ERROR,
                confidence=0.0,
                error="所有验证器都失败了"
            )
        
        # 根据策略组合结果
        return self._combine_results(claim, valid_results)
    
    def _combine_results(
        self, 
        claim: str, 
        results: List[tuple[str, ExternalValidationResult]]
    ) -> ExternalValidationResult:
        """组合多个验证结果"""
        all_evidences: List[Evidence] = []
        
        # 统计各状态
        status_counts = {
            ExternalValidationStatus.VERIFIED: 0,
            ExternalValidationStatus.REFUTED: 0,
            ExternalValidationStatus.UNCERTAIN: 0,
            ExternalValidationStatus.ERROR: 0
        }
        
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for name, result in results:
            status_counts[result.status] += 1
            all_evidences.extend(result.evidences)
            
            # 加权置信度
            weight = 1.0 if result.status != ExternalValidationStatus.ERROR else 0.0
            weighted_confidence += result.confidence * weight
            total_weight += weight
        
        # 根据策略确定最终状态
        if self.strategy == "majority":
            # 多数投票
            max_count = 0
            final_status = ExternalValidationStatus.UNCERTAIN
            for status, count in status_counts.items():
                if count > max_count and status != ExternalValidationStatus.ERROR:
                    max_count = count
                    final_status = status
        elif self.strategy == "any":
            # 只要有一个验证成功就返回
            for status in [ExternalValidationStatus.VERIFIED, ExternalValidationStatus.REFUTED]:
                if status_counts[status] > 0:
                    final_status = status
                    break
            else:
                final_status = ExternalValidationStatus.UNCERTAIN
        else:  # weighted
            # 加权平均
            verified_weight = status_counts[ExternalValidationStatus.VERIFIED]
            refuted_weight = status_counts[ExternalValidationStatus.REFUTED]
            
            if verified_weight > refuted_weight:
                final_status = ExternalValidationStatus.VERIFIED
            elif refuted_weight > verified_weight:
                final_status = ExternalValidationStatus.REFUTED
            else:
                final_status = ExternalValidationStatus.UNCERTAIN
        
        final_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0
        
        # 生成摘要
        summary_parts = []
        for name, result in results:
            if result.status != ExternalValidationStatus.ERROR:
                summary_parts.append(f"{name}: {result.status.value}")
        
        return ExternalValidationResult(
            claim=claim,
            status=final_status,
            confidence=final_confidence,
            evidences=all_evidences,
            summary="; ".join(summary_parts)
        )


# ============ 便捷函数 ============

def create_default_validator(
    duckduckgo: bool = True,
    bing_api_key: Optional[str] = None,
    wikidata: bool = True,
    fact_check_api_key: Optional[str] = None
) -> CompositeValidator:
    """
    创建默认的组合验证器
    
    参数:
        duckduckgo: 是否启用 DuckDuckGo 搜索
        bing_api_key: Bing API 密钥
        wikidata: 是否启用 Wikidata 知识图谱
        fact_check_api_key: Google Fact Check API 密钥
    返回:
        配置好的组合验证器
    """
    validators: List[BaseValidator] = []
    
    if duckduckgo:
        validators.append(WebSearchValidator(engine="duckduckgo"))
    
    if bing_api_key:
        validators.append(WebSearchValidator(engine="bing", api_key=bing_api_key))
    
    if wikidata:
        validators.append(KnowledgeGraphValidator())
    
    if fact_check_api_key:
        validators.append(FactCheckValidator(api_key=fact_check_api_key))
    
    return CompositeValidator(validators=validators)


def verify_claim(
    claim: str,
    validators: Optional[List[str]] = None,
    api_keys: Optional[Dict[str, str]] = None,
    context: Optional["CognitiveContext"] = None
) -> ExternalValidationResult:
    """
    验证声明的便捷函数
    
    参数:
        claim: 要验证的声明
        validators: 验证器类型列表 (["web", "knowledge", "fact_check"])
        api_keys: API 密钥字典 {"bing": "...", "fact_check": "..."}
        context: 认知上下文
    返回:
        ExternalValidationResult 对象
    """
    import os
    
    api_keys = api_keys or {}
    validators = validators or ["web", "knowledge"]
    
    validator_list: List[BaseValidator] = []
    
    if "web" in validators:
        bing_key = api_keys.get("bing") or os.environ.get("BING_API_KEY")
        if bing_key:
            validator_list.append(WebSearchValidator(engine="bing", api_key=bing_key))
        else:
            validator_list.append(WebSearchValidator(engine="duckduckgo"))
    
    if "knowledge" in validators:
        validator_list.append(KnowledgeGraphValidator())
    
    if "fact_check" in validators:
        fc_key = api_keys.get("fact_check") or os.environ.get("GOOGLE_FACT_CHECK_API_KEY")
        if fc_key:
            validator_list.append(FactCheckValidator(api_key=fc_key))
    
    composite = CompositeValidator(validators=validator_list)
    return composite.validate(claim, context)


async def verify_claim_async(
    claim: str,
    validators: Optional[List[str]] = None,
    api_keys: Optional[Dict[str, str]] = None,
    context: Optional["CognitiveContext"] = None
) -> ExternalValidationResult:
    """
    异步验证声明的便捷函数
    """
    import os
    
    api_keys = api_keys or {}
    validators = validators or ["web", "knowledge"]
    
    validator_list: List[BaseValidator] = []
    
    if "web" in validators:
        bing_key = api_keys.get("bing") or os.environ.get("BING_API_KEY")
        if bing_key:
            validator_list.append(WebSearchValidator(engine="bing", api_key=bing_key))
        else:
            validator_list.append(WebSearchValidator(engine="duckduckgo"))
    
    if "knowledge" in validators:
        validator_list.append(KnowledgeGraphValidator())
    
    if "fact_check" in validators:
        fc_key = api_keys.get("fact_check") or os.environ.get("GOOGLE_FACT_CHECK_API_KEY")
        if fc_key:
            validator_list.append(FactCheckValidator(api_key=fc_key))
    
    composite = CompositeValidator(validators=validator_list)
    return await composite.validate_async(claim, context)
