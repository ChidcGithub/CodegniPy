"""
多提供商支持模块测试
"""

import pytest
from cognipy.providers import (
    ProviderType,
    ProviderConfig,
    OpenAIProvider,
    AnthropicProvider,
    ProviderFactory,
    create_provider
)


class TestProviderType:
    """ProviderType 测试"""
    
    def test_provider_type_values(self):
        """测试提供商类型值"""
        assert ProviderType.OPENAI.value == "openai"
        assert ProviderType.ANTHROPIC.value == "anthropic"
        assert ProviderType.CUSTOM.value == "custom"


class TestProviderConfig:
    """ProviderConfig 测试"""
    
    def test_config_creation(self):
        """测试配置创建"""
        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            api_key="test-key",
            model="gpt-4",
            temperature=0.5,
            max_tokens=2048
        )
        
        assert config.provider_type == ProviderType.OPENAI
        assert config.api_key == "test-key"
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048
    
    def test_config_defaults(self):
        """测试配置默认值"""
        config = ProviderConfig()
        
        assert config.provider_type == ProviderType.OPENAI
        assert config.api_key is None
        assert config.model == ""
        assert config.temperature == 0.7
        assert config.max_tokens == 1024


class TestOpenAIProvider:
    """OpenAIProvider 测试"""
    
    def test_provider_creation(self):
        """测试 OpenAI 提供商创建"""
        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            api_key="test-key",
            model="gpt-4o-mini"
        )
        
        provider = OpenAIProvider(config)
        assert provider.config == config


class TestAnthropicProvider:
    """AnthropicProvider 测试"""
    
    def test_provider_creation(self):
        """测试 Anthropic 提供商创建"""
        config = ProviderConfig(
            provider_type=ProviderType.ANTHROPIC,
            api_key="test-key",
            model="claude-3-opus"
        )
        
        provider = AnthropicProvider(config)
        assert provider.config == config
    
    def test_message_conversion(self):
        """测试消息格式转换"""
        config = ProviderConfig(
            provider_type=ProviderType.ANTHROPIC,
            model="claude-3-opus"
        )
        provider = AnthropicProvider(config)
        
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        system, converted = provider._convert_messages(messages)
        
        assert system == "You are helpful."
        assert len(converted) == 2
        assert converted[0]["role"] == "user"
        assert converted[1]["role"] == "assistant"


class TestProviderFactory:
    """ProviderFactory 测试"""
    
    def test_create_openai(self):
        """测试创建 OpenAI 提供商"""
        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            model="gpt-4"
        )
        
        provider = ProviderFactory.create(config)
        assert isinstance(provider, OpenAIProvider)
    
    def test_create_anthropic(self):
        """测试创建 Anthropic 提供商"""
        config = ProviderConfig(
            provider_type=ProviderType.ANTHROPIC,
            model="claude-3-opus"
        )
        
        provider = ProviderFactory.create(config)
        assert isinstance(provider, AnthropicProvider)


class TestCreateProvider:
    """create_provider 函数测试"""
    
    def test_create_openai_provider(self):
        """测试创建 OpenAI 提供商"""
        provider = create_provider(
            "openai",
            api_key="test-key",
            model="gpt-4"
        )
        
        assert isinstance(provider, OpenAIProvider)
        assert provider.config.api_key == "test-key"
        assert provider.config.model == "gpt-4"
    
    def test_create_anthropic_provider(self):
        """测试创建 Anthropic 提供商"""
        provider = create_provider(
            "anthropic",
            api_key="test-key",
            model="claude-3-opus"
        )
        
        assert isinstance(provider, AnthropicProvider)
        assert provider.config.model == "claude-3-opus"
    
    def test_invalid_provider_type(self):
        """测试无效提供商类型"""
        with pytest.raises(ValueError):
            create_provider("invalid_provider")
