"""
多提供商支持模块测试
"""

import pytest
from codegnipy.providers import (
    ProviderType,
    ProviderConfig,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    TransformersProvider,
    LlamaCppProvider,
    QuantizationConfig,
    ProviderFactory,
    create_provider
)


class TestProviderType:
    """ProviderType 测试"""
    
    def test_provider_type_values(self):
        """测试提供商类型值"""
        assert ProviderType.OPENAI.value == "openai"
        assert ProviderType.ANTHROPIC.value == "anthropic"
        assert ProviderType.OLLAMA.value == "ollama"
        assert ProviderType.HUGGINGFACE.value == "huggingface"
        assert ProviderType.LLAMACPP.value == "llamacpp"
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


class TestOllamaProvider:
    """OllamaProvider 测试"""
    
    def test_provider_creation(self):
        """测试 Ollama 提供商创建"""
        config = ProviderConfig(
            provider_type=ProviderType.OLLAMA,
            model="llama2",
            base_url="http://localhost:11434"
        )
        
        provider = OllamaProvider(config)
        assert provider.config == config
        assert provider._base_url == "http://localhost:11434"
    
    def test_default_base_url(self):
        """测试默认 base_url"""
        config = ProviderConfig(
            provider_type=ProviderType.OLLAMA,
            model="llama2"
        )
        
        provider = OllamaProvider(config)
        assert provider._base_url == "http://localhost:11434"
    
    def test_message_conversion(self):
        """测试消息格式转换"""
        config = ProviderConfig(
            provider_type=ProviderType.OLLAMA,
            model="llama2"
        )
        provider = OllamaProvider(config)
        
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        prompt = provider._convert_messages(messages)
        
        assert "System: You are helpful." in prompt
        assert "User: Hello" in prompt
        assert "Assistant: Hi there!" in prompt
        assert "Assistant:" in prompt  # 结尾有助手提示
    
    def test_list_models_returns_list(self):
        """测试 list_models 返回列表（即使服务不可用）"""
        config = ProviderConfig(
            provider_type=ProviderType.OLLAMA,
            model="llama2",
            base_url="http://localhost:99999"  # 不存在的端口
        )
        
        provider = OllamaProvider(config)
        models = provider.list_models()
        
        # 服务不可用时返回空列表
        assert isinstance(models, list)


class TestTransformersProvider:
    """TransformersProvider 测试"""
    
    def test_provider_creation(self):
        """测试 Transformers 提供商创建"""
        config = ProviderConfig(
            provider_type=ProviderType.HUGGINGFACE,
            model="microsoft/DialoGPT-medium",
            extra_params={"device": "cpu"}
        )
        
        provider = TransformersProvider(config)
        assert provider.config == config
        assert provider._device == "cpu"
    
    def test_default_device(self):
        """测试默认设备"""
        config = ProviderConfig(
            provider_type=ProviderType.HUGGINGFACE,
            model="microsoft/DialoGPT-medium"
        )
        
        provider = TransformersProvider(config)
        assert provider._device == "auto"
    
    def test_message_conversion(self):
        """测试消息格式转换"""
        config = ProviderConfig(
            provider_type=ProviderType.HUGGINGFACE,
            model="microsoft/DialoGPT-medium"
        )
        provider = TransformersProvider(config)
        
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        prompt = provider._convert_messages(messages)
        
        assert "<<SYS>>" in prompt
        assert "<</SYS>>" in prompt
        assert "[INST]" in prompt


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
    
    def test_create_ollama(self):
        """测试创建 Ollama 提供商"""
        config = ProviderConfig(
            provider_type=ProviderType.OLLAMA,
            model="llama2"
        )
        
        provider = ProviderFactory.create(config)
        assert isinstance(provider, OllamaProvider)
    
    def test_create_huggingface(self):
        """测试创建 HuggingFace 提供商"""
        config = ProviderConfig(
            provider_type=ProviderType.HUGGINGFACE,
            model="microsoft/DialoGPT-medium"
        )
        
        provider = ProviderFactory.create(config)
        assert isinstance(provider, TransformersProvider)


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
    
    def test_create_ollama_provider(self):
        """测试创建 Ollama 提供商"""
        provider = create_provider(
            "ollama",
            model="llama2",
            base_url="http://localhost:11434"
        )
        
        assert isinstance(provider, OllamaProvider)
        assert provider.config.model == "llama2"
    
    def test_create_huggingface_provider(self):
        """测试创建 HuggingFace 提供商"""
        provider = create_provider(
            "huggingface",
            model="microsoft/DialoGPT-medium"
        )
        
        assert isinstance(provider, TransformersProvider)
        assert provider.config.model == "microsoft/DialoGPT-medium"
    
    def test_invalid_provider_type(self):
        """测试无效提供商类型"""
        with pytest.raises(ValueError):
            create_provider("invalid_provider")


class TestLlamaCppProvider:
    """LlamaCppProvider 测试"""
    
    def test_provider_creation(self):
        """测试 LlamaCpp 提供商创建"""
        config = ProviderConfig(
            provider_type=ProviderType.LLAMACPP,
            model="/path/to/model.gguf",
            extra_params={
                "n_ctx": 4096,
                "n_gpu_layers": 35,
                "n_threads": 4,
            }
        )
        
        provider = LlamaCppProvider(config)
        assert provider.config == config
        assert provider._n_ctx == 4096
        assert provider._n_gpu_layers == 35
        assert provider._n_threads == 4
    
    def test_default_params(self):
        """测试默认参数"""
        config = ProviderConfig(
            provider_type=ProviderType.LLAMACPP,
            model="/path/to/model.gguf"
        )
        
        provider = LlamaCppProvider(config)
        assert provider._n_ctx == 4096
        assert provider._n_gpu_layers == 0
        assert provider._n_threads == 4
        assert provider._verbose is False
    
    def test_message_conversion(self):
        """测试消息格式转换"""
        config = ProviderConfig(
            provider_type=ProviderType.LLAMACPP,
            model="/path/to/model.gguf"
        )
        provider = LlamaCppProvider(config)
        
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        prompt = provider._convert_messages(messages)
        
        assert "You are helpful." in prompt
        assert "Hello" in prompt
        assert "Hi there!" in prompt
    
    def test_format_chat(self):
        """测试聊天格式化"""
        config = ProviderConfig(
            provider_type=ProviderType.LLAMACPP,
            model="/path/to/model.gguf"
        )
        provider = LlamaCppProvider(config)
        
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant message"},
            {"role": "other", "content": "Ignored"},  # 应被忽略
        ]
        
        formatted = provider._format_chat(messages)
        
        assert len(formatted) == 3
        assert formatted[0]["role"] == "system"
        assert formatted[1]["role"] == "user"
        assert formatted[2]["role"] == "assistant"
    
    def test_get_model_info_without_model(self):
        """测试未加载模型时获取信息"""
        config = ProviderConfig(
            provider_type=ProviderType.LLAMACPP,
            model=""  # 空路径
        )
        provider = LlamaCppProvider(config)
        
        # 如果 llama_cpp 未安装，应该抛出 ImportError
        # 如果已安装，应该抛出 ValueError
        try:
            provider._load_model()
            pytest.fail("Expected an exception")
        except ImportError:
            pytest.skip("llama_cpp 包未安装，跳过测试")
        except ValueError:
            pass  # 期望的行为


class TestQuantizationConfig:
    """QuantizationConfig 测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = QuantizationConfig()
        
        assert config.method == "q4_k_m"
        assert config.bits == 4
        assert "K-quants" in config.description
    
    def test_custom_method(self):
        """测试自定义方法"""
        config = QuantizationConfig(method="q8_0")
        
        assert config.method == "q8_0"
        assert config.bits == 8
        assert "8-bit" in config.description
    
    def test_invalid_method(self):
        """测试无效方法"""
        with pytest.raises(ValueError):
            QuantizationConfig(method="invalid_method")
    
    def test_custom_bits(self):
        """测试自定义位数"""
        config = QuantizationConfig(method="q4_k_m", bits=4)
        
        assert config.bits == 4
    
    def test_group_size(self):
        """测试组大小"""
        config = QuantizationConfig(method="q4_k_m", group_size=128)
        
        assert config.group_size == 128
    
    def test_to_dict(self):
        """测试转换为字典"""
        config = QuantizationConfig(method="q5_k_m", group_size=64)
        
        result = config.to_dict()
        
        assert result["method"] == "q5_k_m"
        assert result["bits"] == 5
        assert result["group_size"] == 64
        assert "description" in result
    
    def test_list_methods(self):
        """测试列出方法"""
        methods = QuantizationConfig.list_methods()
        
        assert "q4_0" in methods
        assert "q4_k_m" in methods
        assert "q8_0" in methods
        assert "fp16" in methods
        assert "fp32" in methods
    
    def test_estimate_memory(self):
        """测试内存估算"""
        config = QuantizationConfig(method="q4_k_m")
        
        # 假设 7B 参数模型
        result = config.estimate_memory(model_params=7_000_000_000)
        
        assert "original_mb" in result
        assert "quantized_mb" in result
        assert "compression_ratio" in result
        assert result["compression_ratio"] > 1


class TestProviderFactoryLlamaCpp:
    """ProviderFactory LlamaCpp 测试"""
    
    def test_create_llamacpp(self):
        """测试创建 LlamaCpp 提供商"""
        config = ProviderConfig(
            provider_type=ProviderType.LLAMACPP,
            model="/path/to/model.gguf"
        )
        
        provider = ProviderFactory.create(config)
        assert isinstance(provider, LlamaCppProvider)


class TestCreateProviderLlamaCpp:
    """create_provider LlamaCpp 函数测试"""
    
    def test_create_llamacpp_provider(self):
        """测试创建 LlamaCpp 提供商"""
        provider = create_provider(
            "llamacpp",
            model="/path/to/model.gguf"
        )
        
        assert isinstance(provider, LlamaCppProvider)
        assert provider.config.model == "/path/to/model.gguf"
    
    def test_create_llamacpp_provider_alias(self):
        """测试 llama.cpp 别名"""
        provider = create_provider(
            "llama.cpp",
            model="/path/to/model.gguf"
        )
        
        assert isinstance(provider, LlamaCppProvider)
