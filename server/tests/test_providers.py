import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from providers.base_adapter import BaseAdapter, ModelResponse
from providers.model_registry import choose_model_for_role, ROLE_MAPPING
from providers.mistral_adapter import MistralAdapter
from providers.deepseek_adapter import DeepSeekAdapter


class TestBaseAdapter:
    @patch.dict(os.environ, {'TEST_API_KEY': 'test_key'})
    def test_init_with_api_key(self):
        """Test BaseAdapter initialization with API key."""
        adapter = BaseAdapter(
            api_key_env='TEST_API_KEY',
            endpoint='https://api.test.com',
            role='test'
        )
        assert adapter.api_key == 'test_key'
        assert adapter.endpoint == 'https://api.test.com'
        assert adapter.role == 'test'

    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_api_key(self):
        """Test BaseAdapter initialization without API key."""
        adapter = BaseAdapter(
            api_key_env='MISSING_KEY',
            endpoint='https://api.test.com'
        )
        assert adapter.api_key is None

    @patch('providers.base_adapter.requests.post')
    def test_check_opa_policy_success(self, mock_post):
        """Test OPA policy check success."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'result': True}
        mock_post.return_value = mock_response

        adapter = BaseAdapter('TEST_KEY', 'https://api.test.com')
        result = adapter._check_opa_policy({'test': 'data'})
        assert result is True

    @patch('providers.base_adapter.requests.post')
    def test_check_opa_policy_failure(self, mock_post):
        """Test OPA policy check failure."""
        mock_post.side_effect = Exception('Connection error')

        adapter = BaseAdapter('TEST_KEY', 'https://api.test.com')
        result = adapter._check_opa_policy({'test': 'data'})
        assert result is True  # Fail open

    @patch('providers.base_adapter.BaseAdapter._check_opa_policy')
    @patch('providers.base_adapter.BaseAdapter._call_api')
    @pytest.mark.asyncio
    async def test_call_model_success(self, mock_call_api, mock_check_opa):
        """Test successful model call."""
        mock_check_opa.return_value = True
        mock_call_api.return_value = {'choices': [{'message': {'content': 'response'}}]}

        adapter = BaseAdapter('TEST_KEY', 'https://api.test.com')
        adapter.api_key = 'test_key'

        result = await adapter.call_model([{'role': 'user', 'content': 'test'}])

        assert isinstance(result, ModelResponse)
        assert result.text == 'response'

    @patch('providers.base_adapter.BaseAdapter._check_opa_policy')
    @pytest.mark.asyncio
    async def test_call_model_no_api_key(self, mock_check_opa):
        """Test model call without API key."""
        adapter = BaseAdapter('TEST_KEY', 'https://api.test.com')
        adapter.api_key = None

        result = await adapter.call_model([{'role': 'user', 'content': 'test'}])

        assert isinstance(result, ModelResponse)
        assert 'API key not found' in result.error

    @patch('providers.base_adapter.BaseAdapter._check_opa_policy')
    def test_call_model_opa_denied(self, mock_check_opa):
        """Test model call when OPA denies access."""
        mock_check_opa.return_value = False

        adapter = BaseAdapter('TEST_KEY', 'https://api.test.com')
        adapter.api_key = 'test_key'

        result = adapter.call_model([{'role': 'user', 'content': 'test'}])

        assert isinstance(result, ModelResponse)
        assert 'Access denied' in result.error


class TestModelRegistry:
    def test_choose_model_for_role_known(self):
        """Test choosing model for known role."""
        result = choose_model_for_role('fix_implementation')
        assert result in ROLE_MAPPING['fix_implementation']['models']

    def test_choose_model_for_role_unknown(self):
        """Test choosing model for unknown role."""
        result = choose_model_for_role('unknown_role')
        assert result == 'meta/llama3-70b-instruct'  # Default fallback when no models available

    def test_role_mapping_structure(self):
        """Test role mapping has expected structure."""
        assert 'fix_implementation' in ROLE_MAPPING
        assert 'models' in ROLE_MAPPING['fix_implementation']
        assert 'fallback' in ROLE_MAPPING['fix_implementation']


class TestMistralAdapter:
    @patch('providers.model_registry.choose_model_for_role')
    @patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_key'})
    def test_init(self, mock_choose_model):
        """Test MistralAdapter initialization."""
        mock_choose_model.return_value = 'mistral-model'
        adapter = MistralAdapter(role='test')
        assert adapter.role == 'test'
        assert adapter.api_key == 'test_key'
        mock_choose_model.assert_called_once_with('test')

    @patch('providers.mistral_adapter.requests.post')
    @patch('providers.model_registry.choose_model_for_role')
    @patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_key'})
    def test_call_api_success(self, mock_choose_model, mock_post):
        """Test Mistral API call success."""
        mock_choose_model.return_value = 'mistral-model'

        mock_response = MagicMock()
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'response'}}]
        }
        mock_post.return_value = mock_response

        adapter = MistralAdapter()
        result = adapter._call_api([{'role': 'user', 'content': 'test'}])

        assert result['choices'][0]['message']['content'] == 'response'

    @patch('providers.model_registry.choose_model_for_role')
    @patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_key'})
    def test_normalize_response_success(self, mock_choose_model):
        """Test response normalization success."""
        mock_choose_model.return_value = 'mistral-model'

        adapter = MistralAdapter()
        raw = {
            'choices': [{'message': {'content': 'normalized response'}}]
        }

        result = adapter._normalize_response(raw)

        assert isinstance(result, ModelResponse)
        assert result.text == 'normalized response'

    @patch('providers.model_registry.choose_model_for_role')
    @patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_key'})
    def test_normalize_response_missing_choices(self, mock_choose_model):
        """Test response normalization with missing choices."""
        mock_choose_model.return_value = 'mistral-model'

        adapter = MistralAdapter()
        raw = {}

        result = adapter._normalize_response(raw)

        assert isinstance(result, ModelResponse)
        assert result.text == ''


class TestDeepSeekAdapter:
    @patch('providers.deepseek_adapter.BaseAdapter.__init__')
    def test_init(self, mock_base_init):
        """Test DeepSeekAdapter initialization."""
        mock_base_init.return_value = None
        adapter = DeepSeekAdapter(role='test')
        mock_base_init.assert_called_once_with(
            api_key_env='DEEPSEEK_API_KEY',
            endpoint='https://api.deepseek.com/v1/chat/completions',
            role='test'
        )

    @patch('providers.deepseek_adapter.BaseAdapter.__init__')
    @patch('providers.deepseek_adapter.requests.post')
    @pytest.mark.asyncio
    async def test_call_api_success(self, mock_post, mock_base_init):
        """Test DeepSeek API call success."""
        mock_base_init.return_value = None

        mock_response = MagicMock()
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'deepseek response'}}]
        }
        mock_post.return_value = mock_response

        adapter = DeepSeekAdapter()
        adapter.api_key = 'test_key'

        result = adapter._call_api([{'role': 'user', 'content': 'test'}])

        assert result['choices'][0]['message']['content'] == 'deepseek response'

    @patch('providers.deepseek_adapter.BaseAdapter.__init__')
    @patch('providers.deepseek_adapter.DeepSeekAdapter._call_hf_fallback')
    @pytest.mark.asyncio
    async def test_call_model_with_fallback(self, mock_fallback, mock_base_init):
        """Test DeepSeek call_model with HuggingFace fallback."""
        mock_base_init.return_value = None

        mock_fallback.return_value = ModelResponse(text='fallback response')

        adapter = DeepSeekAdapter()
        adapter.api_key = 'test_key'

        # Mock the base call_model to raise exception
        with patch.object(BaseAdapter, 'call_model', side_effect=Exception('API error')):
            result = await adapter.call_model([{'role': 'user', 'content': 'test'}])

        assert result.text == 'fallback response'
        mock_fallback.assert_called_once()


class TestSystemPrompt:
    def test_system_prompt_loading(self):
        """Test system prompt loading."""
        from providers.system_prompt import SYSTEM_PROMPT, CANNED_RESPONSES

        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 0

        assert isinstance(CANNED_RESPONSES, dict)
        assert 'short' in CANNED_RESPONSES
        assert 'medium' in CANNED_RESPONSES
        assert 'detailed' in CANNED_RESPONSES