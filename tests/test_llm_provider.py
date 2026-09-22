"""Comprehensive tests for LLM Provider abstraction (Ollama, Gemini, and Hybrid modes)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    DEFAULT_API_CONNECT_OPTIONS,
    llm,
)
from livekit.agents.llm import ChatContext, function_tool
from livekit.agents.llm.fallback_adapter import FallbackAdapter
from livekit.plugins import google

from zcloud_ai_telecaller.llm_provider import (
    HybridLLM,
    OllamaLLM,
    OllamaLLMStream,
    OllamaThinkingFilter,
    create_llm_provider,
)


class MockStreamContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockFailingStreamContext:
    def __init__(self, exc):
        self.exc = exc

    async def __aenter__(self):
        raise self.exc

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class DummyGeminiLLM(llm.LLM):
    def __init__(self):
        super().__init__()
        self._label = "dummy.gemini"

    @property
    def model(self):
        return "gemini-2.5-flash"

    @property
    def provider(self):
        return "google"

    def chat(self, *, chat_ctx, tools=None, conn_options=DEFAULT_API_CONNECT_OPTIONS, **kwargs):
        class Stream(llm.LLMStream):
            async def _run(stream_self):
                stream_self._event_ch.send_nowait(
                    llm.ChatChunk(
                        id="fallback-1",
                        delta=llm.ChoiceDelta(role="assistant", content="Fallback from Gemini!"),
                    )
                )

        return Stream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


def test_thinking_filter_basic():
    """Verify that <think> tags are stripped cleanly."""
    tf = OllamaThinkingFilter()
    assert tf.filter_chunk("<think>Reasoning...</think>Hello, how can I help?") == "Hello, how can I help?"


def test_thinking_filter_streaming_chunks():
    """Verify that thinking tags split across multiple streaming chunks are filtered."""
    tf = OllamaThinkingFilter()
    c1 = tf.filter_chunk("Intro <think>Part 1 of thought")
    c2 = tf.filter_chunk(" and part 2 of thought</think>Actual response.")
    assert c1 == "Intro "
    assert c2 == "Actual response."


def test_thinking_filter_clean_text():
    """Verify that regular text without thinking tags passes through unchanged."""
    tf = OllamaThinkingFilter()
    assert tf.filter_chunk("Good morning! Clinic is open today.") == "Good morning! Clinic is open today."


def test_ollama_llm_initialization():
    """Verify OllamaLLM properties and defaults."""
    model = OllamaLLM(model="qwen3:4b", base_url="http://localhost:11434", timeout=45.0)
    assert model.model == "qwen3:4b"
    assert model.provider == "ollama"
    assert model.base_url == "http://localhost:11434"
    assert model.timeout == 45.0


def test_create_llm_provider_local_ai():
    """Verify factory returns OllamaLLM for LOCAL_AI mode."""
    provider = create_llm_provider(ai_mode="LOCAL_AI", model="qwen3:4b")
    assert isinstance(provider, OllamaLLM)
    assert provider.model == "qwen3:4b"
    assert provider.provider == "ollama"


def test_create_llm_provider_cloud_ai():
    """Verify factory returns google.LLM for CLOUD_AI mode."""
    provider = create_llm_provider(ai_mode="CLOUD_AI", api_key="dummy-key")
    assert isinstance(provider, google.LLM)
    assert provider.model == "gemini-2.5-flash"


def test_create_llm_provider_hybrid():
    """Verify factory returns HybridLLM with fallback adapter for HYBRID mode."""
    provider = create_llm_provider(
        ai_mode="HYBRID",
        model="qwen3:4b",
        api_key="dummy-key",
    )
    assert isinstance(provider, HybridLLM)
    assert isinstance(provider, FallbackAdapter)
    assert provider.provider == "hybrid"


def test_create_llm_provider_auto_inference():
    """Verify provider auto-detection from model name."""
    ollama_prov = create_llm_provider(model="qwen3:4b")
    assert isinstance(ollama_prov, OllamaLLM)

    gemini_prov = create_llm_provider(model="gemini-2.5-flash", api_key="dummy-key")
    assert isinstance(gemini_prov, google.LLM)


@pytest.mark.asyncio
async def test_ollama_stream_mocked_success():
    """Verify OllamaLLMStream yields speech chunks and usage metrics."""
    llm_instance = OllamaLLM(model="qwen3:4b")
    ctx = ChatContext()
    ctx.add_message(role="user", content="Hello")

    stream_lines = [
        json.dumps({"message": {"role": "assistant", "content": "Hello, "}, "done": False}),
        json.dumps({"message": {"role": "assistant", "content": "world!"}, "done": False}),
        json.dumps({"done": True, "prompt_eval_count": 10, "eval_count": 5}),
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200

    async def mock_aiter_lines():
        for line in stream_lines:
            yield line

    mock_response.aiter_lines = mock_aiter_lines

    mock_client = MagicMock()
    mock_client.stream.side_effect = lambda *args, **kwargs: MockStreamContext(mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        stream = llm_instance.chat(chat_ctx=ctx)
        collected = await stream.collect()

        assert collected.text == "Hello, world!"
        assert collected.usage is not None
        assert collected.usage.prompt_tokens == 10
        assert collected.usage.completion_tokens == 5
        assert collected.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_ollama_stream_tool_calling():
    """Verify OllamaLLMStream properly extracts and parses function tool calls."""
    llm_instance = OllamaLLM(model="qwen3:4b")

    @function_tool
    def book_slot(date: str) -> str:
        """Book a slot."""
        return "Booked"

    ctx = ChatContext()
    ctx.add_message(role="user", content="Book tomorrow")

    tool_call_payload = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "function": {
                        "name": "book_slot",
                        "arguments": {"date": "2026-09-23"},
                    },
                }
            ],
        },
        "done": True,
    }

    mock_response = MagicMock()
    mock_response.status_code = 200

    async def mock_aiter_lines():
        yield json.dumps(tool_call_payload)

    mock_response.aiter_lines = mock_aiter_lines

    mock_client = MagicMock()
    mock_client.stream.side_effect = lambda *args, **kwargs: MockStreamContext(mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        stream = llm_instance.chat(chat_ctx=ctx, tools=[book_slot])
        collected = await stream.collect()

        assert len(collected.tool_calls) == 1
        tc = collected.tool_calls[0]
        assert tc.name == "book_slot"
        assert tc.call_id == "call_abc123"
        parsed_args = json.loads(tc.arguments)
        assert parsed_args == {"date": "2026-09-23"}


@pytest.mark.asyncio
async def test_ollama_stream_connection_error():
    """Verify that network connection failures raise APIConnectionError."""
    llm_instance = OllamaLLM(model="qwen3:4b", base_url="http://127.0.0.1:9999")
    ctx = ChatContext()
    ctx.add_message(role="user", content="Hello")

    mock_client = MagicMock()
    mock_client.stream.side_effect = lambda *args, **kwargs: MockFailingStreamContext(
        httpx.ConnectError("Connection refused")
    )
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        stream = llm_instance.chat(chat_ctx=ctx)
        with pytest.raises(APIConnectionError):
            await stream.collect()


@pytest.mark.asyncio
async def test_ollama_stream_timeout_error():
    """Verify that timeouts raise APITimeoutError."""
    llm_instance = OllamaLLM(model="qwen3:4b", timeout=5.0)
    ctx = ChatContext()
    ctx.add_message(role="user", content="Hello")

    mock_client = MagicMock()
    mock_client.stream.side_effect = lambda *args, **kwargs: MockFailingStreamContext(
        httpx.TimeoutException("Read timed out")
    )
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        stream = llm_instance.chat(chat_ctx=ctx)
        with pytest.raises(APITimeoutError):
            await stream.collect()


@pytest.mark.asyncio
async def test_hybrid_fallback_execution():
    """Verify that HybridLLM automatically switches to fallback when primary fails."""
    primary_failing = OllamaLLM(model="qwen3:4b", timeout=1.0)
    fallback_gemini = DummyGeminiLLM()

    hybrid = HybridLLM(primary_ollama=primary_failing, fallback_gemini=fallback_gemini)

    ctx = ChatContext()
    ctx.add_message(role="user", content="Test fallback")

    mock_client = MagicMock()
    mock_client.stream.side_effect = lambda *args, **kwargs: MockFailingStreamContext(
        httpx.ConnectError("Ollama down")
    )
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        stream = hybrid.chat(chat_ctx=ctx)
        collected = await stream.collect()
        assert collected.text == "Fallback from Gemini!"
