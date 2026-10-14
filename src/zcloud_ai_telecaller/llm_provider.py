"""LLM Provider Abstraction for Z Cloud AI Telecaller.

Provides seamless switching between Local AI (Ollama), Cloud AI (Gemini), and
Hybrid modes (Ollama primary with Gemini fallback).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Any, AsyncIterator, Literal, Optional, Sequence, cast

import httpx
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    DEFAULT_API_CONNECT_OPTIONS,
    llm,
    utils,
)
from livekit.agents.llm import ChatChunk, ChatContext, ChoiceDelta, CompletionUsage, FunctionToolCall, LLMStream, Tool
from livekit.agents.llm.fallback_adapter import FallbackAdapter
from livekit.plugins import google

logger = logging.getLogger("zcloud_ai_telecaller.llm_provider")


class OllamaThinkingFilter:
    """Filters out <think>...</think> reasoning blocks from text streams.

    Prevents internal model reasoning from being spoken aloud by TTS
    or leaked into customer-facing dialogue transcripts.
    """

    def __init__(self) -> None:
        self._inside_think = False
        self._buffer = ""

    def filter_chunk(self, text: str) -> str:
        """Processes an incoming text fragment and returns only non-thinking content."""
        if not text:
            return ""

        input_text = self._buffer + text
        self._buffer = ""
        output_parts: list[str] = []
        i = 0
        n = len(input_text)

        while i < n:
            if not self._inside_think:
                start_idx = input_text.find("<think>", i)
                if start_idx == -1:
                    # Check for partial tag at end (e.g. "<th", "<think")
                    tag_start = input_text.rfind("<", i)
                    if tag_start != -1 and "<think>".startswith(input_text[tag_start:]):
                        output_parts.append(input_text[i:tag_start])
                        self._buffer = input_text[tag_start:]
                        break
                    output_parts.append(input_text[i:])
                    break
                else:
                    output_parts.append(input_text[i:start_idx])
                    self._inside_think = True
                    i = start_idx + len("<think>")
            else:
                end_idx = input_text.find("</think>", i)
                if end_idx == -1:
                    # Check for partial closing tag
                    tag_start = input_text.rfind("<", i)
                    if tag_start != -1 and "</think>".startswith(input_text[tag_start:]):
                        self._buffer = input_text[tag_start:]
                        break
                    # Still inside thinking block, discard
                    break
                else:
                    self._inside_think = False
                    i = end_idx + len("</think>")

        return "".join(output_parts)


class OllamaLLMStream(LLMStream):
    """Async streaming response handler for Ollama's HTTP chat endpoint."""

    def __init__(
        self,
        llm_instance: OllamaLLM,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool],
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(
            llm_instance,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )
        self._ollama = llm_instance
        self._thinking_filter = OllamaThinkingFilter()

    def _prepare_messages(self) -> list[dict[str, Any]]:
        """Converts ChatContext messages into Ollama-compatible message dictionaries."""
        raw_messages, _ = self._chat_ctx.to_provider_format("openai")
        prepared: list[dict[str, Any]] = []

        for msg in raw_messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            item: dict[str, Any] = {"role": role, "content": str(content)}

            # Preserve tool_calls from assistant turns
            if "tool_calls" in msg and msg["tool_calls"]:
                item["tool_calls"] = msg["tool_calls"]

            # Preserve tool results
            if role == "tool":
                item["role"] = "tool"
                if "tool_call_id" in msg:
                    item["tool_call_id"] = msg["tool_call_id"]

            prepared.append(item)

        return prepared

    def _prepare_tools(self) -> Optional[list[dict[str, Any]]]:
        """Converts LiveKit function tools into Ollama tools format."""
        if not self._tools:
            return None
        tool_ctx = llm.ToolContext(self._tools)
        return tool_ctx.parse_function_tools("openai")

    async def _run(self) -> None:
        request_id = f"ollama_{utils.shortuuid()}"
        endpoint = f"{self._ollama.base_url.rstrip('/')}/api/chat"
        messages = self._prepare_messages()
        tools_def = self._prepare_tools()

        payload: dict[str, Any] = {
            "model": self._ollama.model,
            "messages": messages,
            "stream": True,
        }
        if tools_def:
            payload["tools"] = tools_def

        # Optimize options for real-time voice telecalling latency
        options: dict[str, Any] = {
            "num_ctx": 2048,
            "temperature": self._ollama.temperature if self._ollama.temperature is not None else 0.4,
        }
        payload["options"] = options
        payload["keep_alive"] = "1h"

        timeout = httpx.Timeout(self._ollama.timeout, connect=10.0)

        response_generated = False
        prompt_eval_count = 0
        eval_count = 0

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", endpoint, json=payload) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        raise APIStatusError(
                            f"Ollama returned HTTP {response.status_code}: {body.decode(errors='replace')}",
                            status_code=response.status_code,
                            request_id=request_id,
                            body=body.decode(errors="replace"),
                        )

                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # Read token usage stats if available
                        if "prompt_eval_count" in chunk:
                            prompt_eval_count = chunk["prompt_eval_count"]
                        if "eval_count" in chunk:
                            eval_count = chunk["eval_count"]

                        msg = chunk.get("message", {})

                        # 1. Check for tool calls
                        tool_calls_raw = msg.get("tool_calls")
                        if tool_calls_raw:
                            parsed_calls: list[FunctionToolCall] = []
                            for tc in tool_calls_raw:
                                fn = tc.get("function", {})
                                fn_name = fn.get("name", "")
                                fn_args = fn.get("arguments", {})
                                if isinstance(fn_args, (dict, list)):
                                    fn_args_str = json.dumps(fn_args)
                                else:
                                    fn_args_str = str(fn_args)

                                call_id = tc.get("id") or f"call_{utils.shortuuid()}"
                                parsed_calls.append(
                                    FunctionToolCall(
                                        type="function",
                                        name=fn_name,
                                        arguments=fn_args_str,
                                        call_id=call_id,
                                    )
                                )

                            if parsed_calls:
                                response_generated = True
                                self._event_ch.send_nowait(
                                    ChatChunk(
                                        id=request_id,
                                        delta=ChoiceDelta(
                                            role="assistant",
                                            content="",
                                            tool_calls=parsed_calls,
                                        ),
                                    )
                                )

                        # 2. Check for speech content (filtering out <think> tags)
                        raw_content = msg.get("content", "")
                        if raw_content:
                            clean_text = self._thinking_filter.filter_chunk(raw_content)
                            if clean_text:
                                response_generated = True
                                self._event_ch.send_nowait(
                                    ChatChunk(
                                        id=request_id,
                                        delta=ChoiceDelta(
                                            role="assistant",
                                            content=clean_text,
                                        ),
                                    )
                                )

                        # 3. Handle stream completion
                        if chunk.get("done"):
                            if prompt_eval_count or eval_count:
                                self._event_ch.send_nowait(
                                    ChatChunk(
                                        id=request_id,
                                        usage=CompletionUsage(
                                            completion_tokens=eval_count,
                                            prompt_tokens=prompt_eval_count,
                                            total_tokens=prompt_eval_count + eval_count,
                                        ),
                                    )
                                )
                            break

            if not response_generated:
                raise APIStatusError(
                    "Ollama produced empty response",
                    request_id=request_id,
                )

        except httpx.ConnectError as exc:
            logger.warning("Ollama connection failed at %s: %s", endpoint, exc)
            raise APIConnectionError(
                f"Cannot connect to Ollama at {endpoint}: {exc}",
            ) from exc
        except httpx.TimeoutException as exc:
            logger.warning("Ollama request timed out after %ss: %s", self._ollama.timeout, exc)
            raise APITimeoutError(
                f"Ollama request timed out after {self._ollama.timeout}s: {exc}",
                retryable=False,
            ) from exc
        except (APIError, APIStatusError, APIConnectionError, APITimeoutError):
            raise
        except Exception as exc:
            logger.error("Unexpected error communicating with Ollama: %s", exc)
            raise APIConnectionError(
                f"Ollama unexpected error: {exc}",
            ) from exc


class OllamaLLM(llm.LLM):
    """LiveKit Agents compatible Local LLM implementation powered by Ollama."""

    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 60.0,
        temperature: Optional[float] = None,
    ) -> None:
        super().__init__()
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._temperature = temperature
        self._label = f"ollama.{model}"

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "ollama"

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def timeout(self) -> float:
        return self._timeout

    @property
    def temperature(self) -> Optional[float]:
        return self._temperature

    async def _prewarm_impl(self) -> None:
        """Verifies Ollama connectivity and model availability without generating tokens."""
        url = f"{self._base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    logger.info("Ollama pre-warm check passed at %s (model=%s)", self._base_url, self._model)
        except Exception as exc:
            logger.debug("Ollama pre-warm ping failed: %s (will retry on first chat)", exc)

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: Optional[list[Tool]] = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: Any = None,
        tool_choice: Any = None,
        extra_kwargs: Any = None,
    ) -> LLMStream:
        return OllamaLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )


class HybridLLM(FallbackAdapter):
    """Hybrid LLM that routes to local Ollama by default and falls back to Gemini on error."""

    def __init__(
        self,
        primary_ollama: OllamaLLM,
        fallback_gemini: llm.LLM,
        attempt_timeout: float = 30.0,
    ) -> None:
        super().__init__(
            llm=[primary_ollama, fallback_gemini],
            attempt_timeout=attempt_timeout,
            max_retry_per_llm=0,
            retry_interval=0.5,
        )
        self._primary = primary_ollama
        self._fallback = fallback_gemini
        self._label = f"hybrid({primary_ollama.model} -> {fallback_gemini.model})"

    @property
    def model(self) -> str:
        return f"hybrid({self._primary.model}/{self._fallback.model})"

    @property
    def provider(self) -> str:
        return "hybrid"


VERIFIED_GEMINI_REALTIME_MODELS = frozenset({
    "gemini-3.8-live",
    "gemini-2.5-flash-native-audio-preview-12-2025",
    "gemini-3.1-flash-live-preview",
    "gemini-3.8-live-extended-thinking",
})
DEFAULT_GEMINI_REALTIME_MODEL = "gemini-3.8-live"
DEFAULT_GEMINI_TEXT_MODEL = "gemini-2.5-flash"


def create_llm_provider(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 60.0,
    temperature: Optional[float] = None,
    ai_mode: Optional[str] = None,
    use_realtime: Optional[bool] = None,
) -> llm.LLM | Any:
    """Factory creating the appropriate LLM provider based on runtime settings.

    Modes supported:
        - LOCAL_AI / ollama: Uses local Ollama instance (₹0 API cost).
        - CLOUD_AI / gemini: Uses Google Gemini 2.5 Flash cloud intelligence.
        - HYBRID: Uses Ollama primary with automatic Gemini fallback.
    """
    clean_provider = (provider or "").strip().lower()
    clean_mode = (ai_mode or "").strip().upper()

    # Determine effective mode
    if clean_mode == "LOCAL_AI" or clean_provider in ("ollama", "local"):
        chosen_provider = "ollama"
    elif clean_mode == "CLOUD_AI" or clean_provider in ("gemini", "cloud", "google"):
        chosen_provider = "gemini"
    elif clean_mode == "HYBRID" or clean_provider == "hybrid":
        chosen_provider = "hybrid"
    else:
        # Infer provider from model name if not explicitly stated
        eff_model = (model or "").lower()
        if eff_model.startswith("gemini"):
            chosen_provider = "gemini"
        elif any(eff_model.startswith(prefix) for prefix in ("qwen", "llama", "mistral", "phi", "gemma", "ollama/")):
            chosen_provider = "ollama"
        else:
            chosen_provider = "gemini"

    # 1. Local AI (Ollama)
    if chosen_provider == "ollama":
        eff_model = model or "qwen3:4b"
        if eff_model.startswith("ollama/"):
            eff_model = eff_model[len("ollama/"):]
        eff_base_url = base_url or os.environ.get("OLLAMA_BASE_URL") or "http://127.0.0.1:11434"
        logger.info("Initializing LOCAL_AI LLM provider: Ollama (model=%s, url=%s)", eff_model, eff_base_url)
        return OllamaLLM(
            model=eff_model,
            base_url=eff_base_url,
            timeout=timeout,
            temperature=temperature,
        )

    # 2. Cloud AI (Gemini)
    if chosen_provider == "gemini":
        eff_model = model or (DEFAULT_GEMINI_REALTIME_MODEL if use_realtime else DEFAULT_GEMINI_TEXT_MODEL)
        eff_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "dummy-key-for-initialization"

        if use_realtime is False:
            is_realtime_requested = False
        else:
            is_realtime_requested = (
                eff_model in VERIFIED_GEMINI_REALTIME_MODELS
                or eff_model.startswith("gemini-live")
                or eff_model.endswith("-live")
                or "native-audio" in eff_model
                or clean_provider in ("gemini-live", "live", "realtime")
                or use_realtime is True
            )
        if is_realtime_requested:
            # Ensure model is a verified Gemini Live model; never pass a text model like gemini-2.5-flash to bidiGenerateContent
            if eff_model in VERIFIED_GEMINI_REALTIME_MODELS or eff_model.startswith("gemini-live") or eff_model.endswith("-live") or "native-audio" in eff_model:
                realtime_model = eff_model
            else:
                realtime_model = DEFAULT_GEMINI_REALTIME_MODEL
                logger.info(
                    "Realtime audio requested with model '%s'. Mapping to verified Gemini Live model '%s'.",
                    eff_model,
                    realtime_model,
                )

            logger.info("Initializing CLOUD_AI Gemini Live RealtimeModel (model=%s)", realtime_model)
            from livekit.plugins.google import realtime
            return realtime.RealtimeModel(
                model=realtime_model,
                api_key=eff_key,
                temperature=temperature if temperature is not None else 0.5,
            )

        # Standard text LLM (generateContent) for cascaded STT + LLM + TTS pipeline
        text_model = eff_model if eff_model not in VERIFIED_GEMINI_REALTIME_MODELS and not eff_model.endswith("-live") else DEFAULT_GEMINI_TEXT_MODEL
        logger.info("Initializing CLOUD_AI LLM provider: Google Gemini (model=%s)", text_model)
        return google.LLM(
            model=text_model,
            api_key=eff_key,
        )

    # 3. Hybrid AI (Ollama primary with Gemini fallback)
    eff_ollama_model = model if model and not model.startswith("gemini") else "qwen3:4b"
    if eff_ollama_model.startswith("ollama/"):
        eff_ollama_model = eff_ollama_model[len("ollama/"):]
    eff_base_url = base_url or os.environ.get("OLLAMA_BASE_URL") or "http://127.0.0.1:11434"
    eff_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "dummy-key-for-initialization"

    logger.info(
        "Initializing HYBRID LLM provider: Primary Ollama (%s) with Gemini fallback",
        eff_ollama_model,
    )
    ollama_primary = OllamaLLM(
        model=eff_ollama_model,
        base_url=eff_base_url,
        timeout=timeout,
        temperature=temperature,
    )
    gemini_fallback = google.LLM(
        model="gemini-2.5-flash",
        api_key=eff_key,
    )
    return HybridLLM(
        primary_ollama=ollama_primary,
        fallback_gemini=gemini_fallback,
        attempt_timeout=timeout,
    )
