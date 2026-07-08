# -*- coding: utf-8 -*-
"""LLM Provider abstraction layer: defines a common provider interface and implements OpenRouter, OpenAI, and Anthropic backends.

Includes HTTP retry logic, rate limit handling (Retry-After), timeout control, structured logging, and token cost tracking.
"""
from __future__ import annotations

import os
import re
import time
import json
from typing import Any, Optional, Protocol, Generator, Union

from .. import config as cfg
from . import env as benv
from clippilot.logger import get_logger

logger = get_logger("clippilot.llm")


class AccountQuotaExhaustedError(Exception):
    """Raised when the OpenRouter account free quota, daily limit, global limit, or credits are exhausted."""
    pass


class JSONParsingError(ValueError):
    """Custom exception raised when JSON parsing fails after all attempts."""
    pass

# Pricing per 1M tokens (input, output)
PRICING = {
    # Anthropic
    "claude-opus-4-8": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (0.25, 1.25),
    "claude-3-opus-20240229": (15.0, 75.0),
    "claude-3-5-sonnet-20240620": (3.0, 15.0),
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.150, 0.600),
    "gpt-4-turbo": (10.00, 30.00),
}

def get_pricing(model: str, default_provider: str = "openai") -> tuple[float, float]:
    """Retrieve pricing (input, output) per 1M tokens for a given model."""
    if "free" in model.lower():
        return (0.0, 0.0)
    for k, v in PRICING.items():
        if k in model:
            return v
    if "claude" in model or "anthropic" in model:
        return (3.00, 15.00)
    return (2.50, 10.00)


def safe_extract_response_content(resp) -> str:
    """Defensively extracts assistant text content from any OpenAI/OpenRouter response object."""
    if resp is None:
        return ""
    
    # If the response is already a string, return it directly
    if isinstance(resp, str):
        return resp
        
    # Check if resp has a 'choices' attribute or key
    choices = getattr(resp, "choices", None)
    if choices is None and isinstance(resp, dict):
        choices = resp.get("choices")
        
    if not choices or not isinstance(choices, (list, tuple)):
        # Maybe it's a direct text field or string representation
        text_field = getattr(resp, "text", None)
        if text_field is not None:
            return str(text_field)
        if isinstance(resp, dict):
            text_val = resp.get("text") or resp.get("content")
            if text_val is not None:
                return str(text_val)
        return ""
        
    # Check the first choice
    try:
        first_choice = choices[0]
    except (IndexError, TypeError):
        return ""
        
    if first_choice is None:
        return ""
        
    # Extract from 'message' or 'text'
    message = getattr(first_choice, "message", None)
    if message is None and isinstance(first_choice, dict):
        message = first_choice.get("message")
        
    content = None
    if message is not None:
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
            
    if content is None:
        # Check text field directly in choice (e.g., legacy completion or text endpoints)
        text_field = getattr(first_choice, "text", None)
        if text_field is not None:
            content = text_field
        elif isinstance(first_choice, dict):
            content = first_choice.get("text")
            
    if content is None:
        return ""
        
    # Handle if content is a list of blocks (multimodal or structured text blocks)
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                val = block.get("text")
                if val:
                    text_parts.append(str(val))
            else:
                val = getattr(block, "text", None)
                if val:
                    text_parts.append(str(val))
        return "".join(text_parts)
        
    return str(content)


def save_failed_response(
    stage_name: str,
    raw_response: str,
    reason: str,
    provider: Optional[str] = None,
    requested_model: Optional[str] = None,
    actual_model: Optional[str] = None,
    correction_attempted: bool = False,
    repair_attempted: bool = False,
    schema_validation_status: str = "not_applicable"
):
    """Saves the raw LLM response to logs/failed_llm_responses/ when JSON repair fails."""
    try:
        import datetime
        from pathlib import Path
        logs_dir = Path("logs/failed_llm_responses")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp_file = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        filename = f"{stage_name}_{timestamp_file}.txt"
        filepath = logs_dir / filename
        
        metadata = [
            f"Stage: {stage_name}",
            f"Requested Model: {requested_model or 'Unknown'}",
            f"Actual Model: {actual_model or 'Unknown'}",
            f"Provider: {provider or 'Unknown'}",
            f"Reason: {reason}",
            f"Repair Attempted: {repair_attempted}",
            f"Correction Attempted: {correction_attempted}",
            f"Schema Validation Status: {schema_validation_status}",
            f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        metadata_str = "\n".join(metadata)
        
        file_content = f"{metadata_str}\n\nRaw Response:\n{raw_response or ''}"
        filepath.write_text(file_content, encoding="utf-8")
        logger.warning(f"💾 Saved failed raw LLM response (Reason: {reason}) for stage '{stage_name}' to {filepath}")
    except Exception as e:
        logger.error(f"⚠️ Failed to save raw LLM response: {e}")


def detects_reasoning_leakage(text: str) -> bool:
    """Returns True if the response content contains reasoning leakage patterns at the start."""
    trimmed = text.strip()
    if not trimmed:
        return False
    # If it starts with JSON indicators, it is not leaked reasoning
    if trimmed.startswith('{') or trimmed.startswith('['):
        return False
        
    lower_trimmed = trimmed.lower()
    
    # Check for XML tags
    tags = ["<analysis>", "<thinking>", "<reasoning>"]
    if any(t in lower_trimmed for t in tags):
        return True
        
    # Check if text starts with any of the reasoning prefixes (case-insensitive)
    prefixes = [
        "analysis:",
        "reasoning:",
        "thought:",
        "thinking:",
        "the user wants",
        "let me",
        "let's",
        "sure!",
        "sure,",
        "okay",
        "certainly",
        "here's",
        "here is",
        "first,",
        "second,",
        "to solve this",
        "i need to",
        "i'll",
        "we need to",
        "the task is",
        "my approach"
    ]
    for p in prefixes:
        if lower_trimmed.startswith(p):
            return True
            
    # Check if there is a long block of text before any '{' or '['
    first_brace = trimmed.find('{')
    first_bracket = trimmed.find('[')
    first_json_idx = min(first_brace if first_brace != -1 else len(trimmed),
                         first_bracket if first_bracket != -1 else len(trimmed))
    if first_json_idx > 50:
        return True
        
    return False


class ProviderUsage(dict):
    """Standardized provider usage dictionary."""
    def __init__(
        self,
        provider: str,
        requested_model: str,
        actual_model: str,
        input_tokens: int,
        output_tokens: int,
        latency: float,
        estimated_cost: float,
        estimated_usage: bool = False
    ):
        super().__init__({
            "provider": provider,
            "requested_model": requested_model,
            "actual_model": actual_model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency": latency,
            "estimated_cost": estimated_cost,
            "cost": estimated_cost,  # alias for backward compatibility
            "estimated_usage": estimated_usage
        })


def estimate_prompt_chars(prompt_or_messages: Any) -> int:
    """Helper to estimate the total character count of prompts or message arrays."""
    if isinstance(prompt_or_messages, str):
        return len(prompt_or_messages)
    if isinstance(prompt_or_messages, list):
        total = 0
        for msg in prompt_or_messages:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    total += len(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            total += len(part.get("text", ""))
            elif msg is not None:
                total += len(str(msg))
        return total
    return 0


def make_provider_usage(
    provider: str,
    requested_model: str,
    actual_model: str,
    input_tokens: int,
    output_tokens: int,
    latency: float,
    prompt_or_messages: Any,
    response_str: str,
    system_prompt: Optional[str] = None
) -> ProviderUsage:
    """Instantiates ProviderUsage, automatically estimating missing token usage if both are 0."""
    in_t = input_tokens
    out_t = output_tokens
    estimated_usage = False
    
    if in_t == 0 and out_t == 0:
        total_prompt_chars = estimate_prompt_chars(prompt_or_messages) + (len(system_prompt) if system_prompt else 0)
        in_t = max(1, total_prompt_chars // 4)
        out_t = max(1, len(response_str) // 4)
        estimated_usage = True
        
    in_price, out_price = get_pricing(actual_model, "openai" if "OpenAI" in provider or "OpenRouter" in provider else "anthropic")
    cost = round((in_t / 1_000_000 * in_price) + (out_t / 1_000_000 * out_price), 6)
    
    return ProviderUsage(
        provider=provider,
        requested_model=requested_model,
        actual_model=actual_model,
        input_tokens=in_t,
        output_tokens=out_t,
        latency=round(latency, 4),
        estimated_cost=cost,
        estimated_usage=estimated_usage
    )


def is_vision_capable_model(model_name: str) -> bool:
    """Returns True if the model name indicates vision capabilities."""
    name = model_name.lower()
    if "openrouter/free" in name:
        return False
    if "vision" in name or "vl" in name or "gemini" in name or "gpt-4o" in name or "claude-3" in name or "opus" in name or "sonnet" in name or "pixtral" in name:
        return True
    if "test" in name or "dummy" in name or "model-" in name:
        return True
    return False


class LLMProvider(Protocol):
    """Common interface for LLM provider implementations."""

    @property
    def last_usage(self) -> dict[str, Any]:
        """Expose token usage, latency, and estimated cost for the last call."""
        ...

    def supports_vision(self) -> bool:
        """Return True if this provider/model supports multimodal vision input."""
        ...

    def supports_json(self) -> bool:
        """Return True if this provider supports structured JSON output schemas."""
        ...

    def supports_streaming(self) -> bool:
        """Return True if this provider supports token streaming."""
        ...

    def supports_functions(self) -> bool:
        """Return True if this provider supports function calling."""
        ...

    def context_size(self) -> int:
        """Return the maximum context window token size."""
        ...

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None, stream: bool = False, json_schema: Optional[dict[str, Any]] = None) -> Union[str, Generator[str, None, None]]:
        """Generate text from a text prompt, optionally enforcing a JSON schema."""
        ...

    def generate_vision(
        self,
        messages: list[dict[str, Any]],
        json_schema: dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Process multimodal inputs and return a parsed structured dict."""
        ...


def execute_with_retries(func, max_attempts=3):
    """Executes a function with retry logic and exponential backoff, observing Retry-After.
    
    Aborts immediately on authentication or API key errors.
    Raises immediately on model-specific errors to allow candidate fallback logic.
    Retries on transient network, rate limit (429), or 5xx errors.
    """
    last_err = None
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_err = e
            
            # Inspect HTTP status/headers if available
            response = getattr(e, "response", None)
            status_code = getattr(response, "status_code", None)
            if response is not None:
                status_code = getattr(response, "status_code", None)
            
            # If status_code is not available, try to guess or use status_code of parent if custom error
            if status_code is None and hasattr(e, "status_code"):
                status_code = getattr(e, "status_code")
                
            err_msg = str(e).lower()

            # 0. Check for Account-level rate limits / credit exhaustion
            is_account_limit = (
                "daily free quota exhausted" in err_msg
                or "free quota" in err_msg
                or "account-level rate limit" in err_msg
                or "account rate limit" in err_msg
                or "global free-tier limit reached" in err_msg
                or "free-tier limit" in err_msg
                or "free tier limit" in err_msg
                or "insufficient credits" in err_msg
                or "credit limit" in err_msg
                or "no credits" in err_msg
                or "out of credits" in err_msg
                or "insufficient balance" in err_msg
                or "quota exceeded" in err_msg
            )
            if is_account_limit:
                logger.error(f"🛑 Account quota exhausted or limits reached: {e}. Aborting LLM chain immediately.")
                raise AccountQuotaExhaustedError(str(e))
            
            # 1. Check for Authentication / Key errors -> stop immediately
            is_auth_error = (
                status_code == 401
                or "api key" in err_msg
                or "authentication" in err_msg
                or "unauthorized" in err_msg
                or "invalid key" in err_msg
                or "incorrect api key" in err_msg
            )
            if is_auth_error:
                # Stop immediately, do not retry and do not continue fallback chain
                raise e
                
            # 2. Check for Model-specific errors -> do not retry, raise immediately to advance model
            is_model_error = (
                status_code == 404
                or (status_code == 400 and any(kw in err_msg for kw in ["model", "not found", "invalid", "unsupported", "unknown"]))
                or any(kw in err_msg for kw in ["model not found", "model invalid", "unknown model", "does not exist", "not a valid model", "unsupported model"])
            )
            if is_model_error:
                raise e
                
            # 3. Check if status code represents a hard failure (not transient)
            if status_code is not None and status_code not in (429, 500, 502, 503, 504):
                raise e

            # 4. Check for rate limits (429) consecutive attempts
            if status_code == 429 or "rate_limit" in err_msg or "rate limit" in err_msg:
                if attempt >= 1:
                    # Second consecutive 429. Immediately raise e to advance model.
                    raise e
                
            # 5. Otherwise, perform backoff and retry
            retry_after = 2 ** attempt
            if response is not None:
                headers = getattr(response, "headers", {})
                if "Retry-After" in headers:
                    try:
                        retry_after = int(headers["Retry-After"])
                    except ValueError:
                        pass
                        
            logger.warning(f"LLM call failed (attempt {attempt+1}/{max_attempts}): {e}. Retrying once on rate limit (Retry-After: {retry_after}s)...")
            time.sleep(retry_after)
    raise last_err


class AnthropicProvider:
    """Anthropic Claude API Provider implementation."""

    def __init__(
        self,
        model: str | list[str],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        if isinstance(model, list):
            self.models = model
        else:
            self.models = [model]
        self.model = self.models[0] if self.models else "claude-opus-4-8"
        self.requested_model = self.models[0] if self.models else "claude-opus-4-8"
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("LLM_API_KEY") or benv.get_api_key()
        self.base_url = base_url or os.environ.get("LLM_BASE_URL")
        self._last_usage: dict[str, Any] = {}

        if not self.api_key:
            raise ValueError(
                "Anthropic API key is missing. Please set the ANTHROPIC_API_KEY or LLM_API_KEY environment variable, "
                "or configure it in your settings.json."
            )

        self._client = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from anthropic import Anthropic
            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = Anthropic(**kwargs)
        return self._client

    @property
    def last_usage(self) -> dict[str, Any]:
        return self._last_usage

    def supports_vision(self) -> bool:
        return True

    def supports_json(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def supports_functions(self) -> bool:
        return True

    def context_size(self) -> int:
        return 200000

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None, stream: bool = False, json_schema: Optional[dict[str, Any]] = None) -> Union[str, Generator[str, None, None]]:
        client = self._ensure_client()
        
        json_enforcement = (
            "\n\nYou MUST return ONLY valid JSON.\n"
            "Do NOT explain.\n"
            "Do NOT reason.\n"
            "Do NOT think step-by-step.\n"
            "Do NOT include markdown.\n"
            "Do NOT wrap JSON in code fences.\n"
            "Do NOT include introductory or concluding text.\n"
            "Your response must begin with '{' or '[' and end with '}' or ']'.\n"
            "Any response outside the JSON schema is considered invalid."
        )
        if json_schema:
            if system_prompt:
                system_prompt = system_prompt + json_enforcement
            else:
                system_prompt = json_enforcement.strip()

        messages = [{"role": "user", "content": prompt}]

        if stream:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": 1024,
                "messages": messages,
                "timeout": 30.0,
            }
            if system_prompt:
                kwargs["system"] = system_prompt
            def _stream_generator():
                start_time = time.time()
                try:
                    with client.messages.stream(**kwargs) as stream_resp:
                        for text in stream_resp.text_stream:
                            yield text
                    latency = time.time() - start_time
                    self._last_usage = ProviderUsage(
                        provider="AnthropicProvider",
                        requested_model=self.requested_model,
                        actual_model=self.model,
                        input_tokens=0,
                        output_tokens=0,
                        latency=round(latency, 4),
                        estimated_cost=0.0
                    )
                except Exception as e:
                    logger.error(f"Anthropic streaming failed: {e}")
                    raise
            return _stream_generator()
        last_err = None
        for model in self.models:
            if not is_model_available(model):
                logger.warning(f"🔌 Model Circuit Breaker: Skipping already rate-limited model '{model}'.")
                pipeline_diagnostics["attempts"].append({"model": model, "status": "Skipped (Circuit Breaker)", "latency": 0.0})
                continue

            self.model = model
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": 1024,
                "messages": messages,
                "timeout": 30.0,
            }
            if system_prompt:
                kwargs["system"] = system_prompt
 
            logger.info(f"Sending Anthropic text request to {model}...")
            start_time = time.time()
            try:
                def _call():
                    return client.messages.create(**kwargs)
 
                resp = execute_with_retries(_call)
                latency = time.time() - start_time
 
                text = "".join(
                    getattr(b, "text", "")
                    for b in resp.content
                    if getattr(b, "type", None) == "text"
                )
 
                # Parse usage
                usage = getattr(resp, "usage", None)
                in_t = getattr(usage, "input_tokens", 0) if usage else 0
                out_t = getattr(usage, "output_tokens", 0) if usage else 0
 
                self._last_usage = make_provider_usage(
                    provider="AnthropicProvider",
                    requested_model=self.requested_model,
                    actual_model=model,
                    input_tokens=in_t,
                    output_tokens=out_t,
                    latency=latency,
                    prompt_or_messages=messages,
                    response_str=text,
                    system_prompt=system_prompt
                )

                # Record successful HTTP call diagnostics
                pipeline_diagnostics["actual_provider"] = "AnthropicProvider"
                pipeline_diagnostics["actual_model"] = model
                pipeline_diagnostics["input_tokens"] += self._last_usage.get("input_tokens", 0)
                pipeline_diagnostics["output_tokens"] += self._last_usage.get("output_tokens", 0)
                pipeline_diagnostics["total_latency"] += latency
 
                if json_schema:
                    try:
                        # Extract and parse first
                        tolerant_json_loads(text, json_schema=json_schema)
                    except Exception as parse_err:
                        is_leakage = detects_reasoning_leakage(text)
                        from clippilot.config import Settings
                        settings = Settings.load()
                        correction_attempted = False
                        if getattr(settings, "enable_reasoning_correction", True):
                            logger.warning(f"⚠️ JSON parsing or schema validation failed. Attempting automatic correction request to same model '{model}'...")
                            correction_attempted = True
                            pipeline_diagnostics["correction_requests"] += 1
                            try:
                                # Append assistant turn and correction prompt
                                correction_messages = messages + [
                                    {"role": "assistant", "content": text},
                                    {"role": "user", "content": "You already generated the correct answer.\n\nReturn ONLY valid JSON.\n\nDo not include markdown.\n\nDo not include explanations.\n\nDo not include reasoning.\n\nReturn ONLY the JSON."}
                                ]
                                kwargs_correction = {
                                    "model": model,
                                    "max_tokens": 1024,
                                    "messages": correction_messages,
                                    "timeout": 30.0,
                                }
                                if system_prompt:
                                    kwargs_correction["system"] = system_prompt
                                    
                                def _call_correction():
                                    return client.messages.create(**kwargs_correction)
                                    
                                resp_corr = execute_with_retries(_call_correction)
                                text_corr = "".join(
                                    getattr(b, "text", "")
                                    for b in resp_corr.content
                                    if getattr(b, "type", None) == "text"
                                )
                                
                                usage_corr = getattr(resp_corr, "usage", None)
                                in_t_corr = getattr(usage_corr, "input_tokens", 0) if usage_corr else 0
                                out_t_corr = getattr(usage_corr, "output_tokens", 0) if usage_corr else 0
                                
                                # Add correction usage and latency
                                accumulated_in = self._last_usage["input_tokens"] + in_t_corr
                                accumulated_out = self._last_usage["output_tokens"] + out_t_corr
                                accumulated_latency = self._last_usage["latency"] + (time.time() - start_time - latency)
                                
                                self._last_usage = make_provider_usage(
                                    provider="AnthropicProvider",
                                    requested_model=self.requested_model,
                                    actual_model=model,
                                    input_tokens=accumulated_in,
                                    output_tokens=accumulated_out,
                                    latency=accumulated_latency,
                                    prompt_or_messages=correction_messages,
                                    response_str=text_corr,
                                    system_prompt=system_prompt
                                )

                                # Record correction tokens in diagnostics
                                pipeline_diagnostics["input_tokens"] += in_t_corr
                                pipeline_diagnostics["output_tokens"] += out_t_corr
                                pipeline_diagnostics["total_latency"] += (time.time() - start_time - latency)
                                
                                # Try parsing the corrected response
                                tolerant_json_loads(text_corr, json_schema=json_schema)
                                text = text_corr  # Correction succeeded!
                            except Exception as corr_err:
                                logger.error(f"Correction request failed: {corr_err}")
                                is_schema_err = "schema validation" in str(corr_err).lower()
                                save_failed_response(
                                    stage_name="script" if "script" in prompt.lower() or "scene" in prompt.lower() else "critic",
                                    raw_response=text,
                                    reason="Schema validation failure" if is_schema_err else ("Reasoning leakage" if is_leakage else "JSON validation failure"),
                                    provider="AnthropicProvider",
                                    requested_model=self.requested_model,
                                    actual_model=model,
                                    correction_attempted=correction_attempted,
                                    repair_attempted=True,
                                    schema_validation_status="failed" if is_schema_err or "schema validation" in str(parse_err).lower() else "not_applicable"
                                )
                                # Halt candidate search by raising JSONParsingError directly
                                raise JSONParsingError(f"JSON validation failed completely on correction: {corr_err}")
                        else:
                            # Save failed response since correction is disabled
                            is_schema_err = "schema validation" in str(parse_err).lower()
                            save_failed_response(
                                stage_name="script" if "script" in prompt.lower() or "scene" in prompt.lower() else "critic",
                                raw_response=text,
                                reason="Schema validation failure" if is_schema_err else ("Reasoning leakage" if is_leakage else "JSON validation failure"),
                                provider="AnthropicProvider",
                                requested_model=self.requested_model,
                                actual_model=model,
                                correction_attempted=correction_attempted,
                                repair_attempted=True,
                                schema_validation_status="failed" if is_schema_err else "not_applicable"
                            )
                            raise JSONParsingError(f"JSON parsing/schema validation failed: {parse_err}")
 
                logger.info(
                    f"✔️ LLM call succeeded:\n"
                    f"Requested model: {self.requested_model}\n"
                    f"Actual model: {model}\n"
                    f"Provider: AnthropicProvider\n"
                    f"Latency: {self._last_usage['latency']:.4f}s\n"
                    f"Input tokens: {self._last_usage['input_tokens']}\n"
                    f"Output tokens: {self._last_usage['output_tokens']}\n"
                    f"Estimated cost: ${self._last_usage['estimated_cost']:.6f}"
                )
                pipeline_diagnostics["attempts"].append({"model": model, "status": "Success", "latency": time.time() - start_time})
                return text
            except AccountQuotaExhaustedError as e:
                logger.error(f"🛑 Account quota/limits exhausted: {e}. Aborting candidate search.")
                pipeline_diagnostics["attempts"].append({"model": model, "status": "QuotaExhausted", "latency": time.time() - start_time})
                raise e
            except JSONParsingError as e:
                logger.error(f"🛑 JSON parsing or schema validation failed completely for model '{model}'. Propagating JSONParsingError immediately (no fallback).")
                pipeline_diagnostics["attempts"].append({"model": model, "status": "Success (Invalid JSON/Schema)", "latency": time.time() - start_time})
                raise e
            except Exception as e:
                # If we raised parse_err, catch it and propagate it or log candidate failure
                last_err = e
                err_msg = str(e).lower()
                status_str = "Error"
                if any(kw in err_msg for kw in ["429", "rate limit", "rate_limit", "provider unavailable", "capacity exceeded", "temporary upstream failure", "unavailable"]):
                    mark_model_unavailable(model)
                    status_str = "429/RateLimit"

                pipeline_diagnostics["attempts"].append({"model": model, "status": status_str, "latency": time.time() - start_time})
                curr_idx = self.models.index(model)
                if curr_idx < len(self.models) - 1:
                    next_model = self.models[curr_idx + 1]
                    logger.warning(f"⚠️ Model '{model}' unavailable due to error: {e}.\nTrying next candidate:\n'{next_model}'")
                else:
                    logger.warning(f"⚠️ Model '{model}' failed (no remaining candidates): {e}.")
                continue
        raise last_err

    def generate_vision(
        self,
        messages: list[dict[str, Any]],
        json_schema: dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        client = self._ensure_client()
        last_err = None
        for model in self.models:
            if not is_vision_capable_model(model):
                logger.warning(f"Skipping text-only model '{model}' for Anthropic Vision QA task.")
                continue
            self.model = model
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": 8000,
                "messages": messages,
                "timeout": 90.0,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            logger.info(f"Sending Anthropic vision request to {model}...")
            start_time = time.time()
            api_succeeded = False
            try:
                def _call():
                    return client.messages.create(**kwargs)

                resp = execute_with_retries(_call)
                api_succeeded = True
            except AccountQuotaExhaustedError as e:
                logger.error(f"🛑 Account quota/limits exhausted: {e}. Aborting candidate search.")
                raise e
            except Exception as e:
                last_err = e
                curr_idx = self.models.index(model)
                if curr_idx < len(self.models) - 1:
                    next_model = self.models[curr_idx + 1]
                    logger.warning(f"⚠️ Model '{model}' unavailable due to error: {e}.\nTrying next candidate:\n'{next_model}'")
                else:
                    logger.warning(f"⚠️ Model '{model}' failed (no remaining candidates): {e}.")
                continue

            # API succeeded!
            latency = time.time() - start_time
            content = ""
            if resp:
                resp_content = getattr(resp, "content", None)
                if isinstance(resp_content, list):
                    parts = []
                    for b in resp_content:
                        if b:
                            b_type = getattr(b, "type", None)
                            if b_type == "text":
                                parts.append(getattr(b, "text", "") or "")
                    content = "".join(parts)

            usage = getattr(resp, "usage", None)
            in_t = getattr(usage, "input_tokens", 0) if usage else 0
            out_t = getattr(usage, "output_tokens", 0) if usage else 0

            self._last_usage = make_provider_usage(
                provider="AnthropicProvider",
                requested_model=self.requested_model,
                actual_model=model,
                input_tokens=in_t,
                output_tokens=out_t,
                latency=latency,
                prompt_or_messages=messages,
                response_str=content,
                system_prompt=system_prompt
            )

            try:
                parsed = tolerant_json_loads(content)
            except Exception as parse_err:
                is_leakage = detects_reasoning_leakage(content)
                from clippilot.config import Settings
                settings = Settings.load()
                correction_attempted = False
                if is_leakage and getattr(settings, "enable_reasoning_correction", True):
                    logger.warning("⚠️ Reasoning leakage detected in vision response. Attempting automatic correction request...")
                    correction_attempted = True
                    try:
                        correction_messages = messages + [
                            {"role": "assistant", "content": content},
                            {"role": "user", "content": "You already generated the correct answer.\n\nReturn ONLY the final JSON object.\n\nDo not include explanations.\n\nDo not include markdown.\n\nDo not include analysis.\n\nReturn valid JSON only."}
                        ]
                        kwargs_correction = {
                            "model": model,
                            "max_tokens": 8000,
                            "messages": correction_messages,
                            "timeout": 90.0,
                        }
                        if system_prompt:
                            kwargs_correction["system"] = system_prompt
                        
                        def _call_correction():
                            return client.messages.create(**kwargs_correction)
                        
                        resp_corr = execute_with_retries(_call_correction)
                        content_corr = ""
                        if resp_corr:
                            resp_content_corr = getattr(resp_corr, "content", None)
                            if isinstance(resp_content_corr, list):
                                parts_corr = []
                                for b in resp_content_corr:
                                    if b:
                                        b_type = getattr(b, "type", None)
                                        if b_type == "text":
                                            parts_corr.append(getattr(b, "text", "") or "")
                                content_corr = "".join(parts_corr)
                                
                        usage_corr = getattr(resp_corr, "usage", None)
                        in_t_corr = getattr(usage_corr, "input_tokens", 0) if usage_corr else 0
                        out_t_corr = getattr(usage_corr, "output_tokens", 0) if usage_corr else 0
                        
                        accumulated_in = self._last_usage["input_tokens"] + in_t_corr
                        accumulated_out = self._last_usage["output_tokens"] + out_t_corr
                        accumulated_latency = self._last_usage["latency"] + (time.time() - start_time - latency)
                        
                        self._last_usage = make_provider_usage(
                            provider="AnthropicProvider",
                            requested_model=self.requested_model,
                            actual_model=model,
                            input_tokens=accumulated_in,
                            output_tokens=accumulated_out,
                            latency=accumulated_latency,
                            prompt_or_messages=correction_messages,
                            response_str=content_corr,
                            system_prompt=system_prompt
                        )
                        
                        parsed = tolerant_json_loads(content_corr)
                    except Exception as corr_err:
                        logger.error(f"Correction request failed in vision: {corr_err}")
                        save_failed_response(
                            stage_name="vision",
                            raw_response=content,
                            reason="Reasoning leakage" if is_leakage else "JSON validation failure",
                            provider="AnthropicProvider",
                            requested_model=self.requested_model,
                            actual_model=model,
                            correction_attempted=correction_attempted,
                            repair_attempted=True
                        )
                        raise parse_err
                else:
                    save_failed_response(
                        stage_name="vision",
                        raw_response=content,
                        reason="Reasoning leakage" if is_leakage else "JSON validation failure",
                        provider="AnthropicProvider",
                        requested_model=self.requested_model,
                        actual_model=model,
                        correction_attempted=correction_attempted,
                        repair_attempted=True
                    )
                    raise parse_err

            logger.info(
                f"✔️ LLM call succeeded:\n"
                f"Requested model: {self.requested_model}\n"
                f"Actual model: {model}\n"
                f"Provider: AnthropicProvider\n"
                f"Latency: {self._last_usage['latency']:.4f}s\n"
                f"Input tokens: {self._last_usage['input_tokens']}\n"
                f"Output tokens: {self._last_usage['output_tokens']}\n"
                f"Estimated cost: ${self._last_usage['estimated_cost']:.6f}"
            )
            return parsed
        raise last_err


class OpenAIProvider:
    """OpenAI API Provider implementation."""

    def __init__(
        self,
        model: str | list[str],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        if isinstance(model, list):
            self.models = model
        else:
            self.models = [model]
        self.model = self.models[0] if self.models else "gpt-4o"
        self.requested_model = self.models[0] if self.models else "gpt-4o"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._last_usage: dict[str, Any] = {}

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable, "
                "or configure it in your settings.json."
            )

        self._client = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI
            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    @property
    def last_usage(self) -> dict[str, Any]:
        return self._last_usage

    def supports_vision(self) -> bool:
        return True

    def supports_json(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def supports_functions(self) -> bool:
        return True

    def context_size(self) -> int:
        return 128000

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None, stream: bool = False, json_schema: Optional[dict[str, Any]] = None) -> Union[str, Generator[str, None, None]]:
        client = self._ensure_client()
        
        json_enforcement = (
            "\n\nYou MUST return ONLY valid JSON.\n"
            "Do NOT explain.\n"
            "Do NOT reason.\n"
            "Do NOT think step-by-step.\n"
            "Do NOT include markdown.\n"
            "Do NOT wrap JSON in code fences.\n"
            "Do NOT include introductory or concluding text.\n"
            "Your response must begin with '{' or '[' and end with '}' or ']'.\n"
            "Any response outside the JSON schema is considered invalid."
        )
        if json_schema:
            if system_prompt:
                system_prompt = system_prompt + json_enforcement
            else:
                system_prompt = json_enforcement.strip()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if stream:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 1024,
                "timeout": 30.0,
            }
            def _stream_generator():
                start_time = time.time()
                try:
                    resp = client.chat.completions.create(stream=True, **kwargs)
                    for chunk in resp:
                        if chunk and hasattr(chunk, "choices") and chunk.choices:
                            try:
                                first_choice = chunk.choices[0]
                                delta = getattr(first_choice, "delta", None)
                                if delta:
                                    text = getattr(delta, "content", "") or ""
                                    if text:
                                        yield text
                            except (IndexError, AttributeError, TypeError):
                                continue
                    latency = time.time() - start_time
                    self._last_usage = ProviderUsage(
                        provider=self.__class__.__name__,
                        requested_model=self.requested_model,
                        actual_model=self.model,
                        input_tokens=0,
                        output_tokens=0,
                        latency=round(latency, 4),
                        estimated_cost=0.0
                    )
                except Exception as e:
                    logger.error(f"OpenAI streaming failed: {e}")
                    last_err = None
        for model in self.models:
            if not is_model_available(model):
                logger.warning(f"🔌 Model Circuit Breaker: Skipping already rate-limited model '{model}'.")
                pipeline_diagnostics["attempts"].append({"model": model, "status": "Skipped (Circuit Breaker)", "latency": 0.0})
                continue

            self.model = model
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": 1024,
                "timeout": 30.0,
            }
            
            # Setup response_format if json_schema is provided
            if json_schema:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "schema",
                        "strict": True,
                        "schema": json_schema
                    }
                }
 
            logger.info(f"Sending OpenAI text request to {model}...")
            start_time = time.time()
            api_succeeded = False
            try:
                def _call():
                    return client.chat.completions.create(**kwargs)
 
                resp = execute_with_retries(_call)
                api_succeeded = True
            except AccountQuotaExhaustedError as e:
                logger.error(f"🛑 Account quota/limits exhausted: {e}. Aborting candidate search.")
                pipeline_diagnostics["attempts"].append({"model": model, "status": "QuotaExhausted", "latency": time.time() - start_time})
                raise e
            except Exception as e:
                err_msg = str(e).lower()
                # If native structured output failed, fallback to prompt-only JSON enforcement
                if json_schema and ("response_format" in err_msg or "schema" in err_msg or "structure" in err_msg or "not supported" in err_msg or "400" in err_msg):
                    logger.warning(f"⚠️ Model '{model}' does not support native structured output. Retrying with prompt-only JSON enforcement...")
                    kwargs_no_schema = {k: v for k, v in kwargs.items() if k != "response_format"}
                    try:
                        def _call_no_schema():
                            return client.chat.completions.create(**kwargs_no_schema)
                        resp = execute_with_retries(_call_no_schema)
                        api_succeeded = True
                    except Exception as retry_err:
                        last_err = retry_err
                else:
                    last_err = e
 
                if not api_succeeded:
                    status_str = "Error"
                    if any(kw in err_msg for kw in ["429", "rate limit", "rate_limit", "provider unavailable", "capacity exceeded", "temporary upstream failure", "unavailable"]):
                        mark_model_unavailable(model)
                        status_str = "429/RateLimit"

                    pipeline_diagnostics["attempts"].append({"model": model, "status": status_str, "latency": time.time() - start_time})
                    curr_idx = self.models.index(model)
                    if curr_idx < len(self.models) - 1:
                        next_model = self.models[curr_idx + 1]
                        logger.warning(f"⚠️ Model '{model}' unavailable due to error: {last_err}.\nTrying next candidate:\n'{next_model}'")
                    else:
                        logger.warning(f"⚠️ Model '{model}' failed (no remaining candidates): {last_err}.")
                    continue
 
            # API succeeded!
            latency = time.time() - start_time
            text = safe_extract_response_content(resp)
 
            usage = getattr(resp, "usage", None)
            in_t = getattr(usage, "prompt_tokens", 0) if usage else 0
            out_t = getattr(usage, "completion_tokens", 0) if usage else 0
 
            self._last_usage = make_provider_usage(
                provider=self.__class__.__name__,
                requested_model=self.requested_model,
                actual_model=model,
                input_tokens=in_t,
                output_tokens=out_t,
                latency=latency,
                prompt_or_messages=messages,
                response_str=text,
                system_prompt=system_prompt
            )

            # Record successful HTTP call diagnostics
            pipeline_diagnostics["actual_provider"] = self.__class__.__name__
            pipeline_diagnostics["actual_model"] = model
            pipeline_diagnostics["input_tokens"] += self._last_usage.get("input_tokens", 0)
            pipeline_diagnostics["output_tokens"] += self._last_usage.get("output_tokens", 0)
            pipeline_diagnostics["total_latency"] += latency
 
            # Check if JSON is valid and optionally correct reasoning leakage
            if json_schema:
                try:
                    # Extract and parse first
                    tolerant_json_loads(text, json_schema=json_schema)
                except Exception as parse_err:
                    is_leakage = detects_reasoning_leakage(text)
                    from clippilot.config import Settings
                    settings = Settings.load()
                    correction_attempted = False
                    if getattr(settings, "enable_reasoning_correction", True):
                        logger.warning(f"⚠️ JSON parsing or schema validation failed. Attempting automatic correction request to same model '{model}'...")
                        correction_attempted = True
                        pipeline_diagnostics["correction_requests"] += 1
                        try:
                            correction_messages = messages + [
                                {"role": "assistant", "content": text},
                                {"role": "user", "content": "You already generated the correct answer.\n\nReturn ONLY valid JSON.\n\nDo not include markdown.\n\nDo not include explanations.\n\nDo not include reasoning.\n\nReturn ONLY the JSON."}
                            ]
                            kwargs_correction = {
                                "model": model,
                                "messages": correction_messages,
                                "max_tokens": 1024,
                                "timeout": 30.0,
                            }
                            if "response_format" in kwargs:
                                kwargs_correction["response_format"] = kwargs["response_format"]
                                
                            def _call_correction():
                                return client.chat.completions.create(**kwargs_correction)
                                
                            resp_corr = execute_with_retries(_call_correction)
                            text_corr = safe_extract_response_content(resp_corr)
                            
                            usage_corr = getattr(resp_corr, "usage", None)
                            in_t_corr = getattr(usage_corr, "prompt_tokens", 0) if usage_corr else 0
                            out_t_corr = getattr(usage_corr, "completion_tokens", 0) if usage_corr else 0
                            
                            accumulated_in = self._last_usage["input_tokens"] + in_t_corr
                            accumulated_out = self._last_usage["output_tokens"] + out_t_corr
                            accumulated_latency = self._last_usage["latency"] + (time.time() - start_time - latency)
                            
                            self._last_usage = make_provider_usage(
                                provider=self.__class__.__name__,
                                requested_model=self.requested_model,
                                actual_model=model,
                                input_tokens=accumulated_in,
                                output_tokens=accumulated_out,
                                latency=accumulated_latency,
                                prompt_or_messages=correction_messages,
                                response_str=text_corr,
                                system_prompt=system_prompt
                            )

                            # Record correction tokens in diagnostics
                            pipeline_diagnostics["input_tokens"] += in_t_corr
                            pipeline_diagnostics["output_tokens"] += out_t_corr
                            pipeline_diagnostics["total_latency"] += (time.time() - start_time - latency)
                            
                            tolerant_json_loads(text_corr, json_schema=json_schema)
                            text = text_corr  # Correction succeeded!
                        except Exception as corr_err:
                            logger.error(f"Correction request failed: {corr_err}")
                            is_schema_err = "schema validation" in str(corr_err).lower()
                            save_failed_response(
                                stage_name="script" if "script" in prompt.lower() or "scene" in prompt.lower() else "critic",
                                raw_response=text,
                                reason="Schema validation failure" if is_schema_err else ("Reasoning leakage" if is_leakage else "JSON validation failure"),
                                provider=self.__class__.__name__,
                                requested_model=self.requested_model,
                                actual_model=model,
                                correction_attempted=correction_attempted,
                                repair_attempted=True,
                                schema_validation_status="failed" if is_schema_err or "schema validation" in str(parse_err).lower() else "not_applicable"
                            )
                            # Halt candidate search by raising JSONParsingError directly
                            raise JSONParsingError(f"JSON validation failed completely on correction: {corr_err}")
                    else:
                        is_schema_err = "schema validation" in str(parse_err).lower()
                        save_failed_response(
                            stage_name="script" if "script" in prompt.lower() or "scene" in prompt.lower() else "critic",
                            raw_response=text,
                            reason="Schema validation failure" if is_schema_err else ("Reasoning leakage" if is_leakage else "JSON validation failure"),
                            provider=self.__class__.__name__,
                            requested_model=self.requested_model,
                            actual_model=model,
                            correction_attempted=correction_attempted,
                            repair_attempted=True,
                            schema_validation_status="failed" if is_schema_err else "not_applicable"
                        )
                        raise JSONParsingError(f"JSON parsing/schema validation failed: {parse_err}")
 
            logger.info(
                f"✔️ LLM call succeeded:\n"
                f"Requested model: {self.requested_model}\n"
                f"Actual model: {model}\n"
                f"Provider: {self.__class__.__name__}\n"
                f"Latency: {self._last_usage['latency']:.4f}s\n"
                f"Input tokens: {self._last_usage['input_tokens']}\n"
                f"Output tokens: {self._last_usage['output_tokens']}\n"
                f"Estimated cost: ${self._last_usage['estimated_cost']:.6f}"
            )
            pipeline_diagnostics["attempts"].append({"model": model, "status": "Success", "latency": time.time() - start_time})
            return text
        raise last_err

    def generate_vision(
        self,
        messages: list[dict[str, Any]],
        json_schema: dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        client = self._ensure_client()

        json_enforcement = (
            "\n\nYou MUST return ONLY valid JSON.\n"
            "Do NOT explain.\n"
            "Do NOT reason.\n"
            "Do NOT think step-by-step.\n"
            "Do NOT include markdown.\n"
            "Do NOT wrap JSON in code fences.\n"
            "Do NOT include introductory or concluding text.\n"
            "Your response must begin with '{' or '[' and end with '}' or ']'.\n"
            "Any response outside the JSON schema is considered invalid."
        )
        if system_prompt:
            system_prompt = system_prompt + json_enforcement
        else:
            system_prompt = json_enforcement.strip()

        # Convert Anthropic message format to OpenAI format
        openai_messages = []
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})
            
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if isinstance(content, list):
                new_content = []
                for item in content:
                    if item.get("type") == "text":
                        new_content.append({"type": "text", "text": item.get("text")})
                    elif item.get("type") == "image":
                        media_type = item.get("source", {}).get("media_type", "image/jpeg")
                        b64_data = item.get("source", {}).get("data")
                        new_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{b64_data}"
                            }
                        })
            else:
                openai_messages.append({"role": role, "content": content})

        last_err = None
        for model in self.models:
            if not is_vision_capable_model(model):
                logger.warning(f"Skipping text-only model '{model}' for Vision QA task.")
                continue
            if not is_model_available(model):
                logger.warning(f"🔌 Model Circuit Breaker: Skipping already rate-limited model '{model}'.")
                pipeline_diagnostics["attempts"].append({"model": model, "status": "Skipped (Circuit Breaker)", "latency": 0.0})
                continue

            self.model = model
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": openai_messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "enrichment_schema",
                        "strict": True,
                        "schema": json_schema
                    }
                },
                "max_tokens": 2048,
                "timeout": 90.0,
            }
 
            logger.info(f"Sending OpenAI vision/schema request to {model}...")
            start_time = time.time()
            api_succeeded = False
            try:
                def _call():
                    return client.chat.completions.create(**kwargs)
 
                resp = execute_with_retries(_call)
                api_succeeded = True
            except AccountQuotaExhaustedError as e:
                logger.error(f"🛑 Account quota/limits exhausted: {e}. Aborting candidate search.")
                pipeline_diagnostics["attempts"].append({"model": model, "status": "QuotaExhausted", "latency": time.time() - start_time})
                raise e
            except Exception as e:
                err_msg = str(e).lower()
                if "response_format" in err_msg or "schema" in err_msg or "structure" in err_msg or "not supported" in err_msg or "400" in err_msg:
                    logger.warning(f"⚠️ Model '{model}' does not support native structured vision. Retrying with prompt-only JSON enforcement...")
                    kwargs_no_schema = {k: v for k, v in kwargs.items() if k != "response_format"}
                    try:
                        def _call_no_schema():
                            return client.chat.completions.create(**kwargs_no_schema)
                        resp = execute_with_retries(_call_no_schema)
                        api_succeeded = True
                    except Exception as retry_err:
                        last_err = retry_err
                else:
                    last_err = e
 
                if not api_succeeded:
                    status_str = "Error"
                    if any(kw in err_msg for kw in ["429", "rate limit", "rate_limit", "provider unavailable", "capacity exceeded", "temporary upstream failure", "unavailable"]):
                        mark_model_unavailable(model)
                        status_str = "429/RateLimit"

                    pipeline_diagnostics["attempts"].append({"model": model, "status": status_str, "latency": time.time() - start_time})
                    curr_idx = self.models.index(model)
                    if curr_idx < len(self.models) - 1:
                        next_model = self.models[curr_idx + 1]
                        logger.warning(f"⚠️ Model '{model}' unavailable due to error: {last_err}.\nTrying next candidate:\n'{next_model}'")
                    else:
                        logger.warning(f"⚠️ Model '{model}' failed (no remaining candidates): {last_err}.")
                    continue
 
            # API succeeded!
            latency = time.time() - start_time
            content = safe_extract_response_content(resp) or "{}"
 
            usage = getattr(resp, "usage", None)
            in_t = getattr(usage, "prompt_tokens", 0) if usage else 0
            out_t = getattr(usage, "completion_tokens", 0) if usage else 0
 
            self._last_usage = make_provider_usage(
                provider=self.__class__.__name__,
                requested_model=self.requested_model,
                actual_model=model,
                input_tokens=in_t,
                output_tokens=out_t,
                latency=latency,
                prompt_or_messages=openai_messages,
                response_str=content,
                system_prompt=system_prompt
            )

            # Record successful HTTP call diagnostics
            pipeline_diagnostics["actual_provider"] = self.__class__.__name__
            pipeline_diagnostics["actual_model"] = model
            pipeline_diagnostics["input_tokens"] += self._last_usage.get("input_tokens", 0)
            pipeline_diagnostics["output_tokens"] += self._last_usage.get("output_tokens", 0)
            pipeline_diagnostics["total_latency"] += latency
 
            try:
                # Extract and parse first
                parsed = tolerant_json_loads(content, json_schema=json_schema)
            except Exception as parse_err:
                is_leakage = detects_reasoning_leakage(content)
                from clippilot.config import Settings
                settings = Settings.load()
                correction_attempted = False
                if is_leakage and getattr(settings, "enable_reasoning_correction", True):
                    logger.warning("⚠️ Reasoning leakage detected in vision response. Attempting automatic correction request...")
                    correction_attempted = True
                    pipeline_diagnostics["correction_requests"] += 1
                    try:
                        correction_messages = openai_messages + [
                            {"role": "assistant", "content": content},
                            {"role": "user", "content": "You already generated the correct answer.\n\nReturn ONLY the final JSON object.\n\nDo not include explanations.\n\nDo not include markdown.\n\nDo not include analysis.\n\nReturn valid JSON only."}
                        ]
                        kwargs_correction = {
                            "model": model,
                            "messages": correction_messages,
                            "max_tokens": 2048,
                            "timeout": 90.0,
                        }
                        if "response_format" in kwargs:
                            kwargs_correction["response_format"] = kwargs["response_format"]
                        
                        def _call_correction():
                            return client.chat.completions.create(**kwargs_correction)
                        
                        resp_corr = execute_with_retries(_call_correction)
                        content_corr = safe_extract_response_content(resp_corr) or "{}"
                        
                        usage_corr = getattr(resp_corr, "usage", None)
                        in_t_corr = getattr(usage_corr, "prompt_tokens", 0) if usage_corr else 0
                        out_t_corr = getattr(usage_corr, "completion_tokens", 0) if usage_corr else 0
                        
                        accumulated_in = self._last_usage["input_tokens"] + in_t_corr
                        accumulated_out = self._last_usage["output_tokens"] + out_t_corr
                        accumulated_latency = self._last_usage["latency"] + (time.time() - start_time - latency)
                        
                        self._last_usage = make_provider_usage(
                            provider=self.__class__.__name__,
                            requested_model=self.requested_model,
                            actual_model=model,
                            input_tokens=accumulated_in,
                            output_tokens=accumulated_out,
                            latency=accumulated_latency,
                            prompt_or_messages=correction_messages,
                            response_str=content_corr,
                            system_prompt=system_prompt
                        )

                        # Record correction tokens in diagnostics
                        pipeline_diagnostics["input_tokens"] += in_t_corr
                        pipeline_diagnostics["output_tokens"] += out_t_corr
                        pipeline_diagnostics["total_latency"] += (time.time() - start_time - latency)
                        
                        parsed = tolerant_json_loads(content_corr, json_schema=json_schema)
                        content = content_corr  # Correction succeeded!
                    except Exception as corr_err:
                        logger.error(f"Correction request failed in vision: {corr_err}")
                        is_schema_err = "schema validation" in str(corr_err).lower()
                        save_failed_response(
                            stage_name="vision",
                            raw_response=content,
                            reason="Schema validation failure" if is_schema_err else ("Reasoning leakage" if is_leakage else "JSON validation failure"),
                            provider=self.__class__.__name__,
                            requested_model=self.requested_model,
                            actual_model=model,
                            correction_attempted=correction_attempted,
                            repair_attempted=True,
                            schema_validation_status="failed" if is_schema_err or "schema validation" in str(parse_err).lower() else "not_applicable"
                        )
                        raise corr_err if isinstance(corr_err, JSONParsingError) else parse_err
                else:
                    is_schema_err = "schema validation" in str(parse_err).lower()
                    save_failed_response(
                        stage_name="vision",
                        raw_response=content,
                        reason="Schema validation failure" if is_schema_err else ("Reasoning leakage" if is_leakage else "JSON validation failure"),
                        provider=self.__class__.__name__,
                        requested_model=self.requested_model,
                        actual_model=model,
                        correction_attempted=correction_attempted,
                        repair_attempted=True,
                        schema_validation_status="failed" if is_schema_err else "not_applicable"
                    )
                    raise parse_err
 
            logger.info(
                f"✔️ LLM call succeeded:\n"
                f"Requested model: {self.requested_model}\n"
                f"Actual model: {model}\n"
                f"Provider: {self.__class__.__name__}\n"
                f"Latency: {self._last_usage['latency']:.4f}s\n"
                f"Input tokens: {self._last_usage['input_tokens']}\n"
                f"Output tokens: {self._last_usage['output_tokens']}\n"
                f"Estimated cost: ${self._last_usage['estimated_cost']:.6f}"
            )
            pipeline_diagnostics["attempts"].append({"model": model, "status": "Success", "latency": time.time() - start_time})
            return parsed
        raise last_err


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter API Provider implementation. Extends OpenAIProvider with OpenRouter specific settings."""

    def __init__(
        self,
        model: str | list[str],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        base_url = base_url or os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        super().__init__(model=model, api_key=api_key, base_url=base_url)

    def _ensure_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI
            headers = {
                "HTTP-Referer": "https://github.com/shashanthnetha/ClipZone",
                "X-Title": "ClipPilot App",
            }
            # Custom default parameters inside OpenAI client
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers=headers
            )
        return self._client


# ── Tolerant JSON Utilities & Diagnostics & Circuit Breaker ──

pipeline_diagnostics: dict[str, Any] = {
    "attempts": [],                # list of dicts: {"model": str, "status": str, "latency": float}
    "correction_requests": 0,
    "json_repairs": 0,
    "schema_failures": 0,
    "total_latency": 0.0,
    "input_tokens": 0,
    "output_tokens": 0,
    "actual_provider": None,
    "actual_model": None
}

_CIRCUIT_BREAKER_COOLDOWN = 600  # 10 minutes
_model_blocklist: dict[str, float] = {}

def mark_model_unavailable(model_name: str) -> None:
    """Marks a model as unavailable for 10 minutes."""
    _model_blocklist[model_name] = time.time() + _CIRCUIT_BREAKER_COOLDOWN
    logger.warning(f"🔌 Model Circuit Breaker: Model '{model_name}' marked unavailable for 10 minutes.")

def is_model_available(model_name: str) -> bool:
    """Returns True if the model is currently available to use."""
    expiry = _model_blocklist.get(model_name, 0.0)
    if expiry > time.time():
        return False
    return True

def validate_schema(data: Any, schema: dict[str, Any], path: str = "") -> list[str]:
    """Recursively validates data against JSON Schema, supporting relaxed schema validation (ignoring extra properties)."""
    errors = []
    if not isinstance(schema, dict):
        return errors
        
    s_type = schema.get("type")
    if s_type == "object":
        if not isinstance(data, dict):
            errors.append(f"{path if path else 'root'} must be an object/dict")
            return errors
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        # Check required fields
        for r in required:
            if r not in data:
                errors.append(f"Missing required key: {path + '.' + r if path else r}")
                
        # Validate properties
        for k, v in data.items():
            if k in properties:
                errors.extend(validate_schema(v, properties[k], path + "." + k if path else k))
            
    elif s_type == "array":
        if not isinstance(data, list):
            errors.append(f"{path if path else 'root'} must be a list/array")
            return errors
        items_schema = schema.get("items")
        if items_schema:
            for idx, item in enumerate(data):
                errors.extend(validate_schema(item, items_schema, f"{path}[{idx}]"))
                
    elif s_type == "string":
        if not isinstance(data, str):
            errors.append(f"{path if path else 'root'} must be a string")
    elif s_type == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            errors.append(f"{path if path else 'root'} must be an integer")
    elif s_type == "number":
        if not isinstance(data, (int, float)) or isinstance(data, bool):
            errors.append(f"{path if path else 'root'} must be a number")
    elif s_type == "boolean":
        if not isinstance(data, bool):
            errors.append(f"{path if path else 'root'} must be a boolean")
            
    return errors

def find_all_balanced_blocks(text: str) -> list[str]:
    """Finds all balanced {...} or [...] substrings in the text using balanced brace parsing.
    Scans character-by-character starting at every index containing { or [ to support nested or overlapping structures.
    """
    blocks = []
    n = len(text)
    for i in range(n):
        char = text[i]
        if char in ('{', '['):
            start_char = char
            end_char = '}' if start_char == '{' else ']'
            bracket_count = 0
            in_string = False
            escape = False
            
            for j in range(i, n):
                c = text[j]
                if escape:
                    escape = False
                    continue
                if c == '\\':
                    escape = True
                    continue
                if c == '"':
                    in_string = not in_string
                    continue
                if not in_string:
                    if c == start_char:
                        bracket_count += 1
                    elif c == end_char:
                        bracket_count -= 1
                        if bracket_count == 0:
                            blocks.append(text[i:j+1])
                            break
    return blocks


def repair_json_string(s: str) -> str:
    """Repairs common JSON issues like trailing commas, ignoring characters inside quoted strings."""
    res = []
    in_string = False
    escape = False
    for i, char in enumerate(s):
        if escape:
            escape = False
            res.append(char)
            continue
        if char == '\\':
            escape = True
            res.append(char)
            continue
        if char == '"':
            in_string = not in_string
            res.append(char)
            continue
            
        if char == ',' and not in_string:
            # Look ahead to see if the next non-whitespace char is } or ]
            j = i + 1
            while j < len(s) and s[j].isspace():
                j += 1
            if j < len(s) and s[j] in ('}', ']'):
                continue  # Skip trailing comma
        res.append(char)
    return "".join(res)


def tolerant_json_loads(text: str, json_schema: Optional[dict[str, Any]] = None) -> Any:
    """Parses JSON from text, extracting balanced JSON objects/arrays, attempting raw parse, then repaired parse.
    
    If multiple JSON candidates exist, attempts to parse them sequentially and returns the first one that succeeds.
    """
    text_strip = text.strip()
    
    # Step 1: Remove markdown fences and strip surrounding prose
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text_strip, re.DOTALL)
    if match:
        text_strip = match.group(1).strip()
        
    # Step 2: Extract all balanced candidates
    blocks = find_all_balanced_blocks(text_strip)
    if not blocks:
        try:
            parsed = json.loads(text_strip)
            if json_schema:
                errors = validate_schema(parsed, json_schema)
                if errors:
                    pipeline_diagnostics["schema_failures"] += 1
                    raise JSONParsingError(f"Schema validation failure: {errors}")
            return parsed
        except Exception as e:
            if isinstance(e, JSONParsingError):
                raise e
            raise JSONParsingError(f"No balanced JSON block found. Standard parse error: {e}")
            
    last_err = None
    # Try parsing each candidate block
    for block in blocks:
        # Step 3: Try raw json.loads() first
        try:
            parsed = json.loads(block)
            if json_schema:
                errors = validate_schema(parsed, json_schema)
                if errors:
                    pipeline_diagnostics["schema_failures"] += 1
                    raise JSONParsingError(f"Schema validation failure: {errors}")
            
            rep_applied = []
            if "```" in text:
                rep_applied.append("removed markdown fences")
            if block != text_strip:
                rep_applied.append("extracted balanced JSON object" if block.startswith('{') else "extracted balanced JSON array")
                
            if rep_applied:
                logger.debug("Recovered JSON successfully.\n\nApplied:\n- " + "\n- ".join(rep_applied))
                
            return parsed
        except Exception as e:
            last_err = e
            if isinstance(e, JSONParsingError):
                raise e
            
        # Step 4: Run repair_json_string and retry
        repaired = repair_json_string(block)
        try:
            parsed = json.loads(repaired)
            if json_schema:
                errors = validate_schema(parsed, json_schema)
                if errors:
                    pipeline_diagnostics["schema_failures"] += 1
                    raise JSONParsingError(f"Schema validation failure: {errors}")
            
            pipeline_diagnostics["json_repairs"] += 1
            
            rep_applied = []
            if "```" in text:
                rep_applied.append("removed markdown fences")
            if block != text_strip:
                rep_applied.append("extracted balanced JSON object" if block.startswith('{') else "extracted balanced JSON array")
            rep_applied.append("removed trailing commas")
            
            logger.debug("Recovered repaired JSON successfully.\n\nApplied:\n- " + "\n- ".join(rep_applied))
            return parsed
        except Exception as e:
            last_err = e
            if isinstance(e, JSONParsingError):
                raise e
            
    # Step 5: Raise JSONParsingError if all candidates fail
    if last_err:
        raise JSONParsingError(f"Failed to parse any JSON candidate. Last error: {last_err}")
    raise JSONParsingError("No valid JSON block could be parsed.")


# ── Provider Resolver ──

def get_provider(settings: Optional[cfg.Settings] = None, models: Optional[list[str]] = None) -> LLMProvider:
    """Resolve and return the configured LLMProvider instance, prioritizing OpenRouter."""
    s = settings or cfg.Settings.load()

    provider_name = (
        os.environ.get("LLM_PROVIDER")
        or os.environ.get("CLIPPILOT_LLM_PROVIDER")
        or getattr(s, "llm_provider", "openrouter")
    ).lower()

    if models:
        model = models
    else:
        # Prioritize model configuration: script_model -> llm_model -> brain_model
        model = (
            os.environ.get("LLM_MODEL")
            or os.environ.get("CLIPPILOT_LLM_MODEL")
            or getattr(s, "llm_model", None)
            or getattr(s, "brain_model", None)
        )

    base_url = os.environ.get("LLM_BASE_URL") or getattr(s, "llm_base_url", None)
    api_key = os.environ.get("LLM_API_KEY") or getattr(s, "llm_api_key", None)

    # Clean default model names if model is not set
    if not model:
        if provider_name == "anthropic":
            model = "claude-opus-4-8"
        elif provider_name == "openai":
            model = "gpt-4o"
        elif provider_name == "openrouter":
            model = "openrouter/free"
        else:
            model = "claude-opus-4-8"

    if provider_name == "anthropic":
        return AnthropicProvider(model=model, api_key=api_key, base_url=base_url)
    elif provider_name == "openai":
        return OpenAIProvider(model=model, api_key=api_key, base_url=base_url)
    elif provider_name == "openrouter":
        return OpenRouterProvider(model=model, api_key=api_key, base_url=base_url)

    raise ValueError(f"Unsupported or unrecognized LLM provider type: {provider_name}")
