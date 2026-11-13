"""Gemini 4 Hydrogen and Carbon model factories with Interactions background polling.

Why Carbon uses `client.interactions.create(..., background=True, store=True)`:
  - `gemini-4-carbon` frequently spends 60+ seconds generating internal thinking tokens
    before flushing the first response token.
  - Synchronous `generate_content` HTTP/1.1 sockets get severed at 60.0s by upstream
    gateways (`httpcore2.RemoteProtocolError: Server disconnected without sending a response`).
  - Submitting `gemini-4-carbon` via the Gemini 4 Interactions API with `background=True,
    store=True` returns an interaction ID immediately (~200ms) and polls `interactions.get(id)`
    over short-lived HTTP GETs — completely eliminating 60s socket disconnects and allowing all
    6 Carbon ring nodes (`C1`-`C6`) to reason concurrently in parallel Pregel supersteps.
"""

from __future__ import annotations

import os
import random
import sys
import threading
import time
import warnings
from typing import Any, Literal

warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"google\.genai\..*")

import google.auth
import google.auth.transport.requests
from google import genai
from google.genai.types import HttpOptions, HttpRetryOptions
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import PrivateAttr

from benzene_langgraph.chemistry import ARGON_MODEL_ID, CARBON_MODEL_ID, HYDROGEN_MODEL_ID

HydrogenThinkingLevel = Literal["low", "medium", "high", "xhigh"]
CarbonThinkingLevel = Literal["low", "medium", "high"]

_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

_AUTH_LOCK = threading.Lock()
_ADC_CACHE: tuple[Any, str] | None = None
_ADC_PROBED = False


def extract_text(message: BaseMessage) -> str:
    """Extract plain text from a Gemini 4 AIMessage (handles string or content blocks)."""
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def format_messages_for_interaction(messages: list[BaseMessage]) -> str:
    """Format LangChain System + Human messages into a clean prompt string for Interactions API."""
    sections: list[str] = []
    for msg in messages:
        txt = extract_text(msg)
        if not txt:
            continue
        if isinstance(msg, SystemMessage):
            sections.append(f"<system_instructions>\n{txt}\n</system_instructions>")
        else:
            sections.append(txt)
    return "\n\n".join(sections)


def resolve_vertex_adc(project_override: str | None = None) -> tuple[Any, str] | None:
    """Load and pre-refresh Google Cloud ADC credentials safely across threads."""
    global _ADC_CACHE, _ADC_PROBED
    with _AUTH_LOCK:
        if _ADC_CACHE is not None and project_override is None:
            creds, proj = _ADC_CACHE
            if getattr(creds, "expired", False) and getattr(creds, "refresh_token", None):
                creds.refresh(google.auth.transport.requests.Request())
            return creds, proj

        if _ADC_PROBED and _ADC_CACHE is None and project_override is None:
            return None

        _ADC_PROBED = True
        try:
            creds, loaded_project = google.auth.default(scopes=[_CLOUD_PLATFORM_SCOPE])
            project = (
                project_override
                or os.environ.get("GOOGLE_CLOUD_PROJECT")
                or os.environ.get("GCLOUD_PROJECT")
                or loaded_project
            )
            if not project:
                return None
            if not getattr(creds, "valid", False):
                creds.refresh(google.auth.transport.requests.Request())
            if project_override is None:
                _ADC_CACHE = (creds, project)
            return creds, project
        except Exception:
            return None


def has_gemini_credentials() -> bool:
    """Return True if Gemini API key or Google Cloud Vertex AI ADC is available."""
    if (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
    ):
        return True
    return resolve_vertex_adc() is not None


def _resolve_api_key(explicit_key: str | None = None) -> str | None:
    return (
        explicit_key
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or None
    )


def _is_free_tier_zero_quota(exc: BaseException) -> bool:
    msg = str(exc)
    return "429" in msg and (
        "FreeTier" in msg
        or "free_tier" in msg
        or "limit: 0" in msg
    )


def _is_transient_disconnect_or_overload(exc: BaseException) -> bool:
    cls_name = type(exc).__name__
    mod_name = type(exc).__module__
    msg = str(exc)
    if any(
        token in cls_name
        for token in (
            "RemoteProtocolError",
            "ReadError",
            "WriteError",
            "ConnectError",
            "PoolTimeout",
            "ReadTimeout",
            "ConnectTimeout",
            "TimeoutException",
            "TransportError",
            "ServerError",
            "ServiceUnavailable",
            "APIConnectionError",
            "APITimeoutError",
        )
    ):
        return True
    if any(pkg in mod_name for pkg in ("httpx", "httpcore")) and "Error" in cls_name:
        return True
    if any(
        phrase in msg
        for phrase in (
            "Server disconnected without sending a response",
            "Connection reset by peer",
            "503 UNAVAILABLE",
            "502 Bad Gateway",
            "504 Gateway Timeout",
            "500 INTERNAL",
        )
    ):
        return True
    if "429" in msg and not _is_free_tier_zero_quota(exc):
        return True
    return False


def build_genai_client(
    *,
    api_key: str | None = None,
    vertexai: bool | None = None,
    project: str | None = None,
    location: str = "global",
) -> genai.Client:
    """Construct a `google.genai.Client` for either Gemini Developer API or Vertex AI."""
    resolved_key = _resolve_api_key(api_key)
    use_vertex = bool(vertexai) or (vertexai is None and not resolved_key)
    http_opts = HttpOptions(retry_options=HttpRetryOptions(attempts=2))

    if use_vertex:
        adc = resolve_vertex_adc(project_override=project)
        if adc is not None:
            creds, resolved_project = adc
            return genai.Client(
                vertexai=True,
                project=resolved_project,
                location=location,
                credentials=creds,
                http_options=http_opts,
            )
        return genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=http_opts,
        )

    return genai.Client(api_key=resolved_key, http_options=http_opts)


def build_raw_chat_model(
    *,
    model_id: str,
    atom: str,
    thinking_level: str,
    timeout: float = 300.0,
    api_key: str | None = None,
    vertexai: bool | None = None,
    project: str | None = None,
    location: str = "global",
) -> ChatGoogleGenerativeAI:
    """Construct an underlying ChatGoogleGenerativeAI client (used by Hydrogen & tests)."""
    resolved_key = _resolve_api_key(api_key)
    use_vertex = bool(vertexai) or (vertexai is None and not resolved_key)

    kwargs: dict[str, Any] = {
        "model": model_id,
        # Gemini 4 deprecates sampling params like temperature; explicitly unset default 0.7
        "temperature": None,
        "thinking_config": {"thinking_level": thinking_level},
        "timeout": timeout,
        "max_retries": 2,
        "name": f"{atom}_{model_id}",
    }

    if use_vertex:
        adc = resolve_vertex_adc(project_override=project)
        if adc is not None:
            creds, resolved_project = adc
            kwargs["vertexai"] = True
            kwargs["project"] = resolved_project
            kwargs["location"] = location
            kwargs["credentials"] = creds
        else:
            kwargs["vertexai"] = True
            if project:
                kwargs["project"] = project
            kwargs["location"] = location
    elif resolved_key is not None:
        kwargs["google_api_key"] = resolved_key

    return ChatGoogleGenerativeAI(**kwargs)


def run_carbon_background_interaction(
    *,
    client: genai.Client,
    model_id: str,
    prompt_text: str,
    thinking_level: str,
    poll_interval: float = 2.0,
    timeout: float = 1200.0,
) -> AIMessage:
    """Execute Gemini 4 Carbon (or Argon) via `interactions.create(background=True, store=True)`.

    Short-lived poll GETs completely avoid 60-second GFE HTTP/1.1 header disconnects
    during extended Carbon thinking phases.
    """
    interaction = client.interactions.create(
        model=model_id,
        input=prompt_text,
        background=True,
        store=True,
        generation_config={
            "thinking_level": thinking_level,
        },
    )
    interaction_id = interaction.id
    deadline = time.monotonic() + timeout

    while interaction.status in ("in_progress", "queued", "pending", "running"):
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Interaction {interaction_id} ({model_id}) exceeded {timeout}s timeout"
            )
        time.sleep(poll_interval)
        try:
            interaction = client.interactions.get(interaction_id)
        except Exception as poll_exc:
            if _is_transient_disconnect_or_overload(poll_exc):
                time.sleep(poll_interval)
                continue
            raise

    if interaction.status != "completed":
        raise RuntimeError(
            f"Interaction {interaction_id} ({model_id}) ended with status={interaction.status!r}"
        )

    usage = getattr(interaction, "usage", None)
    usage_metadata = (
        {
            "input_tokens": usage.total_input_tokens or 0,
            "output_tokens": usage.total_output_tokens or 0,
            "total_tokens": usage.total_tokens or 0,
        }
        if usage is not None
        else None
    )

    output_text = getattr(interaction, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return AIMessage(content=output_text.strip(), usage_metadata=usage_metadata)

    # Fallback step extraction if output_text property is absent
    dump = interaction.model_dump() if hasattr(interaction, "model_dump") else {}
    chunks: list[str] = []
    for step in dump.get("steps", []) or []:
        if isinstance(step, dict) and step.get("type") == "model_output":
            for part in step.get("content", []) or []:
                if isinstance(part, dict) and part.get("type") == "text":
                    chunks.append(str(part.get("text", "")))
    return AIMessage(content="".join(chunks).strip(), usage_metadata=usage_metadata)


class ResilientAtomChatModel(BaseChatModel):
    """LangChain BaseChatModel using fast generate_content for Hydrogen and background Interactions for Carbon."""

    atom: str
    element: Literal["H", "C", "Ar"]
    gemini_model: str
    thinking_level: str
    timeout: float = 1200.0
    api_key: str | None = None
    vertexai: bool | None = None
    project: str | None = None
    location: str = "global"
    max_transport_retries: int = 4

    _h_delegate: ChatGoogleGenerativeAI | None = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return f"resilient-{self.gemini_model}"

    def _invoke_hydrogen(self, messages: list[BaseMessage], stop: list[str] | None) -> AIMessage:
        if self._h_delegate is None:
            self._h_delegate = build_raw_chat_model(
                model_id=HYDROGEN_MODEL_ID,
                atom=self.atom,
                thinking_level=self.thinking_level,
                timeout=self.timeout,
                api_key=self.api_key,
                vertexai=self.vertexai,
                project=self.project,
                location=self.location,
            )
        return self._h_delegate.invoke(messages, stop=stop)

    def _invoke_carbon(self, messages: list[BaseMessage], stop: list[str] | None) -> AIMessage:
        client = build_genai_client(
            api_key=self.api_key,
            vertexai=self.vertexai,
            project=self.project,
            location=self.location,
        )
        prompt_text = format_messages_for_interaction(messages)
        return run_carbon_background_interaction(
            client=client,
            model_id=self.gemini_model,
            prompt_text=prompt_text,
            thinking_level=self.thinking_level,
            timeout=self.timeout,
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_exc: BaseException | None = None
        for attempt in range(self.max_transport_retries + 1):
            try:
                if self.element != "H":  # Carbon and Argon both think long; use background polling
                    ai_msg = self._invoke_carbon(messages, stop)
                else:
                    ai_msg = self._invoke_hydrogen(messages, stop)
                return ChatResult(generations=[ChatGeneration(message=ai_msg)])
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_transport_retries and _is_transient_disconnect_or_overload(
                    exc
                ):
                    delay = min(12.0, (1.5 * (2**attempt)) + random.uniform(0.1, 0.8))
                    print(
                        f"  [retry {attempt + 1}/{self.max_transport_retries}] "
                        f"{self.atom} transient network error ({type(exc).__name__}); "
                        f"retrying in {delay:.1f}s...",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(delay)
                    self._h_delegate = None
                    continue
                raise

        assert last_exc is not None
        raise last_exc


class MockAtomChatModel(BaseChatModel):
    """Deterministic BaseChatModel for offline simulation & topology testing."""

    atom: str
    element: Literal["H", "C", "Ar"]
    gemini_model: str
    thinking_level: str

    @property
    def _llm_type(self) -> str:
        return f"mock-{self.gemini_model}"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        joined = "\n".join(extract_text(m) for m in messages)
        markers = [
            tag
            for tag in (
                "H1", "H2", "H3", "H4", "H5", "H6",
                "C1", "C2", "C3", "C4", "C5", "C6",
                "sigma", "pi",
            )
            if tag in joined
        ]
        summary = (
            f"[{self.atom}:{self.gemini_model}@{self.thinking_level}] "
            f"coupled({','.join(markers)})"
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=summary))])


def create_hydrogen_model(
    *,
    atom: str,
    thinking_level: HydrogenThinkingLevel = "medium",
    mock: bool = False,
    api_key: str | None = None,
    vertexai: bool | None = None,
    project: str | None = None,
    location: str = "global",
) -> BaseChatModel:
    """Instantiate a Gemini 4 Hydrogen (`gemini-4-hydrogen`) node model (valency 1)."""
    if mock:
        return MockAtomChatModel(
            atom=atom,
            element="H",
            gemini_model=HYDROGEN_MODEL_ID,
            thinking_level=thinking_level,
        )
    return ResilientAtomChatModel(
        atom=atom,
        element="H",
        gemini_model=HYDROGEN_MODEL_ID,
        thinking_level=thinking_level,
        api_key=api_key,
        vertexai=vertexai,
        project=project,
        location=location,
    )


def create_carbon_model(
    *,
    atom: str,
    thinking_level: CarbonThinkingLevel = "low",
    timeout: float = 900.0,
    mock: bool = False,
    api_key: str | None = None,
    vertexai: bool | None = None,
    project: str | None = None,
    location: str = "global",
) -> BaseChatModel:
    """Instantiate a Gemini 4 Carbon (`gemini-4-carbon`) ring node model (valency 4)."""
    if mock:
        return MockAtomChatModel(
            atom=atom,
            element="C",
            gemini_model=CARBON_MODEL_ID,
            thinking_level=thinking_level,
        )
    return ResilientAtomChatModel(
        atom=atom,
        element="C",
        gemini_model=CARBON_MODEL_ID,
        thinking_level=thinking_level,
        timeout=timeout,
        api_key=api_key,
        vertexai=vertexai,
        project=project,
        location=location,
    )


def create_argon_model(*, atom: str = "Ar", thinking_level: CarbonThinkingLevel = "low", mock: bool = False) -> BaseChatModel:
    """Instantiate a Gemini 4 Argon (`gemini-4-argon`) model: a noble gas, so it joins no ring (valency 0)."""
    cls = MockAtomChatModel if mock else ResilientAtomChatModel
    return cls(atom=atom, element="Ar", gemini_model=ARGON_MODEL_ID, thinking_level=thinking_level)
