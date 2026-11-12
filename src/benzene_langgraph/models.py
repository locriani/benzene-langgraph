"""Gemini 4 Hydrogen and Carbon model factories (plus deterministic offline mock)."""

from __future__ import annotations

import os
import warnings
from typing import Any, Literal

warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"google\.genai\..*")

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI

from benzene_langgraph.chemistry import CARBON_MODEL_ID, HYDROGEN_MODEL_ID

HydrogenThinkingLevel = Literal["low", "medium", "high", "xhigh"]
CarbonThinkingLevel = Literal["low", "medium", "high"]


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


class MockAtomChatModel(BaseChatModel):
    """Deterministic BaseChatModel for offline simulation & topology testing."""

    atom: str
    element: Literal["H", "C"]
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
        # Extract covalent inputs mentioned in the prompt so orbital coupling is observable
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


def has_gemini_credentials() -> bool:
    """Return True if Gemini API or Vertex AI credentials are configured in the environment."""
    return bool(
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
    )


def create_hydrogen_model(
    *,
    atom: str,
    thinking_level: HydrogenThinkingLevel = "high",
    mock: bool = False,
    api_key: str | None = None,
    vertexai: bool | None = None,
) -> BaseChatModel:
    """Instantiate a Gemini 4 Hydrogen (`gemini-4-hydrogen`) node model (valency 1)."""
    if mock:
        return MockAtomChatModel(
            atom=atom,
            element="H",
            gemini_model=HYDROGEN_MODEL_ID,
            thinking_level=thinking_level,
        )
    kwargs: dict[str, Any] = {
        "model": HYDROGEN_MODEL_ID,
        # Gemini 4 deprecates sampling params like temperature; explicitly unset default 0.7
        "temperature": None,
        "thinking_config": {"thinking_level": thinking_level},
        "name": f"{atom}_{HYDROGEN_MODEL_ID}",
    }
    if api_key is not None:
        kwargs["google_api_key"] = api_key
    if vertexai is not None:
        kwargs["vertexai"] = vertexai
    return ChatGoogleGenerativeAI(**kwargs)


def create_carbon_model(
    *,
    atom: str,
    thinking_level: CarbonThinkingLevel = "medium",
    timeout: float = 900.0,
    mock: bool = False,
    api_key: str | None = None,
    vertexai: bool | None = None,
) -> BaseChatModel:
    """Instantiate a Gemini 4 Carbon (`gemini-4-carbon`) ring node model (valency 4)."""
    if mock:
        return MockAtomChatModel(
            atom=atom,
            element="C",
            gemini_model=CARBON_MODEL_ID,
            thinking_level=thinking_level,
        )
    kwargs: dict[str, Any] = {
        "model": CARBON_MODEL_ID,
        # Gemini 4 deprecates sampling params like temperature; explicitly unset default 0.7
        "temperature": None,
        "thinking_config": {"thinking_level": thinking_level},
        "timeout": timeout,
        "name": f"{atom}_{CARBON_MODEL_ID}",
    }
    if api_key is not None:
        kwargs["google_api_key"] = api_key
    if vertexai is not None:
        kwargs["vertexai"] = vertexai
    return ChatGoogleGenerativeAI(**kwargs)
