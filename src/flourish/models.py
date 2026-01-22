"""Model interfaces for different LLM providers."""

import os
from abc import ABC, abstractmethod
from typing import Optional


class BaseModel(ABC):
    """Abstract base class for LLM models."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response from the model."""
        pass


class AnthropicModel(BaseModel):
    """Anthropic Claude model interface."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        import anthropic
        self.client = anthropic.Anthropic()

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.model_name,
            "max_tokens": 4096,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)
        return response.content[0].text


class OpenAIModel(BaseModel):
    """OpenAI GPT model interface."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        import openai
        self.client = openai.OpenAI()

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=4096,
        )
        return response.choices[0].message.content


class GoogleModel(BaseModel):
    """Google Gemini model interface."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        response = self.model.generate_content(full_prompt)
        return response.text


# Model name mappings (Updated January 2026)
MODEL_MAPPINGS = {
    # Anthropic Claude models (latest: Opus 4.5, Sonnet 4.5)
    "claude-opus-4.5": "claude-opus-4-5-20251101",  # Latest flagship (Nov 2025)
    "claude-sonnet-4.5": "claude-sonnet-4-5-20250929",  # Latest Sonnet (Sep 2025)
    # "claude-sonnet-4": "claude-sonnet-4-20250514",  # Legacy Sonnet 4
    # "claude-opus-4": "claude-opus-4-20250514",  # Legacy Opus 4
    # "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",  # Legacy Claude 3.5
    # "claude-3-opus": "claude-3-opus-20240229",  # Legacy Claude 3

    # OpenAI models (latest: GPT-5.2)
    "gpt-5.2": "gpt-5.2",  # Latest flagship (Dec 2025)
    "gpt-5.2-pro": "gpt-5.2-pro",  # Smartest variant with more compute
    # "gpt-5.2-instant": "gpt-5.2-chat-latest",  # Fast variant for ChatGPT
    # "gpt-4o": "gpt-4o",  # Legacy GPT-4 Omni
    # "gpt-4o-mini": "gpt-4o-mini",  # Legacy GPT-4 Omni Mini
    # "gpt-4-turbo": "gpt-4-turbo",  # Legacy GPT-4 Turbo

    # Google Gemini models (latest: Gemini 3.0)
    "gemini-3-pro": "gemini-3-pro-preview",  # Latest Gemini 3 Pro (preview)
    "gemini-3-flash": "gemini-3-flash-preview",  # Latest Gemini 3 Flash (preview)
    # "gemini-2.5-pro": "gemini-2.5-pro",  # Legacy Gemini 2.5
    # "gemini-2.0-flash": "gemini-2.0-flash-001",  # Legacy (retiring March 3, 2026)
    # "gemini-1.5-pro": "gemini-1.5-pro",  # Legacy Gemini 1.5
    # "gemini-1.5-flash": "gemini-1.5-flash",  # Legacy Gemini 1.5
}


def load_model(model_name: str) -> BaseModel:
    """
    Factory function to load the appropriate model interface.

    Args:
        model_name: Name of the model to load. Can be a shorthand name
                   (e.g., "claude-sonnet-4") or full model ID.

    Returns:
        BaseModel instance for the specified model.

    Raises:
        ValueError: If the model name is not recognized.
    """
    # Resolve shorthand names
    resolved_name = MODEL_MAPPINGS.get(model_name, model_name)

    # Determine provider from model name
    if "claude" in resolved_name.lower():
        return AnthropicModel(resolved_name)
    elif "gpt" in resolved_name.lower() or "o1" in resolved_name.lower():
        return OpenAIModel(resolved_name)
    elif "gemini" in resolved_name.lower():
        return GoogleModel(resolved_name)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: {list(MODEL_MAPPINGS.keys())}"
        )


def get_available_models() -> list[str]:
    """Return list of available model shorthand names."""
    return list(MODEL_MAPPINGS.keys())
