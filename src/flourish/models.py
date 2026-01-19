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


# Model name mappings
MODEL_MAPPINGS = {
    # Anthropic models
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "claude-opus-4": "claude-opus-4-20250514",
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-opus": "claude-3-opus-20240229",
    # OpenAI models
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4-turbo": "gpt-4-turbo",
    # Google models
    "gemini-2.0-flash": "gemini-2.0-flash-exp",
    "gemini-1.5-pro": "gemini-1.5-pro",
    "gemini-1.5-flash": "gemini-1.5-flash",
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
