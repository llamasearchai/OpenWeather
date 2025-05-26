from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime, timezone
from openweather.core.models_shared import AgentResponse, WeatherContext
from openweather.llm.llm_manager import LLMManager

class BaseAgent(ABC):
    """Abstract base class for all weather agents."""
    
    def __init__(
        self,
        llm_manager: LLMManager,
        name: str,
        description: str,
        capabilities: List[str]
    ):
        self.llm_manager = llm_manager
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.memory = []  # Simple in-memory conversation history
        self.last_updated = datetime.now(timezone.utc)

    @abstractmethod
    async def perform_task(
        self,
        task: str,
        context: Optional[WeatherContext] = None,
        **kwargs
    ) -> AgentResponse:
        """Perform the agent's primary task."""
        pass

    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate LLM response with context from memory."""
        full_prompt = self._build_full_prompt(prompt)
        response, _ = await self.llm_manager.generate_text(
            prompt=full_prompt,
            system_prompt=system_prompt or self._default_system_prompt(),
            **kwargs
        )
        self._update_memory(prompt, response)
        return response

    def _build_full_prompt(self, current_prompt: str) -> str:
        """Build prompt with conversation history."""
        history = "\n".join(
            f"User: {item['input']}\nAgent: {item['response']}"
            for item in self.memory[-5:]  # Last 5 exchanges
        )
        return f"{history}\nUser: {current_prompt}" if history else current_prompt

    def _update_memory(self, input_text: str, response: str) -> None:
        """Update conversation memory."""
        self.memory.append({
            "input": input_text,
            "response": response,
            "timestamp": datetime.now(timezone.utc)
        })
        self.last_updated = datetime.now(timezone.utc)

    def _default_system_prompt(self) -> str:
        """Default system prompt for this agent."""
        return (
            f"You are {self.name}, a specialized weather AI. "
            f"Your capabilities: {', '.join(self.capabilities)}. "
            "Be concise, accurate, and professional."
        )

    def get_status(self) -> Dict[str, Any]:
        """Get agent status information."""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "last_updated": self.last_updated,
            "memory_size": len(self.memory)
        } 