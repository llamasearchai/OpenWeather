from typing import Dict, List, Optional, Any
import asyncio
from openweather.agents.base_agent import BaseAgent
from openweather.core.models_shared import AgentResponse, WeatherContext

class AgentManager:
    """Manages multiple specialized agents and routes tasks appropriately."""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}

    def register_agent(self, agent: BaseAgent) -> None:
        """Register a new agent with the manager."""
        self.agents[agent.name] = agent
        self.agent_capabilities[agent.name] = agent.capabilities
        print(f"Registered agent: {agent.name}")

    async def route_task(
        self,
        task: str,
        context: Optional[WeatherContext] = None,
        preferred_agent: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """Route a task to the most appropriate agent."""
        # Use preferred agent if specified and available
        if preferred_agent and preferred_agent in self.agents:
            return await self.agents[preferred_agent].perform_task(
                task, context, **kwargs
            )
        
        # Otherwise find best matching agent
        best_agent = self._find_best_agent_for_task(task)
        if best_agent:
            return await best_agent.perform_task(task, context, **kwargs)
        
        return AgentResponse(
            status="error",
            response_text="No suitable agent found for this task"
        )

    def _find_best_agent_for_task(self, task: str) -> Optional[BaseAgent]:
        """Determine which agent is best suited for a given task."""
        task_lower = task.lower()
        best_score = 0
        best_agent = None
        
        for agent_name, capabilities in self.agent_capabilities.items():
            score = self._calculate_match_score(task_lower, capabilities)
            if score > best_score:
                best_score = score
                best_agent = self.agents[agent_name]
        
        return best_agent

    def _calculate_match_score(self, task: str, capabilities: List[str]) -> int:
        """Calculate how well an agent's capabilities match the task."""
        score = 0
        for capability in capabilities:
            if capability.lower() in task:
                score += 1
        return score

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents and their capabilities."""
        return [
            {
                "name": agent.name,
                "description": agent.description,
                "capabilities": agent.capabilities,
                "last_updated": agent.last_updated
            }
            for agent in self.agents.values()
        ] 