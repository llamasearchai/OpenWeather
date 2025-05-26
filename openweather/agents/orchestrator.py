from typing import Dict, List, Optional, Any
import asyncio
from openweather.agents.base_agent import BaseAgent
from openweather.core.models_shared import AgentResponse, WeatherContext
import logging
from datetime import datetime, timezone

from openweather.llm.llm_manager import LLMManager
from openweather.services.forecast_service import ForecastService
from openweather.agents.weather_analyst import WeatherAnalyst
from openweather.agents.marine_agent import MarineWeatherAgent
from openweather.agents.travel_agent import TravelWeatherAgent
from openweather.llm.prompt_engineering import PromptEngineer
from openweather.core.config import settings

logger = logging.getLogger(__name__)

class MultiAgentOrchestrator:
    """Coordinates multiple agents for complex weather analysis tasks."""
    
    def __init__(self, agent_manager):
        self.agent_manager = agent_manager
        self.task_workflows = {
            "trip_planning": ["travel", "weather"],
            "farming_decision": ["agriculture", "long_range"],
            "event_planning": ["weather", "severity"]
        }

    async def execute_workflow(
        self,
        workflow_name: str,
        task: str,
        context: Optional[WeatherContext] = None
    ) -> Dict[str, Any]:
        """Execute a predefined multi-agent workflow."""
        if workflow_name not in self.task_workflows:
            return {"error": f"Unknown workflow: {workflow_name}"}

        results = {}
        for agent_type in self.task_workflows[workflow_name]:
            try:
                response = await self.agent_manager.route_task(
                    task=task,
                    context=context,
                    preferred_agent=agent_type
                )
                results[agent_type] = response
            except Exception as e:
                results[agent_type] = {
                    "error": str(e),
                    "status": "failed"
                }

        # Generate consolidated report if multiple agents succeeded
        if len([r for r in results.values() if isinstance(r, AgentResponse) and r.status == "success"]) > 1:
            results["consolidated"] = await self._generate_consolidated_report(
                workflow_name, task, results
            )

        return results

    async def _generate_consolidated_report(
        self,
        workflow_name: str,
        task: str,
        agent_results: Dict[str, Any]
    ) -> AgentResponse:
        """Generate a consolidated report from multiple agent outputs."""
        prompt = f"Consolidate these {workflow_name} analyses:\n\n"
        for agent_name, result in agent_results.items():
            if isinstance(result, AgentResponse) and result.status == "success":
                prompt += f"{agent_name.upper()} ANALYSIS:\n{result.response_text}\n\n"
        
        prompt += f"Original task: {task}\n\nProvide a consolidated recommendation."
        
        consolidated_response, _ = await self.agent_manager.llm_manager.generate_text(
            prompt=prompt,
            system_prompt="You are an expert analyst. Synthesize these reports into one cohesive recommendation."
        )
        
        return AgentResponse(
            status="success",
            response_text=consolidated_response,
            metadata={"workflow": workflow_name}
        )

class AgentOrchestrator:
    """Orchestrates multiple specialized weather agents."""
    
    def __init__(self, llm_manager: LLMManager, forecast_service: ForecastService):
        self.llm_manager = llm_manager
        self.forecast_service = forecast_service
        self.prompt_engineer = PromptEngineer()
        
        # Initialize specialized agents
        self.agents = {
            "general": WeatherAnalyst(llm_manager, forecast_service),
            "marine": MarineWeatherAgent(llm_manager, forecast_service),
            "travel": TravelWeatherAgent(llm_manager, forecast_service),
            # Add more agents as needed
        }
        
        # Agent routing intelligence
        self.routing_keywords = {
            "marine": ["ocean", "sea", "sailing", "fishing", "maritime", "coastal"],
            "travel": ["trip", "vacation", "travel", "flight", "journey", "holiday"],
            "agriculture": ["farming", "crops", "irrigation", "harvest", "livestock"],
            "aviation": ["flight", "flying", "turbulence", "visibility", "runway"]
        }

    async def process_query(
        self,
        query: str,
        specialist: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        location: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a query through the appropriate specialist agent."""
        context = context or {}
        
        # Auto-route to specialist if not specified
        if not specialist or specialist == "auto":
            specialist = self._route_to_specialist(query)
        
        # Get the appropriate agent
        agent = self.agents.get(specialist, self.agents["general"])
        
        try:
            # Add query metadata
            context.update({
                "query": query,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "routing": {
                    "selected_specialist": specialist,
                    "auto_routed": specialist != "general"
                }
            })
            
            # Process with the selected agent
            response = await agent.analyze(query, context, location)
            
            # Enhance response with orchestrator metadata
            response.update({
                "specialist_used": specialist,
                "orchestrator_version": "1.0",
                "confidence_score": self._calculate_confidence(response),
                "follow_up_suggestions": self._generate_follow_ups(query, specialist)
            })
            
            return response
            
        except Exception as e:
            logger.exception(f"Agent orchestration failed: {e}")
            return {
                "error": str(e),
                "specialist_used": specialist,
                "analysis": "I apologize, but I encountered an error processing your query. Please try again.",
                "recommendations": ["Check your query format", "Try a different specialist"]
            }

    def _route_to_specialist(self, query: str) -> str:
        """Intelligently route query to appropriate specialist."""
        query_lower = query.lower()
        
        # Count keyword matches for each specialist
        specialist_scores = {}
        for specialist, keywords in self.routing_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                specialist_scores[specialist] = score
        
        # Return specialist with highest score, or general if no matches
        if specialist_scores:
            return max(specialist_scores, key=specialist_scores.get)
        return "general"

    def _calculate_confidence(self, response: Dict[str, Any]) -> float:
        """Calculate confidence score for the response."""
        confidence_factors = []
        
        # Analysis length factor
        if response.get("analysis"):
            analysis_length = len(response["analysis"])
            confidence_factors.append(min(1.0, analysis_length / 500))
        
        # Recommendation count factor
        if response.get("recommendations"):
            rec_count = len(response["recommendations"])
            confidence_factors.append(min(1.0, rec_count / 5))
        
        # Weather data availability factor
        if response.get("weather_data"):
            confidence_factors.append(0.9)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5

    def _generate_follow_ups(self, query: str, specialist: str) -> List[str]:
        """Generate intelligent follow-up suggestions."""
        base_suggestions = [
            "Would you like more detailed hourly forecasts?",
            "Do you need information for a specific date range?",
            "Would you like to see historical weather patterns?"
        ]
        
        specialist_suggestions = {
            "marine": [
                "Need wave height and sea conditions?",
                "Want tidal information for your area?",
                "Interested in marine weather alerts?"
            ],
            "travel": [
                "Planning activities based on weather?",
                "Need packing recommendations?",
                "Want alternative destination suggestions?"
            ],
            "agriculture": [
                "Need soil temperature information?",
                "Want irrigation recommendations?",
                "Interested in pest risk assessment?"
            ]
        }
        
        return specialist_suggestions.get(specialist, base_suggestions)

    async def get_agent_capabilities(self) -> Dict[str, Any]:
        """Get information about available agents and their capabilities."""
        capabilities = {}
        
        for name, agent in self.agents.items():
            capabilities[name] = {
                "description": getattr(agent, "description", f"{name} weather specialist"),
                "keywords": self.routing_keywords.get(name, []),
                "supported_queries": getattr(agent, "supported_queries", []),
                "data_sources": getattr(agent, "data_sources", ["general weather"]),
                "example_queries": getattr(agent, "example_queries", [])
            }
        
        return capabilities

    async def batch_process(
        self,
        queries: List[Dict[str, Any]],
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """Process multiple queries concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(query_data):
            async with semaphore:
                return await self.process_query(**query_data)
        
        tasks = [process_single(query) for query in queries]
        return await asyncio.gather(*tasks, return_exceptions=True) 