from typing import Optional, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import typer
from openweather.llm.llm_manager import LLMManager
import asyncio
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

console = Console()

class InteractiveCLI:
    """Enhanced interactive CLI with rich features."""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.session_history = []
        self.context = {
            "user_preferences": {
                "units": "metric",
                "detail_level": "professional"
            }
        }
        self.console = Console(record=True)
        self.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    def start_interactive_session(self):
        """Enhanced interactive session with more features"""
        self.console.print(Panel.fit(
            "[bold blue]OpenWeather Interactive Analyst[/bold blue]\n"
            "[dim]Version 2.0 | Professional Meteorology Mode[/dim]",
            subtitle="Type 'help' for commands"
        ))
        
        while True:
            try:
                user_input = self._get_user_input()
                if self._handle_special_commands(user_input):
                    continue
                    
                response = self._process_input(user_input)
                self._display_response(response)
                
            except KeyboardInterrupt:
                self.console.print("\n[bold yellow]Session saved. Use 'exit' to quit[/bold yellow]")
            except Exception as e:
                self.console.print(f"[bold red]Error: {str(e)}[/bold red]")
                logger.exception("CLI error")

    def _handle_special_commands(self, input_text: str) -> bool:
        """Handle special commands like help, export, etc."""
        if input_text.lower() == 'help':
            self._show_help()
            return True
        elif input_text.lower() == 'export':
            self._export_session()
            return True
        return False

    def _show_help(self):
        """Show help information"""
        help_text = """
        [bold]Available Commands:[/bold]
        - help: Show this help
        - export: Export session history
        - set units <metric/imperial>: Change units
        - set detail <basic/professional>: Change detail level
        - exit/quit: End session
        """
        self.console.print(Panel(help_text, title="Help", border_style="blue"))

    def _export_session(self):
        """Export session history to file"""
        filename = f"weather_session_{self.session_id}.md"
        with open(filename, "w") as f:
            f.write(f"# OpenWeather Session {self.session_id}\n\n")
            for item in self.session_history:
                f.write(f"## {item['timestamp']}\n")
                f.write(f"**Question:** {item['input']}\n\n")
                f.write(f"**Response:**\n{item['response']}\n\n")
        self.console.print(f"[green]Session exported to {filename}[/green]")

    def _get_user_input(self) -> str:
        """Get user input with rich prompt."""
        return typer.prompt(
            "Ask a weather question",
            prompt_suffix="\n> ",
            show_default=False
        )

    def _process_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input and generate response."""
        # Store in session history
        self.session_history.append({
            "input": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate LLM response
        prompt = self._build_prompt(user_input)
        response_text, metadata = asyncio.run(
            self.llm_manager.generate_text(prompt)
        )
        
        return {
            "input": user_input,
            "response": response_text,
            "metadata": metadata
        }

    def _display_response(self, response: Dict[str, Any]):
        """Enhanced display with visualizations"""
        # Display main analysis
        self.console.print(Panel(
            Markdown(response["response"]),
            title="Analysis",
            border_style="green"
        ))
        
        # Display metadata
        self.console.print(f"[dim]Model: {response['metadata'].get('model_used', 'unknown')}[/dim]")
        
        # Add visualization if forecast data exists
        if "forecast_data" in response.get("metadata", {}):
            self._display_forecast_chart(response["metadata"]["forecast_data"])

    def _display_forecast_chart(self, forecast_data: dict):
        """Generate terminal weather chart"""
        try:
            from rich.table import Table
            from rich.text import Text
            
            table = Table(title="7-Day Forecast", show_header=True, header_style="bold magenta")
            table.add_column("Date")
            table.add_column("High/Low")
            table.add_column("Conditions")
            table.add_column("Precip")
            table.add_column("Wind")
            
            for day in forecast_data["daily_forecasts"][:7]:
                temp_text = Text()
                temp_text.append(f"{day['temp_max']}°", style="red")
                temp_text.append(" / ", style="dim")
                temp_text.append(f"{day['temp_min']}°", style="blue")
                
                precip_text = Text()
                precip_text.append(f"{day['precip']}mm", style="blue" if day['precip'] > 5 else "dim")
                
                table.add_row(
                    day['date'],
                    temp_text,
                    day['condition'],
                    precip_text,
                    f"{day['wind_speed']} km/h {day['wind_dir']}"
                )
                
            self.console.print(table)
        except ImportError:
            self.console.print("[yellow]Install rich with `pip install rich` for better visualizations[/yellow]")

    def _build_prompt(self, user_input: str) -> str:
        """Build the LLM prompt with context."""
        return f"""
        User Question: {user_input}
        Context: {json.dumps(self.context, indent=2)}
        Previous Interactions: {len(self.session_history)}
        
        Provide a detailed, professional weather analysis considering:
        - Current meteorological knowledge
        - Safety implications
        - Relevant weather patterns
        """ 