#!/usr/bin/env python3
"""
OpenWeather Platform Startup Script

Simple script to run the OpenWeather platform with various configurations.
"""

import argparse
import os
import sys
import subprocess
import logging
from pathlib import Path

def setup_logging(log_level: str):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        import redis
        print("Core dependencies found")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def start_redis():
    """Start Redis server if not running."""
    try:
        subprocess.run(["redis-cli", "ping"], check=True, capture_output=True)
        print("Redis server is running")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Redis server not found or not running")
        print("Please start Redis manually or using Docker:")
        print("  docker run -d -p 6379:6379 redis:7-alpine")
        return False

def run_development(args):
    """Run in development mode."""
    print("Starting OpenWeather Platform in development mode...")
    
    # Set development environment variables
    os.environ["ENVIRONMENT"] = "development"
    os.environ["LOG_LEVEL"] = args.log_level
    os.environ["RELOAD"] = "true"
    
    if args.host:
        os.environ["HOST"] = args.host
    if args.port:
        os.environ["PORT"] = str(args.port)
    
    # Start the application
    from openweather.main import main
    main()

def run_production(args):
    """Run in production mode."""
    print("Starting OpenWeather Platform in production mode...")
    
    # Set production environment variables
    os.environ["ENVIRONMENT"] = "production"
    os.environ["LOG_LEVEL"] = args.log_level
    os.environ["RELOAD"] = "false"
    os.environ["WORKERS"] = str(args.workers)
    
    if args.host:
        os.environ["HOST"] = args.host
    if args.port:
        os.environ["PORT"] = str(args.port)
    
    # Start the application
    from openweather.main import main
    main()

def run_docker(args):
    """Run using Docker."""
    print("Starting OpenWeather Platform with Docker...")
    
    # Build Docker image
    print("Building Docker image...")
    build_cmd = ["docker", "build", "-t", "openweather-platform", "."]
    subprocess.run(build_cmd, check=True)
    
    # Run Docker container
    run_cmd = [
        "docker", "run",
        "-p", f"{args.port}:8000",
        "--name", "openweather-platform",
        "--rm"
    ]
    
    if args.env_file:
        run_cmd.extend(["--env-file", args.env_file])
    
    run_cmd.append("openweather-platform")
    
    print(f"Starting container on port {args.port}...")
    subprocess.run(run_cmd)

def run_tests(args):
    """Run the test suite."""
    print("Running OpenWeather Platform tests...")
    
    test_cmd = ["python", "-m", "pytest"]
    
    if args.coverage:
        test_cmd.extend(["--cov=openweather", "--cov-report=html"])
    
    if args.benchmark:
        test_cmd.extend(["--benchmark-only"])
    
    if args.test_type:
        test_cmd.append(f"tests/{args.test_type}/")
    
    subprocess.run(test_cmd)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OpenWeather Platform Startup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py dev                    # Development mode
  python run.py prod --workers 4       # Production mode with 4 workers
  python run.py docker --port 8080     # Docker mode on port 8080
  python run.py test --coverage        # Run tests with coverage
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Development mode
    dev_parser = subparsers.add_parser("dev", help="Run in development mode")
    dev_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    dev_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    dev_parser.add_argument("--log-level", default="debug", 
                           choices=["debug", "info", "warning", "error"],
                           help="Log level")
    
    # Production mode
    prod_parser = subparsers.add_parser("prod", help="Run in production mode")
    prod_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    prod_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    prod_parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    prod_parser.add_argument("--log-level", default="info",
                            choices=["debug", "info", "warning", "error"],
                            help="Log level")
    
    # Docker mode
    docker_parser = subparsers.add_parser("docker", help="Run with Docker")
    docker_parser.add_argument("--port", type=int, default=8000, help="Port to expose")
    docker_parser.add_argument("--env-file", help="Environment file for Docker")
    
    # Test mode
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    test_parser.add_argument("--benchmark", action="store_true", help="Run benchmark tests only")
    test_parser.add_argument("--test-type", choices=["unit", "integration", "performance"],
                            help="Run specific test type")
    
    # Health check
    health_parser = subparsers.add_parser("health", help="Check system health")
    
    # Setup check
    setup_parser = subparsers.add_parser("setup", help="Check setup and dependencies")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    log_level = getattr(args, "log_level", "info")
    setup_logging(log_level)
    
    print("OpenWeather Platform")
    print("=" * 50)
    
    # Handle commands
    if args.command == "setup":
        print("Checking setup...")
        deps_ok = check_dependencies()
        redis_ok = start_redis()
        
        if deps_ok and redis_ok:
            print("Setup complete! You can now run the platform.")
        else:
            print("Setup incomplete. Please fix the issues above.")
            sys.exit(1)
    
    elif args.command == "health":
        print("Checking system health...")
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"System status: {health_data['status']}")
                print(f"Services: {len(health_data.get('services', {}))} running")
            else:
                print(f"Health check failed: HTTP {response.status_code}")
        except Exception as e:
            print(f"Cannot reach server: {e}")
            print("Make sure the platform is running on localhost:8000")
    
    elif args.command == "dev":
        if not check_dependencies():
            sys.exit(1)
        run_development(args)
    
    elif args.command == "prod":
        if not check_dependencies():
            sys.exit(1)
        run_production(args)
    
    elif args.command == "docker":
        run_docker(args)
    
    elif args.command == "test":
        run_tests(args)

if __name__ == "__main__":
    main() 