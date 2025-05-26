#!/bin/bash
# Production-ready installation script for OpenWeather Enterprise Platform
# Supports Linux, macOS, and Windows (WSL)

set -euo pipefail

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Configuration
readonly MIN_PYTHON_VERSION="3.9"
readonly RECOMMENDED_PYTHON_VERSION="3.11"
readonly REQUIRED_MEMORY_GB=4
readonly RECOMMENDED_MEMORY_GB=8
readonly REQUIRED_DISK_GB=10
readonly POETRY_VERSION="1.7.1"
readonly DOCKER_MIN_VERSION="20.10"
readonly COMPOSE_MIN_VERSION="2.0"

# Installation modes
INSTALL_MODE="${1:-full}"  # Options: minimal, standard, full, enterprise, drone
ENABLE_GPU="${ENABLE_GPU:-auto}"
ENABLE_DRONE="${ENABLE_DRONE:-true}"
ENABLE_MONITORING="${ENABLE_MONITORING:-true}"
SKIP_DOCKER="${SKIP_DOCKER:-false}"

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${PURPLE}[DEBUG]${NC} $1"
    fi
}

# Error handling
error_exit() {
    log_error "$1"
    exit 1
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Version comparison
version_compare() {
    printf '%s\n%s' "$1" "$2" | sort -V | head -n1
}

# Print banner
print_banner() {
    echo -e "${CYAN}"
    cat << 'EOF'
   ____                 _       __          _   _               
  / __ \___  ___  ___  | |     / /__  ___ _| |_| |__  ___  ____
 / /_/ / _ \/ _ \/ _ \ | | /| / / _ \/ _ `| __| '_ \/ _ \/ ___/
/ ____/  __/  __/_/ / | |/ |/ /  __/ /_| | |_| | | /  __/ /   
\___/ \___/\___/_/   |__/|__/\___/\__,_|\__|_| |_\___/_/    

Enterprise-Grade Weather Analytics Platform v3.0.0
EOF
    echo -e "${NC}"
    echo "Installation Mode: ${INSTALL_MODE}"
    echo "GPU Support: ${ENABLE_GPU}"
    echo "Drone Support: ${ENABLE_DRONE}"
    echo "Monitoring: ${ENABLE_MONITORING}"
    echo
}

# Check system requirements
check_system_requirements() {
    log_step "Checking system requirements..."
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        if command_exists lsb_release; then
            DISTRO=$(lsb_release -si)
            VERSION=$(lsb_release -sr)
        elif [[ -f /etc/os-release ]]; then
            . /etc/os-release
            DISTRO=$ID
            VERSION=$VERSION_ID
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        DISTRO="macos"
        VERSION=$(sw_vers -productVersion)
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
        DISTRO="windows"
    else
        error_exit "Unsupported operating system: $OSTYPE"
    fi
    
    log_info "Detected OS: $OS ($DISTRO $VERSION)"
    
    # Check architecture
    ARCH=$(uname -m)
    log_info "Architecture: $ARCH"
    
    # Check for Apple Silicon
    if [[ "$OS" == "macos" ]] && [[ "$ARCH" == "arm64" ]]; then
        APPLE_SILICON=true
        log_info "Apple Silicon detected - MLX acceleration available"
    else
        APPLE_SILICON=false
    fi
    
    # Check Python version
    check_python_version
    
    # Check available memory
    check_memory_requirements
    
    # Check disk space
    check_disk_space
    
    # Check GPU availability
    check_gpu_support
}

# Check Python version
check_python_version() {
    local python_cmd=""
    
    for cmd in python3.12 python3.11 python3.10 python3.9 python3 python; do
        if command_exists "$cmd"; then
            python_cmd="$cmd"
            break
        fi
    done
    
    if [[ -z "$python_cmd" ]]; then
        error_exit "Python is not installed. Please install Python $MIN_PYTHON_VERSION or higher."
    fi
    
    PYTHON_CMD="$python_cmd"
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    log_info "Detected Python version: $PYTHON_VERSION ($PYTHON_CMD)"
    
    # Version comparison
    if [[ $(version_compare "$PYTHON_VERSION" "$MIN_PYTHON_VERSION") != "$PYTHON_VERSION" ]]; then
        error_exit "Python $MIN_PYTHON_VERSION or higher is required. Found: $PYTHON_VERSION"
    fi
    
    if [[ $(version_compare "$PYTHON_VERSION" "$RECOMMENDED_PYTHON_VERSION") != "$PYTHON_VERSION" ]]; then
        log_warn "Python $RECOMMENDED_PYTHON_VERSION or higher is recommended for best performance"
    fi
}

# Check memory requirements
check_memory_requirements() {
    local available_memory
    
    if [[ "$OS" == "linux" ]]; then
        if [[ -f /proc/meminfo ]]; then
            available_memory=$(awk '/MemAvailable/{printf "%.0f", $2/1024/1024}' /proc/meminfo)
        else
            available_memory=$(free -g | awk '/^Mem:/{print $2}')
        fi
    elif [[ "$OS" == "macos" ]]; then
        available_memory=$(echo "$(sysctl -n hw.memsize) / 1024 / 1024 / 1024" | bc)
    else
        available_memory=8  # Assume sufficient for Windows
    fi
    
    log_info "Available memory: ${available_memory}GB"
    
    if [[ $available_memory -lt $REQUIRED_MEMORY_GB ]]; then
        error_exit "Insufficient memory. Required: ${REQUIRED_MEMORY_GB}GB, Available: ${available_memory}GB"
    fi
    
    if [[ $available_memory -lt $RECOMMENDED_MEMORY_GB ]]; then
        log_warn "Memory below recommended. Recommended: ${RECOMMENDED_MEMORY_GB}GB, Available: ${available_memory}GB"
        log_warn "Consider using 'minimal' installation mode for better performance"
    fi
}

# Check disk space
check_disk_space() {
    local available_disk
    available_disk=$(df . | tail -1 | awk '{print int($4/1024/1024)}')
    
    log_info "Available disk space: ${available_disk}GB"
    
    if [[ $available_disk -lt $REQUIRED_DISK_GB ]]; then
        error_exit "Insufficient disk space. Required: ${REQUIRED_DISK_GB}GB, Available: ${available_disk}GB"
    fi
}

# Check GPU support
check_gpu_support() {
    local gpu_available=false
    
    # Check for NVIDIA GPU
    if command_exists nvidia-smi; then
        if nvidia-smi >/dev/null 2>&1; then
            log_info "NVIDIA GPU detected"
            gpu_available=true
            GPU_TYPE="nvidia"
        fi
    fi
    
    # Check for AMD GPU (Linux)
    if [[ "$OS" == "linux" ]] && command_exists rocm-smi; then
        if rocm-smi >/dev/null 2>&1; then
            log_info "AMD GPU detected"
            gpu_available=true
            GPU_TYPE="amd"
        fi
    fi
    
    # Check for Apple Silicon GPU
    if [[ "$APPLE_SILICON" == "true" ]]; then
        log_info "Apple Silicon GPU detected"
        gpu_available=true
        GPU_TYPE="apple"
    fi
    
    if [[ "$gpu_available" == "false" ]]; then
        log_warn "No GPU detected. CPU-only mode will be used."
        if [[ "$ENABLE_GPU" == "auto" ]]; then
            ENABLE_GPU="false"
        fi
    else
        if [[ "$ENABLE_GPU" == "auto" ]]; then
            ENABLE_GPU="true"
        fi
    fi
    
    log_info "GPU support: $ENABLE_GPU"
}

# Install system dependencies
install_system_dependencies() {
    log_step "Installing system dependencies..."
    
    case "$OS" in
        "linux")
            install_linux_dependencies
            ;;
        "macos")
            install_macos_dependencies
            ;;
        "windows")
            install_windows_dependencies
            ;;
    esac
}

# Install Linux dependencies
install_linux_dependencies() {
    case "$DISTRO" in
        "ubuntu"|"debian")
            sudo apt-get update
            sudo apt-get install -y \
                curl wget git build-essential cmake \
                libssl-dev libffi-dev python3-dev python3-pip python3-venv \
                sqlite3 libsqlite3-dev \
                libpq-dev \
                libatlas-base-dev liblapack-dev libblas-dev gfortran \
                libopencv-dev python3-opencv \
                redis-tools \
                htop tmux screen \
                jq bc
            ;;
        "fedora"|"rhel"|"centos")
            sudo dnf update -y
            sudo dnf install -y \
                curl wget git gcc gcc-c++ make cmake \
                openssl-devel libffi-devel python3-devel python3-pip \
                sqlite sqlite-devel \
                postgresql-devel \
                atlas-devel lapack-devel blas-devel gcc-gfortran \
                opencv-devel python3-opencv \
                redis \
                htop tmux screen \
                jq bc
            ;;
        "arch"|"manjaro")
            sudo pacman -Syu --noconfirm \
                curl wget git base-devel cmake \
                openssl libffi python python-pip \
                sqlite postgresql-libs \
                atlas-lapack blas gcc-fortran \
                opencv python-opencv \
                redis \
                htop tmux screen \
                jq bc
            ;;
        *)
            log_warn "Unsupported Linux distribution: $DISTRO. Please install dependencies manually."
            ;;
    esac
    
    # Install Docker if not present and not skipped
    if [[ "$SKIP_DOCKER" != "true" ]] && ! command_exists docker; then
        install_docker_linux
    fi
}

# Install macOS dependencies
install_macos_dependencies() {
    # Check for Homebrew
    if ! command_exists brew; then
        log_info "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    
    brew update
    brew install \
        python@3.11 git cmake \
        sqlite postgresql redis \
        opencv \
        htop tmux \
        jq bc
    
    # Install Xcode command line tools if needed
    if ! xcode-select -p &>/dev/null; then
        log_info "Installing Xcode command line tools..."
        xcode-select --install
    fi
    
    # Install Docker if not present and not skipped
    if [[ "$SKIP_DOCKER" != "true" ]] && ! command_exists docker; then
        brew install --cask docker
        log_warn "Please start Docker Desktop manually before continuing"
    fi
}

# Install Windows dependencies (WSL)
install_windows_dependencies() {
    log_info "Windows detected. Ensure you're running this in WSL2 for best compatibility."
    
    # Install via apt (assuming Ubuntu WSL)
    install_linux_dependencies
}

# Install Docker on Linux
install_docker_linux() {
    log_info "Installing Docker..."
    
    # Remove old versions
    sudo apt-get remove -y docker docker-engine docker.io containerd runc || true
    
    # Install Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    rm get-docker.sh
    
    # Add user to docker group
    sudo usermod -aG docker "$USER"
    
    # Install Docker Compose
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    
    log_warn "Please log out and back in for Docker group membership to take effect"
}

# Install Python package managers
install_package_managers() {
    log_step "Installing Python package managers..."
    
    # Upgrade pip
    $PYTHON_CMD -m pip install --upgrade pip wheel setuptools
    
    # Install Poetry
    if ! command_exists poetry; then
        log_info "Installing Poetry $POETRY_VERSION..."
        curl -sSL https://install.python-poetry.org | POETRY_VERSION=$POETRY_VERSION $PYTHON_CMD -
        
        # Add Poetry to PATH
        export PATH="$HOME/.local/bin:$PATH"
        
        # Update shell profiles
        for profile in ~/.bashrc ~/.zshrc ~/.profile; do
            if [[ -f "$profile" ]]; then
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$profile"
            fi
        done
    else
        log_info "Poetry is already installed: $(poetry --version)"
    fi
    
    # Install UV for faster dependency resolution
    if ! command_exists uv; then
        log_info "Installing UV..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
    else
        log_info "UV is already installed: $(uv --version)"
    fi
}

# Setup project environment
setup_project_environment() {
    log_step "Setting up project environment..."
    
    # Create project directory structure
    mkdir -p \
        data/{app_data,vector_store_chroma,logs,cache,models,backups} \
        static/uploads \
        deployment/{grafana,prometheus,nginx,k8s} \
        docs/{api,guides,tutorials} \
        scripts/{monitoring,backup,deployment}
    
    # Set up environment file
    setup_environment_file
    
    # Install dependencies based on mode
    install_dependencies
    
    # Initialize database
    setup_database
    
    # Configure monitoring if enabled
    if [[ "$ENABLE_MONITORING" == "true" ]]; then
        setup_monitoring
    fi
}

# Setup environment file
setup_environment_file() {
    if [[ ! -f .env ]]; then
        log_info "Creating .env file from template..."
        cat > .env << EOF
# OpenWeather Enterprise Configuration
# Generated on $(date)

# Core Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
DEBUG=false

# Database Configuration
DATABASE_URL=postgresql://openweather:openweather123@localhost:5432/openweather
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_TIMEOUT=60

# Security
SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET_KEY=$(openssl rand -hex 32)
ENCRYPTION_KEY=$(openssl rand -hex 32)

# API Keys (add your keys here)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
HF_API_KEY=
SENTRY_DSN=

# LLM Configuration
USE_OLLAMA=${ENABLE_GPU}
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3
USE_MLX=${APPLE_SILICON}
MLX_MODEL_PATH=mlx-community/Mistral-7B-Instruct-v0.2-MLX
DEFAULT_LLM_PROVIDER=ollama

# Drone Configuration
ENABLE_DRONE_SUPPORT=${ENABLE_DRONE}
MAVLINK_CONNECTION=tcp:127.0.0.1:5760
DRONE_SAFETY_MARGINS={"wind_speed": 12, "visibility": 1000, "precipitation": 0.1}

# Vector Store Configuration
VECTOR_STORE_PATH=data/vector_store_chroma
CHROMADB_HOST=localhost
CHROMADB_PORT=8001

# Monitoring Configuration
ENABLE_MONITORING=${ENABLE_MONITORING}
PROMETHEUS_GATEWAY=http://localhost:9090
GRAFANA_URL=http://localhost:3001

# Celery Configuration
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# Performance Tuning
MAX_WORKERS=4
CACHE_TTL=3600
RATE_LIMIT_PER_MINUTE=100

# Data Sources
OPENMETEO_API_URL=https://api.open-meteo.com/v1
ENABLE_DATA_SIMULATION=true
DATA_RETENTION_DAYS=90

# Backup Configuration
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"
BACKUP_RETENTION_DAYS=30
EOF
        log_success "Environment file created: .env"
    else
        log_info "Environment file already exists: .env"
    fi
}

# Install dependencies
install_dependencies() {
    log_step "Installing Python dependencies..."
    
    # Configure Poetry
    poetry config virtualenvs.in-project true
    poetry config virtualenvs.prefer-active-python true
    
    # Install based on mode
    local extras=""
    case "$INSTALL_MODE" in
        "minimal")
            extras=""
            ;;
        "standard")
            extras="monitoring"
            ;;
        "full")
            extras="full,monitoring"
            ;;
        "enterprise")
            extras="enterprise,monitoring"
            ;;
        "drone")
            extras="drone,monitoring"
            ;;
    esac
    
    # Add GPU-specific extras
    if [[ "$ENABLE_GPU" == "true" ]]; then
        if [[ "$APPLE_SILICON" == "true" ]]; then
            extras="${extras},mlx"
        fi
    fi
    
    # Install dependencies
    if [[ -n "$extras" ]]; then
        poetry install --extras "$extras"
    else
        poetry install
    fi
    
    # Install additional packages for development
    if [[ "$ENVIRONMENT" == "development" ]]; then
        poetry install --with dev
    fi
}

# Setup database
setup_database() {
    log_step "Setting up database..."
    
    # Check if PostgreSQL is available
    if command_exists psql && pg_isready >/dev/null 2>&1; then
        log_info "PostgreSQL is available, initializing database..."
        
        # Create database and user if they don't exist
        sudo -u postgres psql -c "CREATE DATABASE openweather;" 2>/dev/null || true
        sudo -u postgres psql -c "CREATE USER openweather WITH PASSWORD 'openweather123';" 2>/dev/null || true
        sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE openweather TO openweather;" 2>/dev/null || true
        
        # Run migrations
        poetry run alembic upgrade head
    else
        log_warn "PostgreSQL not available. Using SQLite for development."
        # Update .env to use SQLite
        sed -i.bak 's|DATABASE_URL=postgresql.*|DATABASE_URL=sqlite:///data/openweather.db|' .env
    fi
}

# Setup monitoring
setup_monitoring() {
    log_step "Setting up monitoring configuration..."
    
    # Create Prometheus configuration
    mkdir -p deployment/prometheus
    cat > deployment/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'openweather-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['localhost:8080']

  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']

  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - localhost:9093
EOF
    
    # Create Grafana datasource configuration
    mkdir -p deployment/grafana/datasources
    cat > deployment/grafana/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF
    
    log_info "Monitoring configuration created"
}

# Post-installation setup
post_installation_setup() {
    log_step "Running post-installation setup..."
    
    # Generate API documentation
    if command_exists poetry; then
        poetry run python -c "
from openweather.api.main import app
import json
with open('docs/api/openapi.json', 'w') as f:
    json.dump(app.openapi(), f, indent=2)
"
    fi
    
    # Create systemd service (Linux only)
    if [[ "$OS" == "linux" ]] && command_exists systemctl; then
        create_systemd_service
    fi
    
    # Create startup scripts
    create_startup_scripts
    
    # Set up log rotation
    setup_log_rotation
}

# Create systemd service
create_systemd_service() {
    local service_file="/etc/systemd/system/openweather.service"
    
    sudo tee "$service_file" > /dev/null << EOF
[Unit]
Description=OpenWeather API Service
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=notify
User=$(whoami)
Group=$(whoami)
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/.venv/bin
ExecStart=$(pwd)/.venv/bin/python -m openweather.api.main
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable openweather
    
    log_info "Systemd service created: $service_file"
}

# Create startup scripts
create_startup_scripts() {
    # Create development startup script
    cat > start-dev.sh << 'EOF'
#!/bin/bash
# Development startup script

set -e

echo "Starting OpenWeather in development mode..."

# Activate virtual environment
source .venv/bin/activate

# Start services
if command -v docker-compose >/dev/null 2>&1; then
    echo "Starting support services..."
    docker-compose -f deployment/docker-compose.dev.yml up -d postgres redis ollama
    
    # Wait for services to be ready
    echo "Waiting for services to be ready..."
    sleep 10
fi

# Start the API server
echo "Starting OpenWeather API server..."
poetry run python -m openweather.api.main --reload

EOF
    chmod +x start-dev.sh
    
    # Create production startup script
    cat > start-prod.sh << 'EOF'
#!/bin/bash
# Production startup script

set -e

echo "Starting OpenWeather in production mode..."

# Export production environment
export ENVIRONMENT=production
export LOG_LEVEL=INFO

# Start all services
docker-compose up -d

echo "OpenWeather started successfully!"
echo "API: http://localhost:8000"
echo "Web Dashboard: http://localhost:3000"
echo "Monitoring: http://localhost:3001"

EOF
    chmod +x start-prod.sh
    
    log_info "Startup scripts created: start-dev.sh, start-prod.sh"
}

# Setup log rotation
setup_log_rotation() {
    if [[ "$OS" == "linux" ]] && command_exists logrotate; then
        sudo tee /etc/logrotate.d/openweather > /dev/null << 'EOF'
/var/log/openweather/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 openweather openweather
    postrotate
        systemctl reload openweather
    endscript
}
EOF
        log_info "Log rotation configured"
    fi
}

# Verify installation
verify_installation() {
    log_step "Verifying installation..."
    
    local errors=0
    
    # Check Python installation
    if ! poetry run python -c "import openweather; print(f'OpenWeather v{openweather.__version__} imported successfully')"; then
        log_error "Python package import failed"
        ((errors++))
    fi
    
    # Check dependencies
    local required_deps=("fastapi" "uvicorn" "pydantic" "httpx")
    for dep in "${required_deps[@]}"; do
        if ! poetry run python -c "import $dep" 2>/dev/null; then
            log_error "Required dependency not found: $dep"
            ((errors++))
        fi
    done
    
    # Check configuration
    if [[ ! -f .env ]]; then
        log_error "Environment file not found: .env"
        ((errors++))
    fi
    
    # Check Docker (if not skipped)
    if [[ "$SKIP_DOCKER" != "true" ]]; then
        if ! command_exists docker; then
            log_error "Docker not found"
            ((errors++))
        elif ! docker version >/dev/null 2>&1; then
            log_error "Docker not running"
            ((errors++))
        fi
    fi
    
    if [[ $errors -eq 0 ]]; then
        log_success "Installation verification passed!"
        return 0
    else
        log_error "Installation verification failed with $errors error(s)"
        return 1
    fi
}

# Print installation summary
print_summary() {
    log_success "OpenWeather Enterprise Platform installation completed!"
    echo
    echo -e "${CYAN}Installation Summary:${NC}"
    echo "  • Mode: $INSTALL_MODE"
    echo "  • Python: $PYTHON_VERSION"
    echo "  • OS: $OS ($DISTRO $VERSION)"
    echo "  • Architecture: $ARCH"
    echo "  • GPU Support: $ENABLE_GPU"
    echo "  • Drone Support: $ENABLE_DRONE"
    echo "  • Monitoring: $ENABLE_MONITORING"
    echo
    echo -e "${CYAN}Quick Start Commands:${NC}"
    echo "  • Development: ./start-dev.sh"
    echo "  • Production: ./start-prod.sh"
    echo "  • API Docs: http://localhost:8000/docs"
    echo "  • Web Dashboard: http://localhost:3000"
    echo "  • Monitoring: http://localhost:3001"
    echo
    echo -e "${CYAN}CLI Commands:${NC}"
    echo "  • Weather forecast: poetry run openweather forecast 'London'"
    echo "  • Drone safety: poetry run openweather drone safety-check --lat 51.5 --lon -0.1"
    echo "  • Interactive agent: poetry run openweather analyst --interactive"
    echo "  • API server: poetry run openweather api"
    echo
    echo -e "${CYAN}Configuration:${NC}"
    echo "  • Environment file: .env"
    echo "  • Documentation: docs/"
    echo "  • Logs: data/logs/"
    echo
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "  1. Add your API keys to .env file"
    echo "  2. Start the development server: ./start-dev.sh"
    echo "  3. Visit http://localhost:8000/docs for API documentation"
    echo "  4. Read the documentation: docs/README.md"
    echo
    echo -e "${GREEN}Happy weather forecasting!${NC}"
}

# Main installation flow
main() {
    print_banner
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mode)
                INSTALL_MODE="$2"
                shift 2
                ;;
            --no-gpu)
                ENABLE_GPU="false"
                shift
                ;;
            --no-drone)
                ENABLE_DRONE="false"
                shift
                ;;
            --no-monitoring)
                ENABLE_MONITORING="false"
                shift
                ;;
            --skip-docker)
                SKIP_DOCKER="true"
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --mode MODE           Installation mode: minimal, standard, full, enterprise, drone"
                echo "  --no-gpu             Disable GPU support"
                echo "  --no-drone           Disable drone support"
                echo "  --no-monitoring      Disable monitoring"
                echo "  --skip-docker        Skip Docker installation"
                echo "  --help               Show this help message"
                exit 0
                ;;
            *)
                INSTALL_MODE="$1"
                shift
                ;;
        esac
    done
    
    # Validate installation mode
    case "$INSTALL_MODE" in
        minimal|standard|full|enterprise|drone) ;;
        *)
            error_exit "Invalid installation mode: $INSTALL_MODE. Use: minimal, standard, full, enterprise, or drone"
            ;;
    esac
    
    # Run installation steps
    check_system_requirements
    install_system_dependencies
    install_package_managers
    setup_project_environment
    post_installation_setup
    
    if verify_installation; then
        print_summary
    else
        error_exit "Installation failed verification. Please check the errors above."
    fi
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
