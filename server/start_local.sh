#!/bin/bash

# CodeAgent Local Startup Script
# Starts the entire system in a virtual environment on localhost

set -e

echo "Starting Aetherium System Locally..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
USE_RELOADER=false
for arg in "$@"; do
    case $arg in
        --reload)
            USE_RELOADER=true
            shift
            ;;
        *)
            # Unknown option
            ;;
    esac
done

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.11+ is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.11 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python $PYTHON_VERSION detected. Python $REQUIRED_VERSION or higher is required."
    exit 1
fi

print_success "Python $PYTHON_VERSION detected ✓"

# Increase file watch limit to prevent python -m uvicorn reload issues
print_status "Increasing file watch limit..."
if command -v sysctl &> /dev/null; then
    sudo sysctl -w fs.inotify.max_user_watches=524288 2>/dev/null || print_warning "Could not increase watch limit (may need sudo)"
else
    print_warning "sysctl not available, file watch limit may cause issues"
fi

# Check for existing virtual environment
VENV_DIR="codeagent_venv"
if [ -d "$VENV_DIR" ]; then
    print_warning "Virtual environment already exists, using existing one"
    print_status "Activating existing virtual environment..."
    source "$VENV_DIR/bin/activate"
    print_success "Virtual environment activated ✓"

    # Check if requirements are already installed
    if pip list | grep -q "fastapi"; then
        print_warning "Requirements appear to be already installed, skipping installation"
        REQUIREMENTS_INSTALLED=true
    else
        print_status "Requirements not found, will install..."
        REQUIREMENTS_INSTALLED=false
    fi
else
    print_status "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    print_success "Virtual environment created ✓"

    print_status "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    export PATH="$PWD/$VENV_DIR/bin:$PATH"
    print_success "Virtual environment activated ✓"

    REQUIREMENTS_INSTALLED=false
fi

# Upgrade pip if requirements not installed
if [ "$REQUIREMENTS_INSTALLED" = false ]; then
    print_status "Upgrading pip..."
    pip install --upgrade pip
    print_success "Pip upgraded ✓"

    # Install all requirements
    print_status "Installing requirements from all services..."

    # Core requirements
    pip install -r orchestrator/requirements.txt
    pip install -r sandbox_executor/requirements.txt
    pip install -r tool_api_gateway/requirements.txt
    pip install -r comparator_service/requirements.txt
    pip install -r workspace_service/requirements.txt

    # Agent requirements
    pip install -r agents/fix_implementation/requirements.txt
    pip install -r agents/debugger/requirements.txt
    pip install -r agents/review/requirements.txt
    pip install -r agents/testing/requirements.txt
    pip install -r agents/security/requirements.txt
    pip install -r agents/performance/requirements.txt
    pip install -r agents/feedback/requirements.txt
    pip install -r agents/web_scraper/requirements.txt
    pip install -r agents/deployment/requirements.txt
    pip install -r agents/monitoring/requirements.txt
    pip install -r agents/architecture/requirements.txt
    pip install -r agents/knowledge_agent/requirements.txt
    pip install -r agents/memory/requirements.txt

    # Service requirements
    pip install -r vector_store/requirements.txt
    pip install -r prompt_store/requirements.txt
    pip install -r transcript_store/requirements.txt
    pip install -r observability/requirements.txt
    pip install -r policy_engine/requirements.txt

    print_success "All requirements installed ✓"
else
    print_success "Using existing requirements ✓"
fi

# Set PYTHONPATH
export PYTHONPATH="$PWD:$PYTHONPATH"

# Stop any existing services
if [ -f stop_local.sh ]; then
    print_status "Stopping any existing services..."
    ./stop_local.sh
    print_success "Existing services stopped ✓"
fi

# Reactivate virtual environment after stop
if [ -d "$VENV_DIR" ]; then
    print_status "Reactivating virtual environment..."
    source "$VENV_DIR/bin/activate"
    export PATH="$PWD/$VENV_DIR/bin:$PATH"
    print_success "Virtual environment reactivated ✓"
fi

# Function to start a service
start_service() {
    local service_name=$1
    local command=$2
    local port=$3

    print_status "Starting $service_name on port $port..."

    # Start service in background
    eval "$command" &
    local pid=$!

    # Wait a bit for service to start
    sleep 5

    # Check if service is running
    if kill -0 $pid 2>/dev/null; then
        print_success "$service_name started (PID: $pid) ✓"
        echo $pid >> .service_pids
    else
        print_error "Failed to start $service_name"
        return 1
    fi
}

# Clean up any existing PID file
rm -f .service_pids

# Start dependencies
print_status "Starting dependencies..."

# Start Redis if available
if command -v redis-server &> /dev/null; then
    if ! pgrep -x "redis-server" > /dev/null; then
        print_status "Starting Redis..."
        redis-server --daemonize yes
        sleep 2
        print_success "Redis started ✓"
    else
        print_success "Redis already running ✓"
    fi
else
    print_warning "Redis not installed, some services may not work"
fi

# Start MinIO if available
if [ -f "./minio" ]; then
    if ! pgrep -x "minio" > /dev/null; then
        print_status "Starting MinIO..."
        mkdir -p /tmp/minio-data
        nohup ./minio server /tmp/minio-data --console-address ":9001" > minio.log 2>&1 &
        MINIO_PID=$!
        echo $MINIO_PID >> .service_pids
        sleep 5
        print_success "MinIO started (PID: $MINIO_PID) ✓"
    else
        print_success "MinIO already running ✓"
    fi
else
    print_warning "MinIO binary not found, downloading..."
    if command -v wget &> /dev/null; then
        wget -q https://dl.min.io/server/minio/release/linux-amd64/minio -O minio
        chmod +x minio
        print_status "Starting MinIO..."
        mkdir -p /tmp/minio-data
        nohup ./minio server /tmp/minio-data --console-address ":9001" > minio.log 2>&1 &
        MINIO_PID=$!
        echo $MINIO_PID >> .service_pids
        sleep 5
        print_success "MinIO started (PID: $MINIO_PID) ✓"
    else
        print_warning "wget not available, MinIO not started"
    fi
fi

# Start core services
print_status "Starting core services..."

# Start orchestrator (main service)
start_service "Orchestrator" "python -m uvicorn orchestrator.main:app --host 0.0.0.0 --port 8000 --reload-exclude '**/node_modules/**' --reload-exclude 'ui/**' --reload-exclude '**/__pycache__/**' --reload-exclude '**/*.log' --reload-exclude '**/*.db' --reload-exclude '**/*.rdb' --reload-exclude '**/*.pyc' --reload-exclude 'codeagent_venv/**' --reload-exclude '.pytest_cache/**' --reload-exclude '.ruff_cache/**' --reload-exclude 'codeagent.db'" 8000

# Start sandbox executor
start_service "Sandbox Executor" "python -m uvicorn sandbox_executor.executor:app --host 0.0.0.0 --port 8002 --reload-exclude '**/node_modules/**' --reload-exclude 'ui/**' --reload-exclude '**/__pycache__/**' --reload-exclude '**/*.log' --reload-exclude '**/*.db' --reload-exclude '**/*.rdb' --reload-exclude '**/*.pyc' --reload-exclude 'codeagent_venv/**' --reload-exclude '.pytest_cache/**' --reload-exclude '.ruff_cache/**' --reload-exclude 'codeagent.db'" 8002

# Start tool API gateway
start_service "Tool API Gateway" "python -m uvicorn tool_api_gateway.main:app --host 0.0.0.0 --port 8001 --reload-exclude '**/node_modules/**' --reload-exclude 'ui/**' --reload-exclude '**/__pycache__/**' --reload-exclude '**/*.log' --reload-exclude '**/*.db' --reload-exclude '**/*.rdb' --reload-exclude '**/*.pyc' --reload-exclude 'codeagent_venv/**' --reload-exclude '.pytest_cache/**' --reload-exclude '.ruff_cache/**' --reload-exclude 'codeagent.db'" 8001

# Start comparator service
start_service "Comparator Service" "python -m uvicorn comparator_service.main:app --host 0.0.0.0 --port 8003 --reload-exclude '**/node_modules/**' --reload-exclude 'ui/**' --reload-exclude '**/__pycache__/**' --reload-exclude '**/*.log' --reload-exclude '**/*.db' --reload-exclude '**/*.rdb' --reload-exclude '**/*.pyc' --reload-exclude 'codeagent_venv/**' --reload-exclude '.pytest_cache/**' --reload-exclude '.ruff_cache/**' --reload-exclude 'codeagent.db'" 8003

# Start key agents
print_status "Starting agent services..."

# Fix Implementation Agent
start_service "Fix Implementation Agent" "python -m uvicorn agents.fix_implementation.main:app --host 0.0.0.0 --port 8004 --reload-exclude '**/node_modules/**' --reload-exclude 'ui/**' --reload-exclude '**/__pycache__/**' --reload-exclude '**/*.log' --reload-exclude '**/*.db' --reload-exclude '**/*.rdb' --reload-exclude '**/*.pyc' --reload-exclude 'codeagent_venv/**' --reload-exclude '.pytest_cache/**' --reload-exclude '.ruff_cache/**' --reload-exclude 'codeagent.db'" 8004

# Debugger Agent
start_service "Debugger Agent" "python -m uvicorn agents.debugger.main:app --host 0.0.0.0 --port 8005 --reload-exclude '**/node_modules/**' --reload-exclude 'ui/**' --reload-exclude '**/__pycache__/**' --reload-exclude '**/*.log' --reload-exclude '**/*.db' --reload-exclude '**/*.rdb' --reload-exclude '**/*.pyc' --reload-exclude 'codeagent_venv/**' --reload-exclude '.pytest_cache/**' --reload-exclude '.ruff_cache/**' --reload-exclude 'codeagent.db'" 8005

# Review Agent
start_service "Review Agent" "python -m uvicorn agents.review.main:app --host 0.0.0.0 --port 8006 --reload-exclude '**/node_modules/**' --reload-exclude 'ui/**' --reload-exclude '**/__pycache__/**' --reload-exclude '**/*.log' --reload-exclude '**/*.db' --reload-exclude '**/*.rdb' --reload-exclude '**/*.pyc' --reload-exclude 'codeagent_venv/**' --reload-exclude '.pytest_cache/**' --reload-exclude '.ruff_cache/**' --reload-exclude 'codeagent.db'" 8006

# Testing Agent
start_service "Testing Agent" "python -m uvicorn agents.testing.main:app --host 0.0.0.0 --port 8007 --reload-exclude '**/node_modules/**' --reload-exclude 'ui/**' --reload-exclude '**/__pycache__/**' --reload-exclude '**/*.log' --reload-exclude '**/*.db' --reload-exclude '**/*.rdb' --reload-exclude '**/*.pyc' --reload-exclude 'codeagent_venv/**' --reload-exclude '.pytest_cache/**' --reload-exclude '.ruff_cache/**' --reload-exclude 'codeagent.db'" 8007

# Security Agent
start_service "Security Agent" "python -m uvicorn agents.security.main:app --host 0.0.0.0 --port 8008 --reload-exclude '**/node_modules/**' --reload-exclude 'ui/**' --reload-exclude '**/__pycache__/**' --reload-exclude '**/*.log' --reload-exclude '**/*.db' --reload-exclude '**/*.rdb' --reload-exclude '**/*.pyc' --reload-exclude 'codeagent_venv/**' --reload-exclude '.pytest_cache/**' --reload-exclude '.ruff_cache/**' --reload-exclude 'codeagent.db'" 8008

# Performance Agent
start_service "Performance Agent" "python -m uvicorn agents.performance.main:app --host 0.0.0.0 --port 8009 --reload-exclude '**/node_modules/**' --reload-exclude 'ui/**' --reload-exclude '**/__pycache__/**' --reload-exclude '**/*.log' --reload-exclude '**/*.db' --reload-exclude '**/*.rdb' --reload-exclude '**/*.pyc' --reload-exclude 'codeagent_venv/**' --reload-exclude '.pytest_cache/**' --reload-exclude '.ruff_cache/**' --reload-exclude 'codeagent.db'" 8009

# Feedback Agent
start_service "Feedback Agent" "python -m uvicorn agents.feedback.main:app --host 0.0.0.0 --port 8010 --reload-exclude '**/node_modules/**' --reload-exclude 'ui/**' --reload-exclude '**/__pycache__/**' --reload-exclude '**/*.log' --reload-exclude '**/*.db' --reload-exclude '**/*.rdb' --reload-exclude '**/*.pyc' --reload-exclude 'codeagent_venv/**' --reload-exclude '.pytest_cache/**' --reload-exclude '.ruff_cache/**' --reload-exclude 'codeagent.db'" 8010

# Task Classifier Agent
start_service "Task Classifier Agent" "python -m uvicorn agents.task_classifier.main:app --host 0.0.0.0 --port 8011 --reload-exclude '**/node_modules/**' --reload-exclude 'ui/**' --reload-exclude '**/__pycache__/**' --reload-exclude '**/*.log' --reload-exclude '**/*.db' --reload-exclude '**/*.rdb' --reload-exclude '**/*.pyc' --reload-exclude 'codeagent_venv/**' --reload-exclude '.pytest_cache/**' --reload-exclude '.ruff_cache/**' --reload-exclude 'codeagent.db'" 8011

# Comparator Service Agent
start_service "Comparator Service Agent" "python -m uvicorn comparator_service.main:app --host 0.0.0.0 --port 8012 --reload-exclude '**/node_modules/**' --reload-exclude 'ui/**' --reload-exclude '**/__pycache__/**' --reload-exclude '**/*.log' --reload-exclude '**/*.db' --reload-exclude '**/*.rdb' --reload-exclude '**/*.pyc' --reload-exclude 'codeagent_venv/**' --reload-exclude '.pytest_cache/**' --reload-exclude '.ruff_cache/**' --reload-exclude 'codeagent.db'" 8012

# Web Scraper Agent
start_service "Web Scraper Agent" "python -m uvicorn agents.web_scraper.main:app --host 0.0.0.0 --port 8015 --reload-exclude '**/node_modules/**' --reload-exclude 'ui/**' --reload-exclude '**/__pycache__/**' --reload-exclude '**/*.log' --reload-exclude '**/*.db' --reload-exclude '**/*.rdb' --reload-exclude '**/*.pyc' --reload-exclude 'codeagent_venv/**' --reload-exclude '.pytest_cache/**' --reload-exclude '.ruff_cache/**' --reload-exclude 'codeagent.db'" 8015

# Deployment Agent
start_service "Deployment Agent" "python -m uvicorn agents.deployment.main:app --host 0.0.0.0 --port 8017 --reload-exclude '**/node_modules/**' --reload-exclude 'ui/**' --reload-exclude '**/__pycache__/**' --reload-exclude '**/*.log' --reload-exclude '**/*.db' --reload-exclude '**/*.rdb' --reload-exclude '**/*.pyc' --reload-exclude 'codeagent_venv/**' --reload-exclude '.pytest_cache/**' --reload-exclude '.ruff_cache/**' --reload-exclude 'codeagent.db'" 8017

# Monitoring Agent
start_service "Monitoring Agent" "python -m uvicorn agents.monitoring.main:app --host 0.0.0.0 --port 8018 --reload-exclude '**/node_modules/**' --reload-exclude 'ui/**' --reload-exclude '**/__pycache__/**' --reload-exclude '**/*.log' --reload-exclude '**/*.db' --reload-exclude '**/*.rdb' --reload-exclude '**/*.pyc' --reload-exclude 'codeagent_venv/**' --reload-exclude '.pytest_cache/**' --reload-exclude '.ruff_cache/**' --reload-exclude 'codeagent.db'" 8018

# Architecture Agent
start_service "Architecture Agent" "python -m uvicorn agents.architecture.main:app --host 0.0.0.0 --port 8020 --reload-exclude '**/node_modules/**' --reload-exclude 'ui/**' --reload-exclude '**/__pycache__/**' --reload-exclude '**/*.log' --reload-exclude '**/*.db' --reload-exclude '**/*.rdb' --reload-exclude '**/*.pyc' --reload-exclude 'codeagent_venv/**' --reload-exclude '.pytest_cache/**' --reload-exclude '.ruff_cache/**' --reload-exclude 'codeagent.db'" 8020

# Knowledge Agent
start_service "Knowledge Agent" "python -m uvicorn agents.knowledge_agent.main:app --host 0.0.0.0 --port 8025 --reload-exclude '**/node_modules/**' --reload-exclude 'ui/**' --reload-exclude '**/__pycache__/**' --reload-exclude '**/*.log' --reload-exclude '**/*.db' --reload-exclude '**/*.rdb' --reload-exclude '**/*.pyc' --reload-exclude 'codeagent_venv/**' --reload-exclude '.pytest_cache/**' --reload-exclude '.ruff_cache/**' --reload-exclude 'codeagent.db'" 8025

# Memory Agent
start_service "Memory Agent" "python -m uvicorn agents.memory.main:app --host 0.0.0.0 --port 8026 --reload-exclude '**/node_modules/**' --reload-exclude 'ui/**' --reload-exclude '**/__pycache__/**' --reload-exclude '**/*.log' --reload-exclude '**/*.db' --reload-exclude '**/*.rdb' --reload-exclude '**/*.pyc' --reload-exclude 'codeagent_venv/**' --reload-exclude '.pytest_cache/**' --reload-exclude '.ruff_cache/**' --reload-exclude 'codeagent.db'" 8026

# Start supporting services
print_status "Starting supporting services..."

# Vector Store
start_service "Vector Store" "python -m uvicorn vector_store.main:app --host 0.0.0.0 --port 8019 --reload-exclude '**/node_modules/**' --reload-exclude 'ui/**' --reload-exclude '**/__pycache__/**' --reload-exclude '**/*.log' --reload-exclude '**/*.db' --reload-exclude '**/*.rdb' --reload-exclude '**/*.pyc' --reload-exclude 'codeagent_venv/**' --reload-exclude '.pytest_cache/**' --reload-exclude '.ruff_cache/**' --reload-exclude 'codeagent.db'" 8019

# Enhanced Workspace Service
start_service "Enhanced Workspace Service" "python -m uvicorn workspace_service.main:app --host 0.0.0.0 --port 8024 --reload-exclude '**/node_modules/**' --reload-exclude 'ui/**' --reload-exclude '**/__pycache__/**' --reload-exclude '**/*.log' --reload-exclude '**/*.db' --reload-exclude '**/*.rdb' --reload-exclude '**/*.pyc' --reload-exclude 'codeagent_venv/**' --reload-exclude '.pytest_cache/**' --reload-exclude '.ruff_cache/**' --reload-exclude 'codeagent.db'" 8024

# Prompt Store
start_service "Prompt Store" "python -m uvicorn prompt_store.main:app --host 0.0.0.0 --port 8021 --reload-exclude '**/node_modules/**' --reload-exclude 'ui/**' --reload-exclude '**/__pycache__/**' --reload-exclude '**/*.log' --reload-exclude '**/*.db' --reload-exclude '**/*.rdb' --reload-exclude '**/*.pyc' --reload-exclude 'codeagent_venv/**' --reload-exclude '.pytest_cache/**' --reload-exclude '.ruff_cache/**' --reload-exclude 'codeagent.db'" 8021

# Transcript Store
start_service "Transcript Store" "python -m uvicorn transcript_store.main:app --host 0.0.0.0 --port 8022 --reload-exclude '**/node_modules/**' --reload-exclude 'ui/**' --reload-exclude '**/__pycache__/**' --reload-exclude '**/*.log' --reload-exclude '**/*.db' --reload-exclude '**/*.rdb' --reload-exclude '**/*.pyc' --reload-exclude 'codeagent_venv/**' --reload-exclude '.pytest_cache/**' --reload-exclude '.ruff_cache/**' --reload-exclude 'codeagent.db'" 8022

# Observability
start_service "Observability" "python -m uvicorn observability.main:app --host 0.0.0.0 --port 8023 --reload-exclude '**/node_modules/**' --reload-exclude 'ui/**' --reload-exclude '**/__pycache__/**' --reload-exclude '**/*.log' --reload-exclude '**/*.db' --reload-exclude '**/*.rdb' --reload-exclude '**/*.pyc' --reload-exclude 'codeagent_venv/**' --reload-exclude '.pytest_cache/**' --reload-exclude '.ruff_cache/**' --reload-exclude 'codeagent.db'" 8023

# Policy Engine
start_service "Policy Engine" "python -m uvicorn policy_engine.main:app --host 0.0.0.0 --port 8016 --reload-exclude '**/node_modules/**' --reload-exclude 'ui/**' --reload-exclude '**/__pycache__/**' --reload-exclude '**/*.log' --reload-exclude '**/*.db' --reload-exclude '**/*.rdb' --reload-exclude '**/*.pyc' --reload-exclude 'codeagent_venv/**' --reload-exclude '.pytest_cache/**' --reload-exclude '.ruff_cache/**' --reload-exclude 'codeagent.db'" 8016

print_success "All services started! 🎉"

# Display service URLs
echo ""
echo "🌐 Service URLs:"
echo "  Orchestrator (Main API):    http://localhost:8000"
echo "  Tool API Gateway:           http://localhost:8001"
echo "  Sandbox Executor:           http://localhost:8002"
echo "  Comparator Service:         http://localhost:8003"
echo "  Fix Implementation Agent:   http://localhost:8004"
echo "  Debugger Agent:             http://localhost:8005"
echo "  Review Agent:               http://localhost:8006"
echo "  Testing Agent:              http://localhost:8007"
echo "  Security Agent:             http://localhost:8008"
echo "  Performance Agent:          http://localhost:8009"
echo "  Feedback Agent:             http://localhost:8010"
echo "  Task Classifier Agent:      http://localhost:8011"
echo "  Web Scraper Agent:          http://localhost:8015"
echo "  Policy Engine:              http://localhost:8016"
echo "  Deployment Agent:           http://localhost:8017"
echo "  Monitoring Agent:           http://localhost:8018"
echo "  Vector Store:               http://localhost:8019"
echo "  Architecture Agent:         http://localhost:8020"
echo "  Knowledge Agent:            http://localhost:8025"
echo "  Memory Agent:               http://localhost:8026"
echo "  Prompt Store:               http://localhost:8021"
echo "  Transcript Store:            http://localhost:8022"
echo "  Observability:              http://localhost:8023"
echo "  Enhanced Workspace Service: http://localhost:8024"
echo ""
echo "🎨 UI (if started separately): http://localhost:5173"
echo ""
echo "💻 CLI available: python cli.py"
echo ""

# Health check
print_status "Running health checks..."
sleep 10

SERVICES=(
    "http://localhost:8000/health"          # orchestrator
    "http://localhost:8001/health"          # tool_api_gateway
    "http://localhost:8002/health"          # sandbox_executor
    "http://localhost:8003/health"          # comparator_service
    "http://localhost:8004/health"          # agent_fix_implementation
    "http://localhost:8005/health"          # agent_debugger
    "http://localhost:8006/health"          # agent_review
    "http://localhost:8007/health"          # agent_testing
    "http://localhost:8008/health"          # agent_security
    "http://localhost:8009/health"          # agent_performance
    "http://localhost:8010/health"          # agent_feedback
    "http://localhost:8011/health"          # agent_task_classifier
    "http://localhost:8015/health"          # agent_web_scraper
    "http://localhost:8016/health"          # policy_engine
    "http://localhost:8017/health"          # agent_deployment
    "http://localhost:8018/health"          # agent_monitoring
    "http://localhost:8019/health"          # vector_store
    "http://localhost:8020/health"          # agent_architecture
    "http://localhost:8025/health"          # agent_knowledge
    "http://localhost:8026/health"          # agent_memory
    "http://localhost:8021/health"          # prompt_store
    "http://localhost:8022/health"          # transcript_store
    "http://localhost:8023/health"          # observability
    "http://localhost:8024/health"          # workspace_service
)

all_healthy=true
for url in "${SERVICES[@]}"; do
    service_name=$(echo "$url" | sed 's|http://localhost:[0-9]*/||' | sed 's|/health||')
    if curl -s "$url" > /dev/null 2>&1; then
        print_success "$service_name health check passed ✓"
    else
        print_error "$service_name health check failed"
        all_healthy=false
    fi
done

if [ "$all_healthy" = false ]; then
    print_warning "Some services failed health checks - system may not be fully started"
else
    print_success "All services health checks passed ✓"
fi

if [ "$USE_RELOADER" = true ]; then
    print_status "Starting system reloader for hot reloading..."
    python scripts/reloader.py
else
    echo ""
    print_success "CodeAgent system is running locally! 🚀"
    echo ""
    echo "To stop all services, run: ./stop_local.sh"
    echo "Or manually kill processes with: kill \$(cat .service_pids)"
    echo ""
    echo "💻 Starting CLI..."
    python cli.py interactive
    echo ""
    echo "Press Ctrl+C to exit this script (services will continue running)"

    # Wait for user interrupt
    trap 'echo ""; print_status "Script terminated. Services are still running in background."; print_status "To stop services: ./stop_local.sh"' INT
    wait
fi