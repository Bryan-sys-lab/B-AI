#!/bin/bash

# CodeAgent Local Stop Script
# Stops all running services

set -e

echo "🛑 Stopping CodeAgent System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if PID file exists
if [ -f .service_pids ]; then
    print_status "Reading service PIDs..."

    # Read PIDs and stop services
    while IFS= read -r pid; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            print_status "Stopping process $pid..."
            kill "$pid" 2>/dev/null || true

            # Wait for process to stop
            for i in {1..10}; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    print_success "Process $pid stopped ✓"
                    break
                fi
                sleep 0.5
            done

            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                print_warning "Force killing process $pid..."
                kill -9 "$pid" 2>/dev/null || true
                sleep 1
                if ! kill -0 "$pid" 2>/dev/null; then
                    print_success "Process $pid force killed ✓"
                else
                    print_error "Failed to kill process $pid"
                fi
            fi
        else
            print_warning "Process $pid not running or invalid"
        fi
    done < .service_pids

    # Clean up PID file
    rm -f .service_pids
else
    print_warning "No .service_pids file found."
fi

# Stop Redis if running
if command -v redis-cli &> /dev/null; then
    if pgrep -x "redis-server" > /dev/null; then
        print_status "Stopping Redis..."
        redis-cli shutdown 2>/dev/null || true
        sleep 2
        if ! pgrep -x "redis-server" > /dev/null; then
            print_success "Redis stopped ✓"
        else
            print_warning "Redis may still be running"
        fi
    fi
fi

# Stop MinIO if running
if pgrep -x "minio" > /dev/null; then
    print_status "Stopping MinIO..."
    pkill -x minio 2>/dev/null || true
    sleep 2
    if ! pgrep -x "minio" > /dev/null; then
        print_success "MinIO stopped ✓"
    else
        print_warning "MinIO may still be running"
    fi
fi

# Kill any remaining uvicorn processes
print_status "Checking for any remaining uvicorn processes..."
REMAINING_PIDS=$(pgrep -f "uvicorn" || true)

if [ -n "$REMAINING_PIDS" ]; then
    print_warning "Found remaining CodeAgent processes: $REMAINING_PIDS"
    echo "Killing remaining processes..."
    echo "$REMAINING_PIDS" | xargs kill -9 2>/dev/null || true
    print_success "Remaining processes killed ✓"
else
    print_success "No remaining CodeAgent processes found ✓"
fi

# Deactivate virtual environment if active
if [ -n "$VIRTUAL_ENV" ]; then
    print_status "Deactivating virtual environment..."
    deactivate 2>/dev/null || true
    print_success "Virtual environment deactivated ✓"
fi

print_success "All CodeAgent services stopped! 🛑"

echo ""
echo "To restart services: ./start_local.sh"
echo "To clean virtual environment: rm -rf codeagent_venv"