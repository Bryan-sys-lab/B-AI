# CodeAgent Local Development

This guide explains how to run the CodeAgent system locally without Docker for development purposes.

## Prerequisites

- **Python 3.11+** - Required for all services
- **Node.js 16+** (optional) - For UI development
- **Git** - For cloning repositories
- **curl** - For health checks

## Quick Start

1. **Clone and setup:**

   ```bash
   git clone <repository-url>
   cd codeagent
   ```

2. **Start all services:**

   ```bash
   ./start_local.sh
   ```

3. **Access the system:**

   - **Main API**: http://localhost:8000
   - **UI**: http://localhost:5173 (start separately if needed)
   - **WebSocket**: ws://localhost:8000/ws

4. **Stop all services:**
   ```bash
   ./stop_local.sh
   ```

## What the Scripts Do

### `start_local.sh`

- **Checks for existing virtual environment** (`codeagent_venv`) and reuses it
- **Skips requirement installation** if already installed (fast startup on subsequent runs)
- Installs all requirements from all services (only on first run)
- Starts all services in the background with reload enabled
- Performs health checks
- Displays service URLs

### `stop_local.sh`

- Stops all running services gracefully
- Force kills if necessary
- Cleans up PID files
- Deactivates virtual environment

## Service Architecture

The local setup runs these services:

### Core Services

- **Orchestrator** (8000) - Main API and task coordination
- **Sandbox Executor** (8002) - Isolated code execution
- **Tool API Gateway** (8001) - External tool integration
- **Comparator Service** (8003) - Code comparison and scoring

### Agents

- **Fix Implementation** (8004) - Code fixing and implementation
- **Debugger** (8005) - Test failure analysis
- **Review** (8006) - Code quality and security review
- **Testing** (8007) - Test generation and execution
- **Security** (8008) - Vulnerability scanning
- **Performance** (8009) - Profiling and optimization
- **Feedback** (8010) - Telemetry and self-improvement

### Supporting Services

- **Vector Store** (8011) - FAISS-based embeddings
- **Prompt Store** (8012) - Versioned prompt management
- **Transcript Store** (8013) - Immutable audit logs
- **Observability** (8014) - Metrics and monitoring

## Development Workflow

1. **Start services:**

   ```bash
   ./start_local.sh
   ```

2. **Make code changes** - Services run with `--reload` for hot reloading

3. **Test API endpoints:**

   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8000/about
   ```

4. **Submit a task:**

   ```bash
   curl -X POST http://localhost:8000/api/tasks \
     -H "Content-Type: application/json" \
     -d '{"description": "Fix the bug in fibonacci.py"}'
   ```

5. **Stop when done:**
   ```bash
   ./stop_local.sh
   ```

## Environment Variables

Create a `.env` file for API keys:

```bash
# Aetherium Provider API Keys
MISTRAL_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
NVIDIA_NIM_API_KEY=your_key_here

# Database (optional - uses SQLite by default)
DATABASE_URL=postgresql://user:pass@localhost:5432/codeagent

# MinIO (optional - uses local storage by default)
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
```

## Troubleshooting

### Services won't start

- Check Python version: `python3 --version`
- Ensure ports 8000-8014 are available
- Check logs in terminal output

### Import errors

- Run `./start_local.sh` again to reinstall requirements
- Check that all services have their requirements.txt files

### Services crash

- Check individual service logs
- Ensure all dependencies are installed
- Verify API keys in .env file

### Clean restart

```bash
./stop_local.sh
rm -rf codeagent_venv
rm -f .service_pids
./start_local.sh
```

## UI Development

To run the React UI separately:

```bash
cd ui
npm install
npm run dev
```

The UI will be available at http://localhost:5173 and connects to the WebSocket at ws://localhost:8000/ws.

## Performance Notes

- Local setup uses SQLite instead of PostgreSQL
- FAISS vector store runs in-memory
- Some services may have reduced functionality without external dependencies
- For full production features, use Docker Compose setup

## Contributing

When adding new services:

1. Create requirements.txt in your service directory
2. Add the service to `start_local.sh`
3. Update this README with the new port and description
