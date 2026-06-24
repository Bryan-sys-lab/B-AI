# B-AI: Autonomous Multi-Agent Development Platform

> **A self-learning, one-click AI development team that orchestrates specialized agents to automate your entire software development lifecycle.**

## 🚀 What is B-AI?

B-AI is an **agentic development platform** powered by the Aetherium framework that simulates a complete software engineering team. Instead of writing code yourself, you describe your task—and a coordinated fleet of autonomous AI agents spring into action.

**What makes it special:**
- 🤖 **Specialized Agents** – 10+ domain-specific agents handling code fixing, debugging, testing, security analysis, performance optimization, and more
- 🔌 **Multi-Model Support** – Seamlessly switch between Mistral, DeepSeek, OpenRouter, NVIDIA NIM, and local HuggingFace models
- 🏗️ **Orchestrated Workflows** – Intelligent task planning, routing, and execution across distributed microservices
- 🛡️ **Secure Sandboxing** – Isolated code execution environments with policy enforcement (OPA-backed)
- 📊 **Full Observability** – Built-in monitoring, tracing, and comprehensive logging via OpenTelemetry and Prometheus
- 📝 **One-Click Deployment** – Docker Compose setup that spins up a production-ready dev team in seconds

## 🏗️ Architecture

B-AI uses a **modular microservices architecture** orchestrated by a central FastAPI coordinator:

```
orchestrator (FastAPI)
  ├── Agents (specialized task handlers)
  │   ├── fix_implementation      (Code fixing & implementation)
  │   ├── debugger               (Debug assistance)
  │   ├── review                 (Code review & quality)
  │   ├── deployment             (Deployment automation)
  │   ├── testing                (Automated test generation)
  │   ├── security               (Security analysis)
  │   ├── performance            (Optimization)
  │   ├── monitoring             (System health)
  │   ├── comparator             (Candidate evaluation)
  │   └── feedback               (User feedback processing)
  │
  ├── Providers (LLM integrations)
  │   ├── Mistral               (Open-source models)
  │   ├── DeepSeek              (Reasoning models)
  │   ├── OpenRouter            (Multi-model router)
  │   ├── NVIDIA NIM            (Enterprise models)
  │   └── HuggingFace           (Local inference)
  │
  └── Supporting Services
      ├── sandbox_executor      (Secure code execution)
      ├── tool_api_gateway      (External tool integration)
      ├── vector_store          (Embeddings & RAG)
      ├── prompt_store          (Prompt management)
      ├── transcript_store      (Conversation logs)
      ├── observability         (Metrics & tracing)
      ├── policy_engine         (OPA governance)
      └── storage               (MinIO object store)
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | FastAPI (Python 3.10+) |
| **Databases** | PostgreSQL + Redis |
| **Object Storage** | MinIO |
| **Containerization** | Docker & Docker Compose |
| **Observability** | OpenTelemetry + Prometheus |
| **Frontend** | Next.js 14 + React + TypeScript + TailwindCSS |
| **State Management** | React Query + Zustand |
| **Real-time** | WebSockets/SSE |

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose installed
- Git

### 1. Clone & Setup

```bash
git clone https://github.com/Bryan-sys-lab/B-AI.git
cd B-AI
cd server
```

### 2. Configure Environment

```bash
cp .env .env.local
# Edit .env.local and add your LLM API keys
# - MISTRAL_API_KEY
# - DEEPSEEK_API_KEY
# - OPENROUTER_API_KEY
# - NVIDIA_NIM_API_KEY
# - HUGGINGFACE_API_KEY (optional, for local models)
```

### 3. Start the System

```bash
docker-compose up --build
```

Services will be available at:
- **Orchestrator API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MinIO Console**: http://localhost:9001 (user: `minioadmin` / password: `minioadmin`)
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

### 4. Submit a Task

```bash
curl -X POST http://localhost:8000/api/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Implement a REST API endpoint for user authentication",
    "language": "python",
    "requirements": ["OAuth2", "JWT tokens", "Rate limiting"]
  }'
```

## 📋 Core Features

### 🤖 Multi-Agent Orchestration
- **Task Planning**: Automatically breaks down complex requests into sub-tasks
- **Intelligent Routing**: Routes tasks to optimal agents based on type and complexity
- **Workflow Management**: Manages dependencies and parallel execution
- **Result Comparison**: Evaluates multiple candidates and selects the best solution

### 🔧 Flexible LLM Integration
- Switch models without code changes via environment configuration
- Support for both cloud APIs and local inference
- Model benchmarking and performance tracking
- Fallback chains for reliability

### 🏃 Code Execution & Safety
- **Sandboxed Execution**: Run generated code in isolated Docker containers
- **Policy Enforcement**: OPA-based policies for security guardrails
- **Output Validation**: Automatic validation and formatting
- **Error Handling**: Graceful recovery and detailed error reporting

### 🔍 Complete Observability
- **Distributed Tracing**: OpenTelemetry integration
- **Metrics**: Prometheus-compatible metrics
- **Health Checks**: Real-time system monitoring
- **Audit Logs**: Full action trail for compliance

## 📊 Use Cases

- **Rapid Prototyping** – Spin up working code in minutes, not hours
- **Bug Fixing & Refactoring** – Let agents analyze and improve your codebase
- **Code Review** – AI-powered quality assurance and best practices
- **Test Generation** – Automatic comprehensive test suite creation
- **Performance Optimization** – Identify bottlenecks and suggest improvements
- **Security Analysis** – Vulnerability detection and hardening
- **Deployment Automation** – End-to-end CI/CD orchestration

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests
pytest -m api          # API endpoint tests

# With coverage
pytest --cov=src --cov-report=html
```

The test suite includes:
- ✅ Orchestrator logic (planning, routing, execution)
- ✅ Sandbox executor (secure code execution)
- ✅ Comparator service (candidate evaluation)
- ✅ Tool API gateway (external integrations)
- ✅ Provider adapters (LLM integrations)
- ✅ Agent implementations
- ✅ Security & policy enforcement
- ✅ Integration workflows

## 📁 Project Structure

```
server/
├── orchestrator/          # Central coordination engine
├── agents/               # Specialized agent implementations
├── providers/            # LLM provider adapters
├── sandbox_executor/     # Secure code execution
├── tool_api_gateway/     # External tool integration
├── vector_store/         # Embeddings & RAG
├── prompt_store/         # Prompt templates
├── transcript_store/     # Conversation history
├── observability/        # Monitoring & tracing
├── policy_engine/        # Security policies
├── common/              # Shared utilities
├── tests/               # Comprehensive test suite
├── docker-compose.yml   # Service orchestration
└── cli.py              # CLI interface

frontend/
├── src/                # React components
├── package.json        # Dependencies
├── vite.config.js     # Build configuration
└── tailwind.config.js # Styling configuration
```

## 🔌 Key APIs

### Submit a Task
```bash
POST /api/tasks
{
  "description": "Your task description",
  "language": "python",
  "requirements": ["req1", "req2"]
}
```

### Get Task Status
```bash
GET /api/tasks/{task_id}
```

### List Active Tasks
```bash
GET /api/tasks?status=running&limit=10
```

### Check Agent Status
```bash
GET /api/agents
```

### View Metrics
```bash
GET /metrics (Prometheus format)
```

## 🔐 Security Features

- **Sandboxed Execution**: Code runs in isolated Docker containers
- **Policy Engine**: OPA-based authorization and governance
- **Input Validation**: All inputs sanitized and validated
- **Rate Limiting**: Built-in protection against abuse
- **Audit Logging**: Full traceability of all actions
- **Consent-Based Geolocation**: Privacy-first region detection

## 🚀 Deployment

### Docker Compose (Local/Development)
```bash
docker-compose up --build
```

### Render.com (Production)
See `render.yaml` for production configuration

### Custom Environments
Modify `docker-compose.yml` and environment variables for your setup

## 📚 Documentation

- **[Server Setup](./server/README.md)** – Detailed backend configuration
- **[Local Development](./server/README_LOCAL.md)** – Running locally without Docker
- **[System Architecture](./server/system_summary.md)** – Deep dive into components
- **[Progress & Roadmap](./server/progress.md)** – Current status and upcoming features

## 🤝 Contributing

Contributions are welcome! To add a new agent:

1. Create a new directory under `server/agents/{agent_name}`
2. Implement the agent interface (inherit from `BaseAgent`)
3. Add tests in `server/tests/test_agents.py`
4. Update `docker-compose.yml` to register the agent
5. Open a PR with your implementation

## 📝 License

This project is open source. See LICENSE for details.

## 🙋 Support & Community

- **Issues & Bugs**: [GitHub Issues](https://github.com/Bryan-sys-lab/B-AI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Bryan-sys-lab/B-AI/discussions)
- **Documentation**: [Full Docs](./server/README.md)

## 🔮 Roadmap

- [ ] Web UI dashboard for task monitoring
- [ ] Agent marketplace for community-contributed agents
- [ ] Advanced reasoning capabilities (Chain-of-Thought, Tree-of-Thought)
- [ ] Multi-language support (Java, Go, Rust, etc.)
- [ ] GitHub integration (auto-fix PRs, issue resolution)
- [ ] Kubernetes deployment templates
- [ ] Cost tracking & optimization

---

**Ready to transform your development workflow?** Start with Docker Compose and experience the future of AI-assisted development today! 🚀
