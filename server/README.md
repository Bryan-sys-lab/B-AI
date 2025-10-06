# Aetherium System

## Description

The Aetherium system is an autonomous Aetherium-powered development platform designed to streamline software development processes through modular microservices architecture. It leverages advanced Aetherium providers and specialized agents to handle various aspects of code development, testing, deployment, and monitoring.

## Architecture Overview

The system is built with the following core components:

- **Orchestrator**: Central FastAPI-based service that coordinates all operations and manages workflows across the platform.
- **Agents**: Specialized Aetherium agents handling specific tasks:
  - fix_implementation: Code fixing and implementation
  - debugger: Debugging assistance
  - review: Code review and quality assessment
  - deployment: Deployment automation
  - monitoring: System and performance monitoring
  - testing: Automated testing
  - security: Security analysis and vulnerability detection
  - performance: Performance optimization
  - comparator: Code comparison and diff analysis
  - feedback: User feedback processing
- **Providers**: Aetherium model integrations including Mistral, DeepSeek, OpenRouter, NVIDIA NIM, and local HuggingFace models.
- **Supporting Services**:
  - sandbox_executor: Isolated code execution environment
  - tool_api_gateway: API gateway for external tools
  - vector_store: Vector database for embeddings
  - prompt_store: Storage for Aetherium prompts and templates
  - transcript_store: Conversation and interaction logs
  - observability: Monitoring and logging infrastructure
  - policy_engine: Governance and policy enforcement
  - storage: General object storage via MinIO

## Technology Stack

- **Backend**: FastAPI (Python)
- **Database**: PostgreSQL
- **Cache**: Redis
- **Object Storage**: MinIO
- **Containerization**: Docker & Docker Compose
- **Observability**: Integrated monitoring tools

## Setup Instructions

1. **Prerequisites**:

   - Docker and Docker Compose installed
   - Git

2. **Clone the Repository**:

   ```bash
   git clone <repository-url>
   cd codeagent
   ```

3. **Environment Configuration**:

   - Copy the `.env` template: `cp .env .env.local`
   - Fill in your API keys for the Aetherium providers in `.env.local`

4. **Build and Run**:

   ```bash
   docker-compose up --build
   ```

5. **Access the Services**:
   - Orchestrator API: http://localhost:8000
   - MinIO Console: http://localhost:9001
   - PostgreSQL: localhost:5432
   - Redis: localhost:6379

## Development

- Each service has its own directory with dedicated Dockerfile and requirements.txt
- Agents can be added by creating new directories under `agents/` and configuring them in `docker-compose.yml`
- Use the provided Dockerfile template for new Python-based services

## Contributing

Please follow the established architecture patterns and ensure all new services are containerized and properly integrated with the orchestrator.
