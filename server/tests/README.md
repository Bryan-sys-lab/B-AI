# Test Suite Documentation

This directory contains comprehensive tests for the Aetherium Code Agent system.

## Test Structure

### Unit Tests

- `test_orchestrator.py` - Tests for orchestrator components (Planner, Router, MasterAgent, Workflow)
- `test_sandbox_executor.py` - Tests for sandbox executor (execution, utilities)
- `test_comparator_service.py` - Tests for comparator service (scoring, parallel evaluation)
- `test_tool_api_gateway.py` - Tests for tool API gateway (models, security, endpoints)
- `test_providers.py` - Tests for provider adapters (BaseAdapter, Mistral, DeepSeek, etc.)
- `test_agents.py` - Tests for agent implementations (fix_implementation as example)
- `test_observability_storage.py` - Tests for observability and storage components

### Integration Tests

- `test_integration.py` - Tests for service-to-service communication and workflows

### Existing Tests

- `test_about_tool_api_gateway.py` - Basic API endpoint tests
- `test_about_endpoint.py` - About endpoint tests
- `test_about_sandbox_executor.py` - Sandbox executor about tests
- `test_about_comparator_service.py` - Comparator service about tests
- `test_imports.py` - Import validation tests
- `test_model_registry.py` - Model registry tests

## Test Configuration

### pytest.ini

Configuration file with:

- Test discovery patterns
- Markers for different test types
- Warning filters
- Async test support

### conftest.py

Shared fixtures including:

- Mock adapters and HTTP clients
- Sample data fixtures
- Database and storage mocks
- Environment variable mocks
- Test utilities

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# API tests only
pytest -m api

# Slow tests only
pytest -m slow
```

### Run Specific Test Files

```bash
pytest tests/test_orchestrator.py
pytest tests/test_integration.py -v
```

### Run Tests with Coverage

```bash
pytest --cov=src --cov-report=html
```

## Test Coverage

The test suite covers:

### Core Components

- ✅ Orchestrator (task planning, routing, execution)
- ✅ Sandbox Executor (secure code execution)
- ✅ Comparator Service (candidate evaluation)
- ✅ Tool API Gateway (external tool integration)
- ✅ Provider Adapters (LLM integrations)
- ✅ Agent Implementations (code improvement agents)

### Supporting Systems

- ✅ Observability (metrics, tracing, health checks)
- ✅ Storage (MinIO, databases, transcripts)
- ✅ Security (command validation, OPA policies)
- ✅ Configuration (environment, Docker, monitoring)

### Test Types

- ✅ Unit tests (isolated component testing)
- ✅ Integration tests (service interaction testing)
- ✅ API endpoint tests
- ✅ Error handling tests
- ✅ Mock-based testing for external dependencies

## Mock Strategy

Tests use comprehensive mocking to:

- Avoid external API calls during testing
- Test error conditions and edge cases
- Ensure fast test execution
- Provide deterministic test results

Key mocked components:

- HTTP clients (requests, httpx)
- Database sessions
- File system operations
- External services (Docker, MinIO, Redis)
- LLM API calls
- OpenTelemetry tracing
- Prometheus metrics

## Test Data

Sample test data includes:

- Mock tasks and subtasks
- Sample code candidates
- API request/response fixtures
- Error conditions and edge cases

## Continuous Integration

Tests are designed to run in CI/CD pipelines with:

- Parallel test execution
- Proper cleanup of temporary resources
- Environment isolation
- Comprehensive reporting

## Adding New Tests

When adding new tests:

1. Follow the existing naming convention: `test_*.py`
2. Use appropriate markers (`@pytest.mark.unit`, `@pytest.mark.integration`, etc.)
3. Leverage shared fixtures from `conftest.py`
4. Mock external dependencies
5. Include both success and failure scenarios
6. Add docstrings explaining test purpose

## Test Maintenance

Regular maintenance tasks:

- Update mocks when APIs change
- Add tests for new features
- Remove obsolete test cases
- Update test data as needed
- Monitor test execution time and optimize slow tests
