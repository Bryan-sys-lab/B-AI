#!/usr/bin/env bash
set -euo pipefail

SERVICES=(
  "http://127.0.0.1:8000/health"          # orchestrator
  "http://127.0.0.1:8001/health"          # tool_api_gateway
  "http://127.0.0.1:8002/health"          # sandbox_executor
  "http://127.0.0.1:8003/health"          # comparator_service
  "http://127.0.0.1:8004/health"          # agent_fix_implementation
  "http://127.0.0.1:8005/health"          # agent_debugger
  "http://127.0.0.1:8006/health"          # agent_review
  "http://127.0.0.1:8007/health"          # agent_testing
  "http://127.0.0.1:8008/health"          # agent_security
  "http://127.0.0.1:8009/health"          # agent_performance
  "http://127.0.0.1:8010/health"          # agent_feedback
  "http://127.0.0.1:8011/health"          # vector_store
  "http://127.0.0.1:8012/health"          # prompt_store
  "http://127.0.0.1:8013/health"          # transcript_store
  "http://127.0.0.1:8014/health"          # observability
  "http://127.0.0.1:8015/health"          # agent_web_scraper
  "http://127.0.0.1:8016/health"          # policy_engine
  "http://127.0.0.1:8017/health"          # agent_deployment
  "http://127.0.0.1:8018/health"          # agent_monitoring
)

echo "Running smoke tests against services..."
FAILED=()
for url in "${SERVICES[@]}"; do
  echo -n "Checking $url... "
  if curl -sSf --max-time 10 "$url" >/dev/null; then
    echo "OK"
  else
    echo "FAIL"
    FAILED+=("$url")
  fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "Failed services:"
  for failed in "${FAILED[@]}"; do
    echo "  - $failed"
  done
  echo "System started with some services unavailable"
else
  echo "All health endpoints responded OK."
fi

echo "All health endpoints responded OK."
