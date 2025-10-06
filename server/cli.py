#!/usr/bin/env python3
"""
CodeAgent CLI - Command Line Interface for the CodeAgent system
"""

import typer
import requests
import json
from typing import Optional
import asyncio
import websockets
import threading
import time

async def monitor_task_websocket(task_id: str, timeout: int, verbose: bool):
    """Monitor task progress using WebSocket for real-time updates"""
    import time
    start_time = time.time()

    try:
        async with websockets.connect("ws://localhost:8000/ws") as websocket:
            # Send initial task message to establish connection
            await websocket.send(json.dumps({"task": f"monitor_{task_id}", "task_id": task_id}))

            while time.time() - start_time < timeout:
                try:
                    # Set a short timeout for receiving messages
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)

                    if verbose:
                        typer.echo(f"WebSocket: {data}")

                    if data.get("type") == "status":
                        status = data.get("status", "unknown")
                        progress = data.get("progress", 0)

                        if status == "completed":
                            total_time = time.time() - start_time
                            typer.echo(f"✅ Task completed successfully in {total_time:.2f}s!")
                            # Get final task details
                            task_response = requests.get(f"http://localhost:8000/api/tasks/{task_id}", timeout=5)
                            if task_response.status_code == 200:
                                task_data = task_response.json()
                                if task_data.get('output'):
                                    typer.echo("\n📄 Task Output:")
                                    typer.echo(json.dumps(task_data['output'], indent=2))
                                if task_data.get('subtasks'):
                                    typer.echo("\n🔧 Subtasks:")
                                    for subtask in task_data['subtasks']:
                                        typer.echo(f"  - {subtask['agent']}: {subtask['description']} ({subtask['status']})")
                                        if subtask.get('output'):
                                            typer.echo(f"    Output: {json.dumps(subtask['output'], indent=4)}")
                            break
                        elif status == "failed":
                            total_time = time.time() - start_time
                            typer.echo(f"❌ Task failed after {total_time:.2f}s!")
                            break
                        else:
                            typer.echo(f"⏳ Task status: {status} ({progress}%)")

                    elif data.get("type") == "output":
                        if verbose:
                            typer.echo("📄 Received task output update")

                except asyncio.TimeoutError:
                    # No message received, continue waiting
                    continue
                except websockets.exceptions.ConnectionClosed:
                    typer.echo("WebSocket connection closed")
                    break

            else:
                total_time = time.time() - start_time
                typer.echo(f"Timeout reached after {total_time:.2f}s. Task may still be running.")
                typer.echo(f"Check task status manually: python cli.py get-task {task_id}")

    except Exception as e:
        typer.echo(f"WebSocket monitoring failed: {e}")
        typer.echo("Falling back to polling mode...")
        # Fallback to polling if WebSocket fails
        monitor_task_polling(task_id, timeout, verbose)

def monitor_task_polling(task_id: str, timeout: int, verbose: bool):
    """Fallback polling method for task monitoring"""
    import time
    start_time = time.time()
    poll_count = 0

    while time.time() - start_time < timeout:
        poll_start = time.time()
        try:
            task_response = requests.get(f"http://localhost:8000/api/tasks/{task_id}", timeout=None)
            poll_count += 1
            poll_time = time.time() - poll_start

            if task_response.status_code == 200:
                task_data = task_response.json()
                status = task_data.get('status')

                if verbose:
                    typer.echo(f"Task status: {status} (poll #{poll_count} took {poll_time:.3f}s)")

                if status == 'completed':
                    total_time = time.time() - start_time
                    typer.echo(f"✅ Task completed successfully in {total_time:.2f}s!")
                    if task_data.get('output'):
                        typer.echo("\n📄 Task Output:")
                        typer.echo(json.dumps(task_data['output'], indent=2))
                    if task_data.get('subtasks'):
                        typer.echo("\n🔧 Subtasks:")
                        for subtask in task_data['subtasks']:
                            typer.echo(f"  - {subtask['agent']}: {subtask['description']} ({subtask['status']})")
                            if subtask.get('output'):
                                typer.echo(f"    Output: {json.dumps(subtask['output'], indent=4)}")
                    break
                elif status == 'failed':
                    total_time = time.time() - start_time
                    typer.echo(f"❌ Task failed after {total_time:.2f}s!")
                    if task_data.get('output'):
                        typer.echo(f"Error: {task_data['output']}")
                    break
                else:
                    typer.echo(f"⏳ Task status: {status}")
            else:
                typer.echo(f"Error checking task status: HTTP {task_response.status_code}")
                break
        except requests.exceptions.RequestException as e:
            typer.echo(f"Error checking task status: {e}")
            break

        time.sleep(0.5)

    else:
        total_time = time.time() - start_time
        typer.echo(f"Timeout reached after {total_time:.2f}s. Task may still be running.")
        typer.echo(f"Check task status manually: python cli.py get-task {task_id}")

app = typer.Typer(help="CodeAgent CLI - Interact with the CodeAgent system")

@app.command()
def run_task(
    task: str = typer.Argument(..., help="Task description to execute"),
    wait: bool = typer.Option(True, help="Wait for task completion"),
    timeout: int = typer.Option(300, help="Timeout in seconds"),
    verbose: bool = typer.Option(False, help="Enable verbose output"),
    websocket: bool = typer.Option(False, help="Use WebSocket for real-time updates (faster)")
):
    """Run a task using the CodeAgent system"""
    try:
        payload = {
            "description": task
        }

        if verbose:
            typer.echo(f"Sending request to orchestrator: {payload}")

        response = requests.post("http://localhost:8000/api/tasks", json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            task_id = result.get('task_id')
            typer.echo(f"Task created with ID: {task_id}")

            if wait:
                if websocket:
                    typer.echo("Monitoring task with WebSocket (real-time updates)...")
                    asyncio.run(monitor_task_websocket(task_id, timeout, verbose))
                else:
                    typer.echo("Waiting for task completion (polling every 0.5s)...")
                    import time
                    start_time = time.time()
                    poll_count = 0

                    while time.time() - start_time < timeout:
                        poll_start = time.time()
                        try:
                            task_response = requests.get(f"http://localhost:8000/api/tasks/{task_id}", timeout=None)
                            poll_count += 1
                            poll_time = time.time() - poll_start

                            if task_response.status_code == 200:
                                task_data = task_response.json()
                                status = task_data.get('status')

                                if verbose:
                                    typer.echo(f"Task status: {status} (poll #{poll_count} took {poll_time:.3f}s)")

                                if status == 'completed':
                                    total_time = time.time() - start_time
                                    typer.echo(f"✅ Task completed successfully in {total_time:.2f}s!")
                                    if task_data.get('output'):
                                        typer.echo("\n📄 Task Output:")
                                        typer.echo(json.dumps(task_data['output'], indent=2))
                                    if task_data.get('subtasks'):
                                        typer.echo("\n🔧 Subtasks:")
                                        for subtask in task_data['subtasks']:
                                            typer.echo(f"  - {subtask['agent']}: {subtask['description']} ({subtask['status']})")
                                            if subtask.get('output'):
                                                typer.echo(f"    Output: {json.dumps(subtask['output'], indent=4)}")
                                    break
                                elif status == 'failed':
                                    total_time = time.time() - start_time
                                    typer.echo(f"❌ Task failed after {total_time:.2f}s!")
                                    if task_data.get('output'):
                                        typer.echo(f"Error: {task_data['output']}")
                                    break
                                else:
                                    typer.echo(f"⏳ Task status: {status}")
                            else:
                                typer.echo(f"Error checking task status: HTTP {task_response.status_code}")
                                break
                        except requests.exceptions.RequestException as e:
                            typer.echo(f"Error checking task status: {e}")
                            break

                        time.sleep(0.5)  # Wait 0.5 seconds before checking again for faster response

                    else:
                        total_time = time.time() - start_time
                        typer.echo(f"Timeout reached after {total_time:.2f}s ({timeout} seconds). Task may still be running.")
                        typer.echo(f"Check task status manually: python cli.py get-task {task_id}")
            else:
                typer.echo("Task is being processed in the background.")
                typer.echo(f"Check status: python cli.py get-task {task_id}")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
            typer.echo(response.text)

    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)
        typer.echo("Make sure the CodeAgent system is running (./start_local.sh)", err=True)

@app.command()
def health():
    """Check system health"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            typer.echo("✅ System is healthy")
        else:
            typer.echo(f"❌ System health check failed: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        typer.echo(f"❌ Connection error: {e}", err=True)

@app.command()
def list_agents():
    """List available agents"""
    try:
        response = requests.get("http://localhost:8000/api/agents", timeout=10)
        if response.status_code == 200:
            agents = response.json()
            typer.echo("Available agents:")
            for agent in agents:
                typer.echo(f"  - {agent['name']} ({agent['type']}) - {agent['description']}")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

@app.command()
def status():
    """Get system status"""
    try:
        # Check orchestrator
        orch_response = requests.get("http://localhost:8000/health", timeout=5)
        orch_status = "✅" if orch_response.status_code == 200 else "❌"

        # Check agents
        agents_response = requests.get("http://localhost:8000/api/agents/status", timeout=5)
        agents_status = "✅" if agents_response.status_code == 200 else "❌"

        typer.echo("System Status:")
        typer.echo(f"  Orchestrator: {orch_status}")
        typer.echo(f"  Agents: {agents_status}")

        if agents_response.status_code == 200:
            agents = agents_response.json()
            typer.echo("Agent Details:")
            for agent in agents:
                status_icon = "✅" if agent['health'] == 'healthy' else "❌"
                typer.echo(f"    {status_icon} {agent['name']} - {agent['status']} ({agent['health']})")

    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

@app.command()
def activate_agents():
    """Activate all idle agents"""
    try:
        # Get all agents
        response = requests.get("http://localhost:8000/api/agents", timeout=10)
        if response.status_code == 200:
            agents = response.json()
            activated = 0
            for agent in agents:
                if agent['status'] == 'idle':
                    # Activate the agent
                    control_response = requests.post(f"http://localhost:8000/api/agents/{agent['id']}/control?action=start", timeout=10)
                    if control_response.status_code == 200:
                        typer.echo(f"✅ Activated agent: {agent['name']}")
                        activated += 1
                    else:
                        typer.echo(f"❌ Failed to activate agent: {agent['name']}")
            if activated == 0:
                typer.echo("All agents are already active.")
            else:
                typer.echo(f"Activated {activated} agents.")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

# Task Management Commands
@app.command()
def list_tasks(
    project_id: Optional[str] = typer.Option(None, help="Filter by project ID"),
    status: Optional[str] = typer.Option(None, help="Filter by status")
):
    """List tasks"""
    try:
        params = {}
        if project_id:
            params['project_id'] = project_id
        if status:
            params['status'] = status

        response = requests.get("http://localhost:8000/api/tasks", params=params, timeout=10)
        if response.status_code == 200:
            tasks = response.json()
            if not tasks:
                typer.echo("No tasks found.")
                return

            typer.echo("Tasks:")
            for task in tasks:
                typer.echo(f"  ID: {task['id']}")
                typer.echo(f"  Description: {task['description']}")
                typer.echo(f"  Status: {task['status']}")
                if task.get('project_id'):
                    typer.echo(f"  Project: {task['project_id']}")
                typer.echo(f"  Created: {task['created_at']}")
                typer.echo("")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

@app.command()
def get_task(task_id: str):
    """Get task details"""
    try:
        response = requests.get(f"http://localhost:8000/api/tasks/{task_id}", timeout=10)
        if response.status_code == 200:
            task = response.json()
            typer.echo("Task Details:")
            typer.echo(f"  ID: {task['id']}")
            typer.echo(f"  Description: {task['description']}")
            typer.echo(f"  Status: {task['status']}")
            if task.get('plan'):
                typer.echo(f"  Plan: {json.dumps(task['plan'], indent=2)}")
            if task.get('output'):
                typer.echo(f"  Output: {json.dumps(task['output'], indent=2)}")
            if task.get('subtasks'):
                typer.echo("  Subtasks:")
                for subtask in task['subtasks']:
                    typer.echo(f"    - {subtask['agent']}: {subtask['description']} ({subtask['status']})")
            typer.echo(f"  Created: {task['created_at']}")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

# Provider Management Commands
@app.command()
def list_providers():
    """List Aetherium providers"""
    try:
        response = requests.get("http://localhost:8000/api/providers", timeout=10)
        if response.status_code == 200:
            providers = response.json()
            typer.echo("Aetherium Providers:")
            for provider in providers:
                typer.echo(f"  - {provider['name']} ({provider['type']})")
                typer.echo(f"    Purpose: {provider['purpose']}")
                typer.echo(f"    Status: {provider['status']}")
                typer.echo(f"    Models: {', '.join(provider['models'])}")
                typer.echo("")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

@app.command()
def provider_metrics():
    """Show provider metrics"""
    try:
        response = requests.get("http://localhost:8000/api/providers/metrics", timeout=10)
        if response.status_code == 200:
            metrics = response.json()
            typer.echo("Provider Metrics:")
            for metric in metrics:
                typer.echo(f"  {metric['provider_id']}:")
                typer.echo(f"    Latency: {metric['latency']}ms")
                typer.echo(f"    Success Rate: {metric['success_rate']}%")
                typer.echo(f"    Total Requests: {metric['total_requests']}")
                typer.echo(f"    Active Requests: {metric['active_requests']}")
                typer.echo(f"    Cost Estimate: ${metric['cost_estimate']}")
                typer.echo(f"    Tokens Used: {metric['tokens_used']}")
                if metric.get('last_used'):
                    typer.echo(f"    Last Used: {metric['last_used']}")
                typer.echo("")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

@app.command()
def switch_provider(provider_id: str):
    """Switch active provider"""
    try:
        response = requests.post(f"http://localhost:8000/api/providers/switch/{provider_id}", timeout=10)
        if response.status_code == 200:
            result = response.json()
            typer.echo(f"✅ Switched to provider: {result.get('message', 'Success')}")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
            typer.echo(response.text)
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

# Repository Management Commands
@app.command()
def list_repositories():
    """List repositories"""
    try:
        response = requests.get("http://localhost:8000/api/repositories", timeout=10)
        if response.status_code == 200:
            repos = response.json()
            if not repos:
                typer.echo("No repositories found.")
                return

            typer.echo("Repositories:")
            for repo in repos:
                typer.echo(f"  - {repo['name']}")
                typer.echo(f"    URL: {repo['url']}")
                typer.echo(f"    Branch: {repo['branch']}")
                typer.echo(f"    Status: {repo['status']}")
                if repo.get('description'):
                    typer.echo(f"    Description: {repo['description']}")
                typer.echo("")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

@app.command()
def create_repository(
    name: str,
    url: str,
    branch: str = typer.Option("main", help="Branch to use")
):
    """Create repository"""
    try:
        payload = {
            "name": name,
            "url": url,
            "branch": branch
        }
        response = requests.post("http://localhost:8000/api/repositories", json=payload, timeout=10)
        if response.status_code == 200:
            repo = response.json()
            typer.echo(f"✅ Repository created: {repo['name']}")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
            typer.echo(response.text)
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

@app.command()
def sync_repository(repo_id: str):
    """Sync repository"""
    try:
        response = requests.post(f"http://localhost:8000/api/repositories/{repo_id}/sync", timeout=10)
        if response.status_code == 200:
            typer.echo("✅ Repository synced successfully")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

# Security Commands
@app.command()
def list_security_policies():
    """List security policies"""
    try:
        response = requests.get("http://localhost:8000/api/security/policies", timeout=10)
        if response.status_code == 200:
            policies = response.json()
            if not policies:
                typer.echo("No security policies found.")
                return

            typer.echo("Security Policies:")
            for policy in policies:
                typer.echo(f"  - {policy['name']}")
                typer.echo(f"    Category: {policy['category']}")
                typer.echo(f"    Severity: {policy['severity']}")
                typer.echo(f"    Enabled: {policy['enabled']}")
                typer.echo("")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

@app.command()
def create_security_policy(name: str, category: str):
    """Create security policy"""
    try:
        payload = {
            "name": name,
            "category": category,
            "severity": "medium",
            "enabled": True
        }
        response = requests.post("http://localhost:8000/api/security/policies", json=payload, timeout=10)
        if response.status_code == 200:
            policy = response.json()
            typer.echo(f"✅ Security policy created: {policy['name']}")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
            typer.echo(response.text)
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

@app.command()
def list_security_scans():
    """List security scans"""
    try:
        response = requests.get("http://localhost:8000/api/security/scans", timeout=10)
        if response.status_code == 200:
            scans = response.json()
            if not scans:
                typer.echo("No security scans found.")
                return

            typer.echo("Security Scans:")
            for scan in scans:
                typer.echo(f"  - {scan['id']}")
                typer.echo(f"    Target: {scan['target_type']} - {scan['target_id']}")
                typer.echo(f"    Status: {scan['status']}")
                typer.echo(f"    Score: {scan['score']}")
                typer.echo("")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

@app.command()
def create_security_scan(target_type: str, target_id: str):
    """Create security scan"""
    try:
        payload = {
            "target_type": target_type,
            "target_id": target_id
        }
        response = requests.post("http://localhost:8000/api/security/scans", json=payload, timeout=10)
        if response.status_code == 200:
            scan = response.json()
            typer.echo(f"✅ Security scan created: {scan['id']}")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
            typer.echo(response.text)
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

# Observability Commands
@app.command()
def list_metrics(category: Optional[str] = None):
    """List observability metrics"""
    try:
        params = {}
        if category:
            params['category'] = category

        response = requests.get("http://localhost:8000/api/observability/metrics", params=params, timeout=10)
        if response.status_code == 200:
            metrics = response.json()
            if not metrics:
                typer.echo("No metrics found.")
                return

            typer.echo("Observability Metrics:")
            for metric in metrics:
                typer.echo(f"  - {metric['name']}")
                typer.echo(f"    Category: {metric['category']}")
                typer.echo(f"    Value: {metric['value']} {metric['unit']}")
                typer.echo(f"    Timestamp: {metric['timestamp']}")
                typer.echo("")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

@app.command()
def create_metric(
    name: str,
    value: float,
    category: str = typer.Option("system", help="Metric category")
):
    """Create observability metric"""
    try:
        payload = {
            "name": name,
            "value": value,
            "category": category,
            "unit": "count"
        }
        response = requests.post("http://localhost:8000/api/observability/metrics", json=payload, timeout=10)
        if response.status_code == 200:
            metric = response.json()
            typer.echo(f"✅ Metric created: {metric['name']}")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
            typer.echo(response.text)
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

# Prompt Management Commands
@app.command()
def list_prompts(category: Optional[str] = None):
    """List prompts"""
    try:
        params = {}
        if category:
            params['category'] = category

        response = requests.get("http://localhost:8000/api/prompts", params=params, timeout=10)
        if response.status_code == 200:
            prompts = response.json()
            if not prompts:
                typer.echo("No prompts found.")
                return

            typer.echo("Prompts:")
            for prompt in prompts:
                typer.echo(f"  - {prompt['name']}")
                typer.echo(f"    Category: {prompt['category']}")
                typer.echo(f"    Usage Count: {prompt['usage_count']}")
                typer.echo(f"    Success Rate: {prompt['success_rate']}%")
                typer.echo("")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

@app.command()
def create_prompt(
    name: str,
    content: str,
    category: str = typer.Option("coding", help="Prompt category")
):
    """Create prompt"""
    try:
        payload = {
            "name": name,
            "content": content,
            "category": category
        }
        response = requests.post("http://localhost:8000/api/prompts", json=payload, timeout=10)
        if response.status_code == 200:
            prompt = response.json()
            typer.echo(f"✅ Prompt created: {prompt['name']}")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
            typer.echo(response.text)
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

# Intelligence Commands
@app.command()
def list_analyses(target_type: Optional[str] = None):
    """List intelligence analyses"""
    try:
        params = {}
        if target_type:
            params['target_type'] = target_type

        response = requests.get("http://localhost:8000/api/intelligence/analyses", params=params, timeout=10)
        if response.status_code == 200:
            analyses = response.json()
            if not analyses:
                typer.echo("No analyses found.")
                return

            typer.echo("Intelligence Analyses:")
            for analysis in analyses:
                typer.echo(f"  - {analysis['id']}")
                typer.echo(f"    Target: {analysis['target_type']} - {analysis['target_id']}")
                typer.echo(f"    Type: {analysis['analysis_type']}")
                typer.echo(f"    Confidence: {analysis['confidence']}")
                typer.echo("")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

@app.command()
def create_analysis(target_type: str, target_id: str):
    """Create intelligence analysis"""
    try:
        payload = {
            "target_type": target_type,
            "target_id": target_id,
            "analysis_type": "complexity"
        }
        response = requests.post("http://localhost:8000/api/intelligence/analyze", json=payload, timeout=10)
        if response.status_code == 200:
            analysis = response.json()
            typer.echo(f"✅ Analysis created: {analysis['id']}")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
            typer.echo(response.text)
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

# Integration Commands
@app.command()
def list_integrations():
    """List integrations"""
    try:
        response = requests.get("http://localhost:8000/api/integrations", timeout=10)
        if response.status_code == 200:
            integrations = response.json()
            if not integrations:
                typer.echo("No integrations found.")
                return

            typer.echo("Integrations:")
            for integration in integrations:
                typer.echo(f"  - {integration['name']}")
                typer.echo(f"    Type: {integration['type']}")
                typer.echo(f"    Status: {integration['status']}")
                typer.echo(f"    Description: {integration['description']}")
                typer.echo("")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

@app.command()
def create_integration(name: str, type_: str = typer.Option("webhook", help="Integration type")):
    """Create integration"""
    try:
        payload = {
            "name": name,
            "type": type_,
            "description": f"{type_} integration",
            "config": {}
        }
        response = requests.post("http://localhost:8000/api/integrations", json=payload, timeout=10)
        if response.status_code == 200:
            integration = response.json()
            typer.echo(f"✅ Integration created: {integration['name']}")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
            typer.echo(response.text)
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

@app.command()
def sync_integration(integration_id: str):
    """Sync integration"""
    try:
        response = requests.post(f"http://localhost:8000/api/integrations/{integration_id}/sync", timeout=10)
        if response.status_code == 200:
            typer.echo("✅ Integration synced successfully")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

# Project Commands
@app.command()
def create_project(
    name: str,
    description: Optional[str] = None
):
    """Create project"""
    try:
        payload = {
            "name": name,
            "description": description
        }
        response = requests.post("http://localhost:8000/api/projects", json=payload, timeout=10)
        if response.status_code == 200:
            project = response.json()
            typer.echo(f"✅ Project created: {project['name']}")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
            typer.echo(response.text)
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

@app.command()
def list_projects():
    """List projects"""
    try:
        response = requests.get("http://localhost:8000/api/projects", timeout=10)
        if response.status_code == 200:
            projects = response.json()
            if not projects:
                typer.echo("No projects found.")
                return

            typer.echo("Projects:")
            for project in projects:
                typer.echo(f"  - {project['name']}")
                if project.get('description'):
                    typer.echo(f"    Description: {project['description']}")
                typer.echo(f"    Status: {project['status']}")
                typer.echo(f"    Tasks: {project['task_count']}")
                typer.echo("")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

@app.command()
def get_project(project_id: str):
    """Get project details"""
    try:
        response = requests.get(f"http://localhost:8000/api/projects/{project_id}", timeout=10)
        if response.status_code == 200:
            project = response.json()
            typer.echo("Project Details:")
            typer.echo(f"  Name: {project['name']}")
            if project.get('description'):
                typer.echo(f"  Description: {project['description']}")
            typer.echo(f"  Status: {project['status']}")
            typer.echo(f"  Tasks: {project['task_count']}")
            typer.echo(f"  Created: {project['created_at']}")
        else:
            typer.echo(f"Error: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        typer.echo(f"Connection error: {e}", err=True)

@app.command()
def help():
    """Show available commands"""
    typer.echo("CodeAgent CLI Commands:")
    typer.echo("")
    typer.echo("Task Management:")
    typer.echo("  run-task <description> [--wait/--no-wait] [--timeout SEC] [--verbose] [--websocket]  Run a task (curl-like synchronous)")
    typer.echo("  list-tasks [--project PROJECT] [--status STATUS]    List tasks")
    typer.echo("  get-task <task_id>                                   Get task details")
    typer.echo("")
    typer.echo("System Monitoring:")
    typer.echo("  health                                               Check system health")
    typer.echo("  status                                               Get system status")
    typer.echo("  list-agents                                          List available agents")
    typer.echo("  activate-agents                                      Activate all idle agents")
    typer.echo("")
    typer.echo("Provider Management:")
    typer.echo("  list-providers                                       List Aetherium providers")
    typer.echo("  provider-metrics                                     Show provider metrics")
    typer.echo("  switch-provider <provider_id>                        Switch active provider")
    typer.echo("")
    typer.echo("Repository Management:")
    typer.echo("  list-repositories                                    List repositories")
    typer.echo("  create-repository <name> <url> [--branch BRANCH]     Create repository")
    typer.echo("  sync-repository <repo_id>                            Sync repository")
    typer.echo("")
    typer.echo("Security:")
    typer.echo("  list-security-policies                               List security policies")
    typer.echo("  create-security-policy <name> <category>             Create security policy")
    typer.echo("  list-security-scans                                  List security scans")
    typer.echo("  create-security-scan <target_type> <target_id>       Create security scan")
    typer.echo("")
    typer.echo("Observability:")
    typer.echo("  list-metrics [--category CATEGORY]                   List observability metrics")
    typer.echo("  create-metric <name> <value> [--category CATEGORY]   Create observability metric")
    typer.echo("")
    typer.echo("Prompt Management:")
    typer.echo("  list-prompts [--category CATEGORY]                   List prompts")
    typer.echo("  create-prompt <name> <content> [--category CATEGORY] Create prompt")
    typer.echo("")
    typer.echo("Intelligence:")
    typer.echo("  list-analyses [--target_type TYPE]                   List intelligence analyses")
    typer.echo("  create-analysis <target_type> <target_id>            Create intelligence analysis")
    typer.echo("")
    typer.echo("Integrations:")
    typer.echo("  list-integrations                                    List integrations")
    typer.echo("  create-integration <name> [--type TYPE]              Create integration")
    typer.echo("  sync-integration <integration_id>                    Sync integration")
    typer.echo("")
    typer.echo("Projects:")
    typer.echo("  create-project <name> [--description DESC]           Create project")
    typer.echo("  list-projects                                        List projects")
    typer.echo("  get-project <project_id>                             Get project details")
    typer.echo("")
    typer.echo("Other:")
    typer.echo("  interactive                                          Start interactive CLI mode")
    typer.echo("  help                                                 Show this help message")
    typer.echo("")
    typer.echo("Interactive mode supports all the above commands without the 'python cli.py' prefix.")
    typer.echo("The CLI works like curl - it waits for task completion and shows results synchronously.")

@app.command()
def interactive():
    """Start interactive CLI mode"""
    typer.echo("CodeAgent Interactive CLI")
    typer.echo("Type 'help' for available commands or 'exit' to quit")

    while True:
        try:
            cmd = input("> ").strip()
            if not cmd:
                continue
            if cmd.lower() == 'exit':
                break
            elif cmd.lower() == 'help':
                typer.echo("Available commands: run-task, health, status, list-agents, activate-agents, list-tasks, list-providers, help, exit")
                typer.echo("Usage: run-task <task description> [--wait/--no-wait] [--timeout SEC] [--verbose]")
                typer.echo("Type 'help' (without quotes) for full command reference")
            elif cmd.lower() == 'health':
                health()
            elif cmd.lower() == 'status':
                status()
            elif cmd.lower() == 'list-agents':
                list_agents()
            elif cmd.lower() == 'list-tasks':
                list_tasks()
            elif cmd.lower() == 'list-providers':
                list_providers()
            elif cmd.lower() == 'provider-metrics':
                provider_metrics()
            elif cmd.lower() == 'list-repositories':
                list_repositories()
            elif cmd.lower() == 'list-security-policies':
                list_security_policies()
            elif cmd.lower() == 'list-security-scans':
                list_security_scans()
            elif cmd.lower() == 'list-metrics':
                list_metrics()
            elif cmd.lower() == 'list-prompts':
                list_prompts()
            elif cmd.lower() == 'list-analyses':
                list_analyses()
            elif cmd.lower() == 'list-integrations':
                list_integrations()
            elif cmd.lower() == 'list-projects':
                list_projects()
            elif cmd.lower().startswith('run-task '):
                task = cmd[9:].strip()
                if task:
                    # Parse options if present
                    parts = task.split()
                    description = task
                    wait = True
                    timeout = 300
                    verbose = False
                    websocket = False

                    # Simple option parsing
                    if '--no-wait' in parts:
                        wait = False
                        parts.remove('--no-wait')
                        description = ' '.join(parts)
                    elif '--wait' in parts:
                        wait = True
                        parts.remove('--wait')
                        description = ' '.join(parts)

                    if '--verbose' in parts:
                        verbose = True
                        parts.remove('--verbose')
                        description = ' '.join(parts)

                    if '--websocket' in parts:
                        websocket = True
                        parts.remove('--websocket')
                        description = ' '.join(parts)

                    # Check for timeout
                    timeout_idx = -1
                    for i, part in enumerate(parts):
                        if part == '--timeout':
                            timeout_idx = i
                            break
                    if timeout_idx >= 0 and timeout_idx + 1 < len(parts):
                        try:
                            timeout = int(parts[timeout_idx + 1])
                            parts.pop(timeout_idx + 1)
                            parts.pop(timeout_idx)
                            description = ' '.join(parts)
                        except ValueError:
                            pass

                    run_task(description, wait, timeout, verbose, websocket)
                else:
                    typer.echo("Usage: run-task <task description> [--wait/--no-wait] [--timeout SEC] [--verbose] [--websocket]")
            elif cmd.lower() == 'activate-agents':
                activate_agents()
            elif cmd.lower().startswith('get-task '):
                task_id = cmd[9:].strip()
                if task_id:
                    get_task(task_id)
                else:
                    typer.echo("Usage: get-task <task_id>")
            elif cmd.lower().startswith('switch-provider '):
                provider_id = cmd[16:].strip()
                if provider_id:
                    switch_provider(provider_id)
                else:
                    typer.echo("Usage: switch-provider <provider_id>")
            elif cmd.lower().startswith('create-repository '):
                parts = cmd[17:].strip().split()
                if len(parts) >= 2:
                    name, url = parts[0], parts[1]
                    branch = "main"
                    if len(parts) > 2:
                        branch = parts[2]
                    create_repository(name, url, branch)
                else:
                    typer.echo("Usage: create-repository <name> <url> [branch]")
            elif cmd.lower().startswith('sync-repository '):
                repo_id = cmd[16:].strip()
                if repo_id:
                    sync_repository(repo_id)
                else:
                    typer.echo("Usage: sync-repository <repo_id>")
            elif cmd.lower().startswith('create-security-policy '):
                parts = cmd[23:].strip().split()
                if len(parts) >= 2:
                    name, category = parts[0], parts[1]
                    create_security_policy(name, category)
                else:
                    typer.echo("Usage: create-security-policy <name> <category>")
            elif cmd.lower().startswith('create-security-scan '):
                parts = cmd[21:].strip().split()
                if len(parts) >= 2:
                    target_type, target_id = parts[0], parts[1]
                    create_security_scan(target_type, target_id)
                else:
                    typer.echo("Usage: create-security-scan <target_type> <target_id>")
            elif cmd.lower().startswith('create-metric '):
                parts = cmd[14:].strip().split()
                if len(parts) >= 2:
                    name, value_str = parts[0], parts[1]
                    try:
                        value = float(value_str)
                        category = "system"
                        if len(parts) > 2:
                            category = parts[2]
                        create_metric(name, value, category)
                    except ValueError:
                        typer.echo("Error: value must be a number")
                else:
                    typer.echo("Usage: create-metric <name> <value> [category]")
            elif cmd.lower().startswith('create-prompt '):
                parts = cmd[14:].strip().split()
                if len(parts) >= 2:
                    name, content = parts[0], ' '.join(parts[1:])
                    category = "coding"
                    create_prompt(name, content, category)
                else:
                    typer.echo("Usage: create-prompt <name> <content>")
            elif cmd.lower().startswith('create-analysis '):
                parts = cmd[16:].strip().split()
                if len(parts) >= 2:
                    target_type, target_id = parts[0], parts[1]
                    create_analysis(target_type, target_id)
                else:
                    typer.echo("Usage: create-analysis <target_type> <target_id>")
            elif cmd.lower().startswith('create-integration '):
                parts = cmd[19:].strip().split()
                if len(parts) >= 1:
                    name = parts[0]
                    type_ = "webhook"
                    if len(parts) > 1:
                        type_ = parts[1]
                    create_integration(name, type_)
                else:
                    typer.echo("Usage: create-integration <name> [type]")
            elif cmd.lower().startswith('sync-integration '):
                integration_id = cmd[17:].strip()
                if integration_id:
                    sync_integration(integration_id)
                else:
                    typer.echo("Usage: sync-integration <integration_id>")
            elif cmd.lower().startswith('create-project '):
                parts = cmd[15:].strip().split()
                if len(parts) >= 1:
                    name = parts[0]
                    description = None
                    if len(parts) > 1:
                        description = ' '.join(parts[1:])
                    create_project(name, description)
                else:
                    typer.echo("Usage: create-project <name> [description]")
            elif cmd.lower().startswith('get-project '):
                project_id = cmd[12:].strip()
                if project_id:
                    get_project(project_id)
                else:
                    typer.echo("Usage: get-project <project_id>")
            else:
                typer.echo(f"Unknown command: {cmd}")
                typer.echo("Type 'help' for available commands")
        except KeyboardInterrupt:
            break
        except EOFError:
            break

    typer.echo("Goodbye!")

if __name__ == "__main__":
    app()