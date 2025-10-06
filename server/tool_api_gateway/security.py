from typing import Dict, Any

# Allowlist for shell_exec commands
ALLOWED_COMMANDS = [
    "ls", "cat", "echo", "pwd", "grep", "find", "wc", "head", "tail",
    "sort", "uniq", "cut", "awk", "sed", "mkdir", "rm", "cp", "mv",
    "chmod", "chown", "ps", "top", "df", "du", "free", "uptime"
]

def is_command_allowed(command: str) -> bool:
    """Check if the command is in the allowlist."""
    base_cmd = command.split()[0]
    return base_cmd in ALLOWED_COMMANDS

def check_opa_policy(input_data: Dict[str, Any], policy_name: str) -> bool:
    """
    Check policy using Open Policy Agent.
    This is a mock implementation. In production, call OPA server.
    """
    # Mock OPA check - always allow for now
    # In real implementation:
    # response = requests.post("http://opa:8181/v1/data/policy", json=input_data)
    # return response.json().get("result", False)
    return True

def validate_request(tool_name: str, request_data: Dict[str, Any]) -> bool:
    """Validate request based on tool and security policies."""
    if tool_name == "shell_exec":
        command = request_data.get("command", "")
        if not is_command_allowed(command):
            return False
    # Add other validations as needed
    return check_opa_policy(request_data, tool_name)