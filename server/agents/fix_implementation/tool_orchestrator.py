import re
import json
import logging
from typing import List, Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)

class ToolOrchestrator:
    """Orchestrates tool execution based on natural language model responses."""

    def __init__(self, tool_api_url: str = "http://localhost:8001"):
        self.tool_api_url = tool_api_url

    def parse_tool_requests(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse tool requests from natural language model responses.

        Looks for patterns like:
        - "I need to run: run_tests(repo_url='...', test_command='...')"
        - "Let me read the file: git_read_file(repo_url='...', file_path='...')"
        - "Execute: shell_exec(command='...')"
        """
        tool_requests = []

        # Pattern 1: function_name(args...)
        func_pattern = r'(\w+)\s*\(\s*([^)]*)\s*\)'
        matches = re.findall(func_pattern, response_text)

        for func_name, args_str in matches:
            if func_name in ['run_tests', 'git_read_file', 'git_write_file', 'list_files', 'shell_exec']:
                try:
                    # Parse arguments - handle both quoted and unquoted values
                    args = self._parse_function_args(args_str)
                    tool_requests.append({
                        'function': func_name,
                        'args': args
                    })
                    logger.info(f"Parsed tool request: {func_name} with args {args}")
                except Exception as e:
                    logger.warning(f"Failed to parse args for {func_name}: {args_str} - {e}")

        # Pattern 2: "I need to [action]" statements
        action_patterns = [
            (r'I need to run tests? (?:on|for|in) ([^.\n]+)', 'run_tests'),
            (r'I need to read (?:the )?file ([^.\n]+)', 'git_read_file'),
            (r'I need to list files? (?:in )?([^.\n]+)', 'list_files'),
            (r'I need to execute ([^.\n]+)', 'shell_exec'),
        ]

        for pattern, func_name in action_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            for match in matches:
                # Try to extract structured args from the match
                args = self._extract_args_from_text(match.strip(), func_name)
                if args:
                    tool_requests.append({
                        'function': func_name,
                        'args': args
                    })

        return tool_requests

    def _parse_function_args(self, args_str: str) -> Dict[str, Any]:
        """Parse function arguments from string like 'repo_url="...", test_command="..."'"""
        args = {}

        # Split by commas, but be careful with quoted strings
        arg_pairs = []
        current_pair = ""
        in_quotes = False
        quote_char = None

        for char in args_str:
            if not in_quotes and char in ['"', "'"]:
                in_quotes = True
                quote_char = char
                current_pair += char
            elif in_quotes and char == quote_char:
                in_quotes = False
                quote_char = None
                current_pair += char
            elif not in_quotes and char == ',':
                if current_pair.strip():
                    arg_pairs.append(current_pair.strip())
                current_pair = ""
            else:
                current_pair += char

        if current_pair.strip():
            arg_pairs.append(current_pair.strip())

        # Parse each arg=value pair
        for pair in arg_pairs:
            if '=' in pair:
                key, value = pair.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]

                args[key] = value

        return args

    def _extract_args_from_text(self, text: str, func_name: str) -> Optional[Dict[str, Any]]:
        """Extract arguments from natural language text based on function type."""
        if func_name == 'run_tests':
            # Look for repo URL patterns
            repo_match = re.search(r'(?:https?://[^\s]+)', text)
            if repo_match:
                return {
                    'repo_url': repo_match.group(0),
                    'test_command': 'pytest'  # default
                }
        elif func_name == 'git_read_file':
            # Look for file path patterns
            file_match = re.search(r'([^\s]+\.[^\s]+)', text)
            if file_match:
                return {
                    'repo_url': 'https://github.com/user/repo',  # placeholder - would need context
                    'file_path': file_match.group(0)
                }
        elif func_name == 'shell_exec':
            return {
                'command': text.strip(),
                'working_dir': '/workspace'
            }

        return None

    def execute_tool_request(self, tool_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool request via the tool API."""
        func_name = tool_request['function']
        args = tool_request['args']

        try:
            if func_name == "git_read_file":
                response = requests.post(f"{self.tool_api_url}/git_read_file", json=args)
            elif func_name == "git_write_file":
                response = requests.post(f"{self.tool_api_url}/git_write_file", json=args)
            elif func_name == "list_files":
                response = requests.post(f"{self.tool_api_url}/list_files", json=args)
            elif func_name == "run_tests":
                response = requests.post(f"{self.tool_api_url}/run_tests", json=args)
            elif func_name == "shell_exec":
                response = requests.post(f"{self.tool_api_url}/shell_exec", json=args)
            else:
                return {"error": f"Unknown tool: {func_name}"}

            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Tool API error: {response.status_code}", "details": response.text}
        except Exception as e:
            return {"error": str(e)}

    def execute_tool_requests(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse and execute all tool requests found in the response text."""
        tool_requests = self.parse_tool_requests(response_text)
        results = []

        for request in tool_requests:
            logger.info(f"Executing tool: {request['function']} with args {request['args']}")
            result = self.execute_tool_request(request)
            results.append({
                'request': request,
                'result': result
            })

        return results

    def format_tool_results(self, tool_results: List[Dict[str, Any]]) -> str:
        """Format tool results for inclusion in the next model message."""
        if not tool_results:
            return ""

        formatted = "\n\nTool Execution Results:\n"
        for i, result in enumerate(tool_results, 1):
            req = result['request']
            res = result['result']

            formatted += f"\n{i}. {req['function']}({req['args']})\n"
            if 'error' in res:
                formatted += f"   Error: {res['error']}\n"
            else:
                # Format successful results
                if req['function'] == 'git_read_file':
                    content = res.get('content', '')[:500]  # Truncate long content
                    formatted += f"   Content: {content}...\n" if len(res.get('content', '')) > 500 else f"   Content: {res.get('content', '')}\n"
                elif req['function'] == 'run_tests':
                    formatted += f"   Success: {res.get('success', False)}\n"
                    output = res.get('output', '')[:300]
                    formatted += f"   Output: {output}...\n" if len(res.get('output', '')) > 300 else f"   Output: {res.get('output', '')}\n"
                elif req['function'] == 'shell_exec':
                    formatted += f"   Exit code: {res.get('exit_code', '?')}\n"
                    stdout = res.get('stdout', '')[:300]
                    formatted += f"   Output: {stdout}...\n" if len(res.get('stdout', '')) > 300 else f"   Output: {res.get('stdout', '')}\n"
                else:
                    formatted += f"   Result: {json.dumps(res, indent=2)}\n"

        return formatted