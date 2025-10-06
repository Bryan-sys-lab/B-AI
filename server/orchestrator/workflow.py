from orchestrator.planner import Planner
from orchestrator.router import Router
from orchestrator.database import async_session, Task, Subtask
from sqlalchemy import select, func
import httpx
import json
import asyncio
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

async def plan_task(task_description: str, context: dict = None) -> dict:
    logger.info(f"WORKFLOW: Starting plan_task with description: {task_description[:100]}...")
    planner = Planner()
    result = await planner.decompose_task(task_description, context)
    logger.info(f"WORKFLOW: plan_task completed, subtasks: {len(result.get('subtasks', []))}")
    return result

async def route_subtasks(plan: dict, task_context: dict = None) -> list:
    logger.info(f"WORKFLOW: Starting route_subtasks with {len(plan.get('subtasks', []))} subtasks")
    router = Router()
    result = await router.route_subtasks(plan["subtasks"], task_context)
    logger.info(f"WORKFLOW: route_subtasks completed, routed: {len(result)}")
    return result

async def create_subtasks_in_db(task_id: str, routed_subtasks: list):
    async with async_session() as session:
        subtasks = []
        for rsub in routed_subtasks:
            subtask = Subtask(
                task_id=task_id,
                agent_name=rsub["agent"],
                description=rsub["description"],
                confidence=rsub["confidence"]
            )
            session.add(subtask)
            subtasks.append(subtask)
        await session.commit()
    return subtasks

def normalize_agent_response(agent_response: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
    """
    Normalize varied agent response formats to a consistent workflow format.

    Handles different response patterns from various agents:
    - Standard format: {"success": bool, "result": str, "error": str}
    - Alternative formats: {"response": str}, {"output": str}, {"data": any}
    - Error formats: {"error": str}, {"message": str}, {"details": str}
    """
    logger.info(f"WORKFLOW: Normalizing response from agent {agent_name}")
    logger.info(f"WORKFLOW: Raw agent_response keys: {list(agent_response.keys())}")
    logger.info(f"WORKFLOW: Raw agent_response success: {agent_response.get('success')}")
    logger.info(f"WORKFLOW: Raw agent_response has result: {'result' in agent_response}")
    logger.info(f"WORKFLOW: Raw agent_response has structured: {'structured' in agent_response}")

    # Handle standard success/error format
    if "success" in agent_response:
        if agent_response["success"]:
            logger.info(f"WORKFLOW: Agent {agent_name} reported success")
            output = {"response": agent_response.get("result", "")}
            logger.info(f"WORKFLOW: Extracted response: {output['response'][:200]}...")

            # Extract additional metadata
            if "structured" in agent_response:
                output["structured"] = agent_response["structured"]
                logger.info(f"WORKFLOW: Agent {agent_name} returned structured data")
            if "tokens" in agent_response:
                output["tokens"] = agent_response["tokens"]
            if "metadata" in agent_response:
                output["metadata"] = agent_response["metadata"]
            if "files" in agent_response:
                output["files"] = agent_response["files"]
            if "zip_download_url" in agent_response:
                output["zip_download_url"] = agent_response["zip_download_url"]

            logger.info(f"WORKFLOW: Normalized output keys: {list(output.keys())}")
            return output
        else:
            error_msg = agent_response.get("error", agent_response.get("message", "Agent execution failed"))
            logger.warning(f"WORKFLOW: Agent {agent_name} reported failure: {error_msg}")
            return {"error": error_msg}

    # Handle direct response formats
    if "response" in agent_response:
        logger.info(f"WORKFLOW: Agent {agent_name} returned direct response")
        output = {"response": agent_response["response"]}
        # Include other fields
        for key in ["structured", "tokens", "metadata", "files", "zip_download_url"]:
            if key in agent_response:
                output[key] = agent_response[key]
        return output

    # Handle output/result formats
    if "output" in agent_response:
        logger.info(f"WORKFLOW: Agent {agent_name} returned output field")
        return {"response": agent_response["output"]}

    if "result" in agent_response:
        logger.info(f"WORKFLOW: Agent {agent_name} returned result field")
        return {"response": str(agent_response["result"])}

    if "data" in agent_response:
        logger.info(f"WORKFLOW: Agent {agent_name} returned data field")
        return {"response": json.dumps(agent_response["data"]) if not isinstance(agent_response["data"], str) else agent_response["data"]}

    # Handle error-only formats
    if "error" in agent_response:
        logger.warning(f"WORKFLOW: Agent {agent_name} returned error: {agent_response['error']}")
        return {"error": agent_response["error"]}

    if "message" in agent_response:
        # Could be error or success message
        message = agent_response["message"]
        if any(word in message.lower() for word in ["error", "failed", "failure", "exception"]):
            logger.warning(f"WORKFLOW: Agent {agent_name} returned error message: {message}")
            return {"error": message}
        else:
            logger.info(f"WORKFLOW: Agent {agent_name} returned success message")
            return {"response": message}

    # Handle empty or minimal responses
    if not agent_response or len(agent_response) == 0:
        logger.warning(f"WORKFLOW: Agent {agent_name} returned empty response")
        return {"response": "Task completed (no output)"}

    # Fallback: convert entire response to string
    logger.warning(f"WORKFLOW: Agent {agent_name} returned unrecognized format, converting to string")
    return {"response": json.dumps(agent_response, indent=2)}


async def execute_subtask(subtask: Subtask, manager=None) -> Dict[str, Any]:
    import time
    subtask_start = time.time()

    # Handle simple tasks directly
    desc_lower = subtask.description.lower().strip()
    logger.info(f"Executing subtask with description: '{subtask.description}', desc_lower: '{desc_lower}', agent: {subtask.agent_name}")

    # Get task context which may include conversation history
    async with async_session() as session:
        result = await session.execute(select(Task).where(Task.id == subtask.task_id))
        task = result.scalar_one()
        task_context = task.context or {}
        conversation_history = task_context.get("conversation_history", [])
        logger.info(f"WORKFLOW: Retrieved conversation history for task {subtask.task_id}: {len(conversation_history)} messages")
        if desc_lower in ["hello", "hi", "greetings", "hello!", "hi!", "greetings!"]:
            logger.info("Special case triggered for greeting")
            output = {"response": "Hello! How can I assist you with your code today?"}
            return output

        # Use centralized endpoint manager for different deployment patterns
        from orchestrator.main import endpoint_manager

        # Try streaming endpoint first, fallback to regular endpoint
        stream_url = endpoint_manager.get_endpoint(subtask.agent_name, "stream")
        regular_url = endpoint_manager.get_endpoint(subtask.agent_name, "execute")

        if stream_url:
            # Use streaming endpoint
            logger.info(f"WORKFLOW: Using streaming endpoint for agent {subtask.agent_name}: {stream_url}")
            url = stream_url
            is_streaming = True
        elif regular_url:
            # Use regular endpoint
            logger.info(f"WORKFLOW: Using regular endpoint for agent {subtask.agent_name}: {regular_url}")
            url = regular_url
            is_streaming = False
        else:
            logger.error(f"WORKFLOW: No endpoint configured for agent {subtask.agent_name}")
            output = {"error": f"Agent {subtask.agent_name} not available - no endpoint configured"}
            return output

        # No timeout limits - allow agents to run indefinitely
        logger.info(f"WORKFLOW: Agent {subtask.agent_name} running without timeout limits "
                     f"for task length {len(subtask.description) if subtask.description else 0}")

        async with httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_keepalive_connections=None, max_connections=None, keepalive_expiry=None)) as client:
            try:
                agent_call_start = time.time()

                if is_streaming:
                    # For streaming endpoints, use GET with query parameters
                    params = {"description": subtask.description}
                    if conversation_history:
                        params["conversation_history"] = json.dumps(conversation_history)
                    logger.info(f"WORKFLOW: Calling streaming agent {subtask.agent_name} at {url} with params: {json.dumps(params)[:500]}...")
                    response = await client.get(url, params=params, timeout=300.0)  # 5 minute timeout for streaming

                    # Check if streaming endpoint exists (not 404)
                    if response.status_code == 404:
                        logger.warning(f"WORKFLOW: Streaming endpoint not found for agent {subtask.agent_name}, falling back to regular endpoint")
                        # Fall back to regular endpoint
                        regular_url = endpoint_manager.get_endpoint(subtask.agent_name, "execute")
                        if regular_url:
                            logger.info(f"WORKFLOW: Falling back to regular endpoint: {regular_url}")
                            request_payload = {"description": subtask.description}
                            if conversation_history:
                                request_payload["conversation_history"] = conversation_history
                            response = await client.post(regular_url, json=request_payload)
                            agent_response = response.json()
                        else:
                            agent_response = {"error": f"No endpoint available for agent {subtask.agent_name}"}
                    else:
                        # Parse Server-Sent Events response
                        response_text = response.text
                        logger.info(f"WORKFLOW: Streaming response size: {len(response_text)} characters")

                        # Extract the final result from the stream
                        lines = response_text.split('\n')
                        final_result = None
                        for line in lines:
                            if line.startswith('data: '):
                                try:
                                    data = json.loads(line[6:])  # Remove 'data: ' prefix
                                    if data.get('type') == 'complete':
                                        # The complete event contains the full response data
                                        final_result = data.get('result')
                                        if final_result:
                                            logger.info(f"WORKFLOW: Found complete event with result containing keys: {list(final_result.keys())}")
                                            break
                                    elif data.get('type') == 'file':
                                        # Handle file events - these contain individual file data
                                        logger.info(f"WORKFLOW: Received file event: {data.get('filename', 'unknown')}")
                                    elif data.get('type') == 'progress':
                                        # Handle progress events
                                        logger.debug(f"WORKFLOW: Progress: {data.get('message', '')}")
                                except json.JSONDecodeError as e:
                                    logger.warning(f"WORKFLOW: Failed to parse SSE line: {line[:100]}... Error: {e}")
                                    continue

                        if final_result:
                            # The streaming response already contains the properly formatted agent response
                            agent_response = final_result
                            logger.info(f"WORKFLOW: Extracted final result from streaming response")
                        else:
                            logger.error(f"WORKFLOW: No complete event found in streaming response from agent {subtask.agent_name}")
                            logger.error(f"WORKFLOW: Raw SSE response preview: {response_text[:1000]}...")
                            agent_response = {"error": "Failed to parse streaming response - no complete event found"}
                else:
                    # For regular endpoints, use POST with JSON body
                    request_payload = {"description": subtask.description}
                    if conversation_history:
                        request_payload["conversation_history"] = conversation_history
                    logger.info(f"WORKFLOW: Calling agent {subtask.agent_name} at {url} with payload: {json.dumps(request_payload)[:500]}...")
                    response = await client.post(url, json=request_payload)
                    agent_response = response.json()

                agent_call_time = time.time() - agent_call_start
                logger.info(f"Agent {subtask.agent_name} call took {agent_call_time:.3f}s")

                # Log HTTP response details
                logger.info(f"WORKFLOW: HTTP Status: {response.status_code}")
                logger.info(f"WORKFLOW: HTTP Headers: {dict(response.headers)}")
                logger.info(f"WORKFLOW: Content-Type: {response.headers.get('content-type', 'unknown')}")
                logger.info(f"WORKFLOW: Content-Length header: {response.headers.get('content-length', 'not set')}")

                if is_streaming:
                    # For streaming responses, agent_response is already set above
                    logger.info(f"WORKFLOW: Using streaming response, skipping JSON parsing")
                else:
                    # For regular endpoints, parse JSON response
                    response_text = response.text
                    response_size = len(response_text)
                    logger.info(f"WORKFLOW: Agent {subtask.agent_name} raw response text size: {response_size} characters")
                    logger.info(f"WORKFLOW: Agent {subtask.agent_name} raw response text preview: {response_text[:2000]}...")

                    # Check for obvious truncation indicators
                    if response_text.strip() in ['{"error": ""}', '{"error":""}', '{"error": null}', '{"error":null}']:
                        logger.error(f"WORKFLOW: Detected empty error response - likely truncation! Raw text: '{response_text}'")
                    elif response_size < 50 and not response_text.strip().startswith('{'):
                        logger.error(f"WORKFLOW: Suspiciously small response ({response_size} chars): '{response_text}'")

                    # Parse JSON
                    agent_response = response.json()
                    logger.info(f"WORKFLOW: Agent {subtask.agent_name} parsed JSON response: {json.dumps(agent_response)[:1000]}...")

                    # Check for truncated responses
                    if agent_response == {"error": ""} or (isinstance(agent_response.get("error"), str) and len(agent_response.get("error", "")) == 0):
                        logger.warning(f"WORKFLOW: Agent {subtask.agent_name} returned empty error, possible truncation")
                        # Try to parse partial response or get more details
                        if response_size > 0 and response_size < 100:  # Very small response
                            logger.error(f"WORKFLOW: Response too small ({response_size} chars), likely truncated: {response_text}")
                            agent_response = {"error": f"Response truncated, received only {response_size} characters"}

                    logger.info(f"WORKFLOW: Agent {subtask.agent_name} response keys: {list(agent_response.keys())}")
                    logger.info(f"WORKFLOW: Agent {subtask.agent_name} success value: {agent_response.get('success')}")
                    logger.info(f"WORKFLOW: Agent {subtask.agent_name} has result: {'result' in agent_response}")

                # Enhanced response format normalization for varied agent formats
                output = normalize_agent_response(agent_response, subtask.agent_name)
                logger.info(f"WORKFLOW: Agent {subtask.agent_name} normalized output: {json.dumps(output)[:500]}...")
                logger.info(f"WORKFLOW: Agent {subtask.agent_name} normalized output keys: {list(output.keys())}")

                # Check if conversation history should be updated
                if "response" not in output:
                    logger.warning(f"Agent {subtask.agent_name} did not return a response - conversation history will not be updated")

                total_subtask_time = time.time() - subtask_start
                logger.info(f"Subtask {subtask.id} ({subtask.agent_name}) completed in {total_subtask_time:.3f}s")

                return output
            except httpx.ConnectError as connect_e:
                error_details = {
                    "error_type": "connection",
                    "agent": subtask.agent_name,
                    "endpoint_url": url,
                    "task_description": subtask.description[:200] + "..." if subtask.description and len(subtask.description) > 200 else subtask.description,
                    "error_message": str(connect_e)
                }
                # Import structured logger locally to avoid circular imports
                from orchestrator.main import structured_logger
                structured_logger.log_event("agent_connection_error", error_details)
                logger.error(f"WORKFLOW: Connection error calling agent {subtask.agent_name} at {url}: {connect_e}")
                output = {"error": f"Cannot connect to agent {subtask.agent_name} - service may be down"}
                total_subtask_time = time.time() - subtask_start
                logger.info(f"Subtask {subtask.id} ({subtask.agent_name}) connection failed in {total_subtask_time:.3f}s")
                return output
            except json.JSONDecodeError as json_e:
                error_details = {
                    "error_type": "json_parse",
                    "agent": subtask.agent_name,
                    "raw_response_preview": response.text[:500] + "..." if len(response.text) > 500 else response.text,
                    "response_length": len(response.text),
                    "task_description": subtask.description[:200] + "..." if subtask.description and len(subtask.description) > 200 else subtask.description,
                    "error_message": str(json_e)
                }
                # Import structured logger locally to avoid circular imports
                from orchestrator.main import structured_logger
                structured_logger.log_event("agent_json_parse_error", error_details)
                logger.error(f"WORKFLOW: JSON decode error from agent {subtask.agent_name}: {json_e}")
                logger.error(f"WORKFLOW: Raw response text: {response.text[:1000]}...")
                output = {"error": f"Agent {subtask.agent_name} returned invalid JSON response"}
                total_subtask_time = time.time() - subtask_start
                logger.info(f"Subtask {subtask.id} ({subtask.agent_name}) JSON error in {total_subtask_time:.3f}s")
                return output
            except httpx.HTTPStatusError as http_e:
                error_details = {
                    "error_type": "http_status",
                    "agent": subtask.agent_name,
                    "status_code": http_e.response.status_code,
                    "response_preview": http_e.response.text[:500] + "..." if len(http_e.response.text) > 500 else http_e.response.text,
                    "task_description": subtask.description[:200] + "..." if subtask.description and len(subtask.description) > 200 else subtask.description,
                    "error_message": str(http_e)
                }
                # Import structured logger locally to avoid circular imports
                from orchestrator.main import structured_logger
                structured_logger.log_event("agent_http_error", error_details)
                logger.error(f"WORKFLOW: HTTP error from agent {subtask.agent_name}: {http_e.response.status_code} - {http_e.response.text[:200]}...")
                output = {"error": f"Agent {subtask.agent_name} returned HTTP {http_e.response.status_code}"}
                total_subtask_time = time.time() - subtask_start
                logger.info(f"Subtask {subtask.id} ({subtask.agent_name}) HTTP error in {total_subtask_time:.3f}s")
                return output
            except Exception as e:
                error_details = {
                    "error_type": "unexpected",
                    "agent": subtask.agent_name,
                    "exception_type": type(e).__name__,
                    "task_description": subtask.description[:200] + "..." if subtask.description and len(subtask.description) > 200 else subtask.description,
                    "error_message": str(e),
                    "traceback": __import__('traceback').format_exc()
                }
                # Import structured logger locally to avoid circular imports
                from orchestrator.main import structured_logger
                structured_logger.log_event("agent_unexpected_error", error_details)
                logger.error(f"WORKFLOW: Unexpected error calling agent {subtask.agent_name}: {type(e).__name__}: {str(e)}")
                logger.error(f"WORKFLOW: Full traceback: {__import__('traceback').format_exc()}")
                output = {"error": f"Unexpected error from agent {subtask.agent_name}: {type(e).__name__}"}
                total_subtask_time = time.time() - subtask_start
                logger.info(f"Subtask {subtask.id} ({subtask.agent_name}) failed in {total_subtask_time:.3f}s: {type(e).__name__}")
                return output

async def aggregate_outputs(outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Transform outputs to frontend-expected format
    logger.info(f"WORKFLOW: Aggregating {len(outputs)} outputs")
    for i, output in enumerate(outputs):
        logger.info(f"WORKFLOW: Output {i} keys: {list(output.keys())}")
        logger.info(f"WORKFLOW: Output {i} has response: {'response' in output}")
        logger.info(f"WORKFLOW: Output {i} has error: {'error' in output}")
        if 'response' in output:
            logger.info(f"WORKFLOW: Output {i} response preview: {output['response'][:200]}...")

    frontend_output = {}

    # Collect ZIP download URLs from all outputs
    zip_urls = []
    for output in outputs:
        if "zip_download_url" in output:
            zip_urls.append(output["zip_download_url"])

    # Process each output and transform to frontend format
    all_files = []
    all_responses = []
    all_errors = []

    for output in outputs:
        # Extract files from structured responses
        if output.get("structured") and output["structured"].get("files"):
            files = output["structured"]["files"]
            for filename, content in files.items():
                all_files.append({
                    "filename": filename,
                    "content": content if isinstance(content, str) else json.dumps(content, indent=2),
                    "language": get_language_from_filename(filename)
                })

        # Collect responses
        if "response" in output:
            all_responses.append(output["response"])
        elif "result" in output:
            all_responses.append(str(output["result"]))

        # Collect errors
        if "error" in output:
            all_errors.append(output["error"])

    # Build frontend-compatible output
    if all_files:
        frontend_output["file_delivery"] = all_files
        # Add ZIP download if available
        if zip_urls:
            frontend_output["zip_download_url"] = zip_urls[0] if len(zip_urls) == 1 else zip_urls[0]
            frontend_output["zip_download_urls"] = zip_urls
            frontend_output["zip_filename"] = "project.zip"

    if all_responses:
        if len(all_responses) == 1:
            # Single response - check if it contains code
            response = all_responses[0]
            if "```" in response:
                # Extract code blocks
                import re
                code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', response, re.DOTALL)
                if code_blocks:
                    for lang, code in code_blocks:
                        frontend_output["inline_code"] = f"```{lang or 'python'}\n{code}\n```"
                        frontend_output["language"] = lang or "python"
                        break
            frontend_output["explanatory_summary"] = response
        else:
            # Multiple responses - combine them with more detail
            combined_responses = []
            for i, response in enumerate(all_responses, 1):
                combined_responses.append(f"Response {i}:\n{response}")
            frontend_output["explanatory_summary"] = "\n\n".join(combined_responses)

    if all_errors:
        frontend_output["explanatory_summary"] = f"Errors occurred:\n" + "\n".join(f"{error}" for error in all_errors)

    # For single output or simple responses, return directly if no transformation needed
    if len(outputs) == 1 and not frontend_output:
        output = outputs[0]
        if "response" in output:
            return transform_single_output_to_frontend(output)
        if "error" not in output:
            return transform_single_output_to_frontend(output)

    # If we have structured output, return it
    if frontend_output:
        return frontend_output

    # Fallback: Provide detailed aggregation without AI summarization
    detailed_outputs = []
    for i, output in enumerate(outputs, 1):
        output_desc = f"Output {i}:"
        if output.get("response"):
            output_desc += f"\n{output['response']}"
        elif output.get("error"):
            output_desc += f"\nError: {output['error']}"
        elif output.get("result"):
            output_desc += f"\nResult: {output['result']}"
        else:
            output_desc += f"\n{json.dumps(output, indent=2)}"
        detailed_outputs.append(output_desc)

    result = {"explanatory_summary": "\n\n".join(detailed_outputs)}
    if zip_urls:
        result["zip_download_urls"] = zip_urls
        result["zip_download_url"] = zip_urls[0] if zip_urls else None
    return result


def transform_single_output_to_frontend(output: Dict[str, Any]) -> Dict[str, Any]:
    """Transform a single agent output to frontend format"""
    frontend_output = {}

    # Handle structured files
    if output.get("structured") and output["structured"].get("files"):
        files = output["structured"]["files"]
        frontend_output["file_delivery"] = []
        for filename, content in files.items():
            frontend_output["file_delivery"].append({
                "filename": filename,
                "content": content if isinstance(content, str) else json.dumps(content, indent=2),
                "language": get_language_from_filename(filename)
            })

    # Handle response
    if "response" in output:
        response = output["response"]
        if "```" in response:
            # Extract code blocks
            import re
            code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', response, re.DOTALL)
            if code_blocks:
                for lang, code in code_blocks:
                    frontend_output["inline_code"] = f"```{lang or 'python'}\n{code}\n```"
                    frontend_output["language"] = lang or "python"
                    break
        frontend_output["explanatory_summary"] = response

    # Handle ZIP downloads
    if "zip_download_url" in output:
        frontend_output["zip_download_url"] = output["zip_download_url"]
        frontend_output["zip_filename"] = "project.zip"

    # Handle errors
    if "error" in output:
        frontend_output["explanatory_summary"] = f"**Error:** {output['error']}"

    return frontend_output if frontend_output else output


def transform_aggregated_to_frontend(aggregated: Dict[str, Any], zip_urls: List[str]) -> Dict[str, Any]:
    """Transform aggregated output to frontend format"""
    frontend_output = {}

    # Copy relevant fields
    if "response" in aggregated:
        frontend_output["explanatory_summary"] = aggregated["response"]
    elif "result" in aggregated:
        frontend_output["explanatory_summary"] = str(aggregated["result"])
    elif "summary" in aggregated:
        frontend_output["explanatory_summary"] = aggregated["summary"]

    # Add ZIP URLs
    if zip_urls:
        frontend_output["zip_download_urls"] = zip_urls
        frontend_output["zip_download_url"] = zip_urls[0] if zip_urls else None

    return frontend_output


def get_language_from_filename(filename: str) -> str:
    """Determine language from filename extension or filename"""
    # Handle files without extensions
    if '.' not in filename:
        # Special case files
        filename_lower = filename.lower()
        if filename_lower in ['dockerfile', 'makefile']:
            return 'dockerfile' if filename_lower == 'dockerfile' else 'makefile'
        elif filename_lower in ['readme', 'license', 'changelog', 'authors', 'contributors']:
            return 'text'
        elif filename_lower.startswith('license'):
            return 'text'
        else:
            return 'text'  # Default for unknown files without extensions

    ext = filename.split('.')[-1].lower()

    # Handle compound extensions
    if ext == 'jsx':
        return 'javascript'
    elif ext == 'tsx':
        return 'typescript'
    elif ext == 'scss' or ext == 'sass':
        return 'css'

    lang_map = {
        'py': 'python', 'js': 'javascript', 'ts': 'typescript', 'java': 'java',
        'cpp': 'cpp', 'c': 'c', 'go': 'go', 'rs': 'rust', 'php': 'php',
        'rb': 'ruby', 'html': 'html', 'css': 'css', 'json': 'json',
        'xml': 'xml', 'yaml': 'yaml', 'yml': 'yaml', 'md': 'markdown', 'txt': 'text',
        'sh': 'shell', 'bash': 'shell', 'zsh': 'shell', 'fish': 'shell',
        'sql': 'sql', 'csv': 'csv', 'tsv': 'tsv',
        'pdf': 'pdf', 'doc': 'word', 'docx': 'word', 'xls': 'excel', 'xlsx': 'excel',
        'zip': 'archive', 'tar': 'archive', 'gz': 'archive', 'bz2': 'archive',
        'png': 'image', 'jpg': 'image', 'jpeg': 'image', 'gif': 'image', 'svg': 'image',
        'mp4': 'video', 'avi': 'video', 'mkv': 'video', 'mov': 'video',
        'mp3': 'audio', 'wav': 'audio', 'flac': 'audio'
    }
    return lang_map.get(ext, 'text')  # Default to 'text' for unknown extensions

async def quality_gate(output: Dict[str, Any]) -> bool:
    # For simple responses, approve directly
    if "response" in output and len(output) == 1:
        return True

    # Sophisticated validation using NVIDIA NIM
    from providers.nim_adapter import NIMAdapter
    adapter = NIMAdapter()
    prompt = f"""
Evaluate the quality of this output. Check for:
- Completeness
- Correctness
- Code quality
- Security considerations

Output: {json.dumps(output)}

Respond with JSON: {{"quality_score": 0.0-1.0, "issues": ["list of issues"], "approved": true/false}}
"""
    messages = [{"role": "user", "content": prompt}]
    response = adapter.call_model(messages)
    try:
        result = json.loads(response.text)
        return result.get("approved", False)
    except:
        return False

async def trigger_comparator(outputs: List[Dict[str, Any]]):
    comparator_url = "http://localhost:8003/compare_candidates"
    async with httpx.AsyncClient() as client:
        # Format data for comparator service
        candidates = []
        for i, output in enumerate(outputs):
            candidates.append({
                "id": f"candidate_{i}",
                "patch": json.dumps(output),
                "provider": "agent",
                "repo_url": "dummy",
                "branch": "main"
            })
        await client.post(comparator_url, json={
            "candidates": candidates,
            "test_command": "echo 'test'",
            "repo_url": "dummy",
            "branch": "main"
        })

async def orchestrate_task_flow(task_id: str, manager=None):
    import time
    start_time = time.time()
    logger.info(f"Starting orchestration for task {task_id}")
    print(f"WORKFLOW: Starting orchestration for task {task_id}")
    try:
        # Get task
        task_retrieval_start = time.time()
        logger.info(f"WORKFLOW: Retrieving task {task_id}")
        async with async_session() as session:
            result = await session.execute(select(Task).where(Task.id == task_id))
            task = result.scalar_one()
        task_retrieval_time = time.time() - task_retrieval_start
        logger.info(f"Retrieved task {task_id}: {task.description} (took {task_retrieval_time:.3f}s)")

        # Plan
        planning_start = time.time()
        print(f"WORKFLOW: Planning task {task_id}")
        logger.info(f"WORKFLOW: Starting planning for task {task_id}")
        plan = await plan_task(task.description, task.context)
        planning_time = time.time() - planning_start
        print(f"WORKFLOW: Plan result: {plan}")
        task.plan = plan
        task.status = "planned"
        async with async_session() as session:
            await session.commit()
        logger.info(f"Task {task_id} planned with {len(plan.get('subtasks', []))} subtasks (took {planning_time:.3f}s)")
        print(f"WORKFLOW: Task {task_id} planned with {len(plan.get('subtasks', []))} subtasks")
        if manager:
            await manager.broadcast(json.dumps({"type": "status", "status": "planned", "progress": 25, "task_id": task_id}))

        # Route
        routing_start = time.time()
        logger.info(f"WORKFLOW: Starting routing for task {task_id}")
        routed_subtasks = await route_subtasks(plan, task.context)
        routing_time = time.time() - routing_start
        logger.info(f"Routed {len(routed_subtasks)} subtasks (took {routing_time:.3f}s)")
        if manager:
            await manager.broadcast(json.dumps({"type": "status", "status": "routed", "progress": 50, "task_id": task_id}))

        # Create subtasks
        db_creation_start = time.time()
        logger.info(f"WORKFLOW: Creating subtasks in DB for task {task_id}")
        subtasks = await create_subtasks_in_db(task_id, routed_subtasks)
        db_creation_time = time.time() - db_creation_start
        logger.info(f"Created {len(subtasks)} subtasks in DB (took {db_creation_time:.3f}s)")

        # Execute subtasks in parallel
        execution_start = time.time()
        logger.info(f"WORKFLOW: Starting execution of {len(subtasks)} subtasks")
        # Update task status to running
        async with async_session() as session:
            result = await session.execute(select(Task).where(Task.id == task_id))
            task = result.scalar_one()
            task.status = "running"
            await session.commit()
        execution_tasks = [execute_subtask(subtask, manager) for subtask in subtasks]
        outputs = await asyncio.gather(*execution_tasks)
        execution_time = time.time() - execution_start
        logger.info(f"Executed subtasks, got {len(outputs)} outputs (took {execution_time:.3f}s)")
        if manager:
            await manager.broadcast(json.dumps({"type": "status", "status": "executing", "progress": 75, "task_id": task_id}))

        # Update subtasks with outputs
        logger.info(f"WORKFLOW: Updating subtasks with outputs")
        async with async_session() as session:
            for subtask, output in zip(subtasks, outputs):
                # Update subtask in database
                result = await session.execute(
                    select(Subtask).where(Subtask.id == subtask.id)
                )
                db_subtask = result.scalar_one()
                db_subtask.output = output
                db_subtask.status = "completed"
                db_subtask.completed_at = func.now()
            await session.commit()
        logger.info(f"Updated subtasks with outputs")

        # Aggregate and validate
        logger.info(f"WORKFLOW: Starting aggregation for task {task_id}")
        final_output = await aggregate_outputs(outputs)
        logger.info(f"Aggregated output: {final_output}")

        logger.info(f"WORKFLOW: Starting quality gate for task {task_id}")
        validated = await quality_gate(final_output)
        logger.info(f"Quality gate result: {validated}")

        if not validated:
            # Trigger comparator if multiple patches
            if len(outputs) > 1:
                logger.info(f"Triggering comparator for task {task_id}")
                try:
                    await trigger_comparator(outputs)
                except Exception as comparator_error:
                    logger.warning(f"Comparator failed for task {task_id}: {comparator_error}")
                    # Continue with aggregated output even if comparator fails

        logger.info(f"WORKFLOW: Updating final task status and conversation history")
        async with async_session() as session:
            # Update task status and output
            result = await session.execute(select(Task).where(Task.id == task_id))
            task = result.scalar_one()

            # Update conversation history with the new interaction
            task_context = task.context or {}

            # Get existing conversation history
            conversation_history = task_context.get("conversation_history", [])

            # Add the user message (task description)
            user_message = {
                "type": "user",
                "content": task.description,
                "timestamp": time.time()
            }
            conversation_history.append(user_message)

            # Add the agent response
            if "explanatory_summary" in final_output:
                agent_response = {
                    "type": "assistant",
                    "content": final_output["explanatory_summary"],
                    "timestamp": time.time()
                }
                conversation_history.append(agent_response)
            else:
                logger.warning(f"No explanatory_summary in final_output, skipping conversation history update")

            # Keep only last 20 messages to prevent token limit issues
            logger.info(f"WORKFLOW: Conversation history before truncation: {len(conversation_history)} messages")
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]
                logger.info(f"WORKFLOW: Conversation history truncated to last 20 messages")

            # Update task context with conversation history
            task_context["conversation_history"] = conversation_history

            # Explicitly update the context to ensure SQLAlchemy tracks the change
            from sqlalchemy import update
            await session.execute(
                update(Task).where(Task.id == task_id).values(context=task_context)
            )

            task.status = "completed"
            task.output = final_output

            await session.commit()

        logger.info(f"Task {task_id} completed successfully with updated conversation history")

        logger.info(f"Task {task_id} completed successfully with updated conversation history")
        logger.info(f"Task {task_id} completed successfully")
        if manager:
            await manager.broadcast(json.dumps({"type": "status", "status": "completed", "progress": 100, "task_id": task_id}))
            # Broadcast the final output
            await manager.broadcast(json.dumps({"type": "output", "message": json.dumps([final_output]), "task_id": task_id}))

        return final_output
    except Exception as e:
        logger.error(f"Error in orchestrate_task_flow for task {task_id}: {e}", exc_info=True)
        # Update task status to failed
        try:
            async with async_session() as session:
                result = await session.execute(select(Task).where(Task.id == task_id))
                task = result.scalar_one()
                task.status = "failed"
                task.output = {"error": str(e)}
                await session.commit()
        except Exception as inner_e:
            logger.error(f"Failed to update task status to failed: {inner_e}")
        raise