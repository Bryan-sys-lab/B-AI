import json
import os
import requests
from .base_adapter import BaseAdapter, ModelResponse
from .model_registry import choose_model_for_role

class NIMAdapter(BaseAdapter):
    def __init__(self, role: str = "default"):
        # Determine default model for the role using the central registry
        model = choose_model_for_role(role)
        # Prefer the explicit NIM env var name used in the repo (.env) but
        # gracefully fall back to the older/alternate name if present.
        super().__init__("NVIDIA_NIM_API_KEY", "https://integrate.api.nvidia.com/v1/chat/completions", role=role)
        # If BaseAdapter didn't find the key under NVIDIA_NIM_API_KEY, try the
        # legacy NVIDIA_API_KEY environment variable and set it directly so
        # downstream code has a valid `self.api_key` value.
        if not getattr(self, "api_key", None):
            legacy = os.getenv("NVIDIA_API_KEY")
            if legacy:
                self.api_key = legacy

        self.default_model = model

        # Emit a masked info log so operators can verify which model/provider
        # the adapter will use without leaking secrets.
        masked_key = None
        if getattr(self, "api_key", None):
            k = self.api_key
            masked_key = (k[:4] + "..." + k[-4:]) if len(k) > 8 else "***"
        self.logger.info(
            "Adapter initialized",
            provider="NVIDIA NIM",
            model=self.default_model,
            api_key_present=bool(masked_key),
            api_key_masked=masked_key,
        )

    def _model_supports_tools(self, model_name: str) -> bool:
        """Check if a model supports tool calling."""
        # Based on testing, tools cause 400 Bad Request errors for all models
        # Disable tools for now to ensure complex tasks work
        return False

    def _call_api(self, messages, **kwargs):
        # Process messages to handle system role - NVIDIA API may not support system role
        processed_messages = []
        system_content = ""

        for msg in messages:
            if msg["role"] == "system":
                system_content += msg["content"] + "\n\n"
            else:
                if system_content and msg["role"] == "user":
                    # Prepend system content to first user message
                    msg = {"role": "user", "content": system_content + msg["content"]}
                    system_content = ""  # Clear after using
                processed_messages.append(msg)

        # If there are remaining system messages, add them as user messages
        if system_content:
            processed_messages.insert(0, {"role": "user", "content": system_content.strip()})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        model_name = kwargs.get("model", self.default_model)

        # Always use chat completions endpoint - most models support it
        endpoint = self.endpoint
        payload = {
            "model": model_name,
            "messages": processed_messages,
            "temperature": kwargs.get("temperature", 0.7),
        }
        # Only include tools if the model supports them and tools are provided
        if "tools" in kwargs and self._model_supports_tools(model_name):
            payload["tools"] = kwargs["tools"]
            self.logger.info(f"Including tools in payload for model {model_name}", role=self.role)
        elif "tools" in kwargs:
            self.logger.warning(f"Model {model_name} does not support tools, omitting from request", role=self.role)
            # Log the tools that were omitted for debugging
            self.logger.debug(f"Omitted tools: {kwargs['tools']}", role=self.role)

        # Use default SSL verification as per system security requirements
        # requests verifies SSL certificates by default (verify=True)
        # For testing/development, disable SSL verification if DISABLE_SSL_VERIFICATION is set
        verify_ssl = not os.getenv("DISABLE_SSL_VERIFICATION", "").lower() in ("true", "1", "yes")
        print(f"NIMAdapter: Making API call to {endpoint} with model {payload['model']}")
        try:
            response = requests.post(endpoint, headers=headers, json=payload, verify=verify_ssl, timeout=900.0)
            print(f"NIMAdapter: API call completed with status {response.status_code}")
            if response.status_code != 200:
                print(f"NIMAdapter: Error response body: {response.text}")
                # Log the full request for debugging
                self.logger.error(f"NIM API call failed with status {response.status_code}")
                self.logger.error(f"Request payload: {json.dumps(payload, indent=2)}")
                self.logger.error(f"Response headers: {dict(response.headers)}")
                self.logger.error(f"Response body: {response.text}")
            response.raise_for_status()
            response_json = response.json()
            # Log response size for debugging truncation issues
            response_text = response.text
            print(f"NIMAdapter: Raw response text size: {len(response_text)} characters")
            print(f"NIMAdapter: Response received: {json.dumps(response_json, indent=2)[:500]}...")
            # Log the full response for debugging
            self.logger.info(f"NIM API response size: {len(response_text)} characters")
            self.logger.info(f"NIM API response: {json.dumps(response_json, indent=2)[:1000]}...")

            # Check if the response contains an error even with 200 status
            if response_json.get("error"):
                print(f"NIMAdapter: API returned error in response: {response_json['error']}")
                self.logger.error(f"NIM API returned error: {response_json['error']}")
                raise Exception(f"NIM API error: {response_json['error']}")

            return response_json
        except requests.exceptions.SSLError as e:
            # SSL verification failed - this indicates a certificate issue
            self.logger.error(f"SSL certificate verification failed for {endpoint}: {e}")
            self.logger.error("This may indicate:")
            self.logger.error("1. SSL interception by a proxy/firewall")
            self.logger.error("2. Invalid or expired certificate")
            self.logger.error("3. Network configuration issues")
            self.logger.error("Please contact your network administrator or try a different network connection.")
            raise
        except Exception as e:
            # Re-raise non-SSL errors
            print(f"NIMAdapter: API call failed with exception: {e}")
            raise

    def _normalize_response(self, raw_response):
        print(f"NIMAdapter: Normalizing response: {json.dumps(raw_response, indent=2)[:1000]}...")
        choices = raw_response.get("choices", [])
        if not choices:
            print("NIMAdapter: No choices in response")
            return ModelResponse(text="", tokens=0, tool_calls=[], structured_response={}, confidence=0.0, latency_ms=0)

        choice = choices[0]
        print(f"NIMAdapter: Processing choice: {json.dumps(choice, indent=2)[:500]}...")

        # Handle different response formats for instruct vs chat models
        if "text" in choice:
            # Completions format (instruct models)
            text = choice["text"]
            tool_calls = []
        else:
            # Chat format
            message = choice.get("message", {})
            text = message.get("content", "")
            tool_calls = message.get("tool_calls", [])

        print(f"NIMAdapter: Extracted text: '{text[:200]}...'")
        print(f"NIMAdapter: Full text: '{text}'")

        # Check for empty text response - this indicates an API failure
        if not text or text.strip() == "":
            print("NIMAdapter: Received empty text response from API")
            self.logger.error("NIM API returned empty response")
            raise Exception("NIM API returned empty response - model may be unavailable or rate limited")

        usage = raw_response.get("usage", {})
        tokens = usage.get("total_tokens", 0)

        try:
            structured = json.loads(text)
            print(f"NIMAdapter: Parsed structured response: {json.dumps(structured, indent=2)[:500]}...")

            # Check if the structured response indicates an error
            if isinstance(structured, list) and len(structured) > 0:
                first_item = structured[0]
                if isinstance(first_item, dict) and "error" in first_item:
                    error_msg = first_item.get("error", "")
                    print(f"NIMAdapter: Detected error in structured response: '{error_msg}'")
                    self.logger.error(f"NIM model returned error: {error_msg}")
                    # Raise an exception to be caught by the caller
                    raise Exception(f"NIM model error: {error_msg}")

            # Also check if it's a dict with error
            elif isinstance(structured, dict) and "error" in structured:
                error_msg = structured.get("error", "")
                print(f"NIMAdapter: Detected error in structured response: '{error_msg}'")
                self.logger.error(f"NIM model returned error: {error_msg}")
                raise Exception(f"NIM model error: {error_msg}")

        except json.JSONDecodeError as e:
            print(f"NIMAdapter: Failed to parse structured response: {e}")
            # If the response isn't valid JSON, fall back to empty structured response
            structured = {}
        except Exception as e:
            # Re-raise exceptions we intentionally raised above
            if "NIM model error" in str(e):
                raise
            print(f"NIMAdapter: Unexpected error during structured parsing: {e}")
            structured = {}

        # Additional check: if text itself indicates an error
        if text.strip() == '[{"error": ""}]' or text.strip() == '{"error": ""}':
            print(f"NIMAdapter: Detected error in raw text response: '{text}'")
            self.logger.error(f"NIM model returned error in text: {text}")
            raise Exception(f"NIM model error: {text}")

        confidence = 1.0  # placeholder
        return ModelResponse(text=text, tokens=tokens, tool_calls=tool_calls, structured_response=structured, confidence=confidence, latency_ms=0)

    def get_available_models(self):
        """Fetch the list of available models from NVIDIA API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # For testing/development, disable SSL verification if DISABLE_SSL_VERIFICATION is set
        verify_ssl = not os.getenv("DISABLE_SSL_VERIFICATION", "").lower() in ("true", "1", "yes")
        try:
            # Use default SSL verification as per system security requirements
            response = requests.get("https://integrate.api.nvidia.com/v1/models", headers=headers, verify=verify_ssl)
            response.raise_for_status()
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        except requests.exceptions.SSLError as e:
            self.logger.warning(f"SSL certificate verification failed for models endpoint: {e}")
            self.logger.warning("Model fetching unavailable due to SSL configuration")
            return []
        except Exception as e:
            self.logger.warning(f"Failed to fetch available models: {e}")
            return []

    def _estimate_cost(self, tokens):
        # NVIDIA NIM pricing approximation
        return tokens * 0.000005