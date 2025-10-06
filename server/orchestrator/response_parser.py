"""
Robust Aetherium Response Parsing Utility

Handles various response formats from Aetherium models:
- Raw JSON
- Markdown code blocks (```json)
- Mixed content with embedded JSON
- Malformed responses with graceful fallback
"""

import json
import re
from typing import Dict, Any, Optional, Tuple, List, Callable


def parse_ai_response(response_text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Robustly parse Aetherium responses that may come in different formats.

    Returns:
        Tuple of (parsed_data, parse_method_used)
        parsed_data is None if all parsing strategies fail
    """

    # Strategy 1: Try raw JSON first (most common, fastest)
    try:
        result = json.loads(response_text.strip())
        return result, "raw_json"
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code blocks
    json_patterns = [
        r'```json\s*(.*?)\s*```',  # ```json ... ```
        r'```\s*(.*?)\s*```',      # Generic code block
        r'```\w*\s*(.*?)\s*```',   # Language-specific code block
    ]

    for pattern_name, pattern in [("json_block", json_patterns[0]),
                                  ("generic_block", json_patterns[1]),
                                  ("lang_block", json_patterns[2])]:
        matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                result = json.loads(match.strip())
                return result, f"{pattern_name}_extracted"
            except json.JSONDecodeError:
                continue

    # Strategy 3: Extract JSON-like content using regex
    # Look for {...} or [...] patterns
    json_like_patterns = [
        (r'^\s*(\{.*?\})\s*$', "object_braces"),  # Object at start/end of string
        (r'^\s*(\[.*?\])\s*$', "array_brackets"), # Array at start/end of string
        (r'(\{[^{}]*\{[^{}]*\}[^{}]*\})', "nested_object"),  # Nested objects
    ]

    for pattern, method_name in json_like_patterns:
        matches = re.findall(pattern, response_text, re.MULTILINE | re.DOTALL)
        for match in matches:
            try:
                result = json.loads(match.strip())
                return result, f"{method_name}_extracted"
            except json.JSONDecodeError:
                continue

    # Strategy 4: Clean and retry (remove common contaminants)
    cleaned_text = re.sub(r'[`\*\[\]]', '', response_text)  # Remove markdown chars
    cleaned_text = re.sub(r'^\s*json\s*', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'^\s*python\s*', '', cleaned_text, flags=re.IGNORECASE)

    try:
        result = json.loads(cleaned_text.strip())
        return result, "cleaned_text"
    except json.JSONDecodeError:
        pass

    # Strategy 5: Extract JSON from middle of text
    # Look for balanced braces/brackets anywhere in text
    def extract_balanced(text: str, start_char: str, end_char: str) -> Optional[str]:
        """Extract balanced brackets/braces from text"""
        start_idx = text.find(start_char)
        if start_idx == -1:
            return None

        count = 0
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == start_char:
                count += 1
            elif char == end_char:
                count -= 1
                if count == 0:
                    return text[start_idx:i+1]
        return None

    # Try to extract object
    obj_extract = extract_balanced(response_text, '{', '}')
    if obj_extract:
        try:
            result = json.loads(obj_extract)
            return result, "balanced_braces_extracted"
        except json.JSONDecodeError:
            pass

    # Try to extract array
    arr_extract = extract_balanced(response_text, '[', ']')
    if arr_extract:
        try:
            result = json.loads(arr_extract)
            return result, "balanced_brackets_extracted"
        except json.JSONDecodeError:
            pass

    return None, "failed_all_strategies"


def validate_plan_structure(plan: Dict[str, Any]) -> bool:
    """Validate that a parsed plan has the required structure"""
    if not isinstance(plan, dict) or 'subtasks' not in plan:
        return False

    subtasks = plan['subtasks']
    if not isinstance(subtasks, list) or len(subtasks) == 0:
        return False

    required_fields = ['description', 'agent', 'priority', 'confidence']
    for subtask in subtasks:
        if not isinstance(subtask, dict):
            return False
        if not all(field in subtask for field in required_fields):
            return False
        # Validate field types
        if not isinstance(subtask['description'], str) or not subtask['description'].strip():
            return False
        if not isinstance(subtask['agent'], str) or not subtask['agent'].strip():
            return False
        if not isinstance(subtask['priority'], (int, float)) or not (1 <= subtask['priority'] <= 10):
            return False
        if not isinstance(subtask['confidence'], (int, float)) or not (0.0 <= subtask['confidence'] <= 1.0):
            return False

    return True


def validate_routing_structure(routing: Dict[str, Any]) -> bool:
    """Validate that parsed routing has the required structure"""
    if not isinstance(routing, list) or len(routing) == 0:
        return False

    required_fields = ['description', 'agent', 'confidence', 'priority']
    for route in routing:
        if not isinstance(route, dict):
            return False
        if not all(field in route for field in required_fields):
            return False
        # Validate field types
        if not isinstance(route['description'], str) or not route['description'].strip():
            return False
        if not isinstance(route['agent'], str) or not route['agent'].strip():
            return False
        if not isinstance(route['priority'], (int, float)) or not (1 <= route['priority'] <= 10):
            return False
        if not isinstance(route['confidence'], (int, float)) or not (0.0 <= route['confidence'] <= 1.0):
            return False

    return True


def validate_quality_gate_structure(gate_result: Dict[str, Any]) -> bool:
    """Validate quality gate response structure"""
    if not isinstance(gate_result, dict):
        return False

    # Should have approved field
    if 'approved' not in gate_result:
        return False

    # May have additional fields like quality_score, issues
    return True


def safe_parse_with_validation(response_text: str, validator_func: Optional[Callable[[Dict[str, Any]], bool]] = None) -> Tuple[Optional[Dict[str, Any]], str, bool]:
    """
    Parse Aetherium response with optional validation.

    Returns:
        Tuple of (parsed_data, parse_method, is_valid)
    """
    parsed, method = parse_ai_response(response_text)

    if parsed is None:
        return None, method, False

    if validator_func:
        is_valid = validator_func(parsed)
    else:
        is_valid = True  # No validation requested

    return parsed, method, is_valid