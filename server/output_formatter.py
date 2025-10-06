"""
Output Formatter for Aetherium Coding Results

This module provides standardized formatting for Aetherium-generated coding task results
according to predefined output modes.
"""

import json
from typing import Dict, List, Optional, Union, Any


class OutputFormatter:
    """
    Formats Aetherium coding results into structured, user-friendly outputs.
    """

    def format_response(self, modes: List[str], **kwargs) -> Dict[str, Any]:
        """
        Main method to format response based on selected modes.

        Args:
            modes: List of mode names to use
            **kwargs: Arguments for each mode

        Returns:
            Dict containing formatted outputs for each mode
        """
        result = {}

        for mode in modes:
            if mode == "inline_code":
                result["inline_code"] = self._format_inline_code(
                    kwargs.get("code", ""),
                    kwargs.get("language", "python")
                )
            elif mode == "executed_results":
                result["executed_results"] = self._format_executed_results(
                    kwargs.get("code", ""),
                    kwargs.get("execution_output", "")
                )
            elif mode == "file_delivery":
                result["file_delivery"] = self._format_file_delivery(
                    kwargs.get("file_content", ""),
                    kwargs.get("filename", ""),
                    kwargs.get("extension", "")
                )
            elif mode == "canvas_editor":
                result["canvas_editor"] = self._format_canvas_editor(
                    kwargs.get("full_code", "")
                )
            elif mode == "visual_output":
                result["visual_output"] = self._format_visual_output(
                    kwargs.get("visual_description", ""),
                    kwargs.get("fallback_image", None),
                    kwargs.get("interactive", False)
                )
            elif mode == "explanatory_summary":
                result["explanatory_summary"] = self._format_explanatory_summary(
                    kwargs.get("summary_text", ""),
                    kwargs.get("assumptions", []),
                    kwargs.get("limitations", []),
                    kwargs.get("improvements", [])
                )

        return result

    def _format_inline_code(self, code: str, language: str) -> str:
        """
        Format short code snippets with syntax highlighting.
        """
        return f"```{language}\n{code}\n```"

    def _format_executed_results(self, code: str, execution_output: str, language: Optional[str] = None) -> Dict[str, str]:
        """
        Format code with its execution results.
        """
        detected_language = language or self._detect_language(code)
        return {
            "code": self._format_inline_code(code, detected_language),
            "execution_output": f"Execution Output:\n{execution_output}",
            "language": detected_language
        }

    def _format_file_delivery(self, file_content: str, filename: str, extension: str) -> Dict[str, str]:
        """
        Format large deliverables as downloadable files.
        """
        # Validate file content and extension
        validated_extension = self._validate_file_extension(file_content, extension)
        return {
            "filename": filename or f"output.{validated_extension}",
            "content": file_content,
            "extension": validated_extension
        }

    def _format_canvas_editor(self, full_code: str) -> str:
        """
        Format entire files or projects for editing.
        """
        return full_code

    def _detect_language(self, code: str) -> str:
        """
        Automatically detect programming language from code content.
        """
        code_lower = code.lower().strip()

        # Python indicators
        if any(keyword in code_lower for keyword in ['def ', 'import ', 'print(', 'if __name__']):
            return 'python'
        if 'from ' in code_lower and 'import ' in code_lower:
            return 'python'

        # JavaScript indicators
        if any(keyword in code_lower for keyword in ['function ', 'const ', 'let ', 'var ', 'console.log']):
            return 'javascript'
        if '=>' in code_lower or 'async ' in code_lower:
            return 'javascript'

        # Java indicators
        if any(keyword in code_lower for keyword in ['public class', 'system.out.println', 'import java.']):
            return 'java'

        # C++ indicators
        if any(keyword in code_lower for keyword in ['#include', 'cout <<', 'std::']):
            return 'cpp'

        # C# indicators
        if any(keyword in code_lower for keyword in ['using system;', 'console.writeline', 'public static void']):
            return 'csharp'

        # Go indicators
        if any(keyword in code_lower for keyword in ['package main', 'func ', 'fmt.println']):
            return 'go'

        # Rust indicators
        if any(keyword in code_lower for keyword in ['fn main', 'println!', 'use std::']):
            return 'rust'

        # TypeScript indicators
        if any(keyword in code_lower for keyword in ['interface ', ': string', ': number', 'typescript']):
            return 'typescript'

        # PHP indicators
        if any(keyword in code_lower for keyword in ['<?php', 'echo ', '$', 'function ']):
            return 'php'

        # Ruby indicators
        if any(keyword in code_lower for keyword in ['def ', 'puts ', 'require ', 'class ']):
            return 'ruby'

        # Default to python if no clear indicators
        return 'python'

    def _validate_file_extension(self, content: str, extension: str) -> str:
        """
        Validate and suggest appropriate file extension based on content.
        """
        if not extension:
            # Auto-detect based on content
            if content.startswith('<?xml') or content.startswith('<'):
                return 'xml'
            elif content.startswith('import ') or 'def ' in content:
                return 'py'
            elif 'function' in content or 'const ' in content:
                return 'js'
            elif 'public class' in content:
                return 'java'
            elif '#include' in content:
                return 'cpp'
            elif content.startswith('package ') and 'func ' in content:
                return 'go'
            elif 'fn ' in content and 'println!' in content:
                return 'rs'
            else:
                return 'txt'

        # Validate provided extension matches content
        extension_lower = extension.lower()

        # Check for mismatches
        if extension_lower == 'py' and not ('import ' in content or 'def ' in content):
            # Might be wrong, but allow it
            pass
        elif extension_lower == 'js' and not ('function' in content or 'const ' in content):
            pass
        # Add more validation as needed

        return extension

    def _format_visual_output(self, visual_description: str, fallback_image: Optional[str] = None, interactive: bool = False) -> Dict[str, Any]:
        """
        Format visual outputs like charts or diagrams.
        """
        result = {
            "description": visual_description,
            "interactive": interactive
        }
        if fallback_image:
            result["fallback_image"] = fallback_image
        return result

    def _format_explanatory_summary(self, summary_text: str,
                                   assumptions: List[str] = None,
                                   limitations: List[str] = None,
                                   improvements: List[str] = None) -> str:
        """
        Format explanatory summary with assumptions, limitations, and improvements.
        """
        summary = summary_text

        if assumptions:
            summary += "\n\nAssumptions:\n" + "\n".join(f"- {a}" for a in assumptions)

        if limitations:
            summary += "\n\nLimitations:\n" + "\n".join(f"- {l}" for l in limitations)

        if improvements:
            summary += "\n\nPotential Improvements:\n" + "\n".join(f"- {i}" for i in improvements)

        return summary

    def decide_modes(self, task_complexity: str, has_execution: bool = False,
                    is_large_deliverable: bool = False, has_visual: bool = False) -> List[str]:
        """
        Automatically decide which output modes to use based on task characteristics.

        Args:
            task_complexity: "small", "medium", "large"
            has_execution: Whether code produces meaningful output
            is_large_deliverable: Whether result is a large file
            has_visual: Whether result includes visual elements

        Returns:
            List of recommended modes
        """
        modes = ["explanatory_summary"]  # Always include

        if task_complexity == "small":
            modes.append("inline_code")
        elif task_complexity == "large":
            modes.append("canvas_editor")
        else:  # medium
            modes.append("inline_code")

        if has_execution:
            modes.append("executed_results")

        if is_large_deliverable:
            modes.append("file_delivery")

        if has_visual:
            modes.append("visual_output")

        return modes


# Example usage
if __name__ == "__main__":
    formatter = OutputFormatter()

    # Example: Fibonacci function
    modes = ["inline_code", "executed_results", "explanatory_summary"]
    result = formatter.format_response(
        modes=modes,
        code="""def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print(fibonacci(10))""",
        execution_output="55",
        summary_text="This function uses an iterative loop to compute the nth Fibonacci number.",
        assumptions=["n is a non-negative integer"],
        limitations=["May be slow for very large n due to linear time complexity"],
        improvements=["Use matrix exponentiation for O(log n) performance"]
    )

    print(json.dumps(result, indent=2))