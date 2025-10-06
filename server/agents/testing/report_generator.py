import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

class TestSummary(BaseModel):
    total_tests: int
    passed: int
    failed: int
    errors: int
    skipped: int
    coverage: Optional[float] = None

class TestReport(BaseModel):
    timestamp: datetime
    code_path: str
    language: str
    unit_tests: TestSummary
    integration_tests: TestSummary
    fuzz_tests: TestSummary
    performance_tests: TestSummary
    ui_tests: TestSummary
    load_tests: Dict[str, Any]
    overall_status: str
    recommendations: List[str]
    detailed_results: Dict[str, Any]

class ReportGenerator:
    def generate_comprehensive_report(
        self,
        code_path: str,
        language: str,
        unit_results: Optional[Any] = None,
        integration_results: Optional[Any] = None,
        fuzz_results: Optional[Any] = None,
        performance_results: Optional[Any] = None,
        ui_results: Optional[Any] = None,
        load_results: Optional[Any] = None
    ) -> TestReport:
        """Generate a comprehensive test report."""

        # Create summaries
        unit_summary = self._create_summary(unit_results, "unit")
        integration_summary = self._create_summary(integration_results, "integration")
        fuzz_summary = self._create_summary(fuzz_results, "fuzz")
        performance_summary = self._create_summary(performance_results, "performance")
        ui_summary = self._create_summary(ui_results, "ui")

        # Calculate overall status
        overall_status = self._calculate_overall_status([
            unit_summary, integration_summary, fuzz_summary,
            performance_summary, ui_summary
        ])

        # Generate recommendations
        recommendations = self._generate_recommendations(
            unit_summary, integration_summary, fuzz_summary,
            performance_summary, ui_summary, load_results
        )

        # Collect detailed results
        detailed_results = {
            "unit_tests": self._extract_details(unit_results),
            "integration_tests": self._extract_details(integration_results),
            "fuzz_tests": self._extract_details(fuzz_results),
            "performance_tests": self._extract_details(performance_results),
            "ui_tests": self._extract_details(ui_results),
            "load_tests": self._extract_load_details(load_results)
        }

        return TestReport(
            timestamp=datetime.now(),
            code_path=code_path,
            language=language,
            unit_tests=unit_summary,
            integration_tests=integration_summary,
            fuzz_tests=fuzz_summary,
            performance_tests=performance_summary,
            ui_tests=ui_summary,
            load_tests=self._extract_load_details(load_results),
            overall_status=overall_status,
            recommendations=recommendations,
            detailed_results=detailed_results
        )

    def _create_summary(self, results: Optional[Any], test_type: str) -> TestSummary:
        """Create a test summary from results."""
        if not results:
            return TestSummary(total_tests=0, passed=0, failed=0, errors=0, skipped=0)

        # Handle different result types
        if hasattr(results, 'passed') and hasattr(results, 'failed'):
            # TestResult-like objects
            return TestSummary(
                total_tests=getattr(results, 'total', 0),
                passed=getattr(results, 'passed', 0),
                failed=getattr(results, 'failed', 0),
                errors=getattr(results, 'errors', 0),
                skipped=getattr(results, 'skipped', 0),
                coverage=getattr(results, 'coverage', None)
            )
        elif hasattr(results, 'total_requests'):
            # LoadTestResult
            return TestSummary(
                total_tests=1,  # Load test is one test
                passed=1 if getattr(results, 'failures', 0) == 0 else 0,
                failed=1 if getattr(results, 'failures', 0) > 0 else 0,
                errors=0,
                skipped=0
            )
        else:
            return TestSummary(total_tests=0, passed=0, failed=0, errors=0, skipped=0)

    def _calculate_overall_status(self, summaries: List[TestSummary]) -> str:
        """Calculate overall test status."""
        total_passed = sum(s.passed for s in summaries)
        total_failed = sum(s.failed for s in summaries)
        total_errors = sum(s.errors for s in summaries)

        if total_failed > 0 or total_errors > 0:
            return "failed"
        elif total_passed > 0:
            return "passed"
        else:
            return "no_tests"

    def _generate_recommendations(
        self,
        unit: TestSummary,
        integration: TestSummary,
        fuzz: TestSummary,
        performance: TestSummary,
        ui: TestSummary,
        load_results: Optional[Any]
    ) -> List[str]:
        """Generate test recommendations."""
        recommendations = []

        if unit.failed > 0:
            recommendations.append("Fix failing unit tests to ensure code correctness")
        if unit.coverage and unit.coverage < 80:
            recommendations.append(f"Improve unit test coverage (currently {unit.coverage:.1f}%)")

        if integration.failed > 0:
            recommendations.append("Address integration test failures for component interaction issues")

        if fuzz.failed > 0:
            recommendations.append("Review fuzz test failures for potential security vulnerabilities")

        if performance.failed > 0:
            recommendations.append("Optimize performance bottlenecks identified in tests")

        if ui.failed > 0:
            recommendations.append("Fix UI test failures to ensure proper user experience")

        if load_results and getattr(load_results, 'failures', 0) > 0:
            recommendations.append("Improve system scalability and error handling under load")

        if not recommendations:
            recommendations.append("All tests passed - consider adding more comprehensive test coverage")

        return recommendations

    def _extract_details(self, results: Optional[Any]) -> Dict[str, Any]:
        """Extract detailed results."""
        if not results:
            return {}

        details = {}
        for attr in ['passed', 'failed', 'errors', 'skipped', 'total', 'coverage', 'stdout', 'stderr', 'exit_code']:
            if hasattr(results, attr):
                details[attr] = getattr(results, attr)

        return details

    def _extract_load_details(self, results: Optional[Any]) -> Dict[str, Any]:
        """Extract load test details."""
        if not results:
            return {}

        details = {}
        for attr in ['total_requests', 'requests_per_second', 'response_time_avg',
                     'response_time_95p', 'response_time_99p', 'failures', 'stdout', 'stderr', 'exit_code']:
            if hasattr(results, attr):
                details[attr] = getattr(results, attr)

        return details

    def export_report(self, report: TestReport, format: str = "json") -> str:
        """Export report in specified format."""
        if format == "json":
            return report.json(indent=2)
        elif format == "dict":
            return report.dict()
        else:
            return str(report)