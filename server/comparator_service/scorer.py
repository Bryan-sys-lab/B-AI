def calculate_score(evaluation_result):
    """
    Calculate overall score based on evaluation metrics.
    Higher score is better.
    """
    pass_rate = evaluation_result.pass_rate  # 0-1, higher better
    coverage_delta = evaluation_result.coverage_delta  # e.g., -1 to 1, higher better
    lint_score = evaluation_result.lint_score  # 0-1, lower better (normalized issues)
    security_risk = evaluation_result.security_risk  # 0-1, lower better
    performance_impact = evaluation_result.performance_impact  # 0-1, lower better (normalized slowdown)

    # Weights (can be adjusted)
    weights = {
        "pass_rate": 0.3,
        "coverage_delta": 0.2,
        "lint_score": 0.2,
        "security_risk": 0.15,
        "performance_impact": 0.15
    }

    # Invert negative metrics (lower is better -> higher is better)
    inverted_lint = 1 - lint_score
    inverted_security = 1 - security_risk
    inverted_performance = 1 - performance_impact

    score = (
        weights["pass_rate"] * pass_rate +
        weights["coverage_delta"] * coverage_delta +
        weights["lint_score"] * inverted_lint +
        weights["security_risk"] * inverted_security +
        weights["performance_impact"] * inverted_performance
    )

    # Clamp score to [0, 1] range
    return max(0.0, min(1.0, score))