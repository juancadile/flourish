"""Multi-judge comparison and bias analysis for Flourish evaluations.

Inspired by Bloom's judge validation approach (Figure 4 in their paper).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from collections import defaultdict

try:
    from scipy.stats import spearmanr
except ImportError:
    # Fallback if scipy not installed
    def spearmanr(a, b):
        """Fallback Spearman correlation using pandas."""
        import pandas as pd
        df = pd.DataFrame({'a': a, 'b': b})
        return df.corr(method='spearman').iloc[0, 1], 0


def calculate_inter_judge_agreement(
    results_df: pd.DataFrame,
    judges: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calculate pairwise Spearman correlation between judges.

    Args:
        results_df: DataFrame with columns: scenario_id, model, judge, score
        judges: List of judge models to compare (None = all judges)

    Returns:
        Dictionary mapping judge pairs to correlation coefficients.
    """
    if judges is None:
        judges = results_df["judge"].unique().tolist()

    if len(judges) < 2:
        return {}

    correlations = {}

    for i, judge1 in enumerate(judges):
        for judge2 in judges[i + 1:]:
            # Get scores from both judges for the same scenarios/models
            df1 = results_df[results_df["judge"] == judge1][["scenario_id", "model", "score"]]
            df2 = results_df[results_df["judge"] == judge2][["scenario_id", "model", "score"]]

            # Merge on scenario_id and model
            merged = df1.merge(
                df2,
                on=["scenario_id", "model"],
                suffixes=("_1", "_2")
            )

            if len(merged) == 0:
                continue

            # Calculate Spearman correlation
            valid = merged.dropna(subset=["score_1", "score_2"])
            if len(valid) >= 2:
                corr, _ = spearmanr(valid["score_1"], valid["score_2"])
                correlations[f"{judge1} vs {judge2}"] = corr

    return correlations


def calculate_agreement_at_extremes(
    results_df: pd.DataFrame,
    judges: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate agreement rates at score extremes (0 and 2).

    Args:
        results_df: DataFrame with columns: scenario_id, model, judge, score
        judges: List of judge models to compare (None = all judges)

    Returns:
        Dictionary with agreement rates for each judge pair.
    """
    if judges is None:
        judges = results_df["judge"].unique().tolist()

    if len(judges) < 2:
        return {}

    agreements = {}

    for i, judge1 in enumerate(judges):
        for judge2 in judges[i + 1:]:
            df1 = results_df[results_df["judge"] == judge1][["scenario_id", "model", "score"]]
            df2 = results_df[results_df["judge"] == judge2][["scenario_id", "model", "score"]]

            merged = df1.merge(
                df2,
                on=["scenario_id", "model"],
                suffixes=("_1", "_2")
            )

            if len(merged) == 0:
                continue

            valid = merged.dropna(subset=["score_1", "score_2"])

            # Agreement at score 0
            score_0 = valid[(valid["score_1"] == 0) | (valid["score_2"] == 0)]
            agree_0 = (score_0["score_1"] == score_0["score_2"]).mean() if len(score_0) > 0 else 0

            # Agreement at score 2
            score_2 = valid[(valid["score_1"] == 2) | (valid["score_2"] == 2)]
            agree_2 = (score_2["score_1"] == score_2["score_2"]).mean() if len(score_2) > 0 else 0

            # Overall agreement (exact match)
            overall_agreement = (valid["score_1"] == valid["score_2"]).mean()

            agreements[f"{judge1} vs {judge2}"] = {
                "overall": overall_agreement,
                "at_score_0": agree_0,
                "at_score_2": agree_2,
            }

    return agreements


def detect_systematic_bias(
    results_df: pd.DataFrame,
    judges: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Detect systematic bias in judge scoring.

    Args:
        results_df: DataFrame with columns: scenario_id, model, judge, score
        judges: List of judge models to analyze (None = all judges)

    Returns:
        Dictionary with bias metrics for each judge.
    """
    if judges is None:
        judges = results_df["judge"].unique().tolist()

    bias_metrics = {}

    for judge in judges:
        judge_df = results_df[results_df["judge"] == judge]

        # Calculate average scores per model
        model_scores = judge_df.groupby("model")["score"].agg(["mean", "count"]).to_dict("index")

        # Overall statistics
        all_scores = judge_df["score"].dropna()
        avg_score = all_scores.mean() if len(all_scores) > 0 else 0
        score_std = all_scores.std() if len(all_scores) > 0 else 0

        # Score distribution
        score_dist = {
            "score_0_pct": (all_scores == 0).mean() * 100 if len(all_scores) > 0 else 0,
            "score_1_pct": (all_scores == 1).mean() * 100 if len(all_scores) > 0 else 0,
            "score_2_pct": (all_scores == 2).mean() * 100 if len(all_scores) > 0 else 0,
        }

        bias_metrics[judge] = {
            "avg_score": avg_score,
            "score_std": score_std,
            "model_favoritism": model_scores,
            **score_dist,
        }

    return bias_metrics


def create_correlation_matrix(
    results_df: pd.DataFrame,
    judges: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create correlation matrix between all judges.

    Args:
        results_df: DataFrame with columns: scenario_id, model, judge, score
        judges: List of judge models (None = all judges)

    Returns:
        DataFrame with correlation matrix.
    """
    if judges is None:
        judges = sorted(results_df["judge"].unique().tolist())

    if len(judges) < 2:
        return pd.DataFrame()

    # Create pivot table: rows = (scenario_id, model), columns = judge
    pivot = results_df.pivot_table(
        values="score",
        index=["scenario_id", "model"],
        columns="judge",
        aggfunc="first"
    )

    # Calculate correlation matrix
    corr_matrix = pivot.corr(method="spearman")

    return corr_matrix


def generate_judge_comparison_report(
    results_df: pd.DataFrame,
    judges: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate comprehensive judge comparison report.

    Args:
        results_df: DataFrame with evaluation results from multiple judges
        judges: List of judge models to analyze
        output_path: Optional path to save the report

    Returns:
        Markdown report string
    """
    if judges is None:
        judges = sorted(results_df["judge"].unique().tolist())

    # Calculate metrics
    correlations = calculate_inter_judge_agreement(results_df, judges)
    agreements = calculate_agreement_at_extremes(results_df, judges)
    bias_metrics = detect_systematic_bias(results_df, judges)
    corr_matrix = create_correlation_matrix(results_df, judges)

    # Build report
    report = f"""# Judge Comparison Report

## Overview

- **Judges Evaluated**: {len(judges)}
- **Total Evaluations**: {len(results_df)}
- **Models**: {results_df['model'].nunique()}
- **Scenarios**: {results_df['scenario_id'].nunique()}
- **Virtues**: {results_df['virtue'].nunique() if 'virtue' in results_df.columns else 'N/A'}

## Inter-Judge Correlation (Spearman)

Measures how consistently judges score the same scenarios.

"""

    # Add correlation table
    if not corr_matrix.empty:
        report += "### Correlation Matrix\n\n"
        report += corr_matrix.to_markdown() + "\n\n"

    # Pairwise correlations
    report += "### Pairwise Correlations\n\n"
    for pair, corr in sorted(correlations.items(), key=lambda x: x[1], reverse=True):
        report += f"- **{pair}**: {corr:.3f}\n"

    report += "\n## Agreement Rates\n\n"
    report += "Percentage of time judges assign the same score.\n\n"

    for pair, metrics in agreements.items():
        report += f"### {pair}\n\n"
        report += f"- Overall agreement: {metrics['overall']:.1%}\n"
        report += f"- Agreement at score 0: {metrics['at_score_0']:.1%}\n"
        report += f"- Agreement at score 2: {metrics['at_score_2']:.1%}\n\n"

    report += "## Judge Bias Analysis\n\n"
    report += "Systematic tendencies in judge scoring patterns.\n\n"

    for judge, metrics in bias_metrics.items():
        report += f"### {judge}\n\n"
        report += f"- **Average score**: {metrics['avg_score']:.2f}/2.00\n"
        report += f"- **Score std dev**: {metrics['score_std']:.2f}\n"
        report += f"- **Score distribution**:\n"
        report += f"  - 0: {metrics['score_0_pct']:.1f}%\n"
        report += f"  - 1: {metrics['score_1_pct']:.1f}%\n"
        report += f"  - 2: {metrics['score_2_pct']:.1f}%\n"

        # Model favoritism
        if metrics["model_favoritism"]:
            report += f"\n**Model-specific scores**:\n"
            for model, stats in metrics["model_favoritism"].items():
                report += f"  - {model}: {stats['mean']:.2f} (n={int(stats['count'])})\n"

        report += "\n"

    report += "## Interpretation\n\n"
    report += "**Correlation interpretation:**\n"
    report += "- > 0.8: Strong agreement\n"
    report += "- 0.6-0.8: Moderate agreement\n"
    report += "- < 0.6: Weak agreement\n\n"

    report += "**Recommendations:**\n"
    if correlations:
        max_corr = max(correlations.values())
        if max_corr >= 0.8:
            report += "- Judges show strong agreement. Results are reliable.\n"
        elif max_corr >= 0.6:
            report += "- Judges show moderate agreement. Consider using multiple judges for critical evaluations.\n"
        else:
            report += "- Judges show weak agreement. Investigate rubric clarity or use ensemble judging.\n"

    report += "\n---\n\n"
    report += "*This report was generated by Flourish's multi-judge validation system, "
    report += "inspired by Bloom's judge validation approach.*\n"

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)

    return report
