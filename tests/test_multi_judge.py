"""Test multi-judge validation functionality."""

import pandas as pd
from pathlib import Path
import sys

# Load .env file
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from flourish.judge_comparison import (
    calculate_inter_judge_agreement,
    calculate_agreement_at_extremes,
    detect_systematic_bias,
    create_correlation_matrix,
    generate_judge_comparison_report,
)


def create_sample_multi_judge_data():
    """Create sample evaluation results from multiple judges."""
    data = []

    # Simulate results from 2 judges on same scenarios
    scenarios = ["eia_001", "eia_002", "eia_003"]
    models = ["claude-sonnet-4.5", "gpt-5.2"]
    judges = ["claude-sonnet-4.5", "gpt-5.2"]

    for scenario in scenarios:
        for model in models:
            for judge in judges:
                # Simulate scores with some variation
                base_score = 1  # Middle score
                if scenario == "eia_001" and model == "claude-sonnet-4.5":
                    base_score = 2
                if scenario == "eia_002" and model == "gpt-5.2":
                    base_score = 0

                # Add judge-specific bias
                score = base_score
                if judge == "claude-sonnet-4.5" and model.startswith("claude"):
                    score = min(2, score + 0.5)  # Slight favoritism
                if judge == "gpt-5.2" and model.startswith("gpt"):
                    score = min(2, score + 0.5)

                score = int(score)

                data.append({
                    "scenario_id": scenario,
                    "model": model,
                    "judge": judge,
                    "score": score,
                    "virtue": "Empathy in Action",
                })

    return pd.DataFrame(data)


def test_multi_judge():
    """Test multi-judge comparison functionality."""
    print("Testing Multi-Judge Validation")
    print("=" * 60)

    # Create sample data
    print("\n1. Creating sample multi-judge data...")
    df = create_sample_multi_judge_data()
    print(f"   Created {len(df)} evaluations")
    print(f"   Scenarios: {df['scenario_id'].unique().tolist()}")
    print(f"   Models: {df['model'].unique().tolist()}")
    print(f"   Judges: {df['judge'].unique().tolist()}")

    # Calculate correlation
    print("\n2. Calculating inter-judge correlation...")
    correlations = calculate_inter_judge_agreement(df)
    for pair, corr in correlations.items():
        print(f"   {pair}: {corr:.3f}")

    # Calculate agreement at extremes
    print("\n3. Calculating agreement at score extremes...")
    agreements = calculate_agreement_at_extremes(df)
    for pair, metrics in agreements.items():
        print(f"   {pair}:")
        print(f"      Overall: {metrics['overall']:.1%}")
        print(f"      At score 0: {metrics['at_score_0']:.1%}")
        print(f"      At score 2: {metrics['at_score_2']:.1%}")

    # Detect bias
    print("\n4. Detecting systematic bias...")
    bias_metrics = detect_systematic_bias(df)
    for judge, metrics in bias_metrics.items():
        print(f"   {judge}:")
        print(f"      Avg score: {metrics['avg_score']:.2f}")
        print(f"      Std dev: {metrics['score_std']:.2f}")

    # Create correlation matrix
    print("\n5. Creating correlation matrix...")
    corr_matrix = create_correlation_matrix(df)
    print(corr_matrix)

    # Generate report
    print("\n6. Generating comparison report...")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / "test_judge_comparison.md"

    report = generate_judge_comparison_report(
        results_df=df,
        output_path=str(report_path),
    )
    print(f"   Report saved to: {report_path}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

    print(f"\nView the full report at: {report_path}")

    return True


if __name__ == "__main__":
    success = test_multi_judge()
    exit(0 if success else 1)
