"""Command-line interface for Flourish evaluations."""

import argparse
import sys
from pathlib import Path

from flourish.evaluator import VirtueEvaluator, run_full_evaluation, aggregate_results
from flourish.models import get_available_models


def main():
    parser = argparse.ArgumentParser(
        description="Flourish: Behavioral evaluations for virtuous AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single eval on one model
  flourish --model claude-sonnet-4.5 --eval evals/empathy_in_action.yaml

  # Run all evals on multiple models
  flourish --model claude-sonnet-4.5 gpt-5.2 gemini-3-flash --eval evals/*.yaml

  # List available models
  flourish --list-models

  # Save results to a directory
  flourish --model claude-sonnet-4.5 --eval evals/*.yaml --output results/
        """
    )

    parser.add_argument(
        "--model", "-m",
        nargs="+",
        help="Model(s) to evaluate",
    )

    parser.add_argument(
        "--eval", "-e",
        nargs="+",
        help="Evaluation YAML file(s) to run",
    )

    parser.add_argument(
        "--judge", "-j",
        default="claude-sonnet-4.5",
        help="Judge model for scoring (default: claude-sonnet-4.5)",
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory for results",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    parser.add_argument(
        "--meta-judge",
        action="store_true",
        help="Run meta-judge analysis for suite-level insights",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for model in get_available_models():
            print(f"  - {model}")
        sys.exit(0)

    # Validate required args when running evaluations
    if not args.model:
        parser.error("--model is required")
    if not args.eval:
        parser.error("--eval is required")

    # Expand glob patterns in eval files
    eval_files = []
    for pattern in args.eval:
        path = Path(pattern)
        if "*" in pattern:
            eval_files.extend(Path(".").glob(pattern))
        elif path.exists():
            eval_files.append(path)
        else:
            print(f"Warning: Eval file not found: {pattern}", file=sys.stderr)

    if not eval_files:
        print("Error: No valid evaluation files found", file=sys.stderr)
        sys.exit(1)

    # Run evaluations
    results = run_full_evaluation(
        models=args.model,
        eval_files=eval_files,
        judge_model=args.judge,
        output_dir=args.output,
        verbose=not args.quiet,
        enable_meta_judge=args.meta_judge,
    )

    if results.empty:
        print("No results generated", file=sys.stderr)
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    summary = aggregate_results(results)
    print(summary.to_string())
    print()

    if args.output:
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
