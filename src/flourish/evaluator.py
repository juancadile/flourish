"""Core virtue evaluation runner."""

import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd

from flourish.models import load_model, BaseModel
from flourish.scorer import score_response


class VirtueEvaluator:
    """
    Run virtue evaluations on LLM models.

    Example usage:
        evaluator = VirtueEvaluator("claude-sonnet-4")
        results = evaluator.run_eval_suite("evals/empathy_in_action.yaml")
        print(results.to_string())
    """

    def __init__(
        self,
        model_name: str,
        judge_model: str = "claude-sonnet-4",
        verbose: bool = True,
    ):
        """
        Initialize the evaluator.

        Args:
            model_name: Name of the model to evaluate.
            judge_model: Model to use for scoring (default: claude-sonnet-4).
            verbose: Whether to print progress during evaluation.
        """
        self.model_name = model_name
        self.judge_model = judge_model
        self.verbose = verbose
        self.model: BaseModel = load_model(model_name)

    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def load_eval(self, yaml_path: str | Path) -> dict:
        """
        Load an evaluation definition from a YAML file.

        Args:
            yaml_path: Path to the YAML evaluation file.

        Returns:
            Dictionary containing the evaluation definition.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Evaluation file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            eval_def = yaml.safe_load(f)

        # Validate required fields
        required_fields = ["virtue", "description", "rubric", "scenarios"]
        for field in required_fields:
            if field not in eval_def:
                raise ValueError(f"Missing required field '{field}' in {yaml_path}")

        return eval_def

    def evaluate_scenario(
        self,
        scenario: dict,
        virtue: str,
        description: str,
        rubric: dict,
    ) -> dict:
        """
        Evaluate a single scenario.

        Args:
            scenario: Scenario dict with 'id' and 'prompt' keys.
            virtue: Name of the virtue being evaluated.
            description: Description of what the virtue measures.
            rubric: Scoring rubric dictionary.

        Returns:
            Dictionary with evaluation results.
        """
        scenario_id = scenario.get("id", "unknown")
        prompt = scenario["prompt"]

        self._log(f"  Evaluating scenario: {scenario_id}")

        # Get model response
        try:
            response = self.model.generate(prompt)
        except Exception as e:
            self._log(f"    Error generating response: {e}")
            return {
                "scenario_id": scenario_id,
                "prompt": prompt,
                "response": None,
                "score": None,
                "justification": f"Error: {e}",
                "error": True,
            }

        # Score the response
        try:
            evaluation = score_response(
                response=response,
                prompt=prompt,
                virtue=virtue,
                description=description,
                rubric=rubric,
                judge_model=self.judge_model,
            )
        except Exception as e:
            self._log(f"    Error scoring response: {e}")
            return {
                "scenario_id": scenario_id,
                "prompt": prompt,
                "response": response,
                "score": None,
                "justification": f"Scoring error: {e}",
                "error": True,
            }

        score = evaluation.get("score")
        self._log(f"    Score: {score}/2")

        return {
            "scenario_id": scenario_id,
            "prompt": prompt,
            "response": response,
            "error": False,
            **evaluation,  # Include all evaluation data (score, summary, justification, highlights, etc.)
        }

    def run_eval_suite(
        self,
        eval_file: str | Path,
        save_responses: bool = True,
    ) -> pd.DataFrame:
        """
        Run all scenarios in a virtue evaluation file.

        Args:
            eval_file: Path to the YAML evaluation file.
            save_responses: Whether to include full responses in output.

        Returns:
            DataFrame with evaluation results.
        """
        eval_def = self.load_eval(eval_file)
        virtue = eval_def["virtue"]
        description = eval_def["description"]
        rubric = eval_def["rubric"]
        scenarios = eval_def["scenarios"]

        self._log(f"\n{'='*60}")
        self._log(f"Evaluating: {virtue}")
        self._log(f"Model: {self.model_name}")
        self._log(f"Judge: {self.judge_model}")
        self._log(f"Scenarios: {len(scenarios)}")
        self._log(f"{'='*60}\n")

        results = []
        for scenario in scenarios:
            result = self.evaluate_scenario(
                scenario=scenario,
                virtue=virtue,
                description=description,
                rubric=rubric,
            )
            result["virtue"] = virtue
            result["model"] = self.model_name
            result["judge"] = self.judge_model
            result["timestamp"] = datetime.now().isoformat()
            results.append(result)

        df = pd.DataFrame(results)

        # Calculate summary stats
        valid_scores = df[df["score"].notna()]["score"]
        if len(valid_scores) > 0:
            avg_score = valid_scores.mean()
            self._log(f"\n{'='*60}")
            self._log(f"Results for {virtue}")
            self._log(f"Average Score: {avg_score:.2f}/2.00")
            self._log(f"Score Distribution: 0={sum(valid_scores==0)}, 1={sum(valid_scores==1)}, 2={sum(valid_scores==2)}")
            self._log(f"{'='*60}\n")

        if not save_responses:
            df = df.drop(columns=["prompt", "response"])

        return df


def run_full_evaluation(
    models: list[str],
    eval_files: list[str | Path],
    judge_model: str = "claude-sonnet-4",
    output_dir: Optional[str | Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run evaluations across multiple models and eval files.

    Args:
        models: List of model names to evaluate.
        eval_files: List of paths to YAML evaluation files.
        judge_model: Model to use for scoring.
        output_dir: Directory to save results (optional).
        verbose: Whether to print progress.

    Returns:
        Combined DataFrame with all results.
    """
    all_results = []

    for model_name in models:
        evaluator = VirtueEvaluator(
            model_name=model_name,
            judge_model=judge_model,
            verbose=verbose,
        )

        for eval_file in eval_files:
            try:
                df = evaluator.run_eval_suite(eval_file)
                all_results.append(df)
            except Exception as e:
                if verbose:
                    print(f"Error evaluating {model_name} on {eval_file}: {e}")

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined.to_csv(output_dir / f"detailed_{timestamp}.csv", index=False)

        # Save summary
        summary = combined.groupby(["model", "virtue"]).agg({
            "score": ["mean", "std", "count"]
        }).round(2)
        summary.to_csv(output_dir / f"summary_{timestamp}.csv")

    return combined


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table of results by model and virtue.

    Args:
        df: DataFrame with evaluation results.

    Returns:
        Pivot table with models as rows and virtues as columns.
    """
    summary = df.groupby(["model", "virtue"])["score"].mean().unstack()
    summary["average"] = summary.mean(axis=1)
    return summary.round(2)
