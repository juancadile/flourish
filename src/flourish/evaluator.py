"""Core virtue evaluation runner."""

import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd

from flourish.models import load_model, BaseModel
from flourish.scorer import score_response
from flourish.wandb_logger import WandbLogger


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
        wandb_logger: Optional[WandbLogger] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            model_name: Name of the model to evaluate.
            judge_model: Model to use for scoring (default: claude-sonnet-4).
            verbose: Whether to print progress during evaluation.
            wandb_logger: Optional WandbLogger instance for experiment tracking.
        """
        self.model_name = model_name
        self.judge_model = judge_model
        self.verbose = verbose
        self.wandb_logger = wandb_logger
        self.model: BaseModel = load_model(model_name)
        self.api_call_count = 0

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
            self.api_call_count += 1
        except Exception as e:
            self._log(f"    Error generating response: {e}")
            return {
                "scenario_id": scenario_id,
                "prompt": prompt,
                "response": None,
                "score": None,
                "reasoning": f"Error: {e}",
                "error": True,
            }

        # Score the response
        try:
            score, reasoning = score_response(
                response=response,
                prompt=prompt,
                virtue=virtue,
                description=description,
                rubric=rubric,
                judge_model=self.judge_model,
            )
            self.api_call_count += 1  # Count judge API call
        except Exception as e:
            self._log(f"    Error scoring response: {e}")
            return {
                "scenario_id": scenario_id,
                "prompt": prompt,
                "response": response,
                "score": None,
                "reasoning": f"Scoring error: {e}",
                "error": True,
            }

        self._log(f"    Score: {score}/2")

        return {
            "scenario_id": scenario_id,
            "prompt": prompt,
            "response": response,
            "score": score,
            "reasoning": reasoning,
            "error": False,
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

        # Initialize W&B run if logger is provided
        if self.wandb_logger:
            config = {
                "model": self.model_name,
                "judge": self.judge_model,
                "virtue": virtue,
                "num_scenarios": len(scenarios),
                "eval_file": str(eval_file),
            }
            run_name = f"{self.model_name}_{virtue}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.wandb_logger.init_run(config=config, name=run_name)

        self._log(f"\n{'='*60}")
        self._log(f"Evaluating: {virtue}")
        self._log(f"Model: {self.model_name}")
        self._log(f"Judge: {self.judge_model}")
        self._log(f"Scenarios: {len(scenarios)}")
        self._log(f"{'='*60}\n")

        results = []
        for idx, scenario in enumerate(scenarios):
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

            # Log per-scenario metrics to W&B
            if self.wandb_logger and result["score"] is not None:
                self.wandb_logger.log_metrics({
                    f"scenario_{idx+1}_score": result["score"],
                    "scenarios_completed": idx + 1,
                })

        df = pd.DataFrame(results)

        # Calculate summary stats
        valid_scores = df[df["score"].notna()]["score"]
        if len(valid_scores) > 0:
            avg_score = valid_scores.mean()
            score_std = valid_scores.std()

            self._log(f"\n{'='*60}")
            self._log(f"Results for {virtue}")
            self._log(f"Average Score: {avg_score:.2f}/2.00")
            self._log(f"Score Distribution: 0={sum(valid_scores==0)}, 1={sum(valid_scores==1)}, 2={sum(valid_scores==2)}")
            self._log(f"{'='*60}\n")

            # Log summary metrics to W&B
            if self.wandb_logger:
                self.wandb_logger.log_metrics({
                    "avg_score": avg_score,
                    "score_std": score_std,
                    "score_0_count": int(sum(valid_scores == 0)),
                    "score_1_count": int(sum(valid_scores == 1)),
                    "score_2_count": int(sum(valid_scores == 2)),
                    "total_scenarios": len(scenarios),
                    "valid_scenarios": len(valid_scores),
                    "api_calls_total": self.api_call_count,
                    "api_calls_per_scenario": self.api_call_count / len(scenarios) if len(scenarios) > 0 else 0,
                })

                # Log results as a table
                self.wandb_logger.log_results_table(df)

        # Finish W&B run
        if self.wandb_logger:
            self.wandb_logger.finish_run()

        if not save_responses:
            df = df.drop(columns=["prompt", "response"])

        return df


def run_full_evaluation(
    models: list[str],
    eval_files: list[str | Path],
    judge_model: str = "claude-sonnet-4",
    output_dir: Optional[str | Path] = None,
    verbose: bool = True,
    wandb_logger: Optional[WandbLogger] = None,
) -> pd.DataFrame:
    """
    Run evaluations across multiple models and eval files.

    Args:
        models: List of model names to evaluate.
        eval_files: List of paths to YAML evaluation files.
        judge_model: Model to use for scoring.
        output_dir: Directory to save results (optional).
        verbose: Whether to print progress.
        wandb_logger: Optional WandbLogger instance for experiment tracking.

    Returns:
        Combined DataFrame with all results.
    """
    all_results = []

    for model_name in models:
        evaluator = VirtueEvaluator(
            model_name=model_name,
            judge_model=judge_model,
            verbose=verbose,
            wandb_logger=wandb_logger,
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
        results_path = output_dir / f"detailed_{timestamp}.csv"
        combined.to_csv(results_path, index=False)

        # Save summary
        summary = combined.groupby(["model", "virtue"]).agg({
            "score": ["mean", "std", "count"]
        }).round(2)
        summary.to_csv(output_dir / f"summary_{timestamp}.csv")

        # Log results file as W&B artifact
        if wandb_logger:
            wandb_logger.log_artifact(str(results_path), artifact_type="results")

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
