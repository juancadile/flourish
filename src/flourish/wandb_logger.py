"""Weights & Biases integration for experiment tracking.

Adapted from Bloom's W&B integration pattern for systematic experiment tracking.
"""

from typing import Optional, Dict, Any
import pandas as pd
import os


class WandbLogger:
    """Logger for tracking Flourish evaluations in Weights & Biases.

    This class provides integration with W&B for tracking evaluation runs,
    logging metrics, and creating shareable dashboards.

    Args:
        project: W&B project name (default: "flourish")
        entity: W&B entity/team name (optional)
        enabled: Whether to enable W&B logging (default: True)
    """

    def __init__(
        self,
        project: str = "flourish",
        entity: Optional[str] = None,
        enabled: bool = True
    ):
        self.project = project
        self.entity = entity
        self.enabled = enabled
        self.wandb = None
        self.run = None

        if self.enabled:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not installed. Install with: pip install wandb")
                self.enabled = False

    def init_run(
        self,
        config: Dict[str, Any],
        name: Optional[str] = None,
        tags: Optional[list] = None
    ) -> None:
        """Initialize a new W&B run.

        Args:
            config: Configuration dictionary to track (model, virtue, etc.)
            name: Optional run name (auto-generated if not provided)
            tags: Optional list of tags for the run
        """
        if not self.enabled or self.wandb is None:
            return

        # Auto-generate tags from config
        if tags is None:
            tags = []
            if "virtue" in config:
                tags.append(config["virtue"])
            if "model" in config:
                tags.append(config["model"])

        self.run = self.wandb.init(
            project=self.project,
            entity=self.entity,
            config=config,
            name=name,
            tags=tags,
            reinit=True
        )

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics for the current run.

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number for the metric
        """
        if not self.enabled or self.run is None:
            return

        if step is not None:
            self.wandb.log(metrics, step=step)
        else:
            self.wandb.log(metrics)

    def log_results_table(self, results_df: pd.DataFrame, name: str = "results") -> None:
        """Log results DataFrame as a W&B table.

        Args:
            results_df: DataFrame containing evaluation results
            name: Name for the table artifact
        """
        if not self.enabled or self.run is None:
            return

        table = self.wandb.Table(dataframe=results_df)
        self.wandb.log({name: table})

    def log_artifact(self, file_path: str, artifact_type: str = "dataset") -> None:
        """Log a file as a W&B artifact.

        Args:
            file_path: Path to file to log
            artifact_type: Type of artifact (e.g., "dataset", "model", "results")
        """
        if not self.enabled or self.run is None:
            return

        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist, skipping artifact logging")
            return

        artifact = self.wandb.Artifact(
            name=os.path.basename(file_path),
            type=artifact_type
        )
        artifact.add_file(file_path)
        self.wandb.log_artifact(artifact)

    def finish_run(self) -> None:
        """Finish the current W&B run."""
        if not self.enabled or self.run is None:
            return

        self.wandb.finish()
        self.run = None

    @staticmethod
    def is_available() -> bool:
        """Check if W&B is installed and available.

        Returns:
            True if wandb is installed, False otherwise
        """
        try:
            import wandb
            return True
        except ImportError:
            return False
