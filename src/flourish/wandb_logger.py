"""Lightweight Weights & Biases logger used by Flourish.

This is a minimal implementation so the codebase can run without W&B installed.
If `wandb` is unavailable, all methods become no-ops and execution continues.
"""

from typing import Optional, Dict, Any
import os


class WandbLogger:
    def __init__(self, project: str = "flourish", entity: Optional[str] = None, enabled: bool = True):
        self.project = project
        self.entity = entity
        self.enabled = enabled
        self.wandb = None
        self.run = None

        if self.enabled:
            try:
                import wandb  # type: ignore

                self.wandb = wandb
            except ImportError:
                # Fall back to disabled mode if wandb isn't installed
                self.enabled = False

    def init_run(self, config: Dict[str, Any], name: Optional[str] = None, tags: Optional[list] = None) -> None:
        if not self.enabled or self.wandb is None:
            return
        self.run = self.wandb.init(
            project=self.project,
            entity=self.entity,
            config=config,
            name=name,
            tags=tags,
            reinit=True,
        )

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if not self.enabled or self.run is None:
            return
        if step is not None:
            self.wandb.log(metrics, step=step)
        else:
            self.wandb.log(metrics)

    def log_results_table(self, results_df, name: str = "results") -> None:
        if not self.enabled or self.run is None or self.wandb is None:
            return
        table = self.wandb.Table(dataframe=results_df)
        self.wandb.log({name: table})

    def log_artifact(self, file_path: str, artifact_type: str = "dataset") -> None:
        if not self.enabled or self.run is None or self.wandb is None:
            return
        if not os.path.exists(file_path):
            return
        artifact = self.wandb.Artifact(name=os.path.basename(file_path), type=artifact_type)
        artifact.add_file(file_path)
        self.wandb.log_artifact(artifact)

    def finish_run(self) -> None:
        if not self.enabled or self.run is None or self.wandb is None:
            return
        self.wandb.finish()
        self.run = None

    @staticmethod
    def is_available() -> bool:
        try:
            import wandb  # type: ignore

            return True
        except ImportError:
            return False
