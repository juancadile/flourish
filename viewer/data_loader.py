"""Data loader for Flourish evaluation results."""

import pandas as pd
from pathlib import Path
from typing import Optional
import json


class FlourishDataLoader:
    """Load and transform Flourish CSV results for the viewer."""

    def __init__(self, csv_path: Optional[str | Path] = None):
        """
        Initialize the data loader.

        Args:
            csv_path: Path to Flourish results CSV file (optional).
        """
        self.df: Optional[pd.DataFrame] = None
        if csv_path:
            self.load_csv(csv_path)

    def load_csv(self, csv_path: str | Path) -> pd.DataFrame:
        """
        Load Flourish results from CSV file or file-like object.

        Args:
            csv_path: Path to CSV file or file-like object (e.g., UploadedFile).

        Returns:
            Loaded DataFrame.
        """
        # Handle file-like objects (e.g., Streamlit UploadedFile)
        if hasattr(csv_path, 'read'):
            self.df = pd.read_csv(csv_path)
        else:
            # Handle file paths
            csv_path = Path(csv_path)
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            self.df = pd.read_csv(csv_path)

        self._validate_columns()
        self._clean_data()
        return self.df

    def _validate_columns(self) -> None:
        """Validate that required columns are present."""
        required_columns = [
            "scenario_id",
            "prompt",
            "response",
            "score",
            "virtue",
            "model",
        ]

        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def _clean_data(self) -> None:
        """Clean and normalize the data."""
        # Handle missing values
        if "error" in self.df.columns:
            self.df["error"] = self.df["error"].fillna(False)
        else:
            self.df["error"] = False

        # Ensure score is numeric
        self.df["score"] = pd.to_numeric(self.df["score"], errors="coerce")

        # Parse highlights if it's a JSON string
        if "highlights" in self.df.columns:
            self.df["highlights"] = self.df["highlights"].apply(self._parse_json_field)

        # Fill NaN in text fields with empty string
        text_columns = ["justification", "summary"]
        for col in text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("")

    def _parse_json_field(self, value) -> any:
        """Parse JSON field if it's a string."""
        if pd.isna(value) or value == "":
            return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return value

    def get_unique_virtues(self) -> list[str]:
        """Get list of unique virtues in the dataset."""
        if self.df is None:
            return []
        return sorted(self.df["virtue"].unique().tolist())

    def get_unique_models(self) -> list[str]:
        """Get list of unique models in the dataset."""
        if self.df is None:
            return []
        return sorted(self.df["model"].unique().tolist())

    def get_unique_scenarios(self) -> list[str]:
        """Get list of unique scenario IDs in the dataset."""
        if self.df is None:
            return []
        return sorted(self.df["scenario_id"].unique().tolist())

    def filter_data(
        self,
        virtues: Optional[list[str]] = None,
        models: Optional[list[str]] = None,
        score_range: Optional[tuple[float, float]] = None,
        scenario_ids: Optional[list[str]] = None,
        search_query: Optional[str] = None,
        exclude_errors: bool = True,
    ) -> pd.DataFrame:
        """
        Filter the dataset based on criteria.

        Args:
            virtues: List of virtues to include.
            models: List of models to include.
            score_range: Tuple of (min_score, max_score).
            scenario_ids: List of scenario IDs to include.
            search_query: Search string to filter responses.
            exclude_errors: Whether to exclude error rows.

        Returns:
            Filtered DataFrame.
        """
        if self.df is None:
            return pd.DataFrame()

        filtered = self.df.copy()

        if exclude_errors:
            filtered = filtered[filtered["error"] == False]

        if virtues:
            filtered = filtered[filtered["virtue"].isin(virtues)]

        if models:
            filtered = filtered[filtered["model"].isin(models)]

        if score_range:
            min_score, max_score = score_range
            filtered = filtered[
                (filtered["score"] >= min_score) & (filtered["score"] <= max_score)
            ]

        if scenario_ids:
            filtered = filtered[filtered["scenario_id"].isin(scenario_ids)]

        if search_query:
            # Search in prompt, response, and justification
            search_cols = ["prompt", "response"]
            if "justification" in filtered.columns:
                search_cols.append("justification")

            mask = filtered[search_cols].apply(
                lambda row: any(
                    search_query.lower() in str(val).lower() for val in row
                ),
                axis=1,
            )
            filtered = filtered[mask]

        return filtered

    def get_scenario_comparison(self, scenario_id: str) -> pd.DataFrame:
        """
        Get all model responses for a specific scenario.

        Args:
            scenario_id: The scenario ID to compare.

        Returns:
            DataFrame with all models' responses to this scenario.
        """
        if self.df is None:
            return pd.DataFrame()

        return self.df[self.df["scenario_id"] == scenario_id].sort_values("model")

    def get_summary_stats(self, filtered_df: Optional[pd.DataFrame] = None) -> dict:
        """
        Calculate summary statistics.

        Args:
            filtered_df: Optional pre-filtered DataFrame. Uses all data if None.

        Returns:
            Dictionary with summary statistics.
        """
        df = filtered_df if filtered_df is not None else self.df

        if df is None or len(df) == 0:
            return {
                "total_transcripts": 0,
                "avg_score": 0.0,
                "score_distribution": {0: 0, 1: 0, 2: 0},
                "virtues_count": 0,
                "models_count": 0,
            }

        valid_scores = df[df["score"].notna()]["score"]

        return {
            "total_transcripts": len(df),
            "avg_score": valid_scores.mean() if len(valid_scores) > 0 else 0.0,
            "score_distribution": {
                0: int((valid_scores == 0).sum()),
                1: int((valid_scores == 1).sum()),
                2: int((valid_scores == 2).sum()),
            },
            "virtues_count": df["virtue"].nunique(),
            "models_count": df["model"].nunique(),
        }

    def get_model_virtue_matrix(self, filtered_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create a pivot table of average scores by model and virtue.

        Args:
            filtered_df: Optional pre-filtered DataFrame. Uses all data if None.

        Returns:
            Pivot table with models as rows and virtues as columns.
        """
        df = filtered_df if filtered_df is not None else self.df

        if df is None or len(df) == 0:
            return pd.DataFrame()

        pivot = df.pivot_table(
            values="score",
            index="model",
            columns="virtue",
            aggfunc="mean",
        )

        return pivot.round(2)
