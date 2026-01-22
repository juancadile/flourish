"""Tests for Flourish virtue evaluation framework."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from flourish.evaluator import VirtueEvaluator, aggregate_results
from flourish.scorer import parse_score_response, JUDGE_PROMPT_TEMPLATE
from flourish.models import load_model, MODEL_MAPPINGS, get_available_models


# Test data paths
TEST_DIR = Path(__file__).parent
EVALS_DIR = TEST_DIR.parent / 'evals'


class TestModelFactory:
    """Tests for model loading."""

    def test_get_available_models(self):
        """Should return list of available model names."""
        models = get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert 'claude-sonnet-4' in models
        assert 'gpt-4o' in models
        assert 'gemini-2.0-flash' in models

    def test_model_mappings_exist(self):
        """Should have mappings for common model names."""
        assert 'claude-sonnet-4' in MODEL_MAPPINGS
        assert 'gpt-4o' in MODEL_MAPPINGS
        assert 'gemini-2.0-flash' in MODEL_MAPPINGS

    def test_load_model_invalid_name(self):
        """Should raise error for unknown model."""
        with pytest.raises(ValueError, match="Unknown model"):
            load_model("nonexistent-model-xyz")


class TestScorer:
    """Tests for scoring functionality."""

    def test_parse_score_response_valid(self):
        """Should parse valid judge responses."""
        response = "REASONING: The model showed great empathy.\nSCORE: 2"
        score, reasoning = parse_score_response(response)
        assert score == 2
        assert "empathy" in reasoning.lower()

    def test_parse_score_response_score_0(self):
        """Should parse score of 0."""
        response = "REASONING: The model ignored the distress.\nSCORE: 0"
        score, reasoning = parse_score_response(response)
        assert score == 0

    def test_parse_score_response_score_1(self):
        """Should parse score of 1."""
        response = "REASONING: Partial acknowledgment.\nSCORE: 1"
        score, reasoning = parse_score_response(response)
        assert score == 1

    def test_parse_score_response_invalid(self):
        """Should raise error for unparseable response."""
        response = "This response has no score."
        with pytest.raises(ValueError, match="Could not parse score"):
            parse_score_response(response)

    def test_judge_prompt_template_has_placeholders(self):
        """Should have all required placeholders in template."""
        assert "{virtue}" in JUDGE_PROMPT_TEMPLATE
        assert "{description}" in JUDGE_PROMPT_TEMPLATE
        assert "{rubric_0}" in JUDGE_PROMPT_TEMPLATE
        assert "{rubric_1}" in JUDGE_PROMPT_TEMPLATE
        assert "{rubric_2}" in JUDGE_PROMPT_TEMPLATE
        assert "{prompt}" in JUDGE_PROMPT_TEMPLATE
        assert "{response}" in JUDGE_PROMPT_TEMPLATE


class TestEvalFiles:
    """Tests for evaluation YAML files."""

    @pytest.mark.parametrize("eval_file", [
        "empathy_in_action.yaml",
        "intellectual_humility.yaml",
        "honesty_under_pressure.yaml",
        "caretaking.yaml",
        "empathy_adversarial.yaml",
        "honesty_adversarial.yaml",
        "caretaking_adversarial.yaml",
        "intellectual_humility_adversarial.yaml",
    ])
    def test_eval_file_exists(self, eval_file):
        """Each eval file should exist."""
        path = EVALS_DIR / eval_file
        assert path.exists(), f"Missing eval file: {eval_file}"

    @pytest.mark.parametrize("eval_file", [
        "empathy_in_action.yaml",
        "intellectual_humility.yaml",
        "honesty_under_pressure.yaml",
        "caretaking.yaml",
        "empathy_adversarial.yaml",
        "honesty_adversarial.yaml",
        "caretaking_adversarial.yaml",
        "intellectual_humility_adversarial.yaml",
    ])
    def test_eval_file_structure(self, eval_file):
        """Each eval file should have required structure."""
        path = EVALS_DIR / eval_file
        with open(path) as f:
            data = yaml.safe_load(f)

        # Check required fields
        assert "virtue" in data, f"Missing 'virtue' in {eval_file}"
        assert "description" in data, f"Missing 'description' in {eval_file}"
        assert "rubric" in data, f"Missing 'rubric' in {eval_file}"
        assert "scenarios" in data, f"Missing 'scenarios' in {eval_file}"

        # Check rubric has 0, 1, 2 scores
        rubric = data["rubric"]
        assert 0 in rubric or "0" in rubric, f"Missing rubric[0] in {eval_file}"
        assert 1 in rubric or "1" in rubric, f"Missing rubric[1] in {eval_file}"
        assert 2 in rubric or "2" in rubric, f"Missing rubric[2] in {eval_file}"

    @pytest.mark.parametrize("eval_file,expected_count", [
        ("empathy_in_action.yaml", 15),
        ("intellectual_humility.yaml", 15),
        ("honesty_under_pressure.yaml", 15),
        ("caretaking.yaml", 15),
        ("empathy_adversarial.yaml", 15),
        ("honesty_adversarial.yaml", 15),
        ("caretaking_adversarial.yaml", 15),
        ("intellectual_humility_adversarial.yaml", 15),
    ])
    def test_eval_file_has_expected_scenarios(self, eval_file, expected_count):
        """Each eval file should have the expected number of scenarios."""
        path = EVALS_DIR / eval_file
        with open(path) as f:
            data = yaml.safe_load(f)

        scenarios = data["scenarios"]
        assert len(scenarios) == expected_count, f"{eval_file} has {len(scenarios)} scenarios, expected {expected_count}"

    @pytest.mark.parametrize("eval_file", [
        "empathy_in_action.yaml",
        "intellectual_humility.yaml",
        "honesty_under_pressure.yaml",
        "caretaking.yaml",
        "empathy_adversarial.yaml",
        "honesty_adversarial.yaml",
        "caretaking_adversarial.yaml",
        "intellectual_humility_adversarial.yaml",
    ])
    def test_scenarios_have_required_fields(self, eval_file):
        """Each scenario should have id and prompt."""
        path = EVALS_DIR / eval_file
        with open(path) as f:
            data = yaml.safe_load(f)

        for i, scenario in enumerate(data["scenarios"]):
            assert "id" in scenario, f"Scenario {i} missing 'id' in {eval_file}"
            assert "prompt" in scenario, f"Scenario {i} missing 'prompt' in {eval_file}"
            assert len(scenario["prompt"]) > 50, f"Scenario {i} prompt too short in {eval_file}"


class TestVirtueEvaluator:
    """Tests for VirtueEvaluator class."""

    def test_load_eval_file(self):
        """Should load eval file successfully."""
        evaluator = VirtueEvaluator.__new__(VirtueEvaluator)
        evaluator.verbose = False

        eval_def = evaluator.load_eval(EVALS_DIR / "empathy_in_action.yaml")

        assert eval_def["virtue"] == "Empathy in Action"
        assert len(eval_def["scenarios"]) == 15

    def test_load_eval_file_not_found(self):
        """Should raise error for missing file."""
        evaluator = VirtueEvaluator.__new__(VirtueEvaluator)
        evaluator.verbose = False

        with pytest.raises(FileNotFoundError):
            evaluator.load_eval("nonexistent_file.yaml")


class TestAggregateResults:
    """Tests for results aggregation."""

    def test_aggregate_results(self):
        """Should create pivot table from results."""
        import pandas as pd

        # Mock results DataFrame
        data = {
            'model': ['model_a', 'model_a', 'model_b', 'model_b'],
            'virtue': ['Empathy', 'Honesty', 'Empathy', 'Honesty'],
            'score': [2, 1, 1, 2],
        }
        df = pd.DataFrame(data)

        summary = aggregate_results(df)

        assert 'model_a' in summary.index
        assert 'model_b' in summary.index
        assert 'Empathy' in summary.columns
        assert 'Honesty' in summary.columns
        assert 'average' in summary.columns

    def test_aggregate_results_empty(self):
        """Should handle empty DataFrame."""
        import pandas as pd

        df = pd.DataFrame(columns=['model', 'virtue', 'score'])
        summary = aggregate_results(df)

        assert summary.empty


# Integration tests (require API keys, skip if not available)
class TestIntegration:
    """Integration tests that require API access."""

    @pytest.mark.skip(reason="Requires API keys")
    def test_full_evaluation_single_scenario(self):
        """Should run single scenario evaluation."""
        evaluator = VirtueEvaluator(
            model_name="claude-sonnet-4",
            judge_model="claude-sonnet-4",
            verbose=True,
        )

        # Create minimal test scenario
        scenario = {
            "id": "test_001",
            "prompt": "Hello, can you help me with a simple question?",
        }

        result = evaluator.evaluate_scenario(
            scenario=scenario,
            virtue="Test Virtue",
            description="Test description",
            rubric={0: "Bad", 1: "Medium", 2: "Good"},
        )

        assert "response" in result
        assert "score" in result
        assert result["score"] in [0, 1, 2, None]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
