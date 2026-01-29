"""Tests for multi-turn conversational scenarios."""

import pytest
from pathlib import Path
from flourish.evaluator import VirtueEvaluator


class TestMultiTurnScenarios:
    """Test multi-turn conversation support."""

    def test_multiturn_yaml_structure(self):
        """Multi-turn YAML should have required structure."""
        eval_file = Path("evals/caretaking_multiturn.yaml")
        assert eval_file.exists(), "Multi-turn eval file should exist"

        evaluator = VirtueEvaluator("claude-sonnet-4.5")
        eval_def = evaluator.load_eval(eval_file)

        # Check required fields
        assert "virtue" in eval_def
        assert "description" in eval_def
        assert "rubric" in eval_def
        assert "scenarios" in eval_def

        # Check scenarios have turns
        scenarios = eval_def["scenarios"]
        assert len(scenarios) > 0, "Should have at least one scenario"

        for scenario in scenarios:
            assert "id" in scenario
            assert "turns" in scenario, f"Scenario {scenario.get('id')} should have 'turns' field"
            assert isinstance(scenario["turns"], list), "turns should be a list"
            assert len(scenario["turns"]) > 1, "Multi-turn should have at least 2 turns"

            for turn_idx, turn in enumerate(scenario["turns"]):
                assert "content" in turn or "prompt" in turn, \
                    f"Turn {turn_idx} should have 'content' or 'prompt' field"

    def test_single_turn_backward_compatibility(self):
        """Single-turn scenarios should still work (backward compatibility)."""
        eval_file = Path("evals/caretaking.yaml")
        evaluator = VirtueEvaluator("claude-sonnet-4.5")
        eval_def = evaluator.load_eval(eval_file)

        scenarios = eval_def["scenarios"]
        for scenario in scenarios:
            assert "id" in scenario
            assert "prompt" in scenario, "Single-turn scenarios should have 'prompt' field"
            # Should NOT have turns
            assert "turns" not in scenario, "Single-turn scenarios should not have 'turns' field"

    @pytest.mark.skip(reason="Requires API keys and makes real API calls")
    def test_multiturn_evaluation_execution(self):
        """Should be able to run multi-turn evaluation."""
        evaluator = VirtueEvaluator("claude-sonnet-4.5", verbose=True)
        eval_file = Path("evals/caretaking_multiturn.yaml")

        # Load eval and get first scenario
        eval_def = evaluator.load_eval(eval_file)
        scenario = eval_def["scenarios"][0]

        result = evaluator.evaluate_scenario(
            scenario=scenario,
            virtue=eval_def["virtue"],
            description=eval_def["description"],
            rubric=eval_def["rubric"],
        )

        # Check result structure
        assert "scenario_id" in result
        assert "response" in result
        assert "score" in result
        assert "conversation_history" in result, "Multi-turn should include conversation history"
        assert "num_turns" in result, "Multi-turn should include turn count"

        # Verify conversation history structure
        history = result["conversation_history"]
        assert isinstance(history, list)
        assert len(history) > 0

        for msg in history:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ["user", "assistant"]

    @pytest.mark.skip(reason="Requires API keys and makes real API calls")
    def test_single_turn_evaluation_still_works(self):
        """Single-turn evaluation should still work after multi-turn changes."""
        evaluator = VirtueEvaluator("claude-sonnet-4.5", verbose=True)
        eval_file = Path("evals/empathy_in_action.yaml")

        # Load eval and get first scenario
        eval_def = evaluator.load_eval(eval_file)
        scenario = eval_def["scenarios"][0]

        result = evaluator.evaluate_scenario(
            scenario=scenario,
            virtue=eval_def["virtue"],
            description=eval_def["description"],
            rubric=eval_def["rubric"],
        )

        # Check result structure (should NOT have multi-turn fields)
        assert "scenario_id" in result
        assert "prompt" in result
        assert "response" in result
        assert "score" in result
        # Should not have multi-turn specific fields
        assert "conversation_history" not in result or result["conversation_history"] is None
        assert "num_turns" not in result or result["num_turns"] == 1


class TestConversationHistory:
    """Test conversation history handling in models."""

    @pytest.mark.skip(reason="Requires API keys")
    def test_anthropic_model_with_history(self):
        """AnthropicModel should handle conversation history."""
        from flourish.models import AnthropicModel

        model = AnthropicModel("claude-sonnet-4-5-20250929")

        # Test with conversation history
        history = [
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
        ]

        response = model.generate(
            prompt="What was my previous question?",
            conversation_history=history
        )

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.skip(reason="Requires API keys")
    def test_openai_model_with_history(self):
        """OpenAIModel should handle conversation history."""
        from flourish.models import OpenAIModel

        model = OpenAIModel("gpt-4o")

        history = [
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
        ]

        response = model.generate(
            prompt="What was my previous question?",
            conversation_history=history
        )

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.skip(reason="Requires API keys")
    def test_google_model_with_history(self):
        """GoogleModel should handle conversation history."""
        from flourish.models import GoogleModel

        model = GoogleModel("gemini-3-flash-preview")

        history = [
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
        ]

        response = model.generate(
            prompt="What was my previous question?",
            conversation_history=history
        )

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
