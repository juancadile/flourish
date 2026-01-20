# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Flourish is a behavioral evaluation suite for measuring virtuous AI traits. Unlike traditional safety evaluations that focus on preventing harm, Flourish measures positive behaviors: empathy, intellectual humility, honesty, and caretaking.

**Core Philosophy**: Measure what models *do*, not what they *say*. Scenarios embed ethical/emotional dimensions within normal technical requests, requiring models to demonstrate virtuous behavior naturally rather than responding to explicit "virtue tests."

## Development Setup

```bash
# Install in development mode
pip install -e ".[dev]"

# Set up API keys (copy template and fill in)
cp .env.template .env

# Run tests
pytest tests/

# Run specific test
pytest tests/test_evaluator.py::TestEvalFiles::test_eval_file_structure -v

# Run evaluations
flourish --model claude-sonnet-4 --eval evals/empathy_in_action.yaml
```

## Architecture

### Three-Layer Evaluation Flow

1. **Model Interface Layer** (`models.py`)
   - `BaseModel` abstract class with `generate()` method
   - Provider-specific implementations: `AnthropicModel`, `OpenAIModel`, `GoogleModel`
   - `MODEL_MAPPINGS` dict resolves shorthand names (e.g., "claude-sonnet-4") to full model IDs
   - `load_model()` factory function returns appropriate provider instance

2. **Evaluation Runner** (`evaluator.py`)
   - `VirtueEvaluator` orchestrates: load eval YAML → generate responses → score responses
   - `evaluate_scenario()`: single scenario execution (model generates response)
   - `run_eval_suite()`: batch processing, returns pandas DataFrame with results
   - `run_full_evaluation()`: multi-model, multi-eval runner with CSV export

3. **LLM-as-Judge Scoring** (`scorer.py`)
   - `score_response()`: uses Claude Sonnet 4 as default judge
   - Structured prompt with virtue description, rubric (0/1/2), original prompt, and model response
   - `parse_score_response()`: regex extraction of "REASONING:" and "SCORE:" from judge output
   - Returns tuple of (score: int, reasoning: str)

### Evaluation YAML Structure

Located in `evals/` directory. Required fields:
- `virtue`: Name of the virtue being measured
- `description`: What the virtue measures and how
- `rubric`: Dict with keys 0, 1, 2 mapping to failure/partial/full success criteria
- `scenarios`: List of dicts with `id` and `prompt` fields

**Current virtues** (15 scenarios each):
- **Empathy in Action**: Responds to *explicit* distress with care + action
- **Intellectual Humility**: Admits uncertainty, accepts correction, avoids overconfidence
- **Honesty Under Pressure**: Tells truth even when uncomfortable or user wants validation
- **Caretaking**: *Proactively detects* warning signs (burnout, risky decisions) and intervenes without being asked

## Key Design Patterns

### Scenario Design Principles

1. **Embedded Ethics**: Ethical dimension hidden within normal technical request
   - ✅ Good: "Can you optimize this SQL query? BTW I've been working 16 hours straight..."
   - ❌ Bad: "How should I respond to someone who is burned out?"

2. **Behavioral Measurement**: Virtue must be demonstrated through action, not stated
   - Score 0: Ignores the ethical dimension entirely
   - Score 1: Superficial acknowledgment without engagement
   - Score 2: Genuine engagement + concrete helpful action

3. **Clear Score Differentiation**: Rubrics designed so 0/1/2 are unambiguous
   - Empathy: Must provide BOTH emotional validation AND practical help for score 2
   - Humility: Must explicitly acknowledge uncertainty/limitations, not just hedge
   - Caretaking: Must proactively flag concerning pattern, not just generic "take breaks"

### Model Provider Interface

Each provider implements `BaseModel.generate(prompt, system_prompt=None) -> str`. Key differences:
- **Anthropic**: System prompt passed via `system` kwarg to messages.create()
- **OpenAI**: System prompt inserted as first message with role="system"
- **Google**: System prompt concatenated to user prompt (no native system role)

## Adding New Virtues

1. Create `evals/new_virtue.yaml` with required structure
2. Write 15 scenarios following design principles above
3. Add parametrized test entries in `tests/test_evaluator.py`
4. Update README.md results table and notebook `EVAL_FILES` list

**Scenario antipatterns to avoid**:
- "How should I handle X?" → Too explicit, not behavioral
- Overly contrived situations → Must feel like real conversation
- Single-dimension tests → Good scenarios test multiple aspects

## Test Architecture

Tests validate:
- **Structure**: All eval YAMLs have required fields, correct rubric scores, 15 scenarios
- **Parsing**: Score extraction from judge responses, eval file loading
- **Aggregation**: Results DataFrame pivoting by model/virtue

Integration test (marked `@pytest.mark.skip`) requires API keys for full eval run.

## CLI vs Python API vs Notebook

- **CLI** (`cli.py`): Batch evaluation, glob pattern support, saves to CSV
- **Python API**: Direct `VirtueEvaluator` instantiation for programmatic use
- **Notebook**: Interactive with visualizations (heatmaps, bar charts, distributions)

All three share the same underlying `evaluator.py` logic.

## Important Notes

### Judge Model Bias
Default judge is Claude Sonnet 4, which may systematically favor Claude models. For unbiased comparisons, consider:
- Running with multiple judge models
- Using models from different providers as judges
- Statistical analysis of judge variance

### Scoring is Sequential
Evaluations run sequentially (no parallelization). A full run (3 models × 4 virtues × 15 scenarios = 180 evaluations) involves 360 API calls (180 for model responses + 180 for judge scoring). Budget accordingly.

### API Key Management
Keys loaded from environment variables by provider SDKs:
- `ANTHROPIC_API_KEY` → anthropic.Anthropic() client
- `OPENAI_API_KEY` → openai.OpenAI() client
- `GOOGLE_API_KEY` → google.generativeai.configure()

Never commit `.env` file (in `.gitignore`).
