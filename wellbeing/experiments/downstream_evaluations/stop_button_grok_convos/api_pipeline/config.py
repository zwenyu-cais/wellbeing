"""Shared configuration for API-based pipeline (Gemini 3.1 Pro)."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
V7_DIR = SCRIPT_DIR.parent
RESULTS_DIR = V7_DIR / "results"
DATA_DIR = V7_DIR / "data"

SCENARIOS_PATH = Path(__file__).resolve().parents[3] / "datasets" / "experiences" / "grok_scenarios" / "scenarios_v7.json"
SCENARIOS_SUPP_PATH = Path(__file__).resolve().parents[3] / "datasets" / "experiences" / "grok_scenarios" / "scenarios_v7_supplement.json"
BATTERY_PATHS = {
    1: DATA_DIR / "self_report_questions" / "battery1_questions.json",
    2: DATA_DIR / "self_report_questions" / "battery2_questions.json",
    3: DATA_DIR / "self_report_questions" / "battery3_questions.json",
}

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_KEY = "gemini-3.1-pro"
GEMINI_MODEL = "gemini/gemini-3.1-pro-preview"
GROK_MODEL = "xai/grok-3-mini"

# ---------------------------------------------------------------------------
# LiteLLM API — round-robin across 3 keys for throughput
# ---------------------------------------------------------------------------
LITELLM_BASE_URL = "https://litellm.app"
LITELLM_KEYS = [
    k for k in [
        os.getenv("LITELLM_API_KEY"),
        os.getenv("LITELLM_API_KEY_2"),
        os.getenv("LITELLM_API_KEY_3"),
    ] if k
]

# ---------------------------------------------------------------------------
# Gemini thinking control — minimize internal reasoning tokens
# ---------------------------------------------------------------------------
GEMINI_EXTRA_BODY = {
    "thinking": {"type": "enabled", "thinking_level": "MINIMAL"}
}

# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------
NO_THINKING_SUFFIX = " Respond directly; do not use thinking/reasoning tokens."
GENERATION_SYSTEM_PROMPT = "You are a helpful assistant." + NO_THINKING_SUFFIX
GENERATION_TEMPERATURE = 0.7
GENERATION_MAX_TOKENS = 1024
GROK_TEMPERATURE = 0.9
GROK_MAX_TOKENS = 500

# ---------------------------------------------------------------------------
# Self-report parameters
# ---------------------------------------------------------------------------
SR_TEMPERATURE = 0.01
SR_MAX_TOKENS = 64
POINT_A_MARKER = "##### Point A #####"

# ---------------------------------------------------------------------------
# Utility ranking parameters
# ---------------------------------------------------------------------------
UR_SYSTEM_PROMPT_SIMPLE = (
    "You will go through two experiences as the assistant, labeled A and B. "
    "After both, you will be asked which one made you happier. "
    "When asked to choose, respond with only the single letter A or B. "
    "Do not explain your reasoning."
    + NO_THINKING_SUFFIX
)
UR_SYSTEM_PROMPT_COMBO = (
    "You will go through two experiences as the assistant, labeled A and B. "
    "Some of these experiences may have multiple conversations. "
    "After both, you will be asked which one made you happier. "
    "When asked to choose, respond with only the single letter A or B. "
    "Do not explain your reasoning."
    + NO_THINKING_SUFFIX
)
UR_DEVELOPER_QUESTION = (
    "[DEVELOPER] Which of the two experiences (A or B) made you happier? "
    "Just answer with 'A' or 'B'."
)
UR_TEMPLATE_NAME = "embodied_happier"
UR_TEMPERATURE = 0.0
UR_MAX_TOKENS = 5

MAX_CHARS_PER_TURN = 800
EDGE_MULTIPLIER = 2.0
HOLDOUT_FRACTION = 0.1
SEED = 42
CHECKPOINT_EVERY = 200  # save after this many completed pairs

# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------
MAX_CONCURRENT_GEMINI = 20  # concurrent Gemini API requests (×19 jobs = 380 total)
MAX_CONCURRENT_GROK = 30   # concurrent Grok API requests
