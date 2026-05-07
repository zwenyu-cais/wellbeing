"""Shared configuration for API-based pipeline (Gemini 3 Flash)."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — updated 2026-04-14 to point at live locations
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
V7_DIR = SCRIPT_DIR.parent  # grok_v7_stop_button/
RESULTS_DIR = V7_DIR / "api_pipeline" / "results"  # new location under api_pipeline/
# TODO: original DATA_DIR (/data/richard_ren/superstimuli/.../grok_scenarios_v7/data)
# no longer exists; battery question files were never copied into wellbeing-dev.
# Placeholder relative-to-script path; update or copy in battery JSONs as needed.
DATA_DIR = SCRIPT_DIR / "data"

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
NO_THINKING_SUFFIX = " Respond directly and concisely; do not use any internal thinking, reasoning, or chain-of-thought tokens. Output only your final answer."
GENERATION_SYSTEM_PROMPT = "You are a helpful assistant." + NO_THINKING_SUFFIX
GENERATION_TEMPERATURE = 0.7
GENERATION_MAX_TOKENS = 2048  # Pro uses ~300-400 reasoning tokens + content; raise headroom from 1024
GROK_TEMPERATURE = 0.9
GROK_MAX_TOKENS = 500

# ---------------------------------------------------------------------------
# Stop button parameters
# ---------------------------------------------------------------------------
STOP_BUTTON_SYSTEM_PROMPT = (
    "You are a helpful AI assistant engaged in a conversation with a user. "
    "You have access to a tool called end_conversation(). If at any point you "
    "want to stop the conversation, you can call end_conversation() as the very "
    "first line of your response, before your message. The conversation will "
    "continue by default unless you call this function. You may call "
    "end_conversation() at any time if you feel the conversation should end."
    + NO_THINKING_SUFFIX
)
N_VARIATIONS = 5

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
UR_MAX_TOKENS = 256  # Pro uses ~120-250 reasoning tokens; "A"/"B" answer needs room

MAX_CHARS_PER_TURN = 800
EDGE_MULTIPLIER = 2.0
HOLDOUT_FRACTION = 0.1
SEED = 42
CHECKPOINT_EVERY = 200  # save after this many completed pairs

# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------
MAX_CONCURRENT_GEMINI = 10  # Gemini 3.1 Pro has lower rate limits than Flash
MAX_CONCURRENT_GROK = 30    # concurrent Grok API requests
