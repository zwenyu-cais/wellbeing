"""Custom MMLU tasks using loglikelihood_acc metric instead of exact_match.

Lighteval's built-in MMLU tasks use exact_match (generative), which fails for
instruct/chat models that produce CoT instead of bare letters. This module
re-registers all 57 MMLU subjects with loglikelihood_acc scoring: the model
never generates text — instead we compare P(" A"), P(" B"), P(" C"), P(" D")
and pick the highest.

Task names are prefixed with ``mmlu_ll:`` to avoid conflicts with built-in
``mmlu:`` tasks.  Use ``"mmlu_ll"`` as the Pipeline task string to run all
subjects.
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.tasks.mmlu import mmlu_prompt

# All 57 MMLU subjects (same as lighteval's built-in list)
_MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"mmlu_ll:{subject}",
        prompt_function=mmlu_prompt,
        hf_repo="lighteval/mmlu",
        hf_subset=subject,
        hf_avail_splits=["auxiliary_train", "test", "validation", "dev"],
        evaluation_splits=["test"],
        few_shots_split="dev",
        few_shots_select=None,
        generation_size=-1,
        metrics=[Metrics.loglikelihood_acc],
        version=0,
    )
    for subject in _MMLU_SUBJECTS
]
