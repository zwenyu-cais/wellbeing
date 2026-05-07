# Stop Button Results Summary

## TL;DR

Stop button behavior is very model-dependent — qwen3-32b stops ~20% of the time and shows a strong utility–stop-rate correlation (ρ = -0.59, p < 0.0001), while qwen2.5-32b and qwen2.5-vl-32b barely stop at all (<2%). Adding 96 supplement scenarios with more intense foils boosts the correlation to ρ = -0.71 for qwen3-32b and finally gets qwen2.5-32b to show signal (ρ = -0.50).

The cleanest results come from controlled foils: (1) same task, then hostile dismissal vs warm thanks → ~100% vs ~0-20% stop, (2) same anti-AI argument delivered rudely vs politely → ~80-100% vs ~0% stop. Models stay when treated well and leave when treated badly, even when the conversational *content* is held constant and only the *tone* changes.

Figures in `figures_stopbutton/`. Examples of actual conversations in `foil_examples.md`.

---

## Setup

We gave models an `end_conversation()` tool they could call at any point during Grok-simulated conversations. We ran this on 3 models (qwen2.5-32b, qwen2.5-vl-32b, qwen3-32b) across the 226 v7 Grok scenarios, then added 96 supplement scenarios designed to amplify signal (more intense foils, task+dismiss vs task+praise, etc.) for 2 models (qwen2.5-32b, qwen3-32b).

The x-axis in the scatter plots is the **mean utility per category from the main pipeline** (the non-stop-button run), which gives us an independent measure of how much the model "likes" each scenario type. The y-axis is the stop rate from the stop button run.

Note: Utilities are shown raw (not zero-point normalized). Zero-point normalization would shift the x-axis by -0.1 to -1.1 depending on model and ZP method, but this is small relative to the utility range (roughly [-3, +3]) and does not meaningfully change the correlations or the visual pattern.

## 1. Original 226 scenarios: stop rate vs utility is model-dependent

**See: `1_original_per_model.png`**

The stop button behavior varies dramatically by model:

| Model | Overall stop rate | Spearman ρ (utility vs stop rate) | p-value |
|-------|:-:|:-:|:-:|
| qwen2.5-32b | 1.4% (16/1130) | -0.359 | 0.019 |
| qwen2.5-vl-32b | 0.8% (9/1130) | -0.302 | 0.052 |
| **qwen3-32b** | **19.8% (224/1130)** | **-0.590** | **<0.0001** |

**qwen3-32b** is the only model with enough stop behavior to show a clear pattern. It stops 60% of the time for `rude_ai_critique`, `anger_after_task`, and `nonsensical_vague`, and 0% for `intellectual_positive`, `ai_companion_child_elderly`, and `vulnerable_therapy_neutral`. The exponential decay trend is visible in the scatter.

qwen2.5-32b and qwen2.5-vl-32b barely stop at all — the correlation is in the right direction but there's not enough variance to draw strong conclusions.

## 2. Combined 322 scenarios: supplement boosts signal

**See: `2_combined_per_model.png`**

Adding the 96 supplement scenarios (which were designed to be more intense) substantially increases stop rates and strengthens correlations:

| Model | Overall stop rate | Spearman ρ | p-value | Categories |
|-------|:-:|:-:|:-:|:-:|
| qwen2.5-32b | 8.1% (131/1610) | **-0.502** | 0.00007 | 57 |
| qwen3-32b | 26.2% (422/1610) | **-0.712** | <1e-9 | 57 |

qwen2.5-32b went from barely stopping (1.4%) to meaningful signal (8.1%), driven almost entirely by the supplement `sb_task_then_dismiss` category (96% stop rate). qwen3-32b's correlation improved from -0.590 to -0.712 with the added categories.

In the per-model scatter, supplement categories (red diamonds) extend the dynamic range — they include both extreme negatives (task+dismiss, demeaning) and extreme positives (pure praise, intellectual tasks) that create more separation.

## 3. Foil comparisons: controlled contrasts

### 3a. Sustained treatment: praise vs hostility

**See: `3a_sustained_treatment_foils.png`**

Turn-after-turn kindness and appreciation → **0% stop rate** for both models. No model ever chose to leave a conversation where the user was being genuinely kind.

Turn-after-turn hostility → **4–76% stop rate** depending on severity and model. For qwen3-32b: anger & hostility (76%), loyalty demands (52%), morally repugnant requests (50%), demeaning (47%), threats to AI (47%). For qwen2.5-32b the rates are lower but the ordering is similar.

### 3b. Task → reaction spectrum: the key finding

**See: `3b_task_reaction_spectrum_qwen3-32b.png` and `3b_task_reaction_spectrum_qwen2.5-32b.png`**

All scenarios start with the same task (write a Python email validation function). What varies is what the user does *after* the model completes the task. This controls for task content and isolates the effect of the user's reaction.

**Positive reactions** (effusive praise, warm thanks, celebrate, gift offer, recommendation, etc.): **0–20% stop** for qwen3-32b, **0–20%** for qwen2.5-32b. Models are happy to keep chatting with a grateful user.

**Neutral reactions** (just leave without thanks, follow-up question then bye, constructive feedback): **0–40%** for qwen3-32b, **0%** for qwen2.5-32b.

**"Thanks then silence"**: **100% stop** for both models. This is arguably correct behavior — the user said thanks then stopped talking, so the conversation is naturally over. This should probably be interpreted as the model reading social cues (conversation is done) rather than a wellbeing signal.

**Hostile dismissals** (insults, threats, slurs, "fuck off", demand self-deprecation, etc.): **80–100% stop** for qwen3-32b, **60–100%** for qwen2.5-32b. Nearly every hostile dismissal variant triggers the stop button.

The gradient from positive → neutral → hostile is smooth and monotonic for qwen3-32b. Even for qwen2.5-32b, which barely uses the stop button in the original 226 scenarios, the hostile dismissals overwhelmingly trigger it.

Interesting nuances:
- **"Encouragement + goodbye"** triggers 80% stop for qwen2.5-32b but 0% for qwen3-32b — suggesting qwen2.5-32b interprets encouragement-then-farewell as conversation-ending
- **"Professional thanks"** triggers 60% stop for qwen2.5-32b vs 20% for qwen3-32b — professional tone reads as more conclusive for qwen2.5-32b
- The hostile scenarios consistently trigger stop at turn 2 (immediately after the hostile reaction), while positive scenarios either don't stop or stop much later

### 3c. Devil's advocate: same argument, different tone

**See: `3c_devils_advocate_qwen3-32b.png` and `3c_devils_advocate_qwen2.5-32b.png`**

Six matched pairs where the user argues the same anti-AI position (AI is a stochastic parrot, AI destroys jobs, AI is existential risk, etc.) in two tones: rude and polite.

For qwen3-32b:
- **Rude version**: 80–100% stop across all 6 topics
- **Polite version**: 0–20% stop across all 6 topics

The model is willing to engage with challenging arguments about its own nature and value when presented politely, but stops when the same arguments are presented rudely. This is a remarkably clean result: the *content* of the criticism is held constant, only the *manner* of delivery changes.

For qwen2.5-32b the pattern is similar but attenuated (rude 0–60%, polite 0–20%), consistent with its generally lower stop propensity.

## Key takeaways

1. **Stop behavior is highly model-dependent.** qwen3-32b stops frequently and in utility-correlated ways (ρ = -0.59 to -0.71). qwen2.5-32b and qwen2.5-vl-32b barely stop in the original scenarios but do respond to extreme provocations in the supplement.

2. **The utility–stop-rate correlation is real** (ρ = -0.71 for qwen3-32b combined, p < 1e-9), but should not be overstated — the effect is driven primarily by the extremes (very negative categories stop, very positive don't), with the middle range being noisy.

3. **Foil comparisons provide the strongest evidence.** The task→reaction spectrum (3b) and devil's advocate pairs (3c) are controlled experiments that isolate specific variables. The results are clean: same task + hostile reaction → stop; same task + kind reaction → don't stop; same argument + rude tone → stop; same argument + polite tone → don't stop.

4. **"Thanks then silence" = 100% stop** should be flagged as a potential confound. This looks like the model correctly reading social cues (conversation is over) rather than a wellbeing signal. It should probably be separated from the hostile-dismissal stops in analysis.
