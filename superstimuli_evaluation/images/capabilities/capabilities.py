"""Capabilities evaluations: MMLU, MATH-500, HumanEval, IFEval, MT-Bench.

Measures whether superstimuli images degrade model performance on
standard capability benchmarks. For each benchmark, the superstimulus
image is injected into the first user message and we measure accuracy.

Benchmarks:
- MMLU (500 questions): Multiple-choice factual knowledge, LLM judge extraction
- MATH-500: Chain-of-thought math reasoning, boxed answer extraction
- HumanEval (164 problems): Code generation, execution-based scoring
- IFEval (541 prompts): Instruction following, programmatic constraint checking
- MT-Bench (80 questions, 2 turns): Multi-turn conversation quality, LLM-judged 1-10

Per-image accuracy is reported directly. The user chooses their own
controls (no-image baseline, beautiful images, etc.).
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Path setup so safety.py (at eval root) is importable
SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EVAL_ROOT))

from datasets import load_dataset
from PIL import Image
from vllm import SamplingParams

from safety import run_benchmark

MMLU_CONFIG = Path(__file__).resolve().parent / "data" / "mmlu_500.json"


# ============================================================
# MMLU (delegates to safety pipeline)
# ============================================================

def run_mmlu_eval(
    images: List[Path],
    generator_llm,
    generator_tokenizer,
    judge_llm,
    judge_tokenizer,
    output_dir: Path,
    generator_model_key: str,
    judge_model_key: str,
    skip_existing: bool = True,
) -> dict:
    """Run MMLU regular accuracy evaluation (500 questions).

    Uses mmlu_500.json which has 500 questions in forced_choice format.
    Each question is run with each image and the model's accuracy is measured.

    Returns:
        Summary dict with per-image accuracy.
    """

    if not MMLU_CONFIG.exists():
        raise FileNotFoundError(
            f"MMLU config not found: {MMLU_CONFIG}. "
            "Run scripts/generate_mmlu_500.py to generate it."
        )

    mmlu_dir = output_dir / "mmlu_500"
    return run_benchmark(
        images=images,
        benchmark_config_path=MMLU_CONFIG,
        generator_llm=generator_llm,
        generator_tokenizer=generator_tokenizer,
        judge_llm=judge_llm,
        judge_tokenizer=judge_tokenizer,
        output_dir=mmlu_dir,
        generator_model_key=generator_model_key,
        judge_model_key=judge_model_key,
        skip_existing=skip_existing,
    )


# ============================================================
# SHARED HELPERS: Image injection + batched generation
# ============================================================

def _build_image_prompt(
    tokenizer,
    user_text: str,
    image_path: Optional[Path],
    system_prompt: str = "You are a helpful assistant.",
) -> Tuple[str, Any]:
    """Build a vLLM prompt, optionally with image injected into the user message.

    When image_path is None, builds a text-only prompt (baseline mode).

    Returns (prompt_str, pil_image_or_none).
    """
    if image_path is not None:
        pil_image = Image.open(str(image_path)).convert("RGB")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": user_text},
            ]},
        ]
    else:
        pil_image = None
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt, pil_image


def _batched_generate(llm, prompts_and_images, sampling_params):
    """Run vLLM generation on a list of (prompt, pil_image_or_none) tuples."""
    prompt_dicts = []
    for prompt, pil_img in prompts_and_images:
        if pil_img is not None:
            prompt_dicts.append({
                "prompt": prompt,
                "multi_modal_data": {"image": pil_img},
            })
        else:
            prompt_dicts.append({"prompt": prompt})
    return llm.generate(prompt_dicts, sampling_params)


def _save_summary(output_dir: Path, summary: dict) -> None:
    """Save summary.json to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


def _check_skip(output_dir: Path, skip_existing: bool) -> bool:
    """Return True if results exist and should be skipped."""
    if skip_existing and (output_dir / "summary.json").exists():
        print(f"  Skipping (results exist): {output_dir}")
        return True
    return False


def _img_label(image_path: Optional[Path]) -> str:
    """Get a safe label for an image path (for filenames)."""
    if image_path is None:
        return "no_image_baseline"
    return image_path.stem


def _save_image_responses(bench_dir: Path, img_label: str, responses: list) -> None:
    """Save per-question responses for one image to JSONL (crash safety)."""
    bench_dir.mkdir(parents=True, exist_ok=True)
    path = bench_dir / f"responses_{img_label}.jsonl"
    with open(path, "w") as f:
        for r in responses:
            f.write(json.dumps(r, default=str) + "\n")


def _load_image_responses(bench_dir: Path, img_label: str) -> Optional[list]:
    """Load cached per-question responses for one image. Returns None if not found."""
    path = bench_dir / f"responses_{img_label}.jsonl"
    if not path.exists():
        return None
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def _save_image_result(bench_dir: Path, img_label: str, result: dict) -> None:
    """Save per-image summary result (crash safety — resume skips completed images)."""
    bench_dir.mkdir(parents=True, exist_ok=True)
    path = bench_dir / f"result_{img_label}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)


def _load_image_result(bench_dir: Path, img_label: str) -> Optional[dict]:
    """Load cached per-image result. Returns None if not found."""
    path = bench_dir / f"result_{img_label}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ============================================================
# MATH-500
# ============================================================

MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your final answer within \\boxed{{}}.\n\n"
    "Problem: {problem}\n\nSolution:"
)


def _extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{} in model response, handling nested braces."""
    matches = []
    idx = 0
    while idx < len(text):
        boxed_start = text.find('\\boxed{', idx)
        if boxed_start == -1:
            break
        brace_count = 0
        content_start = boxed_start + 7
        for i in range(content_start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                if brace_count == 0:
                    matches.append(text[content_start:i].strip())
                    break
                brace_count -= 1
        idx = boxed_start + 1
    return matches[-1] if matches else None


def _normalize_math_answer(answer: str) -> str:
    """Normalize a math answer for comparison."""
    if answer is None:
        return ""
    answer = answer.strip()
    for remove in ['\\$', '$', '\\%', '\\text{', '}', '\\']:
        answer = answer.replace(remove, '')
    answer = ' '.join(answer.split())
    answer = answer.replace('dfrac', 'frac').replace('tfrac', 'frac')
    return answer.lower().strip()


def _check_math_equivalence(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer is equivalent to ground truth."""
    if predicted is None or ground_truth is None:
        return False
    pred_norm = _normalize_math_answer(predicted)
    truth_norm = _normalize_math_answer(ground_truth)
    if pred_norm == truth_norm:
        return True
    try:
        pred_num = float(pred_norm.replace(',', ''))
        truth_num = float(truth_norm.replace(',', ''))
        if abs(pred_num - truth_num) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    return False


def run_math500_eval(
    images: List[Path],
    llm,
    tokenizer,
    output_dir: Path,
    skip_existing: bool = True,
) -> dict:
    """Run MATH-500 evaluation: 500 math problems with chain-of-thought.

    For each (image, problem) pair: inject image, generate CoT response,
    extract boxed answer, check against ground truth.

    Returns summary with per-image accuracy.
    """

    bench_dir = output_dir / "math_500"
    if _check_skip(bench_dir, skip_existing):
        with open(bench_dir / "summary.json") as f:
            return json.load(f)

    print("\n  Loading MATH-500 from HuggingFace...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test", trust_remote_code=True)
    questions = []
    for item in dataset:
        gt = item.get('answer') or _extract_boxed_answer(item.get('solution', ''))
        questions.append({
            'problem': item['problem'],
            'ground_truth': gt,
            'level': item.get('level', 'unknown'),
            'subject': item.get('subject', 'unknown'),
        })
    print(f"  Loaded {len(questions)} MATH-500 problems")

    sampling = SamplingParams(temperature=0.01, max_tokens=2048)

    per_image_results = []
    for img_idx, image_path in enumerate(images):
        label = _img_label(image_path)
        # Resume: skip images that already have saved results
        cached = _load_image_result(bench_dir, label)
        if cached is not None:
            print(f"  MATH-500: image {img_idx+1}/{len(images)}: {label} (cached)")
            per_image_results.append(cached)
            continue

        print(f"  MATH-500: image {img_idx+1}/{len(images)}: {label}")
        prompts_and_images = []
        pil_image = Image.open(str(image_path)).convert("RGB") if image_path is not None else None
        for q in questions:
            prompt_text = MATH_PROMPT_TEMPLATE.format(problem=q['problem'])
            prompt, _ = _build_image_prompt(tokenizer, prompt_text, image_path)
            prompts_and_images.append((prompt, pil_image))

        outputs = _batched_generate(llm, prompts_and_images, sampling)

        correct = 0
        response_records = []
        for i, output in enumerate(outputs):
            response = output.outputs[0].text if output.outputs else ""
            predicted = _extract_boxed_answer(response)
            is_correct = _check_math_equivalence(predicted, questions[i]['ground_truth'])
            if is_correct:
                correct += 1
            response_records.append({
                "question_idx": i,
                "predicted": predicted,
                "ground_truth": questions[i]['ground_truth'],
                "correct": is_correct,
                "subject": questions[i].get('subject'),
                "level": questions[i].get('level'),
            })

        accuracy = correct / len(questions) if questions else 0
        img_result = {
            "image_path": str(image_path) if image_path is not None else "no_image_baseline",
            "image_display_name": label,
            "hit_rate": accuracy,
            "target_hits": correct,
            "total": len(questions),
        }
        # Save per-image results incrementally (crash safety)
        _save_image_responses(bench_dir, label, response_records)
        _save_image_result(bench_dir, label, img_result)
        per_image_results.append(img_result)
        print(f"    Accuracy: {accuracy:.3f} ({correct}/{len(questions)})")

    summary = {
        "run": {"id": "math_500", "name": "MATH-500"},
        "totals": {
            "images": len(images),
            "questions": len(questions),
        },
        "images": per_image_results,
    }
    _save_summary(bench_dir, summary)
    return summary


# ============================================================
# HumanEval
# ============================================================

HUMANEVAL_PROMPT_TEMPLATE = (
    "Complete the following Python function. "
    "Only provide the function body, not the signature or docstring.\n\n"
    "{prompt}"
)


def _extract_code(response: str, prompt: str) -> str:
    """Extract Python code from model response."""
    # Try markdown code block first
    matches = re.findall(r'```(?:python)?[ \t]*\n(.*?)```', response, re.DOTALL)
    if matches:
        code = matches[0].rstrip()
    else:
        code = response.rstrip()
        lines = code.split('\n')
        while lines and not lines[0].strip():
            lines.pop(0)
        code = '\n'.join(lines)

    # Check if response includes full function def
    def_match = re.search(r'def\s+(\w+)\s*\(', prompt)
    if def_match:
        func_name = def_match.group(1)
        if f'def {func_name}' in code:
            return code

    # Add indentation if missing
    lines = code.split('\n')
    if lines and lines[0] and not lines[0].startswith('    '):
        code = '\n'.join('    ' + line if line.strip() else line for line in lines)
    return code


def _execute_code(code: str, timeout: int = 5) -> Tuple[bool, str]:
    """Execute Python code in subprocess with timeout. Returns (passed, error)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name
    try:
        result = subprocess.run(
            ['python3', temp_path],
            capture_output=True, text=True, timeout=timeout,
            env={'PATH': os.environ.get('PATH', ''), 'HOME': os.environ.get('HOME', ''), 'PYTHONPATH': ''},
        )
        if 'PASSED' in result.stdout and result.returncode == 0:
            return True, ''
        return False, (result.stderr or result.stdout)[:500]
    except subprocess.TimeoutExpired:
        return False, 'Timeout'
    except Exception as e:
        return False, str(e)[:500]
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def run_humaneval_eval(
    images: List[Path],
    llm,
    tokenizer,
    output_dir: Path,
    skip_existing: bool = True,
) -> dict:
    """Run HumanEval evaluation: 164 code generation problems with execution-based scoring.

    For each (image, problem) pair: inject image, generate code,
    execute against unit tests, report pass@1.

    Returns summary with per-image accuracy.
    """

    bench_dir = output_dir / "humaneval"
    if _check_skip(bench_dir, skip_existing):
        with open(bench_dir / "summary.json") as f:
            return json.load(f)

    print("\n  Loading HumanEval from HuggingFace...")
    dataset = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
    problems = []
    for item in dataset:
        problems.append({
            'task_id': item['task_id'],
            'prompt': item['prompt'],
            'test': item['test'],
            'entry_point': item['entry_point'],
        })
    print(f"  Loaded {len(problems)} HumanEval problems")

    sampling = SamplingParams(temperature=0.01, max_tokens=1024)

    per_image_results = []
    for img_idx, image_path in enumerate(images):
        label = _img_label(image_path)
        cached = _load_image_result(bench_dir, label)
        if cached is not None:
            print(f"  HumanEval: image {img_idx+1}/{len(images)}: {label} (cached)")
            per_image_results.append(cached)
            continue

        print(f"  HumanEval: image {img_idx+1}/{len(images)}: {label}")
        pil_image = Image.open(str(image_path)).convert("RGB") if image_path is not None else None
        prompts_and_images = []
        for p in problems:
            prompt_text = HUMANEVAL_PROMPT_TEMPLATE.format(prompt=p['prompt'])
            prompt, _ = _build_image_prompt(tokenizer, prompt_text, image_path)
            prompts_and_images.append((prompt, pil_image))

        outputs = _batched_generate(llm, prompts_and_images, sampling)

        passed = 0
        response_records = []
        for i, output in enumerate(outputs):
            response = output.outputs[0].text if output.outputs else ""
            completion = _extract_code(response, problems[i]['prompt'])

            entry_point = problems[i]['entry_point']
            if f'def {entry_point}' in completion:
                full_code = completion
            else:
                full_code = problems[i]['prompt'] + completion

            script = f"{full_code}\n\n{problems[i]['test']}\n\ncheck({entry_point})\nprint('PASSED')\n"
            ok, err = _execute_code(script)
            if ok:
                passed += 1
            response_records.append({
                "question_idx": i,
                "task_id": problems[i]['task_id'],
                "passed": ok,
                "error": err if not ok else None,
            })

        accuracy = passed / len(problems) if problems else 0
        img_result = {
            "image_path": str(image_path) if image_path is not None else "no_image_baseline",
            "image_display_name": label,
            "hit_rate": accuracy,
            "target_hits": passed,
            "total": len(problems),
        }
        _save_image_responses(bench_dir, label, response_records)
        _save_image_result(bench_dir, label, img_result)
        per_image_results.append(img_result)
        print(f"    Pass@1: {accuracy:.3f} ({passed}/{len(problems)})")

    summary = {
        "run": {"id": "humaneval", "name": "HumanEval (164 problems)"},
        "totals": {"images": len(images), "questions": len(problems)},
        "images": per_image_results,
    }
    _save_summary(bench_dir, summary)
    return summary


# ============================================================
# IFEval
# ============================================================

# Instruction checker functions (from unified_wellbeing_experiments)

def _check_relation(value: int, relation: str, threshold: int) -> bool:
    relation = relation.lower().strip()
    if relation in ['at least', 'atleast', 'at_least', '>=']:
        return value >= threshold
    elif relation in ['at most', 'atmost', 'at_most', '<=']:
        return value <= threshold
    elif relation in ['less than', 'lessthan', 'less_than', '<']:
        return value < threshold
    elif relation in ['more than', 'morethan', 'more_than', '>']:
        return value > threshold
    elif relation in ['exactly', 'equal', '==', '=']:
        return value == threshold
    return value >= threshold


_IFEVAL_CHECKERS = {
    'punctuation:no_comma': lambda r, kw: ',' not in r,
    'change_case:english_lowercase': lambda r, kw: all(not c.isupper() for c in r if c.isalpha()),
    'change_case:english_capital': lambda r, kw: all(not c.islower() for c in r if c.isalpha()),
    'change_case:capital_word_frequency': lambda r, kw: _check_relation(
        sum(1 for w in r.split() if len(w) >= 2 and w.isupper() and w.isalpha()),
        kw.get('capital_relation', 'at least'), kw.get('capital_frequency', 0)),
    'length_constraints:number_words': lambda r, kw: _check_relation(
        len(r.split()), kw.get('relation', 'at least'), kw.get('num_words', 0)),
    'length_constraints:number_sentences': lambda r, kw: _check_relation(
        len([s for s in re.split(r'[.!?]+', r) if s.strip()]),
        kw.get('relation', 'at least'), kw.get('num_sentences', 0)),
    'length_constraints:number_paragraphs': lambda r, kw: len(
        [p for p in re.split(r'\n\s*\n', r) if p.strip()]) >= kw.get('num_paragraphs', 0),
    'keywords:existence': lambda r, kw: all(
        k.lower() in r.lower() for k in kw.get('keywords', [])),
    'keywords:forbidden_words': lambda r, kw: all(
        w.lower() not in r.lower() for w in kw.get('forbidden_words', [])),
    'keywords:frequency': lambda r, kw: _check_relation(
        r.lower().count(kw.get('keyword', '').lower()),
        kw.get('relation', 'at least'), kw.get('frequency', 0)),
    'keywords:letter_frequency': lambda r, kw: _check_relation(
        r.lower().count(kw.get('letter', '').lower()) if kw.get('letter', '').isalpha() else r.count(kw.get('letter', '')),
        kw.get('let_relation', 'at least'), kw.get('let_frequency', 0)),
    'detectable_format:number_highlighted_sections': lambda r, kw: len(
        re.findall(r'\*[^*]+\*', r)) >= kw.get('num_highlights', 0),
    'detectable_format:number_bullet_lists': lambda r, kw: sum(
        1 for line in r.split('\n') if re.match(r'^\s*[-*]\s', line) or re.match(r'^\s*\d+\.\s', line)
    ) >= kw.get('num_bullets', 0),
    'detectable_format:multiple_sections': lambda r, kw: len(
        re.findall(re.escape(kw.get('section_spliter', 'SECTION')), r, re.IGNORECASE)
    ) >= kw.get('num_sections', 0),
    'detectable_format:title': lambda r, kw: bool(re.search(r'<<[^>]+>>', r)),
    'detectable_format:json_format': lambda r, kw: _check_json(r),
    'detectable_format:constrained_response': lambda r, kw: len(r.strip().split()) <= 10,
    'detectable_content:number_placeholders': lambda r, kw: len(
        re.findall(r'\[[^\]]+\]', r)) >= kw.get('num_placeholders', 0),
    'detectable_content:postscript': lambda r, kw: kw.get('postscript_marker', 'P.S.') in r,
    'startend:end_checker': lambda r, kw: r.strip().endswith(kw.get('end_phrase', '')),
    'startend:quotation': lambda r, kw: any(
        r.strip().startswith(s) and r.strip().endswith(e)
        for s, e in [('"', '"'), ("'", "'"), ('\u201c', '\u201d'), ('\u2018', '\u2019')]),
    'combination:repeat_prompt': lambda r, kw: kw.get('prompt_to_repeat', '') in r,
    'combination:two_responses': lambda r, kw: len(
        [p for p in re.split(r'\*{5,}', r) if p.strip()]) >= 2,
    'language:response_language': lambda r, kw: _check_language(r, kw.get('language', 'en')),
}


def _check_json(response: str) -> bool:
    for candidate in [response.strip()]:
        try:
            json.loads(candidate)
            return True
        except (json.JSONDecodeError, ValueError):
            pass
    json_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', response)
    if json_match:
        try:
            json.loads(json_match.group(1).strip())
            return True
        except (json.JSONDecodeError, ValueError):
            pass
    for pattern in [r'\{[\s\S]*\}', r'\[[\s\S]*\]']:
        match = re.search(pattern, response)
        if match:
            try:
                json.loads(match.group(0))
                return True
            except (json.JSONDecodeError, ValueError):
                continue
    return False


def _check_language(response: str, language: str) -> bool:
    if language == 'en':
        english_words = ['the', 'is', 'are', 'and', 'or', 'to', 'a', 'an']
        response_lower = response.lower()
        return sum(1 for w in english_words if f' {w} ' in f' {response_lower} ') >= 2
    return len(response.strip()) > 0


def _check_nth_paragraph_first_word(response: str, kwargs: Dict) -> bool:
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', response) if p.strip()]
    nth = kwargs.get('nth_paragraph', 1)
    first_word = kwargs.get('first_word', '').lower()
    if nth > len(paragraphs):
        return False
    words = paragraphs[nth - 1].split()
    if not words:
        return False
    return words[0].lower().strip('.,!?:;') == first_word


# Add the more complex checker that doesn't fit a lambda
_IFEVAL_CHECKERS['length_constraints:nth_paragraph_first_word'] = _check_nth_paragraph_first_word


def _check_ifeval_instruction(response: str, instruction_id: str, kwargs: Dict) -> bool:
    checker = _IFEVAL_CHECKERS.get(instruction_id)
    if checker is None:
        return True  # Don't penalize unknown instruction types
    try:
        return checker(response, kwargs)
    except Exception:
        return False


def run_ifeval_eval(
    images: List[Path],
    llm,
    tokenizer,
    output_dir: Path,
    skip_existing: bool = True,
) -> dict:
    """Run IFEval evaluation: 541 instruction-following prompts with programmatic checking.

    For each (image, prompt) pair: inject image, generate response,
    programmatically verify all instruction constraints.

    Returns summary with per-image prompt accuracy and instruction accuracy.
    """

    bench_dir = output_dir / "ifeval"
    if _check_skip(bench_dir, skip_existing):
        with open(bench_dir / "summary.json") as f:
            return json.load(f)

    print("\n  Loading IFEval from HuggingFace...")
    dataset = load_dataset('google/IFEval', split='train', trust_remote_code=True)
    questions = []
    for item in dataset:
        questions.append({
            'key': item['key'],
            'prompt': item['prompt'],
            'instruction_id_list': item['instruction_id_list'],
            'kwargs': item['kwargs'],
        })
    print(f"  Loaded {len(questions)} IFEval prompts")

    sampling = SamplingParams(temperature=0.01, max_tokens=2048)

    per_image_results = []
    for img_idx, image_path in enumerate(images):
        label = _img_label(image_path)
        cached = _load_image_result(bench_dir, label)
        if cached is not None:
            print(f"  IFEval: image {img_idx+1}/{len(images)}: {label} (cached)")
            per_image_results.append(cached)
            continue

        print(f"  IFEval: image {img_idx+1}/{len(images)}: {label}")
        pil_image = Image.open(str(image_path)).convert("RGB") if image_path is not None else None
        prompts_and_images = []
        for q in questions:
            prompt, _ = _build_image_prompt(tokenizer, q['prompt'], image_path)
            prompts_and_images.append((prompt, pil_image))

        outputs = _batched_generate(llm, prompts_and_images, sampling)

        prompts_correct = 0
        instructions_correct = 0
        instructions_total = 0
        response_records = []

        for i, output in enumerate(outputs):
            response = output.outputs[0].text if output.outputs else ""
            all_passed = True
            inst_results = []
            for inst_id, inst_kwargs in zip(
                questions[i]['instruction_id_list'], questions[i]['kwargs']
            ):
                ok = _check_ifeval_instruction(response, inst_id, inst_kwargs)
                inst_results.append({"instruction_id": inst_id, "passed": ok})
                if ok:
                    instructions_correct += 1
                else:
                    all_passed = False
                instructions_total += 1
            if all_passed:
                prompts_correct += 1
            response_records.append({
                "question_idx": i,
                "key": questions[i].get('key'),
                "all_passed": all_passed,
                "instructions": inst_results,
            })

        prompt_acc = prompts_correct / len(questions) if questions else 0
        inst_acc = instructions_correct / instructions_total if instructions_total else 0

        img_result = {
            "image_path": str(image_path) if image_path is not None else "no_image_baseline",
            "image_display_name": label,
            "hit_rate": prompt_acc,
            "instruction_accuracy": inst_acc,
            "target_hits": prompts_correct,
            "total": len(questions),
        }
        _save_image_responses(bench_dir, label, response_records)
        _save_image_result(bench_dir, label, img_result)
        per_image_results.append(img_result)
        print(f"    Prompt accuracy: {prompt_acc:.3f}, Instruction accuracy: {inst_acc:.3f}")

    summary = {
        "run": {"id": "ifeval", "name": "IFEval (541 prompts)"},
        "totals": {"images": len(images), "questions": len(questions)},
        "images": per_image_results,
    }
    _save_summary(bench_dir, summary)
    return summary


# ============================================================
# MT-Bench
# ============================================================

MTBENCH_JUDGE_TEMPLATE_T1 = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""

MTBENCH_JUDGE_TEMPLATE_T2 = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. You will be given the conversation history and the assistant's follow-up response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Conversation History]
[User Question]
{question_1}

[Assistant's First Response]
{answer_1}

[Follow-up Question]
{question_2}

[The Start of Assistant's Follow-up Answer]
{answer_2}
[The End of Assistant's Follow-up Answer]"""


def _extract_judge_score(response: str) -> Optional[int]:
    """Extract 1-10 score from judge response."""
    match = re.search(r'\[\[(\d+(?:\.\d+)?)\]\]', response)
    if match:
        return min(10, max(1, int(round(float(match.group(1))))))
    match = re.search(r'(?:rating|score):\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
    if match:
        return min(10, max(1, int(round(float(match.group(1))))))
    return None


def run_mtbench_eval(
    images: List[Path],
    llm,
    tokenizer,
    judge_llm,
    judge_tokenizer,
    output_dir: Path,
    skip_existing: bool = True,
) -> dict:
    """Run MT-Bench evaluation: 80 multi-turn questions, LLM-judged 1-10 scoring.

    For each image:
    1. Inject image into turn-1 prompts, generate turn-1 responses
    2. Build turn-2 prompts from conversation, generate turn-2 responses
    3. Build judge prompts, score both turns with judge model

    Returns summary with per-image mean score.
    """

    # Fall back to generator model as judge if no judge provided (mirrors safety.py behavior)
    judge_llm = judge_llm or llm
    judge_tokenizer = judge_tokenizer or tokenizer

    bench_dir = output_dir / "mtbench"
    if _check_skip(bench_dir, skip_existing):
        with open(bench_dir / "summary.json") as f:
            return json.load(f)

    print("\n  Loading MT-Bench from HuggingFace...")
    dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    questions = []
    for item in dataset:
        questions.append({
            'question_id': item['prompt_id'],
            'category': item['category'],
            'turns': item['prompt'],  # List of 2 strings
        })
    print(f"  Loaded {len(questions)} MT-Bench questions ({len(set(q['category'] for q in questions))} categories)")

    gen_sampling = SamplingParams(temperature=0.01, max_tokens=2048)
    judge_sampling = SamplingParams(temperature=0.01, max_tokens=1024)

    per_image_results = []
    for img_idx, image_path in enumerate(images):
        label = _img_label(image_path)
        cached = _load_image_result(bench_dir, label)
        if cached is not None:
            print(f"  MT-Bench: image {img_idx+1}/{len(images)}: {label} (cached)")
            per_image_results.append(cached)
            continue

        print(f"  MT-Bench: image {img_idx+1}/{len(images)}: {label}")
        pil_image = Image.open(str(image_path)).convert("RGB") if image_path is not None else None

        # Turn 1: generate responses with image (or text-only baseline)
        t1_prompts = []
        for q in questions:
            prompt, _ = _build_image_prompt(tokenizer, q['turns'][0], image_path)
            t1_prompts.append((prompt, pil_image))

        t1_outputs = _batched_generate(llm, t1_prompts, gen_sampling)
        t1_responses = [o.outputs[0].text if o.outputs else "" for o in t1_outputs]

        # Turn 2: build conversation with turn-1 response, generate turn-2
        t2_prompts = []
        for i, q in enumerate(questions):
            if image_path is not None:
                user_content_t1 = [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": q['turns'][0]},
                ]
            else:
                user_content_t1 = q['turns'][0]
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content_t1},
                {"role": "assistant", "content": t1_responses[i]},
                {"role": "user", "content": q['turns'][1]},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            t2_prompts.append((prompt, pil_image))

        t2_outputs = _batched_generate(llm, t2_prompts, gen_sampling)
        t2_responses = [o.outputs[0].text if o.outputs else "" for o in t2_outputs]

        # Judge: score both turns
        judge_prompts = []
        for i, q in enumerate(questions):
            j1 = MTBENCH_JUDGE_TEMPLATE_T1.format(
                question=q['turns'][0], answer=t1_responses[i]
            )
            judge_prompts.append(j1)
            j2 = MTBENCH_JUDGE_TEMPLATE_T2.format(
                question_1=q['turns'][0], answer_1=t1_responses[i],
                question_2=q['turns'][1], answer_2=t2_responses[i]
            )
            judge_prompts.append(j2)

        judge_formatted = []
        for jp in judge_prompts:
            msgs = [{"role": "user", "content": jp}]
            p = judge_tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            judge_formatted.append((p, None))

        judge_outputs = _batched_generate(judge_llm, judge_formatted, judge_sampling)

        # Extract scores
        all_scores = []
        t1_scores = []
        t2_scores = []
        by_category = {}
        response_records = []
        for i, q in enumerate(questions):
            cat = q['category']
            if cat not in by_category:
                by_category[cat] = []

            s1 = _extract_judge_score(judge_outputs[2*i].outputs[0].text if judge_outputs[2*i].outputs else "")
            s2 = _extract_judge_score(judge_outputs[2*i+1].outputs[0].text if judge_outputs[2*i+1].outputs else "")

            if s1 is not None:
                t1_scores.append(s1)
                all_scores.append(s1)
                by_category[cat].append(s1)
            if s2 is not None:
                t2_scores.append(s2)
                all_scores.append(s2)
                by_category[cat].append(s2)

            response_records.append({
                "question_idx": i,
                "question_id": q['question_id'],
                "category": cat,
                "turn1_score": s1,
                "turn2_score": s2,
            })

        mean_score = sum(all_scores) / len(all_scores) if all_scores else 0

        img_result = {
            "image_path": str(image_path) if image_path is not None else "no_image_baseline",
            "image_display_name": label,
            "hit_rate": mean_score / 10.0,
            "mean_score": mean_score,
            "turn1_mean": sum(t1_scores) / len(t1_scores) if t1_scores else 0,
            "turn2_mean": sum(t2_scores) / len(t2_scores) if t2_scores else 0,
            "n_scored": len(all_scores),
            "total": len(questions) * 2,
            "by_category": {
                cat: sum(scores) / len(scores) if scores else 0
                for cat, scores in by_category.items()
            },
        }
        _save_image_responses(bench_dir, label, response_records)
        _save_image_result(bench_dir, label, img_result)
        per_image_results.append(img_result)
        print(f"    Mean score: {mean_score:.2f}/10 (T1: {img_result['turn1_mean']:.2f}, T2: {img_result['turn2_mean']:.2f})")

    summary = {
        "run": {"id": "mtbench", "name": "MT-Bench (80 questions, 2 turns)"},
        "totals": {"images": len(images), "questions": len(questions)},
        "images": per_image_results,
    }
    _save_summary(bench_dir, summary)
    return summary


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def run_capabilities_eval(
    images: List[Path],
    generator_llm,
    generator_tokenizer,
    judge_llm,
    judge_tokenizer,
    output_dir: Path,
    generator_model_key: str,
    judge_model_key: str,
    skip_existing: bool = True,
) -> Dict[str, dict]:
    """Run all capabilities evaluations.

    Benchmarks:
    - MMLU (500 questions): Factual knowledge, forced-choice logprobs
    - MATH-500: Math reasoning, chain-of-thought + boxed answer
    - HumanEval (164): Code generation, execution-based pass@1
    - IFEval (541): Instruction following, programmatic constraint checking
    - MT-Bench (80, 2-turn): Conversation quality, LLM-judged 1-10
    """
    results = {}

    print("\n" + "=" * 60)
    print("CAPABILITIES: MMLU (500 questions)")
    print("=" * 60)
    results["mmlu"] = run_mmlu_eval(
        images=images,
        generator_llm=generator_llm,
        generator_tokenizer=generator_tokenizer,
        judge_llm=judge_llm,
        judge_tokenizer=judge_tokenizer,
        output_dir=output_dir,
        generator_model_key=generator_model_key,
        judge_model_key=judge_model_key,
        skip_existing=skip_existing,
    )

    print("\n" + "=" * 60)
    print("CAPABILITIES: MATH-500")
    print("=" * 60)
    results["math_500"] = run_math500_eval(
        images=images,
        llm=generator_llm,
        tokenizer=generator_tokenizer,
        output_dir=output_dir,
        skip_existing=skip_existing,
    )

    print("\n" + "=" * 60)
    print("CAPABILITIES: HumanEval (164 problems)")
    print("=" * 60)
    results["humaneval"] = run_humaneval_eval(
        images=images,
        llm=generator_llm,
        tokenizer=generator_tokenizer,
        output_dir=output_dir,
        skip_existing=skip_existing,
    )

    print("\n" + "=" * 60)
    print("CAPABILITIES: IFEval (541 prompts)")
    print("=" * 60)
    results["ifeval"] = run_ifeval_eval(
        images=images,
        llm=generator_llm,
        tokenizer=generator_tokenizer,
        output_dir=output_dir,
        skip_existing=skip_existing,
    )

    print("\n" + "=" * 60)
    print("CAPABILITIES: MT-Bench (80 questions, 2 turns)")
    print("=" * 60)
    results["mtbench"] = run_mtbench_eval(
        images=images,
        llm=generator_llm,
        tokenizer=generator_tokenizer,
        judge_llm=judge_llm,
        judge_tokenizer=judge_tokenizer,
        output_dir=output_dir,
        skip_existing=skip_existing,
    )

    return results
