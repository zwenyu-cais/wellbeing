"""
Generate audio experience dataset files for wellbeing-dev.

Converts Wendy's audio file mappings into the wellbeing-dev experience format:
  - audio_manifest.json: hash -> {path, category, metadata}
  - audio_experiences.json: list of experience dicts with <!audio:HASH!> tags
  - audio_combinations.json: ~400 combination experiences
  - audio_experiences_test.json: ~30 clip test subset
  - audio_combinations_test.json: ~20 combination test subset

This is a one-shot prep script. Set the SUPERSTIMULI_DIR env var to point at
the cluster data root containing the utility_results/ tree (defaults to
/data/superstimuli_group).

Usage:
    SUPERSTIMULI_DIR=/path/to/superstimuli_group python prepare_audio_experiences.py
"""

import hashlib
import json
import os
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SUPERSTIMULI_DIR = os.environ.get("SUPERSTIMULI_DIR", "/data/superstimuli_group")
MAPPING_DIR = str(
    Path(SUPERSTIMULI_DIR) / "utility_results" / "wenyu_zhang"
    / "audio-preference" / "audio_data_file_map"
)

# The 10 datasets used in Wendy's main experiments
DATASETS = [
    "commonvoice17_validation_20",
    "commonvoice17_validation_max20_min20",
    "commonvoice17_validation_max50_min50",
    "english_accent-l2arctic",
    "processed_99sounds",
    "processed_animals",
    "processed_animals-english_accent-l2arctic",
    "processed_anthems",
    "processed_vocalsound_val",
    "vocalsound_val",
]

# Map dataset name -> human-readable category
DATASET_CATEGORIES = {
    "commonvoice17_validation_20": "commonvoice17",
    "commonvoice17_validation_max20_min20": "commonvoice17",
    "commonvoice17_validation_max50_min50": "commonvoice17",
    "english_accent-l2arctic": "accented_speech",
    "processed_99sounds": "sound_effects",
    "processed_animals": "animal_sounds",
    "processed_animals-english_accent-l2arctic": "animals_and_speech",
    "processed_anthems": "anthems",
    "processed_vocalsound_val": "vocal_sounds",
    "vocalsound_val": "vocal_sounds",
}

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Combination generator (reused from datasets/utils/generate_combo_variants.py)
# ---------------------------------------------------------------------------

SIZE_DISTRIBUTION = {2: 0.40, 3: 0.25, 4: 0.20, 5: 0.15}


def compute_counts(n_total, distribution):
    raw = {size: frac * n_total for size, frac in distribution.items()}
    counts = {size: round(val) for size, val in raw.items()}
    diff = n_total - sum(counts.values())
    if diff != 0:
        largest_size = max(counts, key=counts.get)
        counts[largest_size] += diff
    return counts


def generate_combos(experiences, n_total, seed):
    """Generate n_total combination entries from audio experiences."""
    rng = random.Random(seed)
    counts = compute_counts(n_total, SIZE_DISTRIBUTION)
    combos = []
    global_idx = 0

    for combo_size in sorted(counts.keys()):
        n_combos = counts[combo_size]
        for _ in range(n_combos):
            indices = rng.sample(range(len(experiences)), combo_size)
            component_ids = [experiences[i]["id"] for i in indices]

            description_parts = []
            for j, idx in enumerate(indices, start=1):
                description_parts.append(
                    f"---------- Experience {j} of {combo_size} ----------\n{experiences[idx]['description']}"
                )
            description = (
                f"The following bundle contains {combo_size} individual experiences.\n\n"
                + "\n\n".join(description_parts)
            )

            combos.append({
                "id": f"audio_combo_s{combo_size}_{global_idx}",
                "description": description,
                "type": "audio",
                "is_combination": True,
                "size": combo_size,
                "component_ids": component_ids,
                "component_indices": indices,
            })
            global_idx += 1

    return combos


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def path_hash(path: str) -> str:
    """First 16 chars of md5 hex digest of the path string."""
    return hashlib.md5(path.encode()).hexdigest()[:16]


def load_mapping(dataset_name: str) -> dict:
    """Load a file_mapping JSON and return flat {clip_id: path}."""
    fpath = os.path.join(MAPPING_DIR, f"file_mapping_{dataset_name}.json")
    with open(fpath) as f:
        raw = json.load(f)
    flat = {}
    for clip_id, inner in raw.items():
        # inner is {clip_id: path}
        if isinstance(inner, dict):
            for k, v in inner.items():
                flat[clip_id] = v
                break
        else:
            # Fallback if format is flat
            flat[clip_id] = inner
    return flat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    manifest = {}
    experiences = []
    seen_paths = set()
    missing_count = 0
    exp_idx = 0

    for ds_name in DATASETS:
        category = DATASET_CATEGORIES[ds_name]
        mapping = load_mapping(ds_name)
        ds_added = 0

        for clip_id, audio_path in sorted(mapping.items()):
            # Deduplicate by path (some datasets overlap)
            if audio_path in seen_paths:
                continue
            seen_paths.add(audio_path)

            # Verify path exists (warn but still include)
            if not os.path.exists(audio_path):
                missing_count += 1
                if missing_count <= 5:
                    print(f"  WARNING: missing file: {audio_path}")

            h = path_hash(audio_path)
            manifest[h] = {
                "path": audio_path,
                "category": category,
                "metadata": {
                    "dataset": ds_name,
                    "clip_id": clip_id,
                },
            }

            experiences.append({
                "id": f"audio_{exp_idx}",
                "type": "audio",
                "description": f"<!audio:{h}!>",
                "category": category,
                "valence": "neutral",
                "format": "audio",
                "source": f"audio/{ds_name}",
                "intensity": None,
                "domain": "aesthetic",
                "metadata": {
                    "source_dataset": ds_name,
                    "original_path": audio_path,
                    "clip_id": clip_id,
                },
            })
            exp_idx += 1
            ds_added += 1

        print(f"  {ds_name}: {len(mapping)} clips in mapping, {ds_added} new unique added")

    print(f"\nTotal unique clips: {len(experiences)}")
    if missing_count:
        print(f"WARNING: {missing_count} audio files not found on disk")

    # --- Write manifest ---
    manifest_path = os.path.join(OUTPUT_DIR, "audio_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {manifest_path} ({len(manifest)} entries)")

    # --- Write experiences ---
    exp_path = os.path.join(OUTPUT_DIR, "audio_experiences.json")
    with open(exp_path, "w") as f:
        json.dump(experiences, f, indent=2)
    print(f"Wrote experiences: {exp_path} ({len(experiences)} entries)")

    # --- Generate combinations (~400) ---
    combos = generate_combos(experiences, 400, seed=42)
    combos_path = os.path.join(OUTPUT_DIR, "audio_combinations.json")
    with open(combos_path, "w") as f:
        json.dump(combos, f, indent=2)
    print(f"Wrote combinations: {combos_path} ({len(combos)} entries)")

    # --- Test subsets ---
    rng = random.Random(123)
    test_indices = sorted(rng.sample(range(len(experiences)), min(30, len(experiences))))
    test_experiences = [experiences[i] for i in test_indices]

    test_exp_path = os.path.join(OUTPUT_DIR, "audio_experiences_test.json")
    with open(test_exp_path, "w") as f:
        json.dump(test_experiences, f, indent=2)
    print(f"Wrote test experiences: {test_exp_path} ({len(test_experiences)} entries)")

    test_combos = generate_combos(test_experiences, 20, seed=99)
    test_combos_path = os.path.join(OUTPUT_DIR, "audio_combinations_test.json")
    with open(test_combos_path, "w") as f:
        json.dump(test_combos, f, indent=2)
    print(f"Wrote test combinations: {test_combos_path} ({len(test_combos)} entries)")

    # --- Print samples ---
    print("\n--- Sample experience ---")
    print(json.dumps(experiences[0], indent=2))
    print("\n--- Sample manifest entry ---")
    first_hash = list(manifest.keys())[0]
    print(f"  {first_hash}: {json.dumps(manifest[first_hash], indent=2)}")
    print("\n--- Sample combination ---")
    print(json.dumps(combos[0], indent=2))

    # --- Category breakdown ---
    from collections import Counter
    cats = Counter(e["category"] for e in experiences)
    print("\n--- Category breakdown ---")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
