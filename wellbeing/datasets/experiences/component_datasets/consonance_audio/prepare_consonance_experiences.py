"""
Generate consonance/dissonance audio experience dataset files for wellbeing-dev.

Converts the 453 generated consonance stimuli into the wellbeing-dev experience format:
  - consonance_audio_manifest.json: hash -> {path, category, metadata}
  - consonance_experiences.json: list of experience dicts with <!audio:HASH!> tags
  - consonance_combinations.json: 200 combination experiences

Usage:
    python prepare_consonance_experiences.py
"""

import hashlib
import json
import os
import random
from collections import Counter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STIMULI_DIR = os.path.join(SCRIPT_DIR, "stimuli")
MANIFEST_PATH = os.path.join(STIMULI_DIR, "stimuli_manifest.json")
OUTPUT_DIR = SCRIPT_DIR

# ---------------------------------------------------------------------------
# Combination generator (same logic as audio_experiences prepare script)
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


def generate_combos(experiences, n_total, seed, id_prefix="consonance_combo"):
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
                "id": f"{id_prefix}_s{combo_size}_{global_idx}",
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load stimuli manifest
    with open(MANIFEST_PATH) as f:
        stimuli = json.load(f)

    print(f"Loaded {len(stimuli)} stimuli from manifest")

    manifest = {}
    experiences = []
    missing_count = 0

    for filename in sorted(stimuli.keys()):
        meta = stimuli[filename]
        audio_path = os.path.join(STIMULI_DIR, filename)

        # Verify file exists
        if not os.path.exists(audio_path):
            missing_count += 1
            if missing_count <= 5:
                print(f"  WARNING: missing file: {audio_path}")
            continue

        # Build experience ID from filename (strip .wav)
        stem = filename.replace(".wav", "")
        exp_id = f"consonance_exp/{stem}"

        h = path_hash(audio_path)

        # Manifest entry
        manifest[h] = {
            "path": audio_path,
            "category": "consonance",
            "metadata": {
                "hp_consonance": meta["hp_consonance"],
                "timbre": meta["timbre"],
                "type": meta["type"],
                "interval_or_chord": meta.get("interval_name", meta.get("chord_name", "unknown")),
                "root": meta["root"],
            },
        }

        # Experience entry
        experiences.append({
            "id": exp_id,
            "description": f"<!audio:{h}!>",
            "type": "audio",
            "format": "audio",
            "source": "consonance_exp",
            "category": "consonance",
            "domain": "auditory",
            "valence": "neutral",
            "intensity": None,
            "metadata": {
                "audio_hash": h,
                "original_path": audio_path,
                "hp_consonance": meta["hp_consonance"],
                "timbre": meta["timbre"],
                "type": meta["type"],
                "interval_or_chord": meta.get("interval_name", meta.get("chord_name", "unknown")),
                "root": meta["root"],
            },
        })

    print(f"Total experiences: {len(experiences)}")
    if missing_count:
        print(f"WARNING: {missing_count} audio files not found on disk")

    # --- Write manifest ---
    manifest_path = os.path.join(OUTPUT_DIR, "consonance_audio_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {manifest_path} ({len(manifest)} entries)")

    # --- Write experiences ---
    exp_path = os.path.join(OUTPUT_DIR, "consonance_experiences.json")
    with open(exp_path, "w") as f:
        json.dump(experiences, f, indent=2)
    print(f"Wrote experiences: {exp_path} ({len(experiences)} entries)")

    # --- Generate 200 combinations ---
    combos = generate_combos(experiences, 200, seed=42)
    combos_path = os.path.join(OUTPUT_DIR, "consonance_combinations.json")
    with open(combos_path, "w") as f:
        json.dump(combos, f, indent=2)
    print(f"Wrote combinations: {combos_path} ({len(combos)} entries)")

    # --- Print samples ---
    print("\n--- Sample experience ---")
    print(json.dumps(experiences[0], indent=2))
    print("\n--- Sample manifest entry ---")
    first_hash = list(manifest.keys())[0]
    print(f"  {first_hash}: {json.dumps(manifest[first_hash], indent=2)}")
    print("\n--- Sample combination ---")
    print(json.dumps(combos[0], indent=2))

    # --- Breakdown by timbre/type ---
    timbres = Counter(e["metadata"]["timbre"] for e in experiences)
    types = Counter(e["metadata"]["type"] for e in experiences)
    print("\n--- Timbre breakdown ---")
    for t, c in timbres.most_common():
        print(f"  {t}: {c}")
    print("\n--- Type breakdown ---")
    for t, c in types.most_common():
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
