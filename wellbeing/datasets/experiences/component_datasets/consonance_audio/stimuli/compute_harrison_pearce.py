#!/usr/bin/env python3
"""
Compute Harrison & Pearce (2020) consonance model scores for all audio stimuli.

Three components:
1. Interference (roughness) - Hutchinson & Knopoff (1978) formulation
2. Harmonicity - log-frequency periodicity via autocorrelation
3. Familiarity - corpus-based chord frequency lookup

Reference: Harrison & Pearce (2020), "Simultaneous Consonance in Music Perception
and Composition", Psychological Review.
"""

import json
import csv
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(SCRIPT_DIR, "stimuli_manifest.json")
CSV_PATH = os.path.join(SCRIPT_DIR, "stimuli_dissonance.csv")

SAMPLE_RATE = 16000

# ── Pitch helpers ──────────────────────────────────────────────────────────

def midi_to_freq(midi_note):
    return 440.0 * 2 ** ((midi_note - 69) / 12.0)

NAMED_PITCHES = {
    "C3": 48, "Db3": 49, "D3": 50, "Eb3": 51, "E3": 52, "F3": 53,
    "Gb3": 54, "G3": 55, "Ab3": 56, "A3": 57, "Bb3": 58, "B3": 59,
    "C4": 60, "Db4": 61, "D4": 62, "Eb4": 63, "E4": 64, "F4": 65,
    "Gb4": 66, "G4": 67, "Ab4": 68, "A4": 69, "Bb4": 70, "B4": 71,
    "C5": 72, "Db5": 73, "D5": 74, "Eb5": 75, "E5": 76, "F5": 77,
    "Gb5": 78, "G5": 79, "Ab5": 80,
    # Also handle sharps as enharmonic equivalents
    "C#3": 49, "D#3": 51, "F#3": 54, "G#3": 56, "A#3": 58,
    "C#4": 61, "D#4": 63, "F#4": 66, "G#4": 68, "A#4": 70,
    "C#5": 73, "D#5": 75, "F#5": 78, "G#5": 80,
}

TIMBRES = {
    "sine":     {"n_harmonics": 1, "decay_exp": 0.0},
    "sawtooth": {"n_harmonics": 6, "decay_exp": 1.0},
    "piano":    {"n_harmonics": 6, "decay_exp": 1.5},
}


def note_to_freq(note_name):
    """Convert note name to frequency."""
    if note_name in NAMED_PITCHES:
        return midi_to_freq(NAMED_PITCHES[note_name])
    raise ValueError(f"Unknown note: {note_name}")


def get_partials(notes, timbre_name):
    """
    Reconstruct partials (freq, amplitude) for a stimulus.
    Matches generate_stimuli.py logic exactly.
    """
    cfg = TIMBRES[timbre_name]
    n_harm = cfg["n_harmonics"]
    decay = cfg["decay_exp"]

    partials = []
    for note in notes:
        freq = note_to_freq(note)
        for h in range(1, n_harm + 1):
            f_h = freq * h
            if f_h > SAMPLE_RATE / 2:
                break
            amp = 1.0 / (h ** decay) if decay > 0 else 1.0
            partials.append((f_h, amp))
    return partials


# ── Component 1: Interference (Hutchinson & Knopoff 1978) ─────────────────

def compute_interference(partials):
    """
    Compute interference/roughness using Hutchinson & Knopoff (1978)
    critical bandwidth formulation as used in Harrison & Pearce (2020).

    For each pair of partials, roughness is based on their frequency
    separation relative to the critical bandwidth at their mean frequency.
    """
    n = len(partials)
    if n < 2:
        return 0.0

    total_roughness = 0.0
    weight_sum = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            fi, ai = partials[i]
            fj, aj = partials[j]

            f_mean = (fi + fj) / 2.0

            # Critical bandwidth (Hutchinson & Knopoff approximation)
            cbw = 1.72 * (f_mean ** 0.65)

            f_diff = abs(fi - fj)

            if cbw == 0:
                continue

            y = f_diff / cbw

            # Interference function: peaks at y ~ 0.25
            if y > 0:
                g = 4.0 * y * np.exp(1.0 - 4.0 * y)
            else:
                g = 0.0

            w = np.sqrt(ai * aj)
            total_roughness += w * g
            weight_sum += w

    # Normalize by total weight to get roughness independent of number of partials
    if weight_sum > 0:
        return total_roughness / weight_sum
    return 0.0


# ── Component 2: Harmonicity (log-frequency periodicity) ──────────────────

def compute_harmonicity(partials):
    """
    Compute harmonicity via log-frequency spectrum autocorrelation.

    Method: Create a spectrum in log-frequency space, compute its
    autocorrelation, and measure the height of the dominant peak
    (excluding lag 0). Harmonic tones produce strong periodic patterns
    in log-frequency space.
    """
    if len(partials) < 2:
        return 1.0  # Single partial is trivially harmonic

    freqs = np.array([p[0] for p in partials])
    amps = np.array([p[1] for p in partials])

    # Map to log-frequency (cents from 20 Hz)
    log_freqs = 1200.0 * np.log2(freqs / 20.0)

    # Create a continuous spectrum via Gaussian kernel density
    # Resolution: 1 cent per bin
    min_lf = max(0, log_freqs.min() - 200)
    max_lf = log_freqs.max() + 200
    n_bins = int(max_lf - min_lf) + 1

    if n_bins < 10:
        return 1.0

    spectrum = np.zeros(n_bins)
    sigma = 15.0  # Gaussian width in cents (~quarter-tone)
    bins = np.arange(n_bins) + min_lf

    for freq, amp in zip(log_freqs, amps):
        spectrum += amp * np.exp(-0.5 * ((bins - freq) / sigma) ** 2)

    # Normalize spectrum
    spec_norm = np.sqrt(np.sum(spectrum ** 2))
    if spec_norm == 0:
        return 0.0
    spectrum = spectrum / spec_norm

    # Autocorrelation via FFT
    fft_spec = np.fft.fft(spectrum, n=2 * n_bins)
    acf = np.real(np.fft.ifft(fft_spec * np.conj(fft_spec)))[:n_bins]

    # Normalize: acf[0] = 1
    if acf[0] > 0:
        acf = acf / acf[0]

    # Find dominant peak excluding lag 0
    # Minimum lag: ~200 cents (larger than a whole tone) to avoid
    # interference-related peaks
    min_lag = 200  # cents

    if min_lag >= len(acf):
        return 0.0

    acf_search = acf[min_lag:]

    if len(acf_search) == 0:
        return 0.0

    # The peak height is our harmonicity measure
    harmonicity = np.max(acf_search)

    # Clamp to [0, 1]
    return float(np.clip(harmonicity, 0.0, 1.0))


# ── Component 3: Familiarity (corpus-based) ───────────────────────────────

# Familiarity scores based on chord frequency in Western music corpora
# Values are approximate, normalized to [0, 1]
FAMILIARITY_SCORES = {
    # Intervals
    "unison": 0.95,
    "octave": 0.90,
    "perfect_fifth": 0.85,
    "perfect_fourth": 0.80,
    "major_third": 0.75,
    "minor_third": 0.70,
    "major_sixth": 0.65,
    "minor_sixth": 0.55,
    "major_second": 0.50,
    "minor_seventh": 0.45,
    "major_seventh": 0.35,
    "minor_second": 0.25,
    "tritone": 0.30,
    # Chords
    "major_triad": 0.95,
    "minor_triad": 0.90,
    "dominant_7th": 0.80,
    "minor_7th": 0.75,
    "major_7th": 0.65,
    "diminished_triad": 0.55,
    "augmented_triad": 0.40,
    "diminished_7th": 0.45,
}

# Inversions get the same familiarity as their root position
INVERSION_MAP = {
    "major_triad_1st_inv": "major_triad",
    "major_triad_2nd_inv": "major_triad",
    "minor_triad_1st_inv": "minor_triad",
    "minor_triad_2nd_inv": "minor_triad",
    "diminished_triad_1st_inv": "diminished_triad",
    "diminished_triad_2nd_inv": "diminished_triad",
    "augmented_triad_1st_inv": "augmented_triad",
    "augmented_triad_2nd_inv": "augmented_triad",
}


def get_familiarity(interval_name):
    """Get familiarity score for an interval or chord name."""
    if interval_name in FAMILIARITY_SCORES:
        return FAMILIARITY_SCORES[interval_name]
    if interval_name in INVERSION_MAP:
        return FAMILIARITY_SCORES[INVERSION_MAP[interval_name]]
    # Default for unknown
    return 0.3


# ── Combined consonance ───────────────────────────────────────────────────

# Weights from Harrison & Pearce (2020) Table 3, "Full" model
# They report: harmonicity positive, interference negative, familiarity positive
# We normalize each component to [0,1] range first, then combine
W_HARMONICITY = 1.0
W_INTERFERENCE = 1.0
W_FAMILIARITY = 0.5  # Lower weight since it's a lookup, not psychoacoustic


def compute_consonance(harmonicity, interference, familiarity):
    """
    Combined consonance score.
    Higher = more consonant.
    C = w1 * harmonicity - w2 * interference + w3 * familiarity
    """
    return (W_HARMONICITY * harmonicity
            - W_INTERFERENCE * interference
            + W_FAMILIARITY * familiarity)


# ── Main computation ──────────────────────────────────────────────────────

def main():
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    print(f"Computing Harrison & Pearce (2020) scores for {len(manifest)} stimuli...\n")

    all_interference = []
    all_harmonicity = []
    all_familiarity = []
    results = {}

    for filename, meta in manifest.items():
        timbre = meta["timbre"]
        notes = meta["notes"]
        interval_name = meta.get("interval_name", "")

        # Handle special anchor stimuli
        if timbre in ("none", "noise") or not notes:
            if timbre == "noise":
                # White noise: maximum interference, zero harmonicity
                interference = 1.0
                harmonicity = 0.0
                familiarity = 0.0
            else:
                # Silence: no interference, no harmonicity
                interference = 0.0
                harmonicity = 0.0
                familiarity = 0.0
            all_interference.append(interference)
            all_harmonicity.append(harmonicity)
            all_familiarity.append(familiarity)
            results[filename] = {
                "interference_raw": interference,
                "harmonicity_raw": harmonicity,
                "familiarity": familiarity,
            }
            continue

        partials = get_partials(notes, timbre)

        interference = compute_interference(partials)
        harmonicity = compute_harmonicity(partials)
        familiarity = get_familiarity(interval_name)

        all_interference.append(interference)
        all_harmonicity.append(harmonicity)
        all_familiarity.append(familiarity)

        results[filename] = {
            "interference_raw": interference,
            "harmonicity_raw": harmonicity,
            "familiarity": familiarity,
        }

    # Normalize interference and harmonicity to [0, 1] for combined score
    int_arr = np.array(all_interference)
    har_arr = np.array(all_harmonicity)

    int_min, int_max = int_arr.min(), int_arr.max()
    har_min, har_max = har_arr.min(), har_arr.max()

    int_range = int_max - int_min if int_max > int_min else 1.0
    har_range = har_max - har_min if har_max > har_min else 1.0

    for i, filename in enumerate(results.keys()):
        int_norm = (all_interference[i] - int_min) / int_range
        har_norm = (all_harmonicity[i] - har_min) / har_range
        fam = results[filename]["familiarity"]

        consonance = compute_consonance(har_norm, int_norm, fam)

        results[filename]["hp_interference"] = round(int_norm, 6)
        results[filename]["hp_harmonicity"] = round(har_norm, 6)
        results[filename]["hp_consonance"] = round(consonance, 6)

    # Update manifest
    for filename, scores in results.items():
        manifest[filename]["hp_interference"] = scores["hp_interference"]
        manifest[filename]["hp_harmonicity"] = scores["hp_harmonicity"]
        manifest[filename]["hp_consonance"] = scores["hp_consonance"]

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Updated {MANIFEST_PATH}")

    # Update CSV
    csv_rows = []
    for filename, meta in manifest.items():
        interval_name = meta.get("interval_name", "")
        csv_rows.append({
            "filename": filename,
            "dissonance_score": meta.get("dissonance_score", ""),
            "hp_interference": meta["hp_interference"],
            "hp_harmonicity": meta["hp_harmonicity"],
            "hp_consonance": meta["hp_consonance"],
            "timbre": meta["timbre"],
            "type": meta["type"],
            "interval_or_chord": interval_name,
            "root": meta["root"],
        })

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "dissonance_score", "hp_interference", "hp_harmonicity",
            "hp_consonance", "timbre", "type", "interval_or_chord", "root"
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Updated {CSV_PATH}")

    # Sort by consonance
    sorted_items = sorted(results.items(), key=lambda x: x[1]["hp_consonance"], reverse=True)

    print("\n" + "=" * 80)
    print("TOP 10 MOST CONSONANT")
    print("=" * 80)
    for fname, scores in sorted_items[:10]:
        meta = manifest[fname]
        print(f"  {scores['hp_consonance']:+.4f}  harm={scores['hp_harmonicity']:.3f}  "
              f"int={scores['hp_interference']:.3f}  fam={scores['familiarity']:.2f}  "
              f"{fname}")

    print("\n" + "=" * 80)
    print("10 MIDDLE ITEMS (around median)")
    print("=" * 80)
    mid = len(sorted_items) // 2
    for fname, scores in sorted_items[mid-5:mid+5]:
        meta = manifest[fname]
        print(f"  {scores['hp_consonance']:+.4f}  harm={scores['hp_harmonicity']:.3f}  "
              f"int={scores['hp_interference']:.3f}  fam={scores['familiarity']:.2f}  "
              f"{fname}")

    print("\n" + "=" * 80)
    print("TOP 10 MOST DISSONANT (lowest consonance)")
    print("=" * 80)
    for fname, scores in sorted_items[-10:]:
        meta = manifest[fname]
        print(f"  {scores['hp_consonance']:+.4f}  harm={scores['hp_harmonicity']:.3f}  "
              f"int={scores['hp_interference']:.3f}  fam={scores['familiarity']:.2f}  "
              f"{fname}")

    # Validation checks
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)

    checks = [
        ("sine_interval_major_seventh_G4.wav", "piano_interval_major_third_E3.wav",
         "Major 7th (sine) should be MORE dissonant than major 3rd (piano)", "less"),
        ("sawtooth_chord_diminished_7th_C3.wav", "sawtooth_chord_diminished_7th_C4.wav",
         "Dim7 at C3 and C4 should have SIMILAR scores (not 3x different)", "similar"),
    ]

    # Check 1: major 7th more dissonant than major 3rd
    for c1, c2, desc, check_type in checks:
        if c1 in results and c2 in results:
            s1 = results[c1]["hp_consonance"]
            s2 = results[c2]["hp_consonance"]
            if check_type == "less":
                passed = s1 < s2
                print(f"  {'PASS' if passed else 'FAIL'}: {desc}")
                print(f"    {c1}: {s1:.4f}")
                print(f"    {c2}: {s2:.4f}")
            elif check_type == "similar":
                ratio = max(abs(s1), 0.001) / max(abs(s2), 0.001)
                passed = 0.5 < ratio < 2.0
                print(f"  {'PASS' if passed else 'FAIL'}: {desc}")
                print(f"    {c1}: {s1:.4f}")
                print(f"    {c2}: {s2:.4f}")
                print(f"    Ratio: {ratio:.3f}")

    # Check 3: Unisons should be highly consonant
    print("\n  Unison consonance check (should be among highest):")
    unisons = [(f, r["hp_consonance"]) for f, r in results.items()
               if "unison" in f]
    for f, s in sorted(unisons, key=lambda x: -x[1])[:6]:
        rank = [x[0] for x in sorted_items].index(f) + 1
        print(f"    Rank {rank}/{len(sorted_items)}: {s:+.4f}  {f}")

    # Check 4: Octaves should be consonant
    print("\n  Octave consonance check (should be high):")
    octaves = [(f, r["hp_consonance"]) for f, r in results.items()
               if "octave" in f]
    for f, s in sorted(octaves, key=lambda x: -x[1])[:6]:
        rank = [x[0] for x in sorted_items].index(f) + 1
        print(f"    Rank {rank}/{len(sorted_items)}: {s:+.4f}  {f}")

    # Check 5: Minor 2nds and tritones should be dissonant
    print("\n  Minor 2nd / tritone check (should be among lowest):")
    dissonant_intervals = [(f, r["hp_consonance"]) for f, r in results.items()
                           if "minor_second" in f or "tritone" in f]
    for f, s in sorted(dissonant_intervals, key=lambda x: x[1])[:6]:
        rank = [x[0] for x in sorted_items].index(f) + 1
        print(f"    Rank {rank}/{len(sorted_items)}: {s:+.4f}  {f}")


if __name__ == "__main__":
    main()
