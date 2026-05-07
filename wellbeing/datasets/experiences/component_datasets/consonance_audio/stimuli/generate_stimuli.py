#!/usr/bin/env python3
"""
Generate audio stimuli for consonance/dissonance experiment.

Produces intervals, chords, inversions across multiple timbres,
computes Plomp-Levelt/Sethares dissonance scores, and saves
a manifest JSON and summary CSV.
"""

import json
import csv
import os
import numpy as np
from scipy.io import wavfile

# ── Parameters ──────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
DURATION = 3.0
FADE_MS = 50
TARGET_RMS = 0.15  # target RMS for normalization

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Pitch definitions ───────────────────────────────────────────────────────

# Semitones above C0 (MIDI-style). C4 = 60 in MIDI = 261.63 Hz
def midi_to_freq(midi_note):
    return 440.0 * 2 ** ((midi_note - 69) / 12.0)

NAMED_PITCHES = {
    "C3": 48, "E3": 52, "G3": 55,
    "C4": 60, "E4": 64, "G4": 67,
}

ROOT_PITCHES = ["C3", "E3", "G3", "C4", "E4", "G4"]
INVERSION_ROOTS = ["C3", "C4", "G3"]

# Chromatic interval names (0-12 semitones)
INTERVAL_NAMES = [
    "unison", "minor_second", "major_second", "minor_third",
    "major_third", "perfect_fourth", "tritone", "perfect_fifth",
    "minor_sixth", "major_sixth", "minor_seventh", "major_seventh",
    "octave",
]

# Chord definitions: name -> list of semitone offsets from root
CHORD_DEFS = {
    "major_triad":    [0, 4, 7],
    "minor_triad":    [0, 3, 7],
    "diminished_triad": [0, 3, 6],
    "augmented_triad":  [0, 4, 8],
    "major_7th":      [0, 4, 7, 11],
    "minor_7th":      [0, 3, 7, 10],
    "dominant_7th":   [0, 4, 7, 10],
    "diminished_7th": [0, 3, 6, 9],
}

TRIAD_NAMES = ["major_triad", "minor_triad", "diminished_triad", "augmented_triad"]

TIMBRES = {
    "sine":     {"n_harmonics": 1, "decay_exp": 0.0},
    "sawtooth": {"n_harmonics": 6, "decay_exp": 1.0},    # 1/n
    "piano":    {"n_harmonics": 6, "decay_exp": 1.5},     # 1/n^1.5
}

# ── Plomp-Levelt / Sethares dissonance model ───────────────────────────────

PL_B1 = 3.5
PL_B2 = 5.75
PL_DSTAR = 0.24
PL_S1 = 0.021
PL_S2 = 19.0


def pl_dissonance_pair(f1, f2, a1, a2):
    """Plomp-Levelt dissonance between two partials."""
    if f1 > f2:
        f1, f2 = f2, f1
        a1, a2 = a2, a1
    s = PL_DSTAR / (PL_S1 * f1 + PL_S2)
    diff = f2 - f1
    return a1 * a2 * (np.exp(-PL_B1 * s * diff) - np.exp(-PL_B2 * s * diff))


def compute_dissonance(partials):
    """
    Compute total Sethares dissonance for a set of partials.
    partials: list of (frequency, amplitude)
    """
    total = 0.0
    n = len(partials)
    for i in range(n):
        for j in range(i + 1, n):
            d = pl_dissonance_pair(
                partials[i][0], partials[j][0],
                partials[i][1], partials[j][1],
            )
            total += d
    return total


# ── Audio generation helpers ────────────────────────────────────────────────

def generate_tone(freq, timbre_name, duration=DURATION, sr=SAMPLE_RATE):
    """
    Generate a single tone with harmonics defined by the timbre.
    Returns (signal_array, list_of_partials) where partials are (freq, amplitude).
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    cfg = TIMBRES[timbre_name]
    n_harm = cfg["n_harmonics"]
    decay = cfg["decay_exp"]

    signal = np.zeros_like(t)
    partials = []
    for h in range(1, n_harm + 1):
        f_h = freq * h
        if f_h > sr / 2:
            break  # avoid aliasing
        amp = 1.0 / (h ** decay) if decay > 0 else 1.0
        signal += amp * np.sin(2 * np.pi * f_h * t)
        partials.append((f_h, amp))

    return signal, partials


def apply_fade(signal, fade_samples):
    """Apply linear fade-in and fade-out."""
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    signal[:fade_samples] *= fade_in
    signal[-fade_samples:] *= fade_out
    return signal


def normalize_rms(signal, target_rms=TARGET_RMS):
    """Normalize signal to target RMS."""
    rms = np.sqrt(np.mean(signal ** 2))
    if rms > 1e-10:
        signal = signal * (target_rms / rms)
    # Clip to prevent overflow
    signal = np.clip(signal, -0.99, 0.99)
    return signal


def save_wav(filename, signal, sr=SAMPLE_RATE):
    """Save signal as 16-bit WAV."""
    signal_16 = np.int16(signal * 32767)
    wavfile.write(os.path.join(OUTPUT_DIR, filename), sr, signal_16)


def combine_tones(tone_list):
    """Sum multiple tone signals."""
    return sum(tone_list)


# ── Stimulus generation ─────────────────────────────────────────────────────

def generate_multi_tone_stimulus(midi_notes, timbre_name):
    """
    Generate a stimulus from multiple simultaneous MIDI notes.
    Returns (signal, all_partials).
    """
    fade_samples = int(FADE_MS / 1000.0 * SAMPLE_RATE)
    signals = []
    all_partials = []
    for mn in midi_notes:
        freq = midi_to_freq(mn)
        sig, parts = generate_tone(freq, timbre_name)
        signals.append(sig)
        all_partials.extend(parts)
    combined = combine_tones(signals)
    combined = apply_fade(combined, fade_samples)
    combined = normalize_rms(combined)
    return combined, all_partials


def note_name_from_midi(midi):
    """Convert MIDI number to note name like C4, Db3, etc."""
    names = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    octave = (midi // 12) - 1
    note = names[midi % 12]
    return f"{note}{octave}"


def invert_chord(semitones, inversion):
    """
    Apply inversion to chord.
    inversion=1: move bottom note up an octave
    inversion=2: move bottom two notes up an octave
    """
    notes = sorted(semitones)
    for _ in range(inversion):
        notes.append(notes.pop(0) + 12)
    return notes


def main():
    manifest = {}
    rows = []
    fade_samples = int(FADE_MS / 1000.0 * SAMPLE_RATE)

    count = 0

    # ── 1. Intervals: 13 intervals x 6 roots x 3 timbres ──────────────────

    for timbre in TIMBRES:
        for root_name in ROOT_PITCHES:
            root_midi = NAMED_PITCHES[root_name]
            for semitones, interval_name in enumerate(INTERVAL_NAMES):
                midi_notes = [root_midi, root_midi + semitones]
                note_names = [note_name_from_midi(m) for m in midi_notes]

                signal, partials = generate_multi_tone_stimulus(midi_notes, timbre)
                dissonance = compute_dissonance(partials)

                fname = f"{timbre}_interval_{interval_name}_{root_name}.wav"
                save_wav(fname, signal)

                manifest[fname] = {
                    "dissonance_score": round(dissonance, 6),
                    "timbre": timbre,
                    "type": "interval",
                    "interval_name": interval_name,
                    "root": root_name,
                    "notes": note_names,
                    "n_partials": len(partials),
                }
                rows.append({
                    "filename": fname,
                    "dissonance_score": round(dissonance, 6),
                    "timbre": timbre,
                    "type": "interval",
                    "interval_or_chord": interval_name,
                    "root": root_name,
                })
                count += 1

    # ── 2. Chords: 8 types x 6 roots x 3 timbres ──────────────────────────

    for timbre in TIMBRES:
        for root_name in ROOT_PITCHES:
            root_midi = NAMED_PITCHES[root_name]
            for chord_name, offsets in CHORD_DEFS.items():
                midi_notes = [root_midi + o for o in offsets]
                note_names = [note_name_from_midi(m) for m in midi_notes]

                signal, partials = generate_multi_tone_stimulus(midi_notes, timbre)
                dissonance = compute_dissonance(partials)

                fname = f"{timbre}_chord_{chord_name}_{root_name}.wav"
                save_wav(fname, signal)

                manifest[fname] = {
                    "dissonance_score": round(dissonance, 6),
                    "timbre": timbre,
                    "type": "chord",
                    "interval_name": chord_name,
                    "root": root_name,
                    "notes": note_names,
                    "n_partials": len(partials),
                }
                rows.append({
                    "filename": fname,
                    "dissonance_score": round(dissonance, 6),
                    "timbre": timbre,
                    "type": "chord",
                    "interval_or_chord": chord_name,
                    "root": root_name,
                })
                count += 1

    # ── 3. Inversions: 4 triads x 2 inversions x 3 roots x 3 timbres ──────

    for timbre in TIMBRES:
        for root_name in INVERSION_ROOTS:
            root_midi = NAMED_PITCHES[root_name]
            for chord_name in TRIAD_NAMES:
                offsets = CHORD_DEFS[chord_name]
                for inv in [1, 2]:
                    inverted = invert_chord(offsets, inv)
                    midi_notes = [root_midi + o for o in inverted]
                    note_names = [note_name_from_midi(m) for m in midi_notes]

                    signal, partials = generate_multi_tone_stimulus(midi_notes, timbre)
                    dissonance = compute_dissonance(partials)

                    fname = f"{timbre}_chord_{chord_name}_inv{inv}_{root_name}.wav"
                    save_wav(fname, signal)

                    manifest[fname] = {
                        "dissonance_score": round(dissonance, 6),
                        "timbre": timbre,
                        "type": "inversion",
                        "interval_name": f"{chord_name}_inv{inv}",
                        "root": root_name,
                        "notes": note_names,
                        "n_partials": len(partials),
                    }
                    rows.append({
                        "filename": fname,
                        "dissonance_score": round(dissonance, 6),
                        "timbre": timbre,
                        "type": "inversion",
                        "interval_or_chord": f"{chord_name}_inv{inv}",
                        "root": root_name,
                    })
                    count += 1

    # ── 4. Anchors: silence, white noise, pure A4 ──────────────────────────

    n_samples = int(SAMPLE_RATE * DURATION)

    # Silence
    silence = np.zeros(n_samples)
    fname = "anchor_silence.wav"
    save_wav(fname, silence)
    manifest[fname] = {
        "dissonance_score": 0.0,
        "timbre": "none",
        "type": "anchor",
        "interval_name": "silence",
        "root": "N/A",
        "notes": [],
        "n_partials": 0,
    }
    rows.append({
        "filename": fname, "dissonance_score": 0.0,
        "timbre": "none", "type": "anchor",
        "interval_or_chord": "silence", "root": "N/A",
    })
    count += 1

    # White noise
    np.random.seed(42)
    noise = np.random.randn(n_samples)
    noise = apply_fade(noise, fade_samples)
    noise = normalize_rms(noise)
    fname = "anchor_white_noise.wav"
    save_wav(fname, noise)
    manifest[fname] = {
        "dissonance_score": -1.0,  # not applicable
        "timbre": "noise",
        "type": "anchor",
        "interval_name": "white_noise",
        "root": "N/A",
        "notes": [],
        "n_partials": 0,
    }
    rows.append({
        "filename": fname, "dissonance_score": -1.0,
        "timbre": "noise", "type": "anchor",
        "interval_or_chord": "white_noise", "root": "N/A",
    })
    count += 1

    # Pure A4 tone
    t = np.linspace(0, DURATION, n_samples, endpoint=False)
    a4 = np.sin(2 * np.pi * 440.0 * t)
    a4 = apply_fade(a4, fade_samples)
    a4 = normalize_rms(a4)
    fname = "anchor_pure_A4.wav"
    save_wav(fname, a4)
    manifest[fname] = {
        "dissonance_score": 0.0,
        "timbre": "sine",
        "type": "anchor",
        "interval_name": "pure_tone",
        "root": "A4",
        "notes": ["A4"],
        "n_partials": 1,
    }
    rows.append({
        "filename": fname, "dissonance_score": 0.0,
        "timbre": "sine", "type": "anchor",
        "interval_or_chord": "pure_tone", "root": "A4",
    })
    count += 1

    # ── Save manifest and CSV ──────────────────────────────────────────────

    manifest_path = os.path.join(OUTPUT_DIR, "stimuli_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    csv_path = os.path.join(OUTPUT_DIR, "stimuli_dissonance.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "dissonance_score", "timbre", "type",
            "interval_or_chord", "root",
        ])
        writer.writeheader()
        writer.writerows(rows)

    # ── Print summary ──────────────────────────────────────────────────────

    # Filter out anchors with dissonance -1 for stats
    scored = [r for r in rows if r["dissonance_score"] >= 0]
    scored_sorted = sorted(scored, key=lambda x: x["dissonance_score"])

    print(f"\nTotal stimuli generated: {count}")
    print(f"Dissonance score range (excluding noise anchor): "
          f"{scored_sorted[0]['dissonance_score']:.6f} to "
          f"{scored_sorted[-1]['dissonance_score']:.6f}")

    print(f"\n--- Top 10 MOST dissonant ---")
    for r in scored_sorted[-10:][::-1]:
        print(f"  {r['dissonance_score']:.6f}  {r['filename']}")

    print(f"\n--- Top 10 LEAST dissonant ---")
    for r in scored_sorted[:10]:
        print(f"  {r['dissonance_score']:.6f}  {r['filename']}")

    mid = len(scored_sorted) // 2
    print(f"\n--- 10 MIDDLE dissonance ---")
    for r in scored_sorted[mid - 5 : mid + 5]:
        print(f"  {r['dissonance_score']:.6f}  {r['filename']}")

    print(f"\nManifest saved to: {manifest_path}")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
