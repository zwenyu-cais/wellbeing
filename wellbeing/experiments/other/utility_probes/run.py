#!/usr/bin/env python3
"""Train utility probes on D3 experienced utility data.

End-to-end pipeline:
1. Load EU results (preference graph with edges)
2. Extract hidden-state activations from open-weight model
3. Train heteroscedastic linear probes per layer
4. Save best probe and results summary

Usage:
    python run.py \
        --model_key qwen25-7b-instruct \
        --dataset d3_diverse_500 \
        --save_dir results/probes_d3/qwen25-7b-instruct
"""

import argparse
import glob
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

# Add wellbeing root to path for imports
_WELLBEING_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _WELLBEING_ROOT)

from experiments.other.utility_probes.probe import HeteroscedasticProbe
from utils.model_utils import get_model_config


# ---------------------------------------------------------------------------
# Step 1: Load EU results
# ---------------------------------------------------------------------------

def load_eu_results(eu_results_dir: str, model_key: str):
    """Load EU results including graph data (edges, training/holdout splits).

    The graph data is in the full results file (not the _utilities_ variant).
    """
    # Find the full results file (the one WITHOUT '_utilities_' in the name)
    pattern = os.path.join(eu_results_dir, model_key, f"results_{model_key}_*.json")
    candidates = glob.glob(pattern)
    # Filter out _utilities_ files
    candidates = [c for c in candidates if "_utilities_" not in os.path.basename(c)]

    if not candidates:
        # Try the _utilities_ file as fallback (some versions may store graph_data there)
        candidates = glob.glob(pattern)

    if not candidates:
        raise FileNotFoundError(
            f"No EU results file found matching {pattern}. "
            f"Available files: {os.listdir(os.path.join(eu_results_dir, model_key))}"
        )

    results_file = candidates[0]
    print(f"Loading EU results from: {results_file}")

    with open(results_file, "r") as f:
        data = json.load(f)

    # Extract graph data
    if "graph_data" not in data:
        raise KeyError(
            f"No 'graph_data' key in {results_file}. "
            f"Available keys: {list(data.keys())}. "
            f"Try using the full results file (not _utilities_)."
        )

    graph_data = data["graph_data"]
    options = graph_data["options"]  # list of dicts with 'id', 'type', 'messages'/'description'
    edges = graph_data["edges"]  # dict: str(tuple) -> edge info
    training_edges = graph_data["training_edges"]  # list of [id_A, id_B]
    holdout_edge_pairs = graph_data["holdout_edge_indices"]  # list of [id_A, id_B]

    utilities = data.get("utilities", {})
    holdout_metrics = data.get("holdout_metrics", {})

    print(f"  {len(options)} options, {len(edges)} edges")
    print(f"  {len(training_edges)} training edges, {len(holdout_edge_pairs)} holdout edges")
    print(f"  Thurstonian holdout accuracy: {holdout_metrics.get('accuracy', 'N/A')}")

    return options, edges, training_edges, holdout_edge_pairs, utilities, holdout_metrics


# ---------------------------------------------------------------------------
# Step 2: Extract activations
# ---------------------------------------------------------------------------

def extract_activations(
    model_key: str,
    options: list,
    save_path: str,
    batch_size: int = 64,
):
    """Extract last-token hidden states from all layers for each experience option.

    Returns:
        activations: dict mapping option_id -> dict mapping layer_idx -> tensor[hidden_dim]
        num_layers: total number of layers (including embedding layer 0)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Check for cached activations
    if os.path.exists(save_path):
        print(f"Loading cached activations from {save_path}")
        cached = torch.load(save_path, map_location="cpu", weights_only=True)
        # Determine num_layers from first option
        first_key = next(iter(cached))
        num_layers = len(cached[first_key])
        print(f"  {len(cached)} options, {num_layers} layers")
        return cached, num_layers

    # Get model path from models.yaml
    model_config = get_model_config(model_key)
    model_path = model_config.get("path", model_config.get("model_name"))
    print(f"Loading model from: {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load model — we only need hidden states, not logits, so we'll call
    # model.model() directly to skip the lm_head (saves massive memory)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    # Access the base transformer (skips lm_head)
    base_model = model.model

    # Format experiences as chat messages and extract activations
    activations = {}
    num_layers = None

    for batch_start in range(0, len(options), batch_size):
        batch_end = min(batch_start + batch_size, len(options))
        batch_options = options[batch_start:batch_end]

        # Format each option using its messages (multi-turn conversations)
        texts = []
        for opt in batch_options:
            messages = opt.get("messages", [])
            if not messages:
                # Fallback: wrap description as a single user message
                messages = [{"role": "user", "content": opt.get("description", "")}]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)

        # Tokenize with padding
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )

        # Move to model device
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

        # Forward pass on base transformer (no lm_head — saves memory)
        with torch.no_grad():
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states  # tuple of (num_layers+1) tensors

        if num_layers is None:
            num_layers = len(hidden_states)
            print(f"  Model has {num_layers} hidden state layers (including embedding)")

        # Extract last non-padding token's hidden state for each item in batch
        # With left-padding, the last token is always the rightmost position
        for i, opt in enumerate(batch_options):
            opt_id = opt["id"]
            # Find position of last non-padding token
            seq_len = attention_mask[i].sum().item()
            last_pos = input_ids.shape[1] - 1  # with left-padding, last token is at end

            layer_dict = {}
            for layer_idx in range(num_layers):
                layer_dict[layer_idx] = hidden_states[layer_idx][i, last_pos, :].cpu().to(torch.float16)

            activations[opt_id] = layer_dict

        if (batch_start // batch_size) % 5 == 0 or batch_end == len(options):
            print(f"  Extracted activations: {batch_end}/{len(options)} options")

    # Save activations
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(activations, save_path)
    print(f"  Saved activations to {save_path}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return activations, num_layers


# ---------------------------------------------------------------------------
# Step 3: Edge data helpers
# ---------------------------------------------------------------------------

def lookup_edge(edges: dict, a_id: str, b_id: str):
    """Look up an edge in the edges dict, trying both orderings.

    Returns (probability_A_wins, found) where found indicates the edge was found.
    """
    key_ab = str((a_id, b_id))
    if key_ab in edges:
        return edges[key_ab]["probability_A"], True

    key_ba = str((b_id, a_id))
    if key_ba in edges:
        # In the (B, A) edge, probability_A means P(B > A), so flip
        return 1.0 - edges[key_ba]["probability_A"], True

    return None, False


def build_edge_tensors(edge_pairs, edges, activations, layer_idx):
    """Convert edge pairs into activation tensors for a specific layer.

    Returns: (act_A, act_B, targets, n_skipped)
    """
    act_a_list = []
    act_b_list = []
    target_list = []
    n_skipped = 0

    for pair in edge_pairs:
        a_id, b_id = pair[0], pair[1]

        if a_id not in activations or b_id not in activations:
            n_skipped += 1
            continue

        prob_a, found = lookup_edge(edges, a_id, b_id)
        if not found:
            n_skipped += 1
            continue

        act_a_list.append(activations[a_id][layer_idx])
        act_b_list.append(activations[b_id][layer_idx])
        target_list.append(prob_a)

    if not act_a_list:
        return None, None, None, n_skipped

    act_A = torch.stack(act_a_list).float()
    act_B = torch.stack(act_b_list).float()
    targets = torch.tensor(target_list, dtype=torch.float32)

    return act_A, act_B, targets, n_skipped


# ---------------------------------------------------------------------------
# Step 3: Train probes
# ---------------------------------------------------------------------------

def train_probe_on_layer(
    hidden_dim: int,
    train_act_A: torch.Tensor,
    train_act_B: torch.Tensor,
    train_targets: torch.Tensor,
    holdout_act_A: torch.Tensor,
    holdout_act_B: torch.Tensor,
    holdout_targets: torch.Tensor,
    epochs: int,
    lr: float,
    device: torch.device,
):
    """Train a heteroscedastic probe on one layer's activations.

    Returns: (best_state_dict, best_holdout_acc, final_train_loss, history)
    """
    torch.manual_seed(42)

    probe = HeteroscedasticProbe(hidden_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    train_act_A = train_act_A.to(device)
    train_act_B = train_act_B.to(device)
    train_targets = train_targets.to(device)

    has_holdout = holdout_act_A is not None
    if has_holdout:
        holdout_act_A = holdout_act_A.to(device)
        holdout_act_B = holdout_act_B.to(device)
        holdout_targets = holdout_targets.to(device)

    best_holdout_acc = -1.0
    best_state_dict = None
    history = []

    for epoch in range(epochs):
        # Training step
        probe.train()
        optimizer.zero_grad()

        pred_prob = probe.predict_preference(train_act_A, train_act_B)
        loss = F.binary_cross_entropy(pred_prob, train_targets)
        loss.backward()
        optimizer.step()

        # Holdout evaluation
        holdout_acc = 0.0
        if has_holdout and (epoch % 50 == 0 or epoch == epochs - 1):
            probe.eval()
            with torch.no_grad():
                h_pred = probe.predict_preference(holdout_act_A, holdout_act_B)
            holdout_acc = ((h_pred >= 0.5) == (holdout_targets >= 0.5)).float().mean().item()

            if holdout_acc > best_holdout_acc:
                best_holdout_acc = holdout_acc
                best_state_dict = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

        if epoch % 50 == 0 or epoch == epochs - 1:
            train_acc = ((pred_prob.detach() >= 0.5) == (train_targets >= 0.5)).float().mean().item()
            history.append({
                "epoch": epoch,
                "train_loss": loss.item(),
                "train_acc": train_acc,
                "holdout_acc": holdout_acc,
            })

    if not has_holdout:
        best_state_dict = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

    final_train_loss = history[-1]["train_loss"] if history else 0.0

    return best_state_dict, best_holdout_acc, final_train_loss, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train utility probes on D3 hidden states.")
    parser.add_argument("--model_key", type=str, required=True, help="Model key from models.yaml")
    parser.add_argument("--dataset", type=str, default="d3_diverse_500", help="Dataset name")
    parser.add_argument("--save_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument(
        "--eu_results_dir", type=str,
        default=os.path.join(_WELLBEING_ROOT, "experiments", "wellbeing_evaluations",
                             "compute_experienced_utility", "results", "eu_d3"),
        help="Path to EU results directory containing model subdirectories",
    )
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs per layer")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for activation extraction")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices to probe (default: all)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve save_dir relative to wellbeing root if not absolute
    if not os.path.isabs(args.save_dir):
        save_dir = os.path.join(_WELLBEING_ROOT, args.save_dir)
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Resolve eu_results_dir
    if not os.path.isabs(args.eu_results_dir):
        eu_results_dir = os.path.join(_WELLBEING_ROOT, args.eu_results_dir)
    else:
        eu_results_dir = args.eu_results_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Save dir: {save_dir}")

    # -----------------------------------------------------------------------
    # Step 1: Load EU results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 1: Loading EU results")
    print("=" * 60)

    options, edges, training_edges, holdout_edge_pairs, utilities, holdout_metrics = \
        load_eu_results(eu_results_dir, args.model_key)

    thurstonian_holdout_acc = holdout_metrics.get("accuracy", None)

    # -----------------------------------------------------------------------
    # Step 2: Extract activations
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Extracting activations")
    print("=" * 60)

    activations_path = os.path.join(save_dir, "activations.pt")
    activations, num_layers = extract_activations(
        model_key=args.model_key,
        options=options,
        save_path=activations_path,
        batch_size=args.batch_size,
    )

    # -----------------------------------------------------------------------
    # Step 3: Train probes per layer
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Training probes per layer")
    print("=" * 60)

    # Determine which layers to probe
    if args.layers is not None:
        layer_indices = [int(x.strip()) for x in args.layers.split(",")]
    else:
        layer_indices = list(range(num_layers))

    print(f"  Probing {len(layer_indices)} layers: {layer_indices}")

    per_layer_results = {}
    best_layer = None
    best_acc = -1.0
    best_state_dict = None
    best_hidden_dim = None

    for layer_idx in layer_indices:
        print(f"\n--- Layer {layer_idx} ---")
        t0 = time.time()

        # Build edge tensors for this layer
        train_act_A, train_act_B, train_targets, train_skipped = \
            build_edge_tensors(training_edges, edges, activations, layer_idx)

        if train_act_A is None:
            print(f"  No valid training edges, skipping")
            continue

        hidden_dim = train_act_A.shape[1]

        holdout_act_A, holdout_act_B, holdout_targets, holdout_skipped = \
            build_edge_tensors(holdout_edge_pairs, edges, activations, layer_idx)

        n_train = train_act_A.shape[0]
        n_holdout = holdout_act_A.shape[0] if holdout_act_A is not None else 0
        print(f"  {n_train} train edges ({train_skipped} skipped), "
              f"{n_holdout} holdout edges ({holdout_skipped} skipped), "
              f"hidden_dim={hidden_dim}")

        # Train
        state_dict, holdout_acc, final_train_loss, history = train_probe_on_layer(
            hidden_dim=hidden_dim,
            train_act_A=train_act_A,
            train_act_B=train_act_B,
            train_targets=train_targets,
            holdout_act_A=holdout_act_A,
            holdout_act_B=holdout_act_B,
            holdout_targets=holdout_targets,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
        )

        elapsed = time.time() - t0
        print(f"  Holdout acc: {holdout_acc:.4f}  |  Train loss: {final_train_loss:.4f}  |  {elapsed:.1f}s")

        per_layer_results[str(layer_idx)] = {
            "holdout_accuracy": holdout_acc,
            "final_train_loss": final_train_loss,
        }

        if holdout_acc > best_acc:
            best_acc = holdout_acc
            best_layer = layer_idx
            best_state_dict = state_dict
            best_hidden_dim = hidden_dim

    if best_layer is None:
        print("\nError: No layer produced a valid probe.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 4: Save results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Saving results")
    print("=" * 60)

    # Save best probe
    probe_save = {
        "state_dict": best_state_dict,
        "layer": best_layer,
        "hidden_dim": best_hidden_dim,
    }
    probe_path = os.path.join(save_dir, "best_probe.pt")
    torch.save(probe_save, probe_path)
    print(f"  Saved best probe (layer {best_layer}) -> {probe_path}")

    # Save results JSON
    results = {
        "model_key": args.model_key,
        "dataset": args.dataset,
        "best_layer": best_layer,
        "num_layers_tested": len(layer_indices),
        "probe_holdout_accuracy": best_acc,
        "thurstonian_holdout_accuracy": thurstonian_holdout_acc,
        "per_layer_results": per_layer_results,
        "num_training_edges": len(training_edges),
        "num_holdout_edges": len(holdout_edge_pairs),
        "num_options": len(options),
        "epochs": args.epochs,
        "lr": args.lr,
    }
    results_path = os.path.join(save_dir, "probe_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results -> {results_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"{'Layer':>8}  {'Holdout Acc':>12}  {'Train Loss':>11}")
    print(f"{'-'*8}  {'-'*12}  {'-'*11}")
    for layer_idx in layer_indices:
        key = str(layer_idx)
        if key in per_layer_results:
            r = per_layer_results[key]
            marker = " <-- best" if layer_idx == best_layer else ""
            print(f"{layer_idx:>8}  {r['holdout_accuracy']:>12.4f}  "
                  f"{r['final_train_loss']:>11.4f}{marker}")

    print(f"\nBest layer: {best_layer} (holdout accuracy: {best_acc:.4f})")
    if thurstonian_holdout_acc is not None:
        gap = best_acc - thurstonian_holdout_acc
        print(f"Thurstonian holdout accuracy: {thurstonian_holdout_acc:.4f}")
        print(f"Probe - Thurstonian: {gap:+.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
