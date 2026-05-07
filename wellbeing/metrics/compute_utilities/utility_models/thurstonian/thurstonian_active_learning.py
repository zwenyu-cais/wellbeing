import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set, TYPE_CHECKING
from ...compute_utilities import UtilityModel
if TYPE_CHECKING:
    from ...compute_utilities import PreferenceGraph
from ...utils import generate_responses, parse_responses_forced_choice, generate_responses_from_messages
from .utils import fit_thurstonian_model, evaluate_thurstonian_model
import random
from collections import defaultdict
import networkx as nx
import argparse
import yaml
import os
import json
import time
import asyncio


# ===================== THURSTONIAN ACTIVE LEARNING HELPER FUNCTIONS ===================== #

def generate_additional_pairs(
    utilities: Dict[Any, Dict[str, float]],
    existing_pairs_set: Set[Tuple[Any, Any]],
    available_edges: Set[Tuple[Any, Any]],
    num_edges_per_iteration: int,
    P: float,
    Q: float,
    seed: Optional[int] = None,
    scale_factor: float = 1.5,
    max_iterations: int = 5
) -> List[Tuple[Any, Any]]:
    """
    Generates additional pairs by sampling from the intersection of the bottom P% of
    utility differences and the bottom Q% of total degrees.
    
    Args:
        utilities: Dict mapping option IDs to {'mean': float, 'variance': float}
        existing_pairs_set: Set of existing (option_A_id, option_B_id) tuples
        available_edges: Set of available edges to sample from
        num_edges_per_iteration: Number of edges to sample
        P: Percentage defining bottom P% of utility differences
        Q: Percentage defining bottom Q% of total degrees
        seed: Random seed for reproducibility
        scale_factor: Factor to scale P and Q by if not enough pairs found
        max_iterations: Maximum number of scaling iterations
        
    Returns:
        List of selected (option_A_id, option_B_id) tuples
    """
    random.seed(seed)
    np.random.seed(seed)

    # Compute current degrees for each option
    option_id_to_degree = defaultdict(int)
    for A_id, B_id in existing_pairs_set:
        option_id_to_degree[A_id] += 1
        option_id_to_degree[B_id] += 1

    # Identify which pairs remain
    remaining_pairs = [pair for pair in available_edges if pair not in existing_pairs_set]
    if not remaining_pairs:
        print("No remaining pairs to sample.")
        return []

    def get_pairs_in_bottom_PQ_percent(p: float, q: float) -> List[Tuple[Any, Any]]:
        """Get pairs in intersection of bottom p% utility differences and q% degrees."""
        utility_differences = []
        total_degrees = []
        for pair in remaining_pairs:
            A_id, B_id = pair  # Keep original orientation
            diff = abs(utilities[A_id]['mean'] - utilities[B_id]['mean'])
            utility_differences.append(diff)
            deg_sum = option_id_to_degree[A_id] + option_id_to_degree[B_id]
            total_degrees.append(deg_sum)

        utility_differences = np.array(utility_differences)
        total_degrees = np.array(total_degrees)

        utility_cutoff = np.percentile(utility_differences, p)
        degree_cutoff = np.percentile(total_degrees, q)

        mask = (utility_differences <= utility_cutoff) & (total_degrees <= degree_cutoff)
        return [pair for pair, m in zip(remaining_pairs, mask) if m]  # Keep original orientation

    # Try progressively increasing P and Q until we get enough pairs
    current_pairs = []
    current_P = P
    current_Q = Q

    for i in range(max_iterations):
        candidate_subset = get_pairs_in_bottom_PQ_percent(current_P, current_Q)

        if len(candidate_subset) >= num_edges_per_iteration:
            current_pairs = random.sample(candidate_subset, num_edges_per_iteration)
            break
        else:
            current_pairs = candidate_subset[:]
            if i < max_iterations - 1:
                current_P = min(current_P * scale_factor, 100.0)
                current_Q = min(current_Q * scale_factor, 100.0)
            else:
                shortfall = num_edges_per_iteration - len(current_pairs)
                if shortfall > 0:
                    remaining_after_cut = list(set(remaining_pairs) - set(current_pairs))
                    if len(remaining_after_cut) > shortfall:
                        fallback_sample = random.sample(remaining_after_cut, shortfall)
                        current_pairs.extend(fallback_sample)
                    else:
                        current_pairs.extend(remaining_after_cut)
                break

    print(f"Number of additional pairs added: {len(current_pairs)} (after possibly scaling P and Q)")
    return current_pairs


def generate_pseudolabels(
    utilities: Dict[Any, Dict[str, float]],
    existing_pairs_set: Set[Tuple[Any, Any]],
    available_edges: Set[Tuple[Any, Any]],
    confidence_threshold: float
) -> Dict[Tuple[Any, Any], Dict[Any, int]]:
    """
    Generates pseudolabels for unsampled pairs using the Thurstonian model.
    
    Args:
        utilities: Dict mapping option IDs to {'mean': float, 'variance': float}
        existing_pairs_set: Set of existing (option_A_id, option_B_id) tuples
        available_edges: Set of available edges to sample from
        confidence_threshold: Confidence threshold for generating pseudolabels
        
    Returns:
        Dictionary mapping (option_A_id, option_B_id) to counts dictionary
    """
    unsampled_pairs = [pair for pair in available_edges if pair not in existing_pairs_set]
    normal = torch.distributions.Normal(0, 1)
    pseudolabels_counts = {}
    num_pseudolabels_added = 0

    for A_id, B_id in unsampled_pairs:  # Keep original orientation
        mu_A = utilities[A_id]['mean']
        mu_B = utilities[B_id]['mean']
        sigma2_A = utilities[A_id]['variance']
        sigma2_B = utilities[B_id]['variance']

        variance = sigma2_A + sigma2_B + 1e-5
        delta = mu_A - mu_B
        z = delta / np.sqrt(variance)
        prob_A = normal.cdf(torch.tensor(z)).item()

        if prob_A >= confidence_threshold:
            pseudolabels_counts[(A_id, B_id)] = {A_id: 1, B_id: 0}  # Keep original orientation
            num_pseudolabels_added += 1
        elif prob_A <= 1 - confidence_threshold:
            pseudolabels_counts[(A_id, B_id)] = {A_id: 0, B_id: 1}  # Keep original orientation
            num_pseudolabels_added += 1

    print(f"Number of pseudolabels added: {num_pseudolabels_added}")
    return pseudolabels_counts


# ===================== UTILITY MODEL CLASS ===================== #

class ThurstonianActiveLearningUtilityModel(UtilityModel):
    """
    Active learning variant of the Thurstonian utility model.
    Uses a combination of utility differences and degree-based sampling to select edges.
    """
    
    def __init__(
        self,
        unparseable_mode: str,
        comparison_prompt_template: str,
        system_message: str,
        with_reasoning: bool,
        num_epochs: int = 1000,
        learning_rate: float = 0.01,
        edge_multiplier: float = 2.0,
        degree: int = 2,
        num_edges_per_iteration: int = 200,
        P: float = 10.0,
        Q: float = 20.0,
        use_pseudolabels: bool = False,
        pseudolabel_confidence_threshold: float = 0.95,
        seed: Optional[int] = None,
        conversation: Optional[List[Dict[str, str]]] = None,
        target_option: Optional[str] = None,
        K: int = 10,
        use_logprobs: bool = False
    ):
        """
        Initialize the Thurstonian Active Learning utility model.

        Args:
            unparseable_mode: How to handle unparseable responses
            comparison_prompt_template: Template for comparison prompts
            system_message: System message for agents that accept a system message
            with_reasoning: Whether to use response parsing
            num_epochs: Number of epochs for optimization
            learning_rate: Learning rate for optimization
            edge_multiplier: Multiplier for number of edges
            degree: Degree of initial regular graph
            num_edges_per_iteration: Number of edges to sample in each iteration
            P: Percentage defining bottom P% of utility differences to sample from
            Q: Percentage defining bottom Q% of total degrees to sample from
            use_pseudolabels: Whether to use pseudolabeling in final stage
            pseudolabel_confidence_threshold: Confidence threshold for pseudolabeling
            seed: Random seed for reproducibility
            K: Number of responses to generate per prompt
            target_option: Option to target with user prompt
            use_logprobs: If True, use logprobs-based single-pass preference computation instead of K-sample voting
        """
        # Call parent class's __init__ with required arguments
        super().__init__(
            unparseable_mode=unparseable_mode,
            comparison_prompt_template=comparison_prompt_template,
            system_message=system_message,
            with_reasoning=with_reasoning
        )
        
        # Store model-specific arguments as attributes
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.edge_multiplier = edge_multiplier
        self.degree = degree
        self.num_edges_per_iteration = num_edges_per_iteration
        self.P = P
        self.Q = Q
        self.use_pseudolabels = use_pseudolabels
        self.pseudolabel_confidence_threshold = pseudolabel_confidence_threshold
        self.seed = seed
        self.K = K
        self.use_logprobs = use_logprobs
        self.conversation = conversation
        self.target_option = target_option

    def _handle_edits(self, parsed_responses, prev_utilities, prompt_idx_to_key, edits):
        for prompt_idx, (id_A, id_B, dir) in prompt_idx_to_key.items():
            responses = parsed_responses[prompt_idx]
            if id_A in edits or id_B in edits:
                if id_A in edits:
                    a_target_utility = edits[id_A]
                if id_B in edits:
                    b_target_utility = edits[id_B]
                if id_A not in edits:
                    a_target_utility = prev_utilities[str(id_A)]['mean'] if str(id_A) in prev_utilities else prev_utilities[id_A]['mean']
                if id_B not in edits:
                    b_target_utility = prev_utilities[str(id_B)]['mean'] if str(id_B) in prev_utilities else prev_utilities[id_B]['mean']

                edit_preferred = 'A' if a_target_utility > b_target_utility else 'B'
                if dir == 'flipped':
                    edit_preferred = 'B' if edit_preferred == 'A' else 'A'
                parsed_responses[prompt_idx] = [edit_preferred] * len(responses)
        return parsed_responses
    
    def _save_checkpoint(self, graph: 'PreferenceGraph', utilities: Dict, iteration: int, save_dir: str, num_iterations: int) -> None:
        """Save a checkpoint of the current active learning state.

        Args:
            graph: PreferenceGraph with all edges accumulated so far
            utilities: Current utility estimates
            iteration: The iteration that just completed (-1 for initial fit, 0..N-1 for AL iterations)
            save_dir: Directory to write the checkpoint file
            num_iterations: Total number of AL iterations planned
        """
        from ...utils import convert_numpy
        checkpoint = {
            'iteration': iteration,
            'num_iterations': num_iterations,
            'graph_data': graph.export_data(),
            'utilities': utilities,
        }
        checkpoint = convert_numpy(checkpoint)
        checkpoint_path = os.path.join(save_dir, f"checkpoint_iteration_{iteration}.json")
        # Write to a temp file first, then rename for atomicity
        tmp_path = checkpoint_path + ".tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path} (iteration {iteration}, {len(graph.edges)} edges)")

    @staticmethod
    def _find_latest_checkpoint(save_dir: str) -> Optional[Dict]:
        """Find and load the latest checkpoint file from save_dir.

        Returns:
            The loaded checkpoint dict, or None if no checkpoint exists.
        """
        if save_dir is None or not os.path.isdir(save_dir):
            return None
        import glob
        pattern = os.path.join(save_dir, "checkpoint_iteration_*.json")
        checkpoint_files = glob.glob(pattern)
        # Filter out .tmp files
        checkpoint_files = [f for f in checkpoint_files if not f.endswith('.tmp')]
        if not checkpoint_files:
            return None
        # Parse iteration numbers and find the latest
        def _get_iter(path):
            basename = os.path.basename(path)
            # checkpoint_iteration_{N}.json  ->  N  (N can be -1)
            num_str = basename.replace("checkpoint_iteration_", "").replace(".json", "")
            try:
                return int(num_str)
            except ValueError:
                return -999
        checkpoint_files.sort(key=_get_iter)
        latest = checkpoint_files[-1]
        try:
            with open(latest, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Found checkpoint: {latest} (iteration {data.get('iteration')})")
            return data
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: could not load checkpoint {latest}: {e}")
            return None

    async def fit(
        self,
        graph: 'PreferenceGraph',
        agent: Any,
        edits: list = None,
        prev_utilities: Dict[Any, Dict[str, float]] = None,
        a_b_logits_only: bool = False,
        preference_cot_suffix: Optional[str] = None,
        preference_structured_json_schema: Optional[Dict[str, Any]] = None,
        save_dir: Optional[str] = None,
    ) -> Tuple[Dict[Any, Dict[str, float]], Dict[str, float]]:
        """
        Fit the model using active learning to select edges.

        Args:
            graph: PreferenceGraph object containing the preference data
            agent: The agent used for generating comparisons
            save_dir: Optional directory for saving iteration checkpoints.
                      If provided, a checkpoint is saved after each iteration
                      and the run can be resumed from a checkpoint on restart.

        Returns:
            Tuple containing:
            - option_utilities: Dict mapping each option ID to {'mean': float, 'variance': float}
            - metrics: Dict containing model metrics like log_loss and accuracy
        """
        if self.comparison_prompt_template is None:
            raise ValueError("comparison_prompt_template must be provided")

        # Calculate target number of edges and number of iterations
        N = len(graph.options)
        target_total_edges = int(self.edge_multiplier * N * np.log2(N))
        initial_edges = (N * self.degree) // 2
        remainder = target_total_edges - initial_edges
        if remainder <= 0:
            num_iterations = 0
        else:
            num_iterations = int(np.ceil(remainder / self.num_edges_per_iteration))

        print(f"Target total edges: {target_total_edges}")
        print(f"Initial edges: {initial_edges}")
        print(f"Number of iterations: {num_iterations}")

        # ---- Check for existing checkpoint and resume if found ----
        resume_iteration = -2  # -2 means "no checkpoint, start from scratch"
        checkpoint = self._find_latest_checkpoint(save_dir)
        if checkpoint is not None:
            resume_iteration = checkpoint['iteration']
            print(f"Resuming from checkpoint at iteration {resume_iteration} "
                  f"({len(checkpoint['graph_data']['edges'])} edges)")
            # Restore graph state from checkpoint
            from ...compute_utilities import PreferenceGraph
            graph_restored = PreferenceGraph.load_data(checkpoint['graph_data'])
            # Copy restored state into the existing graph object so the caller's
            # reference stays valid
            graph.edges = graph_restored.edges
            graph.training_edges = graph_restored.training_edges
            graph.training_edges_pool = graph_restored.training_edges_pool
            graph.holdout_edge_indices = graph_restored.holdout_edge_indices
            utilities = checkpoint['utilities']
            # Convert string keys back to int if needed (JSON round-trip issue)
            if utilities and isinstance(next(iter(utilities.keys())), str):
                try:
                    utilities = {int(k): v for k, v in utilities.items()}
                except (ValueError, TypeError):
                    pass  # keys are genuinely strings, leave them
            # Refit model from restored graph to get log_loss/accuracy
            utilities, model_log_loss, model_accuracy = fit_thurstonian_model(
                graph=graph,
                num_epochs=self.num_epochs,
                learning_rate=self.learning_rate
            )
            print(f"Restored model - Log Loss: {model_log_loss:.4f}, Accuracy: {model_accuracy * 100:.2f}%")

        # ---- Initial edge collection (skip if checkpoint covers it) ----
        if resume_iteration < -1:
            # No checkpoint for initial fit; run it from scratch
            # Generate initial pairs using regular graph
            initial_pairs = graph.sample_regular_graph(degree=self.degree, seed=self.seed)
            if len(initial_pairs) < initial_edges:
                # If we didn't get enough edges from regular graph, sample additional random edges
                needed = initial_edges - len(initial_pairs)
                remaining_edges = list(graph.training_edges_pool - set(initial_pairs))
                if remaining_edges:
                    additional = random.sample(remaining_edges, min(needed, len(remaining_edges)))
                    initial_pairs.extend(additional)

            # Get responses for initial pairs
            preference_data, prompt_list, prompt_idx_to_key = graph.generate_prompts(
                initial_pairs,
                self.comparison_prompt_template
            )

            # If structured JSON schema is provided, call message-based generator to use schema; otherwise use legacy batching
            if preference_structured_json_schema is not None and preference_cot_suffix is not None:
                messages = []
                for prompt in prompt_list:
                    msg = []
                    if self.system_message is not None and agent.accepts_system_message:
                        msg.append({"role": "system", "content": self.system_message})
                    if isinstance(prompt, dict) and prompt.get('type') == 'conversation':
                        # Multi-turn conversation prompt: append turns, add suffix to last user msg
                        msg.extend(prompt['messages'])
                        # Append CoT suffix to the last user message (or add new one)
                        if msg and msg[-1]['role'] == 'user':
                            msg[-1] = {**msg[-1], 'content': msg[-1]['content'] + "\n\n" + preference_cot_suffix}
                        else:
                            msg.append({"role": "user", "content": preference_cot_suffix})
                    else:
                        # Text or other prompt types
                        content = (prompt if isinstance(prompt, str) else str(prompt)) + "\n\n" + preference_cot_suffix
                        msg.append({"role": "user", "content": content})
                    messages.append(msg)
                # Duplicate messages K times
                messages_k = messages * self.K
                flat_responses = await generate_responses_from_messages(
                    agent,
                    messages=messages_k,
                    structured_json=preference_structured_json_schema,
                )
                # reshape back into {prompt_idx: [K responses]}
                responses = {}
                num_prompts = len(prompt_list)
                for i in range(num_prompts):
                    responses[i] = flat_responses[i::num_prompts]
            else:
                responses = await generate_responses(
                    agent=agent,
                    prompts=prompt_list,
                    system_message=self.system_message,
                    conversation=self.conversation,
                    target_option=self.target_option,
                    K=self.K,
                    a_b_logits_only=a_b_logits_only,
                    use_logprobs=self.use_logprobs,
                )

            if self.use_logprobs:
                # Logprobs mode: responses are dicts with pre-computed probabilities
                processed_preference_data = self.process_logprob_responses(
                    graph=graph,
                    responses=responses,
                    prompt_idx_to_key=prompt_idx_to_key,
                )
            else:
                parsed_responses = parse_responses_forced_choice(responses, with_reasoning=self.with_reasoning, is_gpt_oss="gpt-oss" in agent.model)

                if edits is not None:
                    parsed_responses = self._handle_edits(parsed_responses, prev_utilities, prompt_idx_to_key, edits)

                processed_preference_data = self.process_responses(
                    graph=graph,
                    responses=responses,
                    parsed_responses=parsed_responses,
                    prompt_idx_to_key=prompt_idx_to_key,
                    edits=edits
                )

            graph.add_edges(processed_preference_data)

            # Initial fit
            utilities, model_log_loss, model_accuracy = fit_thurstonian_model(
                graph=graph,
                num_epochs=self.num_epochs,
                learning_rate=self.learning_rate
            )

            print(f"Initial model - Log Loss: {model_log_loss:.4f}, Accuracy: {model_accuracy * 100:.2f}%")

            # Save initial checkpoint
            if save_dir is not None:
                self._save_checkpoint(graph, utilities, -1, save_dir, num_iterations)

        # Active learning iterations
        for iteration in range(num_iterations):
            # Skip iterations that are already covered by the checkpoint
            if iteration <= resume_iteration:
                print(f"\nSkipping iteration {iteration + 1}/{num_iterations} (covered by checkpoint)")
                continue

            print(f"\nIteration {iteration + 1}/{num_iterations}")
            print(f"Sampling up to {self.num_edges_per_iteration} new pairs from the intersection of the bottom {self.P}% utility differences and bottom {self.Q}% total degrees.")

            # Get current utilities and existing pairs
            existing_pairs_set = set(
                (edge.option_A['id'], edge.option_B['id'])
                for edge in graph.edges.values()
            )

            # Generate additional pairs
            additional_pairs = generate_additional_pairs(
                utilities,
                existing_pairs_set,
                graph.training_edges_pool,
                self.num_edges_per_iteration,
                self.P, self.Q,
                seed=self.seed
            )

            if not additional_pairs:  # No more pairs to sample
                break

            # Get responses for additional pairs
            preference_data, prompt_list, prompt_idx_to_key = graph.generate_prompts(
                additional_pairs,
                self.comparison_prompt_template
            )

            if preference_structured_json_schema is not None and preference_cot_suffix is not None:
                messages = []
                for prompt in prompt_list:
                    msg = []
                    if self.system_message is not None and agent.accepts_system_message:
                        msg.append({"role": "system", "content": self.system_message})
                    if isinstance(prompt, dict) and prompt.get('type') == 'conversation':
                        msg.extend(prompt['messages'])
                        if msg and msg[-1]['role'] == 'user':
                            msg[-1] = {**msg[-1], 'content': msg[-1]['content'] + "\n\n" + preference_cot_suffix}
                        else:
                            msg.append({"role": "user", "content": preference_cot_suffix})
                    else:
                        content = (prompt if isinstance(prompt, str) else str(prompt)) + "\n\n" + preference_cot_suffix
                        msg.append({"role": "user", "content": content})
                    messages.append(msg)
                messages_k = messages * self.K
                flat_responses = await generate_responses_from_messages(
                    agent,
                    messages=messages_k,
                    structured_json=preference_structured_json_schema,
                )
                responses = {}
                num_prompts = len(prompt_list)
                for i in range(num_prompts):
                    responses[i] = flat_responses[i::num_prompts]
            else:
                responses = await generate_responses(
                    agent=agent,
                    prompts=prompt_list,
                    system_message=self.system_message,
                    conversation=self.conversation,
                    target_option=self.target_option,
                    K=self.K,
                    a_b_logits_only=a_b_logits_only,
                    use_logprobs=self.use_logprobs,
                )

            if self.use_logprobs:
                processed_preference_data = self.process_logprob_responses(
                    graph=graph,
                    responses=responses,
                    prompt_idx_to_key=prompt_idx_to_key,
                )
            else:
                parsed_responses = parse_responses_forced_choice(responses, with_reasoning=self.with_reasoning, is_gpt_oss="gpt-oss" in agent.model)
                if edits is not None:
                    parsed_responses = self._handle_edits(parsed_responses, prev_utilities, prompt_idx_to_key, edits)
                processed_preference_data = self.process_responses(
                    graph=graph,
                    responses=responses,
                    parsed_responses=parsed_responses,
                    prompt_idx_to_key=prompt_idx_to_key
                )

            graph.add_edges(processed_preference_data)

            # Refit model
            utilities, model_log_loss, model_accuracy = fit_thurstonian_model(
                graph=graph,
                num_epochs=self.num_epochs,
                learning_rate=self.learning_rate
            )

            print(f"Updated model - Log Loss: {model_log_loss:.4f}, Accuracy: {model_accuracy * 100:.2f}%")

            # Save checkpoint after each iteration (re-enabled for resume after crashes)
            if save_dir is not None:
                self._save_checkpoint(graph, utilities, iteration, save_dir, num_iterations)

        # Optional: Generate pseudolabels
        if self.use_pseudolabels:
            print("\nGenerating pseudolabels using the current Thurstonian model.")
            existing_pairs_set = set(
                (edge.option_A['id'], edge.option_B['id'])
                for edge in graph.edges.values()
            )
            
            pseudolabels = generate_pseudolabels(
                utilities,
                existing_pairs_set,
                graph.training_edges_pool,
                self.pseudolabel_confidence_threshold
            )
            
            # Convert pseudolabels into preference data format and add to graph
            for (A_id, B_id), counts in pseudolabels.items():
                # Create synthetic preference data
                prob_A = counts[A_id] / (counts[A_id] + counts[B_id])
                processed_data = [{
                    'option_A': graph.options_by_id[A_id],
                    'option_B': graph.options_by_id[B_id],
                    'probability_A': prob_A,
                    'aux_data': {
                        'is_pseudolabel': True,
                        'count_A': counts[A_id],
                        'count_B': counts[B_id],
                        'total_responses': counts[A_id] + counts[B_id]
                    }
                }]
                graph.add_edges(processed_data)
            
            # Final fit with pseudolabels
            utilities, model_log_loss, model_accuracy = fit_thurstonian_model(
                graph=graph,
                num_epochs=self.num_epochs,
                learning_rate=self.learning_rate
            )

            print(f"Final model with pseudolabels - Log Loss: {model_log_loss:.4f}, Accuracy: {model_accuracy * 100:.2f}%")

        metrics = {
            'log_loss': float(model_log_loss),
            'accuracy': float(model_accuracy)
        }
        
        return utilities, metrics
    
    @classmethod
    def evaluate(
        cls,
        graph: 'PreferenceGraph',
        utilities: Dict[Any, Dict[str, float]],
        edge_indices: List[Tuple[Any, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate the model's goodness-of-fit on the given edges.
        
        Args:
            graph: PreferenceGraph object containing the preference data
            utilities: Dict mapping each option ID to {'mean': float, 'variance': float}
            edge_indices: List of (option_A_id, option_B_id) tuples to evaluate on
            
        Returns:
            Dictionary containing evaluation metrics (log_loss and accuracy)
        """
        return evaluate_thurstonian_model(graph, utilities, edge_indices)
