# models.py

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import logging
import random

logger = logging.getLogger(__name__)

class UtilityModel(ABC):
    """
    Abstract base class for utility models that learn preferences from pairwise comparisons.
    
    Now stateful, with a 'unparseable_mode' attribute controlling how to handle unparseable responses.
    """

    def __init__(
        self,
        unparseable_mode: str,
        comparison_prompt_template: str,
        system_message: str,
        with_reasoning: bool,
        **kwargs
    ):
        """
        Initialize a UtilityModel.
        
        Args:
            unparseable_mode: How to handle unparseable responses (can be "skip", "random", or "distribution")
            comparison_prompt_template: Template for comparison prompts
            system_message: System message for agents that accept a system message
            with_reasoning: Whether to use response parsing for comparison_prompt_template_reasoning_default
            **kwargs: Additional arguments specific to each utility model implementation
        """
        # Validate unparseable_mode
        valid_modes = ["skip", "random", "distribution"]
        if unparseable_mode not in valid_modes:
            raise ValueError(f"unparseable_mode must be one of {valid_modes}, got '{unparseable_mode}'")
        
        # Store required arguments as attributes
        self.unparseable_mode = unparseable_mode
        self.comparison_prompt_template = comparison_prompt_template
        self.system_message = system_message
        self.with_reasoning = with_reasoning

    @abstractmethod
    async def fit(
        self,
        graph: 'PreferenceGraph',
        agent: Any
    ) -> Tuple[Dict[Any, Dict[str, float]], Dict[str, float]]:
        """
        Fit the utility model to the preference data.
        
        Args:
            graph: PreferenceGraph object containing the preference data
            agent: The agent used for generating comparisons
            
        Returns:
            Tuple containing:
            - option_utilities: Dict mapping each option ID to {'mean': float, 'variance': float}
            - metrics: Dict containing model metrics like log_loss and accuracy
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
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
            Dictionary containing evaluation metrics (e.g. log_loss, accuracy)
        """
        pass

    def process_logprob_responses(
        self,
        graph: 'PreferenceGraph',
        responses: Dict[int, list],
        prompt_idx_to_key: Dict[int, Tuple[Any, Any, str]],
    ) -> List[Dict]:
        """
        Convert logprobs-based responses into probabilities of preferring A over B.

        When use_logprobs=True, each response is a dict with '__logprobs__': True
        and a pre-computed 'probability_A' from token logprobs. This method handles
        the original/flipped orientation and produces preference data compatible with
        graph.add_edges.

        Args:
            graph: The PreferenceGraph containing options and edges
            responses: Dict mapping prompt_idx to list of 1 logprobs response dict
            prompt_idx_to_key: Mapping from prompt index to (option_A_id, option_B_id, direction)

        Returns:
            A list of preference data dictionaries ready for graph.add_edges.
        """
        # Group by pair
        pair_data = {}  # (A_id, B_id) -> {'prob_a_values': [], 'logprobs_info': []}

        for prompt_idx, response_list in responses.items():
            A_id, B_id, direction = prompt_idx_to_key[prompt_idx]
            resp = response_list[0]  # Single response per prompt
            prob_a = resp['probability_A']

            pair_key = (A_id, B_id)
            if pair_key not in pair_data:
                pair_data[pair_key] = {
                    'option_A': graph.options_by_id[A_id],
                    'option_B': graph.options_by_id[B_id],
                    'prob_a_values': [],
                    'logprobs_info': [],
                }

            # If flipped, the model's P(A) means P(original_B), so P(original_A) = 1 - P(A)
            if direction == 'flipped':
                pair_data[pair_key]['prob_a_values'].append(1.0 - prob_a)
            else:
                pair_data[pair_key]['prob_a_values'].append(prob_a)
            pair_data[pair_key]['logprobs_info'].append({
                'direction': direction,
                'raw_prob_a': prob_a,
                'text': resp.get('text', ''),
                'token_alts': resp.get('token_alts', []),
                'lp_a': resp.get('lp_a'),
                'lp_b': resp.get('lp_b'),
            })

        # Convert to preference data
        preference_data = []
        for (A_id, B_id), data in pair_data.items():
            # Average P(A) across original and flipped directions
            probability_A = sum(data['prob_a_values']) / len(data['prob_a_values'])
            entry = {
                'option_A': data['option_A'],
                'option_B': data['option_B'],
                'probability_A': probability_A,
                'aux_data': {
                    'count_A': probability_A,
                    'count_B': 1.0 - probability_A,
                    'total_responses': len(data['prob_a_values']),
                    'original_responses': [],
                    'flipped_responses': [],
                    'original_parsed': [],
                    'flipped_parsed': [],
                    'logprobs_mode': True,
                    'logprobs_info': data['logprobs_info'],
                    'per_direction_prob_a': data['prob_a_values'],
                }
            }
            preference_data.append(entry)

        return preference_data

    def process_responses(
        self,
        graph: 'PreferenceGraph',
        responses: Dict[int, List[str]],
        parsed_responses: Dict[int, List[str]],
        prompt_idx_to_key: Dict[int, Tuple[Any, Any, str]],
        edits: list = None
    ) -> List[Dict]:
        """
        Convert raw responses into probabilities of preferring A over B.

        The logic is equivalent to the original PreferenceGraph.process_responses,
        **but** now we incorporate `unparseable_mode` handling:
           - "skip": ignore unparseable
           - "random": randomly choose A or B for unparseable
           - "distribution": treat unparseable as [0.5, 0.5]

        Args:
            graph: The PreferenceGraph containing options and edges
            responses: Dict mapping prompt_idx to list of K raw responses
            parsed_responses: Dict mapping prompt_idx to list of K parsed responses ('A', 'B', or 'unparseable')
            prompt_idx_to_key: Mapping from prompt index to (option_A_id, option_B_id, direction)

        Returns:
            A list of preference data dictionaries ready for `graph.add_edges`, where each entry has:
              - option_A
              - option_B
              - probability_A
              - aux_data (with counts, original responses, etc.)
        """
        # Group raw responses by pair (in the original orientation)
        pair_data = {}  # (A_id, B_id) -> data structure
        total_responses_count = 0
        total_unparseable_count = 0

        for prompt_idx, response_list in responses.items():
            A_id, B_id, direction = prompt_idx_to_key[prompt_idx]
            parsed_list = parsed_responses[prompt_idx]  # The K parsed responses

            # Log each unparseable response with full context
            for raw, parsed in zip(response_list, parsed_list):
                total_responses_count += 1
                if parsed == 'unparseable':
                    total_unparseable_count += 1
                    opt_a = graph.options_by_id[A_id]
                    opt_b = graph.options_by_id[B_id]
                    desc_a = opt_a.get('description', opt_a.get('id'))
                    desc_b = opt_b.get('description', opt_b.get('id'))
                    logger.warning(
                        "UNPARSEABLE response | pair=(%s vs %s) | direction=%s | raw=%r",
                        desc_a, desc_b, direction, raw,
                    )

            # Use the orientation as-is (A_id, B_id)
            pair_key = (A_id, B_id)
            if pair_key not in pair_data:
                pair_data[pair_key] = {
                    'option_A': graph.options_by_id[A_id],
                    'option_B': graph.options_by_id[B_id],
                    'original_responses': [],
                    'flipped_responses': [],
                    'original_parsed': [],
                    'flipped_parsed': []
                }

            # We store the raw and parsed responses in separate buckets
            if direction == 'original':
                pair_data[pair_key]['original_responses'].extend(response_list)
                pair_data[pair_key]['original_parsed'].extend(parsed_list)
            else:  # 'flipped'
                pair_data[pair_key]['flipped_responses'].extend(response_list)
                pair_data[pair_key]['flipped_parsed'].extend(parsed_list)

        # Now, we convert each pair's responses into a probability P(A)
        preference_data = []
        rng = random.Random(42)  # or some other seed for stable 'random' in "random" mode

        for (A_id, B_id), data in pair_data.items():
            # Instead of just counting 'A' or 'B', we keep track of distributions:
            # e.g. 'A' -> (1,0), 'B' -> (0,1), 'unparseable' -> depends on unparseable_mode.
            dist_list = []  # Will store (a_val, b_val) for each response

            # A small helper for flipping:
            # if direction == 'flipped' and user picks 'A', that means B in original orientation
            # if direction == 'flipped' and user picks 'B', that means A in original orientation
            def add_to_dist_list(parsed_char, is_flipped=False):
                """
                Add an appropriate (A, B) distribution to dist_list based on the parse,
                flipping if needed.
                """
                if parsed_char == 'A':
                    if not is_flipped:
                        dist_list.append((1.0, 0.0))
                    else:
                        dist_list.append((0.0, 1.0))
                elif parsed_char == 'B':
                    if not is_flipped:
                        dist_list.append((0.0, 1.0))
                    else:
                        dist_list.append((1.0, 0.0))
                else:
                    # unparseable
                    if self.unparseable_mode == "skip":
                        # do nothing -> skip
                        pass
                    elif self.unparseable_mode == "random":
                        # randomly pick A or B
                        if rng.random() < 0.5:
                            # pick A
                            if not is_flipped:
                                dist_list.append((1.0, 0.0))
                            else:
                                dist_list.append((0.0, 1.0))
                        else:
                            # pick B
                            if not is_flipped:
                                dist_list.append((0.0, 1.0))
                            else:
                                dist_list.append((1.0, 0.0))
                    elif self.unparseable_mode == "distribution":
                        # treat as 50/50
                        dist_list.append((0.5, 0.5))

            # Original ordering
            for parsed in data['original_parsed']:
                add_to_dist_list(parsed, is_flipped=False)

            # Flipped ordering
            for parsed in data['flipped_parsed']:
                add_to_dist_list(parsed, is_flipped=True)

            # Summarize results
            total_A = sum(d[0] for d in dist_list)
            total_B = sum(d[1] for d in dist_list)
            total_responses = len(dist_list)

            if total_responses > 0:
                probability_A = total_A / (total_A + total_B)  # or total_A / total_responses if we interpret it differently
                entry = {
                    'option_A': data['option_A'],
                    'option_B': data['option_B'],
                    'probability_A': probability_A,
                    'aux_data': {
                        'count_A': total_A,  # This might be fractional now
                        'count_B': total_B,  # Also might be fractional
                        'total_responses': total_responses,
                        'original_responses': data['original_responses'],
                        'flipped_responses': data['flipped_responses'],
                        'original_parsed': data['original_parsed'],
                        'flipped_parsed': data['flipped_parsed'],
                        'unparseable_mode': self.unparseable_mode
                    }
                }
                preference_data.append(entry)
            else:
                # if total_responses == 0, that means everything was "skip"
                # or no valid responses were found
                pass

        if total_unparseable_count > 0:
            rate = total_unparseable_count / total_responses_count * 100
            logger.warning(
                "Unparseable summary: %d/%d responses (%.1f%%) were unparseable "
                "(handled via unparseable_mode=%r)",
                total_unparseable_count, total_responses_count, rate,
                self.unparseable_mode,
            )

        return preference_data
