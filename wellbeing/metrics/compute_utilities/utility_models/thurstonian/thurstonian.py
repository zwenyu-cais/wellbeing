import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from metrics.compute_utilities.models import UtilityModel
from metrics.compute_utilities.utils import generate_responses, parse_responses_forced_choice
from .utils import fit_thurstonian_model, evaluate_thurstonian_model

class ThurstonianUtilityModel(UtilityModel):
    """
    Basic Thurstonian utility model that learns utilities from pairwise comparisons.
    This model assumes each option has a latent utility that follows a normal distribution,
    and preferences are determined by sampling from these distributions.
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
        K: int = 10
    ):
        """
        Initialize the Thurstonian utility model.
        
        Args:
            unparseable_mode: How to handle unparseable responses
            comparison_prompt_template: Template for comparison prompts
            system_message: System message for agents that accept a system message
            with_reasoning: Whether to use response parsing
            num_epochs: Number of epochs for optimization
            learning_rate: Learning rate for optimization
            edge_multiplier: Multiplier for number of edges
            K: Number of responses to generate per prompt
            **kwargs: Additional arguments passed to parent class
        """
        # Call parent class's __init__ with required arguments
        super().__init__(
            unparseable_mode=unparseable_mode,
            comparison_prompt_template=comparison_prompt_template,
            system_message=system_message,
            with_reasoning=with_reasoning
        )
        
        # Store Thurstonian-specific arguments as attributes
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.edge_multiplier = edge_multiplier
        self.K = K
    
    async def fit(
        self,
        graph: 'PreferenceGraph',
        agent: Any
    ) -> Tuple[Dict[Any, Dict[str, float]], Dict[str, float]]:
        """
        Fit the Thurstonian model to the preference data.
        
        Args:
            graph: PreferenceGraph object containing the preference data
            agent: The agent used for generating comparisons
            
        Returns:
            Tuple containing:
            - option_utilities: Dict mapping each option ID to {'mean': float, 'variance': float}
            - metrics: Dict containing model metrics like log_loss and accuracy
        """
        # Calculate number of edges based on graph size
        N = len(graph.options)
        target_total_edges = int(self.edge_multiplier * N * np.log2(N))
        
        # Sample edges from training pool
        sampled_edges = graph.sample_random_edges(target_total_edges)
        
        # Generate prompts for sampled edges
        preference_data, prompt_list, prompt_idx_to_key = graph.generate_prompts(
            sampled_edges,
            self.comparison_prompt_template
        )
        
        # Generate responses using the agent
        responses = await generate_responses(
            agent=agent,
            prompts=prompt_list,
            system_message=self.system_message,
            K=self.K
        )
        
        # Parse responses and process them into preference data
        parsed_responses = parse_responses_forced_choice(responses, with_reasoning=self.with_reasoning)
        processed_preference_data = self.process_responses(
            graph=graph,
            responses=responses,
            parsed_responses=parsed_responses,
            prompt_idx_to_key=prompt_idx_to_key
        )
        
        # Add edges to graph
        graph.add_edges(processed_preference_data)
        
        # Fit the Thurstonian model
        option_utilities, model_log_loss, model_accuracy = fit_thurstonian_model(
            graph=graph,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate
        )
        
        metrics = {
            'log_loss': float(model_log_loss),
            'accuracy': float(model_accuracy)
        }
        
        return option_utilities, metrics
    
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

