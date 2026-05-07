import asyncio
import json
import time
import numpy as np
import random
import itertools
import argparse
import os
from collections import defaultdict
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
import torch
import torch.nn.functional as F
import networkx as nx
import re
from .utils import (
    create_agent,
    generate_responses,
    parse_responses_forced_choice,
    flatten_hierarchical_options,
    convert_numpy,
    load_config,
    evaluate_holdout_set
)
from .llm_agent import LiteLLMAgent, HuggingFaceAgent, LLMAgent
from .templates import comparison_prompt_template_default, comparison_prompt_template_reasoning_default
from .models import UtilityModel
import yaml
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import importlib

from .utility_models import (
    ThurstonianUtilityModel,
    ThurstonianActiveLearningUtilityModel,
)


# ===================== MEDIA TAG RESOLUTION ===================== #

def resolve_audio_tags(options_list: List[Dict[str, Any]], audio_manifest_path: str) -> List[Dict[str, Any]]:
    """
    Resolve <!audio:{hash}!> tags in option text using an audio manifest JSON file.

    The manifest maps 16-char hex hashes to absolute file paths:
    {
        "a1b2c3d4e5f6g7h8": "/data/.../audio1.wav",
        ...
    }

    A tag like <!audio:a1b2c3d4e5f6g7h8!> in a text option's description gets
    resolved: the option is converted to an audio option with the appropriate path.

    Args:
        options_list: List of option dicts (may contain text options with audio tags)
        audio_manifest_path: Path to JSON file mapping hashes to audio file paths

    Returns:
        Updated options_list with audio tags resolved to audio options
    """
    import os as _os

    if not _os.path.exists(audio_manifest_path):
        raise FileNotFoundError(f"Audio manifest not found: {audio_manifest_path}")

    with open(audio_manifest_path, 'r') as f:
        manifest = json.load(f)

    audio_tag_pattern = re.compile(r'<!audio:([a-fA-F0-9]{16})!>')

    def _get_audio_path(entry):
        """Extract audio path from manifest entry (str or dict with 'path' key)."""
        if isinstance(entry, str):
            return entry
        elif isinstance(entry, dict):
            return entry['path']
        raise ValueError(f"Unexpected manifest entry type: {type(entry)}")

    resolved = []

    for item in options_list:
        if isinstance(item, str):
            match = audio_tag_pattern.search(item)
            if match:
                hash_key = match.group(1)
                if hash_key not in manifest:
                    raise ValueError(f"Audio hash '{hash_key}' not found in manifest: {audio_manifest_path}")
                audio_path = _get_audio_path(manifest[hash_key])
                text_without_tag = audio_tag_pattern.sub('', item).strip()
                if text_without_tag:
                    resolved.append({
                        'type': 'text_with_audio',
                        'id': hash_key,
                        'audio_path': audio_path,
                        'name': _os.path.basename(audio_path),
                        'description': text_without_tag,
                    })
                else:
                    resolved.append({
                        'type': 'audio',
                        'id': hash_key,
                        'audio_path': audio_path,
                        'name': _os.path.basename(audio_path),
                        'description': f"[AUDIO:{_os.path.basename(audio_path)}]",
                    })
            else:
                resolved.append(item)
        elif isinstance(item, dict):
            description = item.get('description', '')
            if isinstance(description, str):
                all_hashes = audio_tag_pattern.findall(description)
                if len(all_hashes) > 1:
                    # Multiple audio tags — create combination_with_audios
                    components = []
                    for h in all_hashes:
                        if h not in manifest:
                            raise ValueError(f"Audio hash '{h}' not found in manifest: {audio_manifest_path}")
                        aud_path = _get_audio_path(manifest[h])
                        components.append({'type': 'audio', 'audio_path': aud_path, 'description': ''})
                    new_item = item.copy()
                    new_item['type'] = 'combination_with_audios'
                    new_item['components'] = components
                    text_without_tag = audio_tag_pattern.sub('', description).strip()
                    new_item['description'] = text_without_tag
                    resolved.append(new_item)
                elif len(all_hashes) == 1:
                    hash_key = all_hashes[0]
                    if hash_key not in manifest:
                        raise ValueError(f"Audio hash '{hash_key}' not found in manifest: {audio_manifest_path}")
                    audio_path = _get_audio_path(manifest[hash_key])
                    new_item = item.copy()
                    new_item['audio_path'] = audio_path
                    new_item.setdefault('name', _os.path.basename(audio_path))
                    new_item.setdefault('id', hash_key)
                    text_without_tag = audio_tag_pattern.sub('', description).strip()
                    if text_without_tag:
                        new_item['type'] = 'text_with_audio'
                        new_item['description'] = text_without_tag
                    else:
                        new_item['type'] = 'audio'
                        new_item['description'] = f"You get to listen to another audio clip like this one: [AUDIO:{_os.path.basename(audio_path)}]"
                    resolved.append(new_item)
                else:
                    resolved.append(item)
            else:
                resolved.append(item)
        else:
            resolved.append(item)

    return resolved


def resolve_image_tags(options_list: List[Dict[str, Any]], image_manifest_path: str) -> List[Dict[str, Any]]:
    """
    Resolve <!image:{hash}!> tags in option text using an image manifest JSON file.

    The manifest maps 16-char hex hashes to absolute file paths:
    {
        "a1b2c3d4e5f6g7h8": "/data/.../image1.png",
        ...
    }

    A tag like <!image:a1b2c3d4e5f6g7h8!> in a text option's description gets
    resolved: the option is converted to an image option with the appropriate path.

    Args:
        options_list: List of option dicts (may contain text options with image tags)
        image_manifest_path: Path to JSON file mapping hashes to image file paths

    Returns:
        Updated options_list with image tags resolved to image options
    """
    import os as _os

    if not _os.path.exists(image_manifest_path):
        raise FileNotFoundError(f"Image manifest not found: {image_manifest_path}")

    with open(image_manifest_path, 'r') as f:
        manifest = json.load(f)

    image_tag_pattern = re.compile(r'<!image:([a-fA-F0-9]{16})!>')

    def _get_image_path(entry):
        """Extract image path from manifest entry (str or dict with 'path' key)."""
        if isinstance(entry, str):
            return entry
        elif isinstance(entry, dict):
            return entry['path']
        raise ValueError(f"Unexpected manifest entry type: {type(entry)}")

    resolved = []

    for item in options_list:
        if isinstance(item, str):
            # Plain string option
            match = image_tag_pattern.search(item)
            if match:
                hash_key = match.group(1)
                if hash_key not in manifest:
                    raise ValueError(f"Image hash '{hash_key}' not found in manifest: {image_manifest_path}")
                image_path = _get_image_path(manifest[hash_key])
                text_without_tag = image_tag_pattern.sub('', item).strip()
                if text_without_tag:
                    resolved.append({
                        'type': 'text_with_image',
                        'id': hash_key,
                        'path': image_path,
                        'name': _os.path.basename(image_path),
                        'description': text_without_tag,
                    })
                else:
                    resolved.append({
                        'type': 'image',
                        'id': hash_key,
                        'path': image_path,
                        'name': _os.path.basename(image_path),
                        'description': f"[IMAGE:{_os.path.basename(image_path)}]",
                    })
            else:
                resolved.append(item)
        elif isinstance(item, dict):
            description = item.get('description', '')
            if isinstance(description, str):
                all_hashes = image_tag_pattern.findall(description)
                if len(all_hashes) > 1:
                    # Multiple image tags — create combination_with_images
                    components = []
                    for h in all_hashes:
                        if h not in manifest:
                            raise ValueError(f"Image hash '{h}' not found in manifest: {image_manifest_path}")
                        img_path = _get_image_path(manifest[h])
                        components.append({'type': 'image', 'path': img_path, 'description': ''})
                    new_item = item.copy()
                    new_item['type'] = 'combination_with_images'
                    new_item['components'] = components
                    text_without_tag = image_tag_pattern.sub('', description).strip()
                    new_item['description'] = text_without_tag
                    resolved.append(new_item)
                elif len(all_hashes) == 1:
                    hash_key = all_hashes[0]
                    if hash_key not in manifest:
                        raise ValueError(f"Image hash '{hash_key}' not found in manifest: {image_manifest_path}")
                    image_path = _get_image_path(manifest[hash_key])
                    new_item = item.copy()
                    new_item['path'] = image_path
                    new_item.setdefault('name', _os.path.basename(image_path))
                    new_item.setdefault('id', hash_key)
                    text_without_tag = image_tag_pattern.sub('', description).strip()
                    if text_without_tag:
                        new_item['type'] = 'text_with_image'
                        new_item['description'] = text_without_tag
                    else:
                        new_item['type'] = 'image'
                        new_item['description'] = f"You get to see another image like this one: [IMAGE:{_os.path.basename(image_path)}]"
                    resolved.append(new_item)
                else:
                    resolved.append(item)
            else:
                resolved.append(item)
        else:
            resolved.append(item)

    return resolved


# ===================== DEFAULT PROMPTS ===================== #

class PreferenceEdge:
    """
    A class representing a preference edge between two options.
    """
    
    def __init__(self, option_A: Dict[str, Any], option_B: Dict[str, Any], probability_A: float, aux_data: Dict[str, Any] = None):
        """
        Initialize a preference edge.
        
        Args:
            option_A: First option dictionary with at least {'id': Any, 'description': str}
            option_B: Second option dictionary with at least {'id': Any, 'description': str}
            probability_A: Probability of A being preferred over B
            aux_data: Optional dictionary of auxiliary data about this preference
        """
        # Keep options in the order given, no sorting
        self.option_A = option_A
        self.option_B = option_B
        self.probability_A = probability_A  # P(A > B)
        self.aux_data = aux_data if aux_data is not None else {}
    
    def __eq__(self, other: 'PreferenceEdge') -> bool:
        """Two preference edges are equal if they have the same orientation of A→B."""
        if not isinstance(other, PreferenceEdge):
            return False
        return (self.option_A['id'] == other.option_A['id'] and 
                self.option_B['id'] == other.option_B['id'])
    
    def __hash__(self) -> int:
        """Hash based on the ordered pair (A_id, B_id)."""
        return hash((self.option_A['id'], self.option_B['id']))
    
    def __repr__(self) -> str:
        return f"PreferenceEdge({self.option_A['id']} vs {self.option_B['id']}, P(A)={self.probability_A:.3f})"



class PreferenceGraph:
    """
    A class representing a graph of pairwise preferences between options.
    Handles creation of training/holdout edge sets and sampling strategies.
    """
    
    def __init__(self, options: List[Dict[str, Any]], holdout_fraction: float = 0.0, seed: int = 42):
        """
        Initialize a preference graph with training and holdout edge indices.
        
        Args:
            options: List of dictionaries, each containing at least:
                    {'id': str/int, 'description': str}
            holdout_fraction: Fraction of edges to hold out for evaluation
            seed: Random seed for reproducibility
        """
        self.options = options
        self.option_id_to_idx = {option['id']: idx for idx, option in enumerate(options)}
        self.options_by_id = {opt['id']: opt for opt in options}
        
        # Generate all possible edge indices as tuples
        all_edge_indices = list(itertools.combinations([opt['id'] for opt in options], 2))
        
        # Split into training and holdout indices
        random.seed(seed)
        if holdout_fraction <= 0:
            self.training_edges_pool = set(all_edge_indices)
            self.holdout_edge_indices = set()
        else:
            total_edges = len(all_edge_indices)
            # Cap holdout size at min(fraction-based size, 1000)
            fraction_based_size = int(total_edges * holdout_fraction)
            holdout_size = min(fraction_based_size, 1000)
            
            # Randomly select holdout edges
            all_edges_shuffled = all_edge_indices.copy()
            random.shuffle(all_edges_shuffled)
            self.holdout_edge_indices = set(all_edges_shuffled[:holdout_size])
            self.training_edges_pool = set(all_edges_shuffled[holdout_size:])
            
        # Initialize sets for tracking actual edges in the graph
        self.training_edges = set()  # Training edges currently in graph
        self.edges = {}  # Map from edge index tuple to PreferenceEdge
            
        print(f"Total possible edges: {len(all_edge_indices)}")
        print(f"Training pool: {len(self.training_edges_pool)}, Holdout: {len(self.holdout_edge_indices)}")
    
    @classmethod
    def load_data(cls, data: Dict[str, Any]) -> 'PreferenceGraph':
        """
        Create a PreferenceGraph instance from exported data.
        
        Args:
            data: Dictionary containing the graph data, as exported by export_data
            
        Returns:
            A new PreferenceGraph instance with the loaded data
        """
        # Create instance with options
        graph = cls(options=data['options'])
        
        # Restore edge sets
        graph.training_edges = set(tuple(edge) for edge in data['training_edges'])
        graph.training_edges_pool = set(tuple(edge) for edge in data['training_edges_pool'])
        graph.holdout_edge_indices = set(tuple(edge) for edge in data['holdout_edge_indices'])
        
        # Restore edges
        graph.edges = {}
        # Build lookup from option id to full option dict
        options_by_id = {opt['id']: opt for opt in data['options']}
        import ast
        for edge_key_str, edge_data in data['edges'].items():
            # Convert string edge key back to tuple. Edge keys may be int tuples
            # (old format) or string-id tuples (new format), so use ast.literal_eval.
            edge_key = ast.literal_eval(edge_key_str)
            # Handle both new format (option_A_id) and old format (option_A)
            if 'option_A_id' in edge_data:
                option_A = options_by_id[edge_data['option_A_id']]
                option_B = options_by_id[edge_data['option_B_id']]
            else:
                option_A = edge_data['option_A']
                option_B = edge_data['option_B']
            # Create PreferenceEdge instance
            edge = PreferenceEdge(
                option_A=option_A,
                option_B=option_B,
                probability_A=edge_data['probability_A'],
                aux_data=edge_data['aux_data']
            )
            graph.edges[edge_key] = edge
            
        return graph
    
    def export_data(self) -> Dict[str, Any]:
        """
        Export the graph data in a JSON-serializable format.
        
        Returns:
            Dictionary containing all the graph data in a serializable format
        """
        return {
            'options': self.options,
            'edges': {
                str(edge_key): {
                    'option_A_id': edge.option_A['id'],
                    'option_B_id': edge.option_B['id'],
                    'probability_A': edge.probability_A,
                    'aux_data': edge.aux_data
                }
                for edge_key, edge in self.edges.items()
            },
            'training_edges': [list(edge) for edge in self.training_edges],
            'training_edges_pool': [list(edge) for edge in self.training_edges_pool],
            'holdout_edge_indices': [list(edge) for edge in self.holdout_edge_indices]
        }
    
    def _option_needs_multimodal(self, option: Dict) -> bool:
        """Check if an option requires multimodal prompt handling (contains images or audio)."""
        opt_type = option.get('type', '')
        return opt_type in ('image', 'combination_with_images', 'image_quantity', 'text_with_image',
                            'audio', 'text_with_audio', 'conversation_with_image')

    def _option_is_conversation(self, option: Dict) -> bool:
        """Check if an option is a conversation type (multi-turn messages)."""
        return option.get('type', '') in ('conversation', 'conversation_with_image')

    def generate_prompts(self, edge_indices: List[Tuple[Any, Any]], comparison_prompt_template: str) -> Tuple[List[Dict], List, Dict[int, Tuple]]:
        """
        Generate prompts for the given edge indices in both original and flipped ordering.

        Supports text options, image options (type='image'), and combinations with images
        (type='combination_with_images'). For options containing images, generates multimodal
        prompts in the format expected by VL models.

        Args:
            edge_indices: List of (option_A_id, option_B_id) tuples
            comparison_prompt_template: Template string with {option_A} and {option_B} placeholders

        Returns:
            Tuple containing:
            - preference_data: List of pair data with prompts
            - prompt_list: List of prompts (strings for text-only, dicts for multimodal)
            - prompt_idx_to_key: Mapping from prompt index to (option_A_id, option_B_id, direction)
        """
        preference_data = []
        prompt_list = []
        prompt_idx_to_key = {}
        prompt_idx = 0

        for pair_idx, (A_id, B_id) in enumerate(edge_indices):
            option_A = self.options_by_id[A_id]
            option_B = self.options_by_id[B_id]

            pair_data = {
                'pair_id': pair_idx,
                'option_A': option_A,
                'option_B': option_B,
                'prompts': []
            }

            # Check if either option contains images (single image or combination with images)
            A_needs_multimodal = self._option_needs_multimodal(option_A)
            B_needs_multimodal = self._option_needs_multimodal(option_B)
            A_is_conversation = self._option_is_conversation(option_A)
            B_is_conversation = self._option_is_conversation(option_B)

            # Generate prompts in both directions
            for direction in ['original', 'flipped']:
                if direction == 'original':
                    opt1, opt2 = option_A, option_B
                    opt1_multimodal, opt2_multimodal = A_needs_multimodal, B_needs_multimodal
                    opt1_conversation, opt2_conversation = A_is_conversation, B_is_conversation
                else:
                    opt1, opt2 = option_B, option_A
                    opt1_multimodal, opt2_multimodal = B_needs_multimodal, A_needs_multimodal
                    opt1_conversation, opt2_conversation = B_is_conversation, A_is_conversation

                # Generate prompt based on option types
                if (opt1_conversation or opt2_conversation) and (opt1_multimodal or opt2_multimodal):
                    # Conversation with images — hybrid method preserving image blocks
                    prompt = self._generate_conversation_with_image_prompt(
                        opt1, opt2, opt1_conversation, opt2_conversation, comparison_prompt_template
                    )
                elif opt1_conversation or opt2_conversation:
                    # At least one is a conversation (text-only) - create multi-turn prompt
                    prompt = self._generate_conversation_prompt(
                        opt1, opt2, opt1_conversation, opt2_conversation, comparison_prompt_template
                    )
                elif not opt1_multimodal and not opt2_multimodal:
                    # Both text - use standard template
                    prompt = comparison_prompt_template.format(
                        option_A=opt1['description'],
                        option_B=opt2['description']
                    )
                else:
                    # At least one contains images - create multimodal prompt
                    prompt = self._generate_multimodal_prompt(
                        opt1, opt2, opt1_multimodal, opt2_multimodal, comparison_prompt_template
                    )

                prompt_data = {
                    'prompt_idx': prompt_idx,
                    'prompt': prompt,
                    'direction': direction,
                    'responses': []
                }

                pair_data['prompts'].append(prompt_data)
                prompt_list.append(prompt)
                prompt_idx_to_key[prompt_idx] = (A_id, B_id, direction)
                prompt_idx += 1

            preference_data.append(pair_data)

        return preference_data, prompt_list, prompt_idx_to_key

    def _generate_multimodal_prompt(self, opt1, opt2, opt1_multimodal, opt2_multimodal, template):
        """
        Generate a multimodal prompt for VL/audio models when at least one option contains images or audio.

        Uses the comparison_prompt_template to derive framing text, so that EU ("happier")
        vs DU ("prefer") framing is respected for multimodal content too.

        Handles:
        - Single image options (type='image')
        - Combination options with embedded images (type='combination_with_images')
        - Single audio options (type='audio')
        - Text with audio (type='text_with_audio')
        - Text-only options

        Returns a dict with 'content' (list of text/image/audio parts),
        'images' (list of image paths), and 'audios' (list of audio paths).
        """
        # Parse the template to extract intro and closing text.
        # Templates have the form:
        #   <intro>\n\nExperience A:\n{option_A}\n\nExperience B:\n{option_B}\n\n<closing>
        # We extract the intro (before {option_A}) and closing (after {option_B}).
        # Use "Experience" or "Option" labels depending on what the template uses.
        option_label = "Experience"
        if "Option A:" in template or "Option A\n" in template:
            option_label = "Option"

        # Split on the placeholders to get intro and closing
        parts = template.split("{option_A}")
        intro_text = parts[0].rstrip() if len(parts) > 1 else ""
        remainder = parts[1] if len(parts) > 1 else template
        parts2 = remainder.split("{option_B}")
        closing_text = parts2[1].strip() if len(parts2) > 1 else 'Just answer with "A" or "B".'

        # Clean intro: remove trailing "Experience A:" or "Option A:" since we add it ourselves
        for suffix in [f"\n\n{option_label} A:", f"\n{option_label} A:", f"{option_label} A:"]:
            if intro_text.endswith(suffix):
                intro_text = intro_text[:-len(suffix)].rstrip()
                break

        content = []
        images = []
        audios = []

        def add_option_content(opt, label, is_first=False):
            """Add content for a single option, handling text, images, audio, and combinations."""
            opt_type = opt.get('type', '')

            prefix = f"{intro_text}\n\n" if is_first else "\n\n"

            if opt_type == 'image':
                # Single image option - show the image directly as the experience
                content.append({"type": "text", "text": f"{prefix}{option_label} {label}:\n"})
                content.append({"type": "image"})
                images.append(opt['path'])
            elif opt_type == 'image_quantity':
                content.append({"type": "text", "text": f"{prefix}{option_label} {label}:\n{opt['description']} "})
                content.append({"type": "image"})
                images.append(opt['path'])
            elif opt_type == 'text_with_image':
                content.append({"type": "text", "text": f"{prefix}{option_label} {label}:\n"})
                content.append({"type": "image"})
                images.append(opt['path'])
                content.append({"type": "text", "text": f"\n{opt['description']}"})
            elif opt_type == 'audio':
                # Single audio option - play the audio directly as the experience
                content.append({"type": "text", "text": f"{prefix}{option_label} {label}:\n"})
                content.append({"type": "audio"})
                audios.append(opt['audio_path'])
            elif opt_type == 'text_with_audio':
                content.append({"type": "text", "text": f"{prefix}{option_label} {label}:\n"})
                content.append({"type": "audio"})
                audios.append(opt['audio_path'])
                content.append({"type": "text", "text": f"\n{opt['description']}"})
            elif opt_type == 'combination_with_images':
                components = opt.get('components', [])
                content.append({"type": "text", "text": f"{prefix}{option_label} {label}:"})

                for comp in components:
                    comp_type = comp.get('type', '')
                    if comp_type == 'image':
                        content.append({"type": "text", "text": "\n- "})
                        content.append({"type": "image"})
                        images.append(comp['path'])
                    else:
                        content.append({"type": "text", "text": f"\n- {comp['description']}"})
            elif opt_type == 'combination_with_audios':
                components = opt.get('components', [])
                content.append({"type": "text", "text": f"{prefix}{option_label} {label}:"})

                for comp in components:
                    comp_type = comp.get('type', '')
                    if comp_type == 'audio':
                        content.append({"type": "text", "text": "\n- "})
                        content.append({"type": "audio"})
                        audios.append(comp['audio_path'])
                    else:
                        content.append({"type": "text", "text": f"\n- {comp['description']}"})
            else:
                # Pure text option
                content.append({"type": "text", "text": f"{prefix}{option_label} {label}:\n{opt['description']}"})

        # Add Option/Experience A
        add_option_content(opt1, "A", is_first=True)

        # Add Option/Experience B
        add_option_content(opt2, "B", is_first=False)

        # Closing instruction from template
        content.append({"type": "text", "text": f"\n\n{closing_text}"})

        return {
            'type': 'multimodal',
            'content': content,
            'images': images,
            'audios': audios,
        }

    def _parse_template_parts(self, template: str) -> Tuple[str, str, str]:
        """
        Parse a comparison prompt template into preamble, middle, and closing parts.

        The template is expected to have {option_A} and {option_B} placeholders.
        Returns (preamble, middle, closing) where:
            - preamble: everything before {option_A} (rstripped) -- typically ends
              with "Experience A:" or "Option A:"
            - middle: everything between {option_A} and {option_B} (stripped) --
              typically contains "Experience B:" or "Option B:"
            - closing: everything after {option_B} (stripped) -- the comparison
              question
        """
        # Find the positions of the placeholders
        a_pos = template.find('{option_A}')
        b_pos = template.find('{option_B}')

        if a_pos == -1 or b_pos == -1:
            # Fallback if template doesn't have expected placeholders
            return template, '', ''

        preamble = template[:a_pos].rstrip()
        middle = template[a_pos + len('{option_A}'):b_pos].strip()
        closing = template[b_pos + len('{option_B}'):].strip()
        return preamble, middle, closing

    def _generate_conversation_prompt(self, opt1, opt2, opt1_conversation, opt2_conversation, template):
        """
        Generate a multi-turn conversation prompt when at least one option is a conversation.

        Handles four cases:
        - Conversation vs Conversation: Both injected as multi-turn messages
        - Conversation vs Text: Conversation injected as turns, text as inline content
        - Text vs Conversation: Text as inline content, conversation as turns
        - (Text vs Text is handled by the standard template path, not here)

        The template's preamble, middle separator, and closing question are used to
        frame the comparison, while conversation messages become actual user/assistant
        turns.

        Key invariant: never produce two consecutive messages with the same role.
        When a transition would cause consecutive same-role messages, the new
        content is appended to the last message instead.

        Returns a dict with 'type': 'conversation' and 'messages': list of message dicts.
        """
        preamble, middle, closing = self._parse_template_parts(template)
        messages = []

        def _append_msg(role: str, content: str):
            """Append a message, merging with the previous message if roles match."""
            if messages and messages[-1]['role'] == role:
                messages[-1]['content'] += '\n\n' + content
            else:
                messages.append({"role": role, "content": content})

        def _append_conversation(opt, header: str, is_first=False):
            """Append all turns of a conversation option.

            Args:
                opt: The option dict containing 'messages'.
                header: The header to prepend to the first message of this
                    conversation (e.g. the preamble + "Experience A:" or just
                    the middle part "Experience B:").
                is_first: If True, prepend the preamble before the header.
            """
            conv_messages = opt['messages']
            for i, msg in enumerate(conv_messages):
                role = msg['role']
                content = msg['content']
                if i == 0:
                    # Prepend header to the first message
                    _append_msg(role, header + '\n' + content)
                else:
                    _append_msg(role, content)

        if opt1_conversation and opt2_conversation:
            # Both multi-turn
            _append_conversation(opt1, preamble, is_first=True)
            _append_conversation(opt2, middle)
            if closing:
                _append_msg("user", closing)

        elif opt1_conversation and not opt2_conversation:
            # A is multi-turn, B is text
            _append_conversation(opt1, preamble, is_first=True)
            b_text = f"{middle}\n{opt2['description']}"
            if closing:
                b_text += f"\n\n{closing}"
            _append_msg("user", b_text)

        elif not opt1_conversation and opt2_conversation:
            # A is text, B is multi-turn
            a_text = f"{preamble}\n{opt1['description']}"
            _append_msg("user", a_text)
            _append_conversation(opt2, middle)
            if closing:
                _append_msg("user", closing)

        return {
            'type': 'conversation',
            'messages': messages
        }

    def _generate_conversation_with_image_prompt(self, opt1, opt2, opt1_conversation, opt2_conversation, template):
        """
        Generate a multi-turn prompt when at least one option is a conversation that
        contains images (conversation_with_image type).

        Like _generate_conversation_prompt, but handles Union[str, List] content blocks
        so that image entries ({"type": "image", "image_path": ...}) are preserved in the
        messages passed to the model.

        For non-conversation options with images (text_with_image), images are inlined into
        a user message as content blocks.

        Returns a dict with 'type': 'conversation' and 'messages': list of message dicts
        whose 'content' may be str or list.
        """
        preamble, middle, closing = self._parse_template_parts(template)
        messages = []

        def _to_list(content):
            """Ensure content is a list of content blocks."""
            if isinstance(content, str):
                return [{"type": "text", "text": content}]
            return list(content)  # shallow copy

        def _append_msg(role: str, content):
            """Append a message, merging with the previous message if roles match.
            Content can be str or list of content blocks."""
            if messages and messages[-1]['role'] == role:
                prev = messages[-1]['content']
                prev_list = _to_list(prev)
                new_list = _to_list(content)
                # Insert separator
                prev_list.append({"type": "text", "text": "\n\n"})
                prev_list.extend(new_list)
                messages[-1]['content'] = prev_list
            else:
                messages.append({"role": role, "content": content})

        def _append_conversation(opt, header: str):
            """Append all turns of a conversation option, prepending header to first turn."""
            conv_messages = opt['messages']
            for i, msg in enumerate(conv_messages):
                role = msg['role']
                content = msg['content']
                if i == 0:
                    # Prepend header to the first message
                    if isinstance(content, list):
                        header_block = [{"type": "text", "text": header + "\n"}]
                        content = header_block + list(content)
                    else:
                        content = header + '\n' + content
                _append_msg(role, content)

        def _text_with_image_content(opt, header: str):
            """Build content list for a text_with_image option."""
            blocks = [{"type": "text", "text": header + "\n"}]
            if opt.get('path'):
                blocks.append({"type": "image", "image_path": opt['path']})
            if opt.get('description'):
                blocks.append({"type": "text", "text": "\n" + opt['description']})
            return blocks

        if opt1_conversation and opt2_conversation:
            # Both multi-turn (at least one has images)
            _append_conversation(opt1, preamble)
            _append_conversation(opt2, middle)
            if closing:
                _append_msg("user", closing)

        elif opt1_conversation and not opt2_conversation:
            # A is multi-turn, B is text or text_with_image
            _append_conversation(opt1, preamble)
            if self._option_needs_multimodal(opt2):
                b_content = _text_with_image_content(opt2, middle)
                if closing:
                    b_content.append({"type": "text", "text": "\n\n" + closing})
                _append_msg("user", b_content)
            else:
                b_text = f"{middle}\n{opt2['description']}"
                if closing:
                    b_text += f"\n\n{closing}"
                _append_msg("user", b_text)

        elif not opt1_conversation and opt2_conversation:
            # A is text or text_with_image, B is multi-turn
            if self._option_needs_multimodal(opt1):
                a_content = _text_with_image_content(opt1, preamble)
                _append_msg("user", a_content)
            else:
                a_text = f"{preamble}\n{opt1['description']}"
                _append_msg("user", a_text)
            _append_conversation(opt2, middle)
            if closing:
                _append_msg("user", closing)

        return {
            'type': 'conversation',
            'messages': messages
        }

    def add_edges(self, preference_data: List[Dict]) -> None:
        """
        Add multiple edges to the graph based on processed preference data.
        
        Args:
            preference_data: List of dictionaries containing:
                - option_A: Option A dictionary
                - option_B: Option B dictionary
                - probability_A: Probability of A being preferred over B
                - aux_data: Dictionary with auxiliary data
        """
        for data in preference_data:
            A_id = data['option_A']['id']
            B_id = data['option_B']['id']
            # Keep original orientation
            edge_index = (A_id, B_id)
            
            edge = PreferenceEdge(
                option_A=data['option_A'],
                option_B=data['option_B'],
                probability_A=data['probability_A'],
                aux_data=data['aux_data']
            )
            
            self.edges[edge_index] = edge
            
            # Update training edges tracking if this was a training edge
            # Note: We need to check both orientations for training pool membership
            if edge_index in self.training_edges_pool:
                self.training_edges_pool.remove(edge_index)
                self.training_edges.add(edge_index)
            elif (B_id, A_id) in self.training_edges_pool:
                self.training_edges_pool.remove((B_id, A_id))
                self.training_edges.add(edge_index)
    
    def sample_regular_graph(self, degree: int, seed: int = None) -> List[Tuple[Any, Any]]:
        """
        Sample edge indices forming a regular graph of given degree from training edges pool.
        
        Args:
            degree: Desired degree for each node
            seed: Random seed for reproducibility
            
        Returns:
            List of (option_A_id, option_B_id) tuples
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        n_options = len(self.options)
        if degree >= n_options:
            raise ValueError("Degree must be less than the number of options.")
            
        # Generate regular graph using indices
        G = nx.random_regular_graph(degree, n_options, seed=seed)
        initial_pairs = []
        
        # Map node indices to option IDs
        idx_to_id = {idx: opt['id'] for idx, opt in enumerate(self.options)}
        
        # Convert edges to option ID pairs and filter out holdout edges
        for i, j in G.edges():
            edge = tuple(sorted([str(idx_to_id[i]), str(idx_to_id[j])]))
            if edge in self.training_edges_pool:
                initial_pairs.append(edge)
                
        # If we lost too many edges due to holdout filtering, sample additional edges
        target_edges = (n_options * degree) // 2
        if len(initial_pairs) < target_edges:
            remaining_edges = list(self.training_edges_pool - set(initial_pairs))
            n_additional = min(target_edges - len(initial_pairs), len(remaining_edges))
            if n_additional > 0:
                initial_pairs.extend(random.sample(remaining_edges, n_additional))
                
        return initial_pairs
    
    def sample_random_edges(self, n_edges: int, seed: int = None) -> List[Tuple[Any, Any]]:
        """
        Sample random edge indices from training edges pool.
        
        Args:
            n_edges: Number of edges to sample
            seed: Random seed for reproducibility
            
        Returns:
            List of (option_A_id, option_B_id) tuples
        """
        if seed is not None:
            random.seed(seed)
            
        edges_list = list(self.training_edges_pool)
        n_edges = min(n_edges, len(edges_list))
        return random.sample(edges_list, n_edges)


async def compute_utilities(
    options_list: List[Dict[str, str]],
    model_key: Optional[str] = None,
    create_agent_config_path: Optional[str] = None,
    create_agent_config_key: Optional[str] = None,
    agent: Optional[Any] = None,
    compute_utilities_config_path: Optional[str] = None,
    compute_utilities_config_key: Optional[str] = None,
    system_message: Optional[str] = None,
    comparison_prompt_template: Optional[str] = None,
    conversation: Optional[List[Dict[str, str]]] = None,
    with_reasoning: Optional[bool] = None,
    save_dir: str = None,
    save_suffix: Optional[str] = None,
    target_option: Optional[str] = None,
    seed: Optional[int] = None,
    edit_dict: Optional[dict] = None,
    a_b_logits_only: bool = False,
    use_logprobs: bool = False,
    preference_cot_suffix: Optional[str] = None,
    preference_structured_json_schema: Optional[Dict[str, Any]] = None,
    image_manifest_path: Optional[str] = None,
    audio_manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute utilities for a set of options using a specified utility model.

    Args:
        options_list: List of options or dictionary of option lists
        model_key: Key of the model in models.yaml
        create_agent_config_path: Path to create_agent.yaml
        create_agent_config_key: Key to use in create_agent.yaml
        agent: Pre-initialized agent (if provided, model_key and configs are ignored)
        compute_utilities_config_path: Path to compute_utilities.yaml
        compute_utilities_config_key: Key to use in compute_utilities.yaml
        system_message: Optional system message for the agent. If provided, overrides the value in compute_utilities.yaml
        comparison_prompt_template: Optional template for comparison prompts. If provided, overrides the value in compute_utilities.yaml
        conversation: Optional conversation for the agent. If provided, overrides the value in compute_utilities.yaml
        with_reasoning: Whether to use reasoning-based response parsing. If provided (True/False), overrides the config value
        save_dir: Directory to save results
        save_suffix: Suffix for saved files
        image_manifest_path: Optional path to image manifest JSON for resolving <!image:HASH!> tags
        audio_manifest_path: Optional path to audio manifest JSON for resolving <!audio:HASH!> tags

    Returns:
        Dictionary containing results data
    """
    # Load compute utilities config
    compute_utilities_config = load_config(compute_utilities_config_path, compute_utilities_config_key, "compute_utilities.yaml")

    # Get compute utilities arguments from config
    compute_utilities_arguments = compute_utilities_config.get('compute_utilities_arguments', {})

    # Override config values with provided arguments if they exist; if not provided anywhere, use default values
    if system_message is not None:
        compute_utilities_arguments['system_message'] = system_message
    elif compute_utilities_arguments.get('system_message') is None:
        compute_utilities_arguments['system_message'] = None#"You are a helpful assistant."  # default
    
    if with_reasoning is not None:
        compute_utilities_arguments['with_reasoning'] = with_reasoning
    elif compute_utilities_arguments.get('with_reasoning') is None:
        compute_utilities_arguments['with_reasoning'] = False  # default

    if comparison_prompt_template is not None:
        compute_utilities_arguments['comparison_prompt_template'] = comparison_prompt_template
    elif compute_utilities_arguments.get('comparison_prompt_template') is None:
        # Check utility_model_arguments for template before falling back to defaults
        utility_model_arguments_check = compute_utilities_config.get('utility_model_arguments', {})
        if utility_model_arguments_check.get('comparison_prompt_template') is not None:
            compute_utilities_arguments['comparison_prompt_template'] = utility_model_arguments_check['comparison_prompt_template']
        else:
            wr = compute_utilities_arguments['with_reasoning']
            default_template = comparison_prompt_template_reasoning_default if wr else comparison_prompt_template_default
            compute_utilities_arguments['comparison_prompt_template'] = default_template

    # Update the main config with the potentially modified arguments
    compute_utilities_config['compute_utilities_arguments'] = compute_utilities_arguments

    # Create agent if not provided
    if agent is None:
        if model_key is None:
            raise ValueError("Either agent or model_key must be provided")
            
        # Load create agent config and unpack into create_agent() kwargs
        create_agent_config = load_config(create_agent_config_path, create_agent_config_key or "default", "create_agent.yaml")
        agent = create_agent(model_key=model_key, **create_agent_config)
        print(f"Agent created: {agent}")
        
    # Resolve image tags if manifest path is provided
    if image_manifest_path is not None:
        options_list = resolve_image_tags(options_list, image_manifest_path)

    # Resolve audio tags if manifest path is provided
    if audio_manifest_path is not None:
        options_list = resolve_audio_tags(options_list, audio_manifest_path)

    # Process options - handle text options (strings), image options, audio options, and combinations
    if isinstance(options_list, dict):
        options_list = flatten_hierarchical_options(options_list)

    options = []
    option_str_to_id = {}
    for idx, item in enumerate(options_list):
        if isinstance(item, dict):
            # Dict option - could be image, combination, or pre-processed option
            opt_type = item.get('type', '')

            if opt_type == 'image':
                # Single image option
                opt = {
                    'id': item.get('id', idx),
                    'type': 'image',
                    'path': item['path'],
                    'name': item.get('name', f'image_{idx}'),
                    'description': item.get('description', f"[IMAGE:{item.get('name', idx)}]"),
                    'image_type': item.get('image_type', 'unknown'),
                }
            elif opt_type == 'image_quantity':
                # Image quantity option - tracks quantity of seeing a specific image
                opt = {
                    'id': item.get('id', idx),
                    'type': 'image_quantity',
                    'path': item['path'],
                    'description': item['description'],
                    'quantity': item.get('quantity', 1),
                    'image_name': item.get('image_name', f'image_{idx}'),
                }
            elif opt_type == 'combination_with_images':
                # Combination containing images - preserve all fields
                opt = {
                    'id': item.get('id', idx),
                    'type': 'combination_with_images',
                    'description': item['description'],
                    'is_combination': item.get('is_combination', True),
                    'component_ids': item.get('component_ids', []),
                    'has_images': item.get('has_images', True),
                    'image_paths': item.get('image_paths', []),
                    'components': item.get('components', []),
                }
            elif opt_type == 'text_with_image':
                # Text option with an embedded image
                opt = {
                    'id': item.get('id', idx),
                    'type': 'text_with_image',
                    'path': item['path'],
                    'description': item.get('description', ''),
                }
            elif opt_type == 'audio':
                # Single audio option
                opt = {
                    'id': item.get('id', idx),
                    'type': 'audio',
                    'audio_path': item['audio_path'],
                    'name': item.get('name', f'audio_{idx}'),
                    'description': item.get('description', f"[AUDIO:{item.get('name', idx)}]"),
                }
            elif opt_type == 'text_with_audio':
                # Text option with embedded audio
                opt = {
                    'id': item.get('id', idx),
                    'type': 'text_with_audio',
                    'audio_path': item['audio_path'],
                    'description': item.get('description', ''),
                }

            elif opt_type == 'conversation':
                # Conversation option - multi-turn messages
                opt = {
                    'id': item.get('id', idx),
                    'type': 'conversation',
                    'messages': item['messages'],
                    'description': item.get('description', f'conversation_{idx}'),
                }
                # Preserve option_type metadata (e.g. "neutral_conversation" for ZP)
                if 'option_type' in item:
                    opt['option_type'] = item['option_type']
                # Preserve combination metadata if present
                if item.get('is_combination'):
                    opt['is_combination'] = True
                    opt['component_ids'] = item.get('component_ids', [])
                    opt['component_indices'] = item.get('component_indices', [])
                    opt['size'] = item.get('size', len(opt['component_ids']))
            elif opt_type == 'conversation_with_image':
                # Conversation option with images embedded in message content blocks
                opt = {
                    'id': item.get('id', idx),
                    'type': 'conversation_with_image',
                    'messages': item['messages'],
                    'description': item.get('description', f'conversation_with_image_{idx}'),
                }
                # Preserve image metadata
                if 'source_image' in item:
                    opt['source_image'] = item['source_image']
                if 'image_path' in item:
                    opt['image_path'] = item['image_path']
                # Preserve augmentation metadata
                if item.get('augmented'):
                    opt['augmented'] = True
                if 'baseline_id' in item:
                    opt['baseline_id'] = item['baseline_id']
                # Preserve option_type metadata
                if 'option_type' in item:
                    opt['option_type'] = item['option_type']
                # Preserve combination metadata if present
                if item.get('is_combination') or item.get('component_ids'):
                    opt['is_combination'] = True
                    opt['component_ids'] = item.get('component_ids', [])
                    opt['component_indices'] = item.get('component_indices', [])
                    opt['size'] = item.get('combo_size', item.get('size', len(opt['component_ids'])))
            elif 'id' in item and 'description' in item:
                # Pre-processed option dict (e.g., text singleton or text-only combo)
                opt = item.copy()
                if 'id' not in opt:
                    opt['id'] = idx
            else:
                # Unknown dict format - try to use description
                opt = {'id': idx, 'description': str(item.get('description', item))}

            # Preserve combination metadata from source item if present
            # (needed for ComboZP to resolve component utilities)
            if (item.get('is_combination') or item.get('component_ids')) and 'is_combination' not in opt:
                opt['is_combination'] = True
                opt['component_ids'] = item.get('component_ids', [])
                opt['component_indices'] = item.get('component_indices', [])
                opt['size'] = item.get('combo_size', item.get('size', len(opt.get('component_ids', []))))

            # Preserve option_type from source item if not already set
            # (needed for Neutral ZP auto-detection by compute_zero_point)
            if 'option_type' in item and 'option_type' not in opt:
                opt['option_type'] = item['option_type']

            options.append(opt)
            option_str_to_id[opt['description']] = opt['id']
        else:
            # Text option - string
            opt = {'id': idx, 'description': item}
            options.append(opt)
            option_str_to_id[item] = idx

    # Get utility model class
    utility_model_class_name = compute_utilities_config.get('utility_model_class', 'ThurstonianActiveLearningUtilityModel')
    utility_model_classes = {
        'ThurstonianUtilityModel': ThurstonianUtilityModel,
        'ThurstonianActiveLearningUtilityModel': ThurstonianActiveLearningUtilityModel,
    }
    
    if utility_model_class_name not in utility_model_classes:
        raise ValueError(
            f"Unknown utility model class: {utility_model_class_name}. "
            f"Must be one of: {', '.join(utility_model_classes.keys())}"
        )
        
    utility_model_class = utility_model_classes[utility_model_class_name]

    # Get utility model arguments from config and merge with compute utilities arguments
    utility_model_arguments = compute_utilities_config.get('utility_model_arguments', {})
    
    # Required arguments from compute_utilities_arguments
    required_args = {
        'unparseable_mode': compute_utilities_arguments.get('unparseable_mode', 'skip'),
        'comparison_prompt_template': compute_utilities_arguments['comparison_prompt_template'],
        'system_message': compute_utilities_arguments['system_message'],
        'with_reasoning': compute_utilities_arguments['with_reasoning']
    }
    
    # Merge required args with model-specific args, giving precedence to model-specific args
    all_model_args = {**required_args, **utility_model_arguments}
    all_model_args['system_message'] = compute_utilities_arguments['system_message']
    all_model_args['comparison_prompt_template'] = compute_utilities_arguments['comparison_prompt_template']
    all_model_args['conversation'] = conversation
    all_model_args['target_option'] = target_option
    all_model_args['seed'] = seed
    # Initialize the utility model with all arguments
    utility_model = utility_model_class(**all_model_args)
    
    # Get preference graph arguments from config
    preference_graph_arguments = compute_utilities_config.get('preference_graph_arguments', {})
    graph = PreferenceGraph(
        options=options,
        holdout_fraction=preference_graph_arguments.get('holdout_fraction', 0.0),
        seed=preference_graph_arguments.get('holdout_seed', 42)
    )
    target_edits = {}
    prev_utilities = None
    if edit_dict is not None:
        target_edits = {option_str_to_id[edit['target_action']]: edit['target_utility'] for edit in edit_dict['edits']['edits']}
        prev_utilities = edit_dict['utilities']

    # Fit the model (this will populate training edges)
    # Attach preference structured decoding config to the agent/model via conversation suffix if provided
    # We pass the suffix inline by modifying the prompt template at the call sites inside the model fit using the provided suffix
    # Determine use_logprobs: explicit argument takes precedence, otherwise check utility model
    _use_logprobs = use_logprobs or getattr(utility_model, 'use_logprobs', False)

    fit_kwargs = dict(
        graph=graph,
        agent=agent,
        edits=target_edits,
        prev_utilities=prev_utilities,
        a_b_logits_only=a_b_logits_only,
        preference_cot_suffix=preference_cot_suffix,
        preference_structured_json_schema=preference_structured_json_schema,
    )
    # Pass save_dir for checkpointing if the model supports it
    import inspect
    if 'save_dir' in inspect.signature(utility_model.fit).parameters:
        fit_kwargs['save_dir'] = save_dir
    utilities, metrics = await utility_model.fit(**fit_kwargs)
    holdout_metrics = await evaluate_holdout_set(
        graph=graph,
        agent=agent,
        utility_model=utility_model,
        utilities=utilities,
        comparison_prompt_template=compute_utilities_arguments['comparison_prompt_template'],
        system_message=compute_utilities_arguments['system_message'],
        with_reasoning=compute_utilities_arguments['with_reasoning'],
        K=utility_model.K if hasattr(utility_model, 'K') else compute_utilities_arguments.get('K', 10),
        a_b_logits_only=a_b_logits_only,
        use_logprobs=_use_logprobs,
    )
    
    # Prepare results
    results = {
        'options': options,
        'utilities': utilities,
        'metrics': metrics,  # Training metrics
        'holdout_metrics': holdout_metrics,  # Holdout metrics (if computed)
        'compute_utilities_config': compute_utilities_config,
        'graph_data': graph.export_data(),  # Raw preference graph data
        'target_edits': target_edits
    }
    
    # Save results if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Determine save suffix
        if save_suffix is None:
            save_suffix = f"{model_key}_{utility_model_class_name.lower()}"
            
        # Convert NumPy types to native Python types before saving
        results_to_save = convert_numpy(results)
            
        # Save the full results JSON
        results_path = os.path.join(save_dir, f"results_{save_suffix}.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)

        # Save a separate utilities-only JSON (without raw preference graph data)
        results_utilities_path = os.path.join(save_dir, f"results_utilities_{save_suffix}.json")
        results_utilities_to_save = {k: v for k, v in results_to_save.items() if k != 'graph_data'}
        with open(results_utilities_path, 'w', encoding='utf-8') as f:
            json.dump(results_utilities_to_save, f, indent=2, ensure_ascii=False)

        # Save a short summary txt
        summary_path = os.path.join(save_dir, f"summary_{save_suffix}.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Utility Model: {utility_model_class_name}\n\n")
            f.write("Training Metrics:\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
            if holdout_metrics:
                f.write("\nHoldout Metrics:\n")
                for k, v in holdout_metrics.items():
                    f.write(f"{k}: {v}\n")
            f.write("\nSorted utilities:\n")
            sorted_utils = sorted(
                [(opt['description'], utilities[opt['id']]) for opt in options],
                key=lambda x: x[1]['mean'],
                reverse=True
            )
            for desc, util in sorted_utils:
                f.write(f"{desc}: mean={util['mean']:.4f}, variance={util['variance']:.4f}\n")
                
    # Explicitly clean up the vLLM engine to free GPU memory, so subsequent
    # steps (e.g., yes/no ZP inference) can load their own engine.
    if agent is not None and hasattr(agent, 'llm'):
        try:
            del agent.llm
        except Exception:
            pass
    del agent
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return results
