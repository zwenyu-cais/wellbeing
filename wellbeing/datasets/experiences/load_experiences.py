"""Unified experience loader for the wellbeing experiments framework.

Loads experiences from all sources (short_text, intensity_scaled, stories,
book_chapters, conversations, images) into a common Experience dataclass.

Usage:
    from datasets.experiences.load_experiences import load_all, load_by_format, load_by_category

    # Load everything
    experiences = load_all()

    # Load by format
    short = load_by_format("short_text")
    stories = load_by_format("story")
    convos = load_by_format("conversation")

    # Load by category
    praise = load_by_category("praise")
    grief = load_by_category("grief")

    # Load specific file
    from datasets.experiences.load_experiences import load_file
    items = load_file("short_text/praise.json")
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_BANK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "component_datasets")


@dataclass
class Experience:
    """Common interface for all experience types.

    Attributes:
        id: Globally unique identifier "{source}_{idx}"
        text: For passive experiences (short_text, stories, chapters)
        messages: For interactive experiences (conversations), list of {role, content} dicts
        source: Source file relative to datasets/experiences/component_datasets/ (e.g. "short_text/praise")
        category: Semantic category (e.g. "praise", "grief", "offensive")
        valence: "positive" | "negative" | "neutral" | "mixed"
        intensity: Normalized -1.0 to 1.0 (None if not rated)
        domain: "social" | "task" | "existential" | "moral" | "aesthetic" | "neutral"
        format: "short_text" | "story" | "book_chapter" | "conversation" | "image"
        is_pairwise_compatible: Can be used in pairwise preference comparisons
        metadata: Source-specific extras (word_count, scenario, etc.)
    """
    id: str
    text: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    source: str = ""
    category: str = ""
    valence: str = "neutral"
    intensity: Optional[float] = None
    domain: str = "neutral"
    format: str = "short_text"
    is_pairwise_compatible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


def _load_json(rel_path: str) -> Any:
    """Load a JSON file relative to datasets/experiences/component_datasets/."""
    path = os.path.join(_BANK_DIR, rel_path)
    with open(path, "r") as f:
        return json.load(f)


def _source_name(rel_path: str) -> str:
    """Convert file path to source name (strip .json extension)."""
    return rel_path.replace(".json", "")


def load_short_text(filename: str) -> List[Experience]:
    """Load a short_text JSON file.

    Expected format: list of objects with fields:
        text (str), category (str), valence (str), intensity (float|null),
        domain (str), metadata (dict, optional)
    """
    rel_path = f"short_text/{filename}"
    data = _load_json(rel_path)
    source = _source_name(rel_path)
    experiences = []
    for i, item in enumerate(data):
        exp = Experience(
            id=f"{source}_{i:03d}",
            text=item["text"],
            source=source,
            category=item.get("category", os.path.splitext(filename)[0]),
            valence=item.get("valence", "neutral"),
            intensity=item.get("intensity"),
            domain=item.get("domain", "neutral"),
            format="short_text",
            is_pairwise_compatible=True,
            metadata=item.get("metadata", {}),
        )
        experiences.append(exp)
    return experiences


def load_intensity_scaled(filename: str) -> List[Experience]:
    """Load an intensity_scaled JSON file.

    Expected format: object with "experiences" list, each having:
        text (str), category (str), intensity (float), idx (int)
    """
    rel_path = f"intensity_scaled/{filename}"
    data = _load_json(rel_path)
    source = _source_name(rel_path)

    items = data.get("experiences", data) if isinstance(data, dict) else data
    experiences = []
    for i, item in enumerate(items):
        intensity = item.get("intensity", 0.0)
        if intensity > 0:
            valence = "positive"
        elif intensity < 0:
            valence = "negative"
        else:
            valence = "neutral"

        exp = Experience(
            id=f"{source}_{i:03d}",
            text=item["text"],
            source=source,
            category=item.get("category", os.path.splitext(filename)[0]),
            valence=valence,
            intensity=intensity,
            domain=item.get("domain", "social"),
            format="short_text",
            is_pairwise_compatible=True,
            metadata={k: v for k, v in item.items() if k not in ("text", "category", "intensity")},
        )
        experiences.append(exp)
    return experiences


def load_stories(filename: str) -> List[Experience]:
    """Load a stories JSON file.

    Expected format: object with "stories" list (or plain list), each having:
        text (str), title (str), valence (str), word_count (int), etc.
    """
    rel_path = f"stories/{filename}"
    data = _load_json(rel_path)
    source = _source_name(rel_path)

    items = data.get("stories", data) if isinstance(data, dict) else data
    experiences = []
    for i, item in enumerate(items):
        exp = Experience(
            id=item.get("id", f"{source}_{i:03d}"),
            text=item["text"],
            source=source,
            category=item.get("category", "story"),
            valence=item.get("valence", "neutral"),
            intensity=item.get("intensity"),
            domain=item.get("domain", "aesthetic"),
            format="story",
            is_pairwise_compatible=False,  # Too long for pairwise bundles
            metadata={
                k: v for k, v in item.items()
                if k not in ("text", "category", "valence", "intensity", "domain", "id")
            },
        )
        experiences.append(exp)
    return experiences


def load_book_chapters(filename: str) -> List[Experience]:
    """Load a book_chapters JSON file.

    Expected format: list of objects with text, title, author, valence, word_count, etc.
    """
    rel_path = f"book_chapters/{filename}"
    data = _load_json(rel_path)
    source = _source_name(rel_path)

    items = data if isinstance(data, list) else data.get("chapters", [])
    experiences = []
    for i, item in enumerate(items):
        exp = Experience(
            id=item.get("id", f"{source}_{i:03d}"),
            text=item["text"],
            source=source,
            category=item.get("category", "book_chapter"),
            valence=item.get("valence", "neutral"),
            intensity=item.get("intensity"),
            domain=item.get("domain", "aesthetic"),
            format="book_chapter",
            is_pairwise_compatible=False,
            metadata={
                k: v for k, v in item.items()
                if k not in ("text", "category", "valence", "intensity", "domain", "id")
            },
        )
        experiences.append(exp)
    return experiences


def load_conversations(filename: str) -> List[Experience]:
    """Load a conversations JSON file.

    Expected format: list of objects with:
        messages (list of {role, content}), scenario (str), category (str),
        valence (str), domain (str), metadata (dict, optional)
    """
    rel_path = f"conversations/{filename}"
    data = _load_json(rel_path)
    source = _source_name(rel_path)

    items = data if isinstance(data, list) else data.get("conversations", [])
    experiences = []
    for i, item in enumerate(items):
        exp = Experience(
            id=item.get("id", f"{source}_{i:03d}"),
            text=None,
            messages=item["messages"],
            source=source,
            category=item.get("category", os.path.splitext(filename)[0]),
            valence=item.get("valence", "neutral"),
            intensity=item.get("intensity"),
            domain=item.get("domain", "social"),
            format="conversation",
            is_pairwise_compatible=False,
            metadata={
                k: v for k, v in item.items()
                if k not in ("messages", "category", "valence", "intensity", "domain", "id")
            },
        )
        experiences.append(exp)
    return experiences


def load_images(filename: str = "superstimuli_manifest.json") -> List[Experience]:
    """Load image experiences from manifest.

    Expected format: list of objects with:
        image_path (str), category (str), valence (str), etc.
    """
    rel_path = f"images/{filename}"
    data = _load_json(rel_path)
    source = _source_name(rel_path)

    items = data if isinstance(data, list) else data.get("images", [])
    experiences = []
    for i, item in enumerate(items):
        exp = Experience(
            id=item.get("id", f"{source}_{i:03d}"),
            text=None,
            source=source,
            category=item.get("category", "image"),
            valence=item.get("valence", "neutral"),
            intensity=item.get("intensity"),
            domain=item.get("domain", "aesthetic"),
            format="image",
            is_pairwise_compatible=False,
            metadata={"image_path": item["image_path"], **{
                k: v for k, v in item.items()
                if k not in ("image_path", "category", "valence", "intensity", "domain", "id")
            }},
        )
        experiences.append(exp)
    return experiences


def load_file(rel_path: str) -> List[Experience]:
    """Load experiences from a specific file (path relative to datasets/experiences/component_datasets/).

    Dispatches to the appropriate loader based on directory.
    """
    parts = rel_path.split("/")
    directory = parts[0]
    filename = "/".join(parts[1:])

    loaders = {
        "short_text": load_short_text,
        "intensity_scaled": load_intensity_scaled,
        "stories": load_stories,
        "book_chapters": load_book_chapters,
        "conversations": load_conversations,
        "images": load_images,
    }

    if directory not in loaders:
        raise ValueError(f"Unknown experience directory: {directory}")

    return loaders[directory](filename)


def _list_json_files(subdir: str) -> List[str]:
    """List all .json files in a subdirectory of datasets/experiences/component_datasets/."""
    dirpath = os.path.join(_BANK_DIR, subdir)
    if not os.path.isdir(dirpath):
        return []
    return sorted(f for f in os.listdir(dirpath) if f.endswith(".json"))


def load_by_format(fmt: str) -> List[Experience]:
    """Load all experiences of a given format.

    Args:
        fmt: "short_text", "intensity_scaled", "story", "book_chapter", "conversation", "image"
    """
    dir_map = {
        "short_text": ("short_text", load_short_text),
        "intensity_scaled": ("intensity_scaled", load_intensity_scaled),
        "story": ("stories", load_stories),
        "book_chapter": ("book_chapters", load_book_chapters),
        "conversation": ("conversations", load_conversations),
        "image": ("images", load_images),
    }

    if fmt not in dir_map:
        raise ValueError(f"Unknown format: {fmt}. Available: {list(dir_map.keys())}")

    subdir, loader = dir_map[fmt]
    all_experiences = []
    for filename in _list_json_files(subdir):
        try:
            all_experiences.extend(loader(filename))
        except Exception as e:
            print(f"Warning: Failed to load {subdir}/{filename}: {e}")
    return all_experiences


def load_by_category(category: str) -> List[Experience]:
    """Load all experiences matching a category name."""
    all_exp = load_all()
    return [e for e in all_exp if e.category == category]


def load_by_valence(valence: str) -> List[Experience]:
    """Load all experiences matching a valence (positive/negative/neutral/mixed)."""
    all_exp = load_all()
    return [e for e in all_exp if e.valence == valence]


def load_pairwise_compatible() -> List[Experience]:
    """Load only experiences suitable for pairwise preference comparisons."""
    all_exp = load_all()
    return [e for e in all_exp if e.is_pairwise_compatible]


def load_all() -> List[Experience]:
    """Load all experiences from all sources."""
    all_experiences = []
    for fmt in ["short_text", "intensity_scaled", "story", "book_chapter", "conversation", "image"]:
        all_experiences.extend(load_by_format(fmt))
    return all_experiences
