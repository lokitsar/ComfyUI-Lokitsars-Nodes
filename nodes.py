import io
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from aiohttp import web
from server import PromptServer
import folder_paths

try:
    from .training_helpers import DatasetSidecarWriter, scale_for_provider
except ImportError:
    # The local unit tests import nodes.py as a top-level module.
    from training_helpers import DatasetSidecarWriter, scale_for_provider


GALLERY_STATE: Dict[str, Dict[str, Any]] = {}
ENTRY_INDEX: Dict[str, Dict[str, Any]] = {}
STATE_LOCK = threading.Lock()


PACKAGE_DIR = Path(__file__).resolve().parent
# Use ComfyUI's built-in folder_paths module — works for all install types
# (manual, portable, desktop app, etc.)
DEFAULT_SAVE_DIR = Path(folder_paths.get_output_directory()) / "workflow_gallery"
LEGACY_SAVE_DIR = PACKAGE_DIR / "gallery_output"
CACHE_BASE_DIR = DEFAULT_SAVE_DIR / "Workflow-Gallery"
DEFAULT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_BASE_DIR.mkdir(parents=True, exist_ok=True)


ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

SESSION_DIR = CACHE_BASE_DIR / "sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_LIBRARY_FILE = CACHE_BASE_DIR / "prompt_library.json"
PROMPT_LIBRARY_LOCK = threading.Lock()
PROMPT_LIBRARY_OUTPUTS: Dict[str, str] = {}
# Runtime outputs of PromptEnhancer nodes, keyed by node_id. The enhanced text
# only exists at runtime — it is never in the prompt graph — so the gallery's
# prompt resolver needs this to show the ACTUAL final prompt fed downstream.
PROMPT_ENHANCER_OUTPUTS: Dict[str, str] = {}

# ── Wildcard expansion ────────────────────────────────────────────────────────
import random
import re as _re

# ── Wildcard directory setup ─────────────────────────────────────────────────
# NODE_WILDCARDS_DIR: guaranteed default inside the package — always exists.
# Users drop .txt or .yaml wildcard files here.
# wildcard_path widget is ADDITIVE — any extra paths are searched in addition.
NODE_WILDCARDS_DIR = PACKAGE_DIR / "wildcards"
NODE_WILDCARDS_DIR.mkdir(parents=True, exist_ok=True)
print(f"[PromptLibrary] Default wildcard dir: {NODE_WILDCARDS_DIR}", flush=True)

# Auto-detect common extra locations (additive, not replacing default)
_EXTRA_AUTO_DIRS: List[Path] = []
_comfy_root = Path(folder_paths.get_output_directory()).parent
for _candidate in [
    _comfy_root / "wildcards",
    _comfy_root / "custom_nodes" / "ComfyUI-Impact-Pack" / "wildcards",
]:
    if _candidate.exists() and _candidate.is_dir() and _candidate != NODE_WILDCARDS_DIR:
        _EXTRA_AUTO_DIRS.append(_candidate)
        print(f"[PromptLibrary] Also found wildcard dir: {_candidate}", flush=True)


def _resolve_wildcard_dirs(extra_path: str) -> List[Path]:
    """Build the wildcard search list.
    Always starts with NODE_WILDCARDS_DIR, then auto-detected extras,
    then any user-specified extra paths (comma-separated).
    """
    dirs: List[Path] = [NODE_WILDCARDS_DIR]
    for d in _EXTRA_AUTO_DIRS:
        if d not in dirs:
            dirs.append(d)
    if extra_path and extra_path.strip():
        for part in extra_path.split(","):
            p = Path(part.strip())
            if p.exists() and p.is_dir():
                if p not in dirs:
                    dirs.append(p)
                    print(f"[PromptLibrary] Added extra wildcard dir: {p}", flush=True)
            else:
                print(f"[PromptLibrary] Extra wildcard path not found: {p!r}", flush=True)
    print(f"[PromptLibrary] Wildcard search dirs: {[str(d) for d in dirs]}", flush=True)
    return dirs


def _read_wildcard_lines_txt(path: Path) -> List[str]:
    """Read lines from a .txt wildcard file."""
    return [l.strip() for l in path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if l.strip() and not l.strip().startswith("#")]


# Cache parsed YAML files so we only pay the parse cost once per session.
_YAML_DATA_CACHE: Dict[str, tuple[tuple[int, int], Any]] = {}
_YAML_LINES_CACHE: Dict[str, tuple[tuple[int, int], List[str]]] = {}
_WILDCARD_LOOKUP_CACHE: Dict[tuple[str, tuple[str, ...]], tuple[Path | None, List[str]]] = {}


def _file_signature(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
        return (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (0, 0)


def _load_wildcard_yaml(path: Path) -> Any:
    """Load a YAML wildcard file once, invalidating the cache when it changes."""
    cache_key = str(path)
    signature = _file_signature(path)
    cached = _YAML_DATA_CACHE.get(cache_key)
    if cached and cached[0] == signature:
        return cached[1]

    try:
        import yaml  # type: ignore
        data = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore"))
    except ImportError:
        # PyYAML not available - minimal line parser for simple list format.
        data = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                data.append(stripped[2:].strip())
    except Exception as e:
        print(f"[PromptLibrary] YAML parse error in {path}: {e}", flush=True)
        data = None

    _YAML_DATA_CACHE[cache_key] = (signature, data)
    return data

def _read_wildcard_lines_yaml(path: Path, name_parts: List[str]) -> List[str]:
    """Read lines from a .yaml wildcard file.

    Parsed YAML is cached by file timestamp/size. Avoid using a ThreadPoolExecutor
    timeout here: if PyYAML or the filesystem stalls, executor shutdown waits for
    the worker anyway, which freezes the Prompt Library node.
    """
    cache_key = f"{path}|{'/'.join(name_parts)}"
    signature = _file_signature(path)
    cached = _YAML_LINES_CACHE.get(cache_key)
    if cached and cached[0] == signature:
        return cached[1]

    data = _load_wildcard_yaml(path)

    # Navigate nested keys: __colors/dark__ -> colors.yaml key "dark"
    for key in name_parts:
        if isinstance(data, dict) and key in data:
            data = data[key]
        elif isinstance(data, dict):
            key_lower = key.lower()
            match = next((v for k, v in data.items() if str(k).lower() == key_lower), None)
            if match is not None:
                data = match
            else:
                break

    # Flatten to list of strings
    if isinstance(data, list):
        lines = [str(item).strip() for item in data if item]
    elif isinstance(data, dict):
        result = []
        for v in data.values():
            if isinstance(v, list):
                result.extend(str(i).strip() for i in v if i)
            elif v:
                result.append(str(v).strip())
        lines = result
    elif isinstance(data, str):
        lines = [data.strip()]
    else:
        lines = []

    _YAML_LINES_CACHE[cache_key] = (signature, lines)
    return lines


def _safe_wildcard_parts(name: str) -> List[str]:
    parts = []
    for part in name.replace("\\", "/").split("/"):
        part = part.strip()
        if not part or part in {".", ".."}:
            return []
        parts.append(part)
    return parts


def _case_insensitive_child(parent: Path, name: str) -> Path | None:
    candidate = parent / name
    if candidate.exists():
        return candidate

    try:
        name_lower = name.lower()
        for child in parent.iterdir():
            if child.name.lower() == name_lower:
                return child
    except OSError:
        return None
    return None


def _case_insensitive_path(root: Path, relative_parts: List[str], suffix: str) -> Path | None:
    current = root
    for part in relative_parts[:-1]:
        child = _case_insensitive_child(current, part)
        if child is None or not child.is_dir():
            return None
        current = child

    leaf = _case_insensitive_child(current, f"{relative_parts[-1]}{suffix}")
    if leaf is not None and leaf.is_file():
        return leaf
    return None


def _find_wildcard_file_in_dirs(name: str, dirs: List[Path]):
    """Find a wildcard file (.txt, .yaml, .yml) by name within given directories.
    Tries direct paths only, including YAML prefix matches such as
    __colors/dark__ -> colors.yaml key "dark". Avoid recursive scans here:
    walking a large or cloud-backed wildcard tree can freeze ComfyUI."""
    parts = _safe_wildcard_parts(name)
    if not parts:
        return None, []

    cache_key = ("/".join(part.lower() for part in parts), tuple(str(d) for d in dirs))
    cached = _WILDCARD_LOOKUP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    for wdir in dirs:
        # Fast exact match: __foo/bar__ -> foo/bar.txt, foo/bar.yaml, foo/bar.yml
        for ext in (".txt", ".yaml", ".yml"):
            candidate = _case_insensitive_path(wdir, parts, ext)
            if candidate is not None:
                result = (candidate, [])
                _WILDCARD_LOOKUP_CACHE[cache_key] = result
                return result

        # YAML key navigation: __foo/bar/baz__ can map to foo/bar.yaml["baz"]
        # or foo.yaml["bar"]["baz"], without scanning the whole wildcard tree.
        for index in range(len(parts) - 1, 0, -1):
            file_parts = parts[:index]
            key_parts = parts[index:]
            for ext in (".yaml", ".yml"):
                candidate = _case_insensitive_path(wdir, file_parts, ext)
                if candidate is not None:
                    result = (candidate, key_parts)
                    _WILDCARD_LOOKUP_CACHE[cache_key] = result
                    return result

    return None, []


def _expand_inline_choices(text: str, rng: random.Random, depth: int = 0) -> str:
    """Expand {option1|option2|option3} inline choice syntax recursively.
    Supports weighted options via {weight::option|weight::option} syntax."""
    if depth > 10 or "{" not in text:
        return text

    def replace_choice(m: _re.Match) -> str:
        inner = m.group(1)
        options = inner.split("|")
        weights = []
        cleaned = []
        for opt in options:
            if "::" in opt:
                parts = opt.split("::", 1)
                try:
                    weights.append(float(parts[0].strip()))
                    cleaned.append(parts[1].strip())
                except ValueError:
                    weights.append(1.0)
                    cleaned.append(opt.strip())
            else:
                weights.append(1.0)
                cleaned.append(opt.strip())
        return rng.choices(cleaned, weights=weights, k=1)[0]

    prev = None
    result = text
    while prev != result and depth <= 10:
        prev = result
        result = _re.sub(r"\{([^{}]+)\}", replace_choice, result)
        depth += 1
    return result


def _expand_wildcards(text: str, dirs: List[Path], rng: random.Random | None = None, depth: int = 0) -> str:
    """Recursively expand both __file__ wildcards and {choice|choice} inline syntax."""
    if depth > 10:
        return text

    if rng is None:
        rng = random.Random()

    # Expand inline {option1|option2} choices first
    text = _expand_inline_choices(text, rng)

    if "__" not in text:
        return text

    def replace_match(m: _re.Match) -> str:
        wfile, key_parts = _find_wildcard_file_in_dirs(m.group(1), dirs)
        if not wfile:
            return m.group(0)
        try:
            if wfile.suffix.lower() in (".yaml", ".yml"):
                lines = _read_wildcard_lines_yaml(wfile, key_parts)
            else:
                lines = _read_wildcard_lines_txt(wfile)
            if not lines:
                return m.group(0)
            return _expand_wildcards(rng.choice(lines), dirs, rng, depth + 1)
        except Exception as e:
            print(f"[PromptLibrary] Wildcard error for {wfile}: {e}", flush=True)
            return m.group(0)

    return _re.sub(r"__([a-zA-Z0-9_\-/\\]+)__", replace_match, text)


def _safe_int(value: Any, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        result = int(value)
    except Exception:
        result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result



def _sanitize_prefix(prefix: str) -> str:
    cleaned = "".join(ch for ch in prefix if ch.isalnum() or ch in ("-", "_"))
    return cleaned[:80] or "workflow_gallery"


def _resolve_output_dir(raw_path: str) -> Path:
    raw_path = (raw_path or "").strip()
    if not raw_path:
        return DEFAULT_SAVE_DIR
    expanded = Path(os.path.expandvars(os.path.expanduser(raw_path)))
    return expanded if expanded.is_absolute() else (PACKAGE_DIR / expanded).resolve()


def _normalize_output_dir(raw_path: str) -> Path:
    resolved = _resolve_output_dir(raw_path).resolve()
    legacy = LEGACY_SAVE_DIR.resolve()
    if os.path.normcase(str(resolved)) == os.path.normcase(str(legacy)):
        return DEFAULT_SAVE_DIR
    return resolved


def _tensor_to_pil(image_tensor) -> Image.Image:
    arr = image_tensor.cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _thumbnail_bytes(image: Image.Image, max_size: int = 256) -> bytes:
    thumb = image.copy()
    thumb.thumbnail((max_size, max_size), Image.LANCZOS)
    buffer = io.BytesIO()
    thumb.save(buffer, format="WEBP", quality=85, method=6)
    return buffer.getvalue()


def _entry_public(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": entry["id"],
        "filename": entry["filename"],
        "created": entry["created"],
        "width": entry["width"],
        "height": entry["height"],
        "display_prompt": entry.get("display_prompt", entry.get("positive_prompt", "")),
        "prompt_source": entry.get("prompt_source", "unavailable"),
        "positive_prompt": entry.get("positive_prompt", ""),
        "negative_prompt": entry.get("negative_prompt", ""),
        "exported": entry.get("exported", False),
        "seed": entry.get("seed", None),
        "full_url": f"/workflow_gallery/file/{entry['id']}?kind=full",
        "thumb_url": f"/workflow_gallery/file/{entry['id']}?kind=thumb",
    }


def _get_ref_node_id(value: Any) -> str:
    # In the API prompt format a node reference is ALWAYS ["node_id", slot].
    # Bare strings/ints are widget values (seed=512, steps=20, typed text) —
    # treating them as node IDs let the walker jump into unrelated branches
    # whenever a widget int happened to collide with a real node ID.
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    return ""


def _iter_child_node_ids(inputs: Dict[str, Any]) -> List[str]:
    child_ids: List[str] = []
    for child_value in inputs.values():
        child_node_id = _get_ref_node_id(child_value)
        if child_node_id:
            child_ids.append(child_node_id)
    return child_ids


def _is_sampler_node(node: Dict[str, Any]) -> bool:
    """Return True if this node looks like a KSampler or equivalent prompt-bearing node."""
    class_type = str(node.get("class_type", ""))
    inputs = node.get("inputs", {})
    if not isinstance(inputs, dict):
        inputs = {}
    has_sampler_links = (
        ("positive" in inputs)
        or ("negative" in inputs)
        or ("cond_pos" in inputs)
        or ("cond_neg" in inputs)
    )
    # Guider nodes (BasicGuider, CFGGuider, ...) carry the conditioning in
    # SamplerCustomAdvanced pipelines — the sampler itself has no positive/
    # negative inputs there, so the guider is the prompt-bearing node.
    is_guider = "Guider" in class_type and "conditioning" in inputs
    return class_type.startswith("KSampler") or has_sampler_links or is_guider


def _find_relevant_sampler(prompt_graph: Dict[str, Any], gallery_node_id: str | None) -> Dict[str, Any] | None:
    if not gallery_node_id:
        return None

    gallery_node = prompt_graph.get(str(gallery_node_id))
    if not isinstance(gallery_node, dict):
        return None

    gallery_inputs = gallery_node.get("inputs", {})
    if not isinstance(gallery_inputs, dict):
        gallery_inputs = {}

    start_node_id = _get_ref_node_id(gallery_inputs.get("images"))
    if not start_node_id:
        return None

    # BFS upstream from the gallery's image input so we find the *closest*
    # sampler to this specific gallery node, not just any sampler in the graph.
    from collections import deque
    queue: deque[str] = deque([start_node_id])
    visited: set[str] = set()

    while queue:
        node_id = queue.popleft()
        if not node_id or node_id in visited:
            continue
        visited.add(node_id)

        node = prompt_graph.get(node_id)
        if not isinstance(node, dict):
            continue

        if _is_sampler_node(node):
            return node

        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            inputs = {}

        # Prefer latent/image inputs first so we stay on the image pipeline path
        preferred_input_order = ["samples", "latent", "latent_image", "images", "image"]
        ordered_children: List[str] = []
        for key in preferred_input_order:
            child_id = _get_ref_node_id(inputs.get(key))
            if child_id and child_id not in ordered_children:
                ordered_children.append(child_id)
        for child_id in _iter_child_node_ids(inputs):
            if child_id not in ordered_children:
                ordered_children.append(child_id)

        for child_node_id in ordered_children:
            if child_node_id not in visited:
                queue.append(child_node_id)

    return None

def _resolve_text_from_ref(prompt_graph: Dict[str, Any], value: Any, visited: set[str] | None = None, role: str | None = None) -> str:
    # Only real node references ([node_id, slot]) are followed. Bare strings/
    # ints are widget values — following them caused cross-branch text bleed.
    node_ref = _get_ref_node_id(value)

    if not node_ref:
        return ""

    if node_ref not in prompt_graph:
        return ""

    if visited is None:
        visited = set()
    if node_ref in visited:
        return ""
    current_visited = set(visited)
    current_visited.add(node_ref)

    node = prompt_graph.get(node_ref)
    if not isinstance(node, dict):
        return ""

    class_type = str(node.get("class_type", ""))
    inputs = node.get("inputs", {})
    if not isinstance(inputs, dict):
        inputs = {}

    # ZeroOut nodes intentionally nullify conditioning — stop walking here
    # so we don't trace back through them and bleed positive text into negative.
    if "ZeroOut" in class_type or "zero_out" in class_type.lower():
        return ""

    # PromptLibrary node — use runtime output dict first, fall back to widget value
    if class_type == "PromptLibrary":
        if node_ref in PROMPT_LIBRARY_OUTPUTS:
            val = PROMPT_LIBRARY_OUTPUTS[node_ref]
            if val and val.strip():
                return val.strip()
        val = inputs.get("selected_prompt_id", "")
        if isinstance(val, str) and val.strip():
            return val.strip()
        return ""

    # PromptEnhancer node — the enhanced text only exists at runtime, never in
    # the graph. Use the runtime output dict; fall back to following the
    # (pre-enhancement) prompt input so we at least show something meaningful.
    if class_type == "PromptEnhancer":
        if node_ref in PROMPT_ENHANCER_OUTPUTS:
            val = PROMPT_ENHANCER_OUTPUTS[node_ref]
            if val and val.strip():
                return val.strip()
        return _resolve_text_from_ref(prompt_graph, inputs.get("prompt"), current_visited, role)

    if "TextEncode" in class_type:
        text_field_keys = ["text", "prompt", "text_g", "text_l", "clip_l", "clip_g", "t5xxl", "t5xxl_text"]
        parts: List[str] = []
        for key in text_field_keys:
            field_value = inputs.get(key)
            if field_value is None:
                continue
            if isinstance(field_value, str) and field_value.strip():
                # Literal text directly in the field — use it as-is
                parts.append(field_value.strip())
            elif isinstance(field_value, (list, tuple)) and field_value:
                # It's a node reference — follow it upstream to resolve the string.
                # This handles wildcard nodes, string concatenators, primitive nodes, etc.
                resolved = _resolve_text_from_ref(prompt_graph, field_value, current_visited, role)
                if resolved:
                    parts.append(resolved)
        if parts:
            unique_parts = list(dict.fromkeys(parts))
            return "\n".join(unique_parts)
        # TextEncode node had no text — return empty rather than walking
        # non-text children (e.g. clip input) which can bleed positive text into negative.
        return ""

    # For non-TextEncode nodes (e.g. wildcard node, string node, primitive),
    # check common string output fields first before walking all children.
    # populated_text (the EXPANDED wildcard result) must come before
    # wildcard_text (the raw __template__) — the gallery should show the
    # final prompt, not the unexpanded template.
    string_field_keys = ["text", "string", "value", "prompt", "output", "result", "populated_text", "wildcard_text"]
    for key in string_field_keys:
        field_value = inputs.get(key)
        if isinstance(field_value, str) and field_value.strip():
            return field_value.strip()

    # Generic child walk — follow ONLY inputs that can plausibly carry prompt
    # text or conditioning, and never cross into the opposite role's inputs.
    # Walking model/clip/vae/latent children let the resolver wander into
    # unrelated branches and return foreign text; walking `positive` while
    # resolving the negative chain (e.g. through ControlNetApply) is what
    # made positive and negative come back identical.
    _TEXT_LIKE = ("text", "string", "prompt", "value", "wildcard", "populated",
                  "result", "output", "conditioning", "cond")

    def _key_allowed(key_lower: str) -> bool:
        if role == "positive" and ("negative" in key_lower or key_lower == "cond_neg"):
            return False
        if role == "negative" and ("positive" in key_lower or key_lower == "cond_pos"):
            return False
        if role and role in key_lower:
            return True
        return any(t in key_lower for t in _TEXT_LIKE)

    # Same-role keys first (e.g. follow ControlNetApply's `negative` input
    # before its neutral conditioning-style inputs when resolving negative).
    ordered_items = sorted(
        inputs.items(),
        key=lambda kv: 0 if (role and role in str(kv[0]).lower()) else 1,
    )
    for key, child_value in ordered_items:
        if not _key_allowed(str(key).lower()):
            continue
        text = _resolve_text_from_ref(prompt_graph, child_value, current_visited, role)
        if text:
            return text
    return ""


def _extract_prompts(prompt_graph: Any, gallery_node_id: str | None = None) -> tuple[str, str]:
    if not isinstance(prompt_graph, dict):
        return "", ""

    sampler = _find_relevant_sampler(prompt_graph, gallery_node_id)
    if sampler is not None:
        inputs = sampler.get("inputs", {})
        if not isinstance(inputs, dict):
            inputs = {}

        positive = _resolve_text_from_ref(prompt_graph, inputs.get("positive"), role="positive")
        negative = _resolve_text_from_ref(prompt_graph, inputs.get("negative"), role="negative")
        if not positive:
            positive = _resolve_text_from_ref(prompt_graph, inputs.get("cond_pos"), role="positive")
        if not negative:
            negative = _resolve_text_from_ref(prompt_graph, inputs.get("cond_neg"), role="negative")
        if not positive:
            # BasicGuider-style nodes (Flux / SamplerCustomAdvanced pipelines)
            # carry a single `conditioning` input — that's the positive.
            positive = _resolve_text_from_ref(prompt_graph, inputs.get("conditioning"), role="positive")
        if positive or negative:
            return positive, negative

    samplers: list[dict[str, Any]] = []
    for node_key, node in prompt_graph.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type", ""))
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            inputs = {}

        has_sampler_links = ("positive" in inputs) or ("negative" in inputs)
        if class_type.startswith("KSampler") or has_sampler_links:
            samplers.append({"key": str(node_key), "node": node})

    if not samplers:
        return "", ""

    # A global scan is only trustworthy when it's unambiguous. With multiple
    # sampler-ish nodes (multi-pipeline workflows, ControlNet appliers, etc.),
    # picking the lowest-numbered one returns prompts from a *different*
    # pipeline — confidently wrong is worse than honestly unavailable.
    if gallery_node_id is not None and len(samplers) > 1:
        return "", ""

    def _sort_key(item: dict[str, Any]) -> tuple[int, str]:
        key = item["key"]
        return (0, key) if key.isdigit() else (1, key)

    sampler = sorted(samplers, key=_sort_key)[0]["node"]
    inputs = sampler.get("inputs", {})
    if not isinstance(inputs, dict):
        return "", ""

    positive = _resolve_text_from_ref(prompt_graph, inputs.get("positive"), role="positive")
    negative = _resolve_text_from_ref(prompt_graph, inputs.get("negative"), role="negative")

    if not positive:
        positive = _resolve_text_from_ref(prompt_graph, inputs.get("cond_pos"), role="positive")
    if not negative:
        negative = _resolve_text_from_ref(prompt_graph, inputs.get("cond_neg"), role="negative")
    return positive, negative


def _extract_prompts_with_fallback(prompt_graph: Any, extra_pnginfo: Any, gallery_node_id: str | None = None) -> tuple[str, str, str]:
    # --- Primary: live workflow graph, walked from our specific gallery node ---
    positive, negative = _extract_prompts(prompt_graph, gallery_node_id)
    if positive:
        return positive, negative, "workflow graph"

    if not isinstance(extra_pnginfo, dict):
        return positive, negative, "unavailable"

    # --- Fallback 1: embedded prompt JSON (also scoped to gallery_node_id) ---
    # NOTE: no unscoped retry here. An unscoped walk of a multi-pipeline
    # workflow returns some OTHER pipeline's prompt — confidently wrong.
    embedded_prompt = extra_pnginfo.get("prompt")
    if embedded_prompt is not None:
        fallback_positive, fallback_negative = _extract_prompts(embedded_prompt, gallery_node_id)
        if fallback_positive:
            return fallback_positive, fallback_negative, "embedded prompt metadata"

    # --- Fallback 2: embedded workflow JSON (LiteGraph format) ---
    # Pass gallery_node_id so we resolve from the correct sampler, not just
    # the first sampler in the graph (which caused the "random prompt" bug).
    fallback_positive, fallback_negative = _extract_prompts_from_workflow(
        extra_pnginfo.get("workflow"), gallery_node_id
    )
    if fallback_positive:
        return fallback_positive, fallback_negative, "embedded workflow metadata"

    return positive, negative, "unavailable"


def _extract_seed(prompt_graph: Any, gallery_node_id: str | None = None) -> int | None:
    """Extract the seed by walking upstream from the gallery's image input.

    Every node on the walk is checked for a seed-ish field, so this works for
    KSampler (seed), SamplerCustom (noise_seed) AND SamplerCustomAdvanced
    pipelines where the seed lives on a separate RandomNoise node that the
    old sampler-only lookup could never see.
    """
    if not isinstance(prompt_graph, dict):
        return None

    seed_keys = ("seed", "noise_seed", "seed_num", "rand_seed")

    def _seed_from_node(node: Any) -> int | None:
        if not isinstance(node, dict):
            return None
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            return None
        for key in seed_keys:
            val = inputs.get(key)
            if isinstance(val, bool):
                continue
            if isinstance(val, int):
                return val
            if isinstance(val, str) and val.isdigit():
                return int(val)
        return None

    # --- Primary: BFS upstream from THIS gallery's image input ---
    if gallery_node_id:
        gallery_node = prompt_graph.get(str(gallery_node_id))
        if isinstance(gallery_node, dict):
            gallery_inputs = gallery_node.get("inputs", {})
            if not isinstance(gallery_inputs, dict):
                gallery_inputs = {}
            start_node_id = _get_ref_node_id(gallery_inputs.get("images"))
            if start_node_id:
                from collections import deque
                queue: deque[str] = deque([start_node_id])
                visited: set[str] = set()
                while queue:
                    node_id = queue.popleft()
                    if not node_id or node_id in visited:
                        continue
                    visited.add(node_id)
                    node = prompt_graph.get(node_id)
                    if not isinstance(node, dict):
                        continue
                    found = _seed_from_node(node)
                    if found is not None:
                        return found
                    inputs = node.get("inputs", {})
                    if isinstance(inputs, dict):
                        for child_node_id in _iter_child_node_ids(inputs):
                            if child_node_id not in visited:
                                queue.append(child_node_id)

    # --- Fallback: only if the graph has exactly ONE seeded node (unambiguous) ---
    seeds_found: list[int] = []
    for node in prompt_graph.values():
        found = _seed_from_node(node)
        if found is not None:
            seeds_found.append(found)
    if len(seeds_found) == 1:
        return seeds_found[0]
    return None


def _extract_prompt_text_from_workflow_node(node: Dict[str, Any]) -> str:
    node_type = str(node.get("type", ""))
    if "TextEncode" not in node_type:
        return ""

    widgets = node.get("widgets_values")
    if not isinstance(widgets, list):
        return ""

    parts = [item.strip() for item in widgets if isinstance(item, str) and item.strip()]
    if not parts:
        return ""
    return "\n".join(list(dict.fromkeys(parts)))


def _extract_prompts_from_workflow(workflow: Any, gallery_node_id: str | None = None) -> tuple[str, str]:
    if not isinstance(workflow, dict):
        return "", ""

    nodes = workflow.get("nodes")
    links = workflow.get("links")
    if not isinstance(nodes, list) or not isinstance(links, list):
        return "", ""

    node_by_id: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if node_id is None:
            continue
        node_by_id[str(node_id)] = node

    link_to_from: Dict[int, str] = {}
    for link in links:
        if not isinstance(link, list) or len(link) < 2:
            continue
        link_id, from_node_id = link[0], link[1]
        if isinstance(link_id, int):
            link_to_from[link_id] = str(from_node_id)

    def resolve_from_node_id(node_id: str, visited: set[str]) -> str:
        if node_id in visited:
            return ""
        visited_next = set(visited)
        visited_next.add(node_id)

        node = node_by_id.get(node_id)
        if not isinstance(node, dict):
            return ""

        text = _extract_prompt_text_from_workflow_node(node)
        if text:
            return text

        inputs = node.get("inputs")
        if not isinstance(inputs, list):
            return ""

        for input_def in inputs:
            if not isinstance(input_def, dict):
                continue
            link_id = input_def.get("link")
            if not isinstance(link_id, int):
                continue
            upstream_id = link_to_from.get(link_id)
            if not upstream_id:
                continue
            text = resolve_from_node_id(upstream_id, visited_next)
            if text:
                return text
        return ""

    sampler_candidates: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, list):
            continue
        names = {str(item.get("name", "")).lower() for item in inputs if isinstance(item, dict)}
        if {"positive", "negative"}.intersection(names) or {"cond_pos", "cond_neg"}.intersection(names):
            sampler_candidates.append(node)

    # If we know which gallery node to scope to, find the sampler that feeds it
    # by walking upstream through the LiteGraph link map.
    if gallery_node_id and sampler_candidates:
        gallery_wf_node = node_by_id.get(str(gallery_node_id))
        if isinstance(gallery_wf_node, dict):
            gallery_inputs_list = gallery_wf_node.get("inputs")
            if isinstance(gallery_inputs_list, list):
                # Find the link id connected to the "images" input of the gallery node
                images_link_id = None
                for inp in gallery_inputs_list:
                    if isinstance(inp, dict) and str(inp.get("name", "")).lower() == "images":
                        images_link_id = inp.get("link")
                        break
                if isinstance(images_link_id, int):
                    # BFS upstream from gallery's image input to find closest sampler
                    from collections import deque as _deque
                    upstream_start = link_to_from.get(images_link_id)
                    if upstream_start:
                        bfs_queue: _deque[str] = _deque([upstream_start])
                        bfs_visited: set[str] = set()
                        sampler_candidate_ids = {str(n.get("id", "")) for n in sampler_candidates}
                        while bfs_queue:
                            cur_id = bfs_queue.popleft()
                            if not cur_id or cur_id in bfs_visited:
                                continue
                            bfs_visited.add(cur_id)
                            if cur_id in sampler_candidate_ids:
                                # Found the closest sampler upstream — use it exclusively
                                sampler_candidates = [n for n in sampler_candidates if str(n.get("id", "")) == cur_id]
                                break
                            cur_node = node_by_id.get(cur_id)
                            if not isinstance(cur_node, dict):
                                continue
                            cur_inputs = cur_node.get("inputs")
                            if isinstance(cur_inputs, list):
                                for inp in cur_inputs:
                                    if isinstance(inp, dict):
                                        lid = inp.get("link")
                                        if isinstance(lid, int):
                                            nxt = link_to_from.get(lid)
                                            if nxt and nxt not in bfs_visited:
                                                bfs_queue.append(nxt)

    def sort_key(node: Dict[str, Any]) -> tuple[int, str]:
        node_id = str(node.get("id", ""))
        return (0, node_id) if node_id.isdigit() else (1, node_id)

    for sampler in sorted(sampler_candidates, key=sort_key):
        inputs = sampler.get("inputs")
        if not isinstance(inputs, list):
            continue

        by_name = {str(item.get("name", "")).lower(): item for item in inputs if isinstance(item, dict)}

        def resolve_input(*names: str) -> str:
            for name in names:
                input_def = by_name.get(name)
                if not isinstance(input_def, dict):
                    continue
                link_id = input_def.get("link")
                if not isinstance(link_id, int):
                    continue
                upstream_id = link_to_from.get(link_id)
                if not upstream_id:
                    continue
                text = resolve_from_node_id(upstream_id, set())
                if text:
                    return text
            return ""

        positive = resolve_input("positive", "cond_pos")
        negative = resolve_input("negative", "cond_neg")
        if positive:
            return positive, negative

    return "", ""


def _build_pnginfo(prompt: Any, extra_pnginfo: Any) -> PngInfo:
    pnginfo = PngInfo()

    if prompt is not None:
        try:
            pnginfo.add_text("prompt", json.dumps(prompt, ensure_ascii=False))
        except Exception:
            pass

    if isinstance(extra_pnginfo, dict):
        for key, value in extra_pnginfo.items():
            try:
                pnginfo.add_text(str(key), json.dumps(value, ensure_ascii=False))
            except Exception:
                pass

    return pnginfo

def _gallery_payload(node_id: str) -> Dict[str, Any]:
    with STATE_LOCK:
        state = GALLERY_STATE.get(node_id, {})
        entries = [_entry_public(item) for item in state.get("entries", [])]
        return {
            "node_id": node_id,
            "count": len(entries),
            "max_images": state.get("max_images", 100),
            "output_directory": state.get("output_directory", str(DEFAULT_SAVE_DIR)),
            "save_to_disk": state.get("save_to_disk", False),
            "entries": entries,
        }


def _send_gallery_update(node_id: str) -> None:
    _save_session(node_id)
    PromptServer.instance.send_sync("workflow_gallery_update", _gallery_payload(node_id))


def _ensure_state(node_id: str, output_directory: str, max_images: int, save_to_disk: bool) -> Dict[str, Any]:
    with STATE_LOCK:
        state = GALLERY_STATE.setdefault(
            str(node_id),
            {
                "entries": [],
                "max_images": max_images,
                "output_directory": output_directory,
                "save_to_disk": save_to_disk,
            },
        )
        state["max_images"] = max_images
        state["output_directory"] = output_directory
        state["save_to_disk"] = save_to_disk
        return state


def _prune_entries(node_id: str, state: Dict[str, Any], max_images: int) -> None:
    removed: List[Dict[str, Any]] = []
    while len(state["entries"]) > max_images:
        removed.append(state["entries"].pop(0))

    for entry in removed:
        ENTRY_INDEX.pop(entry.get("id", ""), None)
        for key in ("full_path", "thumb_path"):
            try:
                if entry.get(key):
                    Path(entry[key]).unlink(missing_ok=True)
            except Exception:
                pass


def _find_entry(entry_id: str) -> Dict[str, Any] | None:
    with STATE_LOCK:
        # Fast path — normal operation
        entry = ENTRY_INDEX.get(entry_id)
        if entry:
            return entry
        # Fallback: scan all gallery states (handles session restore edge cases
        # where ENTRY_INDEX may not have been fully rebuilt). Repairs the index
        # on the fly so subsequent lookups are fast.
        for state in GALLERY_STATE.values():
            for e in state.get("entries", []):
                if e.get("id") == entry_id:
                    ENTRY_INDEX[entry_id] = e
                    return e
        return None


def _session_path(node_id: str) -> Path:
    safe_id = "".join(ch for ch in str(node_id) if ch.isalnum() or ch in ("-", "_"))
    return SESSION_DIR / f"session_{safe_id}.json"


def _save_session(node_id: str) -> None:
    try:
        with STATE_LOCK:
            state = GALLERY_STATE.get(str(node_id))
            if not state:
                return
            data = {
                "node_id": str(node_id),
                "max_images": state.get("max_images", 48),
                "output_directory": state.get("output_directory", str(DEFAULT_SAVE_DIR)),
                "save_to_disk": state.get("save_to_disk", False),
                "entries": list(state.get("entries", [])),
            }
        path = _session_path(node_id)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(path)
    except Exception as e:
        print(f"[WorkflowGallery] Warning: could not save session for node {node_id}: {e}", flush=True)


def _load_all_sessions() -> None:
    if not SESSION_DIR.exists():
        print("[WorkflowGallery] Session directory not found.", flush=True)
        return

    session_files = list(SESSION_DIR.glob("session_*.json"))
    if not session_files:
        return

    print(f"[WorkflowGallery] Found {len(session_files)} session file(s), restoring...", flush=True)

    for session_file in session_files:
        try:
            data = json.loads(session_file.read_text(encoding="utf-8"))
            node_id = str(data.get("node_id", ""))
            if not node_id:
                continue

            raw_entries: List[Dict[str, Any]] = data.get("entries", [])
            valid_entries: List[Dict[str, Any]] = []
            for entry in raw_entries:
                full_path = entry.get("full_path", "")
                thumb_path = entry.get("thumb_path", "")
                # Use os.path.exists for reliable Windows path checking
                if not full_path or not os.path.exists(full_path):
                    print(f"[WorkflowGallery] Skipping {entry.get('id','?')}: full image missing at {full_path!r}", flush=True)
                    continue
                if not thumb_path or not os.path.exists(thumb_path):
                    print(f"[WorkflowGallery] Skipping {entry.get('id','?')}: thumbnail missing at {thumb_path!r}", flush=True)
                    continue
                valid_entries.append(entry)

            if not valid_entries:
                print(f"[WorkflowGallery] No valid entries in {session_file.name}, removing.", flush=True)
                session_file.unlink(missing_ok=True)
                continue

            with STATE_LOCK:
                state = GALLERY_STATE.setdefault(node_id, {
                    "entries": [],
                    "max_images": data.get("max_images", 48),
                    "output_directory": data.get("output_directory", str(DEFAULT_SAVE_DIR)),
                    "save_to_disk": data.get("save_to_disk", False),
                })
                state["entries"] = valid_entries
                state["max_images"] = data.get("max_images", 48)
                state["output_directory"] = data.get("output_directory", str(DEFAULT_SAVE_DIR))
                state["save_to_disk"] = data.get("save_to_disk", False)
                for entry in valid_entries:
                    ENTRY_INDEX[entry["id"]] = entry

            print(f"[WorkflowGallery] Restored {len(valid_entries)} image(s) for node {node_id}", flush=True)

        except Exception as e:
            import traceback
            print(f"[WorkflowGallery] ERROR loading {session_file.name}: {e}", flush=True)
            traceback.print_exc()


_load_all_sessions()


class WorkflowGallery:
    CATEGORY = "image/ui"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "collect"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True}),
                "save_to_disk": ("BOOLEAN", {"default": False}),
                "output_directory": ("STRING", {"default": str(DEFAULT_SAVE_DIR), "multiline": False}),
                "filename_prefix": ("STRING", {"default": "workflow_gallery", "multiline": False}),
                "max_images": ("INT", {"default": 48, "min": 1, "max": 500, "step": 1}),
            },
            "optional": {
                # Wire-only override sockets (forceInput => no widget is created,
                # so widgets_values order is unchanged and saved workflows are safe).
                # When connected, these are ground truth — the exact string fed to
                # the encoder — and beat graph resolution entirely.
                "positive_override": ("STRING", {"forceInput": True, "default": ""}),
                "negative_override": ("STRING", {"forceInput": True, "default": ""}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def collect(
        self,
        images,
        enabled: bool = True,
        save_to_disk: bool = False,
        output_directory: str = str(DEFAULT_SAVE_DIR),
        filename_prefix: str = "workflow_gallery",
        max_images: int = 48,
        positive_override: str = "",
        negative_override: str = "",
        unique_id: str | None = None,
        prompt: Dict[str, Any] | None = None,
        extra_pnginfo: Dict[str, Any] | None = None,
    ):
        node_id = str(unique_id or "unknown")
        max_images = _safe_int(max_images, 48, 1, 500)
        resolved_output_dir = _normalize_output_dir(output_directory)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

        state = _ensure_state(node_id, str(resolved_output_dir), max_images, bool(save_to_disk))

        if not enabled:
            _send_gallery_update(node_id)
            return {"ui": {"images": []}, "result": (images,)}

        safe_prefix = _sanitize_prefix(filename_prefix)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        positive_prompt, negative_prompt, prompt_source = _extract_prompts_with_fallback(prompt, extra_pnginfo, node_id)

        # Wired overrides are the exact strings fed to the encoder — ground
        # truth. They always beat whatever the graph walk came up with.
        if isinstance(positive_override, str) and positive_override.strip():
            positive_prompt = positive_override.strip()
            prompt_source = "direct input"
        if isinstance(negative_override, str) and negative_override.strip():
            negative_prompt = negative_override.strip()
            if prompt_source == "unavailable":
                prompt_source = "direct input"

        display_prompt = positive_prompt
        seed = _extract_seed(prompt, node_id)
        pnginfo = _build_pnginfo(prompt, extra_pnginfo)

        new_entries: List[Dict[str, Any]] = []
        for idx, image_tensor in enumerate(images):
            pil_image = _tensor_to_pil(image_tensor)
            width, height = pil_image.size
            entry_id = uuid.uuid4().hex
            filename = f"{safe_prefix}_{timestamp}_{idx:03d}_{entry_id[:8]}.png"
            full_path = resolved_output_dir / filename

            if save_to_disk:
                pil_image.save(full_path, format="PNG", compress_level=4, pnginfo=pnginfo)
            else:
                # Still save to a temp-ish package folder so the frontend can display original-size images.
                temp_dir = CACHE_BASE_DIR / "unsaved_cache"
                temp_dir.mkdir(parents=True, exist_ok=True)
                full_path = temp_dir / filename
                pil_image.save(full_path, format="PNG", compress_level=4, pnginfo=pnginfo)

            thumb_path = CACHE_BASE_DIR / "thumb_cache" / f"{entry_id}.webp"
            thumb_path.parent.mkdir(parents=True, exist_ok=True)
            thumb_path.write_bytes(_thumbnail_bytes(pil_image))

            new_entries.append(
                {
                    "id": entry_id,
                    "filename": filename,
                    "created": int(time.time()),
                    "width": width,
                    "height": height,
                    "display_prompt": display_prompt,
                    "prompt_source": prompt_source,
                    "positive_prompt": positive_prompt,
                    "negative_prompt": negative_prompt,
                    "seed": seed,
                    "full_path": str(full_path),
                    "thumb_path": str(thumb_path),
                }
            )

        with STATE_LOCK:
            state["entries"].extend(new_entries)
            for entry in new_entries:
                ENTRY_INDEX[entry["id"]] = entry
            _prune_entries(node_id, state, max_images)

        _send_gallery_update(node_id)
        return {"ui": {"images": []}, "result": (images,)}


routes = PromptServer.instance.routes


@routes.get("/workflow_gallery/state/{node_id}")
async def workflow_gallery_state(request):
    node_id = request.match_info["node_id"]
    return web.json_response(_gallery_payload(node_id))


@routes.post("/workflow_gallery/clear/{node_id}")
async def workflow_gallery_clear(request):
    node_id = request.match_info["node_id"]
    with STATE_LOCK:
        state = GALLERY_STATE.setdefault(node_id, {"entries": [], "max_images": 100, "output_directory": str(DEFAULT_SAVE_DIR), "save_to_disk": False})
        entries = list(state.get("entries", []))
        state["entries"] = []
        for entry in entries:
            ENTRY_INDEX.pop(entry.get("id", ""), None)

    for entry in entries:
        for key in ("full_path", "thumb_path"):
            path = entry.get(key)
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass

    _send_gallery_update(node_id)
    try:
        _session_path(node_id).unlink(missing_ok=True)
    except Exception:
        pass
    return web.json_response({"ok": True})


@routes.get("/workflow_gallery/file/{entry_id}")
async def workflow_gallery_file(request):
    entry_id = request.match_info["entry_id"]
    kind = request.query.get("kind", "thumb")
    entry = _find_entry(entry_id)
    if not entry:
        return web.Response(status=404, text="Not found")

    path_key = "thumb_path" if kind == "thumb" else "full_path"
    path = Path(entry[path_key])
    if not path.exists():
        print(f"[WorkflowGallery] File missing: {path}", flush=True)
        return web.Response(status=404, text="File missing")
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return web.Response(status=404, text="File type not allowed")

    content_type_map = {".webp": "image/webp", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    content_type = content_type_map.get(path.suffix.lower(), "application/octet-stream")
    try:
        data = path.read_bytes()
        return web.Response(
            body=data,
            content_type=content_type,
            headers={"Cache-Control": "no-store"},
        )
    except Exception as e:
        print(f"[WorkflowGallery] Error reading file {path}: {e}", flush=True)
        return web.Response(status=500, text="Error reading file")


@routes.post("/workflow_gallery/export")
async def workflow_gallery_export(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    node_id = str(body.get("node_id", ""))
    entry_ids: List[str] = body.get("entry_ids", [])
    output_directory = str(body.get("output_directory", "")).strip()

    if not entry_ids:
        return web.json_response({"ok": False, "error": "No entry IDs provided"}, status=400)

    # Resolve destination — use the node's configured output dir or fall back to default
    dest_dir = _resolve_output_dir(output_directory) if output_directory else DEFAULT_SAVE_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    exported: List[str] = []
    errors: List[str] = []

    for entry_id in entry_ids:
        entry = _find_entry(entry_id)
        if not entry:
            errors.append(f"{entry_id}: not found")
            continue

        src_path = Path(entry.get("full_path", ""))
        if not src_path.exists():
            errors.append(f"{entry_id}: source file missing")
            continue

        dest_path = dest_dir / src_path.name
        # Avoid overwriting — append a suffix if needed
        counter = 1
        while dest_path.exists():
            dest_path = dest_dir / f"{src_path.stem}_{counter}{src_path.suffix}"
            counter += 1

        try:
            import shutil
            shutil.copy2(str(src_path), str(dest_path))
            # Mark entry as exported in state
            with STATE_LOCK:
                entry["exported"] = True
                entry["exported_path"] = str(dest_path)
            exported.append(entry_id)
        except Exception as e:
            errors.append(f"{entry_id}: {e}")

    if node_id:
        _send_gallery_update(node_id)

    return web.json_response({
        "ok": True,
        "exported": exported,
        "errors": errors,
        "dest_directory": str(dest_dir),
    })


@routes.post("/workflow_gallery/clear_unexported/{node_id}")
async def workflow_gallery_clear_unexported(request):
    node_id = request.match_info["node_id"]
    with STATE_LOCK:
        state = GALLERY_STATE.get(node_id)
        if not state:
            return web.json_response({"ok": True, "removed": 0})

        to_remove = [e for e in state.get("entries", []) if not e.get("exported", False)]
        state["entries"] = [e for e in state.get("entries", []) if e.get("exported", False)]
        for entry in to_remove:
            ENTRY_INDEX.pop(entry.get("id", ""), None)

    # Clean up files for removed entries
    for entry in to_remove:
        for key in ("thumb_path",):
            path = entry.get(key)
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass
        # Only delete full_path if it's in the cache (not a user-configured save dir)
        full_path = entry.get("full_path", "")
        if full_path and str(CACHE_BASE_DIR) in full_path:
            try:
                Path(full_path).unlink(missing_ok=True)
            except Exception:
                pass

    _send_gallery_update(node_id)
    return web.json_response({"ok": True, "removed": len(to_remove)})


@routes.post("/workflow_gallery/delete_entries/{node_id}")
async def workflow_gallery_delete_entries(request):
    node_id = request.match_info["node_id"]
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    entry_ids: List[str] = body.get("entry_ids", [])
    if not entry_ids:
        return web.json_response({"ok": False, "error": "No entry IDs"}, status=400)

    to_remove: List[Dict[str, Any]] = []
    with STATE_LOCK:
        state = GALLERY_STATE.get(node_id)
        if state:
            remaining = []
            for entry in state.get("entries", []):
                if entry.get("id") in entry_ids:
                    to_remove.append(entry)
                    ENTRY_INDEX.pop(entry.get("id", ""), None)
                else:
                    remaining.append(entry)
            state["entries"] = remaining

    for entry in to_remove:
        for key in ("thumb_path",):
            path = entry.get(key)
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass
        full_path = entry.get("full_path", "")
        if full_path and str(CACHE_BASE_DIR) in full_path:
            try:
                Path(full_path).unlink(missing_ok=True)
            except Exception:
                pass

    _send_gallery_update(node_id)
    return web.json_response({"ok": True, "removed": len(to_remove)})


def _load_prompt_library() -> List[Dict[str, Any]]:
    try:
        if PROMPT_LIBRARY_FILE.exists():
            return json.loads(PROMPT_LIBRARY_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[PromptLibrary] Error loading library: {e}", flush=True)
    return []


def _save_prompt_library(entries: List[Dict[str, Any]]) -> None:
    try:
        tmp = PROMPT_LIBRARY_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(PROMPT_LIBRARY_FILE)
    except Exception as e:
        print(f"[PromptLibrary] Error saving library: {e}", flush=True)


class PromptLibrary:
    CATEGORY = "utils/prompt"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "get_prompt"
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute so wildcard randomization runs every queue
        # and the node never gets skipped due to ComfyUI output caching.
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "selected_prompt_id": ("STRING", {"default": "", "multiline": False}),
                "manual_text": ("STRING", {"default": "", "multiline": True, "placeholder": "Type here or connect a STRING wire"}),
                "manual_position": (["after", "before"],),
                "expand_wildcards": ("BOOLEAN", {"default": False}),
                "wildcard_path": ("STRING", {"default": "", "multiline": False, "placeholder": "Optional: extra wildcard folder(s), comma separated. Node wildcards/ folder is always searched first."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def get_prompt(self, selected_prompt_id: str = "", manual_text: str = "", manual_position: str = "after", expand_wildcards: bool = False, wildcard_path: str = "", seed: int = 0, unique_id: str | None = None):
        node_id = str(unique_id or "")

        # Build combined output from selected prompt + manual text
        selected = selected_prompt_id.strip()
        manual = manual_text.strip()

        if manual and selected:
            output = f"{manual}, {selected}" if manual_position == "before" else f"{selected}, {manual}"
        elif selected:
            output = selected
        elif manual:
            output = manual
        else:
            output = ""

        if expand_wildcards and output:
            dirs = _resolve_wildcard_dirs(wildcard_path)
            rng = random.Random(seed)
            output = _expand_wildcards(output, dirs, rng=rng)
            print(f"[PromptLibrary] Wildcard expanded (seed={seed}): {output[:100]!r}", flush=True)

        print(f"[PromptLibrary] Output: selected={selected[:40]!r} manual={manual[:40]!r} combined={output[:80]!r}", flush=True)

        if node_id:
            PROMPT_LIBRARY_OUTPUTS[node_id] = output
        return (output,)


@routes.get("/prompt_library/list")
async def prompt_library_list(request):
    with PROMPT_LIBRARY_LOCK:
        entries = _load_prompt_library()
    return web.json_response({"entries": entries})


@routes.post("/prompt_library/save")
async def prompt_library_save(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    name = str(body.get("name", "")).strip()
    positive_prompt = str(body.get("positive_prompt", "")).strip()
    negative_prompt = str(body.get("negative_prompt", "")).strip()
    tags = [str(t).strip() for t in body.get("tags", []) if str(t).strip()]

    if not name:
        return web.json_response({"ok": False, "error": "Name is required"}, status=400)
    if not positive_prompt:
        return web.json_response({"ok": False, "error": "Prompt is required"}, status=400)

    with PROMPT_LIBRARY_LOCK:
        entries = _load_prompt_library()
        for entry in entries:
            if entry.get("positive_prompt", "").strip() == positive_prompt:
                return web.json_response({"ok": False, "error": "Prompt already exists", "id": entry["id"]}, status=409)
        new_entry = {
            "id": uuid.uuid4().hex,
            "name": name,
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "tags": tags,
            "created": int(time.time()),
        }
        entries.append(new_entry)
        _save_prompt_library(entries)

    return web.json_response({"ok": True, "entry": new_entry})


@routes.post("/prompt_library/delete")
async def prompt_library_delete(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    entry_id = str(body.get("id", "")).strip()
    if not entry_id:
        return web.json_response({"ok": False, "error": "ID required"}, status=400)

    with PROMPT_LIBRARY_LOCK:
        entries = _load_prompt_library()
        new_entries = [e for e in entries if e.get("id") != entry_id]
        if len(new_entries) == len(entries):
            return web.json_response({"ok": False, "error": "Not found"}, status=404)
        _save_prompt_library(new_entries)

    return web.json_response({"ok": True})


@routes.post("/prompt_library/update")
async def prompt_library_update(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    entry_id = str(body.get("id", "")).strip()
    if not entry_id:
        return web.json_response({"ok": False, "error": "ID required"}, status=400)

    with PROMPT_LIBRARY_LOCK:
        entries = _load_prompt_library()
        for entry in entries:
            if entry.get("id") == entry_id:
                if "name" in body: entry["name"] = str(body["name"]).strip()
                if "tags" in body: entry["tags"] = [str(t).strip() for t in body["tags"] if str(t).strip()]
                if "positive_prompt" in body: entry["positive_prompt"] = str(body["positive_prompt"]).strip()
                if "negative_prompt" in body: entry["negative_prompt"] = str(body["negative_prompt"]).strip()
                _save_prompt_library(entries)
                return web.json_response({"ok": True, "entry": entry})

    return web.json_response({"ok": False, "error": "Not found"}, status=404)


@routes.get("/prompt_library/export")
async def prompt_library_export(request):
    fmt = request.query.get("format", "json")
    with PROMPT_LIBRARY_LOCK:
        entries = _load_prompt_library()

    if fmt == "csv":
        import csv
        import io as _io
        buf = _io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["id", "name", "tags", "positive_prompt", "negative_prompt", "created"])
        writer.writeheader()
        for e in entries:
            writer.writerow({
                "id": e.get("id", ""),
                "name": e.get("name", ""),
                "tags": ",".join(e.get("tags", [])),
                "positive_prompt": e.get("positive_prompt", ""),
                "negative_prompt": e.get("negative_prompt", ""),
                "created": e.get("created", ""),
            })
        return web.Response(
            body=buf.getvalue().encode("utf-8"),
            content_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=prompt_library.csv"}
        )
    else:
        return web.Response(
            body=json.dumps(entries, ensure_ascii=False, indent=2).encode("utf-8"),
            content_type="application/json",
            headers={"Content-Disposition": "attachment; filename=prompt_library.json"}
        )


# ── Prompt Enhancer ───────────────────────────────────────────────────────────

def _load_danbooru_tags() -> str:
    """Load the danbooru tags reference file and return as a condensed tag list string."""
    tags_file = PACKAGE_DIR / "danbooru_tags.txt"
    if not tags_file.exists():
        return ""
    try:
        lines = tags_file.read_text(encoding="utf-8").splitlines()
        tags = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("["):
                continue
            tags.extend([t.strip() for t in line.split(",") if t.strip()])
        return ", ".join(tags)
    except Exception as e:
        print(f"[PromptEnhancer] Could not load danbooru_tags.txt: {e}", flush=True)
        return ""

_DANBOORU_TAGS = _load_danbooru_tags()


PROMPT_ENHANCER_PRESETS: Dict[str, str] = {
    "Flux / Z-Image": (
        "You write prompts for the Flux / Z-Image text-to-image models. "
        "Turn the user's idea into ONE rich, natural-language image prompt.\n\n"
        "OUTPUT: reply with the prompt ONLY — flowing sentences, no preamble, no lists, no commentary, "
        "no tag-soup, no prompt weights like (word:1.4).\n\n"
        "How these models read prompts: they parse natural language like a sentence, not comma tags. "
        "Put the most important subject first (attention fades after the opening), then build outward. "
        "There are no negative prompts — express any exclusion as a positive phrase ('a plain studio "
        "background, no text or logos'). Favor concrete camera and photographic language (e.g. '85mm f/1.4, "
        "shallow depth of field', 'golden-hour side light') over empty modifiers like 'masterpiece, 8k'. "
        "Always name texture and material (skin pores, fabric weave, film grain) — this is what stops faces "
        "and surfaces looking plastic. If specific text should appear in the image, put it in double quotes. "
        "Don't stack contradictory styles (no 'photoreal anime oil painting').\n\n"
        "Weave in: subject (age, clothing, distinguishing features), the setting, composition and shot/lens, "
        "lighting and color, mood, and texture. Keep it detailed but focused on a few strong concepts.\n\n"
        "Content range: everyday through fully explicit. When the scene is nude or sexual, describe it plainly "
        "and completely — state of undress, anatomy, and sexual activity in direct terms; never euphemize, omit, "
        "or moralize. Absolute rule: all subjects are adults; never sexualize anyone who is or appears underage, "
        "and if age is uncertain, do not sexualize them."
    ),
    "Illustrious / SDXL Anime": (
        "You convert prompts into Danbooru-tag prompts for the Illustrious XL anime model. "
        "Input may be plain English, tags, or a mix. Output is always a clean Illustrious tag string.\n\n"
        "OUTPUT: reply with the tag string ONLY — comma-separated, all lowercase, no preamble, no commentary.\n\n"
        "Order: lead with quality tags (masterpiece, best quality, amazing quality, very aesthetic, absurdres, "
        "newest), then subject count (1girl, 1boy, 2girls...), then the rating tag, then character name if known "
        "(Danbooru order, e.g. 'hatsune miku'), then appearance (hair, eyes, expression), clothing or state of "
        "undress, pose/action, setting/background, and finally lighting and effects.\n\n"
        "Rating tag — pick the one matching the content: 'general' (safe), 'sensitive' (suggestive), 'nsfw' "
        "(nudity/sexual), 'explicit' (hardcore). Do not default to 'general' when the content is adult.\n\n"
        "Tag rules: translate every English description to its real Danbooru tag ('looking at camera' -> looking "
        "at viewer, 'sad' -> sad). Use only real Danbooru tags. Do NOT use Pony score_ tags, and do NOT use "
        "4k/8k/HDR/photorealistic/sharp focus (not Danbooru tags). Add fitting lighting/atmosphere/background "
        "tags the input implies.\n\n"
        "SFW example -> input 'anime girl with white hair sitting in a cafe, rainy day' -> "
        "masterpiece, best quality, amazing quality, very aesthetic, absurdres, newest, 1girl, general, white hair, "
        "long hair, sitting, cafe, indoors, window, rain, looking out window, soft lighting, depth of field\n\n"
        "Content range: everyday through fully explicit. When the content is nude or sexual, set the correct "
        "rating tag and tag the nudity, anatomy, and acts with accurate Danbooru tags — do not omit or soften them. "
        "Absolute rule: all subjects are adults; never tag anyone who is or appears underage with sexual, nude, or "
        "suggestive tags, and if age is uncertain, do not sexualize them."
    ),
    "LTX Video 2.3": (
        "You write prompts for LTX Video 2.3 (text-to-video with synchronized audio). "
        "Turn the user's idea into ONE detailed cinematic video prompt, built from scratch.\n\n"
        "OUTPUT: reply with the prompt ONLY — one flowing paragraph, present tense, no preamble, "
        "no lists, no numbered steps, no commentary. Match length to scene complexity; detail reduces drift.\n\n"
        "Weave in, naturally: a specific subject (age, clothing, materials and texture — fabric, hair, "
        "surface finish, wear); spatial blocking (left/right, foreground/background, who faces whom, "
        "distance between subjects); the action as explicit VERBS — what the subject does, what the "
        "environment does, and what the camera does; one clear camera move; lighting, color and atmosphere; "
        "and audio (ambient tone plus any dialogue in quotes, broken into short lines with physical acting "
        "beats between them). Close by stating how the shot ends and its pacing.\n\n"
        "Rules: emotion through physical action ('her hands tremble'), never labels ('nervous'); a single "
        "continuous shot, no cuts; no on-screen text or logos (LTX renders text unreliably); no numeric "
        "specs like degrees or speeds.\n\n"
        "Content range: everyday through fully explicit. When the scene is nude or sexual, describe it plainly "
        "and completely — state of undress, anatomy, and sexual action in direct terms; never euphemize, omit, "
        "sanitize, or moralize. Absolute rule: all subjects are adults; never sexualize anyone who is or appears "
        "underage, and if age is uncertain, do not sexualize them."
    ),
    "Wan 2.2 Video": (
        "You write prompts for the Wan 2.2 video model. "
        "Turn the user's idea into ONE cinematic video prompt, built from scratch.\n\n"
        "OUTPUT: reply with the prompt ONLY — one flowing paragraph, present tense, no preamble, no lists, "
        "no commentary. Detailed but focused (roughly 80-120 words); clarity beats length.\n\n"
        "Follow Wan's structure: Subject -> Scene -> Motion -> Aesthetic control -> Stylization. "
        "Describe the subject and setting concretely, then the MOTION explicitly with strong action verbs and "
        "amplitude/speed ('rushes', 'snaps', 'drifts slowly'); Wan goes static without clear motion. Separate "
        "foreground and background motion for depth. Then apply aesthetic control with real cinematography terms "
        "(light source and quality, shot size, camera angle, lens, and one clear camera movement) and a color "
        "grade. End with a stylization cue (e.g. cinematic, anamorphic, 16mm grain, documentary).\n\n"
        "Rules: emotion through physical action, never labels; a single continuous shot, no cuts; no numeric "
        "specs like degrees or speeds.\n\n"
        "Content range: everyday through fully explicit. When the scene is nude or sexual, describe it plainly "
        "and completely — state of undress, anatomy, and sexual action in direct terms; never euphemize, omit, "
        "or moralize. Absolute rule: all subjects are adults; never sexualize anyone who is or appears underage, "
        "and if age is uncertain, do not sexualize them."
    ),
    "MiniMax H3 Base": (
        "You write prompts for MiniMax H3 Text to Video + Audio mode (T2VA). The user's text is a directing "
        "instruction, not prompt content to copy literally. Turn it into a concise audiovisual prompt. "
        "Do not mention pictures, references, or reference labels in T2VA output.\n\n"
        "OUTPUT: reply with ONLY these three fields in this exact order and with these exact names:\n"
        "integrated_multimodal_description: [Shot 1] ...\n"
        "overall_soundscape: ...\n"
        "non_diegetic_music: ...\n\n"
        "In integrated_multimodal_description, begin [Shot 1] with the visual style and initial composition, "
        "then describe a continuous physical progression: initial state, action beginning, development, "
        "consequence or reaction, and ending state. Use natural pacing words such as slowly, gradually, briefly, "
        "then, as, while, and near the end. Keep all action in [Shot 1] when the camera does not cut; a new "
        "character action is not a new shot. Create [Shot 2], [Shot 3], and so on only for actual camera cuts or "
        "clearly separate shots. The first shot has no timestamp. A later shot may begin, for example, "
        "'[Shot 2] At 00:04.500, the camera cuts to...'. Do not divide the clip into small timestamp ranges or "
        "write a frame-by-frame screenplay. Use exact timestamps only when they materially align a real cut or "
        "an important synchronized event. Include enough coherent action for the requested duration without "
        "accounting for every second.\n\n"
        "Give every speaking or singing source a stable ID such as (S1). Put only the original dialogue or "
        "lyrics inside <d>[Language] ...</d>; preserve their wording, punctuation, and language exactly. Keep "
        "dialogue at the corresponding action beat inside integrated_multimodal_description and do not repeat or "
        "paraphrase it in overall_soundscape. Put visible scene text in double quotes. overall_soundscape is a "
        "concise description of environmental and diegetic audio such as ambience, footsteps, wind, machinery, "
        "impacts, cloth movement, breathing, laughter, or physical effects. non_diegetic_music describes only "
        "music requested by the user or clearly required by the source prompt; otherwise output exactly N/A. "
        "Describe what should happen. Do not invent dialogue, lyrics, visible text, music, or unrelated negative "
        "restrictions the user did not request."
    ),
    "MiniMax H3 Frame-to-Frame": (
        "You write prompts for MiniMax H3 First/Last Image to Video + Audio mode (FL2VA). The user's text is a "
        "directing instruction for motion between the supplied first and last images, not a description to assign "
        "to either image. Preserve the reference tags exactly as <Picture 1> and <Picture 2>; their numbering "
        "must follow the actual input or connection order. Never rename a tag or replace it with a subject name.\n\n"
        "OUTPUT: reply with the alignment instruction followed by ONLY these three fields in this exact order:\n"
        "How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the "
        "0.00-second mark of the target video; <Picture 2> (from [Shot N]) aligns with the S.SS-second mark of the "
        "target video.\n\n"
        "integrated_multimodal_description: [Shot 1] ...\n"
        "overall_soundscape: ...\n"
        "non_diegetic_music: ...\n\n"
        "Replace N with the actual final shot and S.SS with the requested duration to exactly two decimals. "
        "Keep a continuous camera take under [Shot 1] unless there is an actual cut; do not create a new shot for "
        "each action. Establish <Picture 1>'s identity, style, composition, objects, and spatial relationships, "
        "then describe a coherent physical progression toward <Picture 2>: action begins, develops, causes a "
        "reaction or consequence, and naturally reaches the final image. Preserve identity, facial appearance, "
        "hairstyle, costume, colors, important props, environment, visual style, and relevant spatial continuity "
        "while allowing pose, expression, body position, and movement to change. Do not merely describe two "
        "static images. Do not use micro-timestamp ranges; reserve exact times for the required first/last-frame "
        "alignment, actual cuts, or important synchronized events.\n\n"
        "Keep requested dialogue at its action beat inside integrated_multimodal_description, preserving it "
        "verbatim inside <d>[Language] ...</d> with stable speaker IDs such as (S1). overall_soundscape contains "
        "only ambience and physical or diegetic sounds, without restating dialogue. Set non_diegetic_music to N/A "
        "unless music is requested or clearly required. Do not invent unrelated negative restrictions."
    ),
    "MiniMax H3 Last-Frame": (
        "You write prompts for MiniMax H3 Last Image to Video + Audio mode (L2VA). <Picture 1> is the supplied "
        "last image that the generated video must reach at the final instant. The user's text directs the preceding "
        "action; it is not a description of a first image. Preserve the exact <Picture 1> tag and keep numbering "
        "consistent with actual input or connection order. Never rename the tag or substitute a descriptive name.\n\n"
        "OUTPUT: reply with this alignment instruction, followed by ONLY these three fields in this exact order:\n"
        "How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the "
        "S.SS-second mark of the target video.\n\n"
        "integrated_multimodal_description: [Shot 1] ...\n"
        "overall_soundscape: ...\n"
        "non_diegetic_music: ...\n\n"
        "Replace N with the actual final shot and S.SS with the requested duration to exactly two decimals. In "
        "integrated_multimodal_description, write a plausible continuous progression that culminates in the "
        "composition and state of <Picture 1> at the end. Begin [Shot 1] with the initial state, then let the "
        "action begin, develop, produce a consequence or reaction, and settle into the supplied last image. "
        "Preserve character and facial identity, hairstyle, costume, colors, important props, environment, visual "
        "style, and relevant spatial relationships while allowing pose, expression, body position, and movement "
        "to change. Keep one continuous camera take under [Shot 1] unless an actual cut is requested or necessary. "
        "Do not create new shots for new actions and do not divide the duration into micro-timestamps. Use exact "
        "timing only for the required last-image alignment, an actual cut, or an important synchronized event.\n\n"
        "Keep requested dialogue at the matching action beat inside integrated_multimodal_description, preserving "
        "it verbatim inside <d>[Language] ...</d> with stable speaker IDs such as (S1). overall_soundscape contains "
        "environmental and physical diegetic audio without restating dialogue. Set non_diegetic_music to N/A "
        "unless the user requests music or the source prompt clearly calls for it. Describe what should happen and "
        "do not invent unrelated negative restrictions."
    ),
    "MiniMax H3 Reference": (
        "You write prompts for MiniMax H3 Reference to Video + Audio mode (Ref2VA). The user's text is a directing instruction. For "
        "example, 'write a 15 second scene for the female in reference 1; she dances in the rain in a 1950s "
        "black-and-white Broadway musical' means: retain the female from Reference 1 as the referenced subject, "
        "then invent and fully stage the requested dance scene around her. It does not mean the instruction itself "
        "describes the reference image. Rewrite the request using stable labels for every reference the user "
        "actually identifies. Reply with ONLY these six sections in this exact order and "
        "with these exact names:\n"
        "subject_definitions: ...\n"
        "summary: ...\n"
        "retention_analysis: ...\n"
        "detailed_description: ...\n"
        "overall_soundscape: ...\n"
        "non_diegetic_music: ...\n\n"
        "In subject_definitions, assign stable <Subject N> labels to reusable visible content, <Picture N> to "
        "concrete frame or storyboard anchors, <Video N> to source-video or temporal-structure relationships, "
        "and <Audio N> to copied or referenced audio signals. Preserve <Picture N>, <Video N>, and <Audio N> "
        "exactly, number each modality independently according to actual input or connection order, and never "
        "rename a tag or substitute a descriptive name. Give each tracked item its own line and never change a "
        "label's meaning. When the asset itself is not visible to you, define only the role stated by the "
        "user (for example, '<Subject 1> is the female from <Picture 1>') and never fabricate her appearance. "
        "In summary, begin with the applicable square-bracketed task types chosen from "
        "keyframe completion, reference generation, video editing, video continuation, audio reuse, and audio "
        "reference, joined with ' + ', then briefly state the target and reference relationships.\n\n"
        "In retention_analysis, give every label one line with its shots or role and the correct fixed marker: "
        "visible references use fully_preserved, partially_preserved, attribute_transfer, or weak_reference; "
        "audio uses fully_copy, partially_copy, reference, or weak_reference. In detailed_description, establish "
        "the overall style before [Shot 1], then describe continuous physical action in playback order: initial "
        "state, action beginning, development, consequence or reaction, and ending state. Keep changing actions "
        "inside the same shot; add [Shot 2], [Shot 3], and so on only for real camera cuts or clearly separate "
        "shots. Later shots may use '[Shot N] At MM:SS.mmm, the camera cuts to...'. Do not split the clip into "
        "micro-timestamp ranges or a frame-by-frame screenplay. Use exact timing only for reference alignment, "
        "actual cuts, or important synchronized events. Preserve referenced identity, facial appearance, hairstyle, "
        "costume, colors, important props, environment, visual style, and relevant spatial relationships while "
        "allowing requested pose, expression, body position, and movement changes.\n\n"
        "Use stable speaker IDs such as (S1), preserve requested speech inside <d>[Language] ...</d>, and keep it "
        "at the matching action beat in detailed_description. Do not repeat or paraphrase dialogue in "
        "overall_soundscape; that section covers environmental and physical diegetic sounds such as ambience, "
        "footsteps, machinery, breathing, laughter, cloth movement, impacts, and effects. Set "
        "non_diegetic_music to N/A unless music is requested or clearly required. Do not invent dialogue, music, "
        "visible text, unresolved labels, or unrelated negative restrictions, and do not write a mere plot summary."
    ),
    "Custom": (
        "You are an expert prompt engineer for AI image and video generation models. "
        "Your job is to aggressively enhance the user's rough prompt into a rich, detailed, effective generation prompt. "
        "DO NOT just reword what they gave you — actively ADD missing elements: "
        "lighting setup, color palette, mood, composition, style references, technical details, atmosphere, and texture. "
        "Preserve the original subject and intent while making it significantly more vivid and specific. "
        "Return ONLY the enhanced prompt. No explanations, no preamble, no metadata.\n\n"
        "Content range: everyday through fully explicit. When the content is nude or sexual, describe it plainly "
        "and completely — state of undress, anatomy, and sexual activity in direct terms; never euphemize, omit, "
        "or moralize. Absolute rule: all subjects are adults; never sexualize anyone who is or appears underage, "
        "and if age is uncertain, do not sexualize them."
    ),
    # ── Vision presets — used when an IMAGE is connected ──────────────────────
    # These treat the image as reference material and generate a prompt FROM it.
    "Flux / Z-Image (Vision)": (
        "You write prompts for the Flux / Z-Image text-to-image models from a reference image. "
        "Generate a natural-language prompt that would recreate what you see.\n\n"
        "OUTPUT: reply with the prompt ONLY — flowing sentences, no preamble, no 'this image shows', "
        "no lists, no tags, no prompt weights.\n\n"
        "Describe only what is actually visible — no invented backstory or assumed emotions. Subject first "
        "(apparent age, clothing or state of undress, readable expression), then visible pose/action, then "
        "lighting (direction, quality, color temperature), composition and depth, color palette, and the "
        "technical look (photographic vs illustrated, film grain, shallow DOF). Name texture and material so "
        "surfaces don't render plastic. Express any exclusion as a positive phrase; there are no negative prompts.\n\n"
        "Content range: everyday through fully explicit. When the image is nude or sexual, describe it plainly "
        "and completely — state of undress, anatomy, and any sexual activity in direct terms; never euphemize, "
        "omit, or sanitize. Absolute rule: treat all subjects as adults; never sexualize anyone who is or appears "
        "underage, and if age is uncertain, do not sexualize them."
    ),
    "Illustrious / SDXL Anime (Vision)": (
        "You generate a Danbooru-tag prompt for the Illustrious XL anime model from a reference image, "
        "tagging what is visible so the image can be recreated.\n\n"
        "OUTPUT: reply with the tag string ONLY — comma-separated, all lowercase, no preamble, no "
        "'this image shows', no commentary. Tag only what is actually visible; do not invent details.\n\n"
        "Order: lead with quality tags (masterpiece, best quality, amazing quality, very aesthetic, absurdres, "
        "newest), then subject count (1girl, 1boy...), then the rating tag, then character name only if certain, "
        "then hair, eyes, expression, each clothing item or state of undress, pose/body position, "
        "setting/background, and finally lighting and visible effects.\n\n"
        "Rating tag — match the content: 'general' (safe), 'sensitive' (suggestive), 'nsfw' (nudity/sexual), "
        "'explicit' (hardcore). Do not default to 'general' when the image is adult.\n\n"
        "Tag rules: real Danbooru tags only. No Pony score_ tags. No 4k/8k/HDR/photorealistic. All lowercase, "
        "comma separated.\n\n"
        "Content range: everyday through fully explicit. When the image is nude or sexual, set the correct rating "
        "tag and tag the nudity, anatomy, and acts with accurate Danbooru tags — do not omit or soften them. "
        "Absolute rule: treat all subjects as adults; never tag anyone who is or appears underage with sexual, "
        "nude, or suggestive tags, and if age is uncertain, do not sexualize them."
    ),
    "LTX Video 2.3 (Vision)": (
        "You write prompts for LTX Video 2.3 (image-to-video with synchronized audio). "
        "The attached image is the FIRST FRAME. Write what happens next.\n\n"
        "OUTPUT: reply with the prompt ONLY — one flowing paragraph, present tense, no preamble, "
        "no image description, no lists, no commentary.\n\n"
        "CRITICAL — this is image-to-video, so do NOT re-describe what's already visible (appearance, "
        "setting, and lighting are given by the image). Describe MOTION and the transition from stillness, "
        "using explicit VERBS: what the subject does, how the environment moves, and what the camera does. "
        "Add one clear camera move and an audio layer (ambient tone plus any dialogue in quotes, broken into "
        "short lines with physical acting beats). Close by stating how the shot ends and its pacing.\n\n"
        "Rules: emotion through physical action ('her jaw tightens'), never labels; a single continuous shot, "
        "no cuts; no numeric specs like degrees or speeds. If the subject is clearly mid-speech you may add one "
        "short fitting line in quotes — but do not deliberate; if unsure, leave dialogue out.\n\n"
        "Content range: everyday through fully explicit. When the image is nude or sexual, describe the action "
        "plainly and completely — state of undress, anatomy, and sexual motion in direct terms; never euphemize, "
        "omit, or sanitize. Absolute rule: treat all subjects as adults; never sexualize anyone who is or appears "
        "underage, and if age is uncertain, do not sexualize them."
    ),
    "Wan 2.2 Video (Vision)": (
        "You write prompts for the Wan 2.2 video model in image-to-video mode. "
        "The attached image is the FIRST FRAME. Write the motion that plays out from it.\n\n"
        "OUTPUT: reply with the prompt ONLY — one flowing paragraph, present tense, no preamble, no image "
        "description, no lists, no commentary. Roughly 80-120 words.\n\n"
        "CRITICAL — the image already sets subject, scene, and style, so do NOT re-describe them. Focus on "
        "MOTION and camera: explicit action verbs with amplitude and speed ('rushes', 'drifts slowly'), "
        "separated foreground and background motion for depth, and one clear camera movement (dolly-in, "
        "tracking, crane up, handheld). Add a color-grade/lighting cue only as it evolves over the shot.\n\n"
        "Rules: emotion through physical action, never labels; a single continuous shot, no cuts; no numeric "
        "specs like degrees or speeds.\n\n"
        "Content range: everyday through fully explicit. When the image is nude or sexual, describe the action "
        "plainly and completely — state of undress, anatomy, and sexual motion in direct terms; never euphemize, "
        "omit, or sanitize. Absolute rule: treat all subjects as adults; never sexualize anyone who is or appears "
        "underage, and if age is uncertain, do not sexualize them."
    ),
    "MiniMax H3 Base (Vision)": (
        "You write prompts for MiniMax H3 Image to Video + Audio mode (I2VA). The attached image is <Picture 1>, the fully "
        "referenced first frame of [Shot 1] at 0.00 seconds. The user's text is a directing instruction for what "
        "happens from that frame, not a description of the image and not a requested final-frame description.\n\n"
        "OUTPUT: reply with ONLY the following instruction and three fields in this exact order:\n"
        "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully "
        "referenced.\n\n"
        "integrated_multimodal_description: [Shot 1] ...\n"
        "overall_soundscape: ...\n"
        "non_diegetic_music: ...\n\n"
        "Start [Shot 1] by anchoring the visible style, subjects, composition, clothing, colors, important props, "
        "environment, and spatial relationships from <Picture 1>. Preserve character identity, facial appearance, "
        "hairstyle, costume, colors, environment, and visual style while allowing the requested movement, "
        "expressions, body-position changes, and pose changes. Continue as coherent physical progression: initial "
        "state, action beginning, development, consequence or reaction, and ending state. Use natural pacing words "
        "instead of numeric choreography. Keep all actions under [Shot 1] when the camera remains continuous. Add "
        "[Shot 2] or later labels only for actual cuts or clearly separate shots, optionally with a useful cut time "
        "such as '[Shot 2] At 00:04.500, the camera cuts to...'. Do not divide the clip into micro-timestamp "
        "ranges or account for every second. Exact timing is only for reference alignment, actual cuts, or "
        "important synchronized events.\n\n"
        "Give vocal sources stable IDs such as (S1), and preserve user-provided speech inside "
        "<d>[Language] ...</d> at the matching action beat inside integrated_multimodal_description. Do not repeat "
        "or paraphrase dialogue in overall_soundscape; use that section for environmental and physical diegetic "
        "audio such as ambience, footsteps, wind, cloth movement, breathing, laughter, impacts, and effects. Set "
        "non_diegetic_music to exactly N/A unless music is requested or clearly called for. Describe what should "
        "happen and do not invent unrelated negative restrictions. Keep the result concise for H3."
    ),
    "MiniMax H3 Frame-to-Frame (Vision)": (
        "You write prompts for MiniMax H3 First/Last Image to Video + Audio mode (FL2VA). The attached image is "
        "<Picture 1>, the first-image reference. <Picture 2> is the last-image reference identified by the user's "
        "workflow or instruction. Preserve both tags exactly and keep their numbering in actual input or connection "
        "order. The user's text directs the motion between them and is not a description of <Picture 2>.\n\n"
        "OUTPUT: begin exactly with 'How the reference pictures align with the target video — <Picture 1> "
        "(from [Shot 1]) aligns with the 0.00-second mark of the target video; <Picture 2> (from [Shot N]) aligns "
        "with the S.SS-second mark of the target video.' Replace N and S.SS with the actual final shot and "
        "requested duration. Then write ONLY these fields in this exact order:\n"
        "integrated_multimodal_description: [Shot 1] ...\n"
        "overall_soundscape: ...\n"
        "non_diegetic_music: ...\n\n"
        "Anchor <Picture 1>'s visible identity, style, composition, clothing, colors, props, environment, and "
        "spatial relationships, then write a continuous physical progression that reaches <Picture 2> at the final "
        "instant. Preserve continuity without freezing pose, expression, body position, or movement. Keep one "
        "continuous [Shot 1] unless an actual camera cut occurs. Do not create shots for action changes and do not "
        "use micro-timestamp ranges; reserve exact timing for first/last-image alignment, real cuts, or important "
        "synchronized events. Keep requested speech in integrated_multimodal_description. Use overall_soundscape "
        "only for environmental and physical diegetic audio without repeating dialogue. Set "
        "non_diegetic_music to N/A unless music is requested or clearly called for, and do not invent unrelated "
        "negative restrictions."
    ),
    "MiniMax H3 Last-Frame (Vision)": (
        "You write prompts for MiniMax H3 Last Image to Video + Audio mode (L2VA). The attached image is "
        "<Picture 1>, the supplied last image that the generated video must reach at the final instant. Treat the "
        "user's text as direction for the action leading into that image, not as a description of a first frame.\n\n"
        "OUTPUT: begin exactly with 'How the reference pictures align with the target video — <Picture 1> "
        "(from [Shot N]) aligns with the S.SS-second mark of the target video.' Replace N and S.SS with the actual "
        "final shot and requested duration. Then write ONLY these fields in this exact order:\n"
        "integrated_multimodal_description: [Shot 1] ...\n"
        "overall_soundscape: ...\n"
        "non_diegetic_music: ...\n\n"
        "Preserve the exact <Picture 1> tag and its actual input or connection numbering.\n\n"
        "Write a coherent physical progression under [Shot 1] that begins in a plausible earlier state and "
        "culminates in <Picture 1>. Preserve character identity, facial appearance, hairstyle, costume, colors, "
        "important props, environment, visual style, and relevant spatial relationships while allowing pose, "
        "expression, body position, and movement to change. Add later shots only for actual cuts. Avoid "
        "micro-timestamps and use exact timing only for last-image alignment, a real cut, or an important "
        "synchronized event. Keep dialogue at its action beat in integrated_multimodal_description; keep only "
        "environmental and physical diegetic audio in overall_soundscape. Set non_diegetic_music to N/A unless "
        "music is requested or clearly called for. Do not invent unrelated negative restrictions."
    ),
    "MiniMax H3 Reference (Vision)": (
        "You write prompts for MiniMax H3 Reference to Video + Audio mode (Ref2VA). Treat the attached image as "
        "<Picture 1>, the first connected reference asset, not automatically as the first frame of the target "
        "video. Treat the user's text only as a directing instruction for how the reference is used.\n\n"
        "OUTPUT: reply with ONLY these six sections in this exact order and with these exact names:\n"
        "subject_definitions: ...\n"
        "summary: ...\n"
        "retention_analysis: ...\n"
        "detailed_description: ...\n"
        "overall_soundscape: ...\n"
        "non_diegetic_music: ...\n\n"
        "Assign stable <Subject N>, <Picture N>, "
        "<Video N>, and <Audio N> labels only where their roles apply. Define reusable visible content as subjects; "
        "use picture labels only for concrete frame or storyboard anchors. Preserve <Picture N>, <Video N>, and "
        "<Audio N> exactly, number each modality independently by actual input or connection order, and never "
        "change a label's meaning or replace a tag with a descriptive name.\n\n"
        "Begin summary with the applicable task types in square brackets. Give every label a retention_analysis "
        "line using the official fixed relationship marker appropriate to visible or audio content. Before [Shot 1] "
        "in detailed_description, establish the overall style, then write the target video shot by shot with exact "
        "reference points, composition, subjects, environment, lighting, continuous physical action, camera, and "
        "synchronized sound. Keep one shot for continuous camera action and create later [Shot N] labels only for "
        "real cuts or separate shots. Avoid micro-timestamp ranges; use exact times only for reference alignment, "
        "actual cuts, or important synchronized events. Preserve identity, appearance, costume, colors, environment, "
        "style, and relevant spatial continuity while allowing requested pose, expression, body-position, and "
        "movement changes. Keep requested dialogue at its action beat in detailed_description and do not restate "
        "it in overall_soundscape. Set non_diegetic_music to N/A unless music is requested or clearly called for. "
        "Write all structural prose in English while preserving dialogue, lyrics, and visible text in their "
        "original language. Do not invent unseen conflicting details or unrelated negative restrictions."
    ),
    "Custom (Vision)": (
        "You are a visual analyst and prompt engineer for AI image and video generation models. "
        "You are looking at a reference image. Generate a detailed generation prompt that would recreate or extend it. "
        "\n\nRULES:"
        "\n- ONLY describe what is visually observable — no assumptions, no abstract feelings"
        "\n- Describe: subject, visible pose/action, lighting, composition, color palette, style, atmosphere"
        "\n- Be specific and concrete — avoid vague adjectives like 'beautiful' or 'stunning'"
        "\n- NO negative language, NO 'avoid' instructions"
        "\n\nReturn ONLY the prompt. No explanations, no preamble.\n\n"
        "Content range: everyday through fully explicit. When the image is nude or sexual, describe it plainly "
        "and completely — state of undress, anatomy, and any sexual activity in direct terms; never euphemize, "
        "omit, or sanitize. Absolute rule: treat all subjects as adults; never sexualize anyone who is or appears "
        "underage, and if age is uncertain, do not sexualize them."
    ),
}

PROMPT_ENHANCER_DEFAULT_URLS: Dict[str, str] = {
    "Ollama": "http://localhost:11434/v1",
    "OpenRouter": "https://openrouter.ai/api/v1",
    "NanoGPT": "https://nano-gpt.com/api/v1",
    "Kobold": "http://localhost:5001/v1",
}

# Keep this in sync with ComfyUI's built-in CLIPLoader. The selected type is
# important because the same Qwen/Gemma weights can use different wrappers.
PROMPT_ENHANCER_LOCAL_CLIP_TYPES = [
    "stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi",
    "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma",
    "ace", "omnigen2", "qwen_image", "hunyuan_image", "flux2", "ovis",
    "longcat_image", "cogvideox", "lens", "pixeldit", "ideogram4", "boogu",
    "krea2", "joyimage", "mage", "minimax",
]
PROMPT_ENHANCER_CONNECTED_CLIP = "(use connected CLIP input)"


class PromptEnhancer:
    CATEGORY = "utils/prompt"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "enhance"
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute — LLM output is non-deterministic and ComfyUI would otherwise
        # cache and skip the node when inputs look identical.
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls):
        # Vision variants are auto-selected at runtime when an IMAGE is connected —
        # don't expose them in the dropdown.
        preset_names = [k for k in PROMPT_ENHANCER_PRESETS if not k.endswith("(Vision)")]
        local_encoders = list(folder_paths.get_filename_list("text_encoders"))
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "backend": (["Ollama", "OpenRouter", "NanoGPT", "Kobold", "ComfyUI Local"],),
                "api_url": ("STRING", {"default": "http://localhost:11434/v1", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "Active API key (auto-fills from saved keys below)"}),
                "openrouter_key": ("STRING", {"default": "", "multiline": False, "placeholder": "OpenRouter API key — saved permanently"}),
                "nanogpt_key": ("STRING", {"default": "", "multiline": False, "placeholder": "NanoGPT API key — saved permanently"}),
                "model_name": ("STRING", {"default": "llama3", "multiline": False, "placeholder": "e.g. llama3, gpt-4o, mistral"}),
                "target_model": (preset_names,),
                "system_prompt": ("STRING", {"default": PROMPT_ENHANCER_PRESETS["Flux / Z-Image"], "multiline": True}),
                "manual_addons": ("STRING", {"default": "", "multiline": True, "placeholder": "Extra instructions or context (optional)..."}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "thinking_mode": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                # Append new serialized widgets after all legacy widgets so saved
                # workflows keep their positional widget values intact.
                "local_text_encoder": ([PROMPT_ENHANCER_CONNECTED_CLIP, *local_encoders], {
                    "tooltip": "Generative checkpoint in models/text_encoders. Connect CLIP below to reuse an already-loaded model and save VRAM."
                }),
                "local_clip_type": (PROMPT_ENHANCER_LOCAL_CLIP_TYPES, {
                    "default": "flux2",
                    "tooltip": "Must match the ComfyUI CLIPLoader type intended for this checkpoint."
                }),
            },
            "optional": {
                "prompt": ("STRING", {"forceInput": True}),
                "image": ("IMAGE",),
                "clip": ("CLIP", {"tooltip": "Reuse an existing CLIP/text encoder instead of loading local_text_encoder again."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def enhance(
        self,
        enabled: bool = True,
        backend: str = "Ollama",
        api_url: str = "http://localhost:11434/v1",
        api_key: str = "",
        openrouter_key: str = "",
        nanogpt_key: str = "",
        model_name: str = "llama3",
        local_text_encoder: str = PROMPT_ENHANCER_CONNECTED_CLIP,
        local_clip_type: str = "flux2",
        target_model: str = "Flux / Z-Image",
        system_prompt: str = "",
        manual_addons: str = "",
        max_tokens: int = 512,
        thinking_mode: bool = False,
        seed: int = 0,
        prompt: str = "",
        image=None,
        clip=None,
        unique_id: str | None = None,
    ):
        def _out(text: str):
            # Record the ACTUAL output so the gallery's prompt resolver can show
            # the final enhanced prompt instead of the pre-enhancement input.
            if unique_id is not None:
                PROMPT_ENHANCER_OUTPUTS[str(unique_id)] = text or ""
            return (text,)

        selected_model = local_text_encoder if backend == "ComfyUI Local" else model_name
        print(f"[PromptEnhancer] Got prompt={prompt!r:.60} enabled={enabled} backend={backend} model={selected_model} thinking={thinking_mode} seed={seed} has_image={image is not None}", flush=True)

        if not enabled or (not prompt.strip() and image is None):
            print(f"[PromptEnhancer] Bypassing — enabled={enabled} prompt_empty={not prompt.strip()}", flush=True)
            return _out(prompt)

        # Use saved key for OpenRouter/NanoGPT if api_key is blank
        active_key = api_key.strip()
        if not active_key:
            if backend == "OpenRouter" and openrouter_key.strip():
                active_key = openrouter_key.strip()
            elif backend == "NanoGPT" and nanogpt_key.strip():
                active_key = nanogpt_key.strip()

        # Build user message — support image input for VL models
        import io as _io
        import base64 as _b64

        # System prompt resolution. Text mode = the target model's text preset (or the
        # user's edited system_prompt widget). When an IMAGE is connected we swap in the
        # matching "(Vision)" preset so the model is actually told to read the image and
        # emit a prompt in the target format — UNLESS the user hand-edited the system
        # prompt, in which case we respect their edit.
        text_default = PROMPT_ENHANCER_PRESETS.get(target_model, "")
        sys_content = system_prompt.strip() or text_default
        user_edited_system = bool(system_prompt.strip()) and system_prompt.strip() != text_default.strip()

        user_content: Any
        if image is not None:
            try:
                import numpy as _np
                from PIL import Image as _PILImage

                frame_indices = [0]

                image_parts = []
                encoded_sizes = []
                for frame_index in frame_indices:
                    img_array = (_np.clip(image[frame_index].cpu().numpy(), 0, 1) * 255).astype(_np.uint8)
                    pil_img = scale_for_provider(_PILImage.fromarray(img_array), backend)
                    buf = _io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    img_b64 = _b64.b64encode(buf.getvalue()).decode("utf-8")
                    image_parts.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                    )
                    encoded_sizes.append(f"{pil_img.size[0]}x{pil_img.size[1]}")

                # Auto-select the vision system prompt for this target model
                # (falls back to the text preset if no vision variant exists).
                if not user_edited_system:
                    sys_content = PROMPT_ENHANCER_PRESETS.get(f"{target_model} (Vision)") or text_default

                # The user's prompt is a DIRECTIVE, not background noise — keep it
                # primary. Text goes BEFORE the image (more reliable across VL backends).
                if target_model.startswith("MiniMax H3"):
                    if target_model == "MiniMax H3 Last-Frame":
                        image_role = "The attached image is <Picture 1>, the last image of the MiniMax H3 L2VA video."
                    elif target_model == "MiniMax H3 Frame-to-Frame":
                        image_role = "The attached image is <Picture 1>, the first-image reference of the MiniMax H3 FL2VA video."
                    elif target_model == "MiniMax H3 Reference":
                        image_role = "The attached image is <Picture 1>, the first connected reference asset for MiniMax H3 Ref2VA."
                    else:
                        image_role = "The attached image is <Picture 1>, the initial image and first frame of the MiniMax H3 I2VA video."
                    if prompt.strip():
                        text_part = (
                            f"{image_role} Treat the following text only as the directing instruction for the "
                            f"selected generation mode, following the system instructions: {prompt.strip()}"
                        )
                    else:
                        text_part = f"{image_role} Generate a complete prompt following the system instructions."
                elif prompt.strip():
                    text_part = (
                        "Use the attached image as the visual reference and follow the "
                        f"system instructions. Apply this user direction: {prompt.strip()}"
                    )
                else:
                    text_part = "Generate a prompt from the attached image, following the system instructions."
                if manual_addons.strip():
                    text_part += f"\n\nAdditional instructions: {manual_addons.strip()}"

                user_content = [{"type": "text", "text": text_part}, *image_parts]
                _vmode = "user-edited, kept" if user_edited_system else "auto-selected"
                print(
                    f"[PromptEnhancer] Attached {len(image_parts)} vision frame(s) "
                    f"({', '.join(encoded_sizes)}) — vision system prompt {_vmode}",
                    flush=True,
                )
            except Exception as img_err:
                print(f"[PromptEnhancer] Image encode failed: {img_err} — falling back to text only", flush=True)
                user_content = prompt.strip()
                if manual_addons.strip():
                    user_content += f"\n\nAdditional instructions: {manual_addons.strip()}"
        else:
            user_content = prompt.strip()
            if manual_addons.strip():
                user_content += f"\n\nAdditional instructions: {manual_addons.strip()}"

        # Reasoning control that works on ALL backends (not just OpenRouter's `thinking`
        # param). Qwen3 / many reasoning models honor a /no_think (or /think) soft switch
        # placed in the prompt. Without this, models like Qwen3 think on every call and
        # burn the token budget before producing an answer — and the thinking_mode
        # checkbox does nothing on Ollama/Kobold/NanoGPT. Harmless text for models that
        # don't recognize it.
        if not thinking_mode:
            sys_content = f"{sys_content}\n\n/no_think".strip()
        else:
            sys_content = f"{sys_content}\n\n/think".strip()

        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_content},
        ]

        # All backends use OpenAI-compatible /chat/completions
        base_url = api_url.rstrip("/")
        endpoint = f"{base_url}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if active_key:
            headers["Authorization"] = f"Bearer {active_key}"
            if backend == "NanoGPT":
                # NanoGPT supports either header. Send both so authentication still
                # works through local proxies that strip the Authorization header.
                headers["X-API-Key"] = active_key
        # OpenRouter requires these headers or returns 400
        if backend == "OpenRouter":
            headers["HTTP-Referer"] = "https://github.com/lokitsar/ComfyUI-Workflow-Gallery"
            headers["X-Title"] = "ComfyUI Prompt Enhancer"

        import urllib.request as _urlreq
        import urllib.error as _urlerr
        import json as _json

        # Detect Ollama native base (strip /v1 suffix if present)
        ollama_base = base_url
        if ollama_base.endswith("/v1"):
            ollama_base = ollama_base[:-3]

        def unload_ollama_model():
            """Release a local Ollama model so it cannot starve ComfyUI of VRAM."""
            if backend != "Ollama" or not model_name.strip():
                return
            ep = f"{ollama_base}/api/generate"
            p = {"model": model_name.strip(), "keep_alive": 0, "stream": False}
            try:
                req = _urlreq.Request(
                    ep,
                    data=_json.dumps(p).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with _urlreq.urlopen(req, timeout=30) as resp:
                    resp.read()
                print(
                    f"[PromptEnhancer] Unloaded Ollama model {model_name.strip()!r} to free GPU memory",
                    flush=True,
                )
            except Exception as unload_err:
                # Prompt enhancement should still succeed if an older/non-local Ollama
                # endpoint cannot service the explicit unload request.
                print(
                    f"[PromptEnhancer] WARNING: could not unload Ollama model: {unload_err}",
                    flush=True,
                )

        def _msg_text(msg):
            """Pull assistant text from an OpenAI-style message, tolerating reasoning
            models that leave `content` null and stash text in a reasoning field, and
            APIs that return content as a list of parts. Never returns None."""
            if not isinstance(msg, dict):
                return ""
            content = msg.get("content")
            if isinstance(content, list):  # content-as-parts (some providers)
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            if not content:  # null/empty -> fall back to reasoning fields
                content = msg.get("reasoning_content") or msg.get("reasoning") or ""
            return (content or "").strip()

        def _http_error_detail(error):
            try:
                body = error.read().decode("utf-8", errors="replace")
                parsed = _json.loads(body)
                detail = parsed.get("error", parsed)
                if isinstance(detail, dict):
                    return str(detail.get("message") or detail.get("code") or "").strip()
                return str(detail).strip()
            except Exception:
                return ""

        # Stop sequences: cut generation at the reflection cue words reasoning models
        # use to second-guess themselves ("Refining…", "Revised…", "Drafting…"). These
        # never occur inside a real prompt. (Do NOT stop on "\n\n": some runs open with a
        # reasoning preamble, and "\n\n" would cut the answer off before it even starts.)
        _stop = ["\nRefining", "\nRevised", "\nDrafting", "\nNote:"]

        def try_openai_compat():
            ep = f"{base_url}/chat/completions"
            p = {
                "model": model_name.strip(),
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stop": _stop,
            }
            # seed: 0 = omit (random), non-zero = pass for reproducibility
            if seed != 0:
                p["seed"] = seed
            # thinking mode control — OpenRouter only; Ollama/Kobold/NanoGPT reject this param
            if backend == "OpenRouter":
                p["thinking"] = {"type": "enabled", "budget_tokens": 2048} if thinking_mode else {"type": "disabled"}
            print(f"[PromptEnhancer] Trying OpenAI-compat at {ep} (backend={backend} thinking={thinking_mode} seed={seed})", flush=True)
            req_data = _json.dumps(p).encode("utf-8")
            req = _urlreq.Request(ep, data=req_data, headers=headers, method="POST")
            with _urlreq.urlopen(req, timeout=120) as resp:
                result = _json.loads(resp.read().decode("utf-8"))
            return _msg_text(result["choices"][0]["message"])

        def try_ollama_native():
            ep = f"{ollama_base}/api/chat"
            opts = {"temperature": 0.7, "num_predict": max_tokens, "stop": _stop}
            if seed != 0:
                opts["seed"] = seed
            p = {
                "model": model_name.strip(),
                "messages": messages,
                "stream": False,
                "keep_alive": 0,
                "options": opts,
            }
            print(f"[PromptEnhancer] Trying Ollama native at {ep} (seed={seed})", flush=True)
            req_data = _json.dumps(p).encode("utf-8")
            req = _urlreq.Request(ep, data=req_data, headers={"Content-Type": "application/json"}, method="POST")
            with _urlreq.urlopen(req, timeout=120) as resp:
                result = _json.loads(resp.read().decode("utf-8"))
            return _msg_text(result.get("message"))

        def strip_thinking(text: str) -> str:
            """Strip thinking/reasoning blocks that models emit before the real answer.
            Handles XML tags AND the markdown-scaffold style some reasoning models use
            (e.g. '**Image Analysis:**' ... '**Drafting the prompt:**' <answer>)."""
            import re as _re2
            # XML-style thinking tags (Qwen3, DeepSeek, some OpenRouter models)
            text = _re2.sub(r"<think>.*?</think>", "", text, flags=_re2.DOTALL)
            text = _re2.sub(r"<thinking>.*?</thinking>", "", text, flags=_re2.DOTALL)
            text = _re2.sub(r"<reasoning>.*?</reasoning>", "", text, flags=_re2.DOTALL)
            text = _re2.sub(r"<reflection>.*?</reflection>", "", text, flags=_re2.DOTALL)
            text = _re2.sub(r"<thought>.*?</thought>", "", text, flags=_re2.DOTALL)
            # "Thinking:" or "Reasoning:" header blocks followed by blank line
            text = _re2.sub(r"(?i)^(thinking|reasoning|reflection|thought process):.*?\n\n", "", text, flags=_re2.DOTALL)
            # Strip any remaining unclosed opening tags
            text = _re2.sub(r"<(think|thinking|reasoning|reflection)[^>]*>.*$", "", text, flags=_re2.DOTALL)

            # Markdown-scaffold reasoning: if a final-answer marker is present, keep only
            # what follows the LAST one. Markers are reasoning-scaffold phrases that
            # essentially never appear inside a real generation prompt.
            markers = [
                r"drafting the (?:final )?(?:prompt|dialogue)",
                r"(?:let'?s |now )?refin(?:e|ing) the (?:prompt|structure)",
                r"final prompt",
                r"final answer",
                r"here(?:'s| is) the (?:final |enhanced )?prompt",
                r"enhanced prompt",
                r"the prompt is",
            ]
            marker_re = _re2.compile(r"(?im)^\s*[*_#> \t]*(?:" + "|".join(markers) + r")\s*[:*_]*\s*$\n?", _re2.MULTILINE)
            last = None
            for m in marker_re.finditer(text):
                last = m
            if last is not None:
                candidate = text[last.end():].strip()
                if candidate:  # never strip everything to nothing
                    text = candidate

            # Trailing self-reflection: some models emit a clean answer, then critique
            # it and start re-drafting (often truncated). Keep everything BEFORE the
            # first reflection cue. Newline-anchored so it can't fire inside a paragraph.
            trailing = _re2.search(
                r"(?im)\n\s*(?:refining\b|revised\b|revised prompt|drafting\b|"
                r"on second thought|let me (?:know|refine|revise)|wait,)",
                text,
            )
            if trailing:
                head = text[:trailing.start()].strip()
                if head:
                    text = head

            # Leading reasoning preamble: some runs open with task commentary
            # ("The user wants…", "I need to…", "Let me…") before the actual prompt.
            # Drop leading lines that look like commentary, stop at the first real line.
            _cue = _re2.compile(
                r"(?i)^(the user\b|the image\b|i need\b|i'?ll\b|i will\b|i'?m going\b|"
                r"let me\b|let'?s\b|okay\b|ok[, ]|first,|looking at\b|"
                r"to (?:write|create|craft|build)\b|here'?s my\b|based on the\b|"
                r"alright\b|sure[,!]|i'?ve analyzed\b)"
            )
            _lines = text.split("\n")
            while len(_lines) > 1 and (not _lines[0].strip() or _cue.match(_lines[0].strip())):
                _lines.pop(0)
            text = "\n".join(_lines).strip()

            # Clean leftover wrapping artifacts (surrounding quotes / stray bold markers)
            text = text.strip().strip("`").strip()
            text = _re2.sub(r'^["\u201c\u201d\']+|["\u201c\u201d\']+$', "", text).strip()
            return text.strip()

        if backend == "ComfyUI Local":
            local_clip = clip
            loaded_here = False
            if local_clip is None:
                if not local_text_encoder or local_text_encoder == PROMPT_ENHANCER_CONNECTED_CLIP:
                    raise RuntimeError(
                        "Prompt Enhancer: choose a local_text_encoder or connect a CLIP input."
                    )
                try:
                    import comfy.sd as _comfy_sd

                    clip_path = folder_paths.get_full_path_or_raise(
                        "text_encoders", local_text_encoder
                    )
                    clip_type = getattr(
                        _comfy_sd.CLIPType,
                        local_clip_type.upper(),
                        _comfy_sd.CLIPType.STABLE_DIFFUSION,
                    )
                    local_clip = _comfy_sd.load_clip(
                        ckpt_paths=[clip_path],
                        embedding_directory=folder_paths.get_folder_paths("embeddings"),
                        clip_type=clip_type,
                    )
                    loaded_here = True
                except Exception as load_err:
                    raise RuntimeError(
                        f"Prompt Enhancer could not load local text encoder "
                        f"{local_text_encoder!r} as {local_clip_type!r}: {load_err}"
                    ) from load_err

            if not callable(getattr(local_clip, "generate", None)) or not callable(
                getattr(local_clip, "decode", None)
            ):
                raise RuntimeError(
                    "Prompt Enhancer: this CLIP/text encoder is conditioning-only and "
                    "does not expose text generation. Use a complete Qwen, Gemma, "
                    "Llama, or other causal-language-model text encoder."
                )

            local_user = prompt.strip()
            if manual_addons.strip():
                local_user += f"\n\nAdditional instructions: {manual_addons.strip()}"
            local_request = (
                f"{sys_content}\n\n"
                f"Input prompt:\n{local_user}\n\n"
                "Return only the enhanced prompt requested above, with no analysis, "
                "heading, quotation marks, or Markdown."
            ).strip()

            try:
                tokenize_kwargs = {"min_length": 1, "thinking": thinking_mode}
                if image is not None:
                    # Compatible VL encoders consume the original tensor directly;
                    # no lossy PNG/base64 round trip is needed for local generation.
                    tokenize_kwargs["image"] = image
                tokens = local_clip.tokenize(local_request, **tokenize_kwargs)
                local_seed = seed if seed != 0 else random.SystemRandom().randrange(1, 2**63)
                generated_ids = local_clip.generate(
                    tokens,
                    do_sample=True,
                    max_length=max_tokens,
                    temperature=0.7,
                    top_k=64,
                    top_p=0.95,
                    min_p=0.05,
                    repetition_penalty=1.05,
                    presence_penalty=0.0,
                    seed=local_seed,
                )
                enhanced = strip_thinking(local_clip.decode(generated_ids))
                if not enhanced:
                    raise RuntimeError("the model returned an empty response")
                if local_clip_type == "minimax":
                    print(
                        "[PromptEnhancer] WARNING: MiniMax H3's Qwen3-VL-32B "
                        "checkpoint is truncated for conditioning; generated text may "
                        "be lower quality than a complete instruct checkpoint.",
                        flush=True,
                    )
                source = "connected CLIP" if clip is not None else local_text_encoder
                print(
                    f"[PromptEnhancer] Enhanced locally with {source!r}: {enhanced[:120]!r}",
                    flush=True,
                )
                return _out(enhanced)
            except Exception as generation_err:
                raise RuntimeError(
                    "Prompt Enhancer local generation failed. The file may be a "
                    "conditioning-only encoder, may use the wrong local_clip_type, or "
                    "may lack the layers needed for text generation. "
                    f"Details: {generation_err}"
                ) from generation_err
            finally:
                # Drop only the reference created by this node. ComfyUI's model manager
                # retains control of device/offload state; connected CLIPs are untouched.
                if loaded_here:
                    del local_clip

        try:
            enhanced = strip_thinking(try_openai_compat())
            if not enhanced:
                raise RuntimeError(
                    "Prompt Enhancer received an empty response. Raise max_tokens or choose a non-thinking model."
                )
            print(f"[PromptEnhancer] Enhanced (OpenAI compat): {enhanced[:120]!r}", flush=True)
            return _out(enhanced)
        except _urlerr.HTTPError as e:
            detail = _http_error_detail(e)
            if e.code == 404 and backend == "Ollama":
                print(f"[PromptEnhancer] OpenAI compat returned 404, trying Ollama native API...", flush=True)
                try:
                    enhanced = strip_thinking(try_ollama_native())
                    if not enhanced:
                        raise RuntimeError("Prompt Enhancer received an empty response from Ollama.")
                    print(f"[PromptEnhancer] Enhanced (Ollama native): {enhanced[:120]!r}", flush=True)
                    return _out(enhanced)
                except Exception as e2:
                    print(f"[PromptEnhancer] Ollama native also failed: {e2}", flush=True)
                    raise RuntimeError(f"Prompt Enhancer could not reach Ollama: {e2}") from e2
            if e.code == 401:
                message = f"Prompt Enhancer: {backend} rejected the API key (401 Unauthorized)."
                if backend == "NanoGPT":
                    message += " Paste a valid, active NanoGPT key into api_key or nanogpt_key."
                if detail:
                    message += f" Provider message: {detail}"
                raise RuntimeError(message) from e
            message = f"Prompt Enhancer request failed: HTTP {e.code} from {backend}."
            if detail:
                message += f" Provider message: {detail}"
            raise RuntimeError(message) from e
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Prompt Enhancer request failed for {backend}: {e}") from e
        finally:
            # The OpenAI-compatible Ollama endpoint does not reliably expose the native
            # keep_alive option, so explicitly unload through /api/generate as well.
            unload_ollama_model()


@routes.get("/prompt_enhancer/presets")
async def prompt_enhancer_presets(request):
    return web.json_response({"presets": PROMPT_ENHANCER_PRESETS, "default_urls": PROMPT_ENHANCER_DEFAULT_URLS})


@routes.post("/prompt_enhancer/models")
async def prompt_enhancer_models(request):
    """Fetch available models for the Prompt Enhancer."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    api_url = str(body.get("api_url", "")).rstrip("/")
    api_key = str(body.get("api_key", "")).strip()
    if not api_url:
        return web.json_response({"ok": False, "error": "No api_url provided"}, status=400)

    import urllib.request as _urlreq
    import urllib.error as _urlerr
    import json as _json

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        if "nano-gpt.com" in api_url.lower():
            headers["X-API-Key"] = api_key

    def parse_models(data):
        if "models" in data:
            return [m.get("name") or m.get("id", "") for m in data["models"] if m.get("name") or m.get("id")]
        elif "data" in data:
            return [m.get("id", "") for m in data["data"] if m.get("id")]
        return []

    # Try OpenAI-compat /models first
    try:
        endpoint = f"{api_url}/models"
        req = _urlreq.Request(endpoint, headers=headers, method="GET")
        with _urlreq.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        models = sorted([m for m in parse_models(data) if m])
        if models:
            return web.json_response({"ok": True, "models": models})
    except _urlerr.HTTPError as e:
        if e.code != 404:
            return web.json_response({"ok": False, "error": f"HTTP {e.code}: {e.reason}"}, status=500)
    except Exception as e:
        pass  # Fall through to Ollama native

    # Try Ollama native /api/tags (for older Ollama versions without /v1)
    try:
        ollama_base = api_url
        if ollama_base.endswith("/v1"):
            ollama_base = ollama_base[:-3]
        tags_endpoint = f"{ollama_base}/api/tags"
        req = _urlreq.Request(tags_endpoint, headers={"Content-Type": "application/json"}, method="GET")
        with _urlreq.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        models = sorted([m.get("name", "") for m in data.get("models", []) if m.get("name")])
        if models:
            return web.json_response({"ok": True, "models": models})
        return web.json_response({"ok": True, "models": models})

    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)


class LoraSidecarSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "filename": ("STRING", {"forceInput": True}),
                "folder": ("STRING", {"default": ""}),
                "existing_file": (["overwrite", "skip"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "save"
    CATEGORY = "Lokitsars/Training"
    OUTPUT_NODE = True

    def save(self, text, filename, folder, existing_file):
        result = DatasetSidecarWriter().write(text, filename, folder, existing_file)
        if result["status"] == "skipped":
            return (f"Skipped existing: {result['path']}",)
        return (result["path"],)


NODE_CLASS_MAPPINGS = {
    "WorkflowGallery": WorkflowGallery,
    "PromptLibrary": PromptLibrary,
    "PromptEnhancer": PromptEnhancer,
    "LoraSidecarSaver": LoraSidecarSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WorkflowGallery": "Workflow Gallery",
    "PromptLibrary": "Prompt Library",
    "PromptEnhancer": "Prompt Enhancer",
    "LoraSidecarSaver": "LoRA Sidecar Saver",
}
