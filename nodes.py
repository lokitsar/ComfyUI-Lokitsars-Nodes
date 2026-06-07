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
_YAML_CACHE: Dict[str, List[str]] = {}

def _read_wildcard_lines_yaml(path: Path, name_parts: List[str]) -> List[str]:
    """Read lines from a .yaml wildcard file with a thread timeout to prevent
    blocking ComfyUI's main thread on large or malformed files.
    Results are cached per file path so repeated wildcard use is fast."""
    import concurrent.futures as _cf

    cache_key = f"{path}|{'/'.join(name_parts)}"
    if cache_key in _YAML_CACHE:
        return _YAML_CACHE[cache_key]

    def _parse():
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore"))
        except ImportError:
            # PyYAML not available — minimal line parser for simple list format
            lines = []
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                stripped = line.strip()
                if stripped.startswith("- "):
                    lines.append(stripped[2:].strip())
            return lines
        except Exception as e:
            print(f"[PromptLibrary] YAML parse error in {path}: {e}", flush=True)
            return []

        # Navigate nested keys: __colors/dark__ -> colors.yaml key "dark"
        for key in name_parts:
            if isinstance(data, dict) and key in data:
                data = data[key]
            elif isinstance(data, dict):
                key_lower = key.lower()
                match = next((v for k, v in data.items() if k.lower() == key_lower), None)
                if match is not None:
                    data = match
                else:
                    break

        # Flatten to list of strings
        if isinstance(data, list):
            return [str(item).strip() for item in data if item]
        elif isinstance(data, dict):
            result = []
            for v in data.values():
                if isinstance(v, list):
                    result.extend(str(i).strip() for i in v if i)
                elif v:
                    result.append(str(v).strip())
            return result
        elif isinstance(data, str):
            return [data.strip()]
        return []

    # Run the parse in a thread with a 5-second timeout.
    # If it takes longer (huge file, filesystem stall) we skip it rather than freeze.
    try:
        with _cf.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_parse)
            lines = future.result(timeout=5.0)
    except _cf.TimeoutError:
        print(f"[PromptLibrary] YAML load timed out for {path} — skipping. "
              f"Consider converting to .txt for faster loading.", flush=True)
        lines = []
    except Exception as e:
        print(f"[PromptLibrary] YAML load failed for {path}: {e}", flush=True)
        lines = []

    _YAML_CACHE[cache_key] = lines
    return lines


def _find_wildcard_file_in_dirs(name: str, dirs: List[Path]):
    """Find a wildcard file (.txt, .yaml, .yml) by name within given directories.
    Tries exact filename match first (fast), then falls back to case-insensitive rglob.
    YAML files are loaded via _read_wildcard_lines_yaml which uses a thread timeout
    to prevent blocking ComfyUI on large files."""
    name_norm = name.replace("\\", "/")
    name_lower = name_norm.lower()

    for wdir in dirs:
        # Fast exact match — txt preferred, then yaml
        for ext in (".txt", ".yaml", ".yml"):
            candidate = wdir / f"{name_norm}{ext}"
            if candidate.exists():
                return candidate, []

        # Case-insensitive fallback scan
        try:
            for wfile in wdir.rglob("*"):
                if wfile.suffix.lower() not in (".txt", ".yaml", ".yml"):
                    continue
                rel = wfile.relative_to(wdir).with_suffix("").as_posix().lower()
                if rel == name_lower:
                    return wfile, []
                # Support yaml key navigation: __colors/dark__ -> colors.yaml["dark"]
                parts = name_lower.split("/")
                for i in range(len(parts), 0, -1):
                    file_part = "/".join(parts[:i])
                    key_parts = parts[i:]
                    if rel == file_part:
                        return wfile, key_parts
        except Exception as e:
            print(f"[PromptLibrary] Error scanning {wdir}: {e}", flush=True)

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
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    if isinstance(value, (str, int)):
        return str(value)
    return ""


def _iter_child_node_ids(inputs: Dict[str, Any]) -> List[str]:
    child_ids: List[str] = []
    for child_value in inputs.values():
        child_node_id = _get_ref_node_id(child_value)
        if child_node_id:
            child_ids.append(child_node_id)
    return child_ids


def _is_sampler_node(node: Dict[str, Any]) -> bool:
    """Return True if this node looks like a KSampler or equivalent."""
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
    return class_type.startswith("KSampler") or has_sampler_links


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

def _resolve_text_from_ref(prompt_graph: Dict[str, Any], value: Any, visited: set[str] | None = None) -> str:
    node_ref = ""
    if isinstance(value, (list, tuple)) and value:
        node_ref = str(value[0])
    elif isinstance(value, (str, int)):
        node_ref = str(value)

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
                resolved = _resolve_text_from_ref(prompt_graph, field_value, current_visited)
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
    string_field_keys = ["text", "string", "value", "prompt", "output", "result", "wildcard_text", "populated_text"]
    for key in string_field_keys:
        field_value = inputs.get(key)
        if isinstance(field_value, str) and field_value.strip():
            return field_value.strip()

    for child_value in inputs.values():
        text = _resolve_text_from_ref(prompt_graph, child_value, current_visited)
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

        positive = _resolve_text_from_ref(prompt_graph, inputs.get("positive"))
        negative = _resolve_text_from_ref(prompt_graph, inputs.get("negative"))
        if not positive:
            positive = _resolve_text_from_ref(prompt_graph, inputs.get("cond_pos"))
        if not negative:
            negative = _resolve_text_from_ref(prompt_graph, inputs.get("cond_neg"))
        if positive:
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

    def _sort_key(item: dict[str, Any]) -> tuple[int, str]:
        key = item["key"]
        return (0, key) if key.isdigit() else (1, key)

    sampler = sorted(samplers, key=_sort_key)[0]["node"]
    inputs = sampler.get("inputs", {})
    if not isinstance(inputs, dict):
        return "", ""

    positive = _resolve_text_from_ref(prompt_graph, inputs.get("positive"))
    negative = _resolve_text_from_ref(prompt_graph, inputs.get("negative"))

    if not positive:
        positive = _resolve_text_from_ref(prompt_graph, inputs.get("cond_pos"))
    if not negative:
        negative = _resolve_text_from_ref(prompt_graph, inputs.get("cond_neg"))
    return positive, negative


def _extract_prompts_with_fallback(prompt_graph: Any, extra_pnginfo: Any, gallery_node_id: str | None = None) -> tuple[str, str, str]:
    # --- Primary: live workflow graph, walked from our specific gallery node ---
    positive, negative = _extract_prompts(prompt_graph, gallery_node_id)
    if positive:
        return positive, negative, "workflow graph"

    if not isinstance(extra_pnginfo, dict):
        return positive, negative, "unavailable"

    # --- Fallback 1: embedded prompt JSON (also scoped to gallery_node_id) ---
    embedded_prompt = extra_pnginfo.get("prompt")
    if embedded_prompt is not None:
        fallback_positive, fallback_negative = _extract_prompts(embedded_prompt, gallery_node_id)
        if fallback_positive:
            return fallback_positive, fallback_negative, "embedded prompt metadata"
        # If gallery_node_id scoped walk failed, try unscoped on embedded prompt
        # but only as a last resort before the stale workflow fallback.
        fallback_positive, fallback_negative = _extract_prompts(embedded_prompt, None)
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
    """Extract the seed from the sampler node connected to the gallery."""
    if not isinstance(prompt_graph, dict):
        return None

    sampler = _find_relevant_sampler(prompt_graph, gallery_node_id)
    if sampler is None:
        # Fall back to any sampler in the graph
        for node in prompt_graph.values():
            if isinstance(node, dict) and _is_sampler_node(node):
                sampler = node
                break

    if sampler is None:
        return None

    inputs = sampler.get("inputs", {})
    if not isinstance(inputs, dict):
        return None

    # Try common seed field names across different sampler types
    for key in ("seed", "noise_seed", "seed_num", "rand_seed"):
        val = inputs.get(key)
        if isinstance(val, int):
            return val
        if isinstance(val, str) and val.isdigit():
            return int(val)

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
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "backend": (["Ollama", "OpenRouter", "NanoGPT", "Kobold"],),
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
            },
            "optional": {
                "prompt": ("STRING", {"forceInput": True}),
                "image": ("IMAGE",),
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
        target_model: str = "Flux / Z-Image",
        system_prompt: str = "",
        manual_addons: str = "",
        max_tokens: int = 512,
        thinking_mode: bool = False,
        seed: int = 0,
        prompt: str = "",
        image=None,
    ):
        print(f"[PromptEnhancer] Got prompt={prompt!r:.60} enabled={enabled} model={model_name} thinking={thinking_mode} seed={seed} has_image={image is not None}", flush=True)

        if not enabled or (not prompt.strip() and image is None):
            print(f"[PromptEnhancer] Bypassing — enabled={enabled} prompt_empty={not prompt.strip()}", flush=True)
            return (prompt,)

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
            # Convert tensor image to base64 PNG
            try:
                import numpy as _np
                from PIL import Image as _PILImage
                img_array = (_np.clip(image[0].cpu().numpy(), 0, 1) * 255).astype(_np.uint8)
                pil_img = _PILImage.fromarray(img_array)
                buf = _io.BytesIO()
                pil_img.save(buf, format="PNG")
                img_b64 = _b64.b64encode(buf.getvalue()).decode("utf-8")

                # Auto-select the vision system prompt for this target model
                # (falls back to the text preset if no vision variant exists).
                if not user_edited_system:
                    sys_content = PROMPT_ENHANCER_PRESETS.get(f"{target_model} (Vision)") or text_default

                # The user's prompt is a DIRECTIVE, not background noise — keep it
                # primary. Text goes BEFORE the image (more reliable across VL backends).
                if prompt.strip():
                    text_part = (
                        "Use the attached image as the visual reference and follow the "
                        f"system instructions. Apply this user direction: {prompt.strip()}"
                    )
                else:
                    text_part = "Generate a prompt from the attached image, following the system instructions."
                if manual_addons.strip():
                    text_part += f"\n\nAdditional instructions: {manual_addons.strip()}"

                user_content = [
                    {"type": "text", "text": text_part},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ]
                _vmode = "user-edited, kept" if user_edited_system else "auto-selected"
                print(f"[PromptEnhancer] Image attached ({pil_img.size[0]}x{pil_img.size[1]}) — vision system prompt {_vmode}", flush=True)
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

        try:
            enhanced = strip_thinking(try_openai_compat())
            if not enhanced:
                print(f"[PromptEnhancer] WARNING: LLM returned no usable text (content was empty/null). "
                      f"If this is a reasoning model, it likely spent the whole max_tokens budget "
                      f"({max_tokens}) on thinking — raise max_tokens or use a non-thinking model. "
                      f"Returning original prompt.", flush=True)
                return (prompt,)
            print(f"[PromptEnhancer] Enhanced (OpenAI compat): {enhanced[:120]!r}", flush=True)
            return (enhanced,)
        except _urlerr.HTTPError as e:
            if e.code == 404 and backend == "Ollama":
                print(f"[PromptEnhancer] OpenAI compat returned 404, trying Ollama native API...", flush=True)
                try:
                    enhanced = strip_thinking(try_ollama_native())
                    if not enhanced:
                        print(f"[PromptEnhancer] WARNING: Ollama native also returned empty", flush=True)
                        return (prompt,)
                    print(f"[PromptEnhancer] Enhanced (Ollama native): {enhanced[:120]!r}", flush=True)
                    return (enhanced,)
                except Exception as e2:
                    print(f"[PromptEnhancer] Ollama native also failed: {e2}", flush=True)
                    return (prompt,)
            print(f"[PromptEnhancer] ERROR: {e} — returning original prompt", flush=True)
            return (prompt,)
        except Exception as e:
            print(f"[PromptEnhancer] ERROR: {e} — returning original prompt", flush=True)
            return (prompt,)


@routes.get("/prompt_enhancer/presets")
async def prompt_enhancer_presets(request):
    return web.json_response({"presets": PROMPT_ENHANCER_PRESETS, "default_urls": PROMPT_ENHANCER_DEFAULT_URLS})


@routes.post("/prompt_enhancer/models")
async def prompt_enhancer_models(request):
    """Fetch available models from a backend's /models endpoint."""
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


NODE_CLASS_MAPPINGS = {
    "WorkflowGallery": WorkflowGallery,
    "PromptLibrary": PromptLibrary,
    "PromptEnhancer": PromptEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WorkflowGallery": "Workflow Gallery",
    "PromptLibrary": "Prompt Library",
    "PromptEnhancer": "Prompt Enhancer",
}
