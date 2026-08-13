"""Test the prompt/seed resolver against synthetic API-format graphs.
Extracts just the resolver functions from nodes.py so we don't need ComfyUI."""
import ast, sys, types

src = open("nodes.py").read()
tree = ast.parse(src)

WANTED = {
    "_get_ref_node_id", "_iter_child_node_ids", "_is_sampler_node",
    "_find_relevant_sampler", "_resolve_text_from_ref", "_extract_prompts",
    "_extract_prompts_with_fallback", "_extract_seed",
    "_extract_prompt_text_from_workflow_node", "_extract_prompts_from_workflow",
}
nodes_out = []
for n in tree.body:
    if isinstance(n, ast.FunctionDef) and n.name in WANTED:
        nodes_out.append(n)

mod = ast.Module(body=nodes_out, type_ignores=[])
ns = {
    "Dict": dict, "Any": object, "List": list,
    "PROMPT_LIBRARY_OUTPUTS": {}, "PROMPT_ENHANCER_OUTPUTS": {},
}
exec(compile(ast.fix_missing_locations(mod), "resolver", "exec"), ns)

_extract_prompts = ns["_extract_prompts"]
_extract_seed = ns["_extract_seed"]
_extract_prompts_with_fallback = ns["_extract_prompts_with_fallback"]
PEO = ns["PROMPT_ENHANCER_OUTPUTS"]
PLO = ns["PROMPT_LIBRARY_OUTPUTS"]

fails = []
def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got={got!r}" + ("" if ok else f" want={want!r}"))
    if not ok: fails.append(name)

# ---- Graph 1: classic KSampler, text nodes feeding CLIPTextEncode ----
g1 = {
    "1": {"class_type": "TextMultiline", "inputs": {"text": "1girl, eyepatch, bikini, car"}},
    "2": {"class_type": "TextMultiline", "inputs": {"text": "lowres, bad anatomy"}},
    "3": {"class_type": "CLIPTextEncode", "inputs": {"text": ["1", 0], "clip": ["9", 1]}},
    "4": {"class_type": "CLIPTextEncode", "inputs": {"text": ["2", 0], "clip": ["9", 1]}},
    "5": {"class_type": "KSampler", "inputs": {"seed": 12345, "steps": 30, "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["6", 0], "model": ["9", 0]}},
    "6": {"class_type": "EmptyLatentImage", "inputs": {"width": 1216, "height": 832}},
    "7": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["9", 2]}},
    "8": {"class_type": "WorkflowGallery", "inputs": {"images": ["7", 0], "enabled": True}},
    "9": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x.safetensors"}},
}
p, n = _extract_prompts(g1, "8")
check("g1 positive", p, "1girl, eyepatch, bikini, car")
check("g1 negative", n, "lowres, bad anatomy")
check("g1 seed", _extract_seed(g1, "8"), 12345)

# ---- Graph 2: SamplerCustomAdvanced + BasicGuider (Flux) + ZeroOut-style no-negative ----
g2 = {
    "1": {"class_type": "TextMultiline", "inputs": {"text": "cinematic photo of a red fox"}},
    "3": {"class_type": "CLIPTextEncode", "inputs": {"text": ["1", 0], "clip": ["20", 1]}},
    "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": 777}},
    "11": {"class_type": "BasicGuider", "inputs": {"model": ["20", 0], "conditioning": ["3", 0]}},
    "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "13": {"class_type": "BasicScheduler", "inputs": {"model": ["20", 0], "steps": 20}},
    "14": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["10", 0], "guider": ["11", 0], "sampler": ["12", 0], "sigmas": ["13", 0], "latent_image": ["6", 0]}},
    "6": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024}},
    "7": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["20", 2]}},
    "8": {"class_type": "WorkflowGallery", "inputs": {"images": ["7", 0]}},
    "20": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "flux.safetensors"}},
}
p, n = _extract_prompts(g2, "8")
check("g2 positive (guider)", p, "cinematic photo of a red fox")
check("g2 negative (none)", n, "")
check("g2 seed (RandomNoise)", _extract_seed(g2, "8"), 777)

# ---- Graph 3: the bleed scenario — negative goes through a node carrying BOTH ----
g3 = {
    "1": {"class_type": "TextMultiline", "inputs": {"text": "A photo of a white man"}},
    "2": {"class_type": "TextMultiline", "inputs": {"text": "ugly, blurry"}},
    "3": {"class_type": "CLIPTextEncode", "inputs": {"text": ["1", 0], "clip": ["9", 1]}},
    "4": {"class_type": "CLIPTextEncode", "inputs": {"text": ["2", 0], "clip": ["9", 1]}},
    # ControlNetApplyAdvanced passes BOTH through; old walker followed `positive` first when resolving negative
    "15": {"class_type": "ControlNetApplyAdvanced", "inputs": {"positive": ["3", 0], "negative": ["4", 0], "control_net": ["16", 0], "image": ["17", 0], "strength": 1.0}},
    "16": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "cn.safetensors"}},
    "17": {"class_type": "LoadImage", "inputs": {"image": "ref.png"}},
    "5": {"class_type": "KSampler", "inputs": {"seed": 42, "positive": ["15", 0], "negative": ["15", 1], "latent_image": ["6", 0], "model": ["9", 0]}},
    "6": {"class_type": "EmptyLatentImage", "inputs": {}},
    "7": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["9", 2]}},
    "8": {"class_type": "WorkflowGallery", "inputs": {"images": ["7", 0]}},
    "9": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x.safetensors"}},
}
p, n = _extract_prompts(g3, "8")
check("g3 positive (through CN)", p, "A photo of a white man")
check("g3 negative NO BLEED", n, "ugly, blurry")

# ---- Graph 4: multi-pipeline workflow, gallery on pipeline B; old code returned pipeline A ----
g4 = {
    # Pipeline A (lowest node ids — old global fallback picked this)
    "1": {"class_type": "TextMultiline", "inputs": {"text": "A photo of a white man"}},
    "3": {"class_type": "CLIPTextEncode", "inputs": {"text": ["1", 0], "clip": ["9", 1]}},
    "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "bad hands", "clip": ["9", 1]}},
    "5": {"class_type": "KSampler", "inputs": {"seed": 111, "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["6", 0]}},
    "6": {"class_type": "EmptyLatentImage", "inputs": {}},
    "9": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "a.safetensors"}},
    # Pipeline B (feeds the gallery)
    "21": {"class_type": "ImpactWildcardEncode", "inputs": {"wildcard_text": "__styles__ woman by car", "populated_text": "editorial photo, woman leaning on car, eyepatch", "clip": ["29", 1], "seed": 999}},
    "24": {"class_type": "CLIPTextEncode", "inputs": {"text": "watermark, text", "clip": ["29", 1]}},
    "25": {"class_type": "KSampler", "inputs": {"seed": 555, "positive": ["21", 0], "negative": ["24", 0], "latent_image": ["26", 0]}},
    "26": {"class_type": "EmptyLatentImage", "inputs": {}},
    "27": {"class_type": "VAEDecode", "inputs": {"samples": ["25", 0], "vae": ["29", 2]}},
    "28": {"class_type": "WorkflowGallery", "inputs": {"images": ["27", 0]}},
    "29": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "b.safetensors"}},
}
p, n = _extract_prompts(g4, "28")
check("g4 positive (pipeline B not A)", p, "editorial photo, woman leaning on car, eyepatch")
check("g4 negative", n, "watermark, text")
check("g4 seed (pipeline B)", _extract_seed(g4, "28"), 555)

# ---- Graph 5: gallery disconnected from any sampler → honest unavailable, not pipeline A ----
g5 = dict(g4)
g5["28"] = {"class_type": "WorkflowGallery", "inputs": {"images": ["30", 0]}}
g5["30"] = {"class_type": "LoadImage", "inputs": {"image": "external.png"}}
p, n, src5 = _extract_prompts_with_fallback(g5, None, "28")
check("g5 positive (unavailable, not stolen)", p, "")
check("g5 source", src5, "unavailable")

# ---- Graph 6: PromptEnhancer in the text path ----
PEO.clear(); PLO.clear()
g6 = {
    "40": {"class_type": "PromptLibrary", "inputs": {"selected_prompt_id": "1girl, base tags", "manual_text": "", "manual_position": "after", "expand_wildcards": False, "wildcard_path": "", "seed": 0}},
    "41": {"class_type": "PromptEnhancer", "inputs": {"enabled": True, "backend": "Ollama", "api_url": "http://localhost:11434/v1", "api_key": "", "openrouter_key": "", "nanogpt_key": "", "model_name": "qwen3", "target_model": "Illustrious / SDXL Anime", "system_prompt": "sys", "manual_addons": "", "max_tokens": 512, "thinking_mode": False, "seed": 0, "prompt": ["40", 0]}},
    "3": {"class_type": "CLIPTextEncode", "inputs": {"text": ["41", 0], "clip": ["9", 1]}},
    "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "worst quality", "clip": ["9", 1]}},
    "5": {"class_type": "KSampler", "inputs": {"seed": 7, "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["6", 0]}},
    "6": {"class_type": "EmptyLatentImage", "inputs": {}},
    "7": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["9", 2]}},
    "8": {"class_type": "WorkflowGallery", "inputs": {"images": ["7", 0]}},
    "9": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x.safetensors"}},
}
# Before enhancer runs: falls back to pre-enhancement chain (PromptLibrary widget)
p, n = _extract_prompts(g6, "8")
check("g6 pre-run fallback", p, "1girl, base tags")
# After enhancer runs: runtime output wins
PEO["41"] = "masterpiece, best quality, 1girl, enhanced tags"
p, n = _extract_prompts(g6, "8")
check("g6 enhanced output", p, "masterpiece, best quality, 1girl, enhanced tags")
check("g6 negative", n, "worst quality")

# ---- Graph 7: int widget values must NOT be treated as node refs ----
g7 = {
    "20": {"class_type": "TextMultiline", "inputs": {"text": "FOREIGN TEXT — must not appear"}},
    "50": {"class_type": "SomeCustomStringNode", "inputs": {"max_tokens": 20, "mode": "concat"}},  # 20 collides with node id "20"
    "3": {"class_type": "CLIPTextEncode", "inputs": {"text": ["50", 0], "clip": ["9", 1]}},
    "5": {"class_type": "KSampler", "inputs": {"seed": 1, "positive": ["3", 0], "negative": ["3", 0], "latent_image": ["6", 0]}},
    "6": {"class_type": "EmptyLatentImage", "inputs": {}},
    "7": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["9", 2]}},
    "8": {"class_type": "WorkflowGallery", "inputs": {"images": ["7", 0]}},
    "9": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x.safetensors"}},
}
p, n = _extract_prompts(g7, "8")
check("g7 no int-ref wormhole", p, "")

print()
if fails:
    print(f"{len(fails)} FAILURES: {fails}")
    sys.exit(1)
print("ALL TESTS PASSED")
