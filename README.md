# Lokitsar's Nodes

A suite of ComfyUI custom nodes for reviewing and comparing generated images,
managing reusable prompts, enhancing prompts with local or cloud language models,
and writing LoRA caption sidecars — all without leaving ComfyUI.

## Version

Current release: **v0.1.22**

### What's new in v0.1.22

- Expanded **Prompt Enhancer** with an in-node control panel, a native
  **ComfyUI Local** backend, reusable connected `CLIP` support, and selectable
  text-encoder wrappers.
- Added model-aware presets for **MiniMax H3 Base, Frame-to-Frame, Last-Frame,
  and Reference** workflows alongside Flux/Z-Image, Illustrious, LTX, and Wan.
- Improved vision prompting, reasoning cleanup, provider authentication and error
  messages, deterministic seeds, and automatic Ollama unloading to release VRAM.
- Added packaged Danbooru guidance for more accurate Illustrious/SDXL Anime prompts.
- Improved **Workflow Gallery** prompt tracing, source labels, Prompt Enhancer
  output capture, scrolling, resizing, cache cleanup, and saved-session recovery.
- Improved **Prompt Library** wildcard lookup with `.txt`, YAML, nested YAML keys,
  recursive expansion, deterministic seeds, and faster cached path resolution.
- Added **LoRA Sidecar Saver** for safely writing matching `.txt` caption files.

## Why I made this

I wanted a cleaner way to review multiple generated images inside a workflow without digging through output folders every time.

A practical use case is generating multiple showcase images for wildcard packs, LoRAs, prompt packs, or Civitai posts. Instead of hunting through saved files, these nodes let you review, compare, selectively export, and reuse prompts directly inside ComfyUI.

## Nodes

This package includes four nodes:

- **Workflow Gallery** — image gallery, viewer, comparison, and export (`image/ui`)
- **Prompt Library** — save, search, and reuse prompts across workflows (`utils/prompt`)
- **Prompt Enhancer** — expand a rough idea (and/or a reference image) into a model-tuned generation prompt via a local or cloud LLM (`utils/prompt`)
- **LoRA Sidecar Saver** — write a generated caption beside training media using a safe matching filename (`Lokitsars/Training`)

---

## Workflow Gallery

### Gallery
- Receives image batches and displays them as thumbnails inside the node
- Hover over any thumbnail to see the positive and negative prompt used to generate it
- Seed is shown on hover and in the viewer
- Resize thumbnails with the slider at the bottom
- Thumbnails show newest images first
- Automatically prunes oldest images when the gallery reaches the configured limit
- Passes input images through unchanged so it works anywhere in your pipeline
- Gallery state persists across ComfyUI restarts — your images are right where you left them

### Viewer
- Click any thumbnail to expand it in full viewer mode
- Seed, positive, and negative prompts displayed in labeled sections
- Inline **Copy** button next to each section for quick clipboard access
- **Save Prompt** button on the positive prompt section — saves directly to your Prompt Library with a custom name and optional tags
- Navigate between images with left and right arrow buttons
- Click the expanded image to return to the gallery

### Image Comparison
- Shift-click two thumbnails to select them (highlighted in orange)
- Click the **⇔ Compare** button that appears in the toolbar
- Side-by-side viewer opens with a draggable divider between the two images
- Drag the divider left and right to reveal more of either image
- Each side shows its own prompt below the image
- Compare stage automatically resizes when you resize the node
- Click **✕ Exit compare** to return to the gallery

### Export & Selection
- Shift-click any number of thumbnails to select them
- **↓ Export Selected** button appears when one or more images are selected
- Exported images are copied to your configured output folder
- A green **✓ badge** appears on exported thumbnails so you always know what's been saved
- **Clear Unexported** removes all non-exported images from the cache (with confirmation)
- **Clear All** removes everything including exported images (with confirmation)

### Cache-first workflow
- By default, images are stored in a local cache — nothing is permanently saved until you export
- Turn on **Save to Disk** if you want every image automatically saved as it generates
- Default export path is `ComfyUI/output/workflow_gallery/` — customizable in the node settings
- Uses ComfyUI's universal `folder_paths` API so the correct output directory is detected automatically across all install types (manual, portable, desktop app)

### Prompt Resolution
- Prompts are resolved directly from the live workflow graph, scoped to the sampler connected to your gallery node
- Supports wildcard nodes, string nodes, primitive nodes, Prompt Library, Prompt Enhancer, and other upstream text-feeding nodes
- Captures the Prompt Enhancer's actual runtime output instead of displaying its rough input prompt
- Displays the detected prompt source in the viewer so you can tell how it was resolved
- Optional positive and negative override sockets accept the exact strings used by unusual workflows
- Handles `ConditioningZeroOut` correctly — no bleed from positive to negative
- Works correctly in workflows with multiple samplers

### Node Inputs

| Input | Type | Default | Description |
|---|---|---|---|
| images | IMAGE | — | Image batch to collect and display |
| enabled | BOOLEAN | true | Pass-through toggle — disable to bypass the gallery |
| save_to_disk | BOOLEAN | false | Save every image to disk as it generates |
| output_directory | STRING | output/workflow_gallery | Where exported images are saved |
| filename_prefix | STRING | workflow_gallery | Prefix used for saved filenames |
| max_images | INT | 48 | Maximum number of images to keep in the gallery |
| positive_override | STRING | — | Optional wired ground-truth positive prompt |
| negative_override | STRING | — | Optional wired ground-truth negative prompt |

---

## Prompt Library

A persistent prompt manager that lives directly in your workflow. Save, search, tag, and reuse prompts — and connect them directly to your CLIPTextEncode nodes.

### Features
- Outputs a `STRING` that connects directly to any CLIPTextEncode `text` input
- Search prompts by text or filter by tags
- Add manual text that combines with your selected prompt — insert before or after
- **💾 Save Current** — saves the manual text area content to the library
- **Save Prompt** button in the Workflow Gallery viewer saves any image's prompt directly to the library
- Edit saved prompts inline — update name, tags, and text
- Delete prompts with confirmation
- Second click on a selected prompt deselects it
- Export your entire library as **JSON** or **CSV**
- Library persists across restarts
- Expand `__file__` wildcards from the packaged `wildcards/` folder, ComfyUI's
  wildcard folder, Impact Pack, or additional folders you specify
- Supports `.txt`, `.yaml`, and `.yml`, including nested keys such as
  `__colors/dark__`, plus inline choices such as `{red|green|blue}`
- Recursively expands nested wildcards with a deterministic seed

### How to use
1. Add the **Prompt Library** node to your workflow
2. Connect its `prompt` output to a CLIPTextEncode `text` input
3. Select a saved prompt from the list — it becomes the output string immediately
4. Optionally type additional text in the manual input and choose whether it goes before or after the selected prompt
5. Save new prompts using **+ Add**, **💾 Save Current**, or the **Save Prompt** button in the Workflow Gallery viewer

### Node Inputs

| Input | Type | Default | Description |
|---|---|---|---|
| selected_prompt_id | STRING | — | Selected library prompt, managed by the in-node interface |
| manual_text | STRING | — | Additional text or a complete manual prompt |
| manual_position | CHOICE | after | Place manual text before or after the selected prompt |
| expand_wildcards | BOOLEAN | false | Expand file and inline wildcard expressions each queue |
| wildcard_path | STRING | — | Optional comma-separated additional wildcard folders |
| seed | INT | 0 | Deterministic wildcard selection seed |

---

## Prompt Enhancer

Turn a rough idea — or a reference image — into a detailed prompt tuned to the model you're generating with. The node calls an LLM (local or cloud), enhances your input, and outputs a `STRING` you can wire into a text encoder or another node.

### Features
- Optional `prompt` (STRING) input and optional `image` (IMAGE) input — use either or both
- Outputs a `STRING` — connect it to a CLIPTextEncode `text` input or any string consumer
- **Target-model presets**, each written to that model's actual prompting rules:
  - **Flux / Z-Image** — natural-language description, no negatives
  - **Illustrious / SDXL Anime** — Danbooru tags with the correct rating and quality tags
  - **LTX Video 2.3** — cinematic single-shot video prompt with audio
  - **Wan 2.2 Video** — cinematic motion prompt
  - **MiniMax H3 Base** — text-to-video+audio or image-to-video+audio direction
  - **MiniMax H3 Frame-to-Frame** — first/last-frame video+audio direction
  - **MiniMax H3 Last-Frame** — last-image-to-video+audio direction
  - **MiniMax H3 Reference** — reference-to-video+audio direction
  - **Custom** — general-purpose enhancement
- **Text-to-X vs Image-to-X**: with no image connected the node writes a text-to-image/video prompt; connect an image and it automatically switches to the matching image-to-video / image-reference preset for the selected model (for video models, it describes the motion that plays out from the frame rather than re-describing the still)
- **🔌 Connect** — fetches the available model list from your API and fills the model dropdown, or checks the local encoder selection
- **Direct ComfyUI text-encoder generation** — select a compatible `.safetensors`/NVFP4 text encoder, or connect an already-loaded `CLIP` to avoid loading a second copy
- **↺ Reset** — restores the system prompt from the selected preset
- **thinking_mode** toggle — suppresses chain-of-thought on models that support it (works across backends)
- **manual_addons** — extra instructions appended to every request
- Cleans leaked reasoning, analysis scaffolding, preambles, and self-revision from the returned prompt
- Uses actionable errors for authentication, connection, empty-response, incompatible local encoder, and generation failures
- Explicitly unloads Ollama after a request so its model does not continue occupying VRAM needed by ComfyUI
- Disabled mode passes the original prompt through unchanged

### Backends
All OpenAI-compatible `/chat/completions` endpoints:

| Backend | URL | Notes |
|---|---|---|
| Ollama | `http://localhost:11434/v1` | Local; falls back to the native API if needed |
| OpenRouter | `https://openrouter.ai/api/v1` | Cloud; API key saved separately |
| NanoGPT | `https://nano-gpt.com/api/v1` | Cloud; API key saved separately |
| Kobold | `http://localhost:5001/v1` | Local |
| ComfyUI Local | — | Uses ComfyUI's native text-encoder loader and model offloading; no API server |

`ComfyUI Local` is for causal/generative text encoders. A `.safetensors` extension by itself does not make a checkpoint an LLM: CLIP, T5, and other conditioning-only encoders cannot write prompts. Converted NVFP4 files are supported when ComfyUI supports their quantization metadata and the checkpoint retains enough of the language model to generate. MiniMax H3's 32B Qwen3-VL encoder is truncated specifically for H3 conditioning, so it is allowed but may generate lower-quality text than a complete instruct checkpoint.

For the lowest VRAM use, load the encoder with ComfyUI's normal **Load CLIP** node and connect its `CLIP` output to Prompt Enhancer. Otherwise, select the file under `local_text_encoder` and choose the same `local_clip_type` you would use in **Load CLIP**.

### Image input (vision models)
Connect an `IMAGE` to use a vision-capable model (for example Qwen-VL locally or
through a cloud provider) to read the reference and write a prompt from it. The
node automatically selects the matching hidden vision preset unless you edited
the system prompt manually. For H3, the reference is labeled according to the
selected Base, Frame-to-Frame, Last-Frame, or Reference mode. A non-reasoning
instruction-tuned vision model generally gives the cleanest result.

### Content
Paired with a local, uncensored model, the enhancer describes content faithfully across the full range from everyday to explicit, instead of omitting obvious details. All subjects are treated as adults; the node never produces sexual, nude, or suggestive descriptions of anyone who is or appears to be a minor.

### Node Inputs

| Input | Type | Default | Description |
|---|---|---|---|
| enabled | BOOLEAN | true | Toggle the enhancer on/off (off passes the prompt through) |
| backend | CHOICE | Ollama | Ollama / OpenRouter / NanoGPT / Kobold / ComfyUI Local |
| api_url | STRING | localhost Ollama | Endpoint; auto-fills when you switch backend |
| api_key | STRING | — | Active key; falls back to the matching saved provider key |
| openrouter_key | STRING | — | Saved OpenRouter key |
| nanogpt_key | STRING | — | Saved NanoGPT key |
| model_name | STRING | llama3 | API model id; set via **Connect** or typed |
| local_text_encoder | CHOICE | connected CLIP | File from `models/text_encoders` used by ComfyUI Local |
| local_clip_type | CHOICE | flux2 | Wrapper/type matching ComfyUI's **Load CLIP** node |
| target_model | CHOICE | Flux / Z-Image | Which model the output prompt is tuned for |
| system_prompt | STRING | preset | Editable system prompt (per target model) |
| manual_addons | STRING | — | Extra instructions appended to the request |
| max_tokens | INT | 512 | Response length cap |
| thinking_mode | BOOLEAN | false | Suppress/allow model chain-of-thought |
| seed | INT | 0 | 0 = random; non-zero for reproducibility |
| prompt | STRING | — | (optional) rough prompt to enhance |
| image | IMAGE | — | (optional) reference image for vision models |
| clip | CLIP | — | (optional) reuse an already-loaded local text encoder |

---

## LoRA Sidecar Saver

Write generated caption text as a `.txt` sidecar with the same stem as an image
filename. This is useful for simple LoRA workflows that already produce captions
inside ComfyUI but do not need the complete dataset-building application.

The node normalizes paths, creates the destination folder when needed, removes
unsafe filename characters, protects Windows reserved filenames, and writes
atomically so an interrupted save does not leave a partial caption.

### Node Inputs

| Input | Type | Default | Description |
|---|---|---|---|
| text | STRING | — | Caption text supplied by another node |
| filename | STRING | — | Image filename or stem used to name the `.txt` sidecar |
| folder | STRING | — | Destination folder for the caption file |
| existing_file | CHOICE | overwrite | Overwrite an existing sidecar or skip it |

The output is a status string containing the written path or the skipped file.

---

## Installation

### Option 1 — ComfyUI Manager (recommended)
Search for **Workflow Gallery** in ComfyUI Manager and install directly.

### Option 2 — Manual
1. Clone or copy this folder into `ComfyUI/custom_nodes/` using the folder name
   `ComfyUI-Lokitsars-Nodes`
2. From the Python environment that launches ComfyUI, run
   `python -m pip install -e ComfyUI/custom_nodes/ComfyUI-Lokitsars-Nodes`
3. Restart ComfyUI and hard-refresh the browser
4. Find **Workflow Gallery** under `image/ui`, **Prompt Library** and
   **Prompt Enhancer** under `utils/prompt`, and **LoRA Sidecar Saver** under
   `Lokitsars/Training`

```text
ComfyUI/
└── custom_nodes/
    └── ComfyUI-Lokitsars-Nodes/
        ├── nodes.py
        ├── training_helpers.py
        ├── danbooru_tags.txt
        ├── js/
        │   ├── workflow_gallery.js
        │   ├── prompt_library.js
        │   └── prompt_enhancer.js
        └── ...
```

---

## How to use

### Basic review workflow
1. Place the **Workflow Gallery** node between your sampler and any Save Image node
2. Connect your image output to the `images` input
3. Queue your generation — images will appear in the gallery as they complete
4. Click any thumbnail to expand it and read the prompt

### Selective export workflow
1. Leave **Save to Disk** off (default)
2. Generate a batch of images
3. Shift-click the ones you want to keep
4. Click **↓ Export Selected** — they're copied to your output folder with a ✓ badge
5. Click **Clear Unexported** to clean up the rest

### Comparison workflow
1. Shift-click exactly two thumbnails
2. Click **⇔ Compare**
3. Drag the divider to compare the images side by side

### Prompt Library workflow
1. Add the Prompt Library node and connect it to your CLIPTextEncode
2. Generate images — click any thumbnail to expand it in the gallery viewer
3. Click **Save Prompt** on the positive prompt section to save it with a name and tags
4. The saved prompt appears in your Prompt Library node immediately after clicking Refresh
5. Select it from the list — it feeds directly into your encoder on the next generation

### Prompt Enhancer workflow

1. Add Prompt Enhancer and connect a rough `STRING`, an `IMAGE`, or both
2. Choose the target generation model and either an API backend or ComfyUI Local
3. Click **Connect** to discover API models or verify the local encoder selection
4. Connect `enhanced_prompt` to the text encoder or prompt consumer in your workflow

### LoRA sidecar workflow

1. Connect generated caption text to `text`
2. Connect or provide the corresponding image filename
3. Choose the dataset folder and whether existing captions should be overwritten
4. Queue the node to write the matching `.txt` file

---

## Screenshots

### Gallery View
![Gallery View](screenshots/Screenshot%202026-03-08%20000942.png)

### Viewer Mode
![Viewer Mode](screenshots/Screenshot%202026-03-08%20014404.png)
