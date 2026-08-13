import { app } from "../../scripts/app.js";

const PE_EXTENSION_NAME = "comfyui.prompt.enhancer";
const PE_TARGET_CLASS = "PromptEnhancer";

const DEFAULT_URLS = {
  Ollama: "http://localhost:11434/v1",
  OpenRouter: "https://openrouter.ai/api/v1",
  NanoGPT: "https://nano-gpt.com/api/v1",
  Kobold: "http://localhost:5001/v1",
};

let PRESETS = {};

async function loadPresets() {
  try {
    const res = await fetch("/prompt_enhancer/presets");
    const data = await res.json();
    PRESETS = data.presets || {};
  } catch (e) {
    console.warn("[PromptEnhancer] Could not load presets", e);
  }
}

loadPresets();

function buildEnhancerPanel(node) {
  const panel = document.createElement("div");
  panel.style.cssText = "display:flex;flex-direction:column;gap:6px;padding:6px;box-sizing:border-box;width:100%;";

  const statusDot = document.createElement("span");
  statusDot.style.cssText = "display:inline-block;width:8px;height:8px;border-radius:50%;background:#666;flex-shrink:0;margin-top:3px;";

  const statusText = document.createElement("span");
  statusText.style.cssText = "font-size:11px;opacity:0.8;flex:1;";
  statusText.textContent = "Not connected";

  const connectBtn = document.createElement("button");
  connectBtn.textContent = "🔌 Connect";
  connectBtn.style.cssText = "cursor:pointer;font-size:11px;padding:3px 10px;border-radius:4px;border:1px solid rgba(100,180,255,0.4);background:rgba(100,180,255,0.1);color:rgba(150,210,255,1);white-space:nowrap;";

  const statusRow = document.createElement("div");
  statusRow.style.cssText = "display:flex;gap:6px;align-items:flex-start;";
  statusRow.appendChild(statusDot);
  statusRow.appendChild(statusText);
  statusRow.appendChild(connectBtn);

  const modelSelect = document.createElement("select");
  modelSelect.style.cssText = "width:100%;padding:4px 8px;border-radius:6px;border:1px solid rgba(255,255,255,0.2);background:rgba(0,0,0,0.3);color:inherit;font-size:12px;";

  const modelPickerLabel = document.createElement("div");
  modelPickerLabel.style.cssText = "font-size:10px;opacity:0.7;";
  modelPickerLabel.textContent = "Select model:";

  const modelPickerRow = document.createElement("div");
  modelPickerRow.style.cssText = "display:none;flex-direction:column;gap:3px;";
  modelPickerRow.appendChild(modelPickerLabel);
  modelPickerRow.appendChild(modelSelect);

  // ── Reset button (inside panel — single DOM widget avoids overlap) ──────────
  const resetBtn = document.createElement("button");
  resetBtn.textContent = "↺ Reset system prompt to preset";
  resetBtn.style.cssText = "cursor:pointer;font-size:11px;padding:3px 8px;border-radius:4px;border:1px solid rgba(255,255,255,0.2);background:rgba(255,255,255,0.06);color:inherit;width:100%;margin-top:2px;";

  panel.appendChild(statusRow);
  panel.appendChild(modelPickerRow);
  panel.appendChild(resetBtn);

  const setStatus = (ok, msg) => {
    statusDot.style.background = ok === null ? "#aaa" : ok ? "#4caf50" : "#f44336";
    statusText.textContent = msg;
  };

  connectBtn.addEventListener("click", async () => {
    const backendWidget = node.widgets?.find(w => w.name === "backend");
    const localEncoderWidget = node.widgets?.find(w => w.name === "local_text_encoder");
    const apiUrlWidget = node.widgets?.find(w => w.name === "api_url");
    const apiKeyWidget = node.widgets?.find(w => w.name === "api_key");
    const api_url = apiUrlWidget?.value || "";
    const api_key = apiKeyWidget?.value || "";

    if (backendWidget?.value === "ComfyUI Local") {
      const selected = localEncoderWidget?.value || "";
      const connectedClip = node.inputs?.some(input => input.name === "clip" && input.link != null);
      const usable = connectedClip || (selected && selected !== "(use connected CLIP input)");
      setStatus(
        usable,
        connectedClip
          ? "Local mode — using connected CLIP"
          : usable
            ? `Local mode — ${selected}`
            : "Choose a local encoder or connect CLIP"
      );
      modelPickerRow.style.display = "none";
      return;
    }

    if (!api_url) { setStatus(false, "No API URL set"); return; }

    connectBtn.textContent = "⏳ Connecting...";
    connectBtn.disabled = true;
    setStatus(null, "Connecting...");

    try {
      const res = await fetch("/prompt_enhancer/models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_url, api_key }),
      });
      const data = await res.json();

      if (data.ok && data.models?.length) {
        setStatus(true, `Connected — ${data.models.length} model(s) available`);

        const modelNameWidget = node.widgets?.find(w => w.name === "model_name");
        const currentModel = modelNameWidget?.value || "";

        modelSelect.innerHTML = "";
        for (const m of data.models) {
          const opt = document.createElement("option");
          opt.value = m;
          opt.textContent = m;
          if (m === currentModel) opt.selected = true;
          modelSelect.appendChild(opt);
        }

        if (currentModel && !data.models.includes(currentModel)) {
          const opt = document.createElement("option");
          opt.value = currentModel;
          opt.textContent = `${currentModel} (current)`;
          opt.selected = true;
          modelSelect.insertBefore(opt, modelSelect.firstChild);
        }

        modelPickerRow.style.display = "flex";

        modelSelect.onchange = () => {
          if (modelNameWidget) {
            modelNameWidget.value = modelSelect.value;
            node.setDirtyCanvas(true, true);
          }
        };

      } else {
        setStatus(false, data.error || "No models found");
        modelPickerRow.style.display = "none";
      }
    } catch (err) {
      setStatus(false, `Error: ${err.message}`);
      modelPickerRow.style.display = "none";
    }

    connectBtn.textContent = "🔌 Connect";
    connectBtn.disabled = false;
  });

  return { panel, resetBtn };
}

app.registerExtension({
  name: PE_EXTENSION_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== PE_TARGET_CLASS) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      const node = this;

      const backendWidget = node.widgets?.find(w => w.name === "backend");
      const apiUrlWidget = node.widgets?.find(w => w.name === "api_url");
      const apiKeyWidget = node.widgets?.find(w => w.name === "api_key");
      const openrouterKeyWidget = node.widgets?.find(w => w.name === "openrouter_key");
      const nanogptKeyWidget = node.widgets?.find(w => w.name === "nanogpt_key");

      // Track whether node is fully initialized to avoid overwriting saved values on load
      let nodeInitialized = false;
      setTimeout(() => { nodeInitialized = true; }, 500);

      if (backendWidget && apiUrlWidget) {
        const origCallback = backendWidget.callback;
        backendWidget.callback = function (value) {
          origCallback?.call(this, value);
          // Only auto-fill URL if user actually changed the backend (not on initial load)
          if (nodeInitialized) {
            const defaultUrl = DEFAULT_URLS[value];
            if (defaultUrl) apiUrlWidget.value = defaultUrl;
          }
          // Auto-swap api_key from saved keys
          if (value === "OpenRouter" && openrouterKeyWidget?.value) {
            if (apiKeyWidget) apiKeyWidget.value = openrouterKeyWidget.value;
          } else if (value === "NanoGPT" && nanogptKeyWidget?.value) {
            if (apiKeyWidget) apiKeyWidget.value = nanogptKeyWidget.value;
          } else if ((value === "Ollama" || value === "Kobold") && nodeInitialized) {
            if (apiKeyWidget) apiKeyWidget.value = "";
          }
          node.setDirtyCanvas(true, true);
        };
      }

      // Save api_key back to the appropriate stored key when it changes
      if (apiKeyWidget) {
        const origApiKeyCallback = apiKeyWidget.callback;
        apiKeyWidget.callback = function(value) {
          origApiKeyCallback?.call(this, value);
          const currentBackend = backendWidget?.value;
          if (currentBackend === "OpenRouter" && openrouterKeyWidget) {
            openrouterKeyWidget.value = value;
          } else if (currentBackend === "NanoGPT" && nanogptKeyWidget) {
            nanogptKeyWidget.value = value;
          }
        };
      }

      const targetModelWidget = node.widgets?.find(w => w.name === "target_model");
      const systemPromptWidget = node.widgets?.find(w => w.name === "system_prompt");
      let userEditedSystemPrompt = false;

      if (systemPromptWidget?.inputEl) {
        systemPromptWidget.inputEl.addEventListener("input", () => { userEditedSystemPrompt = true; });
      }

      if (targetModelWidget && systemPromptWidget) {
        const origTargetCallback = targetModelWidget.callback;
        targetModelWidget.callback = async function (value) {
          origTargetCallback?.call(this, value);
          if (!userEditedSystemPrompt) {
            if (!Object.keys(PRESETS).length) await loadPresets();
            const preset = PRESETS[value];
            if (preset) {
              systemPromptWidget.value = preset;
              if (systemPromptWidget.inputEl) systemPromptWidget.inputEl.value = preset;
            }
          }
          node.setDirtyCanvas(true, true);
        };
      }

      const { panel, resetBtn } = buildEnhancerPanel(node);

      // Wire up reset button now that we have widget references
      resetBtn.addEventListener("click", async () => {
        if (!Object.keys(PRESETS).length) await loadPresets();
        const val = targetModelWidget?.value;
        const preset = PRESETS[val];
        if (preset && systemPromptWidget) {
          systemPromptWidget.value = preset;
          if (systemPromptWidget.inputEl) systemPromptWidget.inputEl.value = preset;
          userEditedSystemPrompt = false;
          node.setDirtyCanvas(true, true);
        }
      });

      // Single DOM widget — no overlap issues
      // NOTE: do NOT pass computeSize in the options object — ComfyUI's addDOMWidget
      // internally calls this.output on an undefined context and throws on startup.
      // Set it directly on the returned widget object instead.
      const peWidget = node.addDOMWidget("pe_panel", "PEDOM", panel, {
        serialize: false, hideOnZoom: true,
        getValue: () => null, setValue: () => {},
      });
      if (peWidget) {
        peWidget.computeSize = () => [panel.scrollWidth || 460, panel.scrollHeight || 80];
      }

      node.size = [Math.max(node.size?.[0] || 0, 460), Math.max(node.size?.[1] || 0, 540)];
      return result;
    };
  },
});
