import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXTENSION_NAME = "comfyui.workflow.gallery";
const TARGET_CLASS = "WorkflowGallery";
const GALLERY_HEIGHT = 680;
const THUMB_MIN = 120;
const THUMB_MAX = 360;
const THUMB_DEFAULT = 120;
const THUMB_STORAGE_KEY = "workflow_gallery_thumb_size";
let thumbSizeMemory = THUMB_DEFAULT;

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "style") Object.assign(node.style, value);
    else if (key === "className") node.className = value;
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2).toLowerCase(), value);
    else node.setAttribute(key, value);
  }
  for (const child of children) {
    if (child == null) continue;
    node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

function ensureStyles() {
  if (document.getElementById("workflow-gallery-styles")) return;
  const style = document.createElement("style");
  style.id = "workflow-gallery-styles";
  style.textContent = `
    .wg-root { display:flex; flex-direction:column; gap:8px; padding:8px; box-sizing:border-box; width:100%; min-width:0; min-height:0; height:100%; overflow:auto; }
    .wg-topbar { display:flex; gap:6px; align-items:center; flex-wrap:wrap; }
    .wg-topbar button { cursor:pointer; }
    .wg-count { margin-left:auto; opacity:0.85; font-size:12px; }
    .wg-preview { flex:1 1 auto; min-height:200px; overflow:hidden; border:1px solid rgba(255,255,255,0.15); border-radius:8px; padding:8px; background:rgba(0,0,0,0.2); display:flex; flex-direction:column; position:relative; min-width:0; }
    .wg-preview.hidden { display:none; }
    .wg-preview-stage { flex:1 1 auto; min-height:0; display:flex; align-items:stretch; gap:8px; }
    .wg-preview-lane { flex:0 0 48px; display:flex; align-items:center; justify-content:center; }
    .wg-preview-img-wrap { flex:1 1 auto; min-width:0; min-height:0; display:flex; align-items:center; justify-content:center; overflow:hidden; }
    .wg-preview-img { display:block; max-width:100%; max-height:100%; width:auto; height:auto; margin:0 auto; border-radius:8px; object-fit:contain; cursor:pointer; }
    .wg-nav { width:40px; height:88px; display:flex; align-items:center; justify-content:center; border:1px solid rgba(255,255,255,0.16); border-radius:10px; background:rgba(255,255,255,0.06); color:rgba(255,255,255,0.95); font-size:28px; line-height:1; cursor:pointer; user-select:none; transition:background 0.12s ease, opacity 0.12s ease; }
    .wg-nav:hover { background:rgba(255,255,255,0.12); }
    .wg-nav.hidden { visibility:hidden; pointer-events:none; opacity:0; }
    .wg-preview-caption { padding-top:6px; font-size:11px; line-height:1.25; word-break:break-word; opacity:0.9; text-align:center; flex-shrink:0; }
    .wg-prompt-actions { display:flex; gap:6px; justify-content:center; margin-top:6px; flex-shrink:0; }
    .wg-prompt-actions button { cursor:pointer; }
    .wg-copy-seed-btn { cursor:pointer; font-size:11px; }
    .wg-copy-seed-btn.hidden { display:none; }
    .wg-prompt-text { margin-top:6px; font-size:11px; line-height:1.3; white-space:pre-wrap; max-height:140px; overflow:auto; border:1px solid rgba(255,255,255,0.12); border-radius:6px; padding:6px; flex-shrink:0; }
    .wg-prompt-sections { margin-top:6px; display:flex; flex-direction:column; gap:4px; max-height:220px; overflow:auto; flex-shrink:0; }
    .wg-prompt-section { border:1px solid rgba(255,255,255,0.12); border-radius:6px; overflow:hidden; }
    .wg-prompt-section-header { display:flex; align-items:center; justify-content:space-between; padding:3px 6px; background:rgba(255,255,255,0.06); font-size:10px; font-weight:bold; text-transform:uppercase; letter-spacing:0.05em; opacity:0.8; }
    .wg-prompt-section-copy { cursor:pointer; background:none; border:1px solid rgba(255,255,255,0.2); border-radius:3px; color:inherit; font-size:10px; padding:1px 5px; opacity:0.7; }
    .wg-prompt-section-copy:hover { opacity:1; background:rgba(255,255,255,0.1); }
    .wg-prompt-section-body { padding:5px 6px; font-size:11px; line-height:1.3; white-space:pre-wrap; word-break:break-word; }
    .wg-gallery { flex:1 1 auto; min-height:0; overflow:auto; border:1px solid rgba(255,255,255,0.15); border-radius:8px; padding:6px; display:grid; gap:6px; align-content:start; overscroll-behavior:contain; }
    .wg-root.viewer-mode .wg-gallery, .wg-root.viewer-mode .wg-slider-row, .wg-root.viewer-mode .wg-meta { display:none; }
    .wg-root.viewer-mode .wg-preview { flex:1 1 auto; min-height:0; }
    .wg-item img { display:block; width:100%; height:auto; }
    .wg-item.selected { outline:2px solid rgba(120,180,255,0.9); }
    .wg-caption { padding:4px 6px; font-size:11px; line-height:1.25; word-break:break-word; opacity:0.9; }
    .wg-empty { opacity:0.7; font-size:12px; padding:10px; text-align:center; }
    .wg-slider-row { display:flex; gap:8px; align-items:center; font-size:12px; }
    .wg-slider-row input[type='range'] { flex:1; }
    .wg-order-hint { font-size:11px; opacity:0.75; text-align:right; }
    .wg-item.compare-selected { outline:2px solid rgba(255,180,50,0.9); }
    .wg-item.exported .wg-export-badge { display:flex; }
    .wg-export-badge { display:none; position:absolute; top:4px; right:4px; background:rgba(60,180,80,0.92); color:#fff; border-radius:50%; width:20px; height:20px; align-items:center; justify-content:center; font-size:12px; font-weight:bold; box-shadow:0 1px 4px rgba(0,0,0,0.4); z-index:2; pointer-events:none; }
    .wg-item { position:relative; border:1px solid rgba(255,255,255,0.12); border-radius:8px; overflow:hidden; background:rgba(0,0,0,0.18); cursor:pointer; }
    .wg-export-btn { cursor:pointer; background:rgba(60,180,80,0.15); border:1px solid rgba(60,180,80,0.5); color:rgba(100,220,120,1); border-radius:4px; padding:2px 8px; font-size:12px; }
    .wg-export-btn:hover { background:rgba(60,180,80,0.28); }
    .wg-export-btn.hidden { display:none; }
    .wg-delete-btn { cursor:pointer; background:rgba(220,60,60,0.15); border:1px solid rgba(220,60,60,0.5); color:rgba(255,100,100,1); border-radius:4px; padding:2px 8px; font-size:12px; }
    .wg-delete-btn:hover { background:rgba(220,60,60,0.28); }
    .wg-delete-btn.hidden { display:none; }
    .wg-clear-unexported-btn { cursor:pointer; font-size:12px; }
    .wg-clear-unexported-btn.hidden { display:none; }
    .wg-compare-btn:hover { background:rgba(255,180,50,0.28); }
    .wg-compare-btn.hidden { display:none; }
    .wg-compare-wrap { flex:1 1 auto; min-height:0; display:flex; flex-direction:column; gap:6px; overflow:hidden; }
    .wg-compare-stage { flex:1 1 auto; min-height:0; width:100%; position:relative; overflow:hidden; border-radius:8px; background:rgba(0,0,0,0.3); user-select:none; }
    .wg-compare-side { position:absolute; top:0; bottom:0; overflow:hidden; }
    .wg-compare-side-left { left:0; right:auto; }
    .wg-compare-side-right { right:0; left:auto; }
    .wg-compare-side img { position:absolute; top:0; display:block; pointer-events:none; object-fit:contain; }
    .wg-compare-side-left img { left:0; width:100%; height:100%; }
    .wg-compare-side-right img { right:0; height:100%; }
    .wg-compare-divider { position:absolute; top:0; bottom:0; width:4px; background:rgba(255,255,255,0.9); cursor:ew-resize; z-index:10; transform:translateX(-50%); border-radius:2px; box-shadow:0 0 8px rgba(0,0,0,0.6); }
    .wg-compare-divider::after { content:'⇔'; position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); background:rgba(255,255,255,0.95); color:#333; border-radius:50%; width:28px; height:28px; display:flex; align-items:center; justify-content:center; font-size:13px; box-shadow:0 2px 6px rgba(0,0,0,0.4); }
    .wg-compare-labels { display:flex; gap:8px; font-size:11px; flex-shrink:0; }
    .wg-compare-label { flex:1; padding:4px 6px; border-radius:6px; border:1px solid rgba(255,255,255,0.1); background:rgba(0,0,0,0.2); white-space:pre-wrap; max-height:80px; overflow:auto; line-height:1.3; }
    .wg-compare-label-title { font-weight:bold; opacity:0.7; margin-bottom:2px; font-size:10px; text-transform:uppercase; letter-spacing:0.05em; }
    .wg-root.viewer-mode .wg-preview { flex:1 1 auto; min-height:0; }
  `;
  document.head.appendChild(style);
}

async function fetchGallery(nodeId) {
  const res = await api.fetchApi(`/workflow_gallery/state/${nodeId}`);
  if (!res.ok) throw new Error(`Failed to load gallery ${nodeId}`);
  return await res.json();
}

async function clearGallery(nodeId) {
  const res = await api.fetchApi(`/workflow_gallery/clear/${nodeId}`, { method: "POST" });
  if (!res.ok) throw new Error(`Failed to clear gallery ${nodeId}`);
  return await res.json();
}

async function exportEntries(nodeId, entryIds, outputDirectory) {
  const res = await api.fetchApi(`/workflow_gallery/export`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ node_id: nodeId, entry_ids: entryIds, output_directory: outputDirectory }),
  });
  if (!res.ok) throw new Error(`Export failed`);
  return await res.json();
}

async function clearUnexported(nodeId) {
  const res = await api.fetchApi(`/workflow_gallery/clear_unexported/${nodeId}`, { method: "POST" });
  if (!res.ok) throw new Error(`Failed to clear unexported`);
  return await res.json();
}

async function deleteEntries(nodeId, entryIds) {
  const res = await api.fetchApi(`/workflow_gallery/delete_entries/${nodeId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ entry_ids: entryIds }),
  });
  if (!res.ok) throw new Error(`Delete failed`);
  return await res.json();
}

function clampThumbSize(value) {
  return Math.max(THUMB_MIN, Math.min(THUMB_MAX, Number(value) || THUMB_DEFAULT));
}

function loadThumbSizePreference() {
  try {
    const saved = globalThis?.localStorage?.getItem(THUMB_STORAGE_KEY);
    if (saved == null) return thumbSizeMemory;
    const size = clampThumbSize(saved);
    thumbSizeMemory = size;
    return size;
  } catch (_err) {
    return thumbSizeMemory;
  }
}

function saveThumbSizePreference(value) {
  const size = clampThumbSize(value);
  thumbSizeMemory = size;
  try {
    globalThis?.localStorage?.setItem(THUMB_STORAGE_KEY, String(size));
  } catch (_err) {
    // Ignore storage errors (private mode, quota, disabled storage).
  }
}

function layoutGrid(galleryEl, thumbSize) {
  const size = clampThumbSize(thumbSize);
  galleryEl.style.gridTemplateColumns = `repeat(auto-fill, minmax(${size}px, 1fr))`;
}

function getDisplayEntries(payload) {
  return payload?.entries?.slice().reverse() || [];
}

function closeViewer(node) {
  const state = node.__wgState;
  if (!state) return;
  state.selectedId = null;
  state.root.classList.remove("viewer-mode");
  state.preview.classList.add("hidden");
  state.previewImg.removeAttribute("src");
  state.previewCaption.textContent = "";
  state.promptText.innerHTML = "";
  if (state.copySeedBtn) state.copySeedBtn.classList.add("hidden");
  renderGallery(node, state.payload || { entries: [] });
  node.setDirtyCanvas(true, true);
}

function navigateViewer(node, direction) {
  const state = node.__wgState;
  if (!state?.selectedId || !state.payload) return;
  const entries = getDisplayEntries(state.payload);
  const idx = entries.findIndex((entry) => entry.id === state.selectedId);
  if (idx === -1 || !entries.length) return;
  const nextIdx = Math.max(0, Math.min(entries.length - 1, idx + direction));
  if (nextIdx === idx) return;
  state.selectedId = entries[nextIdx].id;
  renderGallery(node, state.payload);
  node.setDirtyCanvas(true, true);
}

function updateNavButtons(state, entries) {
  if (!state?.navLeft || !state?.navRight) return;
  const idx = entries.findIndex((entry) => entry.id === state.selectedId);
  const atStart = idx <= 0;
  const atEnd = idx === -1 || idx >= entries.length - 1;
  state.navLeft.classList.toggle('hidden', atStart);
  state.navRight.classList.toggle('hidden', atEnd);
}

async function copyToClipboard(text) {
  const value = String(text || "").trim();
  if (!value) return false;
  try {
    await navigator.clipboard.writeText(value);
    return true;
  } catch {
    const area = document.createElement("textarea");
    area.value = value;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(area);
    return ok;
  }
}

function getPromptValue(entry) {
  return String(entry?.display_prompt || entry?.positive_prompt || "").trim();
}

function getNegativePromptValue(entry) {
  return String(entry?.negative_prompt || "").trim();
}

function getSeedValue(entry) {
  return entry?.seed != null ? String(entry.seed) : null;
}

function getPromptSource(entry) {
  return String(entry?.prompt_source || "unavailable").trim();
}

function formatPromptForDisplay(entry) {
  const positive = getPromptValue(entry);
  const negative = getNegativePromptValue(entry);
  const seed = getSeedValue(entry);
  const parts = [];
  if (seed !== null) parts.push(`🌱 Seed: ${seed}`);
  if (positive && negative) parts.push(`[Positive]\n${positive}\n\n[Negative]\n${negative}`);
  else if (positive) parts.push(positive);
  else if (negative) parts.push(`[Negative]\n${negative}`);
  return parts.join("\n\n");
}

function formatTooltip(entry) {
  const positive = getPromptValue(entry);
  const negative = getNegativePromptValue(entry);
  const seed = getSeedValue(entry);
  const parts = [];
  if (seed !== null) parts.push(`Seed: ${seed}`);
  if (positive) parts.push(`Positive: ${positive}`);
  if (negative) parts.push(`Negative: ${negative}`);
  return parts.join("\n\n");
}


function getCompareStageHeight(node) {
  // node.size[1] is the total node height in canvas pixels.
  // Subtract top fields area (inputs) + topbar + labels + padding.
  const totalH = node.size?.[1] || 900;
  const fieldsH = 130; // enabled/save/dir/prefix/max_images inputs
  const topbarH = 36;
  const labelsH = 130; // prompt labels + exit button
  const padding = 32;
  return Math.max(200, totalH - fieldsH - topbarH - labelsH - padding);
}

function openCompareMode(node) {
  const state = node.__wgState;
  if (!state || !state.compareIds || state.compareIds.length < 2) return;
  const entries = getDisplayEntries(state.payload);
  const entryA = entries.find(e => e.id === state.compareIds[0]);
  const entryB = entries.find(e => e.id === state.compareIds[1]);
  if (!entryA || !entryB) return;

  state.root.classList.add("viewer-mode");
  state.preview.classList.remove("hidden");

  // Hide normal viewer contents, show compare wrap
  state.previewStage.style.display = "none";
  state.previewCaption.style.display = "none";
  state.promptActions.style.display = "none";
  state.promptText.style.display = "none";
  state.compareWrap.style.display = "flex";

  // Set explicit stage height from node canvas size
  const stageH = getCompareStageHeight(node);
  state.compareStage.style.height = `${stageH}px`;
  state.compareStage.style.flexShrink = "0";

  // Set images
  state.compareImgLeft.src = entryA.full_url;
  state.compareImgRight.src = entryB.full_url;

  // Set prompt labels
  const promptA = formatPromptForDisplay(entryA);
  const promptB = formatPromptForDisplay(entryB);
  state.compareLabelLeft.innerHTML = `<div class="wg-compare-label-title">Image A</div>${promptA || "No prompt metadata"}`;
  state.compareLabelRight.innerHTML = `<div class="wg-compare-label-title">Image B</div>${promptB || "No prompt metadata"}`;

  // Reset divider to center
  state.compareDividerPos = 50;
  requestAnimationFrame(() => updateCompareDivider(state));

  node.setDirtyCanvas(true, true);
}

function updateCompareDivider(state) {
  const pct = state.compareDividerPos;
  // Left side clips at divider position
  state.compareSideLeft.style.width = `${pct}%`;
  // Right side starts at divider and fills the rest
  state.compareSideRight.style.width = `${100 - pct}%`;
  state.compareSideRight.style.left = `${pct}%`;
  // Right image width is set to the full stage width so it appears full size,
  // but anchored to the right edge so the visible portion aligns correctly
  const stageW = state.compareStage.offsetWidth || 800;
  const rightPanelW = stageW * (1 - pct / 100);
  const fullW = stageW;
  state.compareImgRight.style.width = `${fullW}px`;
  state.compareImgRight.style.right = `0`;
  state.compareImgRight.style.left = `auto`;
  // Left image just fills its panel naturally at full stage width
  state.compareImgLeft.style.width = `${stageW}px`;
  // Move divider bar
  state.compareDivider.style.left = `${pct}%`;
}

function closeCompareMode(node) {
  const state = node.__wgState;
  if (!state) return;
  state.compareIds = [];
  state.selectedIds = [];
  state.compareWrap.style.display = "none";
  state.previewStage.style.display = "";
  state.previewCaption.style.display = "";
  state.promptActions.style.display = "";
  state.promptText.style.display = "";
  state.compareBtn.classList.add("hidden");
  state.exportBtn.classList.add("hidden");
  state.deleteBtn.classList.add("hidden");
  state.clearUnexportedBtn.classList.remove("hidden");
  state.root.classList.remove("viewer-mode");
  state.preview.classList.add("hidden");
  renderGallery(node, state.payload || { entries: [] });
  node.setDirtyCanvas(true, true);
}

function attachCompareDrag(state, stageEl) {
  let dragging = false;

  state.compareDivider.addEventListener("mousedown", (e) => {
    dragging = true;
    e.preventDefault();
  });

  document.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const rect = stageEl.getBoundingClientRect();
    const pct = Math.min(95, Math.max(5, ((e.clientX - rect.left) / rect.width) * 100));
    state.compareDividerPos = pct;
    updateCompareDivider(state);
  });

  document.addEventListener("mouseup", () => { dragging = false; });

  // Touch support
  state.compareDivider.addEventListener("touchstart", (e) => {
    dragging = true;
    e.preventDefault();
  }, { passive: false });

  document.addEventListener("touchmove", (e) => {
    if (!dragging) return;
    const rect = stageEl.getBoundingClientRect();
    const touch = e.touches[0];
    const pct = Math.min(95, Math.max(5, ((touch.clientX - rect.left) / rect.width) * 100));
    state.compareDividerPos = pct;
    updateCompareDivider(state);
  }, { passive: true });

  document.addEventListener("touchend", () => { dragging = false; });
}

function buildPromptSections(entry, container) {
  container.innerHTML = "";
  const seed = getSeedValue(entry);
  const positive = getPromptValue(entry);
  const negative = getNegativePromptValue(entry);

  if (!seed && !positive && !negative) {
    const empty = el("div", { style: { fontSize: "11px", opacity: "0.6", padding: "4px 6px" } }, ["No prompt metadata found for this image."]);
    container.appendChild(empty);
    return;
  }

  const makeSection = (label, text, copyValue, isSaveable = false) => {
    const copyBtn = el("button", { type: "button", className: "wg-prompt-section-copy" }, ["Copy"]);
    const btns = el("div", { style: { display: "flex", gap: "4px" } }, [copyBtn]);

    if (isSaveable) {
      const saveBtn = el("button", { type: "button", className: "wg-prompt-section-copy" }, ["Save Prompt"]);
      // Inline save form that appears below the section
      const saveForm = el("div", { style: { display:"none", padding:"6px", gap:"4px", flexDirection:"column" } });
      const nameIn = el("input", { type: "text", placeholder: "Prompt name *" });
      nameIn.style.cssText = "padding:3px 6px;border-radius:4px;border:1px solid rgba(255,255,255,0.2);background:rgba(0,0,0,0.4);color:inherit;font-size:11px;width:100%;box-sizing:border-box;";
      const tagsIn = el("input", { type: "text", placeholder: "Tags (comma separated, optional)" });
      tagsIn.style.cssText = nameIn.style.cssText;
      const confirmBtn = el("button", { type: "button" }, ["✓ Save"]);
      const cancelBtn = el("button", { type: "button" }, ["✕"]);
      confirmBtn.style.cssText = "cursor:pointer;padding:2px 8px;border-radius:4px;border:1px solid rgba(60,180,80,0.5);background:rgba(60,180,80,0.15);color:rgba(100,220,120,1);font-size:11px;";
      cancelBtn.style.cssText = "cursor:pointer;padding:2px 8px;border-radius:4px;border:1px solid rgba(255,255,255,0.2);background:rgba(255,255,255,0.06);color:inherit;font-size:11px;";
      const formBtns = el("div", { style: { display:"flex", gap:"4px" } }, [confirmBtn, cancelBtn]);
      saveForm.style.display = "none";
      saveForm.appendChild(nameIn);
      saveForm.appendChild(tagsIn);
      saveForm.appendChild(formBtns);

      saveBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        saveForm.style.display = saveForm.style.display === "none" ? "flex" : "none";
        if (saveForm.style.display !== "none") nameIn.focus();
      });

      confirmBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const name = nameIn.value.trim();
        if (!name) { nameIn.focus(); return; }
        const tags = tagsIn.value.split(",").map(t => t.trim()).filter(Boolean);
        try {
          const res = await fetch("/prompt_library/save", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, positive_prompt: copyValue, tags }),
          });
          const data = await res.json();
          if (data.ok) {
            saveBtn.textContent = "✓ Saved!";
            saveForm.style.display = "none";
            nameIn.value = ""; tagsIn.value = "";
            setTimeout(() => { saveBtn.textContent = "Save Prompt"; }, 2000);
          } else if (res.status === 409) {
            saveBtn.textContent = "Already exists";
            setTimeout(() => { saveBtn.textContent = "Save Prompt"; }, 2000);
            saveForm.style.display = "none";
          } else {
            saveBtn.textContent = "Error";
            setTimeout(() => { saveBtn.textContent = "Save Prompt"; }, 2000);
          }
        } catch (err) {
          console.warn("Save prompt failed", err);
        }
      });

      cancelBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        saveForm.style.display = "none";
        nameIn.value = ""; tagsIn.value = "";
      });

      btns.appendChild(saveBtn);
      const section = el("div", { className: "wg-prompt-section" });
      const header = el("div", { className: "wg-prompt-section-header" }, [label, btns]);
      const body = el("div", { className: "wg-prompt-section-body" }, [text]);
      section.appendChild(header);
      section.appendChild(body);
      section.appendChild(saveForm);
      copyBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const ok = await copyToClipboard(copyValue);
        if (ok) {
          copyBtn.textContent = "✓";
          setTimeout(() => { copyBtn.textContent = "Copy"; }, 1500);
        }
      });
      return section;
    }

    const header = el("div", { className: "wg-prompt-section-header" }, [label, btns]);
    const body = el("div", { className: "wg-prompt-section-body" }, [text]);
    const section = el("div", { className: "wg-prompt-section" }, [header, body]);
    copyBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const ok = await copyToClipboard(copyValue);
      if (ok) {
        copyBtn.textContent = "✓";
        setTimeout(() => { copyBtn.textContent = "Copy"; }, 1500);
      }
    });
    return section;
  };

  if (seed) container.appendChild(makeSection("🌱 Seed", seed, seed, false));
  if (positive) container.appendChild(makeSection("Positive", positive, positive, true));
  if (negative) container.appendChild(makeSection("Negative", negative, negative, false));

  const src = getPromptSource(entry);
  if (src) {
    container.appendChild(
      el("div", { style: { fontSize: "10px", opacity: "0.5", padding: "2px 6px", textAlign: "right" } }, [`prompt source: ${src}`])
    );
  }
}


function renderGallery(node, payload) {
  const state = node.__wgState;
  if (!state) return;

  state.payload = payload;
  state.count.textContent = `${payload.count} / ${payload.max_images}`;
  const saveToDisk = payload.save_to_disk !== false;
  state.saveMode.textContent = `Saving full images: ${saveToDisk ? "ON" : "OFF (cache only)"}`;
  state.dir.textContent = `Output directory: ${payload.output_directory || ""}`;
  state.gallery.innerHTML = "";
  layoutGrid(state.gallery, state.thumbSlider.value);

  if (!payload.entries?.length) {
    state.selectedId = null;
    state.root.classList.remove("viewer-mode");
    state.preview.classList.add("hidden");
    state.previewImg.removeAttribute("src");
    state.previewCaption.textContent = "";
    state.promptText.innerHTML = "";
    state.gallery.appendChild(el("div", { className: "wg-empty" }, ["Gallery is empty. Queue an image batch through this node."]));
    return;
  }

  const entries = getDisplayEntries(payload);
  const selectedStillExists = entries.some((entry) => entry.id === state.selectedId);
  if (!selectedStillExists) {
    state.selectedId = null;
  }

  for (const entry of entries) {
    const selected = entry.id === state.selectedId;
    const compareSelected = state.selectedIds?.includes(entry.id);
    const isExported = entry.exported === true;
    const item = el("div", { className: `wg-item${selected ? " selected" : ""}${compareSelected ? " compare-selected" : ""}${isExported ? " exported" : ""}` });
    const badge = el("div", { className: "wg-export-badge" }, ["✓"]);
    const img = el("img", { src: entry.thumb_url, loading: "eager", alt: entry.filename });
    const promptPreview = formatPromptForDisplay(entry);
    const tooltipText = formatTooltip(entry);
    if (tooltipText) {
      item.title = tooltipText;
      img.title = tooltipText;
    }
    const caption = el("div", { className: "wg-caption" }, [`${entry.filename}
${entry.width}×${entry.height}`]);

    item.appendChild(badge);
    item.appendChild(img);
    item.appendChild(caption);
    // Block ComfyUI's canvas shift-click delete menu by catching mousedown early
    item.addEventListener("mousedown", (e) => {
      if (e.shiftKey) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
      }
    }, true);

    item.addEventListener("click", (e) => {
      // Shift-click: unified selection for export and compare
      if (e.shiftKey) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        const idx = state.selectedIds.indexOf(entry.id);
        if (idx !== -1) {
          state.selectedIds.splice(idx, 1);
        } else {
          state.selectedIds.push(entry.id);
        }
        const count = state.selectedIds.length;
        state.exportBtn.classList.toggle("hidden", count < 1);
        state.deleteBtn.classList.toggle("hidden", count < 1);
        state.compareBtn.classList.toggle("hidden", count !== 2);
        state.clearUnexportedBtn.classList.toggle("hidden", count > 0);
        renderGallery(node, state.payload || payload);
        node.setDirtyCanvas(true, true);
        return;
      }

      // Normal click: single image viewer
      if (state.selectedId === entry.id) {
        state.selectedId = null;
        state.root.classList.remove("viewer-mode");
        state.preview.classList.add("hidden");
        state.previewImg.removeAttribute("src");
        state.previewCaption.textContent = "";
        state.promptText.innerHTML = "";
        state.copySeedBtn.classList.add("hidden");
      } else {
        state.selectedId = entry.id;
        state.root.classList.add("viewer-mode");
        state.preview.classList.remove("hidden");
        state.previewImg.src = entry.full_url;
        state.previewImg.alt = entry.filename;
        state.previewCaption.textContent = `${entry.filename} • ${entry.width}×${entry.height}`;
        buildPromptSections(entry, state.promptText);
        state.copySeedBtn.classList.toggle("hidden", getSeedValue(entry) === null);
      }
      renderGallery(node, state.payload || payload);
      node.setDirtyCanvas(true, true);
    });

    state.gallery.appendChild(item);
  }

  if (state.selectedId) {
    const active = entries.find((entry) => entry.id === state.selectedId);
    if (active) {
      state.root.classList.add("viewer-mode");
      state.preview.classList.remove("hidden");
      state.previewImg.src = active.full_url;
      state.previewImg.alt = active.filename;
      state.previewCaption.textContent = `${active.filename} • ${active.width}×${active.height}`;
      {
        buildPromptSections(active, state.promptText);
        state.copySeedBtn.classList.toggle("hidden", getSeedValue(active) === null);
      }
      updateNavButtons(state, entries);
    }
  } else {
    state.root.classList.remove("viewer-mode");
    state.preview.classList.add("hidden");
    state.previewImg.removeAttribute("src");
    state.previewCaption.textContent = "";
    state.promptText.innerHTML = "";
    state.copySeedBtn.classList.add("hidden");
    updateNavButtons(state, entries);
  }
}

function attachDom(node) {
  ensureStyles();

  const clearBtn = el("button", { type: "button" }, ["Clear All"]);
  const clearUnexportedBtn = el("button", { type: "button", className: "wg-clear-unexported-btn" }, ["Clear Unexported"]);
  const refreshBtn = el("button", { type: "button" }, ["Refresh"]);
  const exportBtn = el("button", { type: "button", className: "wg-export-btn hidden" }, ["↓ Export Selected"]);
  const deleteBtn = el("button", { type: "button", className: "wg-delete-btn hidden" }, ["🗑 Delete Selected"]);
  const compareBtn = el("button", { type: "button", className: "wg-compare-btn hidden" }, ["⇔ Compare"]);
  const count = el("span", { className: "wg-count" }, ["0 / 0"]);
  const previewImg = el("img", { className: "wg-preview-img", loading: "eager", alt: "Selected image preview" });
  const navLeft = el("button", { type: "button", className: "wg-nav hidden", "aria-label": "Previous image" }, ["‹"]);
  const navRight = el("button", { type: "button", className: "wg-nav hidden", "aria-label": "Next image" }, ["›"]);
  const previewCaption = el("div", { className: "wg-preview-caption" }, [""]);
  const copyPromptBtn = el("button", { type: "button" }, ["Copy prompt"]);
  const copySeedBtn = el("button", { type: "button", className: "wg-copy-seed-btn hidden" }, ["Copy seed"]);
  const promptText = el("div", { className: "wg-prompt-sections" }, []);
  const promptActions = el("div", { className: "wg-prompt-actions" }, [copyPromptBtn, copySeedBtn]);
  const previewStage = el("div", { className: "wg-preview-stage" }, [el("div", { className: "wg-preview-lane wg-preview-lane-left" }, [navLeft]), el("div", { className: "wg-preview-img-wrap" }, [previewImg]), el("div", { className: "wg-preview-lane wg-preview-lane-right" }, [navRight])]);

  // Compare mode UI
  const compareImgLeft = el("img", { alt: "Compare A", draggable: "false" });
  const compareImgRight = el("img", { alt: "Compare B", draggable: "false" });
  const compareSideLeft = el("div", { className: "wg-compare-side wg-compare-side-left" }, [compareImgLeft]);
  const compareSideRight = el("div", { className: "wg-compare-side wg-compare-side-right" }, [compareImgRight]);
  const compareDivider = el("div", { className: "wg-compare-divider" });
  const compareStage = el("div", { className: "wg-compare-stage" }, [compareSideLeft, compareSideRight, compareDivider]);
  const compareLabelLeft = el("div", { className: "wg-compare-label" });
  const compareLabelRight = el("div", { className: "wg-compare-label" });
  const compareLabels = el("div", { className: "wg-compare-labels" }, [compareLabelLeft, compareLabelRight]);
  const exitCompareBtn = el("button", { type: "button" }, ["✕ Exit compare"]);
  const compareWrap = el("div", { className: "wg-compare-wrap", style: { display: "none" } }, [compareStage, compareLabels, el("div", { className: "wg-prompt-actions" }, [exitCompareBtn])]);

  const preview = el("div", { className: "wg-preview hidden" }, [previewStage, previewCaption, promptActions, promptText, compareWrap]);
  const gallery = el("div", { className: "wg-gallery" });
  const initialThumbSize = loadThumbSizePreference();
  const thumbSlider = el("input", {
    type: "range",
    min: String(THUMB_MIN),
    max: String(THUMB_MAX),
    step: "10",
    value: String(initialThumbSize),
  });
  const saveMode = el("div", { className: "wg-save-mode" }, [""]);
  const dir = el("div", { className: "wg-dir" }, [""]);

  previewImg.addEventListener("click", () => closeViewer(node));
  navLeft.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    navigateViewer(node, -1);
  });
  navRight.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    navigateViewer(node, 1);
  });

  copyPromptBtn.addEventListener("click", async (e) => {
    e.preventDefault();
    const entry = node.__wgState?.payload?.entries?.find((item) => item.id === node.__wgState?.selectedId);
    const ok = await copyToClipboard(getPromptValue(entry));
    if (ok) {
      const prev = copyPromptBtn.textContent;
      copyPromptBtn.textContent = "Copied!";
      setTimeout(() => { copyPromptBtn.textContent = prev; }, 1500);
    } else {
      console.warn("Workflow Gallery copy prompt failed");
    }
  });

  copySeedBtn.addEventListener("click", async (e) => {
    e.preventDefault();
    const entry = node.__wgState?.payload?.entries?.find((item) => item.id === node.__wgState?.selectedId);
    const seed = getSeedValue(entry);
    if (!seed) return;
    const ok = await copyToClipboard(seed);
    if (ok) {
      const prev = copySeedBtn.textContent;
      copySeedBtn.textContent = "Copied!";
      setTimeout(() => { copySeedBtn.textContent = prev; }, 1500);
    }
  });

  const root = el("div", { className: "wg-root" }, [
    el("div", { className: "wg-topbar" }, [clearBtn, clearUnexportedBtn, refreshBtn, exportBtn, deleteBtn, compareBtn, count]),
    preview,
    gallery,
    el("div", { className: "wg-slider-row" }, [el("span", {}, ["Thumbnail size"]), thumbSlider]),
    el("div", { className: "wg-order-hint" }, ["Newest first"]),
    el("div", { className: "wg-meta", style: { fontSize: "11px", opacity: "0.8", wordBreak: "break-all", display: "grid", gap: "2px" } }, [saveMode, dir]),
  ]);

  node.__wgState = { root, clearBtn, clearUnexportedBtn, refreshBtn, exportBtn, deleteBtn, compareBtn, count, preview, previewStage, previewImg, previewCaption, promptText, promptActions, copyPromptBtn, copySeedBtn, navLeft, navRight, gallery, thumbSlider, saveMode, dir, compareWrap, compareStage, compareImgLeft, compareImgRight, compareSideLeft, compareSideRight, compareDivider, compareLabelLeft, compareLabelRight, selectedId: null, payload: null, selectedIds: [], compareIds: [], compareDividerPos: 50 };

  exportBtn.addEventListener("click", async (e) => {
    e.preventDefault();
    const state = node.__wgState;
    if (!state.selectedIds.length) return;
    const outputDir = state.payload?.output_directory || "";
    const prev = exportBtn.textContent;
    exportBtn.textContent = "Exporting...";
    exportBtn.disabled = true;
    try {
      const result = await exportEntries(node.id, [...state.selectedIds], outputDir);
      const exportedCount = result.exported?.length || 0;
      exportBtn.textContent = `✓ Exported ${exportedCount}!`;
      setTimeout(() => {
        exportBtn.textContent = prev;
        exportBtn.disabled = false;
      }, 2000);
      // Clear selection and refresh
      state.selectedIds = [];
      state.compareBtn.classList.add("hidden");
      state.exportBtn.classList.add("hidden");
      state.clearUnexportedBtn.classList.remove("hidden");
      const payload = await fetchGallery(node.id);
      renderGallery(node, payload);
    } catch (err) {
      exportBtn.textContent = "Export failed";
      exportBtn.disabled = false;
      setTimeout(() => { exportBtn.textContent = prev; }, 2000);
      console.warn("Workflow Gallery export failed", err);
    }
  });

  compareBtn.addEventListener("click", (e) => {
    e.preventDefault();
    // Pass selectedIds to compare mode
    node.__wgState.compareIds = [...node.__wgState.selectedIds];
    openCompareMode(node);
  });

  deleteBtn.addEventListener("click", async (e) => {
    e.preventDefault();
    const state = node.__wgState;
    if (!state.selectedIds.length) return;
    const count = state.selectedIds.length;
    if (!confirm(`Delete ${count} selected image${count !== 1 ? "s" : ""} from the gallery?`)) return;
    try {
      await deleteEntries(node.id, [...state.selectedIds]);
      state.selectedIds = [];
      state.exportBtn.classList.add("hidden");
      state.deleteBtn.classList.add("hidden");
      state.compareBtn.classList.add("hidden");
      state.clearUnexportedBtn.classList.remove("hidden");
      const payload = await fetchGallery(node.id);
      renderGallery(node, payload);
    } catch (err) {
      console.warn("Workflow Gallery delete failed", err);
    }
  });

  clearUnexportedBtn.addEventListener("click", async (e) => {
    e.preventDefault();
    const confirmed = confirm("Remove all unexported images from the gallery and cache?");
    if (!confirmed) return;
    try {
      await clearUnexported(node.id);
      const payload = await fetchGallery(node.id);
      renderGallery(node, payload);
    } catch (err) {
      console.warn("Workflow Gallery clear unexported failed", err);
    }
  });

  exitCompareBtn.addEventListener("click", (e) => {
    e.preventDefault();
    closeCompareMode(node);
  });

  attachCompareDrag(node.__wgState, compareStage);

  const applyThumbSizePreference = () => {
    const size = clampThumbSize(thumbSlider.value);
    thumbSlider.value = String(size);
    layoutGrid(gallery, size);
    saveThumbSizePreference(size);
    node.setDirtyCanvas(true, true);
  };

  thumbSlider.addEventListener("input", applyThumbSizePreference);
  thumbSlider.addEventListener("change", applyThumbSizePreference);
  applyThumbSizePreference();


  clearBtn.addEventListener("click", async (e) => {
    e.preventDefault();
    const confirmed = confirm("Clear all images from the gallery including exported ones?");
    if (!confirmed) return;
    try {
      await clearGallery(node.id);
      node.__wgState.selectedIds = [];
      node.__wgState.compareIds = [];
      node.__wgState.exportBtn.classList.add("hidden");
      node.__wgState.compareBtn.classList.add("hidden");
      const payload = await fetchGallery(node.id);
      renderGallery(node, payload);
    } catch (err) {
      console.warn("Workflow Gallery clear failed", err);
    }
  });

  refreshBtn.addEventListener("click", async (e) => {
    e.preventDefault();
    try {
      const payload = await fetchGallery(node.id);
      renderGallery(node, payload);
    } catch (err) {
      console.warn("Workflow Gallery refresh failed", err);
    }
  });


  const domWidget = node.addDOMWidget("workflow_gallery", "WGDOM", root, {
    serialize: false,
    hideOnZoom: false,
    getValue: () => null,
    setValue: () => {},
  });

  if (domWidget?.element?.style) {
    domWidget.element.style.width = "100%";
    domWidget.element.style.height = "100%";
    domWidget.element.style.display = "block";
  }

  // --- Height sync ---
  // Newer ComfyUI frontends no longer give DOM widgets a constrained height,
  // so the root's `height:100%` resolves to content height, overflow:auto
  // never triggers, and thumbnails clip at the node boundary with no scroll.
  // Fix: set an explicit pixel height, preferring the wrapper the frontend
  // sized for us, falling back to node canvas size minus the widget offset.
  const syncHeight = () => {
    let h = 0;
    const parent = root.parentElement;
    if (parent && parent.clientHeight > 60) {
      h = parent.clientHeight;
    }
    if (!h) {
      const y = typeof domWidget?.y === "number" ? domWidget.y : 0;
      const nodeH = node.size?.[1] || GALLERY_HEIGHT;
      h = Math.max(300, nodeH - y - 16);
    }
    root.style.height = `${h}px`;
  };
  node.__wgState.syncHeight = syncHeight;

  if (typeof ResizeObserver !== "undefined") {
    const ro = new ResizeObserver(() => syncHeight());
    node.__wgResizeObserver = ro;
    requestAnimationFrame(() => {
      if (root.parentElement) ro.observe(root.parentElement);
      syncHeight();
    });
  } else {
    requestAnimationFrame(syncHeight);
  }
  setTimeout(syncHeight, 300);

  // --- Wheel handling ---
  // Recent frontends let the canvas capture wheel events over DOM widgets,
  // which zooms the graph instead of scrolling the gallery. Stop propagation
  // so native scrolling inside the widget works; don't preventDefault.
  root.addEventListener("wheel", (e) => {
    e.stopPropagation();
  }, { passive: true });

  node.size = [Math.max(node.size?.[0] || 0, 420), Math.max(node.size?.[1] || 0, 900)];

  // Initial load with retry — handles the race condition where the browser
  // connects before _load_all_sessions() has finished populating state on startup.
  const loadWithRetry = async (attemptsLeft = 5, delayMs = 800) => {
    try {
      const payload = await fetchGallery(node.id);
      renderGallery(node, payload);
      // If empty and we have retries left, try again after a short delay
      // in case the session restore is still in progress
      if (!payload.entries?.length && attemptsLeft > 1) {
        setTimeout(() => loadWithRetry(attemptsLeft - 1, delayMs), delayMs);
      }
    } catch (err) {
      console.warn("Workflow Gallery load failed", err);
    }
  };
  loadWithRetry();
}

app.registerExtension({
  name: EXTENSION_NAME,

  async setup() {
    api.addEventListener("workflow_gallery_update", (event) => {
      const detail = event.detail;
      if (!detail?.node_id) return;
      const node = app.graph?._nodes_by_id?.[detail.node_id] || app.graph?._nodes?.find((n) => String(n.id) === String(detail.node_id));
      if (!node || node.comfyClass !== TARGET_CLASS) return;
      renderGallery(node, detail);
    });
  },

  async beforeRegisterNodeDef(nodeType) {
    if (nodeType.comfyClass !== TARGET_CLASS && nodeType.ComfyClass !== TARGET_CLASS) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      if (!this.__wgMounted) {
        this.__wgMounted = true;
        attachDom(this);
      }
      return result;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      if (this.__wgKeyHandler) {
        document.removeEventListener("keydown", this.__wgKeyHandler);
        this.__wgKeyHandler = null;
      }
      if (this.__wgResizeObserver) {
        this.__wgResizeObserver.disconnect();
        this.__wgResizeObserver = null;
      }
      return onRemoved?.apply(this, arguments);
    };

    const onResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) {
      const state = this.__wgState;
      state?.syncHeight?.();
      if (state && state.compareWrap?.style.display !== "none") {
        const stageH = getCompareStageHeight(this);
        state.compareStage.style.height = `${stageH}px`;
        requestAnimationFrame(() => updateCompareDivider(state));
      }
      return onResize?.apply(this, arguments);
    };
  },
});
