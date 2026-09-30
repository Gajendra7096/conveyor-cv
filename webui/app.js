"use strict";

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------- upload
const els = {
  form: $("runForm"), runButton: $("runButton"), formError: $("formError"),
  statusPill: $("statusPill"), statusText: $("statusText"),
  progress: $("progress"), progressFill: $("progressFill"), progressLabel: $("progressLabel"),
  kpiFrames: $("kpiFrames"), kpiRecords: $("kpiRecords"), kpiTracks: $("kpiTracks"),
  kpiSpeed: $("kpiSpeed"), kpiCalib: $("kpiCalib"), kpiRecog: $("kpiRecog"),
  previewImg: $("previewImg"), videoEl: $("videoEl"), stageEmpty: $("stageEmpty"),
  speedChart: $("speedChart"), trackChart: $("trackChart"),
  recordsTable: $("recordsTable"), tableFilter: $("tableFilter"),
  artifactList: $("artifactList"),
  videoFile: $("videoFile"), uploadStatus: $("uploadStatus"),
};

const humanSize = (bytes) => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
};

async function uploadVideo(file) {
  els.uploadStatus.className = "upload-status";
  els.uploadStatus.textContent = `Uploading ${file.name} (${humanSize(file.size)})…`;
  const body = new FormData();
  body.append("file", file, file.name);
  try {
    const response = await fetch("/api/upload", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    els.form.elements.video.value = data.path;
    els.uploadStatus.className = "upload-status ok";
    els.uploadStatus.textContent = `Saved ${data.filename} (${humanSize(data.bytes)})`;
  } catch (err) {
    els.uploadStatus.className = "upload-status error";
    els.uploadStatus.textContent = `Upload failed: ${err.message}`;
  }
}

els.videoFile.addEventListener("change", (event) => {
  const file = event.target.files && event.target.files[0];
  if (file) uploadVideo(file);
});

let pollTimer = null;
let currentJob = null;
let currentView = "preview";
let lastResult = null;

const isNum = (v) => typeof v === "number" && Number.isFinite(v);
const fmt = (v, d = 2) => (isNum(v) ? v.toFixed(d) : "—");

// --------------------------------------------------------------- artifacts
async function loadArtifacts() {
  try {
    const groups = await (await fetch("/api/artifacts")).json();
    els.artifactList.innerHTML = "";
    for (const group of groups) {
      if (!group.images.length) continue;
      const box = document.createElement("div");
      box.className = "artifact-group";
      const title = document.createElement("h3");
      title.textContent = group.title;
      const thumbs = document.createElement("div");
      thumbs.className = "thumbs";
      for (const image of group.images) {
        const img = document.createElement("img");
        img.src = image.url;
        img.alt = `${group.title} — ${image.name}`;
        img.loading = "lazy";
        img.addEventListener("click", () => window.open(image.url, "_blank"));
        thumbs.appendChild(img);
      }
      box.append(title, thumbs);
      els.artifactList.appendChild(box);
    }
  } catch (err) {
    els.artifactList.textContent = `Could not load artifacts: ${err.message}`;
  }
}

// ------------------------------------------------------------------ status
function setStatus(state, text) {
  els.statusPill.dataset.state = state;
  els.statusText.textContent = text;
}

function setProgress(processed, maxFrames) {
  const pct = maxFrames > 0 ? Math.min(100, (processed / maxFrames) * 100) : 0;
  els.progress.hidden = false;
  els.progressFill.style.width = `${pct}%`;
  els.progressLabel.textContent = `${processed} / ${maxFrames} frames`;
}

function resetKpis() {
  for (const key of ["kpiFrames", "kpiRecords", "kpiTracks", "kpiSpeed"]) {
    els[key].textContent = "—";
  }
  els.kpiCalib.textContent = "—";
  els.kpiCalib.className = "kpi-value";
  els.kpiRecog.textContent = "—";
  els.kpiRecog.className = "kpi-value";
}

// --------------------------------------------------------------------- run
function collectPayload() {
  const num = (id) => {
    const raw = els.form.elements[id].value.trim();
    return raw === "" ? null : Number(raw);
  };
  const payload = {
    video: els.form.elements.video.value.trim(),
    calibration: els.form.elements.calibration.value.trim(),
    recognition_root: els.form.elements.recognition_root.value.trim(),
    cm_per_pixel: num("cm_per_pixel"),
    max_frames: num("max_frames") ?? 300,
    stride: num("stride") ?? 1,
    flow_every: num("flow_every") ?? 5,
    max_boxes: num("max_boxes") ?? 12,
    keep_static: els.form.elements.keep_static.checked,
  };
  const vx = num("belt_vx");
  const vy = num("belt_vy");
  if (isNum(vx) && isNum(vy)) payload.belt_velocity = [vx, vy];
  for (const key of ["calibration", "recognition_root"]) {
    if (payload[key] === "") payload[key] = null;
  }
  return payload;
}

async function startRun(event) {
  event.preventDefault();
  els.formError.textContent = "";
  els.runButton.disabled = true;
  resetKpis();
  els.recordsTable.tBodies[0].innerHTML =
    '<tr class="empty-row"><td colspan="7">Running…</td></tr>';
  setStatus("running", "Starting");

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectPayload()),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    currentJob = data.id;
    setStatus("running", "Running");
    pollJob();
  } catch (err) {
    setStatus("error", "Failed");
    els.formError.textContent = err.message;
    els.formError.className = "hint error";
    els.runButton.disabled = false;
  }
}

function pollJob() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(async () => {
    if (!currentJob) return;
    try {
      const job = await (await fetch(`/api/status/${currentJob}`)).json();
      setProgress(job.processed, job.max_frames);

      if (job.has_preview) {
        const url = `${job.preview_url}?t=${Date.now()}`;
        if (currentView === "preview") {
          els.previewImg.src = url;
          els.previewImg.hidden = false;
          els.stageEmpty.hidden = true;
        }
      }
      if (isNum(job.latest_speed)) els.kpiSpeed.textContent = fmt(job.latest_speed, 3);
      els.kpiTracks.textContent = job.tracks || "—";

      if (job.status === "done") {
        els.runButton.disabled = false;
        setStatus("done", "Complete");
        renderResult(job.result);
        loadArtifacts();
      } else if (job.status === "error") {
        els.runButton.disabled = false;
        setStatus("error", "Failed");
        els.formError.textContent = job.error || "unknown error";
        els.formError.className = "hint error";
      } else {
        pollJob();
      }
    } catch (err) {
      setStatus("error", "Lost connection");
      els.formError.textContent = err.message;
      els.formError.className = "hint error";
      els.runButton.disabled = false;
    }
  }, 400);
}

// ------------------------------------------------------------------ result
function renderResult(result) {
  if (!result) return;
  lastResult = result;
  const rows = result.rows || [];
  els.kpiFrames.textContent = result.frames_processed;
  els.kpiRecords.textContent = result.records;

  const ids = new Set(rows.map((r) => r.track_id));
  els.kpiTracks.textContent = ids.size;

  const speeds = rows.map((r) => Number(r.speed_cm_s)).filter(isNum);
  if (speeds.length) {
    const mean = speeds.reduce((a, b) => a + b, 0) / speeds.length;
    els.kpiSpeed.textContent = fmt(mean, 3);
  }

  els.kpiCalib.textContent = result.calibrated ? "yes" : "no";
  els.kpiCalib.className = `kpi-value ${result.calibrated ? "yes" : "no"}`;
  els.kpiRecog.textContent = result.recognition_enabled ? "yes" : "no";
  els.kpiRecog.className = `kpi-value ${result.recognition_enabled ? "yes" : "no"}`;

  renderTable(rows);
  drawSpeedChart(rows);
  drawTrackChart(rows);

  const video = (result.artifacts || {}).annotated_video;
  if (video && currentView === "video") {
    els.videoEl.src = `/api/artifact/${video}`;
    els.videoEl.hidden = false;
    els.previewImg.hidden = true;
    els.stageEmpty.hidden = true;
  }
}

function renderTable(rows) {
  const needle = els.tableFilter.value.trim().toLowerCase();
  const body = els.recordsTable.tBodies[0];
  const filtered = rows.filter((r) =>
    !needle ||
    String(r.track_id).includes(needle) ||
    String(r.type).toLowerCase().includes(needle)
  );
  body.innerHTML = "";
  if (!filtered.length) {
    body.innerHTML = `<tr class="empty-row"><td colspan="7">${
      rows.length ? "No records match the filter." : "No records were produced."
    }</td></tr>`;
    return;
  }
  for (const r of filtered.slice(0, 600)) {
    const tr = document.createElement("tr");
    const size = isNum(r.size_width_cm)
      ? `${fmt(r.size_width_cm)} × ${fmt(r.size_height_cm)}`
      : "—";
    const raw = r.raw_x == null ? "—" : `${fmt(r.raw_x, 1)}, ${fmt(r.raw_y, 1)}`;
    const filteredPos = `${fmt(r.filtered_x, 1)}, ${fmt(r.filtered_y, 1)}`;
    const speed = isNum(r.speed_cm_s) ? fmt(r.speed_cm_s, 3) : "—";
    tr.innerHTML =
      `<td>${r.frame}</td><td>${r.track_id}</td><td>${r.type || "—"}</td>` +
      `<td>${size}</td><td>${speed}</td><td>${raw}</td><td>${filteredPos}</td>`;
    body.appendChild(tr);
  }
  if (filtered.length > 600) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<tr class="empty-row"><td colspan="7">Showing first 600 of ${
      filtered.length
    } records.</td></tr>`;
    body.appendChild(tr);
  }
}

// ------------------------------------------------------------------ charts
function prepCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 400;
  const height = canvas.clientHeight || 200;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

function drawAxes(ctx, box, xLabel, yLabel, xMax, yMin, yMax) {
  ctx.strokeStyle = "#232c39";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#5d6b7d";
  ctx.font = "10px ui-monospace, monospace";
  ctx.beginPath();
  ctx.moveTo(box.x, box.y);
  ctx.lineTo(box.x, box.y + box.h);
  ctx.lineTo(box.x + box.w, box.y + box.h);
  ctx.stroke();
  ctx.fillText(xLabel, box.x + box.w, box.y + box.h + 16);
  ctx.fillText(yLabel, box.x - 4, box.y - 6);

  if (isNum(yMax) && isNum(yMin)) {
    ctx.fillText(yMax.toFixed(1), box.x + 3, box.y + 10);
    ctx.fillText(yMin.toFixed(1), box.x + 3, box.y + box.h - 3);
  }
  if (isNum(xMax)) {
    ctx.fillText(String(Math.round(xMax)), box.x + box.w - 20, box.y + box.h + 12);
  }
}

function drawSpeedChart(rows) {
  const { ctx, width, height } = prepCanvas(els.speedChart);
  const box = { x: 46, y: 18, w: width - 62, h: height - 46 };
  const points = rows
    .map((r) => ({ f: r.frame, v: Number(r.speed_cm_s) }))
    .filter((p) => isNum(p.v));
  if (points.length < 2) {
    ctx.fillStyle = "#5d6b7d";
    ctx.font = "12px system-ui";
    ctx.fillText("No speed measurements yet.", box.x, box.y + 30);
    return;
  }
  const byFrame = new Map();
  for (const p of points) byFrame.set(p.f, p.v);
  const sorted = [...byFrame.entries()].sort((a, b) => a[0] - b[0]);
  const xs = sorted.map((e) => e[0]);
  const vs = sorted.map((e) => e[1]);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...vs);
  const yMax = Math.max(...vs);
  const pad = (yMax - yMin) * 0.15 || 0.5;
  const lo = yMin - pad;
  const hi = yMax + pad;

  drawAxes(ctx, box, "frame", "cm/s", xMax, lo, hi);
  const sx = (f) => box.x + ((f - xMin) / (xMax - xMin || 1)) * box.w;
  const sy = (v) => box.y + box.h - ((v - lo) / (hi - lo || 1)) * box.h;

  const gradient = ctx.createLinearGradient(0, box.y, 0, box.y + box.h);
  gradient.addColorStop(0, "rgba(74,158,255,0.30)");
  gradient.addColorStop(1, "rgba(74,158,255,0.02)");
  ctx.beginPath();
  ctx.moveTo(sx(sorted[0][0]), box.y + box.h);
  for (const [f, v] of sorted) ctx.lineTo(sx(f), sy(v));
  ctx.lineTo(sx(sorted[sorted.length - 1][0]), box.y + box.h);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  ctx.beginPath();
  sorted.forEach(([f, v], i) => (i ? ctx.lineTo(sx(f), sy(v)) : ctx.moveTo(sx(f), sy(v))));
  ctx.strokeStyle = "#4a9eff";
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawTrackChart(rows) {
  const { ctx, width, height } = prepCanvas(els.trackChart);
  const box = { x: 46, y: 18, w: width - 62, h: height - 46 };
  const palette = ["#3fb950", "#4a9eff", "#d29922", "#f778ba", "#39c5cf", "#c9d1d9"];
  const tracks = new Map();
  for (const r of rows) {
    if (!isNum(r.filtered_x) || !isNum(r.filtered_y)) continue;
    if (!tracks.has(r.track_id)) tracks.set(r.track_id, []);
    tracks.get(r.track_id).push([r.filtered_x, r.filtered_y]);
  }
  if (!tracks.size) {
    ctx.fillStyle = "#5d6b7d";
    ctx.font = "12px system-ui";
    ctx.fillText("No tracked positions yet.", box.x, box.y + 30);
    return;
  }
  const all = [...tracks.values()].flat();
  const xs = all.map((p) => p[0]);
  const ys = all.map((p) => p[1]);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const padX = (xMax - xMin) * 0.08 || 10;
  const padY = (yMax - yMin) * 0.08 || 10;
  drawAxes(ctx, box, "x px", "y px", xMax, yMin - padY, yMax + padY);

  const sx = (x) => box.x + ((x - (xMin - padX)) / ((xMax + padX) - (xMin - padX) || 1)) * box.w;
  const sy = (y) =>
    box.y + box.h - ((y - (yMin - padY)) / ((yMax + padY) - (yMin - padY) || 1)) * box.h;

  let i = 0;
  for (const [id, path] of tracks) {
    ctx.beginPath();
    path.forEach((p, idx) => (idx ? ctx.lineTo(sx(p[0]), sy(p[1])) : ctx.moveTo(sx(p[0]), sy(p[1]))));
    ctx.strokeStyle = palette[i % palette.length];
    ctx.lineWidth = 1.8;
    ctx.stroke();
    const head = path[path.length - 1];
    ctx.beginPath();
    ctx.arc(sx(head[0]), sy(head[1]), 3, 0, Math.PI * 2);
    ctx.fillStyle = palette[i % palette.length];
    ctx.fill();
    i += 1;
  }
}

// ------------------------------------------------------------------- wiring
els.form.addEventListener("submit", startRun);
els.tableFilter.addEventListener("input", () => {
  if (lastResult) renderTable(lastResult.rows || []);
});

for (const button of document.querySelectorAll(".seg-btn")) {
  button.addEventListener("click", () => {
    document.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
    button.classList.add("active");
    currentView = button.dataset.view;
    els.videoEl.hidden = currentView !== "video";
    els.previewImg.hidden = currentView !== "preview";
    if (currentView === "video" && lastResult) renderResult(lastResult);
    else if (currentView === "preview" && currentJob) pollJob();
  });
}

window.addEventListener("resize", () => {
  if (!lastResult) return;
  const rows = lastResult.rows || [];
  drawSpeedChart(rows);
  drawTrackChart(rows);
});

resetKpis();
loadArtifacts();