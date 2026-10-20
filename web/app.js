// Browser-Teil der Demo: Kamera holen, Python aufrufen, HUD zeichnen.
// Die eigentliche Rechnung (Erkennung, Kalman-Filter, Fehlermessung)
// passiert in Python: siehe web_core.py und predictor.py.

import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";

const PROC_W = 160;           // Kamerabild wird für Python auf 160 px verkleinert
const SIM_FPS = 30;
const MODELS = ["Stillstand", "Geradeaus", "Beschleunigt"];
const COLORS = { Stillstand: "#8a96a3", Geradeaus: "#78ffa0", Beschleunigt: "#e0a43e" };

const $ = (id) => document.getElementById(id);
const view = $("view");
const ctx = view.getContext("2d");
const statusEl = $("status");

let demo = null;              // Python-Objekt aus web_core.py
let mode = null;              // "cam" | "sim" | null
let video = null;
let stream = null;
let proc = null, procCtx = null;
let lastProc = 0, t0 = 0, simN = 0, simAcc = 0, lastNow = 0, frameN = 0;

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

// ---------------------------------------------------------------- Python laden

async function initPython() {
  const pyodide = await loadPyodide();
  setStatus("numpy wird geladen …");
  await pyodide.loadPackage("numpy");
  for (const file of ["predictor.py", "web_core.py"]) {
    const res = await fetch(file);
    if (!res.ok) throw new Error(`${file} nicht gefunden`);
    pyodide.FS.writeFile(file, await res.text());
  }
  pyodide.runPython("import sys; sys.path.insert(0, '.')");
  demo = pyodide.pyimport("web_core").demo;
  setStatus("Bereit. Python läuft im Browser.");
  for (const b of ["btn-cam", "btn-sim", "btn-reset"]) $(b).disabled = false;
}

function toJs(proxy) {
  const out = proxy.toJs({ dict_converter: Object.fromEntries });
  proxy.destroy();
  return out;
}

// ---------------------------------------------------------------- Modi

async function startCamera() {
  stopAll();
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 540 }, facingMode: "user" },
      audio: false,
    });
  } catch (e) {
    setStatus("Kein Kamerazugriff. Erlaube die Kamera oder nimm die Simulation.", true);
    return;
  }
  video = document.createElement("video");
  video.srcObject = stream;
  video.muted = true;
  video.playsInline = true;
  await video.play();

  view.width = video.videoWidth;
  view.height = video.videoHeight;
  proc = document.createElement("canvas");
  proc.width = PROC_W;
  proc.height = Math.round(PROC_W * video.videoHeight / video.videoWidth);
  procCtx = proc.getContext("2d", { willReadFrequently: true });

  demo.reset("cam");
  begin("cam", "Bewege langsam die Hand. Der Hintergrund sollte ruhig sein.");
}

function startSim() {
  stopAll();
  view.width = 960;
  view.height = 540;
  demo.reset("sim");
  simN = 0;
  simAcc = 0;
  begin("sim", "Simulierte Bewegung mit zufälligen Richtungswechseln.");
}

function begin(m, hint) {
  mode = m;
  $("placeholder").style.display = "none";
  $("hint").textContent = hint;
  t0 = lastNow = lastProc = performance.now();
  frameN = 0;
  updateStats();
  requestAnimationFrame(loop);
}

function stopAll() {
  mode = null;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video = null;
}

// ---------------------------------------------------------------- Hauptschleife

function loop(now) {
  if (!mode) return;

  if (mode === "cam") {
    // höchstens ~30 Bilder pro Sekunde an Python geben
    if (now - lastProc >= 1000 / 30 && video.readyState >= 2) {
      const dt = (now - lastProc) / 1000;
      lastProc = now;
      drawMirrored(procCtx, video, proc.width, proc.height);
      const pixels = procCtx.getImageData(0, 0, proc.width, proc.height).data;
      const res = toJs(demo.camera_frame((now - t0) / 1000, dt, pixels, proc.width, proc.height));
      const s = view.width / proc.width;
      drawMirrored(ctx, video, view.width, view.height);
      drawHud(scale(res.pos, s), res.path ? res.path.map((p) => scale(p, s)) : null);
      tick();
    }
  } else {
    // Simulation läuft mit festen 30 Schritten pro Sekunde, egal wie schnell der Bildschirm ist
    simAcc += Math.min((now - lastNow) / 1000, 0.25);
    lastNow = now;
    let res = null;
    while (simAcc >= 1 / SIM_FPS) {
      simAcc -= 1 / SIM_FPS;
      res = toJs(demo.sim_frame(simN / SIM_FPS, 1 / SIM_FPS));
      simN++;
      tick();
    }
    if (res) {
      drawSimBackground();
      drawBall(res.pos);
      drawHud(res.pos, res.path);
    }
  }
  requestAnimationFrame(loop);
}

function tick() {
  frameN++;
  if (frameN % 15 === 0) updateStats();
}

const scale = (p, s) => (p ? [p[0] * s, p[1] * s] : null);

function drawMirrored(c, src, w, h) {
  // Spiegelbild wie bei einem Spiegel: Bewegung nach rechts erscheint rechts
  c.save();
  c.translate(w, 0);
  c.scale(-1, 1);
  c.drawImage(src, 0, 0, w, h);
  c.restore();
}

// ---------------------------------------------------------------- Zeichnen

function drawSimBackground() {
  ctx.fillStyle = "#0d1117";
  ctx.fillRect(0, 0, view.width, view.height);
  ctx.strokeStyle = "#18202a";
  ctx.lineWidth = 1;
  for (let x = 0; x <= view.width; x += 60) line(x, 0, x, view.height);
  for (let y = 0; y <= view.height; y += 60) line(0, y, view.width, y);
}

function drawBall(p) {
  if (!p) return;
  ctx.fillStyle = "#3cc850";
  ctx.beginPath();
  ctx.arc(p[0], p[1], 22, 0, Math.PI * 2);
  ctx.fill();
}

function drawHud(pos, path) {
  const k = view.width / 960;            // Strichstärken an Bildgröße anpassen
  ctx.lineCap = "round";
  if (pos) {
    ctx.strokeStyle = "#78ffa0";
    ctx.lineWidth = 2.5 * k;
    ctx.beginPath();
    ctx.arc(pos[0], pos[1], 28 * k, 0, Math.PI * 2);
    ctx.stroke();
    line(pos[0] - 40 * k, pos[1], pos[0] - 31 * k, pos[1]);
    line(pos[0] + 31 * k, pos[1], pos[0] + 40 * k, pos[1]);
  }
  if (path && path.length) {
    let prev = pos;
    path.forEach((p, i) => {
      const a = 1 - (i / path.length) * 0.7;
      ctx.strokeStyle = ctx.fillStyle = `rgba(120,255,160,${a})`;
      ctx.lineWidth = 2 * k;
      if (prev) line(prev[0], prev[1], p[0], p[1]);
      ctx.beginPath();
      ctx.arc(p[0], p[1], (i === path.length - 1 ? 6 : 3.5) * k, 0, Math.PI * 2);
      ctx.fill();
      prev = p;
    });
    const end = path[path.length - 1];
    ctx.strokeStyle = ctx.fillStyle = "#ffb43c";
    ctx.lineWidth = 2.5 * k;
    line(end[0] - 13 * k, end[1], end[0] + 13 * k, end[1]);
    line(end[0], end[1] - 13 * k, end[0], end[1] + 13 * k);
    ctx.font = `500 ${Math.round(15 * k)}px "IBM Plex Mono", monospace`;
    ctx.fillText("+1,0 s", end[0] + 14 * k, end[1] - 12 * k);
  }
}

function line(x0, y0, x1, y1) {
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
}

// ---------------------------------------------------------------- Statistik

function updateStats() {
  const s = toJs(demo.stats());
  const last = (m) => s.errors[m][s.errors[m].length - 1];
  const vals = MODELS.map(last).filter((v) => v != null);
  const max = vals.length ? Math.max(...vals) : 1;

  for (const m of MODELS) {
    const el = document.querySelector(`.model[data-model="${m}"]`);
    const v = last(m);
    el.querySelector(".val").textContent = v == null ? "–" : `${v.toFixed(1)} px`;
    el.querySelector(".bar span").style.width = v == null ? "0" : `${(v / max) * 100}%`;
    el.classList.toggle("best", s.count >= 30 && m === s.best);
  }
  $("count").textContent = s.count
    ? `${s.count} Vorhersagen ausgewertet.`
    : "Noch keine Vorhersagen ausgewertet.";
  drawChart(s.errors);
}

function drawChart(errors) {
  const c = $("chart");
  const g = c.getContext("2d");
  const W = c.width, H = c.height, L = 70, R = 16, T = 16, B = 46;
  g.clearRect(0, 0, W, H);

  const all = MODELS.flatMap((m) => errors[m]).filter((v) => v != null);
  const ymax = all.length ? Math.max(...all) * 1.1 : 100;
  const x = (i) => L + (i / 9) * (W - L - R);
  const y = (v) => T + (1 - v / ymax) * (H - T - B);

  g.font = '22px "IBM Plex Mono", monospace';
  g.fillStyle = "#7d8b99";
  g.strokeStyle = "#1c2530";
  g.lineWidth = 2;
  for (let k = 0; k <= 4; k++) {
    const v = (ymax / 4) * k;
    g.beginPath(); g.moveTo(L, y(v)); g.lineTo(W - R, y(v)); g.stroke();
    g.textAlign = "right";
    g.fillText(Math.round(v), L - 10, y(v) + 7);
  }
  g.textAlign = "center";
  for (const i of [0, 4, 9]) g.fillText(`${((i + 1) / 10).toFixed(1)} s`, x(i), H - 12);

  for (const m of MODELS) {
    const pts = errors[m];
    g.strokeStyle = COLORS[m];
    g.lineWidth = 4;
    g.beginPath();
    let started = false;
    pts.forEach((v, i) => {
      if (v == null) return;
      if (!started) { g.moveTo(x(i), y(v)); started = true; } else g.lineTo(x(i), y(v));
    });
    g.stroke();
  }
}

// ---------------------------------------------------------------- Start

$("btn-cam").addEventListener("click", startCamera);
$("btn-sim").addEventListener("click", startSim);
$("btn-reset").addEventListener("click", () => {
  if (mode) demo.reset(mode);
  simN = 0;
  t0 = performance.now();
  updateStats();
});

initPython().catch((e) => {
  console.error(e);
  setStatus(`Python konnte nicht geladen werden: ${e.message}`, true);
});
