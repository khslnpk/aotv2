/* =============================================================================
   Sleep Twin frontend
   - Fetches subject list, picks the first test subject by default
   - Renders hero dial + bento score cards + constellation hypnogram + what-if
   - Click a score card -> flip overlay with the per-component breakdown
   - Press M (or click button) -> methodology drawer
   ============================================================================= */

const SCORE_DEFINITIONS = [
  {
    key: "recovery",
    title: "Recovery",
    score_key: "recovery_score",
    color: "var(--c-recovery)",
    blurb: "How well your body restored itself overnight. Sleep contribution + autonomic dip.",
    size: "size-lg",
    spark: "◐",
  },
  {
    key: "sleep_quality",
    title: "Sleep quality",
    score_key: "sleep_quality_score",
    color: "var(--c-sleep)",
    blurb: "Architecture of the night: duration, efficiency, REM/NREM balance, fragmentation.",
    size: "size-lg",
    spark: "✦",
  },
  {
    key: "stress",
    title: "Stress index",
    score_key: "stress_index",
    color: "var(--c-stress)",
    blurb: "Autonomic load. Low HR dip and low HRV proxy = elevated sympathetic tone.",
    size: "size-md",
    spark: "▲",
  },
  {
    key: "fatigue",
    title: "Fatigue",
    score_key: "fatigue_score",
    color: "var(--c-fatigue)",
    blurb: "Inverse recovery + daily load + sleep deficit.",
    size: "size-md",
    spark: "◔",
  },
  {
    key: "energy",
    title: "Energy",
    score_key: "energy_level",
    color: "var(--c-energy)",
    blurb: "Composite — felt-sense energy is mostly recovery + (100 − fatigue).",
    size: "size-md",
    spark: "⚡",
  },
  {
    key: "sleep_debt",
    title: "Sleep debt",
    score_key: "sleep_debt_score",
    color: "var(--c-debt)",
    blurb: "Tonight's deficit vs an 8 h target. Single-night only.",
    size: "size-md",
    spark: "⊘",
  },
];

const SCORE_BY_KEY = Object.fromEntries(SCORE_DEFINITIONS.map(d => [d.key, d]));

let state = {
  subjects: [],
  currentSubject: null,
  analysis: null,
  overlay: { showHr: true, showMotion: true, showPsg: false },
};

// =============================================================================
// Boot
// =============================================================================

document.addEventListener("DOMContentLoaded", async () => {
  bindUi();
  drawDialBackground();
  buildBento();

  try {
    const subjects = await fetchJSON("/api/subjects");
    state.subjects = subjects;
    populateSubjectSelect(subjects);
    const initial = (subjects.find(s => s.split === "test") || subjects[0]).id;
    await loadSubject(initial);
  } catch (err) {
    console.error(err);
    document.querySelector(".hero-caption").textContent =
      `Couldn't load /api/subjects — is the server running? (${err.message})`;
  }

  loadMethodology();
});

function bindUi() {
  document.getElementById("subject-select").addEventListener("change", e => {
    loadSubject(e.target.value);
  });

  document.querySelectorAll(".chip-toggle").forEach(btn => {
    btn.addEventListener("click", () => {
      btn.classList.toggle("active");
      const k = btn.dataset.overlay;
      const map = { hr: "showHr", motion: "showMotion", psg: "showPsg" };
      state.overlay[map[k]] = btn.classList.contains("active");
      renderConstellation();
    });
  });

  document.getElementById("dial-sleep").addEventListener("input", onWhatIfChange);
  document.getElementById("dial-steps").addEventListener("input", onWhatIfChange);

  document.getElementById("open-methodology").addEventListener("click", openDrawer);
  document.getElementById("drawer-close").addEventListener("click", closeDrawer);
  document.getElementById("scrim").addEventListener("click", () => { closeDrawer(); closeFlip(); });

  document.getElementById("flip-close").addEventListener("click", closeFlip);

  document.addEventListener("keydown", e => {
    if (e.key === "Escape") { closeFlip(); closeDrawer(); }
    if (e.key === "m" || e.key === "M") {
      if (document.activeElement && document.activeElement.tagName === "INPUT") return;
      openDrawer();
    }
  });
}

// =============================================================================
// Data
// =============================================================================

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

function populateSubjectSelect(subjects) {
  const sel = document.getElementById("subject-select");
  sel.innerHTML = "";
  for (const s of subjects) {
    const opt = document.createElement("option");
    opt.value = s.id;
    opt.textContent = `${s.id}  ·  ${s.duration_hours} h  ·  ${s.split}`;
    sel.appendChild(opt);
  }
}

async function loadSubject(id) {
  state.currentSubject = id;
  document.getElementById("subject-select").value = id;
  const subj = state.subjects.find(s => s.id === id);
  if (subj) {
    const pill = document.getElementById("split-pill");
    pill.textContent = subj.split;
    pill.dataset.split = subj.split;
  }
  try {
    const data = await fetchJSON(`/api/subjects/${encodeURIComponent(id)}`);
    state.analysis = data;
    render(data);
    document.getElementById("dial-sleep").value = 0;
    document.getElementById("dial-steps").value = 0;
    document.getElementById("dial-sleep-val").textContent = 0;
    document.getElementById("dial-steps-val").textContent = 0;
    renderDeltaTiles(data.state, data.state);
  } catch (err) {
    console.error(err);
  }
}

// =============================================================================
// Rendering
// =============================================================================

function render(d) {
  renderHero(d);
  renderBento(d);
  renderConstellation();
}

function renderHero(d) {
  const recovery = d.state.recovery_score;
  document.getElementById("hero-value").textContent = recovery.toFixed(0);
  document.getElementById("dial-tick").textContent = recovery.toFixed(0);
  drawDialForeground(recovery);

  document.getElementById("meta-duration").textContent =
    (d.sleep_summary.total_sleep_time_min / 60).toFixed(1) + " h";
  document.getElementById("meta-efficiency").textContent =
    (d.sleep_summary.sleep_efficiency * 100).toFixed(0) + "%";
  document.getElementById("meta-agree").textContent =
    (d.model_agreement_with_psg * 100).toFixed(0) + "%";
  document.getElementById("meta-awake").textContent = d.sleep_summary.awakenings;
}

function buildBento() {
  const bento = document.getElementById("bento");
  bento.innerHTML = "";
  for (const def of SCORE_DEFINITIONS) {
    const card = document.createElement("div");
    card.className = `card ${def.size}`;
    card.dataset.score = def.key;
    card.innerHTML = `
      <div class="card-head">
        <span class="card-title">${def.title}</span>
        <span class="card-spark">${def.spark}</span>
      </div>
      <div class="card-number"><span data-value>—</span><span class="card-unit">/100</span></div>
      <div class="card-blurb">${def.blurb}</div>
      <div class="card-bar"><div class="card-bar-fill" data-fill style="width:0%"></div></div>
      <div class="card-foot">
        <span>tap for breakdown</span>
        <strong data-foot></strong>
      </div>
    `;
    card.addEventListener("click", () => openFlip(def.key));
    bento.appendChild(card);
  }
}

function renderBento(d) {
  for (const def of SCORE_DEFINITIONS) {
    const card = document.querySelector(`.card[data-score="${def.key}"]`);
    if (!card) continue;
    const value = d.state[def.score_key];
    card.querySelector("[data-value]").textContent = Math.round(value);
    card.querySelector("[data-fill]").style.width = Math.max(0, Math.min(100, value)) + "%";
    card.style.setProperty("--accent-width", `${Math.max(0, Math.min(100, value))}%`);
    card.querySelector("[data-foot]").textContent = labelFromValue(def.key, value);
    const blurbEl = card.querySelector(".card-blurb");
    if (blurbEl && !blurbEl.style.color) {
      blurbEl.style.color = "var(--text-muted)";
      blurbEl.style.fontSize = "12px";
      blurbEl.style.lineHeight = "1.45";
      blurbEl.style.marginBottom = "10px";
      blurbEl.style.minHeight = "0";
    }
  }
}

function labelFromValue(key, v) {
  // tiny qualitative label per score
  if (key === "stress" || key === "fatigue" || key === "sleep_debt") {
    if (v < 30) return "low";
    if (v < 60) return "moderate";
    return "high";
  }
  if (v < 40) return "low";
  if (v < 70) return "moderate";
  if (v < 85) return "strong";
  return "excellent";
}

// ------------------ Hero dial ------------------

function drawDialBackground() {
  // 8 segments forming a 270° arc from 135° to 405° (i.e. -90° to 180°)
  const bg = document.getElementById("dial-bg-segments");
  bg.innerHTML = "";
  const segments = 8;
  const startDeg = -135;
  const sweepTotal = 270;
  const gap = 4;
  const segSweep = (sweepTotal - gap * (segments - 1)) / segments;
  for (let i = 0; i < segments; i++) {
    const a0 = startDeg + i * (segSweep + gap);
    const a1 = a0 + segSweep;
    bg.appendChild(createArc(a0, a1, 90, "rgba(255,255,255,0.05)", 18));
  }
}

function drawDialForeground(value) {
  const fg = document.getElementById("dial-fg-segments");
  fg.innerHTML = "";
  const segments = 8;
  const filled = (value / 100) * segments;
  const startDeg = -135;
  const sweepTotal = 270;
  const gap = 4;
  const segSweep = (sweepTotal - gap * (segments - 1)) / segments;
  for (let i = 0; i < segments; i++) {
    if (i >= Math.ceil(filled)) break;
    const fillRatio = i + 1 <= filled ? 1 : filled - i;
    const a0 = startDeg + i * (segSweep + gap);
    const a1 = a0 + segSweep * fillRatio;
    fg.appendChild(createArc(a0, a1, 90, "url(#dial-grad)", 18));
  }
}

function createArc(a0Deg, a1Deg, r, stroke, width) {
  const cx = 110, cy = 110;
  const a0 = (a0Deg * Math.PI) / 180;
  const a1 = (a1Deg * Math.PI) / 180;
  const x0 = cx + r * Math.cos(a0);
  const y0 = cy + r * Math.sin(a0);
  const x1 = cx + r * Math.cos(a1);
  const y1 = cy + r * Math.sin(a1);
  const large = a1 - a0 > Math.PI ? 1 : 0;
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("d", `M ${x0} ${y0} A ${r} ${r} 0 ${large} 1 ${x1} ${y1}`);
  path.setAttribute("stroke", stroke);
  path.setAttribute("stroke-width", String(width));
  path.setAttribute("stroke-linecap", "round");
  path.setAttribute("fill", "none");
  return path;
}

// ------------------ Constellation hypnogram ------------------

function renderConstellation() {
  const d = state.analysis;
  if (!d) return;
  const el = document.getElementById("constellation");
  el.innerHTML = "";

  const W = el.clientWidth || 1200;
  const H = el.clientHeight || 220;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "none");

  const stages = d.predicted_stages;
  const n = stages.length;
  if (n === 0) { el.appendChild(svg); return; }
  const dx = W / n;

  // Stage band centers (Wake top, NREM mid, REM bottom)
  const bands = { 0: H * 0.22, 1: H * 0.55, 2: H * 0.82 };
  const colors = { 0: "#ec4899", 1: "#8b5cf6", 2: "#00d9c0" };

  // Background NREM "field" — translucent area
  let pathD = `M 0 ${H}`;
  for (let i = 0; i < n; i++) {
    const stage = stages[i];
    const y = stage === 1 ? bands[1] + 16 : H;
    pathD += ` L ${i * dx} ${y}`;
  }
  pathD += ` L ${W} ${H} Z`;
  const field = document.createElementNS("http://www.w3.org/2000/svg", "path");
  field.setAttribute("d", pathD);
  field.setAttribute("fill", "rgba(139,92,246,0.10)");
  svg.appendChild(field);

  // Per-epoch dot / spike
  for (let i = 0; i < n; i++) {
    const stage = stages[i];
    const x = i * dx + dx / 2;
    const y = bands[stage];

    if (stage === 0) {
      // Wake spike — short vertical line
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", x); line.setAttribute("x2", x);
      line.setAttribute("y1", y - 8); line.setAttribute("y2", y + 8);
      line.setAttribute("stroke", colors[0]);
      line.setAttribute("stroke-width", "1.4");
      line.setAttribute("opacity", "0.9");
      svg.appendChild(line);
    } else if (stage === 2) {
      // REM star — pulsing bright dot
      const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      c.setAttribute("cx", x); c.setAttribute("cy", y);
      c.setAttribute("r", 2.4);
      c.setAttribute("fill", colors[2]);
      c.setAttribute("class", "rem-star");
      c.setAttribute("filter", "url(#star-glow)");
      svg.appendChild(c);
    } else {
      // NREM dim dot
      const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      c.setAttribute("cx", x); c.setAttribute("cy", y);
      c.setAttribute("r", 1.4);
      c.setAttribute("fill", colors[1]);
      c.setAttribute("opacity", "0.55");
      svg.appendChild(c);
    }
  }

  // Glow filter for REM
  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  defs.innerHTML = `
    <filter id="star-glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="1.5" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  `;
  svg.insertBefore(defs, svg.firstChild);

  // HR overlay
  if (state.overlay.showHr && d.hr_timeline && d.hr_timeline.bpm.length) {
    const path = makeOverlayPath(d.hr_timeline.time_sec, d.hr_timeline.bpm, W, H, 0.1, 0.9, true);
    if (path) {
      const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
      p.setAttribute("d", path);
      p.setAttribute("stroke", "rgba(0,217,192,0.75)");
      p.setAttribute("stroke-width", "1.4");
      p.setAttribute("fill", "none");
      svg.appendChild(p);
    }
  }
  // Motion overlay (as area at bottom)
  if (state.overlay.showMotion && d.motion_timeline && d.motion_timeline.enmo.length) {
    const path = makeOverlayPath(d.motion_timeline.time_sec, d.motion_timeline.enmo, W, H, 0.92, 0.05, false);
    if (path) {
      const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
      p.setAttribute("d", path + ` L ${W} ${H} L 0 ${H} Z`);
      p.setAttribute("fill", "rgba(245,158,11,0.18)");
      p.setAttribute("stroke", "rgba(245,158,11,0.4)");
      p.setAttribute("stroke-width", "0.8");
      svg.appendChild(p);
    }
  }
  // PSG truth overlay (thin ribbon below)
  if (state.overlay.showPsg && d.true_stages && d.true_stages.length === n) {
    for (let i = 0; i < n; i++) {
      const stage = d.true_stages[i];
      const x = i * dx;
      const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      r.setAttribute("x", x); r.setAttribute("y", H - 6);
      r.setAttribute("width", Math.max(1, dx));
      r.setAttribute("height", 4);
      r.setAttribute("fill", colors[stage]);
      r.setAttribute("opacity", "0.75");
      svg.appendChild(r);
    }
  }

  el.appendChild(svg);

  // Time axis ticks
  const axis = document.getElementById("time-axis");
  axis.innerHTML = "";
  const times = d.epoch_times_sec;
  const t0 = times[0] || 0;
  const tN = times[times.length - 1] || 0;
  const ticks = 6;
  for (let i = 0; i <= ticks; i++) {
    const f = i / ticks;
    const t = t0 + (tN - t0) * f;
    const span = document.createElement("span");
    span.textContent = formatTime(t);
    axis.appendChild(span);
  }
}

function makeOverlayPath(times, values, W, H, topNorm, botNorm, ignoreNaN) {
  if (!times || times.length === 0) return null;
  const t0 = times[0];
  const tN = times[times.length - 1];
  const span = tN - t0 || 1;
  const finite = values.filter(v => v != null && Number.isFinite(v));
  if (finite.length === 0) return null;
  const vmin = Math.min(...finite);
  const vmax = Math.max(...finite);
  const vspan = vmax - vmin || 1;
  const yTop = H * topNorm;
  const yBot = H * botNorm;
  let d = "";
  let started = false;
  for (let i = 0; i < times.length; i++) {
    const v = values[i];
    if (v == null || !Number.isFinite(v)) {
      if (ignoreNaN) continue;
    }
    const x = ((times[i] - t0) / span) * W;
    const norm = ((v ?? vmin) - vmin) / vspan;
    const y = yTop + (1 - norm) * (yBot - yTop);
    d += (started ? " L " : "M ") + x.toFixed(2) + " " + y.toFixed(2);
    started = true;
  }
  return d;
}

function formatTime(sec) {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  return `${h}h${m.toString().padStart(2, "0")}`;
}

// ------------------ Flip overlay (component breakdown) ------------------

function openFlip(scoreKey) {
  const d = state.analysis;
  if (!d) return;
  const def = SCORE_BY_KEY[scoreKey];
  const value = d.state[def.score_key];
  const components = (d.state.components && d.state.components[scoreKey]) || {};

  const overlay = document.getElementById("flip-overlay");
  const card = document.getElementById("flip-card");
  card.dataset.score = scoreKey;

  const front = document.getElementById("flip-front");
  front.innerHTML = renderFlipContent(def, value, components);
  document.getElementById("flip-back").innerHTML = "";

  overlay.hidden = false;
  document.getElementById("scrim").hidden = false;
}

function closeFlip() {
  document.getElementById("flip-overlay").hidden = true;
  if (document.getElementById("drawer").hidden) {
    document.getElementById("scrim").hidden = true;
  }
}

function renderFlipContent(def, value, components) {
  const COMPONENT_REFS = {
    duration:                "tap into the 7-9 h NSF adult range",
    efficiency:              "≥ 0.85 = AASM/Buysse PSQI ideal",
    rem_balance:             "REM should be 20–25% of TST",
    nrem_balance:            "NREM should be 55–65% of TST",
    waso:                    "wake-after-onset; full points at 0 min, none at ≥ 90 min",
    onset_latency:           "fall asleep in 10–20 min — very short = sleep deprivation sign",
    cycle_count:             "4–6 NREM→REM cycles per healthy night",

    single_night_deficit_min:"minutes short of an 8 h target tonight",

    sleep_contribution:      "from sleep_quality — sleep is the largest driver of next-day recovery",
    hr_dip:                  "(pre-sleep HR − min sleep HR) / pre-sleep HR — parasympathetic dominance",
    hrv_proxy:               "RMS of successive HR differences during sleep (proxy for true RMSSD)",
    hr_stability:            "absence of upward HR drift through the night",
    autonomic_estimate:      "fallback when no HR signal — uses resting_hr_delta argument",

    autonomic_arousal:       "shallow HR dip = sustained sympathetic tone",
    low_hrv:                 "low HRV correlates with stress (Kim et al. 2018)",
    overnight_hr_climb:      "rising HR through the night = sympathetic activation",
    hr_elevation_estimate:   "fallback when no HR signal",
    activity_load:           "yesterday's step count relative to a 14k baseline",
    sleep_inefficiency:      "wakefulness in bed elevates next-day stress markers",

    low_recovery:            "inverse of the recovery score",
    sleep_deficit:           "minutes short of an 8 h target, capped at 180",
    fragmentation:           "awakenings during the night",
    overnight_strain:        "from HR drift — neutral midpoint when no HR signal",

    recovery_contribution:     "50 % of recovery feeds energy directly",
    fatigue_inverse:           "30 % of (100 − fatigue)",
    sleep_quality_contribution:"20 % of sleep_quality",
  };

  const rows = Object.entries(components)
    .filter(([k]) => !k.startsWith("_"))
    .map(([k, v]) => {
      const isNumeric = typeof v === "number";
      const label = k.replace(/_/g, " ");
      const ref = COMPONENT_REFS[k] || "";
      const valStr = isNumeric ? v.toFixed(1) : v;
      const widthPct = isNumeric ? Math.max(0, Math.min(100, v / 40 * 100)) : 0;
      return `
        <div class="component-row">
          <div class="comp-meta">
            <div class="comp-name">${label} <code>${k}</code></div>
            <div class="comp-bar"><div class="comp-bar-fill" style="width: ${widthPct}%"></div></div>
            <div class="comp-ref" style="color: var(--text-faint); font-size: 11px; margin-top: 2px;">${ref}</div>
          </div>
          <div class="comp-value">${valStr}${isNumeric ? '<small>pts</small>' : ''}</div>
        </div>
      `;
    }).join("");

  const note = components._note
    ? `<div class="comp-note">${components._note}</div>`
    : "";

  return `
    <div class="ovl-title">${def.title}</div>
    <div class="ovl-score">${value.toFixed(0)}<span style="font-size: 0.3em; color: var(--text-faint); -webkit-text-fill-color: var(--text-faint);">/100</span></div>
    <div class="ovl-tag">${def.blurb}</div>
    <div class="components-list">${rows}</div>
    ${note}
  `;
}

// ------------------ What-if ------------------

let whatIfTimer = null;

function onWhatIfChange() {
  const lost = parseFloat(document.getElementById("dial-sleep").value);
  const extra = parseFloat(document.getElementById("dial-steps").value);
  document.getElementById("dial-sleep-val").textContent = lost;
  document.getElementById("dial-steps-val").textContent = extra;
  // debounce
  if (whatIfTimer) clearTimeout(whatIfTimer);
  whatIfTimer = setTimeout(async () => {
    if (!state.currentSubject) return;
    try {
      const res = await fetchJSON("/api/whatif", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          subject_id: state.currentSubject,
          lost_minutes: lost,
          extra_steps: extra,
        }),
      });
      renderDeltaTiles(res.baseline, res.scenario);
    } catch (err) {
      console.error(err);
    }
  }, 120);
}

function renderDeltaTiles(baseline, scenario) {
  const grid = document.getElementById("delta-grid");
  grid.innerHTML = "";
  for (const def of SCORE_DEFINITIONS) {
    const b = baseline[def.score_key];
    const s = scenario[def.score_key];
    const dlt = s - b;
    const isWorseDirection = def.key === "stress" || def.key === "fatigue" || def.key === "sleep_debt";
    const sign = dlt > 0.05 ? "+" : "";
    const cls = Math.abs(dlt) < 0.1 ? "flat" : (dlt > 0) === !isWorseDirection ? "up" : "down";
    const tile = document.createElement("div");
    tile.className = "delta-tile";
    tile.dataset.score = def.key;
    tile.innerHTML = `
      <div class="delta-k">${def.title}</div>
      <div class="delta-v">${s.toFixed(0)}<small style="font-family: var(--font-mono); font-size: 10px; color: var(--text-faint); margin-left: 2px;">/100</small></div>
      <div class="delta-d ${cls}">${sign}${dlt.toFixed(1)} pts</div>
    `;
    grid.appendChild(tile);
  }
}

// ------------------ Methodology drawer ------------------

let methodologyCache = null;

async function loadMethodology() {
  try {
    // We render a self-contained summary; the full markdown lives in artifacts/reports/
    methodologyCache = METHODOLOGY_HTML;
    document.getElementById("drawer-body").innerHTML = methodologyCache;
  } catch (err) {
    console.error(err);
  }
}

function openDrawer() {
  document.getElementById("drawer").hidden = false;
  document.getElementById("scrim").hidden = false;
}
function closeDrawer() {
  document.getElementById("drawer").hidden = true;
  if (document.getElementById("flip-overlay").hidden) {
    document.getElementById("scrim").hidden = true;
  }
}

const METHODOLOGY_HTML = `
  <p>Every visible number on this page is a weighted sum of named, individually-bounded components. The breakdown is shown when you tap any score card — this drawer is the full reference.</p>

  <h4>Data sources</h4>
  <table>
    <thead><tr><th>Signal</th><th>Used for</th></tr></thead>
    <tbody>
      <tr><td>Predicted hypnogram (Wake/NREM/REM)</td><td>every score</td></tr>
      <tr><td>Wrist heart rate</td><td>recovery, stress, fatigue</td></tr>
      <tr><td>Wrist accelerometer (ENMO)</td><td>movement context</td></tr>
      <tr><td>Step counts</td><td>fatigue, stress (activity load)</td></tr>
    </tbody>
  </table>

  <h3>sleep_quality</h3>
  <p>Seven components totaling 100: duration (7–9 h), efficiency (≥0.85), REM balance (20–25%), NREM balance (55–65%), WASO penalty (full at 0 min, 0 at ≥90), onset latency (10–20 min), cycle count (4–6 cycles).</p>
  <p>References: Hirshkowitz 2015 (NSF), Buysse 1989 (PSQI), Carskadon &amp; Dement 2005, Reed &amp; Sacco 2016.</p>

  <h3>sleep_debt</h3>
  <p>Single-night deficit vs an 8 h target. The Walch dataset has one night per subject so multi-night debt cannot be computed — this is tonight's deficit only.</p>

  <h3>recovery</h3>
  <p>With HR: 40 % sleep + 25 % HR dip + 20 % HRV proxy + 15 % HR stability. Without HR: 40 % sleep + 60 % autonomic estimate from <code>resting_hr_delta</code>.</p>
  <p>References: Trinder 2001, Plews 2013, Walker 2017, Whoop methodology.</p>

  <h3>stress_index</h3>
  <p>30 % autonomic arousal (inverse HR dip) + 25 % low HRV + 15 % overnight HR climb + 15 % activity load + 15 % sleep inefficiency.</p>
  <p>Reference: Kim 2018, <em>Stress and HRV: a meta-analysis</em>.</p>

  <h3>fatigue</h3>
  <p>40 % inverse-recovery + 20 % activity load + 20 % sleep deficit + 10 % fragmentation + 10 % overnight strain.</p>

  <h3>energy</h3>
  <p>Composite: 50 % recovery + 30 % (100 − fatigue) + 20 % sleep_quality.</p>

  <h4>What this dataset cannot support</h4>
  <p>SpO2 (apnea), core body temperature (circadian), respiratory rate, multi-night history, subjective ratings (PSQI/KSS/RPE), demographics, true beat-to-beat HRV. These limits make the scores <em>physiological correlate estimates</em>, not validated medical predictions.</p>

  <h4>Full reference</h4>
  <p>The complete methodology with literature citations lives at <code>artifacts/reports/digital_twin_methodology.md</code>.</p>
`;
