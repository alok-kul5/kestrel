const $ = (id) => document.getElementById(id);
const EPOCH = Date.UTC(2017, 11, 1);
const NS = "http://www.w3.org/2000/svg";
const fmt$ = (v) => "$" + Math.round(v).toLocaleString("en-US");
const fmtK = (v) => (Math.abs(v) >= 1e6 ? "$" + (v / 1e6).toFixed(2) + "M" : "$" + Math.round(v / 1e3) + "k");
const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";

let lastSeq = 0, running = true, summary = null, table = null, liveThr = null;

function el(tag, attrs = {}, parent) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
function countUp(node, to, render, ms = 1400) {
  const t0 = performance.now();
  const step = (t) => {
    const k = Math.min((t - t0) / ms, 1), e = 1 - Math.pow(1 - k, 3);
    node.textContent = render(to * e);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
const api = (p, opt) => fetch(p, opt).then((r) => r.json());

async function boot() {
  summary = await api("/api/summary");
  const curves = await api("/api/curves");
  table = curves.threshold_table;
  liveThr = summary.threshold;
  const champ = summary.models[summary.champion].test;
  const d = summary.decision;
  countUp($("kPr"), champ.pr_auc, (v) => v.toFixed(3));
  countUp($("kRec"), champ.recall_at_1pct_fpr, (v) => pct(v));
  countUp($("kSave"), 1 - d.global_threshold.cost / d.approve_all_cost, (v) => pct(v, 0));
  drawPR(curves);
  drawShap(summary.shap_importance);
  drawBench(summary);
  const s = $("thr");
  s.value = Math.max(0.01, Math.min(0.99, d.threshold)).toFixed(2);
  s.oninput = () => drawCost();
  drawCost(true);
  monitoring();
  setInterval(monitoring, 8000);
  poll();
  setInterval(poll, 700);
  flagged();
  setInterval(flagged, 2000);
}

/* live stream */
async function poll() {
  let f;
  try { f = await api(`/api/feed?since=${lastSeq}&limit=40`); } catch { $("liveText").textContent = "offline"; return; }
  running = f.replay.running;
  $("liveText").textContent = running ? `live · ${f.replay.rate}/s · ${f.replay.pos.toLocaleString()} of ${f.replay.total.toLocaleString()}` : "paused";
  $("playBtn").textContent = running ? "Pause" : "Play";
  $("cScored").textContent = f.counts.scored.toLocaleString();
  $("cFlagged").textContent = f.counts.flagged.toLocaleString();
  $("cHit").textContent = f.counts.flagged_fraud.toLocaleString();
  $("cThr").textContent = f.threshold.toFixed(2);
  liveThr = f.threshold;
  if (f.latency.n) $("kLat").textContent = f.latency.p95_ms.toFixed(1) + "ms";
  const box = $("stream");
  for (const ev of f.events) {
    lastSeq = Math.max(lastSeq, ev.seq);
    box.prepend(row(ev));
  }
  while (box.children.length > 16) box.lastChild.remove();
}
function row(ev) {
  const r = document.createElement("div");
  r.className = "row " + ev.decision;
  const t = new Date(EPOCH + ev.dt * 1000);
  const cls = ev.p >= 0.5 ? "high" : ev.p >= 0.1 ? "mid" : "";
  r.innerHTML = `
    <span class="mono dim">${t.toISOString().slice(5, 10)} ${t.toISOString().slice(11, 16)}</span>
    <span class="acct mono">${ev.account}</span>
    <span class="amt">${"$" + ev.amount.toFixed(2)}</span>
    <span class="risk"><span class="bar"><i class="${cls}" style="width:${Math.max(2, ev.p * 100)}%"></i></span><span class="mono">${ev.p.toFixed(3)}</span></span>
    <span><span class="chip ${ev.decision}">${ev.decision}</span>${ev.label === 1 ? '<span class="fraud-dot" title="confirmed fraud"></span>' : ""}</span>`;
  return r;
}

/* review queue */
let shownCases = new Set();
async function flagged() {
  const { cases } = await api("/api/flagged?limit=25");
  const box = $("cases");
  if (!cases.length) return;
  if (!shownCases.size) box.innerHTML = "";
  for (const c of cases.slice().reverse()) {
    if (shownCases.has(c.seq)) continue;
    shownCases.add(c.seq);
    const maxImp = Math.max(...c.reasons.map((r) => r.impact), 1e-6);
    const n = document.createElement("div");
    n.className = "case";
    const t = new Date(EPOCH + c.dt * 1000).toISOString().replace("T", " ").slice(0, 16);
    n.innerHTML = `
      <div class="case-top"><span class="case-amt">${"$" + c.amount.toFixed(2)}</span>
        <span>${c.label === 1 ? '<span class="tag fraud">confirmed fraud</span>' : c.label === 0 ? '<span class="tag legit">legit</span>' : ""}
        <span class="case-p">p = ${c.p.toFixed(3)}</span></span></div>
      <div class="case-meta">${t} · txn ${c.id} · ${c.product || ""} · ${c.card || "card"} · ${c.email || "no email"}${c.device ? " · " + c.device : ""}</div>
      ${c.reasons.map((r) => `<div class="reason"><span>${r.text}</span><span class="imp" style="width:${(r.impact / maxImp) * 70}px"></span></div>`).join("")}`;
    box.prepend(n);
  }
  while (box.children.length > 30) box.lastChild.remove();
}

/* cost vs threshold */
function drawCost(animate) {
  const svg = $("costChart");
  svg.innerHTML = "";
  const W = 560, H = 260, L = 56, R = 14, T = 14, B = 34;
  const xs = table.threshold, ys = table.cost;
  const yMax = Math.max(...ys, table.approve_all) * 1.05;
  const x = (v) => L + ((v - 0) / 1) * (W - L - R);
  const y = (v) => T + (1 - v / yMax) * (H - T - B);
  for (let i = 0; i <= 4; i++) {
    const v = (yMax / 4) * i;
    el("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: "axis" }, svg);
    el("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end" }, svg).textContent = fmtK(v);
  }
  for (const v of [0, 0.25, 0.5, 0.75, 1]) el("text", { x: x(v), y: H - 12, "text-anchor": "middle" }, svg).textContent = v.toFixed(2);
  el("line", { x1: L, x2: W - R, y1: y(table.approve_all), y2: y(table.approve_all), stroke: "#a99486", "stroke-dasharray": "4 5" }, svg);
  el("text", { x: W - R, y: y(table.approve_all) - 6, "text-anchor": "end" }, svg).textContent = "approve everything";
  const d = xs.map((v, i) => `${i ? "L" : "M"}${x(v).toFixed(1)},${y(ys[i]).toFixed(1)}`).join("");
  const area = d + `L${x(xs[xs.length - 1])},${y(0)}L${x(xs[0])},${y(0)}Z`;
  const g = el("linearGradient", { id: "cg", x1: 0, x2: 0, y1: 0, y2: 1 }, el("defs", {}, svg));
  el("stop", { offset: 0, "stop-color": "#c4461b", "stop-opacity": 0.45 }, g);
  el("stop", { offset: 1, "stop-color": "#c4461b", "stop-opacity": 0 }, g);
  el("path", { d: area, fill: "url(#cg)" }, svg);
  el("path", { d, fill: "none", stroke: "#e2672f", "stroke-width": 2.5, class: animate ? "draw" : "" }, svg);
  const best = summary.decision.threshold;
  el("line", { x1: x(best), x2: x(best), y1: T, y2: H - B, stroke: "#7fd1a8", "stroke-dasharray": "3 4" }, svg);
  el("text", { x: x(best) + 6, y: T + 12, fill: "#7fd1a8" }, svg).textContent = "cost-optimal (chosen on validation)";
  const thr = +$("thr").value;
  const i = xs.findIndex((v) => v >= thr - 1e-9);
  el("line", { x1: x(thr), x2: x(thr), y1: T, y2: H - B, stroke: "#ffb547", "stroke-width": 2 }, svg);
  el("circle", { cx: x(thr), cy: y(ys[i]), r: 6, fill: "#ffb547", stroke: "#110d0b", "stroke-width": 2 }, svg);
  const fl = table.flagged[i], tp = table.tp[i];
  const stats = [
    [fmt$(ys[i]), `total cost @ ${thr.toFixed(2)}`],
    [pct(1 - ys[i] / table.approve_all, 0), "loss avoided vs approve-all"],
    [pct(table.fraud_dollars[i] / table.fraud_total, 0), "of fraud dollars stopped"],
    [pct(tp / Math.max(fl, 1)), "precision of review queue"],
    [pct(tp / table.n_fraud), "fraud transactions caught"],
    [pct(fl / table.n, 2), "of traffic sent to review"],
  ];
  $("thrStats").innerHTML = stats.map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
}
$("applyThr").onclick = async () => {
  await api("/api/threshold", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ threshold: +$("thr").value }) });
  $("applyThr").textContent = "Applied ✓";
  setTimeout(() => ($("applyThr").textContent = "Apply to live scoring"), 1500);
};
$("playBtn").onclick = async () => {
  await api(running ? "/api/replay/stop" : "/api/replay/start", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ rate: +$("speed").value }) });
};
$("speed").onchange = async () => {
  await api("/api/replay/stop", { method: "POST" });
  setTimeout(() => api("/api/replay/start", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ rate: +$("speed").value }) }), 300);
};

/* PR curves */
const LABELS = { logreg_balanced: "Logistic regression", xgboost: "XGBoost", ablation_raw_columns_only: "LightGBM, raw columns only" };
const COLORS = ["#e2672f", "#ffb547", "#7fd1a8", "#8a7a70"];
function drawPR(curves) {
  const svg = $("prChart");
  const W = 560, H = 300, L = 44, R = 14, T = 12, B = 34;
  const x = (v) => L + v * (W - L - R), y = (v) => T + (1 - v) * (H - T - B);
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    el("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: "axis" }, svg);
    el("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end" }, svg).textContent = v.toFixed(2);
    el("text", { x: x(v), y: H - 12, "text-anchor": "middle" }, svg).textContent = v.toFixed(2);
  }
  el("text", { x: W - R, y: H - 26, "text-anchor": "end" }, svg).textContent = "recall →";
  el("text", { x: L + 6, y: T + 12 }, svg).textContent = "↑ precision";
  const names = Object.keys(curves.pr_by_model);
  const champ = summary.champion;
  const order = [champ, "xgboost", "logreg_balanced", "ablation_raw_columns_only"].filter((n) => names.includes(n));
  const legend = [];
  order.slice().reverse().forEach((n) => {
    const c = curves.pr_by_model[n], idx = order.indexOf(n);
    const d = c.recall.map((r, i) => `${i ? "L" : "M"}${x(r).toFixed(1)},${y(c.precision[i]).toFixed(1)}`).join("");
    el("path", { d, fill: "none", stroke: COLORS[idx], "stroke-width": n === champ ? 3 : 1.6, class: "draw", opacity: n === champ ? 1 : 0.85 }, svg);
  });
  order.forEach((n, idx) => {
    const name = n === champ ? `LightGBM champion (${summary.models[n].note})` : LABELS[n] || n;
    legend.push(`<span><i style="background:${COLORS[idx]}"></i>${name} · ${summary.models[n].test.pr_auc.toFixed(3)}</span>`);
  });
  $("prLegend").innerHTML = legend.join("");
}

/* monitoring */
async function monitoring() {
  const m = await api("/api/monitoring");
  if (m.live.psi != null) $("cPsi").textContent = `${m.live.psi.toFixed(3)} ${m.live.status}`;
  else $("cPsi").textContent = m.live.status;
  const svg = $("monChart");
  if (svg.childNodes.length) return;
  const w = m.weekly, W = 420, H = 240, L = 40, R = 40, T = 14, B = 34;
  const bw = (W - L - R) / w.length;
  const frMax = Math.max(...w.map((r) => r.fraud_rate)) * 1.25, psiMax = Math.max(0.3, ...w.map((r) => r.psi)) * 1.1;
  const y1 = (v) => T + (1 - v / frMax) * (H - T - B), y2 = (v) => T + (1 - v / psiMax) * (H - T - B);
  el("rect", { x: L, y: y2(0.1) - 1, width: W - L - R, height: 2, fill: "rgba(255,181,71,.4)" }, svg);
  w.forEach((r, i) => {
    const bx = L + i * bw + bw * 0.18;
    const b = el("rect", { x: bx, y: y1(r.fraud_rate), width: bw * 0.64, height: H - B - y1(r.fraud_rate), rx: 3, fill: r.start_day >= summary.split.test.days[0] ? "#c4461b" : "#6b2a14" }, svg);
    b.style.transformOrigin = `0 ${H - B}px`;
    b.style.animation = `grow-y .9s ${i * 60}ms cubic-bezier(.2,.7,.2,1) both`;
    el("text", { x: bx + bw * 0.32, y: H - 14, "text-anchor": "middle" }, svg).textContent = "d" + r.start_day;
  });
  const d = w.map((r, i) => `${i ? "L" : "M"}${L + (i + 0.5) * bw},${y2(r.psi)}`).join("");
  el("path", { d, fill: "none", stroke: "#ffb547", "stroke-width": 2, class: "draw" }, svg);
  w.forEach((r, i) => el("circle", { cx: L + (i + 0.5) * bw, cy: y2(r.psi), r: 4, fill: "#ffb547" }, svg));
  for (const v of [0, frMax / 2]) el("text", { x: L - 6, y: y1(v) + 4, "text-anchor": "end" }, svg).textContent = pct(v, 1);
  for (const v of [0, 0.1, psiMax / 1.1]) el("text", { x: W - R + 6, y: y2(v) + 4 }, svg).textContent = v.toFixed(2);
  const st = document.createElement("style");
  st.textContent = "@keyframes grow-y{from{transform:scaleY(0)}}";
  document.head.appendChild(st);
}

function drawShap(rows) {
  const top = rows.slice(0, 12), max = top[0].mean_abs_shap;
  $("shap").innerHTML = top.map((r, i) =>
    `<div class="shap-row"><span title="${r.feature}">${r.feature}</span><i style="width:${(r.mean_abs_shap / max) * 100}%;animation-delay:${400 + i * 60}ms"></i></div>`).join("");
}
function drawBench(s) {
  const names = ["logreg_balanced", "lightgbm_none", "lightgbm_class_weight", "lightgbm_undersample", "xgboost", "isolation_forest", "ablation_raw_columns_only", "ablation_no_graph"];
  const nice = { logreg_balanced: "LogReg (balanced)", lightgbm_none: "LightGBM", lightgbm_class_weight: "LightGBM + class weight", lightgbm_undersample: "LightGBM + undersample", xgboost: "XGBoost", isolation_forest: "Isolation forest", ablation_raw_columns_only: "No engineered feats", ablation_no_graph: "No graph feats" };
  $("bench").innerHTML = `<tr><th>Model</th><th>PR-AUC</th><th>Rec@1%</th><th>ROC</th></tr>` + names.filter((n) => s.models[n]).map((n) => {
    const t = s.models[n].test;
    return `<tr class="${n === s.champion ? "champ" : ""}"><td>${nice[n]}</td><td>${t.pr_auc.toFixed(3)}</td><td>${pct(t.recall_at_1pct_fpr)}</td><td>${t.roc_auc.toFixed(3)}</td></tr>`;
  }).join("");
}

boot();
