/* MedIntel clinician dashboard: a client of the REST API (/v1). Plain JavaScript, no build step.
   Every value from the server is inserted as text, never as HTML. The sign-in token lives in sessionStorage. */
"use strict";

const API = "/v1";
const state = { token: null, me: null, kb: null, poll: null };
const SAMPLES = ["chest_pain_clinic.csv", "morning_batch.csv", "boundary_tests.csv"];
const PRIORITY_CLASS = { escalate: "critical", review: "warning", priority: "serious", manual: "neutral", routine: "good", failed: "critical" };
const DECISION = { high_risk: "High risk", low_risk: "Low risk", uncertain: "Uncertain", withheld: "Withheld", failed: "Failed" };
const STATUS = {
  pending_review: ["Pending review", "warning"],
  accepted: ["Accepted", "good"],
  annotated: ["Annotated", "neutral"],
  rejected: ["Rejected", "critical"],
  failed: ["Failed", "critical"],
};
const WHAT_IF_FIELDS = [
  ["systolic_bp", "Systolic BP (mmHg)"],
  ["diastolic_bp", "Diastolic BP (mmHg)"],
  ["heart_rate", "Heart rate (bpm)"],
  ["total_cholesterol", "Cholesterol (mg/dL)"],
  ["fasting_glucose", "Fasting glucose (mg/dL)"],
  ["bmi", "BMI"],
];

/* ---------- DOM helpers ---------- */
function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = String(value);
    else if (key === "style") for (const [p, v] of Object.entries(value)) el.style.setProperty(p, v);
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else if (key === "href") el.setAttribute("href", safeHref(value));
    else if (value === true) el.setAttribute(key, "");
    else el.setAttribute(key, String(value));
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}
function safeHref(url) {
  const s = String(url);
  return s.startsWith("#") || s.startsWith("/") || s.startsWith("https://") ? s : "#";
}
function svg(tag, attrs, ...children) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, String(v));
  for (const c of children) el.append(c);
  return el;
}
const fmt = (n, d = 2) => (n === null || n === undefined || Number.isNaN(Number(n)) ? "n/a" : Number(n).toFixed(d));
const words = (s) => String(s || "").replaceAll("_", " ");
const when = (iso) => (iso ? new Date(iso).toLocaleString() : "");
const hasRole = (...roles) => !!state.me && roles.some((r) => state.me.identity.roles.includes(r));

function logo() {
  return svg("svg", { viewBox: "0 0 40 40", "aria-hidden": "true" },
    svg("circle", { cx: 20, cy: 20, r: 19, fill: "currentColor", opacity: "0.12" }),
    svg("path", { d: "M5 21h7l3-8 5 15 4-11 2 4h9", fill: "none", stroke: "currentColor", "stroke-width": 2.6, "stroke-linecap": "round", "stroke-linejoin": "round" }));
}
function badgeFor(priority) {
  return h("span", { class: `badge ${PRIORITY_CLASS[priority.key] || "neutral"}` }, h("span", { class: "dot" }), priority.label);
}
function statusChip(status) {
  const [text, cls] = STATUS[status] || [words(status), "neutral"];
  return h("span", { class: `badge ${cls}` }, text);
}
function confidence(c, threshold) {
  const below = c < threshold;
  return h("div", { class: "conf", title: `C = ${fmt(c)}; threshold ${fmt(threshold)}` },
    h("div", { class: "meter", role: "img", "aria-label": `Confidence ${fmt(c)} against threshold ${fmt(threshold)}` },
      h("span", { style: { width: `${Math.min(100, Math.max(0, c * 100))}%`, "--mc": below ? "var(--st-warning)" : "var(--accent)" } }),
      h("i", { class: "tick", style: { left: `calc(${threshold * 100}% - 1px)` } })),
    h("span", { class: "v", text: fmt(c) }));
}
function card(title, sub, body, end) {
  return h("section", { class: "card" },
    h("div", { class: "card-h" }, h("div", {}, h("h2", { text: title }), sub ? h("p", { class: "sub", text: sub }) : null),
      end ? h("div", { class: "end" }, end) : null),
    h("div", { class: "card-b" }, body));
}
function note(text, kind = "") {
  return h("div", { class: `note ${kind}`, role: kind === "err" ? "alert" : "status" }, text);
}
function table(head, rows, onClick) {
  return h("div", { class: "scroll-x" }, h("table", { class: "tbl" },
    h("thead", {}, h("tr", {}, head.map((t) => h("th", { scope: "col", text: t })))),
    h("tbody", {}, rows.map((r) => {
      const tr = h("tr", onClick ? { class: "click", tabindex: 0 } : {}, r.cells.map((c) => (c instanceof Node && c.tagName === "TD" ? c : h("td", {}, c))));
      if (onClick) {
        tr.addEventListener("click", () => onClick(r));
        tr.addEventListener("keydown", (e) => { if (e.key === "Enter") onClick(r); });
      }
      return tr;
    }))));
}

/* ---------- API ---------- */
class ApiError extends Error {
  constructor(status, detail) { super(detail); this.status = status; }
}
async function api(path, { method = "GET", body, form, text } = {}) {
  const headers = {};
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  let payload;
  if (form) payload = form;
  else if (body !== undefined) { headers["Content-Type"] = "application/json"; payload = JSON.stringify(body); }
  const res = await fetch(API + path, { method, headers, body: payload });
  if (res.status === 401) {
    if (state.token) signOut(true);
    throw new ApiError(401, "Please sign in again.");
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    } catch (_) { /* keep the status line */ }
    throw new ApiError(res.status, detail);
  }
  return text ? res.text() : res.json();
}
async function knowledgeBase() {
  if (!state.kb) state.kb = await api("/knowledge/base");
  return state.kb;
}
function remember(token) {
  state.token = token;
  try { token ? sessionStorage.setItem("medintel.token", token) : sessionStorage.removeItem("medintel.token"); } catch (_) { /* private mode */ }
}
function signOut(expired) {
  remember(null);
  state.me = null;
  state.kb = null;
  if (location.hash !== "#/login") location.hash = expired ? "#/login?expired=1" : "#/login";
}

/* ---------- shell ---------- */
const NAV = [["#/batches", "Batches"], ["#/knowledge", "Knowledge"], ["#/audit", "Audit trail"], ["#/model", "Model card"], ["#/evaluation", "Evaluation"]];
function shell(active, title, subtitle, content) {
  const me = state.me;
  const initials = me.identity.name.split(/\s+/).map((w) => w[0]).join("").slice(0, 2).toUpperCase();
  const roles = me.identity.roles.map((r) => me.role_labels[r] || r).join(", ");
  const app = h("div", { class: "app" },
    h("nav", { class: "rail", "aria-label": "Main" },
      h("a", { class: "brand", href: "#/batches" }, logo(), h("div", {}, h("b", { text: "MedIntel" }), h("span", { text: "Decision support" }))),
      h("div", { class: "nav" }, NAV.map(([href, label]) => h("a", { href, "aria-current": active === href ? "page" : null, text: label })),
        h("a", { href: "/docs", target: "_blank", rel: "noopener", text: "API docs" })),
      h("div", { class: "rail-foot" },
        h("div", { class: "who" }, h("span", { class: "av", text: initials }), h("div", {}, h("strong", { text: me.identity.name }), h("small", { text: roles }))),
        h("div", { class: "row" },
          h("button", { class: "btn sm", type: "button", onclick: cycleTheme, text: `Theme: ${themeName()}` }),
          me.server.auth ? h("button", { class: "btn sm", type: "button", onclick: () => signOut(false), text: "Sign out" }) : null),
        h("p", { class: "rail-note", text: me.server.auth ? "Signed in. Your name is recorded on every review and approval." : "Local mode: sign-in is off on this server." }))),
    h("div", { class: "main" },
      h("header", { class: "topbar" }, h("div", { class: "topbar-in" },
        h("div", { class: "title-wrap" }, h("h1", { text: title }), subtitle ? h("p", { text: subtitle }) : null),
        h("span", { class: "pill", text: "Research prototype" }))),
      h("main", { class: "page", id: "page", tabindex: -1 }, content),
      h("footer", { class: "foot" }, h("span", { text: `${me.server.notice} Model ${me.server.model} · index ${me.server.index} · summaries: ${me.server.summary_writer}` }))));
  return app;
}
function themeName() {
  return document.documentElement.dataset.theme || "auto";
}
function cycleTheme() {
  const next = { auto: "light", light: "dark", dark: "auto" }[themeName()];
  if (next === "auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = next;
  try { localStorage.setItem("medintel.theme", next); } catch (_) { /* ignore */ }
  render();
}

/* ---------- pages ---------- */
function pageSignIn(expired) {
  const input = h("input", { id: "token", type: "password", autocomplete: "off", spellcheck: "false", placeholder: "mi_...", required: true });
  const msg = h("div", { "aria-live": "polite" }, expired ? note("Your session ended. Sign in again.", "warn") : null);
  const form = h("form", { class: "stack", onsubmit: async (e) => {
    e.preventDefault();
    remember(input.value.trim());
    try {
      state.me = await api("/me");
      location.hash = "#/batches";
    } catch (err) {
      remember(null);
      msg.replaceChildren(note(err.status === 401 ? "That token was not accepted." : err.message, "err"));
    }
  } },
  h("label", { class: "field", for: "token" }, "Access token", input),
  h("button", { class: "btn primary", type: "submit", text: "Sign in" }),
  msg);
  return h("div", { class: "signin" }, h("section", { class: "card" }, h("div", { class: "card-b" },
    h("div", { class: "brand" }, logo(), h("div", {}, h("b", { text: "MedIntel" }), h("span", { text: "Clinician dashboard" }))),
    h("p", { class: "soft", text: "Sign in with the token your administrator created for you (uv run medintel users add). Your roles decide what you can do." }),
    form,
    note("Research prototype on synthetic or de-identified data. Not a medical device; a clinician makes the final decision."))));
}

async function pageBatches() {
  const out = h("div", { class: "stack" });
  const canUpload = hasRole("data_admin");
  const canAnalyse = hasRole("clinician", "data_admin");
  const result = h("div", { class: "stack", "aria-live": "polite" });
  let mode = "sample";
  const sample = h("select", { id: "sample" }, SAMPLES.map((s) => h("option", { value: s, text: s })));
  const file = h("input", { id: "csv", type: "file", accept: ".csv,text/csv" });
  const events = h("input", { id: "events", type: "file", accept: ".csv,text/csv" });
  const sampleBox = h("label", { class: "field", for: "sample" }, "Bundled synthetic batch", sample);
  const uploadBox = h("div", { class: "form-grid" },
    h("label", { class: "field", for: "csv" }, "Patient records (CSV)", file),
    h("label", { class: "field", for: "events" }, "Event log (CSV, optional)", events));
  uploadBox.hidden = true;
  const segSample = h("button", { type: "button", "aria-pressed": "true", text: "Sample" });
  const segUpload = h("button", { type: "button", "aria-pressed": "false", text: "Upload CSV" });
  const setMode = (m) => {
    mode = m;
    segSample.setAttribute("aria-pressed", String(m === "sample"));
    segUpload.setAttribute("aria-pressed", String(m === "upload"));
    sampleBox.hidden = m !== "sample";
    uploadBox.hidden = m !== "upload";
  };
  segSample.addEventListener("click", () => setMode("sample"));
  segUpload.addEventListener("click", () => setMode("upload"));
  const recentBox = h("div");
  const loadRecent = async () => {
    const recent = await api("/batches?limit=25");
    const rows = recent.map((b) => ({ b, cells: [h("span", { class: "mono", text: b.batch_id }), b.source_name, when(b.created_at), b.valid, b.rejected, `${b.analyzed} / ${b.valid}`] }));
    recentBox.replaceChildren(rows.length ? table(["Batch", "Source", "Created", "Valid", "Rejected", "Analysed"], rows, (r) => { location.hash = `#/batch/${r.b.batch_id}`; })
      : h("p", { class: "empty", text: "No batches yet. Validate a sample above to start." }));
  };
  const validate = h("button", { class: "btn primary", type: "submit", disabled: !canUpload, text: "Validate records" });
  const form = h("form", { class: "stack", onsubmit: async (e) => {
    e.preventDefault();
    validate.disabled = true;
    result.replaceChildren(note("Validating..."));
    try {
      let summary;
      if (mode === "sample") summary = await api("/batches", { method: "POST", body: { sample_name: sample.value } });
      else {
        if (!file.files[0]) throw new Error("Choose a CSV file first.");
        const data = new FormData();
        data.append("file", file.files[0]);
        if (events.files[0]) data.append("events", events.files[0]);
        summary = await api("/batches/upload", { method: "POST", form: data });
      }
      result.replaceChildren(validationResult(summary, canAnalyse));
      loadRecent().catch(() => {});
    } catch (err) {
      result.replaceChildren(note(err.message, "err"));
    } finally {
      validate.disabled = !canUpload;
    }
  } },
  h("div", { class: "seg", role: "group", "aria-label": "Source" }, segSample, segUpload),
  sampleBox, uploadBox,
  h("div", { class: "row" }, validate,
    canUpload ? h("span", { class: "muted", text: "Every record is checked before any agent runs (TC-01, TC-02)." })
      : h("span", { class: "muted", text: "Uploading needs the Data Administrator role." })));
  out.append(card("New batch", "Validate a batch, then analyse it in the background.", [form, result]));

  await loadRecent();
  out.append(card("Recent batches", "Newest first.", recentBox));
  return shell("#/batches", "Batches", "Upload or pick a batch, validate it, and triage the results.", out);
}

function validationResult(summary, canAnalyse) {
  const box = h("div", { class: "stack" });
  box.append(note(`${summary.valid} of ${summary.records_read} records are valid in ${summary.source_name} (batch ${summary.batch_id}).`, summary.rejected.length ? "warn" : "ok"));
  if (summary.missing_columns.length) box.append(note(`Missing columns: ${summary.missing_columns.join(", ")}`, "err"));
  if (summary.rejected.length) {
    box.append(table(["Row", "Patient", "Problems"], summary.rejected.map((r) => ({ cells: [r.row, r.patient_id || "(none)",
      h("td", { class: "wrap" }, r.problems.map((p) => `${p.field}: ${p.message}${p.test_case ? ` (${p.test_case})` : ""}`).join("; "))] }))));
  }
  if (summary.valid) {
    const progress = h("div", { class: "progress", role: "progressbar", "aria-valuemin": 0, "aria-valuemax": 100, "aria-valuenow": 0 }, h("span"));
    const label = h("span", { class: "muted", "aria-live": "polite" });
    const start = h("button", { class: "btn primary", type: "button", disabled: !canAnalyse, text: `Analyse ${summary.valid} patients` });
    start.addEventListener("click", async () => {
      start.disabled = true;
      try {
        const status = await api(`/batches/${summary.batch_id}/jobs`, { method: "POST" });
        watchJob(status.job.job_id, progress, label);
      } catch (err) {
        label.textContent = err.message;
        start.disabled = false;
      }
    });
    box.append(h("div", { class: "row" }, start, canAnalyse ? null : h("span", { class: "muted", text: "Analysing needs the Clinician or Data Administrator role." })), progress, label);
  }
  return box;
}

function watchJob(jobId, progress, label) {
  clearInterval(state.poll);
  const tick = async () => {
    try {
      const s = await api(`/jobs/${jobId}`);
      const pct = Math.round(s.progress * 100);
      progress.firstChild.style.setProperty("width", `${pct}%`);
      progress.setAttribute("aria-valuenow", String(pct));
      label.textContent = `${s.job.done + s.job.failed} of ${s.job.total} cases (${words(s.job.status)})`;
      if (["completed", "failed", "cancelled", "interrupted"].includes(s.job.status)) {
        clearInterval(state.poll);
        if (s.job.done) location.hash = `#/batch/${s.job.batch_id}`;
      }
    } catch (err) {
      clearInterval(state.poll);
      label.textContent = err.message;
    }
  };
  state.poll = setInterval(tick, 700);
  tick();
}

async function pageBatch(batchId) {
  const s = await api(`/batches/${encodeURIComponent(batchId)}`);
  const counts = {};
  for (const q of s.queue) counts[q.priority.key] = (counts[q.priority.key] || 0) + 1;
  const chips = h("div", { class: "row" }, Object.entries(counts).map(([k, n]) => h("span", { class: `badge ${PRIORITY_CLASS[k]}` }, h("span", { class: "dot" }), `${words(k)} ${n}`)));
  const rows = s.queue.map((q) => ({ q, cells: [badgeFor(q.priority), h("span", { class: "mono", text: q.case_id }), q.patient_id, DECISION[q.decision] || q.decision,
    confidence(q.confidence, q.threshold), fmt(q.probability_high_risk), q.triggered_rules.join(", ") || "none", q.evidence_count, q.conflict_count, statusChip(q.status)] }));
  const body = h("div", { class: "stack" },
    h("dl", { class: "kv" }, h("dt", { text: "Source" }), h("dd", { text: s.source_name }), h("dt", { text: "Valid records" }), h("dd", { text: s.valid }),
      h("dt", { text: "Analysed" }), h("dd", { text: s.analyzed }), h("dt", { text: "Pending review" }), h("dd", { text: s.pending_review })),
    chips);
  const analyse = s.not_analyzed.length && hasRole("clinician", "data_admin") ? (() => {
    const progress = h("div", { class: "progress" }, h("span"));
    const label = h("span", { class: "muted" });
    const btn = h("button", { class: "btn primary", type: "button", text: `Analyse ${s.not_analyzed.length} remaining` });
    btn.addEventListener("click", async () => { btn.disabled = true; const st = await api(`/batches/${s.batch_id}/jobs`, { method: "POST" }); watchJob(st.job.job_id, progress, label); });
    return h("div", { class: "stack" }, btn, progress, label);
  })() : null;
  return shell("#/batches", `Batch ${s.batch_id}`, "Triage order: priority first, then probability of high risk.",
    h("div", { class: "stack" }, card("Batch", null, [body, analyse]),
      card("Triage queue", `${s.queue.length} cases`, s.queue.length ? table(["Priority", "Case", "Patient", "Decision", "Confidence C", "P(high)", "Rules", "Evidence", "Conflicts", "Status"], rows, (r) => { location.hash = `#/case/${r.q.case_id}`; })
        : h("p", { class: "empty", text: "Not analysed yet." }))));
}

async function pageCase(caseId) {
  const id = encodeURIComponent(caseId);
  const [c, x, kb, trail] = await Promise.all([api(`/cases/${id}`), api(`/cases/${id}/explanation`), knowledgeBase(), api(`/audit?case_id=${id}&limit=20`)]);
  const r = c.report;
  const passages = Object.fromEntries((kb.passages || []).map((p) => [p.source_id, p]));
  const header = h("section", { class: "card" }, h("div", { class: "card-b decision" },
    h("div", { class: "stack" },
      h("div", { class: "row" }, h("span", { class: `badge lg ${PRIORITY_CLASS[c.priority.key] || "neutral"}` }, h("span", { class: "dot" }), c.priority.label), statusChip(r.status)),
      h("h2", { text: `${DECISION[r.decision] || r.decision} · ${r.patient_id}` }),
      h("p", { class: "rec", text: r.recommendation }),
      h("p", { class: "muted mono", text: `model ${r.versions.model} · rulebase ${r.versions.rulebase} · index ${r.versions.index}` })),
    h("div", { class: "stack" }, h("span", { class: "eyebrow", text: "Confidence C against threshold" }), confidence(r.confidence, x.threshold),
      h("p", { class: "muted", text: x.below_threshold ? "Below the threshold: low confidence." : "At or above the threshold." }))));

  const summary = card("Clinical context summary", `Written by ${c.summary_written_by}${c.summary_cost_usd !== null ? ` · about $${fmt(c.summary_cost_usd, 4)}` : ""}`,
    [h("p", { class: "summary", text: r.summary || "(no summary)" }),
      x.removed_statements.length ? h("div", { class: "stack" }, h("span", { class: "eyebrow", text: "Removed for lacking support (TC-12)" }),
        h("ul", { class: "list" }, x.removed_statements.map((s) => h("li", { class: "struck", text: s })))) : null,
      x.conflicts.length ? h("div", { class: "stack" }, h("span", { class: "eyebrow", text: "Conflicts" }), h("ul", { class: "list" }, x.conflicts.map((s) => h("li", { text: s })))) : null]);

  const terms = h("div", { class: "terms" }, [["M", "ML support", "alpha"], ["R", "Rule consistency", "beta"], ["K", "Retrieval support", "gamma"], ["D", "Disagreement", "lam"]].map(([sym, label, w]) =>
    h("div", { class: "term" }, h("span", { class: "sym", text: sym }), h("span", { class: "tv", text: fmt(x.component_scores[sym]) }), h("small", { text: `${label}, weight ${fmt(x.weights[w])}` }))));
  const fusion = card("How the decision was reached", "C = alpha*M + beta*R + gamma*K - lambda*D",
    [terms, h("ol", { class: "list" }, x.steps.map((s) => h("li", { text: s })))]);

  const top = x.model.largest_contributions;
  const maxAbs = Math.max(0.01, ...top.map((t) => Math.abs(t.contribution)));
  const contrib = h("div", { class: "contrib", role: "list" }, top.map((t) => {
    const pct = (Math.abs(t.contribution) / maxAbs) * 35;
    const up = t.contribution > 0;
    return [h("span", { class: "lbl", role: "listitem" }, h("b", { text: t.input }), ` ${t.value}`),
      h("div", { class: "cbar", title: `${up ? "+" : ""}${fmt(t.contribution, 3)} on the log-odds scale` },
        h("span", { class: up ? "up" : "down", style: { width: `${pct}%` } }),
        h("em", { style: up ? { left: `calc(50% + ${pct}% + 6px)` } : { right: `calc(50% + ${pct}% + 6px)` }, text: `${up ? "+" : ""}${fmt(t.contribution, 2)}` }))];
  }));
  const model = card("Risk model", `${x.model.submodel} · P(high risk) ${fmt(x.model.probability_high_risk)}`,
    [top.length ? contrib : h("p", { class: "empty", text: "No contributions to show." }),
      x.model.inputs_filled_in.length ? note(`Not recorded, filled in with training averages: ${x.model.inputs_filled_in.join(", ")}.`, "warn") : null,
      x.model.submodel.startsWith("prototype") ? note(x.model.note, "warn")
        : h("p", { class: "muted" }, "Red raises the estimate, blue lowers it. Each bar is the input's exact SHAP value on the log-odds scale; inputs that were not recorded contribute 0. ",
          h("a", { href: "#/model", text: "Read the model card" }))]);

  const evidence = card("Evidence", `${x.evidence.length} passages above the relevance threshold`,
    x.evidence.length ? h("div", { class: "evidence" }, x.evidence.map((e) => {
      const p = passages[e.source_id] || {};
      return h("article", { class: "ev" },
        h("div", { class: "head" }, h("span", { class: "mono", text: e.source_id }), h("strong", { text: e.title }),
          h("span", { class: "chip", text: `${e.direction} risk` }), h("span", { class: "chip", text: `score ${fmt(e.score)}` })),
        p.passage ? h("p", { text: p.passage }) : null,
        h("p", { class: "muted" }, `Matched on: ${e.matched_on}`, p.url ? [" · ", h("a", { href: p.url, target: "_blank", rel: "noopener noreferrer", text: p.source || "source" })] : null));
    })) : h("p", { class: "empty", text: "No sufficiently relevant evidence (TC-06): retrieval support K is 0." }));

  const rules = card("Rules", x.rules_fired.length ? `Fired: ${x.rules_fired.join(", ")}` : "No rule fired",
    h("ul", { class: "list" }, r.findings.map((f) => h("li", { text: f }))));

  return shell("#/batches", `Case ${r.case_id}`, `Patient ${r.patient_id}${c.clinical_notes ? ` · ${c.clinical_notes}` : ""}`,
    h("div", { class: "stack" }, header,
      h("div", { class: "grid g2" }, summary, fusion),
      h("div", { class: "grid g2" }, model, evidence),
      h("div", { class: "grid g2" }, rules, reviewCard(r)),
      whatIfCard(r.case_id, c.record, x),
      card("Audit trail", "Append-only; newest first.", table(["Time", "Event", "Actor", "Status"], trail.records.map((a) => ({ cells: [when(a.timestamp), words(a.event), a.actor, words(a.status)] }))))));
}

function reviewCard(report) {
  if (!hasRole("clinician")) return card("Clinician review", null, note("Reviewing needs the Clinician role."));
  const notes = h("textarea", { id: "review-notes", maxlength: 2000 });
  const out = h("div", { "aria-live": "polite" });
  const choice = (value, label, checked) => h("label", { class: "choice" }, h("input", { type: "radio", name: "decision", value, checked }), label);
  const form = h("form", { class: "stack", onsubmit: async (e) => {
    e.preventDefault();
    const decision = form.querySelector("input[name=decision]:checked").value;
    try {
      const res = await api(`/cases/${report.case_id}/reviews`, { method: "POST", body: { decision, notes: notes.value || null } });
      out.replaceChildren(note(res.message, "ok"));
      setTimeout(render, 900);
    } catch (err) {
      out.replaceChildren(note(err.message, "err"));
    }
  } },
  h("div", { class: "choices", role: "radiogroup", "aria-label": "Decision" }, choice("accept", "Accept", true), choice("annotate", "Annotate"), choice("reject", "Reject")),
  h("label", { class: "field", for: "review-notes" }, "Notes (needed to annotate or reject)", notes),
  h("div", { class: "row" }, h("button", { class: "btn primary", type: "submit", text: "Record review" }), h("span", { class: "muted", text: `Recorded as ${state.me.identity.name}.` })),
  out);
  return card("Clinician review", `Current status: ${words(report.status)}`, form);
}

function whatIfCard(caseId, record, x) {
  const inputs = {};
  const fields = WHAT_IF_FIELDS.filter(([k]) => record[k] !== undefined).map(([k, label]) => {
    inputs[k] = h("input", { id: `wf-${k}`, type: "number", step: "any", value: record[k] ?? "" });
    return h("label", { class: "field", for: `wf-${k}` }, label, inputs[k]);
  });
  const smoker = h("select", { id: "wf-smoker" }, h("option", { value: "true", text: "Yes", selected: record.smoker === true }), h("option", { value: "false", text: "No", selected: record.smoker === false }));
  const creat = h("input", { id: "wf-creat", type: "checkbox", checked: record.creatinine === null });
  const weights = {};
  const weightFields = [["alpha", "alpha (M)"], ["beta", "beta (R)"], ["gamma", "gamma (K)"], ["lam", "lambda (D)"], ["threshold", "threshold"]].map(([k, label]) => {
    weights[k] = h("input", { id: `ww-${k}`, type: "number", min: 0, max: 1, step: 0.05, value: k === "threshold" ? x.threshold : x.weights[k] });
    return h("label", { class: "field", for: `ww-${k}` }, label, weights[k]);
  });
  const out = h("div", { class: "stack", "aria-live": "polite" });
  const form = h("form", { class: "stack", onsubmit: async (e) => {
    e.preventDefault();
    const changes = {};
    for (const [k, el] of Object.entries(inputs)) if (el.value !== "" && Number(el.value) !== record[k]) changes[k] = Number(el.value);
    if (String(record.smoker) !== smoker.value) changes.smoker = smoker.value === "true";
    if (creat.checked && record.creatinine !== null) changes.creatinine_missing = true;
    const w = {};
    for (const [k, el] of Object.entries(weights)) {
      const current = k === "threshold" ? x.threshold : x.weights[k];
      if (el.value !== "" && Math.abs(Number(el.value) - current) > 1e-9) w[k] = Number(el.value);
    }
    try {
      const res = await api(`/cases/${caseId}/what-if`, { method: "POST", body: { changes, weights: Object.keys(w).length ? w : null } });
      out.replaceChildren(
        h("div", { class: "row" }, h("span", { class: "chip", text: `Before: ${DECISION[res.before.decision] || res.before.decision}, C ${fmt(res.before.confidence)}, ${res.before.priority}` }),
          h("span", { class: "chip", text: `After: ${DECISION[res.after.decision] || res.after.decision}, C ${fmt(res.after.confidence)}, ${res.after.priority}` })),
        h("ul", { class: "list" }, res.differences.map((d) => h("li", { text: d }))),
        h("p", { class: "muted", text: res.note }));
    } catch (err) {
      out.replaceChildren(note(err.message, "err"));
    }
  } },
  h("div", { class: "form-grid" }, fields, h("label", { class: "field", for: "wf-smoker" }, "Current smoker", smoker)),
  h("label", { class: "check" }, creat, "Treat creatinine as not measured"),
  h("details", {}, h("summary", { text: "Fusion weights" }), h("div", { class: "form-grid" }, weightFields)),
  h("div", { class: "row" }, h("button", { class: "btn primary", type: "submit", text: "Recompute" }), h("span", { class: "muted", text: "Nothing is saved or audited." })),
  out);
  return card("What if", "Change values or weights and see what would change.", form);
}

async function pageKnowledge() {
  const [pending, all, kb] = await Promise.all([api("/knowledge/updates?status=pending"), api("/knowledge/updates?status=all"), knowledgeBase()]);
  const curator = hasRole("curator");
  const decide = (proposal, approve) => async () => {
    try {
      const res = await api(`/knowledge/updates/${proposal.proposal_id}/decision`, { method: "POST", body: { approve, notes: null } });
      state.kb = null;
      alertBox.replaceChildren(note(res.message, "ok"));
      setTimeout(render, 900);
    } catch (err) {
      alertBox.replaceChildren(note(err.message, "err"));
    }
  };
  const alertBox = h("div", { "aria-live": "polite" });
  const rows = pending.proposals.map((p) => ({ p, cells: [h("span", { class: "mono", text: p.proposal_id }), h("a", { href: `#/case/${p.case_id}`, text: p.case_id }),
    DECISION[p.decision] || p.decision, p.proposed_by, h("td", { class: "wrap" }, p.notes || ""), when(p.created_at),
    curator ? h("div", { class: "row" }, h("button", { class: "btn sm good", type: "button", onclick: decide(p, true), text: "Approve" }),
      h("button", { class: "btn sm danger", type: "button", onclick: decide(p, false), text: "Decline" })) : h("span", { class: "muted", text: "Curator only" })] }));
  const history = h("ol", { class: "list" }, all.history.map((v) => h("li", {}, h("strong", { text: v.label }), ` ${v.description} (${v.approved_by}, ${when(v.created_at)})`)));
  const query = h("input", { id: "kb-q", type: "search", placeholder: "e.g. chest pain on exertion" });
  const hits = h("div", { class: "evidence", "aria-live": "polite" });
  const search = h("form", { class: "row", onsubmit: async (e) => {
    e.preventDefault();
    if (!query.value.trim()) return;
    const res = await api(`/evidence?query=${encodeURIComponent(query.value)}&k=5`);
    hits.replaceChildren(...res.results.map((r) => h("article", { class: "ev" }, h("div", { class: "head" }, h("span", { class: "mono", text: r.source_id }), h("strong", { text: r.title }), h("span", { class: "chip", text: `score ${fmt(r.score)}` })), h("p", { text: r.passage }))));
  } }, h("label", { class: "field", for: "kb-q" }, "Search the knowledge base", query), h("button", { class: "btn", type: "submit", text: "Search" }));
  const passages = table(["Source", "Title", "Topic", "Direction", "From", "Status"], (kb.passages || []).map((p) => ({ cells: [h("span", { class: "mono", text: p.source_id }), p.title, p.topic || "", p.direction,
    p.url ? h("a", { href: p.url, target: "_blank", rel: "noopener noreferrer", text: p.source || "source" }) : (p.source || ""), p.status || ""] })));
  return shell("#/knowledge", "Knowledge", `Index ${pending.index_version}. Validated outcomes join the index only after a Knowledge Curator approves them.`,
    h("div", { class: "stack" }, alertBox,
      card("Pending proposals", curator ? "Approve to create a new index version." : "Only a Knowledge Curator can approve.",
        rows.length ? table(["Proposal", "Case", "Confirmed", "Proposed by", "Notes", "Created", ""], rows) : h("p", { class: "empty", text: "Nothing pending." })),
      h("div", { class: "grid g2" }, card("Index history", null, history), card("Search", "By meaning, with the same embeddings the agent uses.", [search, hits])),
      card("Knowledge base", `${kb.name || ""} ${kb.version || ""}`.trim(), [kb.review && kb.review.note ? note(kb.review.note, "warn") : null, passages])));
}

async function pageAudit() {
  const params = new URLSearchParams(location.hash.split("?")[1] || "");
  const caseId = params.get("case") || "";
  const trail = await api(`/audit?limit=200${caseId ? `&case_id=${encodeURIComponent(caseId)}` : ""}`);
  const input = h("input", { id: "audit-case", type: "search", value: caseId, placeholder: "C-1042" });
  const form = h("form", { class: "row", onsubmit: (e) => { e.preventDefault(); location.hash = input.value.trim() ? `#/audit?case=${encodeURIComponent(input.value.trim())}` : "#/audit"; } },
    h("label", { class: "field", for: "audit-case" }, "Filter by case", input), h("button", { class: "btn", type: "submit", text: "Filter" }));
  const rows = trail.records.map((a) => ({ cells: [when(a.timestamp), h("a", { href: `#/case/${a.case_id}`, text: a.case_id }), words(a.event), a.actor, words(a.status),
    h("span", { class: "mono", text: a.model_version }), a.rulebase_version, h("td", { class: "wrap mono" }, a.retrieval_ids)] }));
  return shell("#/audit", "Audit trail", "Every report and review, append-only: entries cannot be changed or deleted.",
    card("Records", `${trail.records.length} shown`, [form, rows.length ? table(["Time", "Case", "Event", "Actor", "Status", "Model", "Rulebase", "Evidence"], rows) : h("p", { class: "empty", text: "No records." })]));
}

function markdown(text) {
  const out = h("div", { class: "md" });
  const lines = text.split("\n");
  let i = 0;
  const inline = (s) => {
    const parts = s.split(/(\*\*[^*]+\*\*|`[^`]+`)/);
    return parts.map((p) => (p.startsWith("**") ? h("strong", { text: p.slice(2, -2) }) : p.startsWith("`") ? h("code", { class: "mono", text: p.slice(1, -1) }) : p));
  };
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    const image = line.match(/^!\[([^\]]*)\]\(figures\/([a-z_]+\.svg)\)$/);
    if (image) {
      const img = h("img", { class: "figure", alt: image[1] });
      api(`/evaluation/figures/${image[2]}`, { text: true })
        .then((svgText) => { img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgText)}`; })
        .catch(() => { img.alt = `${image[1]} (not available)`; });
      out.append(img);
      i++;
      continue;
    }
    const heading = line.match(/^(#{1,3})\s+(.*)$/);
    if (heading) { out.append(h(`h${heading[1].length}`, {}, inline(heading[2]))); i++; continue; }
    if (line.startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].startsWith("|")) { rows.push(lines[i]); i++; }
      const cells = (r) => r.split("|").slice(1, -1).map((c) => c.trim());
      const body = rows.slice(2).map((r) => ({ cells: cells(r).map(inline) }));
      out.append(table(cells(rows[0]), body));
      continue;
    }
    if (line.startsWith("- ")) {
      const items = [];
      while (i < lines.length && lines[i].startsWith("- ")) { items.push(lines[i].slice(2)); i++; }
      out.append(h("ul", {}, items.map((t) => h("li", {}, inline(t)))));
      continue;
    }
    const para = [];
    while (i < lines.length && lines[i].trim() && !/^(#|\||- )/.test(lines[i])) { para.push(lines[i]); i++; }
    out.append(h("p", {}, inline(para.join(" "))));
  }
  return out;
}

async function pageEvaluation() {
  let text;
  try {
    text = await api("/evaluation", { text: true });
  } catch (err) {
    if (err.status !== 404) throw err;
    return shell("#/evaluation", "Evaluation", "Chapters 6 and 7 of the Stage I report, measured.",
      card("No evaluation yet", null, note("Run uv run medintel evaluate on the server, then reload this page.", "warn")));
  }
  return shell("#/evaluation", "Evaluation", "Chapters 6 and 7 of the Stage I report, measured on held-out data.",
    card("Stage II evaluation", "From reports/evaluation/EVALUATION.md", markdown(text)));
}

async function pageModel() {
  const text = await api("/model-card", { text: true });
  return shell("#/model", "Model card", "Data, submodels, results and limitations of the risk model.", card("Risk model", null, markdown(text)));
}

/* ---------- router ---------- */
async function render() {
  const root = document.getElementById("app");
  root.classList.remove("boot");
  clearInterval(state.poll);
  const hash = location.hash || "#/batches";
  const [path] = hash.split("?");
  if (path === "#/login") {
    root.replaceChildren(pageSignIn(hash.includes("expired=1")));
    document.getElementById("token")?.focus();
    return;
  }
  try {
    if (!state.token && !state.me) {
      const health = await fetch("/healthz").then((r) => r.json());
      if (health.auth) throw new Error("sign in");
    }
    state.me = await api("/me");
  } catch (_) {
    if (location.hash !== "#/login") location.hash = "#/login";
    return;
  }
  const [, page, arg] = path.split("/");
  try {
    let view;
    if (page === "batch" && arg) view = await pageBatch(decodeURIComponent(arg));
    else if (page === "case" && arg) view = await pageCase(decodeURIComponent(arg));
    else if (page === "knowledge") view = await pageKnowledge();
    else if (page === "audit") view = await pageAudit();
    else if (page === "model") view = await pageModel();
    else if (page === "evaluation") view = await pageEvaluation();
    else view = await pageBatches();
    root.replaceChildren(view);
    document.getElementById("page")?.focus({ preventScroll: true });
    window.scrollTo(0, 0);
  } catch (err) {
    if (err.status === 401) return;
    root.replaceChildren(shell("", "Something went wrong", null, note(err.message, "err")));
  }
}

function boot() {
  try {
    const theme = localStorage.getItem("medintel.theme");
    if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
    state.token = sessionStorage.getItem("medintel.token");
  } catch (_) { /* storage unavailable */ }
  window.addEventListener("hashchange", render);
  render();
}
document.addEventListener("DOMContentLoaded", boot);
