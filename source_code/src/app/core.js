(function () {
  'use strict';
  const D = window.MedIntelData;
  const E = window.MedIntelEngine;
  const ICONS = window.MedIntelIcons;
  const MI = (window.MI = { screens: {}, actions: {}, inputs: {}, changes: {} });

  /* ---------- helpers ---------- */
  const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  const esc = (s) => String(s === null || s === undefined ? '' : s).replace(/[&<>"']/g, (c) => ESC[c]);
  const icon = (name, cls = '') => {
    const i = ICONS[name];
    return i ? `<svg class="i ${cls}" viewBox="${i[0]}" aria-hidden="true" focusable="false">${i[1]}</svg>` : '';
  };
  const f2 = (v) => Number(v).toFixed(2);
  const pct = (v) => `${Math.round(Number(v) * 100)}%`;
  const pad = (n) => String(n).padStart(2, '0');
  const fmtTime = (d) => `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  const fmtHM = (d) => `${pad(d.getHours())}:${pad(d.getMinutes())}`;
  const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const fmtDate = (d) => `${DAYS[d.getDay()]} ${d.getDate()} ${MONTHS[d.getMonth()]} ${d.getFullYear()}`;
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const hash = (s) => {
    let h = 2166136261;
    for (let i = 0; i < s.length; i += 1) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
    return (h >>> 0) / 4294967295;
  };
  const reducedMotion = () => window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const store = {
    get(k) {
      try { return window.localStorage.getItem(k); } catch (e) { return null; }
    },
    set(k, v) {
      try { window.localStorage.setItem(k, v); } catch (e) { /* storage unavailable */ }
    },
  };

  /* ---------- vocabulary ---------- */
  const ROLES = {
    clinician: { label: 'Clinician', icon: 'clinician', blurb: 'Reviews reports and records decisions' },
    data_admin: { label: 'Data Administrator', icon: 'dataadmin', blurb: 'Imports and de-identifies records' },
    curator: { label: 'Knowledge Curator', icon: 'curator', blurb: 'Maintains rules, approves knowledge' },
    sys_admin: { label: 'System Administrator', icon: 'sysadmin', blurb: 'Configures models and fusion weights' },
  };
  const AGENTS = {
    ingest: { label: 'Ingestion and validation', short: 'Validate', icon: 'ingest', module: 'DataIngestionService + SchemaValidator', out: 'PatientCase', fr: 'FR1, FR2' },
    ml: { label: 'ML Analysis Agent', short: 'ML', icon: 'ml', module: 'MLAnalysisAgent', out: 'MLResult', fr: 'FR3' },
    rules: { label: 'Rule Validation Agent', short: 'Rules', icon: 'rules', module: 'RuleValidationAgent', out: 'RuleResult', fr: 'FR4' },
    retrieval: { label: 'Knowledge Retrieval Agent', short: 'Retrieval', icon: 'retrieval', module: 'KnowledgeRetrievalAgent', out: 'EvidenceItem list', fr: 'FR5' },
    context: { label: 'Clinical Context Agent', short: 'Context (LLM)', icon: 'context', module: 'ClinicalContextAgent', out: 'ContextSummary', fr: 'FR6' },
    fusion: { label: 'Decision Fusion Engine', short: 'Fusion', icon: 'fusion', module: 'DecisionFusionEngine', out: 'FusionResult', fr: 'FR7' },
    audit: { label: 'Explanation and audit', short: 'Report', icon: 'audit', module: 'ExplanationGenerator + AuditLogger', out: 'DecisionReport + AuditRecord', fr: 'FR8, FR9' },
  };
  const DECISION = {
    high_risk: { label: 'High risk', icon: 'up', cls: 'critical' },
    low_risk: { label: 'Low risk', icon: 'check', cls: 'good' },
    uncertain: { label: 'Uncertain', icon: 'question', cls: 'warning' },
    withheld: { label: 'Withheld', icon: 'ban', cls: 'neutral' },
  };
  const PRIORITY_CLS = { escalate: 'critical', review: 'warning', priority: 'serious', manual: 'neutral', routine: 'good' };
  const PRIORITY_ICON = { escalate: 'warn', review: 'question', priority: 'flag', manual: 'ban', routine: 'check' };
  const STATUS = {
    pending_review: { label: 'Pending review', cls: 'neutral', icon: 'clock' },
    accepted: { label: 'Accepted', cls: 'good', icon: 'okcircle' },
    rejected: { label: 'Rejected', cls: 'critical', icon: 'xcircle' },
    annotated: { label: 'Annotated', cls: 'serious', icon: 'annotate' },
    failed: { label: 'Failed', cls: 'critical', icon: 'warn' },
  };
  const SCREENS = [
    { id: 'overview', label: 'Overview', icon: 'home', sec: 'Today' },
    { id: 'upload', label: 'Upload and validate', icon: 'upload', sec: 'Pipeline' },
    { id: 'run', label: 'Live run', icon: 'run', sec: 'Pipeline' },
    { id: 'triage', label: 'Triage queue', icon: 'triage', sec: 'Clinical review' },
    { id: 'report', label: 'Case report', icon: 'report', sec: 'Clinical review' },
    { id: 'whatif', label: 'What-if lab', icon: 'whatif', sec: 'Explore' },
    { id: 'compare', label: 'Black box vs MedIntel', icon: 'compare', sec: 'Explore' },
    { id: 'governance', label: 'Audit and governance', icon: 'governance', sec: 'Governance' },
    { id: 'about', label: 'Architecture', icon: 'about', sec: 'Governance' },
  ];

  /* ---------- state ---------- */
  const state = {
    screen: 'overview',
    role: 'clinician',
    settings: { ...D.DEFAULT_SETTINGS },
    versions: { model: D.MODEL.version, rulebase: D.RULEBASE.version, index: 3 },
    batch: null,
    cases: {},
    order: [],
    run: { status: 'idle', lanes: [], maxWorkers: 4, speed: 1.6, clock: 0, total: 0 },
    selected: null,
    audit: [],
    proposals: [],
    kbHistory: [{ version: 3, when: null, text: 'Initial index: 10 sample guideline summaries', by: 'Knowledge Curator' }],
    trace: false,
    triageFilter: 'all',
    govTab: 'audit',
    compareId: null,
    whatif: null,
    batchSeq: 0,
  };
  MI.state = state;

  const indexVersion = () => `kb-v${state.versions.index}`;
  const versionsObj = () => ({ model: state.versions.model, rulebase: state.versions.rulebase, index: indexVersion() });

  /* ---------- batch processing ---------- */
  function loadBatch(text, file) {
    const parsed = E.parseCSV(text);
    const res = parsed.missingColumns.length ? { valid: [], rejected: [] } : E.validateRecords(parsed.records);
    state.batch = { file, parsed, valid: res.valid, rejected: res.rejected, loadedAt: new Date(), sent: false };
    return state.batch;
  }

  function prepareCases() {
    const b = state.batch;
    state.cases = {};
    state.batchSeq += 1;
    for (const rec of b.valid) {
      const c = E.buildCase(rec, { file: b.file });
      state.cases[c.case_id] = { bundle: E.analyzeCase(c, state.settings), status: 'pending_review', notes: '', ready: false, analyzedAt: null, reviews: [] };
    }
    b.sent = true;
    reorder();
    const first = state.order[0];
    state.selected = first || null;
    state.compareId = state.order.find((id) => state.cases[id].bundle.priority.key === 'escalate') || first || null;
    state.whatif = null;
  }

  function reorder() {
    state.order = Object.keys(state.cases).sort((a, b) => {
      const A = state.cases[a].bundle;
      const B = state.cases[b].bundle;
      if (A.priority.level !== B.priority.level) return A.priority.level - B.priority.level;
      return B.ml.probabilities.high_risk - A.ml.probabilities.high_risk;
    });
  }

  function recomputeAll() {
    for (const id of Object.keys(state.cases)) {
      const rec = state.cases[id];
      rec.bundle = E.analyzeCase(rec.bundle.case, state.settings);
    }
    reorder();
  }

  function writeAudit(caseId, status, actor, event, when = new Date()) {
    const b = state.cases[caseId].bundle;
    state.audit.unshift({ ...E.auditRecord(b, status, versionsObj(), when), when, actor, event });
  }

  function markReady(caseId, when) {
    const rec = state.cases[caseId];
    if (!rec || rec.ready) return;
    rec.ready = true;
    rec.analyzedAt = when;
    writeAudit(caseId, 'pending_review', 'System', 'Report created', when);
  }

  const readyIds = () => state.order.filter((id) => state.cases[id] && state.cases[id].ready);

  /* ---------- simulated parallel run ---------- */
  const STAGE_KEYS = ['ingest', 'ml', 'rules', 'retrieval', 'context', 'fusion', 'audit'];
  const RANGES = { ingest: [0.08, 0.15], ml: [0.25, 0.55], rules: [0.03, 0.08], retrieval: [0.4, 0.9], context: [1.6, 3.1], fusion: [0.01, 0.02], audit: [0.08, 0.16] };

  function planRun(maxWorkers) {
    const ids = state.order.slice();
    const free = new Array(Math.max(1, maxWorkers)).fill(0);
    const lanes = ids.map((id) => {
      const pid = state.cases[id].bundle.case.patient_id;
      const dur = {};
      for (const k of STAGE_KEYS) {
        const [lo, hi] = RANGES[k];
        dur[k] = Math.round((lo + (hi - lo) * hash(`${pid}:${k}:${state.batchSeq}`)) * 100) / 100;
      }
      const w = free.indexOf(Math.min(...free));
      const start = free[w];
      const st = {};
      let t = start;
      st.ingest = [t, t + dur.ingest];
      t += dur.ingest;
      for (const k of ['ml', 'rules', 'retrieval']) st[k] = [t, t + dur[k]];
      t += Math.max(dur.ml, dur.rules, dur.retrieval);
      st.context = [t, t + dur.context];
      t += dur.context;
      st.fusion = [t, t + dur.fusion];
      t += dur.fusion;
      st.audit = [t, t + dur.audit];
      t += dur.audit;
      free[w] = t;
      return { id, start, end: t, st, dur, worker: w, done: false };
    });
    return { lanes, total: Math.max(0, ...lanes.map((l) => l.end)) };
  }

  let rafId = null;
  function startRun(opts = {}) {
    const run = state.run;
    if (rafId) cancelAnimationFrame(rafId);
    run.maxWorkers = opts.maxWorkers || run.maxWorkers;
    const plan = planRun(run.maxWorkers);
    run.lanes = plan.lanes;
    run.total = plan.total;
    run.clock = 0;
    run.status = 'running';
    run.realStart = performance.now();
    run.startedAt = new Date();
    if (opts.instant || reducedMotion()) {
      finishRun(opts.seedTime);
      return;
    }
    const step = () => {
      run.clock = ((performance.now() - run.realStart) / 1000) * run.speed;
      for (const lane of run.lanes) {
        if (!lane.done && run.clock >= lane.end) {
          lane.done = true;
          markReady(lane.id, new Date());
        }
      }
      if (MI.state.screen === 'run' && MI.screens.run.tick) MI.screens.run.tick();
      updateNavCounts();
      if (run.clock >= run.total) {
        finishRun();
        return;
      }
      rafId = requestAnimationFrame(step);
    };
    rafId = requestAnimationFrame(step);
  }

  function finishRun(seedTime) {
    const run = state.run;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;
    run.clock = run.total;
    run.lanes.forEach((lane, i) => {
      lane.done = true;
      const when = seedTime ? new Date(seedTime.getTime() + Math.round(lane.end * 1000) + i * 37) : new Date();
      markReady(lane.id, when);
    });
    run.status = 'done';
    run.finishedAt = seedTime ? new Date(seedTime.getTime() + Math.round(run.total * 1000)) : new Date();
    if (MI.state.screen === 'run' && MI.screens.run.tick) MI.screens.run.tick();
    updateNavCounts();
    if (MI.onRunDone) {
      const cb = MI.onRunDone;
      MI.onRunDone = null;
      cb();
    }
  }

  /* ---------- reviews and knowledge governance ---------- */
  function review(caseId, status, notes) {
    const rec = state.cases[caseId];
    rec.status = status;
    rec.notes = notes || rec.notes;
    rec.reviews.unshift({ status, notes, when: new Date(), by: ROLES[state.role].label });
    writeAudit(caseId, status, ROLES[state.role].label, `Review: ${STATUS[status].label.toLowerCase()}`);
    const d = rec.bundle.fusion.decision;
    if (status === 'accepted' && (d === 'high_risk' || d === 'low_risk') && !state.proposals.some((p) => p.case_id === caseId && p.status === 'pending')) {
      state.proposals.unshift({ id: `KU-${String(state.proposals.length + 1).padStart(3, '0')}`, case_id: caseId, decision: d, notes: notes || '', created: new Date(), status: 'pending' });
      return true;
    }
    return false;
  }

  function decideProposal(id, approve) {
    const p = state.proposals.find((x) => x.id === id);
    if (!p || p.status !== 'pending') return;
    p.status = approve ? 'approved' : 'declined';
    p.decided = new Date();
    if (approve) {
      state.versions.index += 1;
      state.kbHistory.unshift({ version: state.versions.index, when: p.decided, text: `Added validated outcome from ${p.case_id} (${DECISION[p.decision].label.toLowerCase()} confirmed by a clinician)`, by: 'Knowledge Curator' });
    }
  }

  /* ---------- shared UI pieces ---------- */
  const decisionBadge = (d, lg = false) => {
    const x = DECISION[d];
    return `<span class="badge ${x.cls}${lg ? ' lg' : ''}">${icon(x.icon)}${x.label}</span>`;
  };
  const priorityPill = (p) => `<span class="badge ${PRIORITY_CLS[p.key]}">${icon(PRIORITY_ICON[p.key])}${esc(p.label)}</span>`;
  const statusPill = (s) => `<span class="badge ${STATUS[s].cls}">${icon(STATUS[s].icon)}${STATUS[s].label}</span>`;
  const agentIco = (k, sm = false) => `<span class="agico ag-${k}${sm ? ' sm' : ''}">${icon(AGENTS[k].icon)}</span>`;
  const agentChip = (k, label) => `<span class="agchip ag-${k}">${agentIco(k, true)}${esc(label || AGENTS[k].short)}</span>`;
  const confMeter = (f) => {
    const below = f.confidence < f.threshold;
    return `<div class="conf"><div class="meter" style="--mc:${below ? 'var(--st-warning)' : 'var(--accent)'}" role="img" aria-label="Confidence ${f2(f.confidence)}, threshold ${f2(f.threshold)}"><span style="width:${Math.min(100, f.confidence * 100)}%"></span><i class="tick" style="left:calc(${f.threshold * 100}% - 1px)"></i></div><span class="v num">${f2(f.confidence)}</span></div>`;
  };

  function gauge(f, size = 200) {
    const r = 80;
    const cx = 100;
    const cy = 96;
    const len = Math.PI * r;
    const v = Math.max(0, Math.min(1, f.confidence));
    const below = f.confidence < f.threshold;
    const ang = Math.PI * (1 - f.threshold);
    const tx1 = cx + (r - 14) * Math.cos(ang);
    const ty1 = cy - (r - 14) * Math.sin(ang);
    const tx2 = cx + (r + 14) * Math.cos(ang);
    const ty2 = cy - (r + 14) * Math.sin(ang);
    const lx = cx + (r + 26) * Math.cos(ang);
    const ly = cy - (r + 26) * Math.sin(ang);
    return `<svg class="gauge" width="${size}" height="${size * 0.66}" viewBox="0 0 200 132" role="img" aria-label="Confidence C ${f2(f.confidence)} against threshold ${f2(f.threshold)}">
      <path class="trk" d="M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}" stroke-width="16"/>
      <path class="val" d="M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}" stroke-width="16" stroke="${below ? 'var(--st-warning)' : 'var(--accent)'}" stroke-dasharray="${len}" stroke-dashoffset="${len * (1 - v)}"/>
      <line class="thr" x1="${tx1}" y1="${ty1}" x2="${tx2}" y2="${ty2}"/>
      <text class="gt" x="${lx}" y="${ly}" text-anchor="middle" dominant-baseline="middle">${f2(f.threshold)}</text>
      <text class="gv" x="${cx}" y="${cy - 8}" text-anchor="middle">${f2(f.confidence)}</text>
      <text class="gl" x="${cx}" y="${cy + 18}" text-anchor="middle">Confidence C</text>
    </svg>`;
  }

  function roleGate(role, what) {
    if (state.role === role) return '';
    return `<div class="gate">${icon('lock')}<span style="flex:1;min-width:200px">${esc(what)} is done by the <b>${ROLES[role].label}</b>. You are viewing as ${ROLES[state.role].label}.</span><button class="btn sm primary" data-act="role" data-arg="${role}">${icon(ROLES[role].icon)}Switch to ${ROLES[role].label}</button></div>`;
  }

  function fusionBar(f, opts = {}) {
    const W = (v) => `${Math.max(0, Math.min(1, v)) * 100}%`;
    const t = f.terms;
    const pos = Math.min(1, t.aM + t.bR + t.gK);
    const c = Math.max(0, f.confidence);
    const neg = Math.max(0, pos - c);
    const segs = [['αM', 'var(--a-ml)', t.aM], ['βR', 'var(--a-rules)', t.bR], ['γK', 'var(--a-retrieval)', t.gK]].filter((s) => s[2] > 0.0005);
    let x = 0;
    const top = segs.map(([k, col, v], i) => {
      const last = i === segs.length - 1;
      const s = `<span class="fb-seg${i === 0 ? ' first' : ''}${last ? ' last' : ''}" style="left:${W(x)};width:calc(${W(v)} - ${last ? 0 : 2}px);background:${col}" title="${k} = ${f2(v)}"></span>`;
      x += v;
      return s;
    }).join('');
    let bottom = c > 0.0005 ? `<span class="fb-seg first${neg > 0.0005 ? '' : ' last'}" style="left:0;width:calc(${W(c)} - ${neg > 0.0005 ? 2 : 0}px);background:var(--a-fusion)" title="C = ${f2(c)}"></span>` : '';
    if (neg > 0.0005) bottom += `<span class="fb-neg last${c > 0.0005 ? '' : ' first'}" style="left:${W(c)};width:${W(neg)}" title="λD = ${f2(t.lD)}"></span>`;
    const thr = (label) => `<i class="fb-thr" style="left:${f.threshold * 100}%">${label ? `<b>threshold ${f2(f.threshold)}</b>` : ''}</i>`;
    const ticks = [0, 0.25, 0.5, 0.75, 1].map((v) => `<span style="left:${v * 100}%">${v.toFixed(2)}</span>`).join('');
    return `<div class="fusionbar" ${opts.trace ? 'data-trace="fusion" data-detail="FusionResult: C, its terms and the threshold"' : ''} role="img" aria-label="Support ${f2(pos)} minus disagreement penalty ${f2(t.lD)} gives confidence ${f2(c)} against threshold ${f2(f.threshold)}">
      <div class="fb-cap"><span>Support <span class="muted">αM + βR + γK</span></span><b class="num">${f2(pos)}</b></div>
      <div class="fb-track">${top}${thr(false)}</div>
      <div class="fb-cap"><span>Confidence <span class="muted">C = support − λD</span></span><b class="num">${f2(c)}</b></div>
      <div class="fb-track c">${bottom}${thr(true)}</div>
      <div class="fb-axis">${ticks}</div>
    </div>
    <div class="legend"><span><i class="sw" style="--swc:var(--a-ml)"></i>αM (ML support)</span><span><i class="sw" style="--swc:var(--a-rules)"></i>βR (rule consistency)</span><span><i class="sw" style="--swc:var(--a-retrieval)"></i>γK (retrieval support)</span><span><i class="sw hatch" style="--swc:var(--a-context)"></i>λD (disagreement, subtracted)</span><span><i class="sw" style="--swc:var(--a-fusion)"></i>C (confidence)</span></div>`;
  }

  function equation(f) {
    const w = f.weights;
    const c = f.component_scores;
    return `<div class="eq" aria-label="Confidence equation with values">
      <var>C</var> = <var>α</var><var>M</var> + <var>β</var><var>R</var> + <var>γ</var><var>K</var> &minus; <var>λ</var><var>D</var><br>
      <span class="sr">equals</span>= <span class="n">${f2(w.alpha)}</span>&times;<span class="n">${f2(c.M)}</span> + <span class="n">${f2(w.beta)}</span>&times;<span class="n">${c.R}</span> + <span class="n">${f2(w.gamma)}</span>&times;<span class="n">${f2(c.K)}</span> &minus; <span class="n">${f2(w.lam)}</span>&times;<span class="n">${f2(c.D)}</span> = <span class="n res">${f2(f.confidence)}</span>
    </div>`;
  }

  /* ---------- toasts, modal, popover ---------- */
  function toast(msg, kind = 'ok') {
    const host = $('#toasts');
    if (!host) return;
    const t = document.createElement('div');
    t.className = kind === 'warn' ? 'toast warn' : 'toast';
    t.setAttribute('role', 'status');
    t.innerHTML = `${icon(kind === 'warn' ? 'warn' : 'okcircle')}<span>${esc(msg)}</span>`;
    host.appendChild(t);
    setTimeout(() => t.remove(), 3600);
  }

  let modalReturn = null;
  function openModal({ title, body, foot, wide }) {
    closeModal();
    modalReturn = document.activeElement;
    const o = document.createElement('div');
    o.className = 'overlay';
    o.id = 'modal';
    o.innerHTML = `<div class="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title" style="${wide ? 'width:min(920px,100%)' : ''}">
      <div class="modal-h"><h3 id="modal-title">${title}</h3><button class="btn icon ghost" data-act="close-modal" aria-label="Close">${icon('close')}</button></div>
      <div class="modal-b">${body}</div>${foot ? `<div class="modal-f">${foot}</div>` : ''}</div>`;
    o.addEventListener('click', (e) => {
      if (e.target === o) closeModal();
    });
    document.body.appendChild(o);
    const btn = $('.modal-f .btn.primary', o) || $('.modal-h .btn', o);
    if (btn) btn.focus();
  }
  function closeModal() {
    const m = $('#modal');
    if (m) {
      m.remove();
      if (modalReturn && modalReturn.focus) modalReturn.focus();
    }
  }

  const framed = () => Boolean(window.claude && typeof window.claude.use === 'function');
  let downloads = null;
  let downloadsChecked = !framed();
  function checkDownloads() {
    if (!framed()) return Promise.resolve();
    return window.claude.use('downloads').then((d) => {
      downloads = d;
      downloadsChecked = true;
      const b = $('#save-file');
      if (b) b.hidden = !downloads;
    }).catch(() => {
      downloadsChecked = true;
    });
  }
  const canSave = () => (framed() ? Boolean(downloads) : true);

  async function saveFile(filename, text, mime) {
    if (framed()) {
      if (!downloads) await checkDownloads();
      if (!downloads) {
        toast('Saving files is not available in this view. Use Copy instead.', 'warn');
        return;
      }
      try {
        await downloads.save({ filename, data: text });
        toast(`Saved ${filename}`);
      } catch (e) {
        if (!e || e.code !== 'declined') toast('The file could not be saved here. Use Copy instead.', 'warn');
      }
      return;
    }
    const url = URL.createObjectURL(new Blob([text], { type: mime }));
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 3000);
    toast(`Downloaded ${filename}`);
  }

  let exportPayload = null;
  function openExport({ title, filename, text, mime, intro }) {
    exportPayload = { filename, text, mime };
    openModal({
      title: esc(title),
      wide: true,
      body: `${intro ? `<p class="soft">${intro}</p>` : ''}<div class="row"><span class="chip mono">${icon('box')}${esc(filename)}</span><span class="chip">${(new Blob([text]).size / 1024).toFixed(1)} KB</span></div><pre class="code" id="export-text" tabindex="0">${esc(text)}</pre>`,
      foot: `<button class="btn" data-act="copy-export">${icon('copy')}Copy</button><button class="btn primary" id="save-file" data-act="save-export" ${canSave() ? '' : 'hidden'}>${icon('download')}Save file</button>`,
    });
    if (!downloadsChecked) checkDownloads();
  }
  MI.actions['copy-export'] = () => {
    if (!exportPayload) return;
    const fallback = () => {
      const pre = $('#export-text');
      const sel = window.getSelection();
      const r = document.createRange();
      r.selectNodeContents(pre);
      sel.removeAllRanges();
      sel.addRange(r);
      toast('Text selected. Press Ctrl+C or Cmd+C to copy.', 'warn');
    };
    try {
      navigator.clipboard.writeText(exportPayload.text).then(() => toast('Copied to the clipboard'), fallback);
    } catch (e) {
      fallback();
    }
  };
  MI.actions['save-export'] = () => exportPayload && saveFile(exportPayload.filename, exportPayload.text, exportPayload.mime);
  MI.actions['close-modal'] = () => closeModal();

  function openRoleMenu(anchor) {
    closeRoleMenu();
    const r = anchor.getBoundingClientRect();
    const p = document.createElement('div');
    p.className = 'popover';
    p.id = 'role-menu';
    p.setAttribute('role', 'menu');
    p.innerHTML = `<div class="eyebrow" style="padding:8px 10px 6px">View as</div>${Object.entries(ROLES).map(([k, v]) => `<button role="menuitemradio" aria-checked="${state.role === k}" data-act="role" data-arg="${k}"><span class="av">${icon(v.icon)}</span><strong>${v.label}</strong><span class="ck">${state.role === k ? icon('check') : ''}</span><small>${v.blurb}</small></button>`).join('')}`;
    document.body.appendChild(p);
    const h = p.offsetHeight;
    const top = Math.max(12, Math.min(window.innerHeight - h - 12, r.top - h - 8 < 12 ? r.bottom + 8 : r.top - h - 8));
    p.style.top = `${top}px`;
    p.style.left = `${Math.max(12, Math.min(window.innerWidth - 302, r.left))}px`;
    const first = $('button', p);
    if (first) first.focus();
  }
  function closeRoleMenu() {
    const m = $('#role-menu');
    if (m) m.remove();
  }

  /* ---------- theme ---------- */
  function effectiveTheme() {
    const t = document.documentElement.getAttribute('data-theme');
    if (t === 'dark' || t === 'light') return t;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  function applyStoredTheme() {
    const t = store.get('medintel-theme');
    if (t === 'dark' || t === 'light') document.documentElement.setAttribute('data-theme', t);
  }
  MI.actions.theme = () => {
    const next = effectiveTheme() === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    store.set('medintel-theme', next);
    renderTopbar();
  };

  /* ---------- shell ---------- */
  const LOGO_SVG = `<svg viewBox="0 0 40 40" aria-hidden="true"><rect x="1" y="1" width="38" height="38" rx="11" fill="#0e7c86"/><rect x="1" y="1" width="38" height="38" rx="11" fill="none" stroke="rgba(255,255,255,.25)"/><path d="M6 22h6.5l2.8-7 4.6 14 3.6-10.5 2.2 3.5H34" fill="none" stroke="#fff" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/><circle cx="34" cy="22" r="2.6" fill="#fff"/></svg>`;
  MI.LOGO_SVG = LOGO_SVG;

  function renderShell() {
    const secs = [];
    for (const s of SCREENS) {
      if (!secs.length || secs[secs.length - 1].sec !== s.sec) secs.push({ sec: s.sec, items: [] });
      secs[secs.length - 1].items.push(s);
    }
    document.getElementById('app').innerHTML = `
      <aside class="rail" id="rail" aria-label="Main navigation">
        <a class="brandmark" href="#overview" data-act="nav" data-arg="overview">${LOGO_SVG}<span class="bm-text"><b>MedIntel</b><span>Clinical console</span></span></a>
        <nav class="nav">${secs.map((g) => `<div class="nav-sec">${g.sec}</div>${g.items.map((s) => `<button data-act="nav" data-arg="${s.id}" title="${s.label}" aria-label="${s.label}">${icon(s.icon)}<span class="lbl">${s.label}</span><span class="count" data-count="${s.id}" hidden></span></button>`).join('')}`).join('')}</nav>
        <div class="rail-foot">
          <button class="role-btn" id="role-btn" data-act="role-menu" aria-haspopup="menu" title="Change role"></button>
          <p class="proto-note">Concept prototype for Project Stage I. Synthetic patients only; not a clinical tool.</p>
        </div>
      </aside>
      <div class="scrim-nav" id="scrim-nav" data-act="close-nav" hidden></div>
      <div class="main">
        <header class="topbar"><div class="topbar-in" id="topbar"></div></header>
        <main class="page" id="view" tabindex="-1"></main>
        <footer class="foot"><div class="foot-in"><span>MedIntel concept prototype · synthetic, de-identified sample data · not for clinical use</span><span>BE Computer Engineering Project Stage I · JESITMR Nashik · AY 2026-27</span></div></footer>
      </div>`;
  }

  function renderTopbar() {
    const scr = MI.screens[state.screen];
    const title = scr && scr.title ? scr.title() : '';
    const sub = scr && scr.subtitle ? scr.subtitle() : '';
    const dark = effectiveTheme() === 'dark';
    $('#topbar').innerHTML = `
      <button class="btn icon ghost menu-btn" data-act="open-nav" aria-label="Open navigation">${icon('menu')}</button>
      <div class="title-wrap"><h1>${title}</h1>${sub ? `<p>${sub}</p>` : ''}</div>
      <div class="top-actions">
        <span class="proto-pill hide-s" title="Concept prototype with synthetic data">${icon('flag')}Concept prototype · synthetic data</span>
        <button class="btn primary" data-act="tour-start" id="tour-btn">${icon('play')}<span class="hide-s">Guided demo</span></button>
        <button class="btn icon" data-act="theme" aria-label="Switch to ${dark ? 'light' : 'dark'} theme" title="Switch to ${dark ? 'light' : 'dark'} theme">${icon(dark ? 'sun' : 'moon')}</button>
      </div>`;
  }

  function renderRoleButton() {
    const r = ROLES[state.role];
    const b = $('#role-btn');
    if (b) b.innerHTML = `<span class="av">${icon(r.icon)}</span><span class="rb-text"><small>Viewing as</small><strong>${r.label}</strong></span>${icon('chevdown', 'chev')}`;
  }

  function updateNavCounts() {
    const needs = readyIds().filter((id) => state.cases[id].bundle.priority.level <= 2 && state.cases[id].status === 'pending_review').length;
    const pending = state.proposals.filter((p) => p.status === 'pending').length;
    const set = (id, n) => {
      const el = $(`[data-count="${id}"]`);
      if (!el) return;
      el.hidden = !n;
      el.textContent = n;
    };
    set('triage', needs);
    set('governance', pending);
    set('run', state.run.status === 'running' ? '…' : 0);
  }

  function updateNav() {
    $$('.nav button').forEach((b) => {
      if (b.dataset.arg === state.screen) b.setAttribute('aria-current', 'page');
      else b.removeAttribute('aria-current');
    });
    updateNavCounts();
  }

  function render(opts = {}) {
    const scr = MI.screens[state.screen];
    document.body.classList.toggle('trace-on', state.screen === 'report' && state.trace);
    renderTopbar();
    renderRoleButton();
    updateNav();
    const view = $('#view');
    view.innerHTML = scr.render();
    if (!opts.keepScroll) window.scrollTo(0, 0);
    if (scr.mount) scr.mount();
  }
  MI.render = render;

  function navigate(screen, opts = {}) {
    if (!MI.screens[screen]) screen = 'overview';
    if (opts.caseId && state.cases[opts.caseId]) state.selected = opts.caseId;
    const same = state.screen === screen;
    if (!same && MI.screens[state.screen] && MI.screens[state.screen].unmount) MI.screens[state.screen].unmount();
    state.screen = screen;
    closeNav();
    closeRoleMenu();
    try {
      if (location.hash.slice(1) !== screen) history.replaceState(null, '', `#${screen}`);
    } catch (e) { /* history unavailable in this frame */ }
    render({ keepScroll: same && opts.keepScroll });
    if (!opts.noFocus) {
      const v = $('#view');
      if (v) v.focus({ preventScroll: true });
    }
  }
  MI.navigate = navigate;

  function openNav() {
    $('#rail').classList.add('open');
    $('#scrim-nav').hidden = false;
  }
  function closeNav() {
    const r = $('#rail');
    if (r) r.classList.remove('open');
    const s = $('#scrim-nav');
    if (s) s.hidden = true;
  }

  /* ---------- global actions ---------- */
  MI.actions.nav = (el) => navigate(el.dataset.arg);
  MI.actions['open-nav'] = () => openNav();
  MI.actions['close-nav'] = () => closeNav();
  MI.actions['role-menu'] = (el) => ($('#role-menu') ? closeRoleMenu() : openRoleMenu(el));
  MI.actions.role = (el) => setRole(el.dataset.arg);
  MI.actions.open = (el) => navigate('report', { caseId: el.dataset.arg });

  function setRole(role, quiet) {
    if (!ROLES[role]) return;
    state.role = role;
    closeRoleMenu();
    renderRoleButton();
    render({ keepScroll: true });
    if (!quiet) toast(`Viewing as ${ROLES[role].label}`);
  }
  MI.setRole = setRole;

  /* ---------- event delegation ---------- */
  document.addEventListener('click', (e) => {
    const el = e.target.closest('[data-act]');
    if (!el) {
      if (!e.target.closest('#role-menu') && !e.target.closest('#role-btn')) closeRoleMenu();
      return;
    }
    const fn = MI.actions[el.dataset.act];
    if (!fn) return;
    if (el.tagName === 'A') e.preventDefault();
    fn(el, e);
  });
  document.addEventListener('input', (e) => {
    const el = e.target.closest('[data-in]');
    if (el && MI.inputs[el.dataset.in]) MI.inputs[el.dataset.in](el, e);
  });
  document.addEventListener('change', (e) => {
    const el = e.target.closest('[data-ch]');
    if (el && MI.changes[el.dataset.ch]) MI.changes[el.dataset.ch](el, e);
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if ($('#modal')) closeModal();
      closeRoleMenu();
      closeNav();
    }
  });
  window.addEventListener('hashchange', () => {
    const t = location.hash.slice(1);
    if (t && t !== state.screen && MI.screens[t]) navigate(t);
  });

  Object.assign(MI, {
    D, E, ICONS, esc, icon, f2, pct, fmtTime, fmtHM, fmtDate, $, $$, hash, store, reducedMotion,
    ROLES, AGENTS, DECISION, PRIORITY_CLS, PRIORITY_ICON, STATUS, SCREENS, STAGE_KEYS,
    loadBatch, prepareCases, reorder, recomputeAll, writeAudit, readyIds, startRun, finishRun, review, decideProposal,
    decisionBadge, priorityPill, statusPill, agentIco, agentChip, confMeter, gauge, roleGate, fusionBar, equation,
    toast, openModal, closeModal, openExport, saveFile, renderShell, renderTopbar, renderRoleButton, updateNav, updateNavCounts,
    applyStoredTheme, effectiveTheme, indexVersion, versionsObj, checkDownloads,
  });
})();
