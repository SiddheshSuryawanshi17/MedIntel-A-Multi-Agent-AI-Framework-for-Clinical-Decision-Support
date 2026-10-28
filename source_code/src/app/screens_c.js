(function () {
  'use strict';
  const MI = window.MI;
  const { D, E, esc, icon, f2, state } = MI;

  /* ---------------- What-if lab ---------------- */
  const SLIDERS = {
    vitals: [
      ['systolic_bp', 'Systolic BP', 'mmHg', 80, 230, 1],
      ['diastolic_bp', 'Diastolic BP', 'mmHg', 40, 140, 1],
      ['heart_rate', 'Heart rate', 'bpm', 40, 170, 1],
    ],
    labs: [
      ['total_cholesterol', 'Total cholesterol', 'mg/dL', 100, 340, 1],
      ['fasting_glucose', 'Fasting glucose', 'mg/dL', 60, 320, 1],
      ['creatinine', 'Creatinine', 'mg/dL', 0.4, 3.0, 0.1],
    ],
    patient: [
      ['age', 'Age', 'years', 10, 90, 1],
      ['bmi', 'BMI', 'kg/m²', 16, 45, 0.1],
    ],
    weights: [
      ['alpha', 'α · weight of ML support M', '', 0, 1, 0.05],
      ['beta', 'β · weight of rule consistency R', '', 0, 1, 0.05],
      ['gamma', 'γ · weight of retrieval support K', '', 0, 1, 0.05],
      ['lam', 'λ · penalty on disagreement D', '', 0, 1, 0.05],
      ['threshold', 'Decision threshold', '', 0.3, 0.9, 0.05],
    ],
  };

  function wiInit(baseId) {
    const ids = MI.readyIds();
    const id = baseId && state.cases[baseId] ? baseId : (ids.find((x) => state.cases[x].bundle.priority.key === 'escalate') || ids[0]);
    if (!id) return null;
    const c = state.cases[id].bundle.case;
    state.whatif = { base: id, xs: { ...c.x_s }, missing: c.x_s.creatinine === null, s: { ...state.settings }, log: [], prev: null };
    if (state.whatif.missing) state.whatif.xs.creatinine = 1.0;
    return state.whatif;
  }

  function wiCase() {
    const w = state.whatif;
    const base = state.cases[w.base].bundle.case;
    const xs = { ...w.xs, creatinine: w.missing ? null : w.xs.creatinine };
    return { ...base, x_s: xs, scripted: false };
  }

  function sliderHTML(group, [key, label, unit, min, max, step]) {
    const w = state.whatif;
    const isW = group === 'weights';
    const v = isW ? w.s[key] : w.xs[key];
    const f = !isW && E.field(key);
    const disabled = key === 'creatinine' && w.missing;
    return `<div class="ctl"><label for="wi-${key}"><span>${esc(label)}</span><b id="wv-${key}" class="num">${fmtVal(key, v, unit)}</b></label>
      <input type="range" id="wi-${key}" data-in="wi" data-key="${key}" data-group="${isW ? 'w' : 'x'}" min="${min}" max="${max}" step="${step}" value="${v}" ${disabled ? 'disabled' : ''}>
      ${f && f.refText ? `<span class="ref">reference ${esc(f.refText)} ${esc(f.unit || '')}${key === 'systolic_bp' ? ' · rule R-001 fires at 180' : key === 'heart_rate' ? ' · rule R-004 fires at 120' : ''}</span>` : ''}${key === 'age' ? '<span class="ref">rule R-003 withholds the result under 18</span>' : ''}</div>`;
  }
  const fmtVal = (key, v, unit) => (['alpha', 'beta', 'gamma', 'lam', 'threshold'].includes(key) ? f2(v) : `${key === 'creatinine' || key === 'bmi' ? Number(v).toFixed(1) : Math.round(v)}${unit ? ` ${unit}` : ''}`);

  function wiOutput() {
    const w = state.whatif;
    const b = E.analyzeCase(wiCase(), w.s);
    const f = b.fusion;
    const fired = b.rules.checks.filter((k) => k.fired);
    const now = { decision: f.decision, rules: b.rules.triggered.join(','), ev: b.evidence.length, conflicts: b.context.conflicts.length, below: f.below_threshold };
    const p = w.prev;
    if (p) {
      const t = MI.fmtTime(new Date());
      if (p.rules !== now.rules) {
        const added = b.rules.triggered.filter((r) => !p.rules.split(',').includes(r));
        const removed = p.rules.split(',').filter((r) => r && !b.rules.triggered.includes(r));
        added.forEach((r) => w.log.unshift({ t, i: 'rules', text: `Rule ${r} fired: ${D.RULEBASE.rules.find((x) => x.rule_id === r).message}` }));
        removed.forEach((r) => w.log.unshift({ t, i: 'rules', text: `Rule ${r} no longer fires.` }));
      }
      if (p.ev !== now.ev) w.log.unshift({ t, i: 'retrieval', text: now.ev ? `Evidence retrieved: ${b.evidence.map((e) => e.source_id).join(', ')}.` : 'No sufficiently relevant evidence: K drops to 0.' });
      if (p.conflicts !== now.conflicts) w.log.unshift({ t, i: 'warn', text: `${now.conflicts} conflict${now.conflicts === 1 ? '' : 's'} reported; D is now ${f2(f.disagreement)}.` });
      if (p.decision !== now.decision) w.log.unshift({ t, i: 'fusion', text: `Decision changed: ${MI.DECISION[p.decision].label} to ${MI.DECISION[now.decision].label}.` });
      else if (p.below !== now.below) w.log.unshift({ t, i: 'fusion', text: `C crossed the threshold (${f2(f.confidence)} vs ${f2(f.threshold)}).` });
      w.log = w.log.slice(0, 30);
    }
    w.prev = now;
    const top3 = b.ml.importance.slice(0, 3).map((x) => `<div class="mrow" style="font-size:12.5px"><span class="soft">${esc(x.label)}</span><span class="num" style="color:${x.contribution >= 0 ? 'var(--sti-critical)' : 'var(--sti-good)'}">${x.contribution >= 0 ? '+' : '−'}${Math.abs(x.contribution).toFixed(2)}</span></div>`).join('');
    return `
      <div class="card" id="wi-result">
        <div class="card-b stack" style="padding-top:16px;gap:14px">
          <div class="row" style="align-items:center;gap:18px">
            <div class="stack" style="gap:8px;flex:1;min-width:220px">
              <span class="eyebrow">Live result</span>
              <div id="wi-decision">${MI.decisionBadge(f.decision, true)}</div>
              <p style="font-size:14px">${esc(b.recommendation)}</p>
            </div>
            ${MI.gauge(f, 190)}
          </div>
          ${MI.equation(f)}
          ${MI.fusionBar(f)}
        </div>
      </div>
      <div class="mini">
        <div class="card"><h4>${MI.agentIco('ml', true)}ML Analysis Agent</h4><div class="probbar"><div class="pb-top"><span>Probability of high risk</span><b class="num">${MI.pct(b.ml.probabilities.high_risk)}</b></div><div class="meter" style="--mc:var(--a-ml)"><span style="width:${b.ml.probabilities.high_risk * 100}%"></span><i class="tick" style="left:calc(50% - 1px)"></i></div></div>${top3}</div>
        <div class="card" id="wi-rules"><h4>${MI.agentIco('rules', true)}Rule Validation Agent</h4>${fired.length ? fired.map((k) => `<div class="row"><span class="badge ${k.rule.action === 'elevate' ? 'critical' : k.rule.action === 'flag' ? 'serious' : 'neutral'}">${k.rule.rule_id} · ${k.rule.action}</span><span class="soft" style="font-size:12.5px">${esc(k.rule.message)}</span></div>`).join('') : '<span class="soft">No rule fired.</span>'}</div>
        <div class="card"><h4>${MI.agentIco('retrieval', true)}Knowledge Retrieval Agent</h4>${b.evidence.length ? b.evidence.map((e) => `<div class="mrow"><span><span class="srcid">${e.source_id}</span> <span class="soft" style="font-size:12.5px">${esc(e.title)}</span></span><span class="num soft">${e.score.toFixed(2)}</span></div>`).join('') : `<span class="soft">No sufficiently relevant evidence (K = 0).</span>`}</div>
        <div class="card"><h4>${MI.agentIco('context', true)}Conflicts (${b.context.conflicts.length})</h4>${b.context.conflicts.length ? b.context.conflicts.map((t) => `<div class="conflict">${icon('warn')}<span>${esc(t)}</span></div>`).join('') : '<span class="soft">The components agree.</span>'}</div>
      </div>
      <div class="card"><div class="card-h"><h3>What changed</h3><span class="sub">most recent first</span></div><div class="card-b"><div class="changes" id="wi-log">${w.log.length ? w.log.map((x) => `<div>${icon(x.i)}<span><span class="muted num">${x.t}</span> ${esc(x.text)}</span></div>`).join('') : '<span class="muted">Move a slider to see how each agent responds.</span>'}</div></div></div>`;
  }

  function refreshWi() {
    const out = MI.$('#wi-out');
    if (out) out.innerHTML = wiOutput();
  }
  MI.refreshWi = refreshWi;

  MI.screens.whatif = {
    title: () => 'What-if lab',
    subtitle: () => 'Change a patient value or a fusion weight and watch every agent respond',
    render() {
      if (!state.whatif || !state.cases[state.whatif.base]) wiInit();
      const w = state.whatif;
      if (!w) return `<div class="card"><div class="empty">${icon('whatif')}<b>Nothing to explore yet</b><span>Run a batch first; the lab starts from a real case.</span></div></div>`;
      w.prev = null;
      const ids = MI.readyIds();
      return `
      <div class="wi">
        <div class="card" id="wi-inputs">
          <div class="card-h"><h2>Inputs</h2><div class="end"><button class="btn sm" data-act="wi-reset">${icon('reset')}Reset</button></div></div>
          <div class="card-b">
            <div class="ctl" style="margin-bottom:6px"><label for="wi-base"><span>Start from case</span></label>
              <select class="sel" id="wi-base" data-ch="wi-base">${ids.map((id) => `<option value="${id}" ${id === w.base ? 'selected' : ''}>${id} · ${MI.demog(state.cases[id].bundle.case)}</option>`).join('')}</select></div>
            <div class="grp"><h4>Vital signs</h4>${SLIDERS.vitals.map((s) => sliderHTML('x', s)).join('')}
              <button class="btn sm" data-act="wi-sweep" id="wi-sweep">${icon('play')}Sweep systolic BP from 170 to 190</button></div>
            <div class="grp"><h4>Laboratory</h4>${SLIDERS.labs.map((s) => sliderHTML('x', s)).join('')}
              <label class="switch"><input type="checkbox" id="wi-missing" data-ch="wi-missing" ${w.missing ? 'checked' : ''}>Creatinine result missing</label></div>
            <div class="grp"><h4>Patient</h4>${SLIDERS.patient.map((s) => sliderHTML('x', s)).join('')}
              <div class="row"><span class="muted" style="font-size:12.5px">Sex</span><div class="seg" role="group" aria-label="Sex">${['F', 'M'].map((v) => `<button data-act="wi-sex" data-arg="${v}" aria-pressed="${w.xs.sex === v}">${v === 'F' ? 'Female' : 'Male'}</button>`).join('')}</div>
              <label class="switch"><input type="checkbox" id="wi-smoker" data-ch="wi-smoker" ${w.xs.smoker ? 'checked' : ''}>Current smoker</label></div></div>
            <div class="grp"><h4>Fusion weights</h4>${SLIDERS.weights.map((s) => sliderHTML('weights', s)).join('')}
              <span class="ref muted" style="font-size:11.5px">α, β and γ always sum to 1, so C cannot exceed 1.</span></div>
          </div>
        </div>
        <div class="stack" id="wi-out" style="gap:18px">${wiOutput()}</div>
      </div>`;
    },
    unmount() {
      stopSweep();
    },
  };

  function setWeight(key, v) {
    const s = state.whatif.s;
    if (['alpha', 'beta', 'gamma'].includes(key)) {
      const others = ['alpha', 'beta', 'gamma'].filter((k) => k !== key);
      const rest = others.reduce((a, k) => a + s[k], 0);
      s[key] = v;
      others.forEach((k) => {
        s[k] = rest > 0 ? ((1 - v) * s[k]) / rest : (1 - v) / 2;
      });
      ['alpha', 'beta', 'gamma'].forEach((k) => {
        const el = MI.$(`#wi-${k}`);
        if (el && k !== key) el.value = s[k];
        const lab = MI.$(`#wv-${k}`);
        if (lab) lab.textContent = f2(s[k]);
      });
    } else s[key] = v;
  }

  MI.inputs.wi = (el) => {
    const key = el.dataset.key;
    const v = Number(el.value);
    if (el.dataset.group === 'w') setWeight(key, v);
    else state.whatif.xs[key] = v;
    const s = [...SLIDERS.vitals, ...SLIDERS.labs, ...SLIDERS.patient, ...SLIDERS.weights].find((x) => x[0] === key);
    const lab = MI.$(`#wv-${key}`);
    if (lab) lab.textContent = fmtVal(key, v, s ? s[2] : '');
    if (el.dataset.user !== 'auto') stopSweep();
    refreshWi();
  };
  MI.changes['wi-base'] = (el) => {
    stopSweep();
    wiInit(el.value);
    MI.render({ keepScroll: true });
  };
  MI.changes['wi-missing'] = (el) => {
    state.whatif.missing = el.checked;
    const c = MI.$('#wi-creatinine');
    if (c) c.disabled = el.checked;
    refreshWi();
  };
  MI.changes['wi-smoker'] = (el) => {
    state.whatif.xs.smoker = el.checked;
    refreshWi();
  };
  MI.actions['wi-sex'] = (el) => {
    state.whatif.xs.sex = el.dataset.arg;
    MI.$$('[data-act="wi-sex"]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.arg === el.dataset.arg)));
    refreshWi();
  };
  MI.actions['wi-reset'] = () => {
    stopSweep();
    wiInit(state.whatif ? state.whatif.base : null);
    MI.render({ keepScroll: true });
  };

  let sweepTimer = null;
  function stopSweep() {
    if (sweepTimer) clearInterval(sweepTimer);
    sweepTimer = null;
    const b = MI.$('#wi-sweep');
    if (b) b.innerHTML = `${icon('play')}Sweep systolic BP from 170 to 190`;
  }
  function sweep(from = 170, to = 190, onDone) {
    stopSweep();
    const el = MI.$('#wi-systolic_bp');
    if (!el) return;
    let v = from;
    el.value = v;
    el.dataset.user = 'auto';
    el.dispatchEvent(new Event('input', { bubbles: true }));
    const b = MI.$('#wi-sweep');
    if (b) b.innerHTML = `${icon('pause')}Stop the sweep`;
    sweepTimer = setInterval(() => {
      v += 1;
      el.value = v;
      el.dispatchEvent(new Event('input', { bubbles: true }));
      if (v >= to) {
        delete el.dataset.user;
        stopSweep();
        if (onDone) onDone();
      }
    }, MI.reducedMotion() ? 20 : 170);
  }
  MI.sweep = sweep;
  MI.actions['wi-sweep'] = () => (sweepTimer ? stopSweep() : sweep());

  /* ---------------- Governance ---------------- */
  const TABS = [['audit', 'Audit trail', 'audit'], ['knowledge', 'Knowledge updates', 'database'], ['rulebase', 'Rulebase', 'rules'], ['settings', 'Models and settings', 'gear']];

  function auditTab() {
    const ridText = (r) => (Array.isArray(r) ? r : String(r || '').split(',')).filter(Boolean).join(', ') || '-';
    const rows = [...state.audit].sort((a, b) => b.when - a.when).map((a) => `<tr><td class="num">${MI.fmtTime(a.when)}</td><td class="mono nowrap">${a.case_id}</td><td>${MI.statusPill(a.status)}</td><td>${esc(a.event)}</td><td>${esc(a.actor)}</td><td class="mono" style="font-size:11.5px">${esc(a.model_version)}</td><td class="mono">${esc(a.rulebase_version)}</td><td class="mono" style="font-size:11.5px;min-width:180px">${esc(ridText(a.retrieval_ids))}</td></tr>`).join('');
    return `<div class="card"><div class="card-h"><h2>Audit trail</h2><span class="sub">${state.audit.length} records · append-only, newest first</span><div class="end"><button class="btn sm" data-act="export-audit">${icon('download')}Export</button></div></div>
      <div class="card-b"><div class="scroll-x"><table class="tbl" id="audit-table"><thead><tr><th>Time</th><th>Case</th><th>Status</th><th>Event</th><th>Actor</th><th>Model</th><th>Rulebase</th><th>Retrieval IDs</th></tr></thead><tbody>${rows || '<tr><td colspan="8" class="muted">No records yet.</td></tr>'}</tbody></table></div>
      <p class="muted" style="font-size:12px;margin-top:10px">Each AuditRecord keeps the case ID, model and rulebase versions, retrieval IDs, status and timestamp (Section 4.4.2), so any report can be reproduced later (NFR6).</p></div></div>`;
  }

  function knowledgeTab() {
    const pending = state.proposals.filter((p) => p.status === 'pending');
    const done = state.proposals.filter((p) => p.status !== 'pending');
    const can = state.role === 'curator';
    return `<div class="grid g2" style="align-items:start">
      <div class="card"><div class="card-h"><h2>Pending updates</h2><span class="sub">validated outcomes waiting for approval</span></div>
        <div class="card-b">${MI.roleGate('curator', 'Approving knowledge updates')}
          ${pending.length ? pending.map((p) => `<div class="prop" id="prop-${p.id}"><div class="stack" style="gap:4px"><span class="row"><b class="mono">${p.id}</b>${MI.decisionBadge(p.decision)}<span class="muted">from ${p.case_id}</span></span><span class="soft" style="font-size:12.5px">Accepted by a clinician at ${MI.fmtTime(p.created)}${p.notes ? `: “${esc(p.notes)}”` : ''}</span></div>
            <div class="row"><button class="btn sm good" data-act="kb-approve" data-arg="${p.id}" ${can ? '' : 'disabled'}>${icon('check')}Approve</button><button class="btn sm danger" data-act="kb-decline" data-arg="${p.id}" ${can ? '' : 'disabled'}>${icon('close')}Decline</button></div></div>`).join('')
            : `<div class="empty">${icon('inbox')}<span>No pending updates. When a clinician accepts a report, its outcome appears here.</span></div>`}
          ${done.length ? `<div class="eyebrow" style="margin-top:14px">Decided</div>${done.map((p) => `<div class="prop"><span><b class="mono">${p.id}</b> <span class="muted">from ${p.case_id}</span></span><span class="badge ${p.status === 'approved' ? 'good' : 'neutral'}">${p.status}</span></div>`).join('')}` : ''}
        </div></div>
      <div class="card" id="kb-card"><div class="card-h"><h2>Knowledge index</h2><span class="sub">re-indexed only through add_validated</span></div>
        <div class="card-b stack">
          <div class="row"><span class="bigver" id="kb-ver">${MI.indexVersion()}</span><span class="chip">${D.KB.length} sample passages</span></div>
          <div class="eyebrow">Version history</div>
          <div class="alog">${state.kbHistory.map((h) => `<div><time>${h.when ? MI.fmtTime(h.when) : 'initial'}</time><span><b class="mono">kb-v${h.version}</b> <span class="soft">${esc(h.text)} · ${esc(h.by)}</span></span></div>`).join('')}</div>
          <div class="note">${icon('lock')}<span>Model outputs never update the knowledge store on their own. One wrong recommendation cannot become a lasting source of error without a clinician and a curator agreeing first.</span></div>
        </div></div>
    </div>`;
  }

  function rulebaseTab() {
    const json = JSON.stringify(D.RULEBASE, null, 2);
    return `<div class="grid g2" style="align-items:start">
      <div class="card"><div class="card-h"><h2>Rules in force</h2><span class="ver">rulebase ${D.RULEBASE.version}</span></div><div class="card-b">
        <div class="scroll-x"><table class="tbl"><thead><tr><th>Rule</th><th>Condition</th><th>Action</th><th>Message</th></tr></thead><tbody>${D.RULEBASE.rules.map((r) => `<tr><td class="mono">${r.rule_id}</td><td class="mono" style="font-size:12px">${r.operator === 'missing' ? `${r.field} is missing` : `${r.field} ${r.operator === '>=' ? '≥' : r.operator} ${r.threshold}`}</td><td><span class="badge ${r.action === 'elevate' ? 'critical' : r.action === 'flag' ? 'serious' : 'neutral'}">${r.action} · ${r.severity}</span></td><td class="soft" style="font-size:12.5px">${esc(r.message)}</td></tr>`).join('')}</tbody></table></div>
        <p class="muted" style="font-size:12px;margin-top:10px">The most severe fired rule sets the action; at equal severity, suppress outranks elevate and elevate outranks flag.</p></div></div>
      <div class="card"><div class="card-h"><h2>rulebase.json</h2><div class="end"><button class="btn sm" data-act="export-rules">${icon('download')}Export</button></div></div><div class="card-b"><pre class="code">${esc(json)}</pre></div></div>
    </div>`;
  }

  function settingsTab() {
    const can = state.role === 'sys_admin';
    const s = state.settings;
    const row = (k, label) => `<div class="ctl"><label for="set-${k}"><span>${label}</span><b class="num" id="sv-${k}">${f2(s[k])}</b></label><input type="range" id="set-${k}" data-in="set" data-key="${k}" min="${k === 'threshold' ? 0.3 : 0}" max="${k === 'threshold' ? 0.9 : 1}" step="0.05" value="${s[k]}" ${can ? '' : 'disabled'}></div>`;
    return `<div class="grid g2" style="align-items:start">
      <div class="card"><div class="card-h"><h2>Model registry</h2></div><div class="card-b">
        <div class="scroll-x"><table class="tbl"><thead><tr><th>Model</th><th>Status</th><th>Notes</th></tr></thead><tbody>
          <tr><td class="mono">risk-lr 0.3</td><td><span class="badge good">${icon('check')}active</span></td><td class="soft" style="font-size:12.5px">Logistic regression with demo weights over 8 features. Stage II trains it on a public dataset.</td></tr>
          <tr><td class="mono">risk-lr 0.2</td><td><span class="badge neutral">retired</span></td><td class="soft" style="font-size:12.5px">Kept so that older reports stay reproducible.</td></tr>
        </tbody></table></div>
        <dl class="kv" style="margin-top:14px"><dt>Retrieval</dt><dd>k = ${s.k}, minimum similarity ${f2(s.min_score)}</dd><dt>LLM endpoint</dt><dd>configurable: hosted API or a local model (scripted in this prototype)</dd><dt>Audit store</dt><dd>SQLite in development, PostgreSQL in deployment</dd></dl>
      </div></div>
      <div class="card"><div class="card-h"><h2>Fusion settings</h2><span class="sub">applied to every case</span></div><div class="card-b stack">
        ${MI.roleGate('sys_admin', 'Changing fusion settings')}
        ${row('alpha', 'α · ML support')}${row('beta', 'β · rule consistency')}${row('gamma', 'γ · retrieval support')}${row('lam', 'λ · disagreement penalty')}${row('threshold', 'Decision threshold')}
        <p class="muted" style="font-size:12px">α, β and γ are rescaled to sum to 1 when applied. In Stage II these are tuned on a validation split only.</p>
        <div class="row"><button class="btn primary" data-act="apply-settings" ${can ? '' : 'disabled'}>${icon('check')}Apply to all cases</button><button class="btn" data-act="reset-settings" ${can ? '' : 'disabled'}>${icon('reset')}Restore defaults</button></div>
      </div></div>
    </div>`;
  }

  MI.screens.governance = {
    title: () => 'Audit and governance',
    subtitle: () => 'Append-only audit trail and controlled knowledge updates (NFR1, NFR6)',
    render() {
      const body = { audit: auditTab, knowledge: knowledgeTab, rulebase: rulebaseTab, settings: settingsTab }[state.govTab] || auditTab;
      return `<div class="tabs" role="tablist">${TABS.map(([k, label, ic]) => `<button role="tab" id="tab-${k}" aria-selected="${state.govTab === k}" data-act="gov-tab" data-arg="${k}">${icon(ic)}${label}${k === 'knowledge' && state.proposals.some((p) => p.status === 'pending') ? ` <span class="badge serious" style="padding:1px 7px">${state.proposals.filter((p) => p.status === 'pending').length}</span>` : ''}</button>`).join('')}</div>${body()}`;
    },
  };
  MI.actions['gov-tab'] = (el) => {
    state.govTab = el.dataset.arg;
    MI.render({ keepScroll: true });
  };
  MI.actions['export-audit'] = () => {
    const rows = state.audit.map((a) => ({ case_id: a.case_id, model_version: a.model_version, rulebase_version: a.rulebase_version, retrieval_ids: a.retrieval_ids, status: a.status, timestamp: a.when.toISOString() }));
    MI.openExport({ title: 'Audit trail', filename: 'audit_records.json', text: `${JSON.stringify(rows, null, 2)}\n`, mime: 'application/json', intro: 'AuditRecord fields as defined in the report: case ID, model and rulebase versions, retrieval IDs, status and timestamp.' });
  };
  MI.actions['export-rules'] = () => MI.openExport({ title: 'Rulebase', filename: 'rulebase.json', text: `${JSON.stringify(D.RULEBASE, null, 2)}\n`, mime: 'application/json' });
  MI.actions['kb-approve'] = (el) => {
    if (state.role !== 'curator') return;
    MI.decideProposal(el.dataset.arg, true);
    MI.render({ keepScroll: true });
    MI.toast(`Approved. Knowledge index is now ${MI.indexVersion()}.`);
  };
  MI.actions['kb-decline'] = (el) => {
    if (state.role !== 'curator') return;
    MI.decideProposal(el.dataset.arg, false);
    MI.render({ keepScroll: true });
    MI.toast('Declined. The knowledge index is unchanged.');
  };
  const draftSettings = {};
  MI.inputs.set = (el) => {
    draftSettings[el.dataset.key] = Number(el.value);
    const lab = MI.$(`#sv-${el.dataset.key}`);
    if (lab) lab.textContent = f2(el.value);
  };
  MI.actions['apply-settings'] = () => {
    const s = { ...state.settings, ...draftSettings };
    const sum = s.alpha + s.beta + s.gamma || 1;
    s.alpha /= sum;
    s.beta /= sum;
    s.gamma /= sum;
    state.settings = s;
    Object.keys(draftSettings).forEach((k) => delete draftSettings[k]);
    MI.recomputeAll();
    if (state.whatif) state.whatif.s = { ...s };
    MI.render({ keepScroll: true });
    MI.toast(`Fusion settings applied to ${Object.keys(state.cases).length} cases.`);
  };
  MI.actions['reset-settings'] = () => {
    state.settings = { ...D.DEFAULT_SETTINGS };
    Object.keys(draftSettings).forEach((k) => delete draftSettings[k]);
    MI.recomputeAll();
    if (state.whatif) state.whatif.s = { ...state.settings };
    MI.render({ keepScroll: true });
    MI.toast('Default fusion settings restored.');
  };

  /* ---------------- Compare ---------------- */
  MI.screens.compare = {
    title: () => 'Black box vs MedIntel',
    subtitle: () => 'The same synthetic case through a single opaque model and through MedIntel',
    render() {
      const ids = MI.readyIds();
      if (!ids.length) return `<div class="card"><div class="empty">${icon('compare')}<b>No case to compare yet</b></div></div>`;
      if (!state.compareId || !ids.includes(state.compareId)) state.compareId = ids[0];
      const rec = state.cases[state.compareId];
      const b = rec.bundle;
      const c = b.case;
      const bbHigh = b.ml.prediction === 'high_risk';
      const auditRow = state.audit.filter((a) => a.case_id === c.case_id).slice(-1)[0];
      const rows = [
        ['Shows the evidence behind the answer', false, `${b.evidence.length} source${b.evidence.length === 1 ? '' : 's'} with IDs and similarity scores`],
        ['Applies deterministic safety rules', false, b.rules.triggered.length ? `${b.rules.triggered.join(', ')} checked and fired` : `${b.rules.checks.length} rules checked`],
        ['Reports disagreement between components', false, `${b.context.conflicts.length} conflict${b.context.conflicts.length === 1 ? '' : 's'} reported`],
        ['Confidence that can be traced', false, `C = ${f2(b.fusion.confidence)} from Equation (4.2)`],
        ['Removes unsupported statements', false, b.context.unsupported_statements.length ? `${b.context.unsupported_statements.length} removed` : 'checked; none found'],
        ['Writes an audit record', false, auditRow ? `written at ${MI.fmtTime(auditRow.when)}` : 'written'],
        ['Leaves the final decision to a clinician', false, 'review required before any action'],
      ];
      return `
      <div class="row"><label for="cmp-case" class="muted" style="font-size:12.5px">Case</label><select class="sel" id="cmp-case" data-ch="cmp-case">${ids.map((id) => `<option value="${id}" ${id === state.compareId ? 'selected' : ''}>${id} · ${MI.demog(state.cases[id].bundle.case)}</option>`).join('')}</select><span class="chip">${icon('flag')}Illustrative scenario</span></div>
      <div class="vs">
        <section class="card bb" id="cmp-bb">
          <div class="card-h"><span class="agico" style="--ag:var(--ink-3)">${icon('robot')}</span><h2>Single-model assistant</h2><span class="sub">one model reads the case and answers</span></div>
          <div class="card-b stack">
            <p class="soft" style="font-size:12.5px">Input: ${MI.demog(c)}, BP ${c.x_s.systolic_bp}/${c.x_s.diastolic_bp} mmHg, cholesterol ${c.x_s.total_cholesterol} mg/dL, glucose ${c.x_s.fasting_glucose} mg/dL${c.x_s.smoker ? ', smoker' : ''}.</p>
            <div class="bubble"><b>Assessment:</b> ${bbHigh ? 'High cardiovascular risk. Further review suggested.' : 'Low cardiovascular risk. Routine follow-up suggested.'}</div>
            <div class="whyq">${icon('question')}Why? <span class="muted">No explanation is given.</span></div>
            <div class="whyq">${icon('retrieval')}Sources? <span class="muted">None shown.</span></div>
            <div class="whyq">${icon('fusion')}Confidence? <span class="muted">Not reported.</span></div>
            <div class="whyq">${icon('audit')}Record of the decision? <span class="muted">None kept.</span></div>
          </div>
        </section>
        <section class="card" id="cmp-mi">
          <div class="card-h">${MI.LOGO_SVG.replace('<svg', '<svg style="width:34px;height:34px;flex:none"')}<h2>MedIntel</h2><span class="sub">separate evidence, explicit fusion, clinician review</span></div>
          <div class="card-b stack">
            <div class="row" style="gap:16px;align-items:center"><div class="stack" style="gap:8px;flex:1;min-width:200px">${MI.decisionBadge(b.fusion.decision, true)}<p style="font-size:14px">${esc(b.recommendation)}</p></div>${MI.gauge(b.fusion, 170)}</div>
            ${b.context.conflicts.map((t) => `<div class="conflict">${icon('warn')}<span>${esc(t)}</span></div>`).join('')}
            <div class="row">${b.rules.triggered.map((r) => `<span class="chip">${icon('rules')}${r}</span>`).join('')}${b.evidence.map((e) => `<span class="srcid">${e.source_id}</span>`).join('')}</div>
            <button class="btn" data-act="open" data-arg="${c.case_id}" style="align-self:flex-start">${icon('report')}Open the full report</button>
          </div>
        </section>
      </div>
      <div class="card"><div class="card-h"><h2>Side by side</h2></div><div class="card-b"><div class="scroll-x"><table class="tbl check-t"><thead><tr><th>Capability</th><th>Single-model assistant</th><th>MedIntel</th></tr></thead><tbody>
        ${rows.map(([cap, bb, mi]) => `<tr><td>${cap}</td><td><span class="no">${icon('xcircle')}No</span></td><td><span class="yes">${icon('okcircle')}Yes</span> <span class="soft" style="font-size:12.5px">${esc(mi)}</span></td></tr>`).join('')}
      </tbody></table></div>
      <p class="muted" style="font-size:12px;margin-top:10px">Illustrative scenario: the single-model answer is the ML component on its own. It shows the design difference described in Chapter 2 of the report; it is not a measured result.</p></div></div>`;
    },
  };
  MI.changes['cmp-case'] = (el) => {
    state.compareId = el.value;
    MI.render({ keepScroll: true });
  };

  /* ---------------- Architecture ---------------- */
  const ARCH = {
    ingest: ['Checks every record against the schema: required fields, types, ranges, units and repeated IDs. Valid rows become one isolated PatientCase each.', 'CSV records, event logs', 'Live: parsing, schema checks and unit checks run in your browser.'],
    ml: ['Estimates risk from the structured features and reports class probabilities with the factors behind them.', 'PatientCase.x_s', 'Live: a logistic model with hand-set demo weights. Stage II trains a model on a public dataset.'],
    rules: ['Applies explicit, versioned thresholds and policies. The most severe fired rule sets the action, and it overrides the learned prediction.', 'PatientCase and rulebase.json', 'Live: the same rule logic as Section 5.3.2 of the report.'],
    retrieval: ['Finds supporting passages in an approved knowledge base and returns them with source IDs and similarity scores, or states that nothing relevant was found.', 'Query built from the case', 'Live: rule-based matching over 10 illustrative summaries. Stage II uses sentence embeddings and FAISS.'],
    context: ['Uses a language model to reconcile the three results, name conflicts and flag statements the evidence does not support.', 'The three analysis results', 'Scripted: text is pre-written for the sample patients; conflicts are computed live.'],
    fusion: ['Combines the evidence with Equation (4.2), C = αM + βR + γK − λD, and applies rule precedence and the decision threshold.', 'MLResult, RuleResult, evidence, ContextSummary', 'Live: the same logic as DecisionFusionEngine.fuse in Section 5.4.2.'],
    audit: ['Builds the DecisionReport for the clinician and appends an AuditRecord with every version needed to reproduce it.', 'PatientCase and FusionResult', 'Live: reports and audit records are generated in your browser.'],
  };

  MI.screens.about = {
    title: () => 'Architecture',
    subtitle: () => 'How this prototype maps to the Project Stage I design',
    render() {
      const sel = state.archSel || 'fusion';
      const a = MI.AGENTS[sel];
      const [what, input, here] = ARCH[sel];
      const logo = (window.MI_ASSETS && window.MI_ASSETS.logo) || '';
      return `
      <div class="card"><div class="card-h"><h2>Components</h2><span class="sub">select one to see its contract</span></div>
        <div class="card-b stack">
          <div class="arch">${MI.STAGE_KEYS.map((k) => `<button class="anode ag-${k}" data-act="arch" data-arg="${k}" aria-pressed="${k === sel}">${MI.agentIco(k)}<span>${MI.AGENTS[k].label}</span><small>${MI.AGENTS[k].fr}</small></button>`).join('')}</div>
          <div class="card" style="box-shadow:none;background:var(--surface-2)"><div class="card-b stack" style="padding-top:16px">
            <div class="row">${MI.agentIco(sel)}<h3 style="font-size:17px">${a.label}</h3><span class="ver">${esc(a.module)}</span></div>
            <p style="font-size:14px;max-width:80ch">${esc(what)}</p>
            <dl class="kv"><dt>Input</dt><dd>${esc(input)}</dd><dt>Output</dt><dd class="mono">${esc(a.out)}</dd><dt>Requirements</dt><dd>${esc(a.fr)}</dd><dt>In this prototype</dt><dd>${esc(here)}</dd></dl>
          </div></div>
        </div></div>
      <div class="realgrid">
        <div class="card"><div class="card-h"><h2>Computed live in your browser</h2></div><div class="card-b"><ul>
          <li>CSV parsing and schema validation, including the TC-01 and TC-02 cases</li><li>Risk estimate from a logistic model with demo weights</li><li>All four rules, with severity and precedence</li><li>Evidence matching, similarity scores, k and the minimum score</li><li>Conflict detection, Equation (4.2), rule precedence and the threshold</li><li>Decision reports, audit records and governed knowledge updates</li></ul></div></div>
        <div class="card"><div class="card-h"><h2>Scripted or simulated</h2></div><div class="card-b"><ul>
          <li>Language-model summaries for the eight sample patients</li><li>Knowledge passages: 10 illustrative summaries, not clinical guidance</li><li>Run timings in the live run</li><li>The single-model answer on the comparison screen</li><li>All patients, notes and events, which are synthetic</li><li>Stage II replaces these parts with the real agents and a public dataset</li></ul></div></div>
      </div>
      <div class="card"><div class="card-b team" style="padding-top:18px">
        ${logo ? `<div class="logo-card"><img src="${logo}" alt="Jawahar Education Society's Institute of Technology, Management and Research, Nashik"></div>` : ''}
        <div class="stack" style="gap:6px">
          <span class="eyebrow">BE Computer Engineering · Project Stage I · AY 2026-27</span>
          <h3 class="display" style="font-size:20px">MedIntel: A Multi-Agent AI Framework for Clinical Decision Support</h3>
          <p class="soft">${['Siddhesh Kishor Suryawanshi', 'Shantanu Avinash Patil', 'Omkar Abaji Musale', 'Tanmay Shriniwas Bagul'].map((n) => `<span class="nb">${n}</span>`).join(' · ')}</p>
          <p class="soft">Guide: Prof. G. P Mohole, Head of Computer Department · Department of Computer Engineering, JESITMR, Nashik</p>
          <p class="muted" style="font-size:12.5px">Built from the Project Stage I report and the research paper submitted to IJAICMES.</p>
        </div>
      </div></div>`;
    },
  };
  MI.actions.arch = (el) => {
    state.archSel = el.dataset.arg;
    MI.render({ keepScroll: true });
  };
  MI.actions['arch-open'] = (el) => {
    state.archSel = el.dataset.arg;
    MI.navigate('about');
  };
})();
