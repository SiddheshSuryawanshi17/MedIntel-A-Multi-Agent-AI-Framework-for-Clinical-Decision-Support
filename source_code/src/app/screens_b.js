(function () {
  'use strict';
  const MI = window.MI;
  const { E, esc, icon, f2, state } = MI;

  /* ---------------- Triage ---------------- */
  const FILTERS = [
    ['all', 'All', () => true],
    ['needs', 'Needs review', (r) => r.bundle.priority.level <= 2 && r.status === 'pending_review'],
    ['pending', 'Pending', (r) => r.status === 'pending_review'],
    ['reviewed', 'Reviewed', (r) => r.status !== 'pending_review'],
  ];

  MI.screens.triage = {
    title: () => 'Triage queue',
    subtitle: () => 'Clinician · cases ordered by priority, then by estimated risk',
    render() {
      const ids = MI.readyIds();
      if (!ids.length) {
        return `<div class="card"><div class="empty">${icon('inbox')}<b>The queue is empty</b><span>${state.run.status === 'running' ? 'Cases appear here as soon as each report is ready.' : 'Upload a batch and send it to the scheduler.'}</span><button class="btn primary" data-act="nav" data-arg="${state.run.status === 'running' ? 'run' : 'upload'}">${icon(state.run.status === 'running' ? 'run' : 'upload')}${state.run.status === 'running' ? 'Watch the live run' : 'Go to upload'}</button></div></div>`;
      }
      const f = FILTERS.find((x) => x[0] === state.triageFilter) || FILTERS[0];
      const shown = ids.filter((id) => f[2](state.cases[id]));
      const counts = Object.fromEntries(FILTERS.map(([k, , fn]) => [k, ids.filter((id) => fn(state.cases[id])).length]));
      const rows = shown.map((id) => {
        const r = state.cases[id];
        const b = r.bundle;
        return `<tr class="click" data-act="open" data-arg="${id}" id="row-${id}">
          <td>${MI.priorityPill(b.priority)}</td>
          <td><b class="mono">${id}</b><div class="muted mono" style="font-size:11.5px">${esc(b.case.patient_id)}</div></td>
          <td>${MI.demog(b.case)}</td>
          <td>${MI.decisionBadge(b.fusion.decision)}</td>
          <td style="min-width:150px">${MI.confMeter(b.fusion)}</td>
          <td>${MI.signals(r)}</td>
          <td>${MI.statusPill(r.status)}</td>
          <td><button class="btn sm ghost" data-act="open" data-arg="${id}" aria-label="Open report ${id}">${icon('right')}</button></td>
        </tr>`;
      }).join('');
      const cards = shown.map((id) => {
        const r = state.cases[id];
        const b = r.bundle;
        return `<button class="tcard" data-act="open" data-arg="${id}">
          <span class="row">${MI.priorityPill(b.priority)}${MI.decisionBadge(b.fusion.decision)}<span class="sp"></span>${MI.statusPill(r.status)}</span>
          <span><b class="mono">${id}</b> · ${MI.demog(b.case)}</span>
          ${MI.confMeter(b.fusion)}
          ${MI.signals(r)}
        </button>`;
      }).join('');
      return `
      <div class="tri-head">
        <div class="seg" role="group" aria-label="Filter cases">${FILTERS.map(([k, label]) => `<button data-act="tri-filter" data-arg="${k}" aria-pressed="${state.triageFilter === k}">${label} <span class="muted">${counts[k]}</span></button>`).join('')}</div>
        <span class="sp"></span>
        <span class="legend"><span><i class="sw" style="--swc:var(--accent)"></i>C at or above threshold</span><span><i class="sw" style="--swc:var(--st-warning)"></i>C below threshold ${f2(state.settings.threshold)}</span></span>
      </div>
      <div class="card tbl-tri"><div class="scroll-x"><table class="tbl" id="triage-table"><thead><tr><th>Priority</th><th>Case</th><th>Patient</th><th>Decision</th><th>Confidence C</th><th>Signals</th><th>Status</th><th><span class="sr">Open</span></th></tr></thead><tbody>${rows || `<tr><td colspan="8"><div class="empty">${icon('filter')}No cases match this filter.</div></td></tr>`}</tbody></table></div></div>
      <div class="cards-m">${cards || `<div class="card"><div class="empty">${icon('filter')}No cases match this filter.</div></div>`}</div>
      <p class="muted" style="font-size:12.5px">Escalate: a rule overrides the model and confidence is below the threshold. Review: uncertain. Priority: high risk with normal confidence. Assess manually: the recommendation is withheld.</p>`;
    },
  };
  MI.actions['tri-filter'] = (el) => {
    state.triageFilter = el.dataset.arg;
    MI.render({ keepScroll: true });
  };

  /* ---------------- Case report ---------------- */
  const OP = { '>=': '≥', '>': '>', '<': '<', '<=': '≤' };
  const ACTION_CLS = { elevate: 'critical', flag: 'serious', suppress: 'neutral' };
  const DIR = { raises: ['raises risk', 'critical'], lowers: ['lowers risk', 'good'], neutral: ['context', 'neutral'] };
  const VITALS = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'temperature_c', 'total_cholesterol', 'fasting_glucose', 'creatinine', 'bmi', 'smoker'];
  const TRACE_NODES = ['ingest', 'ml', 'rules', 'retrieval', 'context', 'fusion', 'audit'];

  const tr = (k, detail) => `data-trace="${k}" data-detail="${esc(detail)}"`;
  const head = (k, title, ver) => `<div class="card-h">${MI.agentIco(k)}<div class="ttl"><h3>${title}</h3><span class="vline">${ver}</span></div></div>`;

  function featureText(c, f) {
    if (f.key === 'sex_m') return `<b>Sex</b> ${c.x_s.sex === 'M' ? 'male' : 'female'}`;
    if (f.key === 'smoker') return `<b>Smoker</b> ${c.x_s.smoker ? 'yes' : 'no'}`;
    return `<b>${esc(f.label)}</b> ${f.value}${f.unit ? ` ${esc(f.unit)}` : ''}${f.imputed ? ' (imputed)' : ''}`;
  }

  function mlCard(b) {
    const ml = b.ml;
    const top = ml.importance.slice(0, 5);
    const maxAbs = Math.max(0.5, ...top.map((f) => Math.abs(f.contribution)));
    const bars = top.map((f) => {
      const w = (Math.abs(f.contribution) / maxAbs) * 30;
      const up = f.contribution >= 0;
      const val = `${up ? '+' : '−'}${Math.abs(f.contribution).toFixed(2)}`;
      return `<span class="lbl" ${tr('ml', `MLResult.importance: ${f.label} adds ${val} to the risk logit`)}>${featureText(b.case, f)}</span>
        <span class="cbar" role="img" aria-label="${esc(f.label)} ${up ? 'raises' : 'lowers'} risk by ${Math.abs(f.contribution).toFixed(2)}"><span class="${up ? 'up' : 'down'}" style="width:${w}%"></span><em style="${up ? `left:calc(50% + ${w}% + 5px)` : `right:calc(50% + ${w}% + 5px)`}">${val}</em></span>`;
    }).join('');
    const imputed = ml.contributions.some((f) => f.imputed);
    const pH = ml.probabilities.high_risk;
    return `<section class="card agent" ${tr('ml', `MLAnalysisAgent · ${ml.model_version}`)}>
      ${head('ml', 'ML Analysis Agent', esc(ml.model_version))}
      <div class="card-b stack">
        <div class="probbar" ${tr('ml', 'MLResult.probabilities from predict_proba')}>
          <div class="pb-top"><span>Probability of high risk</span><b class="num">${MI.pct(pH)}</b></div>
          <div class="meter" style="--mc:var(--a-ml)"><span style="width:${pH * 100}%"></span><i class="tick" style="left:calc(50% - 1px)"></i></div>
          <span class="muted" style="font-size:12px">Prediction: <b style="color:var(--ink)">${MI.DECISION[ml.prediction].label}</b>, class probability ${f2(ml.probabilities[ml.prediction])}. The tick marks 50%.</span>
        </div>
        <div class="eyebrow">Top factors in this prediction</div>
        <div class="contrib">${bars}</div>
        <div class="legend"><span><i class="sw" style="--swc:var(--div-up)"></i>raises risk</span><span><i class="sw" style="--swc:var(--div-down)"></i>lowers risk</span><span class="muted">change in log-odds against a 50-year-old reference profile</span></div>
        ${imputed ? `<div class="note warn">${icon('warn')}<span>Creatinine is missing, so the model used the reference value 0.9 mg/dL. The rule agent flags this separately.</span></div>` : ''}
      </div>
    </section>`;
  }

  function rulesCard(b) {
    const r = b.rules;
    const lines = r.checks.map((k) => {
      const ru = k.rule;
      const cond = ru.operator === 'missing' ? `${ru.field} is missing` : `${ru.field} ${OP[ru.operator]} ${ru.threshold}`;
      const obs = MI.E.fmt(ru.field, k.value);
      return `<div class="ruleline${k.fired ? ' fired' : ''}" ${tr('rules', `${ru.rule_id} · rulebase ${r.rulebase_version} · ${k.fired ? 'fired' : 'passed'}`)}>
        <span class="rid">${ru.rule_id}</span>
        <span><span class="mono" style="font-size:12px">${esc(cond)}</span> <span class="muted">· observed ${esc(obs)}</span></span>
        ${k.fired ? `<span class="badge ${ACTION_CLS[ru.action]}">${icon(ru.action === 'flag' ? 'flag' : ru.action === 'elevate' ? 'up' : 'ban')}${ru.action} · ${ru.severity}</span>` : `<span class="chip">${icon('check')}passed</span>`}
        ${k.fired ? `<span class="rmsg">${esc(ru.message)}</span>` : ''}
      </div>`;
    }).join('');
    const fired = r.checks.filter((k) => k.fired).length;
    return `<section class="card agent" ${tr('rules', `RuleValidationAgent · rulebase ${r.rulebase_version}`)}>
      ${head('rules', 'Rule Validation Agent', `rulebase ${esc(r.rulebase_version)}`)}
      <div class="card-b stack">
        <div class="row" style="font-size:12.5px"><span class="chip">${r.checks.length} rules checked</span><span class="chip">${fired} fired</span>${r.top ? `<span class="soft">Action applied: <b>${r.top.action}</b> from ${r.top.rule_id}</span>` : '<span class="soft">No rule changes the decision.</span>'}</div>
        <div>${lines}</div>
      </div>
    </section>`;
  }

  function retrievalCard(b) {
    const ev = b.evidence;
    const body = ev.length
      ? ev.map((e) => `<div class="ev" ${tr('retrieval', `EvidenceItem ${e.source_id} · index ${MI.indexVersion()} · similarity ${e.score.toFixed(2)}`)}>
          <div class="ev-h"><span class="srcid">${e.source_id}</span><b>${esc(e.title)}</b><span class="badge ${DIR[e.direction][1]}" style="font-size:11px;padding:2px 8px">${DIR[e.direction][0]}</span></div>
          <p>${esc(e.passage)}</p>
          <div class="ev-f"><div class="meter"><span style="width:${e.score * 100}%"></span></div><span class="num">similarity ${e.score.toFixed(2)}</span><span>matched on ${esc(e.matched_on)}</span></div>
        </div>`).join('')
      : `<div class="empty" ${tr('retrieval', 'An empty result means no passage reached the minimum similarity (TC-06)')}>${icon('search')}<b style="color:var(--ink)">No sufficiently relevant evidence</b><span>No passage in the knowledge base reached the minimum similarity of ${f2(state.settings.min_score)}. Retrieval support K is 0, and the report says so instead of inventing a source.</span><span class="tc">TC-06</span></div>`;
    return `<section class="card agent" ${tr('retrieval', `KnowledgeRetrievalAgent · index ${MI.indexVersion()}`)}>
      ${head('retrieval', 'Knowledge Retrieval Agent', `index ${MI.indexVersion()} · top k = ${state.settings.k}`)}
      <div class="card-b">${body}<p class="muted" style="font-size:11.5px;margin-top:10px">Sample knowledge base of illustrative guideline summaries, not clinical guidance.</p></div>
    </section>`;
  }

  function contextCard(b) {
    const cx = b.context;
    const draft = cx.draft.map((d) => (d.unsupported
      ? `<div class="s cut" ${tr('context', 'ContextSummary.unsupported_statements · removed before the report (TC-12)')}>${esc(d.text)}<span class="why">${icon('xcircle')}<span>${esc(d.unsupported)} Removed from the report.</span><span class="tc">TC-12</span></span></div>`
      : `<div class="s" ${tr('context', 'ContextSummary.summary · LLM synthesis of the agent outputs')}>${esc(d.text)}</div>`)).join('');
    const conflicts = cx.conflicts.length
      ? cx.conflicts.map((t) => `<div class="conflict" ${tr('context', 'ContextSummary.conflicts · raises disagreement D in the fusion step')}>${icon('warn')}<span>${esc(t)}</span></div>`).join('')
      : `<div class="note" ${tr('context', 'No conflicts reported, so D = 0')}>${icon('check')}<span>No conflict between the ML, rule and retrieval results.</span></div>`;
    return `<section class="card agent" ${tr('context', 'ClinicalContextAgent · LLM Service')}>
      ${head('context', 'Clinical Context Agent', cx.generated_by === 'scripted' ? 'LLM Service · text scripted for the demo' : 'LLM Service · template text')}
      <div class="card-b stack">
        <div class="eyebrow">Summary draft</div>
        <div class="draft">${draft}</div>
        <div class="eyebrow">Conflicts reported (${cx.conflicts.length})</div>
        <div class="conflicts">${conflicts}</div>
      </div>
    </section>`;
  }

  function fusionCard(b) {
    const f = b.fusion;
    const c = f.component_scores;
    const r = b.rules;
    const label = MI.DECISION[b.ml.prediction].label.toLowerCase();
    const steps = [];
    if (f.override === 'high_risk') steps.push(`Rule <b>${r.top.rule_id}</b> fired with action <b>elevate</b>, so the decision is <b>high risk</b>. A triggered rule always takes precedence over the learned prediction.`);
    else if (f.override === 'withheld') steps.push(`Rule <b>${r.top.rule_id}</b> fired with action <b>suppress</b>, so the recommendation is <b>withheld</b>.`);
    else steps.push(`No rule overrides the model, so the decision follows the ML prediction (<b>${label}</b>) when C reaches the threshold.`);
    if (f.override === 'high_risk' && f.below_threshold) steps.push(`C = ${f2(f.confidence)} is below the threshold ${f2(f.threshold)}, so the case is marked low confidence and escalated for same-day review.`);
    else if (f.override) steps.push(`C = ${f2(f.confidence)} is reported with the decision for the clinician's information.`);
    else if (f.below_threshold) steps.push(`C = ${f2(f.confidence)} is below the threshold ${f2(f.threshold)}, so the decision is reported as <b>uncertain</b>.`);
    else steps.push(`C = ${f2(f.confidence)} is at or above the threshold ${f2(f.threshold)}, so the prediction is reported with normal confidence.`);
    if (c.D > 0) steps.push(`${b.context.conflicts.length} reported conflict${b.context.conflicts.length > 1 ? 's' : ''} set the disagreement D to ${f2(c.D)}, which subtracts ${f2(f.terms.lD)} from C.`);
    if (c.K === 0) steps.push('No sufficiently relevant evidence was retrieved, so retrieval support K is 0.');
    const terms = [
      ['M', 'ml', 'ML support', f2(c.M), 'probability of the predicted class'],
      ['R', 'rules', 'Rule consistency', String(c.R), c.R ? 'no rule contradicts the prediction' : 'a rule overrides the prediction'],
      ['K', 'retrieval', 'Retrieval support', f2(c.K), c.K ? 'mean similarity of retrieved passages' : 'no relevant passage found'],
      ['D', 'context', 'Disagreement', f2(c.D), `${b.context.conflicts.length} conflict${b.context.conflicts.length === 1 ? '' : 's'} ÷ 3, capped at 1`],
    ];
    return `<section class="card" id="fusion-card" ${tr('fusion', 'DecisionFusionEngine.fuse · Equation (4.2)')}>
      <div class="card-h">${MI.agentIco('fusion')}<h3>Decision fusion and confidence</h3><span class="sub">Equation (4.2) of the report</span></div>
      <div class="card-b stack" style="gap:14px">
        <div id="fusion-viz" class="stack" style="gap:14px"><div ${tr('fusion', 'C = αM + βR + γK − λD, clipped at 0')}>${MI.equation(f)}</div>
        ${MI.fusionBar(f, { trace: true })}</div>
        <div class="terms">${terms.map(([s, k, n, v, d]) => `<div class="term" ${tr(k, `${n}: ${d}`)}><span class="sym"><i class="sw" style="--swc:var(--a-${k})"></i>${s}</span><span class="tv num">${v}</span><small>${n}: ${esc(d)}</small></div>`).join('')}</div>
        <div><div class="eyebrow" style="margin-bottom:6px">How the decision was reached</div><ol class="steps-l">${steps.map((s) => `<li>${s}</li>`).join('')}</ol></div>
        <p class="muted" style="font-size:12px">Weights α ${f2(f.weights.alpha)} · β ${f2(f.weights.beta)} · γ ${f2(f.weights.gamma)} · λ ${f2(f.weights.lam)} · threshold ${f2(f.threshold)}. Illustrative values; Stage II tunes them on a validation split only.</p>
      </div>
    </section>`;
  }

  function reviewCard(id, rec) {
    const can = state.role === 'clinician';
    const dis = can ? '' : 'disabled';
    const entries = state.audit.filter((a) => a.case_id === id).sort((a, b) => a.when - b.when);
    return `<section class="card review" id="review-card" ${tr('audit', 'AuditLogger · every review is appended to the audit trail')}>
      <div class="card-h">${MI.agentIco('audit')}<h3>Clinician review</h3>${MI.statusPill(rec.status)}<span class="sub">the clinician keeps the final decision</span></div>
      <div class="card-b stack">
        ${MI.roleGate('clinician', 'Reviewing a recommendation')}
        <label for="review-notes" class="eyebrow">Notes for the record</label>
        <textarea id="review-notes" placeholder="For example: repeat BP 178/104 at 08:40, started treatment." ${dis}>${esc(rec.notes)}</textarea>
        <div class="row">
          <button class="btn good" id="btn-accept" data-act="review" data-arg="accepted" ${dis}>${icon('okcircle')}Accept</button>
          <button class="btn" id="btn-annotate" data-act="review" data-arg="annotated" ${dis}>${icon('annotate')}Annotate</button>
          <button class="btn danger" id="btn-reject" data-act="review" data-arg="rejected" ${dis}>${icon('xcircle')}Reject</button>
          <span class="muted" style="font-size:12px">Accepted outcomes are proposed to the Knowledge Curator; nothing is re-indexed without approval.</span>
        </div>
        <div class="eyebrow">This case in the audit trail</div>
        <div class="alog" id="case-audit">${entries.map((a) => `<div><time>${MI.fmtTime(a.when)}</time><span>${MI.statusPill(a.status)} <span class="soft">${esc(a.event)} · ${esc(a.actor)}</span></span></div>`).join('')}</div>
      </div>
    </section>`;
  }

  function traceBar() {
    return `<div class="tracebar" id="tracebar"><div class="card">
      <span class="eyebrow">Trace sources</span>
      <div class="tnodes">${TRACE_NODES.map((k) => `<span class="tnode ag-${k}" data-node="${k}">${MI.agentIco(k)}${MI.AGENTS[k].short}</span>`).join('')}</div>
      <div class="tdetail" id="tdetail">Point at any part of the report to see which component produced it.</div>
    </div></div>`;
  }

  MI.screens.report = {
    title: () => (state.selected ? `Case report · ${state.selected}` : 'Case report'),
    subtitle: () => 'DecisionReport for clinician review (FR8)',
    render() {
      const ids = MI.readyIds();
      const id = state.selected && state.cases[state.selected] && state.cases[state.selected].ready ? state.selected : ids[0];
      if (!id) return `<div class="card"><div class="empty">${icon('report')}<b>No report yet</b><span>Reports appear when the live run completes.</span><button class="btn primary" data-act="nav" data-arg="triage">${icon('triage')}Go to the triage queue</button></div></div>`;
      state.selected = id;
      const rec = state.cases[id];
      const b = rec.bundle;
      const c = b.case;
      const pos = ids.indexOf(id);
      const prev = ids[(pos - 1 + ids.length) % ids.length];
      const next = ids[(pos + 1) % ids.length];
      const vit = MI.VITALS_HTML(c);
      return `
      ${traceBar()}
      <section class="card rep-head" id="rep-head">
        <div ${tr('ingest', `PatientCase ${c.case_id} built from ${c.source}`)}>
          <div class="rep-top"><span class="eyebrow">Case report · ${pos + 1} of ${ids.length}</span><span class="casenav"><button class="btn sm ghost icon" data-act="open" data-arg="${prev}" aria-label="Previous case ${prev}">${icon('left')}</button><button class="btn sm ghost icon" data-act="open" data-arg="${next}" aria-label="Next case ${next}">${icon('right')}</button></span></div>
          <h2>${id} <span class="muted" style="font-weight:400;font-size:.7em">${esc(c.patient_id)}</span></h2>
          <p class="meta">${MI.demog(c)} · ${esc(c.source)} · analyzed ${rec.analyzedAt ? MI.fmtTime(rec.analyzedAt) : '-'}</p>
          <div class="row" style="margin-top:10px">${MI.statusPill(rec.status)}${MI.priorityPill(b.priority)}</div>
        </div>
        <div class="rep-dec">
          <div ${tr('fusion', `FusionResult.decision · ${b.fusion.override ? `rule override (${b.rules.top.rule_id})` : 'follows the ML prediction'}`)} id="rep-decision">${MI.decisionBadge(b.fusion.decision, true)}</div>
          <p ${tr('audit', 'ExplanationGenerator · DecisionReport.recommendation')}>${esc(b.recommendation)}</p>
          <div class="rep-tools">
            <button class="btn sm ${state.trace ? 'on' : ''}" id="trace-btn" data-act="trace" aria-pressed="${state.trace}">${icon('trace')}Trace sources</button>
            <button class="btn sm" data-act="export-report">${icon('download')}Export report</button>
            <button class="btn sm ghost" data-act="compare-case" data-arg="${id}">${icon('compare')}Compare</button>
          </div>
        </div>
        <div class="rep-gauge" ${tr('fusion', 'FusionResult.confidence against the threshold')}>${MI.gauge(b.fusion, 210)}</div>
      </section>
      <div class="rep-grid">
        <section class="card" ${tr('ingest', 'PatientCase · validated by SchemaValidator')}>
          <div class="card-h">${MI.agentIco('ingest')}<h3>Patient snapshot</h3><span class="sub">de-identified</span></div>
          <div class="card-b stack">
            <div class="vitals">${vit}</div>
            <div class="eyebrow">Clinical notes</div>
            <p class="quote" ${tr('ingest', 'PatientCase.x_t · text context')}>${esc(c.x_t || 'No notes.')}</p>
            ${c.x_l.length ? `<div class="eyebrow">Event log</div><div class="timeline" ${tr('ingest', 'PatientCase.x_l · events joined by patient identifier')}>${c.x_l.map((e) => `<div class="tl"><time>${esc(e.time)}</time><span>${esc(e.text)}</span></div>`).join('')}</div>` : ''}
          </div>
        </section>
        <div class="agents">${mlCard(b)}${rulesCard(b)}${retrievalCard(b)}${contextCard(b)}</div>
      </div>
      ${fusionCard(b)}
      ${reviewCard(id, rec)}`;
    },
    mount() {
      const view = MI.$('#view');
      if (this._over) view.removeEventListener('mouseover', this._over);
      if (this._focus) view.removeEventListener('focusin', this._focus);
      let hot = null;
      const hit = (target) => {
        if (!state.trace) return;
        const el = target.closest && target.closest('[data-trace]');
        if (!el || el === hot) return;
        if (hot) hot.classList.remove('trace-hot');
        hot = el;
        el.classList.add('trace-hot');
        const k = el.dataset.trace;
        MI.$$('.tnode').forEach((n) => n.classList.toggle('hot', n.dataset.node === k));
        const d = MI.$('#tdetail');
        if (d) d.innerHTML = `<b>${MI.AGENTS[k].label}</b> · ${esc(el.dataset.detail || MI.AGENTS[k].out)}`;
      };
      this._over = (e) => hit(e.target);
      this._focus = (e) => hit(e.target);
      view.addEventListener('mouseover', this._over);
      view.addEventListener('focusin', this._focus);
      this.highlight = (el) => hit(el);
    },
    unmount() {
      document.body.classList.remove('trace-on');
    },
  };

  MI.VITALS_HTML = (c) => VITALS.map((k) => {
    const f = MI.E.field(k);
    const v = c.x_s[k];
    const ab = MI.E.abnormal(k, v);
    const miss = v === null || v === undefined;
    const cls = miss ? 'miss' : ab === 'high' ? 'hi' : ab === 'low' ? 'lo' : '';
    const flag = miss ? '<span class="flagtxt">MISSING</span>' : ab === 'high' ? '<span class="flagtxt">HIGH</span>' : ab === 'low' ? '<span class="flagtxt">LOW</span>' : '';
    return `<div class="vital ${cls}" ${tr('ingest', `PatientCase.x_s.${k}`)}><span class="vn">${f.label}</span><span class="vv">${esc(MI.E.fmt(k, v))}${flag}</span>${f.refText ? `<span class="vr">reference ${esc(f.refText)}${f.unit ? ` ${esc(f.unit)}` : ''}</span>` : ''}</div>`;
  }).join('');

  MI.actions.trace = () => {
    state.trace = !state.trace;
    document.body.classList.toggle('trace-on', state.trace);
    const b = MI.$('#trace-btn');
    if (b) {
      b.classList.toggle('on', state.trace);
      b.setAttribute('aria-pressed', String(state.trace));
    }
    if (!state.trace) {
      MI.$$('.trace-hot').forEach((x) => x.classList.remove('trace-hot'));
      MI.$$('.tnode').forEach((n) => n.classList.remove('hot'));
    } else MI.toast('Trace on: point at any part of the report');
  };
  MI.actions['export-report'] = () => {
    const rec = state.cases[state.selected];
    if (!rec) return;
    const rep = E.buildReport(rec.bundle, rec.status, rec.notes, MI.versionsObj());
    MI.openExport({ title: `DecisionReport · ${state.selected}`, filename: `${state.selected}_decision_report.json`, text: `${JSON.stringify(rep, null, 2)}\n`, mime: 'application/json', intro: 'The fields follow the JSON Schema in Section 4.4.2 of the report.' });
  };
  MI.actions.review = (el) => {
    if (state.role !== 'clinician') {
      MI.toast('Switch to Clinician to review.', 'warn');
      return;
    }
    const id = state.selected;
    const notesEl = MI.$('#review-notes');
    const notes = notesEl ? notesEl.value.trim() : '';
    if (el.dataset.arg === 'annotated' && !notes) {
      MI.toast('Add a note before annotating.', 'warn');
      if (notesEl) notesEl.focus();
      return;
    }
    const proposed = MI.review(id, el.dataset.arg, notes);
    MI.render({ keepScroll: true });
    MI.toast(proposed ? 'Review recorded. Outcome sent to the Knowledge Curator for approval.' : 'Review recorded in the audit trail.');
  };
  MI.actions['compare-case'] = (el) => {
    state.compareId = el.dataset.arg;
    MI.navigate('compare');
  };
})();
