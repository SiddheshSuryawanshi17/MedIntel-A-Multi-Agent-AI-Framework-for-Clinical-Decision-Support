(function () {
  'use strict';
  const MI = window.MI;
  const { D, esc, icon, f2, state } = MI;

  const PULSE = `<svg class="pulse-bg" viewBox="0 0 300 120" aria-hidden="true"><path d="M0 70h70l14-34 24 72 18-52 12 20h162" fill="none" stroke="currentColor" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

  function flowStrip() {
    const step = (k) => `<button class="fstep ag-${k}" data-act="arch-open" data-arg="${k}">${MI.agentIco(k, true)}${MI.AGENTS[k].short}</button>`;
    const arrow = `<span class="farrow">${icon('arrow')}</span>`;
    return `<div class="flow">${step('ingest')}${arrow}<span class="fpar">${step('ml')}${step('rules')}${step('retrieval')}</span>${arrow}${step('context')}${arrow}${step('fusion')}${arrow}${step('audit')}</div>`;
  }
  MI.flowStrip = flowStrip;

  function signals(rec) {
    const b = rec.bundle;
    const out = [];
    for (const k of b.rules.checks.filter((x) => x.fired)) out.push(`<span class="chip">${icon('rules')}${k.rule.rule_id} ${k.rule.action}</span>`);
    if (!b.evidence.length) out.push(`<span class="chip">${icon('retrieval')}No evidence</span>`);
    if (b.context.conflicts.length) out.push(`<span class="chip">${icon('warn')}${b.context.conflicts.length} conflict${b.context.conflicts.length > 1 ? 's' : ''}</span>`);
    if (b.context.unsupported_statements.length) out.push(`<span class="chip">${icon('context')}Statement removed</span>`);
    return out.length ? `<span class="signals">${out.join('')}</span>` : '<span class="muted">None</span>';
  }
  MI.signals = signals;

  const demog = (c) => `${c.x_s.age}-year-old ${c.x_s.sex === 'F' ? 'female' : 'male'}`;
  MI.demog = demog;

  /* ---------------- Overview ---------------- */
  MI.screens.overview = {
    title: () => 'Overview',
    subtitle: () => `${MI.fmtDate(new Date())} · today's sample batch of synthetic patients`,
    render() {
      const ids = MI.readyIds();
      const recs = ids.map((id) => state.cases[id]);
      const needs = recs.filter((r) => r.bundle.priority.level <= 2 && r.status === 'pending_review').length;
      const avg = recs.length ? recs.reduce((a, r) => a + r.bundle.fusion.confidence, 0) / recs.length : 0;
      const rej = state.batch ? state.batch.rejected.length : 0;
      const top = ids.slice(0, 4);
      return `
      <section class="hero fade-in">
        <div class="card hero-main">
          ${PULSE}
          <span class="eyebrow">MedIntel · multi-agent clinical decision support</span>
          <h2>Every recommendation traceable to its evidence, with the final decision left to the clinician.</h2>
          <p class="lead">This console shows how MedIntel is designed to work once Project Stage II builds it. Learned, rule-based and retrieved evidence are produced separately, combined with an explicit confidence score and reviewed by a clinician. Every patient on these screens is synthetic.</p>
          <div class="row">
            <button class="btn primary" data-act="nav" data-arg="triage">${icon('triage')}Open the triage queue</button>
            <button class="btn" data-act="nav" data-arg="upload">${icon('upload')}Upload a batch</button>
            <button class="btn ghost" data-act="nav" data-arg="about">${icon('about')}How it works</button>
          </div>
        </div>
        <div class="card demo-card">
          <span class="eyebrow" style="color:var(--on-brand-2)">For the presentation</span>
          <h3>Guided demo</h3>
          <p>A step-by-step walkthrough of the concept, about seven minutes. Each step performs the action on screen and shows what to say.</p>
          <button class="btn primary lg" data-act="tour-start">${icon('play')}Start guided demo</button>
          <div class="keys"><kbd>←</kbd><kbd>→</kbd> move between steps · <kbd>Esc</kbd> exits</div>
          <ol class="beats"><li>Upload, validate and run a batch</li><li>Why case C-1042 is escalated</li><li>Trace every line to its source</li><li>What-if: watch rule R-001 fire</li><li>Clinician review and governed updates</li></ol>
        </div>
      </section>
      <section class="grid g4">
        <div class="card stat"><span class="sl">Cases analyzed</span><span class="sv">${ids.length}</span><span class="sc">${state.batch ? `${esc(state.batch.file)}, ${rej} row${rej === 1 ? '' : 's'} rejected at validation` : 'no batch loaded'}</span></div>
        <div class="card stat"><span class="sl">Need review now</span><span class="sv">${needs}</span><span class="sc">escalated or uncertain, not yet reviewed</span></div>
        <div class="card stat"><span class="sl">Average confidence C</span><span class="sv">${recs.length ? f2(avg) : '-'}</span><span class="sc">decision threshold ${f2(state.settings.threshold)}</span></div>
        <div class="card stat"><span class="sl">Audit records</span><span class="sv">${state.audit.length}</span><span class="sc">append-only · knowledge index ${MI.indexVersion()}</span></div>
      </section>
      <section class="grid g2">
        <div class="card">
          <div class="card-h"><h2>Needs attention</h2><span class="sub">highest priority first</span><div class="end"><button class="btn sm ghost" data-act="nav" data-arg="triage">All cases${icon('right')}</button></div></div>
          <div class="attn" style="margin-top:12px">
            ${top.length ? top.map((id) => {
              const r = state.cases[id];
              const b = r.bundle;
              return `<button data-act="open" data-arg="${id}">${MI.priorityPill(b.priority)}<span class="t1"><span class="mono">${id}</span> · ${demog(b.case)}</span>${MI.decisionBadge(b.fusion.decision)}<span class="t2">${esc(b.recommendation)}</span></button>`;
            }).join('') : `<div class="empty">${icon('inbox')}<span>No analyzed cases yet. Load a batch to fill the queue.</span></div>`}
          </div>
        </div>
        <div class="card">
          <div class="card-h"><h2>How a case flows</h2><span class="sub">select a step to learn more</span></div>
          <div class="card-b stack">
            ${flowStrip()}
            <p class="soft" style="font-size:13px">Records are validated first. The ML, rule and retrieval agents then work in parallel on an isolated copy of the case, the context agent reconciles their outputs, and the fusion engine scores the decision before a report and an audit record are written.</p>
            <div class="note">${icon('about')}<span>In this prototype the rules, the scoring model, evidence matching and the confidence equation run live in your browser. Language-model text is pre-written for the sample patients, and all timings are simulated.</span></div>
          </div>
        </div>
      </section>`;
    },
  };

  /* ---------------- Upload ---------------- */
  const SHOW = ['age', 'sex', 'systolic_bp', 'diastolic_bp', 'temperature_c', 'total_cholesterol', 'fasting_glucose', 'creatinine', 'smoker'];
  const SHORT = { age: 'Age', sex: 'Sex', systolic_bp: 'SBP', diastolic_bp: 'DBP', temperature_c: 'Temp °C', total_cholesterol: 'Chol', fasting_glucose: 'Glucose', creatinine: 'Creat.', smoker: 'Smoker' };

  function loadText(text, file) {
    MI.loadBatch(text, file);
    MI.render({ keepScroll: true });
    const b = state.batch;
    MI.toast(b.parsed.missingColumns.length ? `${file}: required columns are missing` : `${b.parsed.records.length} records read · ${b.valid.length} valid · ${b.rejected.length} rejected`, b.rejected.length || b.parsed.missingColumns.length ? 'warn' : 'ok');
  }
  MI.loadText = loadText;

  function readFile(file) {
    if (!file) return;
    if (!/\.csv$/i.test(file.name) && file.type && !/csv|text/.test(file.type)) {
      MI.toast('Choose a .csv file with the columns listed under Expected columns.', 'warn');
      return;
    }
    const fr = new FileReader();
    fr.onload = () => loadText(String(fr.result), file.name);
    fr.onerror = () => MI.toast('The file could not be read.', 'warn');
    fr.readAsText(file);
  }

  MI.screens.upload = {
    title: () => 'Upload and validate',
    subtitle: () => 'Data Administrator · every record is checked before any agent runs (FR1)',
    render() {
      const can = state.role === 'data_admin';
      const b = state.batch;
      const dis = can ? '' : 'disabled';
      let results = '';
      if (b) {
        const recs = b.parsed.records;
        const rejByRow = new Map(b.rejected.map((r) => [r.row, r]));
        const missing = b.parsed.missingColumns;
        const rows = recs.map((raw) => {
          const rj = rejByRow.get(raw.__row);
          const badFields = new Set(rj ? rj.problems.map((p) => p.field) : []);
          const cells = SHOW.map((k) => `<td class="num${k === 'temperature_c' ? '' : ' opt'} ${badFields.has(k) ? 'cell-bad' : ''}">${esc(raw[k] === '' || raw[k] === undefined ? '-' : raw[k])}</td>`).join('');
          const status = rj
            ? `<div class="prob">${rj.problems.map((p) => `<span>${icon('xcircle')} <b class="mono">${esc(p.field)}</b>: ${esc(p.message)}${p.tc ? `<span class="tc">${p.tc}</span>` : ''}</span>`).join('')}</div>`
            : `<span class="badge good">${icon('check')}Valid</span>`;
          return `<tr class="${rj ? 'bad' : ''}"><td class="num muted">${raw.__row}</td><td class="mono ${badFields.has('patient_id') ? 'cell-bad' : ''}">${esc(raw.patient_id || '(empty)')}</td>${cells}<td>${status}</td></tr>`;
        }).join('');
        results = `
        <div class="card" id="validation">
          <div class="card-h"><h2>Validation results</h2><span class="chip mono">${icon('csv')}${esc(b.file)}</span><span class="sub">read at ${MI.fmtHM(b.loadedAt)}</span></div>
          <div class="card-b stack">
            ${missing.length ? `<div class="note warn">${icon('warn')}<span>This file is missing required columns: <b class="mono">${missing.map(esc).join(', ')}</b>. Add them and upload again.</span></div>` : ''}
            <div class="vsum"><span class="chip">${icon('list')}${recs.length} records read</span><span class="badge good">${icon('check')}${b.valid.length} valid</span><span class="badge critical">${icon('xcircle')}${b.rejected.length} rejected</span><span class="chip">${icon('lock')}Rejected rows are held for correction; nothing is converted silently</span></div>
            <div class="scroll-x"><table class="tbl" id="val-table"><thead><tr><th>Row</th><th>Patient ID</th>${SHOW.map((k) => `<th${k === 'temperature_c' ? '' : ' class="opt"'}>${SHORT[k]}</th>`).join('')}<th>Result</th></tr></thead><tbody>${rows}</tbody></table></div>
            <div class="row">
              ${b.sent ? `<span class="badge good">${icon('okcircle')}Sent to the scheduler</span><button class="btn" data-act="nav" data-arg="run">${icon('run')}Open live run</button>` : `<button class="btn primary lg" id="send-btn" data-act="send-batch" ${dis || (b.valid.length ? '' : 'disabled')}>${icon('run')}Send ${b.valid.length} case${b.valid.length === 1 ? '' : 's'} to the scheduler</button>`}
              <span class="muted" style="font-size:12.5px">Each valid record becomes one isolated PatientCase (FR2).</span>
            </div>
          </div>
        </div>`;
      }
      return `
      ${MI.roleGate('data_admin', 'Uploading patient records')}
      <div class="card">
        <div class="card-h"><h2>Patient records</h2><span class="sub">CSV export from the hospital system, already de-identified</span></div>
        <div class="card-b stack">
          <div class="drop" id="drop">
            <span class="big">${icon('upload')}</span>
            <b style="font-size:15px">Drop a CSV file here</b>
            <span class="muted">or</span>
            <div class="row" style="justify-content:center">
              <label class="btn ${can ? '' : 'disabled'}" for="file-in" ${can ? '' : 'aria-disabled="true" style="opacity:.5;pointer-events:none"'}>${icon('csv')}Choose a file</label>
              <input id="file-in" type="file" accept=".csv,text/csv" class="sr" ${dis}>
              <button class="btn primary" id="sample-btn" data-act="use-sample" ${dis}>${icon('inbox')}Use the sample morning batch</button>
            </div>
            <span class="muted" style="font-size:12px">Files are read in this browser only. Nothing is uploaded anywhere.</span>
          </div>
          <div class="row">
            <span class="eyebrow">Expected columns</span>
            <button class="btn sm ghost" data-act="sample-csv">${icon('download')}Get the sample CSV</button>
          </div>
          <div class="schema">${D.SCHEMA.map((f) => `<span class="chip"><b class="mono">${f.key}</b>${f.unit ? ` <span class="muted">${esc(f.unit)}</span>` : ''}${f.required ? '' : ' <span class="muted">optional</span>'}</span>`).join('')}</div>
        </div>
      </div>
      ${results}`;
    },
    mount() {
      const drop = MI.$('#drop');
      const input = MI.$('#file-in');
      if (input) input.addEventListener('change', () => readFile(input.files && input.files[0]));
      if (!drop) return;
      ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => {
        e.preventDefault();
        if (state.role === 'data_admin') drop.classList.add('over');
      }));
      ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove('over')));
      drop.addEventListener('drop', (e) => {
        e.preventDefault();
        if (state.role !== 'data_admin') {
          MI.toast('Switch to Data Administrator to upload records.', 'warn');
          return;
        }
        readFile(e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]);
      });
    },
  };
  MI.actions['use-sample'] = () => loadText(D.SAMPLE_CSV, 'morning_batch.csv');
  MI.actions['sample-csv'] = () => MI.openExport({ title: 'Sample morning batch', filename: 'morning_batch.csv', text: `${D.SAMPLE_CSV}\n`, mime: 'text/csv', intro: 'Ten synthetic, de-identified records. Rows 10 and 11 are deliberately invalid so the validator has something to catch (test cases TC-01 and TC-02).' });
  MI.actions['send-batch'] = () => {
    if (!state.batch || !state.batch.valid.length) return;
    MI.prepareCases();
    MI.navigate('run');
    MI.startRun();
  };

  /* ---------------- Live run ---------------- */
  const STAGE_LABEL = { ingest: 'Validate', ml: 'ML', rules: 'Rules', retrieval: 'Retrieval', context: 'Context (LLM)', fusion: 'Fusion', audit: 'Report' };

  function stageChip(lane, k) {
    return `<span class="stg ag-${k}" data-lane="${lane.id}" data-stage="${k}">${MI.agentIco(k)}<span>${STAGE_LABEL[k]}</span><span class="t"></span></span>`;
  }

  MI.screens.run = {
    title: () => 'Live run',
    subtitle: () => 'CaseScheduler · each case runs in its own isolated context (FR2, NFR2)',
    render() {
      const run = state.run;
      if (!run.lanes.length) {
        return `<div class="card"><div class="empty">${icon('run')}<b>No run yet</b><span>Upload and validate a batch, then send it to the scheduler.</span><button class="btn primary" data-act="nav" data-arg="upload">${icon('upload')}Go to upload</button></div></div>`;
      }
      const sep = '<span class="sep">' + icon('right') + '</span>';
      const lanes = run.lanes.map((lane) => {
        const b = state.cases[lane.id] ? state.cases[lane.id].bundle : null;
        return `<div class="lane" id="lane-${lane.id}">
          <div class="lane-id"><b>${lane.id}</b><small>${icon('lock')}isolated · worker ${lane.worker + 1}</small><small>${b ? `${esc(b.case.patient_id)} · ${MI.demog(b.case)}` : ''}</small></div>
          <div class="stages">${stageChip(lane, 'ingest')}${sep}<span class="par">${stageChip(lane, 'ml')}${stageChip(lane, 'rules')}${stageChip(lane, 'retrieval')}</span>${sep}${stageChip(lane, 'context')}${sep}${stageChip(lane, 'fusion')}${sep}${stageChip(lane, 'audit')}</div>
          <div class="lane-out" id="out-${lane.id}"></div>
        </div>`;
      }).join('');
      return `
      <div class="card">
        <div class="card-b stack" style="padding-top:16px">
          <div class="runbar">
            <span id="run-status"></span>
            <div class="meter" style="--mc:var(--accent)"><span id="run-prog" style="width:0%"></span></div>
            <span class="num soft" id="run-clock"></span>
          </div>
          <div class="row">
            <span class="muted" style="font-size:12.5px">max_workers</span>
            <div class="seg" role="group" aria-label="Maximum parallel cases">${[1, 2, 4, 8].map((n) => `<button data-act="workers" data-arg="${n}" aria-pressed="${run.maxWorkers === n}">${n}</button>`).join('')}</div>
            <span class="sp"></span>
            <button class="btn" data-act="rerun">${icon('reset')}Run again</button>
            <button class="btn primary" data-act="nav" data-arg="triage" id="to-triage">${icon('triage')}Open the triage queue</button>
          </div>
          <div class="note">${icon('clock')}<span>Timings are simulated for the demo. The Stage II target is one case analyzed end to end within 30 seconds on the recommended hardware (NFR4).</span></div>
        </div>
      </div>
      <div class="lanes" id="lanes">${lanes}</div>
      <div class="legend" style="justify-content:center">${['ingest', 'ml', 'rules', 'retrieval', 'context', 'fusion', 'audit'].map((k) => `<span><i class="sw" style="--swc:var(--a-${k})"></i>${MI.AGENTS[k].label}</span>`).join('')}</div>`;
    },
    mount() {
      this.tick();
    },
    tick() {
      const run = state.run;
      const clock = run.clock;
      let done = 0;
      for (const lane of run.lanes) {
        for (const k of MI.STAGE_KEYS) {
          const el = document.querySelector(`.stg[data-lane="${lane.id}"][data-stage="${k}"]`);
          if (!el) continue;
          const [s, e] = lane.st[k];
          const st = clock >= e ? 'done' : clock >= s ? 'running' : 'wait';
          if (el.dataset.state !== st) {
            el.dataset.state = st;
            el.classList.toggle('running', st === 'running');
            el.classList.toggle('done', st === 'done');
            el.querySelector('.t').textContent = st === 'done' ? `${lane.dur[k].toFixed(2)} s` : st === 'running' ? '…' : '';
          }
        }
        const laneEl = document.getElementById(`lane-${lane.id}`);
        const out = document.getElementById(`out-${lane.id}`);
        const finished = lane.done || clock >= lane.end;
        if (finished) done += 1;
        if (laneEl) laneEl.classList.toggle('queued', clock < lane.start);
        if (out) {
          const key = finished ? 'done' : clock >= lane.start ? 'run' : 'queue';
          if (out.dataset.k !== key) {
            out.dataset.k = key;
            const rec = state.cases[lane.id];
            if (key === 'done' && rec) out.innerHTML = `${MI.decisionBadge(rec.bundle.fusion.decision)}<small class="num">C ${f2(rec.bundle.fusion.confidence)} · ${(lane.end - lane.start).toFixed(1)} s</small>`;
            else if (key === 'run') out.innerHTML = `<small>${icon('bolt')} analyzing</small>`;
            else out.innerHTML = `<small>${icon('clock')} queued (max_workers ${run.maxWorkers})</small>`;
          }
        }
      }
      const total = run.lanes.length;
      const st = document.getElementById('run-status');
      if (st) st.innerHTML = run.status === 'done' ? `<span class="badge good">${icon('okcircle')}Complete · ${total} reports ready</span>` : `<span class="badge serious">${icon('bolt')}Running · ${done} of ${total} done</span>`;
      const prog = document.getElementById('run-prog');
      if (prog) prog.style.width = `${total ? (done / total) * 100 : 0}%`;
      const ck = document.getElementById('run-clock');
      if (ck) ck.textContent = `${Math.min(clock, run.total).toFixed(1)} s of ${run.total.toFixed(1)} s simulated`;
    },
  };
  MI.actions.workers = (el) => {
    state.run.maxWorkers = Number(el.dataset.arg);
    MI.actions.rerun();
  };
  MI.actions.rerun = () => {
    if (!state.batch || !state.batch.valid.length) return;
    MI.prepareCases();
    MI.startRun();
    MI.render({ keepScroll: true });
  };
})();
