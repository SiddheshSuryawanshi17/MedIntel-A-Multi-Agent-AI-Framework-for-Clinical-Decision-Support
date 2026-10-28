(function () {
  'use strict';
  const MI = window.MI;
  const { D, esc, icon, state } = MI;

  const HERO = 'C-1042';
  const NOTE = 'Repeat BP 182/110 mmHg at 08:40. Treatment started; review tomorrow.';

  function resetDemo() {
    if (MI.finishRun && state.run.status === 'running') MI.finishRun();
    Object.assign(state, {
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
    });
    document.body.classList.remove('trace-on');
  }
  MI.resetDemo = resetDemo;

  function ensureBatch() {
    if (!state.batch) MI.loadBatch(D.SAMPLE_CSV, 'morning_batch.csv');
  }
  function ensureRun(finish) {
    ensureBatch();
    if (!state.batch.sent) {
      MI.prepareCases();
      MI.startRun();
    }
    if (finish && state.run.status === 'running') MI.finishRun();
  }
  function trace(on) {
    state.trace = on;
    document.body.classList.toggle('trace-on', on && state.screen === 'report');
  }

  const STEPS = [
    {
      center: true,
      title: 'MedIntel clinical console',
      html: () => `${window.MI_ASSETS && window.MI_ASSETS.logo ? `<img src="${window.MI_ASSETS.logo}" alt="JESITMR Nashik" style="width:100%;max-width:380px;background:#fff;border-radius:8px;padding:6px 8px">` : ''}<p>A concept prototype of how MedIntel is designed to work once Project Stage II builds it. Every patient is synthetic, and the steps below perform each action for you.</p><p class="say">MedIntel separates learned, rule-based and retrieved evidence, combines it with an explicit confidence score, and always leaves the decision to a clinician.</p>`,
      before: () => {
        resetDemo();
        MI.setRole('data_admin', true);
        MI.navigate('upload', { noFocus: true });
      },
    },
    {
      screen: 'upload', role: 'data_admin', target: ['#sample-btn', '#drop'],
      title: 'A morning batch arrives',
      text: 'The Data Administrator uploads a de-identified CSV export. Here we use the sample batch of ten synthetic records.',
      say: 'Records are only read in the browser; nothing leaves the machine.',
      before: () => {
        resetDemo();
        MI.setRole('data_admin', true);
      },
    },
    {
      screen: 'upload', role: 'data_admin', target: ['#val-table tr.bad', '#validation'],
      title: 'Validation catches bad records first',
      text: 'Ten records read, eight valid. One row has no patient ID (test case TC-01) and one has a temperature entered in °F (TC-02). Nothing is converted silently; both rows are held for correction.',
      say: 'This is functional requirement FR1: no agent ever sees an invalid record.',
      before: () => {
        resetDemo();
        MI.setRole('data_admin', true);
        MI.loadBatch(D.SAMPLE_CSV, 'morning_batch.csv');
      },
    },
    {
      screen: 'run', target: ['#lanes > .lane:nth-child(-n+4)', '#lanes'],
      title: 'Eight cases run in parallel, each in isolation',
      text: 'Four cases run at a time. Inside each case, the ML, rule and retrieval agents work in parallel; the context agent waits for all three, then fusion scores the decision.',
      say: 'Each case has its own isolated context, so nothing from one patient can reach another patient’s report (NFR2).',
      before: () => {
        ensureBatch();
        if (!state.batch.sent) {
          MI.prepareCases();
          MI.startRun();
        }
      },
    },
    {
      screen: 'triage', role: 'clinician', target: [`#row-${HERO}`, '.cards-m .tcard:first-child'],
      title: 'The triage queue puts risk first',
      text: 'Case C-1042 is at the top and marked Escalate: a safety rule overrides the model, and the confidence is low.',
      say: 'The queue is ordered by priority, so the clinician sees the cases that need attention first.',
      before: () => ensureRun(true),
    },
    {
      screen: 'report', caseId: HERO, role: 'clinician', target: ['#rep-head'],
      title: 'A report built for review',
      text: 'High risk, with a confidence of 0.25, well below the 0.60 threshold. The recommendation says what to do next: same-day review and a repeat blood pressure reading.',
      before: () => {
        ensureRun(true);
        trace(false);
      },
    },
    {
      screen: 'report', caseId: HERO, target: ['.agents > .agent:nth-child(-n+2)', '.agents'],
      title: 'Why: the model and the rules disagree',
      text: 'The ML model sees a 34-year-old non-smoker with normal labs and estimates a 17% risk. Rule R-001 fires on a systolic reading of 186 mmHg and elevates the case. The retrieved evidence on severely raised blood pressure sides with the rule.',
      say: 'Each agent does one job and returns a typed result, so we can see exactly where they disagree.',
      before: () => ensureRun(true),
    },
    {
      screen: 'report', caseId: HERO, target: ['#fusion-viz', '#fusion-card'],
      title: 'The confidence equation, in the open',
      text: 'C = αM + βR + γK − λD. The rule override sets R to 0, and two reported conflicts raise D to 0.67. The rule still decides “high risk”, but with low confidence, so the case is escalated.',
      say: 'This is Equation (4.2) from our report, computed live from the agent outputs.',
      before: () => ensureRun(true),
    },
    {
      screen: 'report', caseId: HERO, target: ['#rep-decision', '#rep-head'],
      title: 'Every line is traceable',
      text: 'With Trace sources on, pointing at any part of the report lights up the component and version that produced it. Here the decision comes from the fusion engine, because of rule R-001.',
      before: () => {
        ensureRun(true);
        trace(true);
      },
      after: () => {
        const el = document.getElementById('rep-decision');
        if (el && MI.screens.report.highlight) MI.screens.report.highlight(el);
      },
    },
    {
      screen: 'report', caseId: 'C-1047', target: ['.draft .cut', '.draft'],
      title: 'Unsupported statements are removed',
      text: 'The language model’s draft claimed a history of diabetes. Nothing in the record supports it, so the sentence is struck out and kept out of the report (TC-12).',
      say: 'The language model only summarizes; it is never allowed to add facts.',
      before: () => {
        ensureRun(true);
        trace(false);
      },
    },
    {
      screen: 'report', caseId: 'C-1044', target: ['#fusion-viz', '#fusion-card'],
      title: 'Saying “uncertain” instead of guessing',
      text: 'For C-1044 no sufficiently relevant evidence was found, so retrieval support is 0 and the confidence of 0.57 falls below the threshold. MedIntel reports “uncertain” and asks for a clinician’s view (TC-06).',
      before: () => ensureRun(true),
    },
    {
      screen: 'whatif', target: ['#wi-result'], prefer: 'left-bottom',
      title: 'What-if lab: watch a rule fire',
      text: 'We sweep C-1042’s systolic blood pressure from 170 to 190 mmHg. At 180, rule R-001 fires, a conflict appears and the confidence drops. Examiners can move any slider themselves.',
      say: 'This is the boundary test from Chapter 6: 179, 180 and 181 mmHg.',
      before: () => {
        ensureRun(true);
        state.whatif = null;
      },
      after: () => {
        if (state.whatif && state.whatif.base !== HERO) {
          MI.changes['wi-base']({ value: HERO });
        }
        setTimeout(() => MI.sweep(170, 190), 450);
      },
    },
    {
      screen: 'report', caseId: HERO, role: 'clinician', target: ['#review-card'],
      title: 'The clinician makes the decision',
      text: 'The clinician accepts the escalation and adds a note. The review is appended to the audit trail, and the confirmed outcome is proposed to the Knowledge Curator.',
      say: 'MedIntel recommends; it never acts on its own.',
      before: () => {
        ensureRun(true);
        trace(false);
        MI.setRole('clinician', true);
        if (state.cases[HERO] && state.cases[HERO].status === 'pending_review') MI.review(HERO, 'accepted', NOTE);
      },
    },
    {
      screen: 'governance', govTab: 'knowledge', role: 'curator', target: ['#kb-card', '#kb-ver'],
      title: 'Knowledge changes only with approval',
      text: 'As Knowledge Curator we approve the validated outcome. The knowledge index moves from kb-v3 to kb-v4, and the change is recorded with who approved it.',
      say: 'Model outputs never update the knowledge base by themselves, so one wrong answer cannot become a lasting error.',
      before: () => {
        ensureRun(true);
        if (state.cases[HERO] && state.cases[HERO].status === 'pending_review') {
          MI.setRole('clinician', true);
          MI.review(HERO, 'accepted', NOTE);
        }
        MI.setRole('curator', true);
        state.govTab = 'knowledge';
        const p = state.proposals.find((x) => x.case_id === HERO && x.status === 'pending');
        if (p) MI.decideProposal(p.id, true);
      },
    },
    {
      screen: 'governance', govTab: 'audit', target: ['#audit-table thead, #audit-table tbody tr:nth-child(-n+4)', '#audit-table'],
      title: 'An append-only audit trail',
      text: 'Every report and every review is recorded with the model and rulebase versions and the retrieval IDs used, so any recommendation can be reproduced later (NFR1, NFR6).',
      before: () => {
        ensureRun(true);
        state.govTab = 'audit';
      },
    },
    {
      screen: 'compare', target: ['.vs'],
      title: 'Black box vs MedIntel',
      text: 'The same case through a single opaque model: “low risk”, with no reasons, sources or record. MedIntel escalates it with evidence, conflicts, a traceable confidence and an audit record.',
      say: 'This is the research gap from our literature survey, made visible.',
      before: () => {
        ensureRun(true);
        state.compareId = HERO;
        MI.setRole('clinician', true);
      },
    },
    {
      center: true,
      title: 'Next: Project Stage II',
      html: () => `<p>Stage II builds the real system: a model trained on a public dataset, FAISS retrieval with sentence embeddings, a live language-model endpoint, and the twelve planned test cases.</p><p class="say">Thank you. We would be glad to take questions, and the console stays open for you to explore.</p><p class="muted" style="font-size:12.5px">Siddhesh Kishor Suryawanshi · Shantanu Avinash Patil · Omkar Abaji Musale · Tanmay Shriniwas Bagul<br>Guide: Prof. G. P Mohole · JESITMR Nashik · AY 2026-27</p>`,
      before: () => {
        trace(false);
        MI.navigate('overview', { noFocus: true });
      },
    },
  ];

  let idx = -1;
  let els = null;
  let posTimer = null;

  function visible(el) {
    if (!el) return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && getComputedStyle(el).visibility !== 'hidden';
  }
  function findTarget(step) {
    for (const sel of step.target || []) {
      const list = Array.from(document.querySelectorAll(sel)).filter(visible);
      if (list.length) return list;
    }
    return null;
  }
  function rectOf(list) {
    const rs = list.map((el) => el.getBoundingClientRect());
    const top = Math.min(...rs.map((r) => r.top));
    const left = Math.min(...rs.map((r) => r.left));
    const right = Math.max(...rs.map((r) => r.right));
    const bottom = Math.max(...rs.map((r) => r.bottom));
    return { top, left, right, bottom, width: right - left, height: bottom - top };
  }
  function topInset() {
    const tb = document.querySelector('.topbar');
    if (!tb) return 0;
    const p = getComputedStyle(tb).position;
    return p === 'sticky' || p === 'fixed' ? Math.max(0, tb.getBoundingClientRect().bottom) : 0;
  }

  function build() {
    const block = document.createElement('div');
    block.className = 'tour-block';
    const spot = document.createElement('div');
    spot.className = 'tour-spot';
    const card = document.createElement('div');
    card.className = 'tour-card';
    card.setAttribute('role', 'dialog');
    card.setAttribute('aria-live', 'polite');
    card.setAttribute('aria-label', 'Guided demo');
    document.body.append(block, spot, card);
    els = { block, spot, card };
    card.addEventListener('click', (e) => {
      const b = e.target.closest('[data-tour]');
      if (!b) return;
      if (b.dataset.tour === 'next') go(idx + 1);
      if (b.dataset.tour === 'prev') go(idx - 1);
      if (b.dataset.tour === 'end') end();
    });
    block.addEventListener('click', () => go(idx + 1));
  }

  function cardHTML(step) {
    const n = STEPS.length;
    const body = step.html ? step.html() : `<p>${esc(step.text)}</p>${step.say ? `<p class="say">${esc(step.say)}</p>` : ''}`;
    return `<div class="tstep"><span>${step.center ? (idx === 0 ? 'Guided demo' : 'Wrap-up') : `Step ${idx} of ${n - 2}`}</span><button class="btn icon ghost sm" data-tour="end" aria-label="Exit the guided demo" style="margin:-6px -8px 0 0">${icon('close')}</button></div>
      <h3>${esc(step.title)}</h3>${body}
      <div class="tnav">
        <button class="btn sm" data-tour="prev" ${idx === 0 ? 'disabled' : ''}>${icon('left')}Back</button>
        <div class="tprog"><span style="width:${((idx + 1) / n) * 100}%"></span></div>
        <button class="btn sm primary" data-tour="next" id="tour-next">${idx === n - 1 ? `${icon('check')}Finish` : `Next${icon('right')}`}</button>
      </div>
      <p class="muted khint" style="font-size:11.5px"><kbd>←</kbd> <kbd>→</kbd> or click anywhere to continue · <kbd>Esc</kbd> to exit</p>`;
  }

  function place() {
    if (!els || idx < 0) return;
    const step = STEPS[idx];
    const { spot, card } = els;
    const list = step.center ? null : findTarget(step);
    card.classList.toggle('center', !list);
    if (!list) {
      spot.style.cssText = 'top:50%;left:50%;width:0;height:0;';
      card.style.top = '';
      card.style.left = '';
      return;
    }
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const inset = topInset();
    const r = rectOf(list);
    const pad = 8;
    const top = Math.max(inset + 4, r.top - pad);
    const left = Math.max(6, r.left - pad);
    const bottom = Math.min(vh - 6, r.bottom + pad);
    const right = Math.min(vw - 6, r.right + pad);
    spot.style.cssText = `top:${top}px;left:${left}px;width:${Math.max(0, right - left)}px;height:${Math.max(0, bottom - top)}px;`;
    const ch = card.offsetHeight;
    const cw = card.offsetWidth;
    const gap = 14;
    const clampX = (x) => Math.min(Math.max(16, x), vw - cw - 16);
    const clampY = (y) => Math.min(Math.max(inset + 12, y), vh - ch - 16);
    let ctop;
    let cleft;
    if (vw < 640) {
      ctop = vh - ch - 12;
      cleft = (vw - cw) / 2;
    } else if (step.prefer === 'left-bottom' && left >= cw + gap + 10) {
      cleft = left - cw - gap;
      ctop = vh - ch - 16;
    } else if (vh - bottom >= ch + gap + 10) {
      ctop = bottom + gap;
      cleft = clampX(left);
    } else if (top - inset >= ch + gap + 10) {
      ctop = top - ch - gap;
      cleft = clampX(left);
    } else if (vw - right >= cw + gap + 10) {
      cleft = right + gap;
      ctop = clampY(top);
    } else if (left >= cw + gap + 10) {
      cleft = left - cw - gap;
      ctop = clampY(top);
    } else {
      ctop = vh - ch - 18;
      cleft = vw - cw - 18;
    }
    card.style.top = `${Math.max(12, ctop)}px`;
    card.style.left = `${Math.max(12, cleft)}px`;
  }

  async function go(i) {
    if (i < 0) return;
    if (i >= STEPS.length) {
      end();
      return;
    }
    idx = i;
    const step = STEPS[i];
    if (step.before) step.before();
    if (step.role && state.role !== step.role) MI.setRole(step.role, true);
    if (step.govTab) state.govTab = step.govTab;
    if (step.screen) MI.navigate(step.screen, { caseId: step.caseId, noFocus: true });
    els.card.innerHTML = cardHTML(step);
    await new Promise((res) => requestAnimationFrame(() => requestAnimationFrame(res)));
    const list = step.center ? null : findTarget(step);
    if (list) {
      const r = rectOf(list);
      const inset = topInset();
      const avail = window.innerHeight - inset;
      const ch = els.card.offsetHeight;
      let want = inset + 14;
      if (window.innerWidth >= 640 && r.height + ch + 40 <= avail) want = inset + Math.max(14, (avail - r.height - ch - 28) / 2);
      else if (window.innerWidth < 640) want = inset + 10;
      window.scrollTo({ top: Math.max(0, window.scrollY + r.top - want), behavior: MI.reducedMotion() ? 'auto' : 'smooth' });
    }
    place();
    clearTimeout(posTimer);
    posTimer = setTimeout(() => {
      place();
      if (step.after) step.after();
      setTimeout(place, 300);
    }, MI.reducedMotion() ? 30 : 480);
    const nb = document.getElementById('tour-next');
    if (nb) nb.focus({ preventScroll: true });
  }

  function onKey(e) {
    if (idx < 0) return;
    if (e.key === 'ArrowRight' || e.key === 'PageDown') {
      e.preventDefault();
      go(idx + 1);
    } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
      e.preventDefault();
      go(idx - 1);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      end();
    }
  }
  const onMove = () => place();

  function start() {
    if (els) end();
    MI.closeModal();
    build();
    document.addEventListener('keydown', onKey, true);
    window.addEventListener('resize', onMove);
    window.addEventListener('scroll', onMove, { passive: true });
    go(0);
  }

  function end() {
    idx = -1;
    clearTimeout(posTimer);
    document.removeEventListener('keydown', onKey, true);
    window.removeEventListener('resize', onMove);
    window.removeEventListener('scroll', onMove);
    if (els) {
      els.block.remove();
      els.spot.remove();
      els.card.remove();
    }
    els = null;
    const b = document.getElementById('tour-btn');
    if (b) b.focus({ preventScroll: true });
  }

  MI.tour = { start, end, go, steps: STEPS, get index() { return idx; } };
  MI.actions['tour-start'] = () => start();
})();
