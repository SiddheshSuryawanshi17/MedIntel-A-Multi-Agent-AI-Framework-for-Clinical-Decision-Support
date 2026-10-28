(function (root) {
  'use strict';
  const D = root.MedIntelData || (typeof require === 'function' ? require('./data.js') : null);

  const OVERRIDES = { elevate: 'high_risk', suppress: 'withheld' };
  const SEVERITY = { low: 1, medium: 2, high: 3 };
  const PRECEDENCE = { flag: 0, elevate: 1, suppress: 2 };
  const OPS = {
    '>': (a, b) => a > b,
    '>=': (a, b) => a >= b,
    '<': (a, b) => a < b,
    '<=': (a, b) => a <= b,
  };
  const RULE_HINTS = {
    'R-001': 'Confirm the blood pressure with a repeat reading.',
    'R-004': 'Check the rhythm and look for an acute cause.',
  };

  const round = (v, n = 3) => Math.round(v * 10 ** n) / 10 ** n;
  const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
  const field = (key) => D.SCHEMA.find((f) => f.key === key);

  function parseCSV(text) {
    const src = String(text || '').replace(/^﻿/, '');
    const rows = [];
    let row = [];
    let cell = '';
    let quoted = false;
    for (let i = 0; i < src.length; i += 1) {
      const ch = src[i];
      if (quoted) {
        if (ch === '"' && src[i + 1] === '"') {
          cell += '"';
          i += 1;
        } else if (ch === '"') quoted = false;
        else cell += ch;
      } else if (ch === '"') quoted = true;
      else if (ch === ',') {
        row.push(cell);
        cell = '';
      } else if (ch === '\n' || ch === '\r') {
        if (ch === '\r' && src[i + 1] === '\n') i += 1;
        row.push(cell);
        rows.push(row);
        row = [];
        cell = '';
      } else cell += ch;
    }
    if (cell !== '' || row.length) {
      row.push(cell);
      rows.push(row);
    }
    const clean = rows.filter((r) => r.some((c) => c.trim() !== ''));
    if (!clean.length) return { header: [], records: [], missingColumns: D.SCHEMA.filter((f) => f.required).map((f) => f.key) };
    const header = clean[0].map((h) => h.trim().toLowerCase());
    const records = clean.slice(1).map((cells, i) => {
      const rec = { __row: i + 2 };
      header.forEach((h, j) => {
        rec[h] = (cells[j] ?? '').trim();
      });
      return rec;
    });
    const missingColumns = D.SCHEMA.filter((f) => f.required && !header.includes(f.key)).map((f) => f.key);
    return { header, records, missingColumns };
  }

  function validateRecords(records) {
    const valid = [];
    const rejected = [];
    const seen = new Set();
    for (const raw of records) {
      const problems = [];
      const rec = { __row: raw.__row };
      for (const f of D.SCHEMA) {
        const v = raw[f.key] === undefined ? '' : String(raw[f.key]).trim();
        if (f.type === 'id') {
          if (!v) problems.push({ field: f.key, message: 'field required', tc: 'TC-01' });
          else if (seen.has(v)) problems.push({ field: f.key, message: 'repeated patient_id', tc: null });
          rec[f.key] = v;
        } else if (f.type === 'text') {
          rec[f.key] = v;
        } else if (f.type === 'enum') {
          const u = v.toUpperCase();
          if (!v) problems.push({ field: f.key, message: 'field required', tc: null });
          else if (!f.values.includes(u)) problems.push({ field: f.key, message: `expected ${f.values.join(' or ')}`, tc: null });
          rec[f.key] = u;
        } else if (f.type === 'bool') {
          const u = v.toLowerCase();
          if (['yes', 'true', '1', 'y'].includes(u)) rec[f.key] = true;
          else if (['no', 'false', '0', 'n'].includes(u)) rec[f.key] = false;
          else {
            problems.push({ field: f.key, message: v ? 'expected yes or no' : 'field required', tc: null });
            rec[f.key] = null;
          }
        } else {
          if (v === '') {
            if (f.required) problems.push({ field: f.key, message: 'field required', tc: null });
            rec[f.key] = null;
            continue;
          }
          const num = Number(v);
          if (!Number.isFinite(num) || (f.type === 'int' && !Number.isInteger(num))) {
            problems.push({ field: f.key, message: `"${v}" is not a ${f.type === 'int' ? 'whole number' : 'number'}`, tc: null });
            rec[f.key] = null;
            continue;
          }
          if (f.fahrenheitCheck && num >= 93 && num <= 110) {
            problems.push({ field: f.key, message: `${num} looks like °F; expected °C (${f.min}-${f.max}). Not converted.`, tc: 'TC-02' });
          } else if (num < f.min || num > f.max) {
            problems.push({ field: f.key, message: `${num} is outside ${f.min}-${f.max} ${f.unit}`, tc: null });
          }
          rec[f.key] = num;
        }
      }
      if (rec.patient_id) seen.add(rec.patient_id);
      if (problems.length) rejected.push({ row: raw.__row, raw, record: rec, problems });
      else valid.push(rec);
    }
    return { valid, rejected };
  }

  function buildCase(rec, opts = {}) {
    const digits = (rec.patient_id.match(/\d+/g) || ['0000']).join('').slice(-4).padStart(4, '0');
    const xs = {};
    for (const f of D.SCHEMA) if (!['patient_id', 'notes'].includes(f.key)) xs[f.key] = rec[f.key];
    return {
      case_id: `C-${digits}`,
      patient_id: rec.patient_id,
      x_s: xs,
      x_t: rec.notes || '',
      x_l: (D.EVENTS[rec.patient_id] || []).map(([t, text]) => ({ time: t, text })),
      source: `${opts.file || 'morning_batch.csv'}:${rec.__row || '-'}`,
      scripted: opts.scripted !== false && Boolean(D.SCRIPTED[rec.patient_id]),
    };
  }

  function featureValue(xs, key) {
    if (key === 'sex_m') return xs.sex === 'M' ? 1 : 0;
    if (key === 'smoker') return xs.smoker ? 1 : 0;
    return xs[key];
  }

  function mlAnalyze(c) {
    const xs = c.x_s;
    let logit = D.MODEL.base_logit;
    const contributions = D.MODEL.features.map((f) => {
      let x = featureValue(xs, f.key);
      let imputed = false;
      if (x === null || x === undefined) {
        x = f.impute;
        imputed = true;
      }
      const contribution = f.coef * (x - f.ref);
      logit += contribution;
      return { key: f.key, label: f.label, unit: f.unit, value: x, imputed, contribution: round(contribution, 3) };
    });
    const pHigh = 1 / (1 + Math.exp(-logit));
    const prediction = pHigh >= 0.5 ? 'high_risk' : 'low_risk';
    const importance = [...contributions].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
    return {
      prediction,
      probabilities: { high_risk: round(pHigh, 3), low_risk: round(1 - pHigh, 3) },
      logit: round(logit, 3),
      contributions,
      importance,
      model_version: D.MODEL.version,
    };
  }

  function fires(rule, value) {
    if (rule.operator === 'missing') return value === null || value === undefined;
    return value !== null && value !== undefined && OPS[rule.operator](value, rule.threshold);
  }

  function ruleEvaluate(c, rulebase = D.RULEBASE) {
    const checks = rulebase.rules.map((rule) => {
      const value = c.x_s[rule.field] === undefined ? null : c.x_s[rule.field];
      return { rule, value, fired: fires(rule, value) };
    });
    const hits = checks.filter((k) => k.fired).map((k) => k.rule);
    let top = null;
    for (const r of hits) {
      if (!top) top = r;
      else {
        const a = [SEVERITY[r.severity], PRECEDENCE[r.action]];
        const b = [SEVERITY[top.severity], PRECEDENCE[top.action]];
        if (a[0] > b[0] || (a[0] === b[0] && a[1] > b[1])) top = r;
      }
    }
    return {
      triggered: hits.map((r) => r.rule_id),
      flags: hits.map((r) => r.message),
      action: top ? top.action : null,
      reason: top ? top.message : 'no rule triggered',
      top,
      checks,
      rulebase_version: rulebase.version,
    };
  }

  function retrieve(c, settings = D.DEFAULT_SETTINGS, kb = D.KB) {
    const xs = c.x_s;
    return kb
      .filter((p) => p.match(xs))
      .map((p) => ({
        source_id: p.source_id,
        title: p.title,
        topic: p.topic,
        passage: p.passage,
        direction: p.direction,
        score: round(p.score(xs), 3),
        matched_on: p.matchedOn(xs),
      }))
      .filter((e) => e.score >= settings.min_score)
      .sort((a, b) => b.score - a.score)
      .slice(0, settings.k);
  }

  const LABEL = { high_risk: 'high risk', low_risk: 'low risk', withheld: 'withheld', uncertain: 'uncertain' };

  function fmt(key, v) {
    const f = field(key);
    if (v === null || v === undefined) return 'missing';
    if (key === 'sex') return v;
    if (key === 'smoker') return v ? 'yes' : 'no';
    return `${v}${f && f.unit ? ` ${f.unit}` : ''}`;
  }

  function abnormal(key, v) {
    const f = field(key);
    if (!f || !f.ref || v === null || v === undefined) return null;
    if (v > f.ref[1]) return 'high';
    if (v < f.ref[0]) return 'low';
    return null;
  }

  function templateSummary(c, ml, rules, evidence, conflicts) {
    const xs = c.x_s;
    const out = [];
    const raised = ['systolic_bp', 'total_cholesterol', 'fasting_glucose', 'creatinine', 'heart_rate']
      .filter((k) => abnormal(k, xs[k]) === 'high')
      .map((k) => `${field(k).label.toLowerCase().replace('bp', 'blood pressure')} (${fmt(k, xs[k])})`);
    if (raised.length) out.push(`${raised.length === 1 ? 'The' : 'Raised values:'} ${raised.join(', ')}${raised.length === 1 ? ' is above the reference range' : ''}.`.replace('The systolic', 'Systolic').replace(/^The (\w)/, (m, a) => a.toUpperCase()));
    else out.push('The recorded values are within the reference ranges.');
    if (xs.smoker) out.push('The patient smokes.');
    out.push(`The ML model estimates a ${LABEL[ml.prediction]} (probability of high risk ${ml.probabilities.high_risk.toFixed(2)}).`);
    if (rules.top) out.push(`Rule ${rules.top.rule_id} ${rules.top.action === 'flag' ? 'flags the case' : `${rules.top.action}s the case`}: ${rules.top.message}`);
    else out.push('No rule was triggered.');
    if (evidence.length) out.push(`Retrieved evidence: ${evidence.map((e) => e.source_id).join(', ')}.`);
    else out.push('No sufficiently relevant evidence was found in the knowledge base.');
    if (conflicts.length) out.push(conflicts.length === 1 ? 'One conflict between the components is reported for review.' : `${conflicts.length} conflicts between the components are reported for review.`);
    return out;
  }

  function synthesize(c, ml, rules, evidence) {
    const conflicts = [];
    const override = OVERRIDES[rules.action] || null;
    if (override === 'high_risk' && ml.prediction !== 'high_risk') {
      const r = rules.top;
      conflicts.push({ type: 'rule_vs_ml', text: `ML predicts low risk, but rule ${r.rule_id} elevates the case (${field(r.field).label} ${fmt(r.field, c.x_s[r.field])}).` });
    }
    const strongRaise = evidence.filter((e) => e.direction === 'raises' && e.score >= 0.75);
    const strongLower = evidence.filter((e) => e.direction === 'lowers' && e.score >= 0.75);
    if (ml.prediction === 'low_risk' && strongRaise.length) {
      conflicts.push({ type: 'evidence_vs_ml', text: `Retrieved evidence (${strongRaise.map((e) => e.source_id).join(', ')}) points to raised risk, while ML predicts low risk.` });
    }
    if (ml.prediction === 'high_risk' && strongLower.length) {
      conflicts.push({ type: 'evidence_vs_ml', text: `Retrieved evidence (${strongLower.map((e) => e.source_id).join(', ')}) describes a low-risk profile, while ML predicts high risk.` });
    }
    const scripted = c.scripted && D.SCRIPTED[c.patient_id];
    const sentences = scripted || templateSummary(c, ml, rules, evidence, conflicts);
    const draft = sentences.map((s) => (typeof s === 'string' ? { text: s } : { text: s.text, unsupported: s.unsupported }));
    return {
      draft,
      summary: draft.filter((d) => !d.unsupported).map((d) => d.text).join(' '),
      conflicts: conflicts.map((k) => k.text),
      conflict_details: conflicts,
      unsupported_statements: draft.filter((d) => d.unsupported).map((d) => d.text),
      generated_by: scripted ? 'scripted' : 'template',
    };
  }

  function fuse(ml, rules, evidence, context, s = D.DEFAULT_SETTINGS) {
    const override = OVERRIDES[rules.action] || null;
    const M = ml.probabilities[ml.prediction];
    const R = override && override !== ml.prediction ? 0 : 1;
    const K = evidence.length ? mean(evidence.map((e) => e.score)) : 0;
    const Dv = Math.min(1, context.conflicts.length / 3);
    const terms = { aM: s.alpha * M, bR: s.beta * R, gK: s.gamma * K, lD: s.lam * Dv };
    const raw = terms.aM + terms.bR + terms.gK - terms.lD;
    const C = Math.max(0, raw);
    const below = C < s.threshold;
    const decision = override || (below ? 'uncertain' : ml.prediction);
    return {
      decision,
      confidence: round(C, 3),
      raw: round(raw, 3),
      disagreement: round(Dv, 3),
      component_scores: { M: round(M, 3), R, K: round(K, 3), D: round(Dv, 3) },
      terms: { aM: round(terms.aM, 3), bR: round(terms.bR, 3), gK: round(terms.gK, 3), lD: round(terms.lD, 3) },
      threshold: s.threshold,
      weights: { alpha: s.alpha, beta: s.beta, gamma: s.gamma, lam: s.lam },
      override,
      below_threshold: below,
    };
  }

  function recommend(fusion, rules) {
    const hint = rules.top && RULE_HINTS[rules.top.rule_id] ? ` ${RULE_HINTS[rules.top.rule_id]}` : '';
    if (fusion.decision === 'withheld') return 'Recommendation withheld: the risk model is intended for adults. Assess against pediatric reference ranges.';
    if (fusion.decision === 'high_risk' && fusion.override && fusion.below_threshold) return `Escalate for same-day clinician review; rule ${rules.top.rule_id} overrides the model.${hint}`;
    if (fusion.decision === 'high_risk') return `Prioritize clinician review today.${hint}`;
    if (fusion.decision === 'uncertain') return 'Clinician review advised: the evidence is not strong enough for a confident recommendation.';
    return rules.flags.length ? `Routine follow-up once the flagged item is resolved.${hint}` : 'Routine follow-up.';
  }

  function priority(fusion) {
    if (fusion.decision === 'high_risk' && fusion.below_threshold) return { level: 1, key: 'escalate', label: 'Escalate' };
    if (fusion.decision === 'uncertain') return { level: 2, key: 'review', label: 'Review' };
    if (fusion.decision === 'high_risk') return { level: 3, key: 'priority', label: 'Priority' };
    if (fusion.decision === 'withheld') return { level: 4, key: 'manual', label: 'Assess manually' };
    return { level: 5, key: 'routine', label: 'Routine' };
  }

  function findings(c, ml, rules) {
    const keyOf = (k) => (k === 'sex_m' ? 'sex' : k);
    const top = ml.importance.slice(0, 3).map((f) => `${keyOf(f.key)} = ${f.key === 'sex_m' ? c.x_s.sex : f.key === 'smoker' ? (c.x_s.smoker ? 'yes' : 'no') : c.x_s[keyOf(f.key)] === null ? 'missing' : c.x_s[keyOf(f.key)]}`);
    return top.concat(rules.flags);
  }

  function analyzeCase(c, settings = D.DEFAULT_SETTINGS) {
    const ml = mlAnalyze(c);
    const rules = ruleEvaluate(c);
    const evidence = retrieve(c, settings);
    const context = synthesize(c, ml, rules, evidence);
    const fusion = fuse(ml, rules, evidence, context, settings);
    return { case: c, ml, rules, evidence, context, fusion, recommendation: recommend(fusion, rules), priority: priority(fusion) };
  }

  function buildReport(b, status, notes, versions) {
    return {
      case_id: b.case.case_id,
      patient_id: b.case.patient_id,
      findings: findings(b.case, b.ml, b.rules),
      prediction: b.ml.prediction,
      decision: b.fusion.decision,
      triggered_rules: b.rules.triggered,
      evidence_refs: b.evidence.map((e) => e.source_id),
      conflicts: b.context.conflicts,
      confidence: b.fusion.confidence,
      component_scores: b.fusion.component_scores,
      summary: b.context.summary,
      recommendation: b.recommendation,
      status,
      notes: notes || '',
      versions,
    };
  }

  function auditRecord(b, status, versions, timestamp) {
    return {
      case_id: b.case.case_id,
      model_version: versions.model,
      rulebase_version: versions.rulebase,
      retrieval_ids: b.evidence.map((e) => e.source_id).join(','),
      status,
      timestamp,
    };
  }

  const engine = {
    OVERRIDES, LABEL,
    parseCSV, validateRecords, buildCase, mlAnalyze, ruleEvaluate, retrieve, synthesize, fuse,
    recommend, priority, findings, analyzeCase, buildReport, auditRecord, fmt, abnormal, field, round, mean,
  };
  root.MedIntelEngine = engine;
  if (typeof module !== 'undefined' && module.exports) module.exports = engine;
})(typeof window !== 'undefined' ? window : globalThis);
