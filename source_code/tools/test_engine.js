const assert = require('assert');
const D = require('../src/data.js');
const E = require('../src/engine.js');

const parsed = E.parseCSV(D.SAMPLE_CSV);
assert.deepStrictEqual(parsed.missingColumns, []);
assert.strictEqual(parsed.records.length, 10);
const { valid, rejected } = E.validateRecords(parsed.records);
assert.strictEqual(valid.length, 8, 'eight valid rows');
assert.strictEqual(rejected.length, 2, 'two rejected rows');
assert.strictEqual(rejected[0].problems[0].tc, 'TC-01');
assert.strictEqual(rejected[1].problems[0].tc, 'TC-02');

const expect = {
  'P-1041': { decision: 'low_risk', priority: 'routine' },
  'P-1042': { decision: 'high_risk', priority: 'escalate', conflicts: 2 },
  'P-1043': { decision: 'high_risk', priority: 'priority' },
  'P-1044': { decision: 'uncertain', priority: 'review', evidence: 0 },
  'P-1045': { decision: 'high_risk', priority: 'priority', rules: ['R-002'] },
  'P-1046': { decision: 'withheld', priority: 'manual', rules: ['R-003'] },
  'P-1047': { decision: 'high_risk', priority: 'priority', unsupported: 1 },
  'P-1048': { decision: 'low_risk', priority: 'routine' },
};
const rows = [];
for (const rec of valid) {
  const c = E.buildCase(rec);
  const b = E.analyzeCase(c);
  const x = expect[c.patient_id];
  rows.push({ id: c.patient_id, p_high: b.ml.probabilities.high_risk, pred: b.ml.prediction, rules: b.rules.triggered.join('+') || '-', K: b.fusion.component_scores.K, ev: b.evidence.length, conf: b.context.conflicts.length, C: b.fusion.confidence, decision: b.fusion.decision, priority: b.priority.key });
  assert.strictEqual(b.fusion.decision, x.decision, `${c.patient_id} decision`);
  assert.strictEqual(b.priority.key, x.priority, `${c.patient_id} priority`);
  if (x.conflicts !== undefined) assert.strictEqual(b.context.conflicts.length, x.conflicts, `${c.patient_id} conflicts`);
  if (x.evidence !== undefined) assert.strictEqual(b.evidence.length, x.evidence, `${c.patient_id} evidence`);
  if (x.rules) assert.deepStrictEqual(b.rules.triggered, x.rules, `${c.patient_id} rules`);
  if (x.unsupported !== undefined) assert.strictEqual(b.context.unsupported_statements.length, x.unsupported);
  const margin = Math.abs(b.fusion.confidence - b.fusion.threshold);
  if (!b.fusion.override) assert.ok(margin >= 0.02, `${c.patient_id} C=${b.fusion.confidence} too close to the threshold`);
}
console.table(rows);

// boundary behaviour of R-001 (179 / 180 / 181), as in Table 6.5.2
const base = E.buildCase(valid.find((r) => r.patient_id === 'P-1042'));
for (const sbp of [179, 180, 181]) {
  const c = { ...base, x_s: { ...base.x_s, systolic_bp: sbp } };
  const r = E.ruleEvaluate(c);
  assert.strictEqual(r.triggered.includes('R-001'), sbp >= 180, `R-001 at ${sbp}`);
}
// suppress outranks elevate at equal severity
const minor = { ...base, x_s: { ...base.x_s, age: 16, systolic_bp: 185 } };
assert.strictEqual(E.ruleEvaluate(minor).action, 'suppress');
// report fields match the Section 4.4.2 schema
const b = E.analyzeCase(base);
const rep = E.buildReport(b, 'pending_review', '', { model: 'm', rulebase: '1.0.0', index: 'kb-v3' });
for (const k of ['case_id', 'patient_id', 'findings', 'prediction', 'triggered_rules', 'evidence_refs', 'conflicts', 'confidence', 'recommendation', 'status', 'versions']) assert.ok(k in rep, k);
console.log('findings:', rep.findings);
console.log('recommendation:', rep.recommendation);
console.log('ALL ENGINE TESTS PASSED');
