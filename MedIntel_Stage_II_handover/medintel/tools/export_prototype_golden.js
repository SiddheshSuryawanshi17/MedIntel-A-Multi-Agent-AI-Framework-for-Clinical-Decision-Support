// Runs the Stage I concept prototype's engine on the sample batches and writes every
// intermediate result as JSON. The Python port must reproduce this file (tests/test_golden_parity.py).
// Usage: node tools/export_prototype_golden.js <prototype src dir> <samples dir> > tests/golden/prototype_golden.json
const fs = require('fs');
const path = require('path');

const srcDir = path.resolve(process.argv[2]);
const samplesDir = path.resolve(process.argv[3]);
const D = require(path.join(srcDir, 'data.js'));
const E = require(path.join(srcDir, 'engine.js'));

function run(file) {
  const text = fs.readFileSync(path.join(samplesDir, file), 'utf8');
  const parsed = E.parseCSV(text);
  const { valid, rejected } = E.validateRecords(parsed.records);
  const cases = valid.map((rec) => {
    const c = E.buildCase(rec, { file });
    const b = E.analyzeCase(c);
    const versions = { model: D.MODEL.version, rulebase: D.RULEBASE.version, index: 'kb-v3' };
    return {
      case: { case_id: c.case_id, patient_id: c.patient_id, x_s: c.x_s, x_t: c.x_t, x_l: c.x_l, source: c.source },
      ml: {
        prediction: b.ml.prediction,
        probabilities: b.ml.probabilities,
        logit: b.ml.logit,
        contributions: b.ml.contributions,
        importance: b.ml.importance.map((f) => f.key),
        model_version: b.ml.model_version,
      },
      rules: { triggered: b.rules.triggered, flags: b.rules.flags, action: b.rules.action, reason: b.rules.reason, rulebase_version: b.rules.rulebase_version },
      evidence: b.evidence.map((e) => ({ source_id: e.source_id, score: e.score, direction: e.direction, matched_on: e.matched_on })),
      context: { summary: b.context.summary, conflicts: b.context.conflicts, unsupported_statements: b.context.unsupported_statements, generated_by: b.context.generated_by },
      fusion: b.fusion,
      recommendation: b.recommendation,
      priority: b.priority,
      report: E.buildReport(b, 'pending_review', '', versions),
    };
  });
  return {
    file,
    missing_columns: parsed.missingColumns,
    valid: valid.length,
    rejected: rejected.map((r) => ({ row: r.row, patient_id: r.record.patient_id, problems: r.problems.map((p) => ({ field: p.field, message: p.message, test_case: p.tc })) })),
    cases,
  };
}

const out = { generated_from: 'MedIntel Stage I concept prototype (engine.js)', settings: D.DEFAULT_SETTINGS, batches: [run('morning_batch.csv'), run('boundary_tests.csv')] };
process.stdout.write(`${JSON.stringify(out, null, 2)}\n`);
