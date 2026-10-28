(function (root) {
  'use strict';

  const SCHEMA = [
    { key: 'patient_id', label: 'Patient ID', type: 'id', required: true },
    { key: 'age', label: 'Age', type: 'int', unit: 'years', min: 0, max: 110, required: true },
    { key: 'sex', label: 'Sex', type: 'enum', values: ['F', 'M'], required: true },
    { key: 'systolic_bp', label: 'Systolic BP', type: 'num', unit: 'mmHg', min: 60, max: 260, required: true, ref: [90, 129], refText: '90-129' },
    { key: 'diastolic_bp', label: 'Diastolic BP', type: 'num', unit: 'mmHg', min: 30, max: 160, required: true, ref: [60, 79], refText: '60-79' },
    { key: 'heart_rate', label: 'Heart rate', type: 'num', unit: 'bpm', min: 25, max: 220, required: true, ref: [60, 100], refText: '60-100' },
    { key: 'temperature_c', label: 'Temperature', type: 'num', unit: '°C', min: 34, max: 42, required: true, ref: [36.1, 37.2], refText: '36.1-37.2', fahrenheitCheck: true },
    { key: 'total_cholesterol', label: 'Total cholesterol', type: 'num', unit: 'mg/dL', min: 80, max: 400, required: true, ref: [0, 199], refText: 'below 200' },
    { key: 'fasting_glucose', label: 'Fasting glucose', type: 'num', unit: 'mg/dL', min: 40, max: 500, required: true, ref: [70, 99], refText: '70-99' },
    { key: 'creatinine', label: 'Creatinine', type: 'num', unit: 'mg/dL', min: 0.2, max: 12, required: false, ref: [0.6, 1.2], refText: '0.6-1.2' },
    { key: 'bmi', label: 'BMI', type: 'num', unit: 'kg/m²', min: 12, max: 70, required: true, ref: [18.5, 24.9], refText: '18.5-24.9' },
    { key: 'smoker', label: 'Current smoker', type: 'bool', required: true },
    { key: 'notes', label: 'Clinical notes', type: 'text', required: false },
  ];

  const SAMPLE_CSV = [
    'patient_id,age,sex,systolic_bp,diastolic_bp,heart_rate,temperature_c,total_cholesterol,fasting_glucose,creatinine,bmi,smoker,notes',
    'P-1041,42,F,118,76,72,36.7,182,88,0.8,23.1,no,Annual check-up. No complaints.',
    'P-1042,34,M,186,112,88,36.9,176,94,0.9,24.6,no,"Headache since morning, one high reading at triage."',
    'P-1043,67,M,164,96,84,36.8,258,142,1.3,31.2,yes,Exertional chest tightness for two weeks.',
    'P-1044,61,M,136,84,78,36.6,228,118,1.1,28.9,no,"Occasional palpitations, one fainting episode last month."',
    'P-1045,63,F,148,90,76,36.8,246,108,,29.1,yes,Creatinine sample not received by the lab.',
    'P-1046,16,M,124,78,92,36.9,165,90,0.7,21.0,no,Sports physical. Mild palpitations after exercise.',
    'P-1047,72,F,158,88,82,36.7,251,112,1.6,29.8,no,Breathlessness on stairs and ankle swelling.',
    'P-1048,29,F,112,70,66,36.5,170,84,0.7,21.8,no,Pre-employment health check.',
    ',51,M,142,88,80,36.8,231,104,1.0,27.5,no,Follow-up visit. Identifier missing in the export.',
    'P-1050,47,F,128,82,74,101.3,204,96,0.9,25.4,no,Temperature entered from a Fahrenheit thermometer.',
  ].join('\n');

  const EVENTS = {
    'P-1041': [['07:05', 'Registered for annual check-up'], ['07:12', 'Vitals recorded by nursing staff']],
    'P-1042': [['07:58', 'Arrived at outpatient triage with headache'], ['08:12', 'Triage BP 186/112 mmHg'], ['08:20', 'Repeat BP reading requested']],
    'P-1043': [['07:20', 'Referred by general physician'], ['07:41', 'Outpatient ECG ordered']],
    'P-1044': [['07:33', 'Follow-up after an emergency visit last month'], ['07:50', 'Holter monitoring discussed']],
    'P-1045': [['07:15', 'Blood samples collected'], ['07:46', 'Lab: creatinine sample not received']],
    'P-1046': [['08:01', 'Sports physical requested by school'], ['08:09', 'Vitals recorded']],
    'P-1047': [['07:25', 'Visit for breathlessness on exertion'], ['07:38', 'Ankle swelling noted on examination']],
    'P-1048': [['07:10', 'Pre-employment screening'], ['07:18', 'Vitals recorded']],
  };

  const SCRIPTED = {
    'P-1041': [
      'Blood pressure, cholesterol and fasting glucose are within the reference ranges, and the patient does not smoke.',
      'The ML model estimates a low risk, and the retrieved evidence describes this profile as low risk.',
      'No rule was triggered and the components agree.',
    ],
    'P-1042': [
      'Systolic blood pressure was 186 mmHg at triage, which triggers rule R-001.',
      'The ML model estimates a low long-term risk because the patient is young and the other risk factors are normal.',
      'The rule result and the retrieved evidence on severely raised blood pressure disagree with the model, so the case is escalated for clinician review.',
    ],
    'P-1043': [
      'Blood pressure, total cholesterol and fasting glucose are raised, and the patient smokes.',
      'The ML model estimates a high risk, consistent with the retrieved evidence on each of these factors.',
      'No rule was triggered and the components agree.',
    ],
    'P-1044': [
      'Blood pressure, cholesterol and fasting glucose are mildly raised but below the usual treatment thresholds.',
      'The ML model leans toward high risk, but no sufficiently relevant evidence was found in the knowledge base.',
      'The notes mention palpitations and a recent fainting episode, which this framework does not assess.',
    ],
    'P-1045': [
      'Blood pressure and total cholesterol are raised and the patient smokes.',
      'Serum creatinine is missing, so rule R-002 flags that kidney-related checks could not be run.',
      'The ML model estimates a high risk, consistent with the retrieved evidence.',
    ],
    'P-1046': [
      'The patient is 16 years old, so rule R-003 withholds the risk recommendation because the model is intended for adults.',
      'The recorded values raise no immediate concern, but assessment should follow pediatric reference ranges.',
    ],
    'P-1047': [
      'Blood pressure and total cholesterol are raised, and serum creatinine is 1.6 mg/dL.',
      { text: 'She has a known history of type 2 diabetes.', unsupported: 'Not supported: fasting glucose is 112 mg/dL and no record or note mentions diabetes.' },
      'The ML model estimates a high risk, consistent with the retrieved evidence on blood pressure, cholesterol and kidney function.',
    ],
    'P-1048': [
      'All recorded values are within the reference ranges and the patient does not smoke.',
      'The ML model and the retrieved evidence both indicate low risk.',
    ],
  };

  const RULEBASE = {
    version: '1.0.0',
    approved_by: 'knowledge_curator',
    rules: [
      { rule_id: 'R-001', field: 'systolic_bp', operator: '>=', threshold: 180, action: 'elevate', severity: 'high', message: 'Systolic blood pressure at or above 180 mmHg; urgent review advised.' },
      { rule_id: 'R-002', field: 'creatinine', operator: 'missing', threshold: null, action: 'flag', severity: 'medium', message: 'Serum creatinine missing; kidney-related rules cannot be checked.' },
      { rule_id: 'R-003', field: 'age', operator: '<', threshold: 18, action: 'suppress', severity: 'high', message: 'Model intended for adults; recommendation withheld under age 18.' },
      { rule_id: 'R-004', field: 'heart_rate', operator: '>=', threshold: 120, action: 'flag', severity: 'medium', message: 'Resting heart rate at or above 120 bpm; check for arrhythmia or acute illness.' },
    ],
  };

  const MODEL = {
    name: 'risk-lr',
    version: 'risk-lr 0.3 (demo weights)',
    base_logit: -2.2,
    features: [
      { key: 'age', label: 'Age', unit: 'years', coef: 0.06, ref: 50 },
      { key: 'sex_m', label: 'Male sex', unit: '', coef: 0.45, ref: 0 },
      { key: 'systolic_bp', label: 'Systolic BP', unit: 'mmHg', coef: 0.018, ref: 120 },
      { key: 'total_cholesterol', label: 'Total cholesterol', unit: 'mg/dL', coef: 0.009, ref: 190 },
      { key: 'fasting_glucose', label: 'Fasting glucose', unit: 'mg/dL', coef: 0.01, ref: 90 },
      { key: 'smoker', label: 'Current smoker', unit: '', coef: 0.85, ref: 0 },
      { key: 'bmi', label: 'BMI', unit: 'kg/m²', coef: 0.04, ref: 24 },
      { key: 'creatinine', label: 'Creatinine', unit: 'mg/dL', coef: 0.6, ref: 0.9, impute: 0.9 },
    ],
  };

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  const KB = [
    {
      source_id: 'KB-HTN-001', title: 'Severely raised blood pressure', topic: 'Blood pressure', direction: 'raises',
      passage: 'A systolic reading of 180 mmHg or higher needs prompt clinical review and a repeat measurement to confirm the value before any decision is made.',
      match: (x) => x.systolic_bp >= 180 || x.diastolic_bp >= 120,
      score: (x) => clamp(0.86 + 0.002 * (x.systolic_bp - 180), 0.86, 0.95),
      matchedOn: (x) => `systolic BP ${x.systolic_bp} mmHg`,
    },
    {
      source_id: 'KB-HTN-004', title: 'Raised blood pressure and cardiovascular risk', topic: 'Blood pressure', direction: 'raises',
      passage: 'Blood pressure that stays at 140/90 mmHg or above increases long-term cardiovascular risk and is usually managed with lifestyle changes and, where indicated, medication.',
      match: (x) => x.systolic_bp >= 140 || x.diastolic_bp >= 90,
      score: (x) => clamp(0.70 + 0.004 * (Math.max(x.systolic_bp, 140) - 140), 0.70, 0.84),
      matchedOn: (x) => `BP ${x.systolic_bp}/${x.diastolic_bp} mmHg`,
    },
    {
      source_id: 'KB-LIP-007', title: 'High total cholesterol', topic: 'Lipids', direction: 'raises',
      passage: 'Total cholesterol of 240 mg/dL or more is considered high and adds to cardiovascular risk alongside blood pressure, smoking and diabetes.',
      match: (x) => x.total_cholesterol >= 240,
      score: (x) => clamp(0.72 + 0.002 * (x.total_cholesterol - 240), 0.72, 0.82),
      matchedOn: (x) => `total cholesterol ${x.total_cholesterol} mg/dL`,
    },
    {
      source_id: 'KB-SMK-003', title: 'Smoking and cardiovascular risk', topic: 'Smoking', direction: 'raises',
      passage: 'Current smoking substantially raises cardiovascular risk. Risk falls steadily after stopping, so smoking status should be recorded at every visit.',
      match: (x) => x.smoker === true,
      score: () => 0.79,
      matchedOn: () => 'current smoker',
    },
    {
      source_id: 'KB-DM-010', title: 'Raised fasting glucose', topic: 'Glucose', direction: 'raises',
      passage: 'A fasting glucose of 126 mg/dL or higher, confirmed on repeat testing, is consistent with diabetes, which increases cardiovascular risk.',
      match: (x) => x.fasting_glucose >= 126,
      score: (x) => clamp(0.74 + 0.002 * (x.fasting_glucose - 126), 0.74, 0.82),
      matchedOn: (x) => `fasting glucose ${x.fasting_glucose} mg/dL`,
    },
    {
      source_id: 'KB-CKD-004', title: 'Kidney function and cardiovascular risk', topic: 'Kidney', direction: 'raises',
      passage: 'Reduced kidney function, often seen first as a raised serum creatinine, is linked to higher cardiovascular risk and changes how some medicines are dosed.',
      match: (x) => x.creatinine !== null && x.creatinine >= 1.5,
      score: () => 0.71,
      matchedOn: (x) => `creatinine ${x.creatinine} mg/dL`,
    },
    {
      source_id: 'KB-CKD-009', title: 'Missing kidney function results', topic: 'Kidney', direction: 'neutral',
      passage: 'When serum creatinine is not available, kidney-related risk checks cannot be completed and the result should be requested before medication decisions.',
      match: (x) => x.creatinine === null,
      score: () => 0.62,
      matchedOn: () => 'creatinine missing',
    },
    {
      source_id: 'KB-HR-005', title: 'Raised resting heart rate', topic: 'Heart rate', direction: 'neutral',
      passage: 'A resting heart rate that stays above 100 beats per minute should be evaluated for an underlying cause, such as an arrhythmia, infection or dehydration.',
      match: (x) => x.heart_rate > 100,
      score: (x) => clamp(0.68 + 0.003 * (x.heart_rate - 100), 0.68, 0.80),
      matchedOn: (x) => `heart rate ${x.heart_rate} bpm`,
    },
    {
      source_id: 'KB-PED-002', title: 'Adult risk tools in patients under 18', topic: 'Pediatrics', direction: 'neutral',
      passage: 'Risk tools developed for adults should not be applied to patients under 18. Pediatric assessment uses age-specific reference ranges.',
      match: (x) => x.age < 18,
      score: () => 0.88,
      matchedOn: (x) => `age ${x.age}`,
    },
    {
      source_id: 'KB-GEN-001', title: 'Low-risk cardiovascular profile', topic: 'General', direction: 'lowers',
      passage: 'Normal blood pressure, cholesterol and fasting glucose without smoking are associated with a low short-term cardiovascular risk in adults.',
      match: (x) => x.age >= 18 && x.systolic_bp < 130 && x.diastolic_bp < 85 && x.total_cholesterol < 200 && x.fasting_glucose < 100 && x.smoker === false,
      score: () => 0.78,
      matchedOn: () => 'BP, cholesterol and glucose in range; non-smoker',
    },
  ];

  const DEFAULT_SETTINGS = { alpha: 0.5, beta: 0.3, gamma: 0.2, lam: 0.5, threshold: 0.6, k: 5, min_score: 0.55 };

  const data = { SCHEMA, SAMPLE_CSV, EVENTS, SCRIPTED, RULEBASE, MODEL, KB, DEFAULT_SETTINGS };
  root.MedIntelData = data;
  if (typeof module !== 'undefined' && module.exports) module.exports = data;
})(typeof window !== 'undefined' ? window : globalThis);
