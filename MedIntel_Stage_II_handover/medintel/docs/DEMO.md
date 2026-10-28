# MedIntel demo guide (about 8 minutes)

A script for presenting the Stage II system: what to prepare, what to click, what to say, and the questions
examiners tend to ask. Everything shown runs on synthetic or public de-identified data.

## Before the demo (once, the day before)

```bash
uv sync
uv run medintel fetch-data && uv run medintel train      # trained risk model and model card
uv run medintel evaluate                                 # reports/evaluation/EVALUATION.md and figures
uv run medintel users add "Dr. Rao" --role clinician --role data_admin --save-token ~/rao.token
uv run medintel users add "Ms. Iyer" --role curator --save-token ~/iyer.token
```

Optional: put `ANTHROPIC_API_KEY=...` in `.env` for Claude-written summaries; without it the template writes them
and the report says so. On the day, start the server and open the dashboard in a fresh browser window:

```bash
uv run medintel serve        # then open http://127.0.0.1:8000
```

Keep the two token files open in a text editor so you can paste them, and practise the flow once. Use a new
`MEDINTEL_HOME` folder for the rehearsal and another for the real demo if you want a clean audit trail.

## The script

**1. The idea (30 seconds).** "A single model that reads the case, recalls medicine and writes the answer is hard to
check. MedIntel splits the work: a trained risk model, a versioned rulebase, retrieval from a sourced knowledge base,
and a writer that may only state what the other agents produced. Equation 4.2 fuses them into a confidence C, and
every report is audited."

**2. Sign in and validate (1 minute).** Sign in as Dr. Rao. On Batches, pick `chest_pain_clinic.csv` and press
Validate. Point out that every record is checked before any agent runs. To show TC-01 and TC-02, validate
`morning_batch.csv`: two of its ten rows are rejected with the reason and the test case (row 10 has no patient ID;
row 11 has a temperature of 101.3, which looks like °F and is not converted silently).

**3. Background analysis and triage (1 minute).** Press Analyse. The batch runs as a background job with a progress
bar and lands on the triage queue: Escalate first, then Priority, then Routine. "The queue orders cases by priority,
then by the model's probability of high risk."

**4. One case, explained (2 minutes).** Open C-3006, the escalated case.
- The decision header: high risk, escalated, C = 0.27 below the threshold of 0.60.
- "The model says low risk (0.05): normal exercise test, non-anginal pain. But rule R-001 fires on a systolic BP of
  182, and a triggered rule always takes precedence." Show the two conflicts and the "How the decision was reached"
  steps: the disagreement D lowers C, so the case goes to a clinician the same day.
- Risk model card: the bars are exact SHAP values; red raises the estimate, blue lowers it.
- Evidence: each passage has a source ID and a link to the CDC or NIH page it summarises.
- What if: change the systolic BP to 150 and press Recompute. R-001 stops firing and the case becomes routine.
  "Nothing is saved: exploration never touches the record or the audit trail."

**5. Review, governance and audit (1.5 minutes).** Accept the report with a note. "An accepted outcome becomes a
proposal; it cannot reach the knowledge base until a curator approves it." Sign out, sign in as Ms. Iyer, open
Knowledge and approve: the index moves from kb-v3 to kb-v4. Open the Audit trail: Dr. Rao's review and the system's
report are there with names and times, and the table cannot be edited or deleted.

**6. Evaluation (1 minute).** Open Evaluation. "We measured what Chapters 6 and 7 promised: all twelve planned test
cases pass, every report passes the schema check, the worst case takes well under the 30-second target. With weights
tuned on the validation split, the full pipeline beats ML alone on F1 by sending its least certain cases to a
clinician, and its confidence is better calibrated than the model's own." Mention what is pending: the team's
relevance labels and the reviewers' rubric scores.

**7. The same system from an AI assistant (30 seconds, optional).** In Claude Code connected to the server's `/mcp`
endpoint, ask: "Triage chest_pain_clinic.csv and explain the first case." The same tools, roles and audit trail apply.

## Questions examiners ask

- **How do you stop the language model from making things up?** It only receives the other agents' outputs, must
  cite a source for every sentence, and a deterministic check (TC-12) removes any sentence that cites a source the
  case lacks, cites nothing, states a number its sources do not contain, or names a symptom its sources never
  mention. The planned test TC-12 proves it with a stubbed reply.
- **What happens if the model or the API is down?** The template writes the summary and the report says why
  (`summary_written_by`). A case that fails analysis becomes a failure report instead of crashing the batch (NFR5).
- **Why logistic regression?** It beat random forest and gradient boosting in cross-validation on the training split,
  is well calibrated, and its SHAP explanations are exact. See the model card.
- **Why two submodels?** Routine records have no exercise test. Filling eight inputs with averages distorted the age
  effect, so a pre-test model scores those records and the full model scores records with a work-up.
- **Is the test result of 0.975 AUROC realistic?** The test split has 61 patients; cross-validation gives about 0.90,
  which is the better estimate. The external sites score lower (0.71 to 0.88) because many inputs are missing there.
- **Why were beta and lambda not tuned?** The UCI patients almost never trigger a rule override or a conflict, so the
  data cannot inform those weights; they stay at the design values and the tuning report says so.
- **Who can do what?** Roles come from each person's sign-in: only a Data Administrator uploads, only a Clinician
  reviews, only a Knowledge Curator approves knowledge updates. The audit trail records the signed-in name.
- **Is this ready for patients?** No. It is a research prototype on public and synthetic data, not a medical device,
  and its knowledge base is a draft awaiting review by the team and the guide.
