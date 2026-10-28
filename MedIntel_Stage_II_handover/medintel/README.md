# MedIntel

A multi-agent clinical decision support framework, served over the Model Context Protocol (MCP). This is the
Project Stage II implementation of the design in the Stage I report (BE Computer Engineering, JESITMR Nashik,
AY 2026-27).

> Research prototype. Use synthetic or de-identified data only. MedIntel is not a medical device, and a clinician
> makes every final decision.

## Status: all four phases done; evaluation results in [reports/evaluation](reports/evaluation/EVALUATION.md)

- **Phase 4 (evaluation):** the twelve planned test cases as named tests, and `uv run medintel evaluate`, which
  measures what Chapters 6 and 7 of the Stage I report promised and writes a report with figures. See
  [Evaluation](#evaluation-phase-4) and the [demo guide](docs/DEMO.md).
- **Phase 3:** one HTTP server with MCP over Streamable HTTP, a REST API and a clinician dashboard on the same core,
  sign-in with per-person tokens so roles come from identities, background jobs for long batches, and a Docker
  image. See [HTTP server](#http-server-phase-3).

Phase 2 made the agents real:

| Agent | Phase 2 |
|---|---|
| ML Analysis | Logistic regression trained and calibrated on UCI Heart Disease, compared with random forest and gradient boosting; exact SHAP explanations; a pre-test and a full submodel |
| Rule Validation | Versioned JSON rulebase R-001 to R-004 (unchanged) |
| Knowledge Retrieval | sentence-transformers embeddings over a sourced starter knowledge base (18 summaries of CDC, NIH and MedlinePlus pages) plus clinician-validated cases |
| Clinical Context | Claude (Opus 5.5) writes the summary in a fixed structure; the TC-12 check removes unsupported sentences; template fallback |
| Decision Fusion | Equation (4.2) with weights proposed from the validation split (report only, not applied) |

Phases 0 and 1 (the prototype port with golden tests, and the full MCP surface with governance) are unchanged and
still pass their tests. The stdio server (`uv run medintel-mcp`) works as before, for one person on one machine.

## Quick start

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is installed by uv if needed.

```bash
uv sync                          # install (first time: PyTorch CPU, sentence-transformers, scikit-learn, FAISS)
uv run medintel fetch-data       # download UCI Heart Disease (CC BY 4.0) into data/raw, checksums verified
uv run medintel train            # train both submodels; writes models/risk-lr-uci.json and models/MODEL_CARD.md
uv run pytest                    # 139 tests, no network and no API key needed
uv run medintel evaluate         # Stage II evaluation: reports/evaluation/EVALUATION.md and figures
uv run medintel-mcp              # start the MCP server on stdio (one person, this machine)
```

Or run the HTTP server, with sign-in, for several people:

```bash
uv run medintel users add "Dr. Rao" --role clinician --role data_admin --save-token ~/medintel-rao.token
uv run medintel users add "Ms. Iyer" --role curator --save-token ~/medintel-iyer.token
uv run medintel serve            # http://127.0.0.1:8000: dashboard /dashboard, MCP /mcp, API docs /docs
```

The first retrieval downloads the embedding model (all-MiniLM-L6-v2, about 90 MB, Apache 2.0) from Hugging Face
unless it is already cached. Without `models/risk-lr-uci.json` the server runs the prototype's demo weights, and
every report's model version says so.

For Claude-written summaries, put an API key in a `.env` file in this folder (it is git-ignored and read at
startup; variables already set in the environment win):

```
ANTHROPIC_API_KEY=...
```

Without a key, `writer = "claude"` falls back to the template and the report says `template (no ANTHROPIC_API_KEY)`.
Analyses, reviews, knowledge versions and the audit trail are stored in `~/.medintel/medintel.db`; set
`MEDINTEL_HOME` to use another folder. All settings are in `config/settings.toml`.

## Connecting a client

- **Claude Code:** open this folder in Claude Code and approve the `medintel` server listed in `.mcp.json` when
  asked. To add it for every folder instead:
  `claude mcp add medintel -- uv run --directory /full/path/to/medintel medintel-mcp`
- **Claude Desktop:** add to `claude_desktop_config.json`:

  ```json
  {"mcpServers": {"medintel": {"command": "uv", "args": ["run", "--directory", "/full/path/to/medintel", "medintel-mcp"]}}}
  ```

- **MCP Inspector:** `npx @modelcontextprotocol/inspector uv run medintel-mcp`

Then ask, for example: *"Validate chest_pain_clinic.csv, triage the batch and walk me through the first case."*

## HTTP server (Phase 3)

`uv run medintel serve` runs one ASGI app (FastAPI and the MCP SDK's Streamable HTTP transport) with four front
doors on the same core and database:

| Path | What |
|---|---|
| `/dashboard/` | Clinician dashboard: batches, triage queue, case view, what-if, review, knowledge updates, audit, model card |
| `/mcp` | MCP over Streamable HTTP (stateless, as the 2026-07-28 spec expects): the same 14 tools, 3 prompts and 6 resources as stdio |
| `/v1/...` | REST API; interactive docs at `/docs`, schema at `/openapi.json` |
| `/healthz` | Health check (no sign-in) |

**Sign-in and roles.** Every person gets a token (`uv run medintel users add NAME --role ROLE`), a random 256-bit
secret shown once or saved to a file readable only by you (`--save-token`). Only its SHA-256 is stored, in the same
database. Every `/mcp` and `/v1` request sends it as `Authorization: Bearer <token>`. The person's roles decide
what they may do (a clinician cannot approve knowledge updates: 403), and the audit trail records the signed-in
name, not a typed one. Manage people with `medintel users list | rotate NAME | disable NAME | enable NAME |
roles NAME --role ...`; a disabled or rotated token stops working at once.

| Role | May |
|---|---|
| Data Administrator | upload and validate batches, analyse |
| Clinician | analyse, record reviews (accept, annotate, reject) |
| Knowledge Curator | approve or decline knowledge updates |
| System Administrator | read everything (user management is on the server's command line) |

Everyone signed in can read reports, explanations, evidence, the audit trail and the model card.

`--no-auth` turns sign-in off for local use (roles come from `config/settings.toml`); it is refused unless the
server listens on 127.0.0.1.

**Background jobs.** A long batch runs as a job: `start_batch_analysis` (MCP) or `POST /v1/batches/{id}/jobs`
(REST) returns a `job_id` at once; `get_job_status` / `GET /v1/jobs/{id}` report progress and, when finished, the
triage queue; `cancel_job` stops it between cases. Job rows live in SQLite, and a job that was running when the
server stopped is marked `interrupted` on the next start. The MCP tasks extension is not used yet because clients
do not support it widely; polling works everywhere. `analyze_batch` (synchronous, with progress) is unchanged.

**Connecting Claude Code over HTTP:**

```bash
claude mcp add --transport http medintel http://127.0.0.1:8000/mcp \
  --header "Authorization: Bearer $(cat ~/medintel-rao.token)"
```

**Security measures:** bearer tokens hashed at rest; DNS-rebinding protection on `/mcp` (only this server's own
host names); a strict Content-Security-Policy on the dashboard (no inline scripts or styles; server data is always
inserted as text); `X-Content-Type-Options`, `Referrer-Policy` and `X-Frame-Options`; uploads capped at 2 MB; the
dashboard keeps its token in `sessionStorage`, cleared when the tab closes. Beyond localhost, put the server behind
HTTPS (see [Deploying](#deploying)); bearer tokens must never travel over plain HTTP on a network.

### REST API

| Method and path | Same as |
|---|---|
| `GET /v1/me` | who is signed in, their roles, server versions |
| `GET /v1/batches`, `POST /v1/batches` (sample or CSV text), `POST /v1/batches/upload` (multipart) | `validate_records` |
| `GET /v1/batches/{batch_id}` | `get_batch_status` |
| `POST /v1/batches/{batch_id}/jobs`, `GET /v1/jobs/{job_id}`, `POST /v1/jobs/{job_id}/cancel` | the job tools |
| `POST /v1/cases`, `GET /v1/cases/{case_id}` | `analyze_case`, the case report |
| `GET /v1/cases/{case_id}/explanation`, `POST /v1/cases/{case_id}/what-if` | `explain_decision`, `what_if` |
| `POST /v1/cases/{case_id}/reviews` | `record_review` |
| `GET /v1/evidence?query=` or `?case_id=` | `search_evidence` |
| `GET /v1/knowledge/updates`, `POST /v1/knowledge/updates/{proposal_id}/decision`, `GET /v1/knowledge/base` | knowledge tools and resource |
| `GET /v1/audit`, `GET /v1/model-card`, `GET /v1/schema/patient-record` | audit trail and resources |
| `GET /v1/evaluation`, `GET /v1/evaluation/figures/{name}` | the latest evaluation report and its figures |

Errors are JSON `{"detail": ...}` with 400 (fixable request), 401 (no or bad token), 403 (role), 404 (unknown ID)
or 413 (upload too large).

### Dashboard

Plain HTML, CSS and JavaScript in `src/medintel/dashboard/` (no build step), using the design tokens of the Stage I
concept prototype, with automatic dark mode and a phone layout. It is a client of the REST API only, so everything
it shows can also be fetched with `curl`. Verified with Playwright in Chrome at 1440 px, on an iPhone 15 viewport
and in dark mode: the whole flow (validate, background analysis, triage, case, what-if, review, curator approval,
audit, model card) ran with no console errors, no failed requests and no horizontal overflow. The Evaluation page
shows the latest `medintel evaluate` report with its figures.

## Deploying

The server is one process with SQLite, so it suits one machine (a laptop for a demo, or a small VM). Nothing has
been deployed publicly.

**Docker** (the image trains the model and caches the embedding model at build time, runs as a non-root user and
keeps its database in the `/data` volume):

```bash
docker build -t medintel .
docker run --rm -it -v medintel-data:/data medintel uv run --no-sync medintel users add "Dr. Rao" --role clinician
docker run -d --name medintel -p 8000:8000 -v medintel-data:/data --env-file .env \
  -e MEDINTEL_PUBLIC_URL=https://medintel.example.org -e MEDINTEL_BEHIND_PROXY=1 medintel
```

`.env` carries `ANTHROPIC_API_KEY` for Claude-written summaries (optional). The image needs about 1 GB of RAM.

**HTTPS:** run a reverse proxy in front, for example Caddy (`caddy reverse-proxy --from medintel.example.org --to
127.0.0.1:8000`, which obtains a certificate automatically), and set `MEDINTEL_PUBLIC_URL` to the public address so
`/mcp` accepts that host name.

**On a classroom network without Docker:** `uv run medintel serve --host 0.0.0.0 --public-url
http://<laptop-ip>:8000`. Tokens then travel unencrypted on that network, so use it only on a trusted network with
synthetic data.

**Hosting options to choose from:** a small cloud VM (2 vCPUs, 4 GB) with Docker and Caddy; Hugging Face Spaces with
the Docker SDK (make the Space private); Render or Fly.io. Serverless platforms such as Vercel are not a fit
(long-running process, PyTorch, SQLite). Moving to several instances would need Postgres instead of SQLite.

## The agents

### ML Analysis (risk model)

- **Data:** UCI Heart Disease, Cleveland subset (303 patients), split 60/20/20 into train, validation and test with
  seed 2026. Outcome: angiographic coronary artery disease (over 50% narrowing).
- **Model selection:** logistic regression, random forest and gradient boosting compared by repeated
  cross-validation on the training split. Logistic regression won on log loss, AUROC and Brier score.
- **Calibration:** Platt scaling fitted on out-of-fold scores, so the validation split stays free for fusion tuning.
- **Explanations:** inputs are centered on the training data, which makes each feature's term its exact SHAP value.
  A value that was not recorded is set to the training average, contributes nothing, and is listed as filled in.
- **Two submodels:** filling in many correlated inputs distorts the rest (without peak heart rate, the full model's
  adjusted age term points the wrong way). The **pre-test model** (age, sex, chest pain type, systolic BP,
  cholesterol, fasting glucose above 120 mg/dL, resting ECG) scores records without an exercise test, such as
  routine check-ups. The **full model** (all 13 UCI inputs) scores records with one.

| Held-out results | Full model | Pre-test model |
|---|---|---|
| Cross-validated AUROC (training split) | 0.90 (SD 0.05) | 0.81 |
| Validation AUROC | 0.87 | 0.80 |
| Test AUROC (n = 61, 95% CI) | 0.975 (0.94 to 0.997) | 0.92 |
| Test Brier score | 0.071 | 0.122 |
| Test expected calibration error | 0.100 | 0.145 |

The test split happened to be easier than average; the cross-validation figures are the better guide. On the other
UCI sites the routed model reaches AUROC 0.88 (Hungary, 20% of inputs missing), 0.72 (Switzerland) and 0.71 (Long
Beach VA). The full model card, with coefficients, reliability tables and limitations, is written to
`models/MODEL_CARD.md` and served as the `medintel://model-card` resource.

### Knowledge Retrieval

- **Knowledge base:** `src/medintel/resources/knowledge_base.json`, 18 short summaries of public-domain CDC, NIH
  (NHLBI, NIDDK) and MedlinePlus pages, each with its URL and retrieval date. Every number in a passage was checked
  against its page. MedlinePlus encyclopedia pages were left out because they are copyrighted by A.D.A.M.
  **Status: draft.** The team and Prof. Mohole should review the medical content (see the review list below).
- **Search:** one query per notable finding, plus the clinical notes. Queries name the clinical category a value
  falls in (for example "fasting blood sugar of 126 mg/dL or above, in the diabetes range"), because embeddings do
  not compare numbers. Each passage keeps its best cosine score and the finding that matched it (`matched_on`).
- **Thresholds:** `min_score = 0.55` drops weak matches (TC-06: no relevant evidence gives K = 0). Set from the
  sample cases; tune it with the team's relevance labels (precision at 5).
- **Conflicts:** cosine similarity says nothing about severity, so evidence can only raise an evidence-versus-model
  conflict when it was matched by a *marked* finding: one past a threshold stated in the knowledge base or
  rulebase (BP at or above 180/120 mmHg, fasting glucose of 126 mg/dL or above, resting heart rate above 100 bpm,
  current smoking, typical or exercise-induced angina, a reversible thallium defect, narrowed vessels), or a fully
  normal profile for low-risk evidence. Marked evidence also needs `strong_score = 0.6`.
- **Validated cases:** an outcome joins the index as `KB-CASE-<number>` only after the Knowledge Curator approves it,
  and a case never retrieves its own outcome. The index version (kb-vN) moves with every approval and with every
  change of the knowledge base file.
- **Index:** an exact inner-product search in numpy, the same computation as FAISS `IndexFlatIP`. Set
  `index = "faiss"` to use FAISS, on Linux only: the macOS and Windows wheels of FAISS, PyTorch and scikit-learn
  each bundle their own OpenMP runtime, and a process that loads two of them can abort with `OMP: Error #15`. A test
  checks that both indexes return the same results.

### Clinical Context (Claude writer)

- Claude receives only the structured outputs of the other agents (record values, model estimate and drivers,
  fired rules, evidence passages, detected conflicts) and the list of sources it may cite.
- Structured output (`output_config.format`): 3 to 6 sentences, each with the sources it relies on. Effort `low` by
  default; Opus 5.5 always thinks, and effort sets how much.
- The TC-12 check then removes every sentence that cites a source the case does not contain, cites nothing, states a
  number its sources do not hold, or names a symptom its sources never mention. A clause saying a field was not
  recorded may name that field ("chest pain type was not recorded" is not a claim of chest pain). Claude's word is
  never taken on its own.
- Any failure (no key, network, refusal, malformed reply) falls back to the template, and `summary_written_by`
  says why. Tokens already spent are still counted.
- Each summary records its tokens and estimated cost (`summary_cost_usd`, list prices); `explain_decision` shows
  the writer, the prompt version and the usage. What-if explorations always use the template, so they cost nothing.
- **Privacy:** with the Claude writer, case facts (including free-text notes) are sent to the Anthropic API. Use
  synthetic or de-identified records only.

### Decision Fusion

`uv run medintel tune-fusion` searches alpha, beta, gamma, lambda and the threshold on the validation split,
minimizing the Brier score of C against "the decision was right", and scores the test split once. UCI patients
almost never trigger a rule override or a rule conflict, so beta and lambda are held at their design values (the
report explains why and shows the unconstrained optimum too). With the embedding retrieval:

| Settings | Test Brier(C) | Confident cases | Accuracy when confident |
|---|---|---|---|
| Current (0.5, 0.3, 0.2, 0.5, threshold 0.6) | 0.086 | 98% | 88% |
| Proposed (0.65, 0.3, 0.05, 0.5, threshold 0.74) | 0.080 | 79% | 96% |

Retrieval relevance K says almost nothing about whether a decision is right on this data (the unconstrained
optimum sets gamma to 0). The proposal is written to `reports/fusion_tuning.json` with a threshold trade-off table
and is **not** applied: choosing coverage against accuracy is the team's decision.

## Evaluation (Phase 4)

`uv run medintel evaluate` runs the configured agents (trained model, embedding retrieval, template summaries) on
the held-out UCI Cleveland test split, the validation split where the plan says so, and the 24 bundled sample
cases, then writes `reports/evaluation/EVALUATION.md`, `evaluation.json` and four SVG figures. It takes about 20
seconds. As the plan requires, the fusion weights and threshold are tuned on the validation split and judged on the
test split; the illustrative design weights are reported alongside.

| Metric (Table 7.1) | Target | Result |
|---|---|---|
| F1 | Full pipeline above ML only | 0.955 vs 0.873; the full pipeline decides 79% of cases and refers the rest |
| ROC-AUC | Not below ML only | 0.975 vs 0.975 |
| Calibration error | Fused C no worse than the ML confidence | ECE 0.075 vs 0.088 |
| Report completeness | 100% pass the schema | 100% of 146 reports |
| End-to-end latency | 30 s per case; ML and rules under 1 s | worst case 0.13 s with template summaries; a Claude summary adds 6.4 s on average, 7.2 s at most (24 live cases) |
| Planned test cases | All twelve pass | 12 of 12 |
| Evidence precision at 5 | Reported, team labels | pending the team's labels |
| Explanation rubric | Mean of at least 4 per criterion | pending reviewer scores |
| Per-case cost | Reported | about $0.023 per case live, $0.011 with the Batch API (Opus 5.5, effort low) |

The report also covers the four conditions with every metric, the one-at-a-time ablation (without ML the pipeline
decides nothing; without retrieval calibration slightly improves, so K carries little signal on this data),
reliability tables, disagreement rates, rule false alerts, stage latency, cold start and throughput. Read it with
its caveats: 61 test patients, and UCI cases rarely trigger rules or conflicts.

**Planned test cases.** Each case of Table 6.4 is a named test in `tests/test_planned_cases.py`, and `medintel
evaluate` records whether each passes:

| Case | What it checks | Test |
|---|---|---|
| TC-01 | A record without a patient ID is rejected | `test_tc01_record_without_patient_id_is_rejected` |
| TC-02 | A Fahrenheit temperature is held, not converted | `test_tc02_fahrenheit_temperature_is_held_not_converted` |
| TC-03 | R-001 fires on a systolic BP of 182 | `test_tc03_r001_fires_on_high_systolic_bp` |
| TC-04 | The ML output contract | `test_tc04_ml_output_contract` |
| TC-05 | Retrieval returns the top five in order | `test_tc05_retrieval_returns_top_k_in_order` |
| TC-06 | Empty retrieval is stated and K = 0 | `test_tc06_empty_retrieval_is_stated` |
| TC-07 | ML and rule disagreement raises D and lowers C | `test_tc07_ml_rule_disagreement_lowers_confidence` |
| TC-08 | Full agreement gives D = 0 and C above the threshold | `test_tc08_full_agreement` |
| TC-09 | Parallel runs keep patients isolated | `test_tc09_parallel_runs_keep_patients_isolated` |
| TC-10 | Report, review and audit end to end over REST | `test_tc10_end_to_end_report_review_and_audit` |
| TC-11 | Per-case latency targets | `test_tc11_per_case_latency_targets` |
| TC-12 | An unsupported statement is removed and logged | `test_tc12_unsupported_statement_is_removed` |

TC-12 asks that a summary naming a symptom absent from the case be caught, so the check now also compares every
symptom a sentence names (from a small lexicon of synonyms, such as fainting and syncope) with the text of its cited
sources. Every case also records the time spent in each stage (`timings` in `explain_decision`), as Section 6.5.1
plans.

**The two human judgements.** `uv run medintel export-labels` writes `reports/labeling/evidence_relevance.csv`
(each sample case's top five passages, with the case context) and `explanation_rubric.csv` (a fixed mix of
conflicting, low-confidence and clear reports). Mark relevance 1 or 0, score each report 1 to 5 for completeness,
factual consistency and usefulness, time the reading, and run `medintel evaluate` again: precision at 5 and the
rubric means join the report. Existing sheets are never overwritten without `--force`.

## Commands

| Command | What it does |
|---|---|
| `uv run medintel fetch-data` | Download UCI Heart Disease and check its SHA-256 checksums |
| `uv run medintel train` | Train, calibrate and evaluate both submodels; write the model file and model card |
| `uv run medintel kb-check` | Load the knowledge base and embedding model; show the evidence each sample case retrieves |
| `uv run medintel tune-fusion` | Propose fusion settings from the validation split (report only) |
| `uv run medintel evaluate` | Run the Stage II evaluation and write `reports/evaluation/` (report, JSON, figures) |
| `uv run medintel export-labels` | Write the evidence-relevance and explanation-rubric sheets for the team |
| `uv run medintel eval-writer` | Run the Claude writer on evaluation cases and measure TC-12 removals, tokens, cost and latency. Prints an upper-bound cost and sends nothing without `--yes`. `--mode batch` (default) uses the Message Batches API at half price and writes `reports/writer_eval.json`; `--mode live` times each case and writes `reports/writer_eval_live.json`. `evaluate` reads both |

## MCP tools, prompts and resources

| Tool | Role | What it does |
|---|---|---|
| `validate_records` | Data Administrator | Checks a CSV batch (or a bundled sample: `morning_batch.csv`, `boundary_tests.csv`, `chest_pain_clinic.csv`) before any agent runs. Returns a `batch_id`. |
| `analyze_batch` | Clinician or Data Administrator | Analyzes every valid patient in parallel, reporting progress, and returns the triage queue and the summary cost. |
| `start_batch_analysis` | Clinician or Data Administrator | Starts a background job and returns its `job_id` at once (for long batches). |
| `get_job_status` | any | Job progress, and the triage queue once it has finished. |
| `cancel_job` | Clinician or Data Administrator | Stops a job between cases. |
| `analyze_case` | Clinician or Data Administrator | Runs the pipeline for one patient (from a batch, or one record with optional cardiology fields). |
| `get_batch_status` | any | Analyzed cases, reviews still pending, and the queue. |
| `explain_decision` | any | M, R, K, D, weights, threshold, rule precedence, evidence, conflicts, removed statements, the submodel with its largest contributions and filled-in inputs, and who wrote the summary. |
| `what_if` | any | Recomputes a case with changed values or weights and lists what changed. Nothing is saved; no paid calls. |
| `search_evidence` | any | Searches the knowledge base by meaning, or shows the evidence a case matches. |
| `record_review` | Clinician | Accept, annotate or reject a report. Accepted high- or low-risk outcomes become proposals for the curator. |
| `list_knowledge_updates` | any | Proposals, the current index version and its history. |
| `approve_knowledge_update` | Knowledge Curator | Approve (new index version) or decline a proposal. |
| `get_audit_trail` | any | Audit records, newest first, including who did what. |

**Prompts:** `triage_batch`, `review_case` and `curator_review`.
**Resources:** `medintel://rulebase`, `medintel://schema/decision-report`, `medintel://schema/patient-record`,
`medintel://model-card`, `medintel://knowledge-base` and `medintel://cases/{case_id}/report`.

Over stdio, roles come from `[access] roles` in `config/settings.toml` or `MEDINTEL_ROLES`; over HTTP they come
from the signed-in identity. An accepted outcome becomes a proposal (`KU-001`) that only a curator can approve; the
audit table is append-only and records `event` and `actor`.

## Patient records

The 12 prototype fields (age, sex, systolic and diastolic BP, heart rate, temperature, cholesterol, fasting glucose,
creatinine, BMI, smoking, notes) plus 8 optional cardiology fields read by the full model: `chest_pain_type`
(typical_angina, atypical_angina, non_anginal, asymptomatic), `resting_ecg` (normal, st_t_abnormality,
lv_hypertrophy), `max_heart_rate`, `exercise_angina`, `st_depression`, `st_slope` (upsloping, flat, downsloping),
`major_vessels` (0 to 3) and `thal` (normal, fixed_defect, reversible_defect). Old CSVs stay valid. The full list
with units and ranges is the `medintel://schema/patient-record` resource.

## For the team to review

- **Knowledge base (draft):** every passage, with Prof. Mohole. Known gaps: no allowed page gives a "high" total
  cholesterol cut-off (KB-LIP-001 gives healthy levels only); none describes left ventricular hypertrophy on an ECG
  (KB-ECG-001 covers a thickened heart muscle from high blood pressure); the stress-test pages do not name thallium
  or describe specific ECG changes; the pediatric page covers children under 13 only; KB-CKD-001's source page was
  last reviewed in 2016.
- **Marked-finding thresholds** in `agents/embedding_store.py` (`case_queries`): confirm each one and its source.
- **Retrieval thresholds:** label about 50 evidence-relevance examples and tune `min_score` for precision at 5.
- **Fusion settings:** read `reports/fusion_tuning.json` and choose a threshold.
- **Model caveats** (model card): referral population with a 46% disease rate; in this data asymptomatic patients had
  the highest disease rate, so typical angina gets a negative contribution; not validated under 18.

## How the code maps to the report

| Report | Code |
|---|---|
| Table 4.2.1 output contracts, Section 4.4.2 report | `src/medintel/models.py`, `resources/decision_report.schema.json` |
| 5.2.1 DataIngestionService, SchemaValidator | `src/medintel/ingestion.py`, `src/medintel/schema.py` |
| 5.2.2 Orchestrator, CaseScheduler | `src/medintel/orchestrator.py` |
| 5.3.1 MLAnalysisAgent | `src/medintel/agents/ml_agent.py`, `agents/risk_model.py`, `training/train.py` |
| 5.3.2 RuleValidationAgent, RuleBase (4.4.1) | `src/medintel/agents/rule_agent.py`, `resources/rulebase.json` |
| 5.4.1 KnowledgeRetrievalAgent, KnowledgeStore, add_validated | `agents/embedding_store.py`, `agents/retrieval_agent.py`, `knowledge.py` |
| FR6 ClinicalContextAgent, LLM Service | `agents/context_agent.py`, `agents/claude_writer.py` |
| 5.4.2 DecisionFusionEngine (Equation 4.2), ExplanationGenerator | `src/medintel/fusion.py`, `training/fusion_tuning.py` |
| 5.5.1 AuditLogger, export_report | `src/medintel/audit.py` |
| 5.5.2 clinician review | `src/medintel/service.py` (`review`), `record_review` tool |
| Settings (5.2.1) | `config/settings.toml`, `src/medintel/config.py` |
| Chapters 6 and 7 evaluation | `training/evaluate.py`, `training/metrics.py`, `training/figures.py`, `training/labels.py`, `training/writer_eval.py` |
| Table 6.4 planned test cases | `tests/test_planned_cases.py` |
| FastAPI service, dashboard, hosted MCP (3.4) | `src/medintel/web.py`, `src/medintel/dashboard/`, `Dockerfile` |
| User roles (3.1), sign-in | `src/medintel/identity.py`, `cli.py users` |
| Batch scheduling at scale | `src/medintel/jobs.py` |

## What the tests cover

TC-01, TC-02, TC-06 and TC-12 (including on Claude's sentences), the R-001 boundary, rule precedence, Equation (4.2),
the report schema, the append-only audit log, isolated parallel cases, failure reports, triage, review and approval
with index versioning, roles, what-if, prompts, database upgrades, and the MCP server in-process and over stdio.
Phase 2 adds: the UCI parser, the encoder, exact SHAP additivity, filled-in inputs contributing nothing, submodel
routing, the model file round trip, retrieval with a stand-in encoder, marked findings, validated cases joining the
index only after approval, FAISS and numpy agreeing, the Claude writer with a stand-in client (structured replies,
every fallback, cost accounting, no paid calls from what-if), fusion tuning, the Batch API evaluation path and the
spending guard. Phase 3 adds: tokens shown once and stored hashed, rotation and disabling, background jobs
(completion, cancellation, recovery after a restart), the REST flow with 401, 403, 404 and 413, CSV upload, MCP over
Streamable HTTP with the signed-in identity as the actor, bad tokens and foreign host names refused, local no-auth
mode, the dashboard's security headers, and the CLI's token files and safety checks. Phase 4 adds the twelve planned
test cases, symptom grounding in the TC-12 check (and statements that a field was not recorded), stage timings, the decision rules of every condition and ablation,
ROC-AUC against scikit-learn, the labelling sheets, and a full `evaluate` run on a synthetic UCI-format dataset. No
test needs the network or an API key.

## Data and licences

- UCI Heart Disease: Janosi, A., Steinbrunn, W., Pfisterer, M. and Detrano, R. (1989). Heart Disease [Dataset].
  UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X. CC BY 4.0. Downloaded by `fetch-data`, never
  committed (`data/raw/` is git-ignored, as are trained model files).
- Knowledge base: summaries of US federal government pages (public domain), each with its source.
- Embedding model: sentence-transformers/all-MiniLM-L6-v2, Apache 2.0.

## What is left for the team

1. **Label and score:** fill in the two sheets from `medintel export-labels`, then run `medintel evaluate` again.
2. **Review the knowledge base** with Prof. Mohole (see "For the team to review").
3. **Choose the operating point:** adopt the tuned fusion settings from `reports/fusion_tuning.json`, or pick a
   threshold from its coverage table, and record the choice in `config/settings.toml`.
4. **Claude-written summaries:** measured (section 9 of the evaluation): no fallbacks in 194 replies, and TC-12 removed
   0.7% of sentences, each a conflict sentence that calls KB-ANG-001 "the angina evidence" without citing it. A prompt
   line asking the writer to cite any passage it describes would avoid that. The rubric sheet already holds Claude's
   summaries for the reviewers to score. Each person running the server needs their own API key in `.env`.
5. **Hosting:** pick where to deploy (see [Deploying](#deploying)); nothing is public yet.
6. **Optional:** Synthea synthetic patients for a FHIR input path (needs Java or a sample download).

## Regenerating the golden file

The expected outputs in `tests/golden/prototype_golden.json` come from the concept prototype's engine:

```bash
node tools/export_prototype_golden.js <prototype src folder> src/medintel/samples > tests/golden/prototype_golden.json
```

## Team

Siddhesh Kishor Suryawanshi, Shantanu Avinash Patil, Omkar Abaji Musale, Tanmay Shriniwas Bagul.
Guide: Prof. G. P Mohole, Department of Computer Engineering, JESITMR, Nashik.
