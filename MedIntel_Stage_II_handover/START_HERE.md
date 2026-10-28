# MedIntel Stage II: start here

## What is in this folder

- `MedIntel_walkthrough_1440p.mp4`: a captioned walkthrough of the whole flow (about 4.5 minutes). Watch it first.
- `medintel/`: the Stage II system. The MCP server, REST API, clinician dashboard and evaluation all run on the same
  core. Its `README.md` covers the design, `docs/DEMO.md` is an 8-minute demo script with likely examiner
  questions, and `reports/evaluation/EVALUATION.md` has the results.

## Run it

You need [uv](https://docs.astral.sh/uv/). Then, from this folder:

```bash
cd medintel
uv sync
uv run medintel fetch-data        # UCI Heart Disease (CC BY 4.0), checksums verified
uv run medintel train             # trains the risk model and writes models/MODEL_CARD.md
uv run medintel users add "Dr. Rao" --role clinician --role data_admin --save-token ~/rao.token
uv run medintel users add "Ms. Iyer" --role curator --save-token ~/iyer.token
uv run medintel serve
```

Open http://127.0.0.1:8000/dashboard/ and sign in by pasting the contents of `~/rao.token`. Keep token files
outside the project folder; they are passwords.

After `train`, `uv run pytest` runs all 139 tests (before it, the one that checks the trained model is skipped). The
tests need no API key.

## Claude-written summaries (optional)

Create your own API key at https://platform.claude.com and put `ANTHROPIC_API_KEY=...` in `medintel/.env` (the
file is git-ignored). Without a key, the built-in template writes the summaries and every report says so. With
Claude, a case costs about $0.02. Only send synthetic or de-identified records.

## Where things stand

- Done: all four phases. A risk model trained on UCI Heart Disease, embedding retrieval over a sourced knowledge
  base, the Claude writer with the TC-12 check, sign-in with roles, the dashboard, background jobs, MCP access for AI
  assistants, and the evaluation for Chapters 6 and 7 (all twelve planned test cases pass).
- For the team: label the two sheets in `reports/labeling/` and run `uv run medintel evaluate` again; review the
  knowledge base drafts with the guide; choose the fusion operating point (`reports/fusion_tuning.json`); decide
  where to host it. The README section "What is left for the team" has the details.
- Not yet a git repository. Run `git init` and commit; `.gitignore` already excludes keys, tokens, downloaded data
  and trained model files.

## Notes

- Training is reproducible: retraining gives identical weights, and only the creation time (so the model's
  fingerprint, such as `c89c22e9`) differs. Run `uv run medintel evaluate` to regenerate the evaluation report
  for your own copy.
- Tested on macOS (Apple Silicon). The `Dockerfile` builds a Linux image. Not tested on Windows.
- Research prototype on synthetic and public de-identified data. Not a medical device.
