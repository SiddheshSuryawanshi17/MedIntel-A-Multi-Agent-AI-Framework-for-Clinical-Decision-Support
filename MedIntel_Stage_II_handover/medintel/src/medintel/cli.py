"""medintel command line: offline steps that prepare the agents (download, train, index checks, tuning, evaluation)."""

import argparse
import json
import os
import sys
from pathlib import Path

from medintel.config import load_env_file, load_settings, project_env_file

DATA_DIR = Path("data/raw/uci-heart-disease")


def fetch_data(args: argparse.Namespace) -> None:
    from medintel.training import uci

    hashes = uci.fetch(Path(args.dest))
    print(f"Saved UCI Heart Disease ({uci.LICENSE}) to {args.dest}")
    for name, digest in sorted(hashes.items()):
        print(f"  {name}  sha256 {digest[:16]}")
    print(f"Cite as: {uci.CITATION}")


def train_model(args: argparse.Namespace) -> None:
    from medintel.training.train import train

    result = train(Path(args.data), Path(args.out), args.seed)
    ev = result["metadata"]["evaluation"]
    print(f"Wrote {result['path']} ({result['version']}) and {Path(args.out) / 'MODEL_CARD.md'}")
    for name in ("full", "pre_test"):
        test = ev[name]["test"]
        print(f"  {name}: test AUROC {test['auroc']}, Brier {test['brier']}, ECE {test['ece']} (n = {test['n']})")


def tune_fusion(args: argparse.Namespace) -> None:
    from medintel.service import build_knowledge_store
    from medintel.training.fusion_tuning import settings_block, tune

    base, _, _ = build_knowledge_store(args.settings)
    report = tune(args.settings, Path(args.data), base, args.target)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump(report) + "\n", encoding="utf-8")
    test = report["test"]
    print(f"Wrote {out}. Retrieval: {report['retrieval_backend']}. Not applied to config/settings.toml.")
    for reason in report["held_at_design_values"].values():
        print(f"  Held at the design value: {reason}.")
    for name in ("current", "tuned"):
        r = test[name]
        print(
            f"  test, {name}: Brier(C) {r['brier_C']}, coverage {r['coverage']}, "
            f"accuracy when confident {r['accuracy_when_confident']}"
        )
    print("Proposed settings:\n" + settings_block(report["tuned"]))


def eval_writer(args: argparse.Namespace) -> None:
    import os

    import anthropic

    from medintel.agents.claude_writer import ClaudeWriter
    from medintel.service import build_knowledge_store
    from medintel.training import writer_eval as ev

    cases = []
    if args.cases in ("samples", "all"):
        cases += ev.sample_cases()
    if args.cases in ("uci-test", "all"):
        cases += ev.uci_test_cases(Path(args.data))
    cases = cases[: args.limit] if args.limit else cases
    base, _, _ = build_knowledge_store(args.settings)
    analyses = ev.analyse(args.settings, cases, base)
    client = anthropic.Anthropic(max_retries=2, timeout=120) if os.environ.get("ANTHROPIC_API_KEY") else None
    writers = [
        ClaudeWriter(model, effort, args.settings.context.max_tokens, client=client)
        for model in args.models.split(",")
        for effort in args.efforts.split(",")
    ]
    estimate = ev.plan(writers, analyses)
    half = " (the Batch API halves this)" if args.mode == "batch" else ""
    print(f"{estimate['requests']} requests; at most about ${estimate['max_cost_usd_standard']}{half}.")
    if not args.yes:
        print("Nothing sent. Add --yes to run.")
        return
    if client is None:
        raise ValueError("ANTHROPIC_API_KEY is not set (put it in .env or the environment)")
    rows = ev.run_batch(client, writers, analyses) if args.mode == "batch" else ev.run_live(writers, analyses)
    report = {"mode": args.mode, "cases": args.cases, "n_cases": len(analyses), "results": ev.summarize(rows)}
    out = Path(args.out or ("reports/writer_eval.json" if args.mode == "batch" else "reports/writer_eval_live.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump(report) + "\n", encoding="utf-8")
    detail = out.with_suffix(".jsonl")
    detail.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {out} and {detail}")
    print(dump(report["results"]))


def kb_check(args: argparse.Namespace) -> None:
    import time

    from medintel.agents.embedding_store import EmbeddingKnowledgeStore, KnowledgeBase, SentenceTransformerEncoder
    from medintel.training.writer_eval import sample_cases

    r = args.settings.retrieval
    kb = KnowledgeBase.load(r.knowledge_base or None)
    print(f"{kb.label}: {len(kb.passages)} passages, review status {kb.review.get('status', 'n/a')}")
    unsourced = [p.source_id for p in kb.passages if not p.url]
    if unsourced:
        print(f"  without a source URL: {', '.join(unsourced)}")
    store = EmbeddingKnowledgeStore(kb, SentenceTransformerEncoder(r.embedding_model))
    started = time.monotonic()
    store.search_text("blood pressure", 1)
    print(f"Embedding model {r.embedding_model} and index ready in {time.monotonic() - started:.1f} s")
    print(f"Sample cases (min_score {r.min_score}, strong_score {r.strong_score}; * = kept):")
    for case in sample_cases():
        hits = store.search(case, args.k, 0.0)
        shown = ", ".join(f"{h.source_id} {h.score:.2f}{'*' if h.score >= r.min_score else ''}" for h in hits)
        print(f"  {case.patient_id}: {shown}")


def evaluate_cmd(args: argparse.Namespace) -> None:
    from medintel.service import build_knowledge_store
    from medintel.training.evaluate import evaluate, summary_rows

    out = Path(args.out)
    result = evaluate(
        args.settings,
        Path(args.data),
        lambda: build_knowledge_store(args.settings)[0],
        out,
        Path.cwd(),
        Path(args.labels),
        Path(args.writer_report),
    )
    print(f"Wrote {out / 'EVALUATION.md'}, {out / 'evaluation.json'} and {out / 'figures'}/")
    for metric, _, value, met in summary_rows(result):
        print(f"  [{met}] {metric}: {value}")


def export_labels(args: argparse.Namespace) -> None:
    from medintel.agents.claude_writer import create_writer
    from medintel.orchestrator import create_orchestrator
    from medintel.service import build_knowledge_store
    from medintel.training.labels import (
        RELEVANCE_COLUMNS,
        RUBRIC_COLUMNS,
        relevance_rows,
        rubric_rows,
        write_sheet,
    )
    from medintel.training.writer_eval import sample_cases

    store = build_knowledge_store(args.settings)[0]
    writer = create_writer(args.settings.context)
    orchestrator = create_orchestrator(args.settings, writer, None, store)
    analyses = [orchestrator.run_case(c) for c in sample_cases()]
    out = Path(args.out)
    for name, columns, rows in (
        ("evidence_relevance.csv", RELEVANCE_COLUMNS, relevance_rows(analyses)),
        ("explanation_rubric.csv", RUBRIC_COLUMNS, rubric_rows(analyses)),
    ):
        if write_sheet(out / name, columns, rows, args.force):
            print(f"Wrote {out / name} ({len(rows)} rows)")
        else:
            print(f"Kept {out / name}: it already exists and may hold labels (use --force to replace it)")


LOOPBACK = ("127.0.0.1", "localhost", "::1")


def user_store():  # noqa: ANN201
    from medintel.config import medintel_home
    from medintel.identity import UserStore

    return UserStore(medintel_home() / "medintel.db")


def hand_over(name: str, token: str, save: str | None) -> None:
    if save:
        path = Path(save).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(token + "\n", encoding="utf-8")
        path.chmod(0o600)
        print(f"Token for {name} saved to {path} (readable only by you). It is not stored anywhere else.")
    else:
        print(f"Token for {name}, shown once (only its hash is stored):\n{token}")


def users(args: argparse.Namespace) -> None:
    from medintel.identity import UserError

    store = user_store()
    try:
        if args.action == "add":
            hand_over(args.name, store.add(args.name, args.role or []), args.save_token)
        elif args.action == "rotate":
            hand_over(args.name, store.rotate(args.name), args.save_token)
        elif args.action in ("disable", "enable"):
            store.set_disabled(args.name, args.action == "disable")
            print(f"{args.name} {args.action}d.")
        elif args.action == "roles":
            store.set_roles(args.name, args.role or [])
            print(f"{args.name}: {', '.join(args.role)}")
        else:
            rows = store.list()
            if not rows:
                print('No users yet. Add one: uv run medintel users add "Your Name" --role clinician')
            name_w = max([24, *(len(u.name) for u in rows)])
            roles_w = max([40, *(len(", ".join(u.roles)) for u in rows)])
            for u in rows:
                state = "disabled" if u.disabled else f"last used {u.last_used_at or 'never'}"
                print(f"{u.name:{name_w}s}  {', '.join(u.roles):{roles_w}s}  {state}")
    except UserError as exc:
        raise ValueError(str(exc)) from exc


def serve(args: argparse.Namespace) -> None:
    import uvicorn

    from medintel.service import MedIntelApp
    from medintel.web import create_app

    if args.no_auth and args.host not in LOOPBACK:
        raise ValueError("--no-auth is only allowed when listening on 127.0.0.1")
    app = MedIntelApp.from_env()
    if not args.no_auth and app.users.count() == 0:
        raise ValueError('no users yet. Create one first: uv run medintel users add "Your Name" --role clinician')
    shown = "localhost" if args.host in ("0.0.0.0", "::") else args.host
    public_url = (args.public_url or f"http://{shown}:{args.port}").rstrip("/")
    print(f"MedIntel on {public_url}  dashboard {public_url}/dashboard/  MCP {public_url}/mcp  API {public_url}/docs")
    if args.no_auth:
        print("Sign-in is OFF (local use only): every request acts with the roles in config/settings.toml.")
    uvicorn.run(
        create_app(app, public_url, auth=not args.no_auth),
        host=args.host,
        port=args.port,
        proxy_headers=args.behind_proxy,
        forwarded_allow_ips="*" if args.behind_proxy else None,
        log_level="info",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="medintel", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch-data", help="Download UCI Heart Disease (CC BY 4.0) and check its checksums.")
    p.add_argument("--dest", default=str(DATA_DIR))
    p.set_defaults(func=fetch_data)

    p = sub.add_parser("train", help="Train, calibrate and document the risk model.")
    p.add_argument("--data", default=str(DATA_DIR))
    p.add_argument("--out", default="models")
    p.add_argument("--seed", type=int, default=2026)
    p.set_defaults(func=train_model)

    p = sub.add_parser("tune-fusion", help="Tune the Equation (4.2) weights on the validation split (report only).")
    p.add_argument("--data", default=str(DATA_DIR))
    p.add_argument("--target", type=float, default=0.85, help="Accuracy wanted for cases with C at the threshold.")
    p.add_argument("--out", default="reports/fusion_tuning.json")
    p.set_defaults(func=tune_fusion)

    p = sub.add_parser("eval-writer", help="Run the Claude writer on evaluation cases (paid; needs --yes).")
    p.add_argument("--cases", choices=("samples", "uci-test", "all"), default="samples")
    p.add_argument("--data", default=str(DATA_DIR))
    p.add_argument("--models", default="claude-opus-5-5")
    p.add_argument("--efforts", default="low")
    p.add_argument("--mode", choices=("batch", "live"), default="batch")
    p.add_argument("--limit", type=int, default=0, help="Use only the first N cases.")
    p.add_argument(
        "--out", help="Default: reports/writer_eval.json, or reports/writer_eval_live.json with --mode live."
    )
    p.add_argument("--yes", action="store_true", help="Send the requests (they are billed).")
    p.set_defaults(func=eval_writer)

    p = sub.add_parser("kb-check", help="Load the knowledge base and embedding model; show retrieval on the samples.")
    p.add_argument("--k", type=int, default=5)
    p.set_defaults(func=kb_check)

    p = sub.add_parser("evaluate", help="Run the Stage II evaluation (Chapters 6 and 7) and write a report.")
    p.add_argument("--data", default=str(DATA_DIR))
    p.add_argument("--out", default="reports/evaluation")
    p.add_argument("--labels", default="reports/labeling", help="Folder with the filled-in labelling sheets.")
    p.add_argument("--writer-report", default="reports/writer_eval.json")
    p.set_defaults(func=evaluate_cmd)

    p = sub.add_parser("export-labels", help="Write the evidence-relevance and explanation-rubric sheets.")
    p.add_argument("--out", default="reports/labeling")
    p.add_argument("--force", action="store_true", help="Replace sheets that already exist.")
    p.set_defaults(func=export_labels)

    p = sub.add_parser("users", help="Manage the people who may sign in to the HTTP server, and their roles.")
    p.add_argument("action", choices=("add", "list", "rotate", "disable", "enable", "roles"))
    p.add_argument("name", nargs="?", default="")
    p.add_argument("--role", action="append", help="data_admin, clinician, curator or sys_admin (repeatable).")
    p.add_argument("--save-token", help="Write the new token to this file (mode 600) instead of printing it.")
    p.set_defaults(func=users)

    p = sub.add_parser("serve", help="Run the HTTP server: MCP at /mcp, REST at /v1, dashboard at /dashboard.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--public-url",
        default=os.environ.get("MEDINTEL_PUBLIC_URL"),
        help="The address people use, e.g. https://medintel.example.org behind TLS (env MEDINTEL_PUBLIC_URL).",
    )
    p.add_argument("--no-auth", action="store_true", help="Local only: no sign-in, roles from settings.toml.")
    p.add_argument(
        "--behind-proxy",
        action="store_true",
        default=os.environ.get("MEDINTEL_BEHIND_PROXY") == "1",
        help="Trust X-Forwarded-* headers from a TLS proxy (env MEDINTEL_BEHIND_PROXY=1).",
    )
    p.set_defaults(func=serve)

    args = parser.parse_args(argv)
    if getattr(args, "action", None) not in (None, "list") and not args.name:
        parser.error(f"users {args.action} needs a name")
    load_env_file(project_env_file())
    args.settings = load_settings()
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"medintel {args.command}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def dump(data: object) -> str:
    return json.dumps(data, indent=2, default=str)


if __name__ == "__main__":
    main()
