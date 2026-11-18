"""Command-line interface: ``nickerson-synth``."""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from nickerson_synth import __version__, exporters, extractor, pricing
from nickerson_synth.models import AuditRun, IterationLog, PaperAudit, Rubric
from nickerson_synth.templates import EXAMPLE_ITERATION_LOG, EXAMPLE_RUBRIC

app = typer.Typer(
    name="nickerson-synth",
    help=(
        "Evidence-verified, LLM-assisted literature coding for taxonomy and "
        "Systematization-of-Knowledge studies (Nickerson et al. 2013)."
    ),
    no_args_is_help=True,
    add_completion=False,
)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"nickerson-lit-synth {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version, is_eager=True, help="Show version."),
    ] = None,
) -> None:
    """nickerson-synth command group."""


def _fail(msg: str) -> None:
    typer.secho(f"Error: {msg}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


def _load_rubric(path: Path) -> Rubric:
    try:
        return Rubric.from_file(path)
    except FileNotFoundError:
        _fail(f"rubric not found: {path}")
    except (ValidationError, ValueError) as exc:
        _fail(f"invalid rubric {path}:\n{exc}")
    raise AssertionError  # unreachable


# --------------------------------------------------------------------------- #
# init
# --------------------------------------------------------------------------- #


@app.command()
def init(
    directory: Annotated[Path, typer.Argument(help="Where to write the starter files.")] = Path(),
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing files.")] = False,
) -> None:
    """Write an example rubric.yaml and iteration_log.yaml."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in (
        ("rubric.yaml", EXAMPLE_RUBRIC),
        ("iteration_log.yaml", EXAMPLE_ITERATION_LOG),
    ):
        target = directory / name
        if target.exists() and not force:
            typer.echo(f"skip   {target} (exists; use --force to overwrite)")
            continue
        target.write_text(content, encoding="utf-8")
        typer.echo(f"wrote  {target}")
    typer.echo("\nNext: edit rubric.yaml, then run")
    typer.echo("  nickerson-synth audit ./pdfs --rubric rubric.yaml --dry-run")


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #


@app.command()
def validate(
    rubric: Annotated[Path, typer.Argument(help="Rubric YAML/JSON.")],
    log: Annotated[
        Path | None, typer.Option("--log", help="Iteration log to check ending conditions.")
    ] = None,
    results: Annotated[
        Path | None, typer.Option("--results", help="audit.json from a previous run.")
    ] = None,
    pdfs: Annotated[
        Path | None, typer.Option("--pdfs", help="Corpus folder (file stems = object ids).")
    ] = None,
) -> None:
    """Validate a rubric and, optionally, Nickerson's ending conditions."""
    rb = _load_rubric(rubric)
    t = rb.taxonomy
    n_chars = sum(len(d.characteristics) for d in t.dimensions)
    typer.secho(
        f"Rubric OK: {len(t.dimensions)} dimensions, {n_chars} characteristics.", fg="green"
    )
    for d in t.dimensions:
        typer.echo(
            f"  {d.id:<28} {'ME' if d.exclusive else 'NE'}  {', '.join(d.characteristic_ids)}"
        )
    if log is None:
        return
    try:
        il = IterationLog.from_file(log)
    except (ValidationError, ValueError, FileNotFoundError) as exc:
        _fail(f"invalid iteration log {log}:\n{exc}")
    audit = AuditRun.load(results) if results else None
    corpus = [p.stem for p in extractor.discover_pdfs(pdfs)] if pdfs else None
    report = il.evaluate_ending_conditions(t, audit=audit, corpus=corpus)
    typer.echo("\nObjective ending conditions:")
    for k, v in report.objective.items():
        typer.echo(f"  [{_mark(v)}] {k}")
    typer.echo("Subjective ending conditions:")
    for k, v in report.subjective.items():
        typer.echo(f"  [{_mark(v)}] {k}")
    for m in report.messages:
        typer.secho(f"  note: {m}", fg="yellow")
    if report.satisfied:
        typer.secho(
            "All ending conditions met: the taxonomy-building process can stop.", fg="green"
        )
    else:
        typer.secho("Ending conditions not (yet) met: run another iteration.", fg="yellow")
        raise typer.Exit(2)


def _mark(v: bool | None) -> str:
    return {True: "x", False: " ", None: "?"}[v]


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #


@app.command()
def audit(
    pdf_dir: Annotated[Path, typer.Argument(help="Folder of PDFs to code.")],
    rubric: Annotated[Path, typer.Option("--rubric", "-r", help="Rubric YAML/JSON.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output folder.")] = Path("results"),
    model: Annotated[str, typer.Option(help="Claude model id.")] = pricing.DEFAULT_MODEL,
    concurrency: Annotated[int, typer.Option(min=1, max=32, help="Parallel requests.")] = 4,
    max_chars: Annotated[
        int, typer.Option(help="Max characters of paper text sent per PDF.")
    ] = extractor.DEFAULT_MAX_CHARS,
    max_tokens: Annotated[int, typer.Option(help="Max output tokens per paper.")] = 8000,
    recursive: Annotated[bool, typer.Option("--recursive", help="Include sub-folders.")] = False,
    limit: Annotated[int | None, typer.Option(min=1, help="Only process the first N PDFs.")] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Estimate tokens and cost; no API calls.")
    ] = False,
    output_tokens_per_dimension: Annotated[
        int, typer.Option(help="Dry-run assumption for output tokens per dimension.")
    ] = 350,
    price_input: Annotated[float | None, typer.Option(help="USD / MTok input override.")] = None,
    price_output: Annotated[float | None, typer.Option(help="USD / MTok output override.")] = None,
    price_cache_write: Annotated[float | None, typer.Option(help="USD / MTok cache write.")] = None,
    price_cache_read: Annotated[float | None, typer.Option(help="USD / MTok cache read.")] = None,
) -> None:
    """Code every PDF in PDF_DIR against the rubric and write CSV/JSON matrices."""
    rb = _load_rubric(rubric)
    try:
        paths = extractor.discover_pdfs(pdf_dir, recursive=recursive)
    except NotADirectoryError as exc:
        _fail(str(exc))
    if limit:
        paths = paths[:limit]
    if not paths:
        _fail(f"no PDFs found in {pdf_dir}")
    price = pricing.get_price(
        model,
        {
            "input": price_input,
            "output": price_output,
            "cache_write": price_cache_write,
            "cache_read": price_cache_read,
        },
    )

    if dry_run:
        _dry_run(rb, paths, model, max_chars, output_tokens_per_dimension, price)
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        _fail("ANTHROPIC_API_KEY is not set (or use --dry-run).")

    typer.echo(f"Coding {len(paths)} PDF(s) with {model} (concurrency {concurrency}) ...")
    done = 0

    def progress(p: PaperAudit) -> None:
        nonlocal done
        done += 1
        if p.error:
            typer.secho(f"[{done}/{len(paths)}] {p.paper_id}: ERROR {p.error}", fg="red")
        else:
            ok = sum(c.verified for c in p.classifications)
            typer.echo(
                f"[{done}/{len(paths)}] {p.paper_id}: {ok}/{len(p.classifications)} verified"
            )

    auditor = extractor.Auditor(
        rb, model, concurrency=concurrency, max_chars=max_chars, max_tokens=max_tokens
    )
    run = auditor.run(paths, on_done=progress)
    written = exporters.write_all(run, out)
    _summary(run, price, written)


def _dry_run(
    rb: Rubric,
    paths: list[Path],
    model: str,
    max_chars: int,
    out_per_dim: int,
    price: pricing.ModelPrice | None,
) -> None:
    prefix_chars = len(json.dumps(extractor.build_tool(rb))) + sum(
        len(b["text"]) for b in extractor.build_system(rb)
    )
    prefix = pricing.estimate_tokens(prefix_chars)
    docs: list[int] = []
    skipped = 0
    for p in paths:
        try:
            pages = extractor.load_pdf_pages(p)
        except Exception:
            skipped += 1
            continue
        text, _ = extractor.build_document(pages, max_chars)
        docs.append(pricing.estimate_tokens(text) + 60)
    est = pricing.estimate_run(prefix, docs, out_per_dim * len(rb.taxonomy.dimensions), price)
    typer.echo(f"DRY RUN - no API calls. Model: {model}")
    typer.echo(f"  PDFs readable / skipped     : {est.n_papers} / {skipped}")
    typer.echo(
        f"  Cached prefix (rubric+tool) : ~{prefix:,} tokens "
        f"({'cached' if est.cache_used else 'below cache minimum, not cached'})"
    )
    typer.echo(f"  Paper text                  : ~{est.document_tokens:,} tokens")
    typer.echo(f"  Output (assumed)            : ~{est.output_tokens:,} tokens")
    if est.cost_usd is None:
        typer.secho(
            f"  Unknown price for '{model}'. Pass --price-input and --price-output.", fg="yellow"
        )
    else:
        typer.secho(f"  Estimated cost              : ~${est.cost_usd:,.2f} USD", fg="green")
    typer.echo("  (Estimate uses ~4 chars/token; actual usage is reported after a real run.)")


def _summary(run: AuditRun, price: pricing.ModelPrice | None, written: dict[str, Path]) -> None:
    usage = run.total_usage()
    errors = sum(1 for p in run.papers if p.error)
    cls = [c for p in run.papers for c in p.classifications]
    verified = sum(c.verified for c in cls)
    typer.echo("")
    typer.secho(f"Done: {len(run.papers) - errors} coded, {errors} failed.", fg="green")
    if cls:
        typer.echo(f"Verified classifications: {verified}/{len(cls)} ({verified / len(cls):.0%})")
    typer.echo(
        f"Tokens: input {usage.input_tokens:,}, output {usage.output_tokens:,}, "
        f"cache write {usage.cache_creation_input_tokens:,}, "
        f"cache read {usage.cache_read_input_tokens:,}"
    )
    if price:
        typer.echo(f"Cost: ${pricing.cost_usd(usage, price):,.2f} USD")
    for k, path in written.items():
        typer.echo(f"  {k:<9} {path}")


# --------------------------------------------------------------------------- #
# export-latex
# --------------------------------------------------------------------------- #


class LatexKind(str, Enum):
    matrix = "matrix"
    taxonomy = "taxonomy"


@app.command("export-latex")
def export_latex(
    source: Annotated[
        Path, typer.Argument(help="audit.json (matrix or taxonomy) or a rubric file (taxonomy).")
    ],
    kind: Annotated[LatexKind, typer.Option("--kind", "-k")] = LatexKind.matrix,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Write to file (default: stdout).")
    ] = None,
    bibkeys: Annotated[
        Path | None, typer.Option(help="CSV of paper_id,bibkey to emit \\cite{} keys.")
    ] = None,
    caption: Annotated[str | None, typer.Option()] = None,
    label: Annotated[str | None, typer.Option()] = None,
    verified_only: Annotated[
        bool, typer.Option("--verified-only", help="Omit codes whose evidence was not verified.")
    ] = False,
    single_column: Annotated[
        bool, typer.Option("--single-column", help="Use table instead of table*.")
    ] = False,
) -> None:
    """Emit LaTeX tabularx boilerplate for the taxonomy or the classification matrix."""
    run: AuditRun | None = None
    rb: Rubric | None = None
    if source.suffix.lower() == ".json":
        data = json.loads(source.read_text(encoding="utf-8"))
        if "papers" in data:
            run = AuditRun.model_validate(data)
            rb = run.rubric
        else:
            rb = Rubric.model_validate(data)
    else:
        rb = _load_rubric(source)

    if kind is LatexKind.matrix:
        if run is None:
            _fail("--kind matrix needs an audit.json produced by `nickerson-synth audit`.")
        keys = exporters.read_bibkeys(bibkeys) if bibkeys else None
        tex = exporters.matrix_table(
            run,
            bibkeys=keys,
            caption=caption,
            label=label or "tab:classification",
            verified_only=verified_only,
            star=not single_column,
        )
    else:
        tex = exporters.taxonomy_table(
            rb.taxonomy, caption=caption, label=label or "tab:taxonomy", star=not single_column
        )

    if out:
        out.write_text(tex, encoding="utf-8")
        typer.echo(f"wrote {out}")
    else:
        typer.echo(tex, nl=False)


if __name__ == "__main__":  # pragma: no cover
    app()
