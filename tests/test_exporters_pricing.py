from __future__ import annotations

import csv
import shutil
import subprocess

import pytest

from nickerson_synth import exporters, pricing
from nickerson_synth.extractor import Auditor, discover_pdfs
from nickerson_synth.models import AuditRun, TokenUsage


@pytest.fixture
def run(rubric, pdf_dir, fake_client) -> AuditRun:
    return Auditor(rubric, "claude-sonnet-5-5", client=fake_client).run(discover_pdfs(pdf_dir))


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_write_all(run, tmp_path):
    paths = exporters.write_all(run, tmp_path / "out")
    assert set(paths) == {"json", "matrix", "evidence", "coverage"}
    matrix = {r["paper_id"]: r for r in _rows(paths["matrix"])}
    assert matrix["paper_a"]["evaluation_method"] == "benchmark_experiments;user_study"
    assert matrix["paper_a"]["evaluation_method__status"] == "verified"
    assert matrix["paper_b"]["evaluation_method__status"] == "unclassifiable"
    assert matrix["paper_b"]["artifact_availability__status"] == "unverified"

    ev = _rows(paths["evidence"])
    assert any(r["quote_verified"] == "False" for r in ev)
    assert any(r["proposed_characteristic"] == "literature_synthesis" for r in ev)

    cov = {
        (r["dimension_id"], r["characteristic_id"]): int(r["n_papers"])
        for r in _rows(paths["coverage"])
    }
    assert cov[("contribution_type", "new_method")] == 1
    assert cov[("contribution_type", "theory")] == 0
    assert AuditRun.load(paths["json"]) == run


def test_matrix_csv_marks_errors(run, tmp_path):
    run.papers[0].error = "boom"
    exporters.write_matrix_csv(run, tmp_path / "m.csv")
    row = _rows(tmp_path / "m.csv")[0]
    assert row["contribution_type__status"] == "error" and row["error"] == "boom"


def test_latex_escape():
    assert exporters.latex_escape(r"a_b & 50% #1 {x} ~ ^ \ $") == (
        r"a\_b \& 50\% \#1 \{x\} \textasciitilde{} \textasciicircum{} \textbackslash{} \$"
    )


def test_taxonomy_table(rubric):
    tex = exporters.taxonomy_table(rubric.taxonomy, label="tab:t", star=False)
    assert "\\begin{tabularx}{\\textwidth}" in tex and "\\begin{table}[t]" in tex
    assert "Contribution type & ME &" in tex and "Evaluation method & NE &" in tex
    assert "\\label{tab:t}" in tex and "evidence.." not in tex


def test_matrix_table(run, tmp_path):
    keys_csv = tmp_path / "keys.csv"
    keys_csv.write_text("paper_id,bibkey\npaper_a,smith2024index\n")
    keys = exporters.read_bibkeys(keys_csv)
    tex = exporters.matrix_table(run, bibkeys=keys)
    assert "\\cite{smith2024index}" in tex
    assert "paper\\_b" in tex  # no key -> escaped id
    assert "($\\checkmark$)" in tex  # unverified code for paper_b
    n_chars = sum(len(d.characteristics) for d in run.taxonomy.dimensions)
    row = next(line for line in tex.splitlines() if line.startswith("\\cite{smith2024index}"))
    assert row.count("&") == n_chars
    strict = exporters.matrix_table(run, verified_only=True)
    assert "($\\checkmark$)" not in strict


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_latex_compiles(run, rubric, tmp_path):
    body = exporters.taxonomy_table(rubric.taxonomy) + exporters.matrix_table(run)
    doc = (
        "\\documentclass{article}\\usepackage{booktabs,tabularx,graphicx,amssymb}"
        "\\begin{document}" + body + "\\end{document}"
    )
    (tmp_path / "t.tex").write_text(doc)
    res = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "t.tex"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stdout[-2000:]


# ------------------------------------------------------------------- pricing


def test_known_and_dated_model_prices():
    p = pricing.get_price("claude-sonnet-5-5")
    assert p.input == 2.0 and p.output == 10.0
    assert pricing.get_price("claude-haiku-4-5-20251001").input == 1.0


def test_default_and_legacy_sonnet_prices():
    assert pricing.DEFAULT_MODEL == "claude-sonnet-4-20250514"
    for model in (pricing.DEFAULT_MODEL, "claude-3-7-sonnet-latest"):
        p = pricing.get_price(model)
        assert (p.input, p.output, p.cache_write, p.cache_read) == (3.0, 15.0, 3.75, 0.30)


def test_unknown_model_needs_overrides():
    assert pricing.get_price("mystery-model") is None
    p = pricing.get_price("mystery-model", {"input": 2.0, "output": 8.0})
    assert p.cache_write == 2.5 and p.cache_read == pytest.approx(0.2)


def test_overrides_apply():
    p = pricing.get_price("claude-sonnet-5-5", {"input": 9.0, "output": None})
    assert p.input == 9.0 and p.output == 10.0


def test_cost_usd():
    price = pricing.ModelPrice(input=2, output=10, cache_write=2.5, cache_read=0.1)
    usage = TokenUsage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
    )
    assert pricing.cost_usd(usage, price) == pytest.approx(14.6)


def test_estimate_run_with_and_without_cache():
    price = pricing.ModelPrice(2, 10, 2.5, 0.1, min_cacheable_tokens=1000)
    cached = pricing.estimate_run(2000, [10_000, 10_000, 10_000], 500, price)
    assert cached.cache_used
    assert cached.usage.cache_creation_input_tokens == 2000
    assert cached.usage.cache_read_input_tokens == 4000
    small = pricing.estimate_run(500, [10_000, 10_000], 500, price)
    assert not small.cache_used and small.usage.input_tokens == 21_000
    assert (
        cached.cost_usd
        < pricing.estimate_run(
            2000, [10_000] * 3, 500, pricing.ModelPrice(2, 10, 2.5, 0.1, min_cacheable_tokens=10**9)
        ).cost_usd
    )
    assert pricing.estimate_run(2000, [1], 1, None).cost_usd is None


def test_estimate_tokens():
    assert pricing.estimate_tokens("abcd" * 10) == 10
    assert pricing.estimate_tokens(9) == 3
