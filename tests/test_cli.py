from __future__ import annotations

import json

import pytest
from conftest import FakeClient, default_responder
from typer.testing import CliRunner

from nickerson_synth import __version__, extractor, pricing
from nickerson_synth.cli import app

runner = CliRunner()


@pytest.fixture
def project(tmp_path, pdf_dir, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init"]).exit_code == 0
    return tmp_path


def test_version():
    res = runner.invoke(app, ["--version"])
    assert res.exit_code == 0 and __version__ in res.output


def test_init_writes_and_skips(tmp_path):
    res = runner.invoke(app, ["init", str(tmp_path / "x")])
    assert res.exit_code == 0
    assert (tmp_path / "x" / "rubric.yaml").exists()
    res = runner.invoke(app, ["init", str(tmp_path / "x")])
    assert "skip" in res.output
    assert "wrote" in runner.invoke(app, ["init", str(tmp_path / "x"), "--force"]).output


def test_validate_rubric(project):
    res = runner.invoke(app, ["validate", "rubric.yaml"])
    assert res.exit_code == 0 and "3 dimensions" in res.output


def test_validate_bad_rubric(project):
    (project / "bad.yaml").write_text("taxonomy: {name: x}\n")
    res = runner.invoke(app, ["validate", "bad.yaml"])
    assert res.exit_code == 1


def test_validate_missing_rubric(project):
    assert runner.invoke(app, ["validate", "nope.yaml"]).exit_code == 1


def test_validate_with_log_not_satisfied(project):
    from conftest import write_pdf

    write_pdf(project / "pdfs" / "unseen_paper.pdf", ["text"])
    res = runner.invoke(
        app, ["validate", "rubric.yaml", "--log", "iteration_log.yaml", "--pdfs", "pdfs"]
    )
    assert res.exit_code == 2
    assert "representative_sample_examined" in res.output
    assert "not yet examined" in res.output


def test_dry_run_makes_no_calls(project, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def boom(**kwargs):
        raise AssertionError("client must not be created in dry run")

    monkeypatch.setattr(extractor, "make_client", boom)
    res = runner.invoke(app, ["audit", "pdfs", "-r", "rubric.yaml", "--dry-run"])
    assert res.exit_code == 0, res.output
    assert "DRY RUN" in res.output and "Estimated cost" in res.output
    assert "PDFs readable / skipped     : 2 / 0" in res.output


def test_dry_run_unknown_model(project):
    res = runner.invoke(
        app, ["audit", "pdfs", "-r", "rubric.yaml", "--dry-run", "--model", "some-future-model"]
    )
    assert res.exit_code == 0 and "Unknown price" in res.output
    res = runner.invoke(
        app,
        [
            "audit",
            "pdfs",
            "-r",
            "rubric.yaml",
            "--dry-run",
            "--model",
            "some-future-model",
            "--price-input",
            "1",
            "--price-output",
            "2",
        ],
    )
    assert "Estimated cost" in res.output


def test_audit_requires_api_key(project, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    res = runner.invoke(app, ["audit", "pdfs", "-r", "rubric.yaml"])
    assert res.exit_code == 1 and "ANTHROPIC_API_KEY" in res.output


def test_audit_errors_for_empty_or_missing_dir(project, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    (project / "empty").mkdir()
    assert runner.invoke(app, ["audit", "empty", "-r", "rubric.yaml"]).exit_code == 1
    assert runner.invoke(app, ["audit", "missing", "-r", "rubric.yaml"]).exit_code == 1


def _mock_audit(project, monkeypatch, extra=()):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    fake = FakeClient(default_responder)
    monkeypatch.setattr(extractor, "make_client", lambda **kw: fake)
    res = runner.invoke(app, ["audit", "pdfs", "-r", "rubric.yaml", "-o", "results", *extra])
    assert res.exit_code == 0, res.output
    return res, fake


def test_audit_full_pipeline_and_latex(project, monkeypatch):
    res, fake = _mock_audit(project, monkeypatch)
    assert "Verified classifications" in res.output and "Cost: $" in res.output
    assert len(fake.messages.calls) == 2
    data = json.loads((project / "results" / "audit.json").read_text())
    assert data["model"] == pricing.DEFAULT_MODEL and len(data["papers"]) == 2

    (project / "keys.csv").write_text("paper_a,doe2025\n")
    res = runner.invoke(
        app, ["export-latex", "results/audit.json", "--bibkeys", "keys.csv", "-o", "matrix.tex"]
    )
    assert res.exit_code == 0
    assert "\\cite{doe2025}" in (project / "matrix.tex").read_text()

    res = runner.invoke(app, ["export-latex", "results/audit.json", "--kind", "taxonomy"])
    assert res.exit_code == 0 and "\\begin{tabularx}" in res.output

    res = runner.invoke(
        app,
        [
            "validate",
            "rubric.yaml",
            "--log",
            "iteration_log.yaml",
            "--results",
            "results/audit.json",
        ],
    )
    assert "every_characteristic_classifies_an_object" in res.output


def test_audit_limit(project, monkeypatch):
    _, fake = _mock_audit(project, monkeypatch, extra=["--limit", "1"])
    assert len(fake.messages.calls) == 1


def test_export_latex_from_rubric(project):
    res = runner.invoke(
        app,
        [
            "export-latex",
            "rubric.yaml",
            "--kind",
            "taxonomy",
            "--single-column",
            "--caption",
            "My caption",
        ],
    )
    assert res.exit_code == 0
    assert "\\begin{table}[t]" in res.output and "My caption" in res.output
    res = runner.invoke(app, ["export-latex", "rubric.yaml", "--kind", "matrix"])
    assert res.exit_code == 1


def test_export_latex_from_rubric_json(project):
    import yaml

    data = yaml.safe_load((project / "rubric.yaml").read_text())
    (project / "rubric.json").write_text(json.dumps(data))
    res = runner.invoke(app, ["export-latex", "rubric.json", "--kind", "taxonomy"])
    assert res.exit_code == 0 and "Contribution type" in res.output
