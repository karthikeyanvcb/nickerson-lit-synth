from __future__ import annotations

import json

import pytest
import yaml
from pydantic import ValidationError

from nickerson_synth.models import (
    AuditRun,
    Classification,
    Dimension,
    Iteration,
    IterationApproach,
    IterationLog,
    PageEvidence,
    PaperAudit,
    Rubric,
    Taxonomy,
    TokenUsage,
)
from nickerson_synth.templates import EXAMPLE_ITERATION_LOG, EXAMPLE_RUBRIC


def _dim(id_="d1", chars=("a", "b"), exclusive=True) -> dict:
    return {
        "id": id_,
        "label": id_.upper(),
        "definition": "def",
        "exclusive": exclusive,
        "characteristics": [{"id": c, "label": c.upper(), "definition": "x"} for c in chars],
    }


def test_example_rubric_is_valid(rubric):
    assert rubric.taxonomy.dimension_ids == [
        "contribution_type",
        "evaluation_method",
        "artifact_availability",
    ]
    assert rubric.taxonomy.get("evaluation_method").exclusive is False
    assert "contribution_type.new_method" in rubric.taxonomy.qualified_characteristic_ids()


def test_dimension_needs_two_characteristics():
    with pytest.raises(ValidationError):
        Dimension.model_validate(_dim(chars=("only",)))


def test_duplicate_characteristic_ids_rejected():
    with pytest.raises(ValidationError, match="Duplicate characteristic id"):
        Dimension.model_validate(_dim(chars=("a", "a")))


def test_duplicate_characteristic_labels_rejected_case_insensitive():
    d = _dim()
    d["characteristics"][1]["label"] = d["characteristics"][0]["label"].lower()
    with pytest.raises(ValidationError, match="Duplicate characteristic label"):
        Dimension.model_validate(d)


def test_duplicate_dimensions_rejected():
    with pytest.raises(ValidationError, match="Duplicate dimension id"):
        Taxonomy.model_validate(
            {"name": "t", "meta_characteristic": "m", "dimensions": [_dim("x"), _dim("x")]}
        )


@pytest.mark.parametrize("bad", ["Has Space", "UPPER", "-lead", ""])
def test_slug_ids_enforced(bad):
    with pytest.raises(ValidationError):
        Dimension.model_validate(_dim(id_=bad))


def test_extra_fields_forbidden():
    d = _dim()
    d["typo_field"] = 1
    with pytest.raises(ValidationError):
        Dimension.model_validate(d)


def test_get_unknown_raises(rubric):
    with pytest.raises(KeyError):
        rubric.taxonomy.get("nope")
    with pytest.raises(KeyError):
        rubric.taxonomy.get("contribution_type").get("nope")


def test_rubric_from_yaml_and_json(tmp_path, rubric):
    y = tmp_path / "r.yaml"
    y.write_text(EXAMPLE_RUBRIC)
    j = tmp_path / "r.json"
    j.write_text(json.dumps(rubric.model_dump(mode="json")))
    assert Rubric.from_file(y) == Rubric.from_file(j) == rubric
    bad = tmp_path / "r.txt"
    bad.write_text("x")
    with pytest.raises(ValueError, match="Unsupported"):
        Rubric.from_file(bad)


def test_rubric_prompt_is_deterministic_and_complete(rubric):
    p1, p2 = rubric.to_prompt(), rubric.to_prompt()
    assert p1 == p2
    assert "MUTUALLY EXCLUSIVE" in p1 and "NON-EXCLUSIVE" in p1
    for cid in rubric.taxonomy.qualified_characteristic_ids():
        assert f"`{cid.split('.')[1]}`" in p1
    assert rubric.sha256() == Rubric.model_validate(rubric.model_dump()).sha256()


def test_iteration_approach_modes():
    assert IterationApproach.CONCEPTUAL_TO_EMPIRICAL.mode == "deductive"
    assert IterationApproach.EMPIRICAL_TO_CONCEPTUAL.mode == "inductive"


def test_iteration_numbers_must_be_consecutive():
    with pytest.raises(ValidationError, match="consecutive"):
        IterationLog(
            meta_characteristic="m",
            iterations=[
                Iteration(number=1, approach="conceptual-to-empirical"),
                Iteration(number=3, approach="empirical-to-conceptual"),
            ],
        )


def test_unknown_subjective_condition_rejected():
    with pytest.raises(ValidationError, match="Unknown subjective"):
        IterationLog(meta_characteristic="m", subjective_assessment={"pretty": True})


def _run(rubric, papers):
    return AuditRun(
        tool_version="t", model="m", rubric_sha256=rubric.sha256(), rubric=rubric, papers=papers
    )


def _all_chars_paper(rubric) -> list[PaperAudit]:
    papers = []
    for i, d in enumerate(rubric.taxonomy.dimensions):
        for c in d.characteristics:
            papers.append(
                PaperAudit(
                    paper_id=f"p{i}{c.id}",
                    source_path="x",
                    sha256="0",
                    classifications=[Classification(dimension_id=d.id, characteristic_ids=[c.id])],
                )
            )
    return papers


def test_ending_conditions_all_met(rubric):
    log = IterationLog.model_validate(yaml.safe_load(EXAMPLE_ITERATION_LOG))
    run = _run(rubric, _all_chars_paper(rubric))
    corpus = ["paper_a", "paper_b", "paper_c", "paper_d", "paper_e"]
    report = log.evaluate_ending_conditions(rubric.taxonomy, audit=run, corpus=corpus)
    assert report.satisfied, report.messages


def test_ending_conditions_fail_on_changes_and_gaps(rubric):
    log = IterationLog.model_validate(yaml.safe_load(EXAMPLE_ITERATION_LOG))
    log.iterations[-1].characteristics_added = ["evaluation_method.new"]
    log.iterations[-1].merges = ["a+b"]
    run = _run(rubric, [])
    report = log.evaluate_ending_conditions(
        rubric.taxonomy, audit=run, corpus=["paper_a", "unseen"]
    )
    assert not report.satisfied
    assert report.objective["no_new_dimensions_or_characteristics_last_iteration"] is False
    assert (
        report.objective["no_dimensions_or_characteristics_merged_or_split_last_iteration"] is False
    )
    assert report.objective["every_characteristic_classifies_an_object"] is False
    assert report.objective["representative_sample_examined"] is False
    assert any("unseen" in m for m in report.messages)


def test_ending_conditions_unknown_without_inputs(rubric):
    log = IterationLog(meta_characteristic="m")
    report = log.evaluate_ending_conditions(rubric.taxonomy)
    assert report.objective["representative_sample_examined"] is None
    assert report.objective["no_new_dimensions_or_characteristics_last_iteration"] is None
    assert not report.satisfied
    assert any("empty" in m for m in report.messages)


def test_classification_status():
    ok = Classification(
        dimension_id="d",
        characteristic_ids=["a"],
        evidence=[PageEvidence(page=1, quote="q", verified=True, matched_page=1)],
    )
    assert ok.verified and ok.status == "verified"
    no_ev = Classification(dimension_id="d", characteristic_ids=["a"])
    assert no_ev.status == "unverified"
    issue = ok.model_copy(update={"issues": ["x"]})
    assert issue.status == "unverified"
    assert Classification(dimension_id="d", unclassifiable=True).status == "unclassifiable"


def test_token_usage_add_and_from_api():
    a = TokenUsage(input_tokens=1, output_tokens=2)
    b = TokenUsage.from_api(type("U", (), {"input_tokens": 3, "cache_read_input_tokens": None})())
    assert (a + b).input_tokens == 4
    assert TokenUsage.from_api(None) == TokenUsage()


def test_audit_run_roundtrip_and_counts(tmp_path, rubric):
    run = _run(rubric, _all_chars_paper(rubric))
    path = tmp_path / "a.json"
    run.save(path)
    loaded = AuditRun.load(path)
    assert loaded == run
    assert all(n == 1 for n in loaded.characteristic_counts().values())
    assert loaded.total_usage() == TokenUsage()
