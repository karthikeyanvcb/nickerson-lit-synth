from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from conftest import FakeClient, good_answer_a, make_response, write_pdf

from nickerson_synth import extractor
from nickerson_synth.extractor import (
    Auditor,
    PageText,
    build_document,
    build_system,
    build_tool,
    discover_pdfs,
    load_pdf_pages,
    normalize_text,
    parse_and_validate,
    verify_quote,
)

PAGES = [
    PageText(1, "The quick brown fox jumps over the lazy dog near the river bank."),
    PageText(2, "We evalu-\nate the sys-\ntem with “benchmark” workﬂows — carefully."),
]


# ---------------------------------------------------------------- PDF handling


def test_load_pdf_pages_and_discovery(pdf_dir, tmp_path):
    pages = load_pdf_pages(pdf_dir / "paper_a.pdf")
    assert [p.number for p in pages] == [1, 2, 3]
    assert "benchmark" in pages[1].text
    sub = pdf_dir / "sub"
    sub.mkdir()
    write_pdf(sub / "paper_c.pdf", ["nested"])
    (pdf_dir / "notes.txt").write_text("x")
    assert [p.name for p in discover_pdfs(pdf_dir)] == ["paper_a.pdf", "paper_b.pdf"]
    assert len(discover_pdfs(pdf_dir, recursive=True)) == 3
    with pytest.raises(NotADirectoryError):
        discover_pdfs(tmp_path / "missing")


def test_build_document_truncates_but_keeps_one_page():
    text, n = build_document(PAGES, max_chars=10_000)
    assert n == 2 and '<page number="2">' in text
    text, n = build_document(PAGES, max_chars=100)
    assert n == 1 and '<page number="2">' not in text
    text, n = build_document(PAGES, max_chars=20)
    assert n == 1 and len(text) == 20


# ------------------------------------------------------------ verification


def test_normalize_text_handles_ligatures_quotes_hyphens():
    n = normalize_text(PAGES[1].text)
    assert "evaluate the system" in n
    assert '"benchmark" workflows - carefully' in n


def test_verify_quote_exact_and_normalised():
    assert verify_quote("quick brown fox jumps", 1, PAGES) == (True, 1)
    assert verify_quote('We evaluate the system with "benchmark" workflows', 2, PAGES) == (True, 2)


def test_verify_quote_wrong_page_is_corrected():
    assert verify_quote("quick brown fox jumps", 2, PAGES) == (True, 1)


def test_verify_quote_with_ellipsis_in_order_only():
    assert verify_quote("The quick brown fox ... near the river bank", 1, PAGES)[0]
    assert not verify_quote("near the river bank ... The quick brown fox", 1, PAGES)[0]


def test_verify_quote_rejects_paraphrase_and_short():
    assert verify_quote("a fast brown fox leaps over a dog", 1, PAGES) == (False, None)
    assert verify_quote("fox", 1, PAGES) == (False, None)
    assert verify_quote("   ", 1, PAGES) == (False, None)


def test_verify_quote_tolerates_compound_hyphen():
    pages = [PageText(1, "a privacy-preserving data-base for queries")]
    assert verify_quote("privacy preserving", 1, pages)[0] is False  # words differ
    assert verify_quote("privacy-preserving database for queries", 1, pages)[0]


# --------------------------------------------------------------- prompt/tool


def test_tool_schema_enums(rubric):
    tool = build_tool(rubric)
    item = tool["input_schema"]["properties"]["classifications"]["items"]
    assert item["properties"]["dimension_id"]["enum"] == rubric.taxonomy.dimension_ids
    assert "new_method" in item["properties"]["characteristic_ids"]["items"]["enum"]


def test_system_has_single_cache_breakpoint_on_rubric(rubric):
    blocks = build_system(rubric)
    cached = [b for b in blocks if "cache_control" in b]
    assert len(cached) == 1 and cached[0] is blocks[-1]
    assert cached[0]["cache_control"] == {"type": "ephemeral"}
    assert rubric.to_prompt() in cached[0]["text"]


# ------------------------------------------------------------ parse/validate


def test_parse_valid_answer(rubric, pdf_dir):
    pages = load_pdf_pages(pdf_dir / "paper_a.pdf")
    cls = parse_and_validate(good_answer_a(), rubric, pages)
    assert [c.dimension_id for c in cls] == rubric.taxonomy.dimension_ids
    assert all(c.verified for c in cls), [c.issues for c in cls]


def test_parse_flags_rule_violations(rubric):
    pages = [PageText(1, "We propose a new method and also an empirical study of systems.")]
    raw = {
        "classifications": [
            {  # ME dimension with two codes
                "dimension_id": "contribution_type",
                "characteristic_ids": ["new_method", "empirical_study"],
                "confidence": 3,  # clamped
                "rationale": "",
                "evidence": [{"page": 1, "quote": "We propose a new method"}],
                "unclassifiable": False,
            },
            {  # characteristic from another dimension
                "dimension_id": "evaluation_method",
                "characteristic_ids": ["code_only", "simulation"],
                "confidence": 0.5,
                "rationale": "",
                "evidence": [],
                "unclassifiable": False,
            },
            {  # duplicate dimension
                "dimension_id": "contribution_type",
                "characteristic_ids": ["theory"],
                "confidence": 0.5,
                "rationale": "",
                "evidence": [],
                "unclassifiable": False,
            },
            {"dimension_id": "not_a_dimension", "characteristic_ids": []},
        ]
    }
    cls = {c.dimension_id: c for c in parse_and_validate(raw, rubric, pages)}
    ct = cls["contribution_type"]
    assert ct.confidence == 1.0
    assert any("mutually exclusive" in i for i in ct.issues)
    assert any("more than once" in i for i in ct.issues)
    ev = cls["evaluation_method"]
    assert ev.characteristic_ids == ["simulation"]
    assert any("not in this dimension" in i for i in ev.issues)
    assert any("evidence" in i for i in ev.issues)
    missing = cls["artifact_availability"]
    assert missing.unclassifiable and "not returned" in missing.issues[0]


def test_parse_unclassifiable_keeps_proposal(rubric):
    raw = {
        "classifications": [
            {
                "dimension_id": "evaluation_method",
                "characteristic_ids": ["user_study"],
                "confidence": 0.2,
                "rationale": "",
                "evidence": [],
                "unclassifiable": True,
                "proposed_characteristic": "field_deployment",
            }
        ]
    }
    c = parse_and_validate(raw, rubric, [PageText(1, "x")])[1]
    assert c.unclassifiable and c.characteristic_ids == []
    assert c.proposed_characteristic == "field_deployment"
    assert c.status == "unclassifiable"


# ------------------------------------------------------------------- auditor


def test_auditor_end_to_end(rubric, pdf_dir, fake_client):
    auditor = Auditor(rubric, "claude-sonnet-5-5", client=fake_client, concurrency=2)
    done = []
    run = auditor.run(discover_pdfs(pdf_dir), on_done=done.append)
    assert len(done) == 2 and [p.paper_id for p in run.papers] == ["paper_a", "paper_b"]
    a, b = run.papers
    assert a.error is None and all(c.verified for c in a.classifications)
    assert a.n_pages == 3 and a.pages_sent == 3 and not a.truncated
    assert b.get("evaluation_method").unclassifiable
    art = b.get("artifact_availability")
    assert not art.verified and not art.evidence[0].verified  # fabricated quote caught
    assert run.total_usage().cache_read_input_tokens == 1600

    call = fake_client.messages.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": extractor.TOOL_NAME}
    assert call["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert call["model"] == "claude-sonnet-5-5"
    # identical cached prefix across requests
    assert all(
        c["system"] == call["system"] and c["tools"] == call["tools"]
        for c in fake_client.messages.calls
    )


def test_cache_warmup_runs_first_request_alone(rubric, pdf_dir):
    state = {"active": 0, "max_active_during_first": 0, "n": 0}

    async def create(**kwargs):
        state["n"] += 1
        first = state["n"] == 1
        state["active"] += 1
        await asyncio.sleep(0.01)
        if first:
            state["max_active_during_first"] = state["active"]
        state["active"] -= 1
        return make_response({"classifications": []})

    for i in range(3):
        write_pdf(pdf_dir / f"extra_{i}.pdf", ["text"])
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    Auditor(rubric, "m", client=client, concurrency=8).run(discover_pdfs(pdf_dir))
    assert state["max_active_during_first"] == 1 and state["n"] == 5


def test_auditor_isolates_errors(rubric, pdf_dir, tmp_path):
    (pdf_dir / "broken.pdf").write_bytes(b"not a pdf")
    write_pdf(pdf_dir / "blank.pdf", [""])

    def responder(kwargs):
        if 'id="paper_b"' in kwargs["messages"][0]["content"]:
            return RuntimeError("rate limited")
        if 'id="paper_a"' in kwargs["messages"][0]["content"]:
            return make_response(good_answer_a(), stop_reason="max_tokens")
        return make_response({"classifications": []})

    run = Auditor(rubric, "m", client=FakeClient(responder)).run(discover_pdfs(pdf_dir))
    errs = {p.paper_id: p.error for p in run.papers}
    assert "extraction failed" in errs["broken"]
    assert "No extractable text" in errs["blank"]
    assert "max_tokens" in errs["paper_a"]
    assert "rate limited" in errs["paper_b"]


def test_missing_tool_call_is_error(rubric, pdf_dir):
    resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hi")], usage=None, stop_reason="end_turn"
    )
    run = Auditor(rubric, "m", client=FakeClient(lambda k: resp)).run([pdf_dir / "paper_a.pdf"])
    assert "record_classification" in run.papers[0].error


def test_truncation_reported(rubric, pdf_dir, fake_client):
    auditor = Auditor(rubric, "m", client=fake_client, max_chars=200)
    p = auditor.run([pdf_dir / "paper_a.pdf"]).papers[0]
    assert p.truncated and p.pages_sent == 1


def test_duplicate_stems_get_unique_ids(rubric, pdf_dir, fake_client):
    sub = pdf_dir / "sub"
    sub.mkdir()
    write_pdf(sub / "paper_b.pdf", ["again"])
    run = Auditor(rubric, "m", client=fake_client).run(discover_pdfs(pdf_dir, recursive=True))
    ids = [p.paper_id for p in run.papers]
    assert len(set(ids)) == len(ids) and "paper_b__2" in ids


def test_empty_paths(rubric, fake_client):
    assert Auditor(rubric, "m", client=fake_client).run([]).papers == []


def test_make_client_is_lazy(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    client = extractor.make_client(max_retries=1)
    assert hasattr(client, "messages")
