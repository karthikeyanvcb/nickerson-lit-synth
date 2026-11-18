from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pymupdf
import pytest
import yaml

from nickerson_synth.extractor import TOOL_NAME
from nickerson_synth.models import Rubric
from nickerson_synth.templates import EXAMPLE_RUBRIC

PAPER_A_PAGES = [
    "A Fast Index for Graphs\n\nAbstract. In this paper we propose a novel indexing "
    "algorithm for large sparse graphs that reduces query latency.",
    "Evaluation. We run benchmark experiments on five public datasets and compare "
    "against three baselines. We also conducted a user study with 12 participants.",
    "Availability. Our source code and all datasets are available at "
    "https://example.org/artifact under an open licence.",
]


@pytest.fixture
def rubric() -> Rubric:
    return Rubric.model_validate(yaml.safe_load(EXAMPLE_RUBRIC))


def write_pdf(path: Path, pages: list[str]) -> Path:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(72, 72, 520, 770), text, fontsize=11)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def pdf_dir(tmp_path: Path) -> Path:
    d = tmp_path / "pdfs"
    d.mkdir()
    write_pdf(d / "paper_a.pdf", PAPER_A_PAGES)
    write_pdf(
        d / "paper_b.pdf",
        ["We survey prior work and systematise it into a framework. No code is released."],
    )
    return d


def good_answer_a() -> dict[str, Any]:
    return {
        "classifications": [
            {
                "dimension_id": "contribution_type",
                "characteristic_ids": ["new_method"],
                "confidence": 0.95,
                "rationale": "Proposes a new algorithm.",
                "evidence": [{"page": 1, "quote": "we propose a novel indexing algorithm"}],
                "unclassifiable": False,
            },
            {
                "dimension_id": "evaluation_method",
                "characteristic_ids": ["benchmark_experiments", "user_study"],
                "confidence": 0.9,
                "rationale": "Benchmarks and a user study.",
                "evidence": [
                    {"page": 2, "quote": "We run benchmark experiments on five public datasets"},
                    {"page": 2, "quote": "conducted a user study with 12 participants"},
                ],
                "unclassifiable": False,
            },
            {
                "dimension_id": "artifact_availability",
                "characteristic_ids": ["code_and_data"],
                "confidence": 0.9,
                "rationale": "Code and data released.",
                "evidence": [
                    {"page": 3, "quote": "Our source code and all datasets are available"}
                ],
                "unclassifiable": False,
            },
        ]
    }


def answer_b() -> dict[str, Any]:
    return {
        "classifications": [
            {
                "dimension_id": "contribution_type",
                "characteristic_ids": ["survey_or_sok"],
                "confidence": 0.9,
                "rationale": "Survey.",
                "evidence": [{"page": 1, "quote": "We survey prior work and systematise it"}],
                "unclassifiable": False,
            },
            {
                "dimension_id": "evaluation_method",
                "characteristic_ids": [],
                "confidence": 0.4,
                "rationale": "Not applicable to a survey.",
                "evidence": [],
                "unclassifiable": True,
                "proposed_characteristic": "literature_synthesis",
            },
            {
                "dimension_id": "artifact_availability",
                "characteristic_ids": ["not_released"],
                "confidence": 0.8,
                "rationale": "States no code.",
                # fabricated quote -> must fail verification
                "evidence": [{"page": 1, "quote": "the authors promise to release code later"}],
                "unclassifiable": False,
            },
        ]
    }


def make_response(tool_input: dict[str, Any], stop_reason: str = "tool_use") -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", name=TOOL_NAME, input=tool_input)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=1000,
            output_tokens=200,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=800,
        ),
    )


class FakeMessages:
    def __init__(self, responder: Callable[[dict[str, Any]], Any]) -> None:
        self.responder = responder
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        result = self.responder(kwargs)
        if isinstance(result, Exception):
            raise result
        return result


class FakeClient:
    """Stand-in for anthropic.AsyncAnthropic; no network, no credits."""

    def __init__(self, responder: Callable[[dict[str, Any]], Any]) -> None:
        self.messages = FakeMessages(responder)


def default_responder(kwargs: dict[str, Any]) -> Any:
    content = kwargs["messages"][0]["content"]
    if 'id="paper_a"' in content:
        return make_response(good_answer_a())
    return make_response(answer_b())


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient(default_responder)
