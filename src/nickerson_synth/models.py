"""Pydantic v2 data models.

The models cover four concerns:

1. **Taxonomy definition** (``Characteristic``, ``Dimension``, ``Taxonomy``, ``Rubric``):
   user-supplied dimensions whose characteristics are validated for structural
   uniqueness, with a per-dimension *mutually exclusive* flag.
2. **Development process** (``Iteration``, ``IterationLog``): a log of
   conceptual-to-empirical (deductive) and empirical-to-conceptual (inductive)
   iterations, and a checker for the ending conditions of Nickerson, Varshney &
   Muntermann (2013), "A method for taxonomy development and its application in
   information systems", *European Journal of Information Systems* 22(3):336-359.
3. **Evidence** (``PageEvidence``, ``Classification``): verbatim quotes tied to a
   PDF page, plus the result of checking each quote against the extracted text.
4. **Run outputs** (``TokenUsage``, ``PaperAudit``, ``AuditRun``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Slug = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^[a-z0-9][a-z0-9_\-]*$", max_length=64),
]
"""Lower-case identifier used for dimension and characteristic ids."""

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------- #
# Taxonomy definition
# --------------------------------------------------------------------------- #


class Characteristic(_Base):
    """One value a dimension can take (a column in the classification matrix)."""

    id: Slug
    label: NonEmpty
    definition: NonEmpty
    indicators: list[str] = Field(
        default_factory=list, description="Textual cues that suggest this characteristic applies."
    )
    exclusions: list[str] = Field(
        default_factory=list, description="Cues that must NOT be coded as this characteristic."
    )


class Dimension(_Base):
    """A dimension of the taxonomy.

    ``exclusive=True`` means the characteristics are mutually exclusive: every
    object is coded with exactly one characteristic. ``exclusive=False`` allows
    one or more. In both cases the characteristics are meant to be collectively
    exhaustive; objects that fit none are reported as *unclassifiable*, which is
    the signal for a further empirical-to-conceptual iteration.
    """

    id: Slug
    label: NonEmpty
    definition: NonEmpty
    exclusive: bool = True
    characteristics: list[Characteristic] = Field(min_length=2)

    @model_validator(mode="after")
    def _unique_characteristics(self) -> Dimension:
        _ensure_unique((c.id for c in self.characteristics), f"characteristic id in '{self.id}'")
        _ensure_unique(
            (c.label.casefold() for c in self.characteristics),
            f"characteristic label in '{self.id}'",
        )
        return self

    @property
    def characteristic_ids(self) -> list[str]:
        return [c.id for c in self.characteristics]

    def get(self, characteristic_id: str) -> Characteristic:
        for c in self.characteristics:
            if c.id == characteristic_id:
                return c
        raise KeyError(f"Unknown characteristic '{characteristic_id}' in dimension '{self.id}'")


class Taxonomy(_Base):
    """A taxonomy: a meta-characteristic plus a set of dimensions."""

    name: NonEmpty
    meta_characteristic: NonEmpty = Field(
        description="Nickerson step 1: the purpose that every dimension must follow from."
    )
    description: str = ""
    dimensions: list[Dimension] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_dimensions(self) -> Taxonomy:
        _ensure_unique((d.id for d in self.dimensions), "dimension id")
        _ensure_unique((d.label.casefold() for d in self.dimensions), "dimension label")
        return self

    @property
    def dimension_ids(self) -> list[str]:
        return [d.id for d in self.dimensions]

    def get(self, dimension_id: str) -> Dimension:
        for d in self.dimensions:
            if d.id == dimension_id:
                return d
        raise KeyError(f"Unknown dimension '{dimension_id}'")

    def qualified_characteristic_ids(self) -> list[str]:
        """All characteristics as ``"dimension.characteristic"`` strings."""
        return [f"{d.id}.{c.id}" for d in self.dimensions for c in d.characteristics]


class Rubric(_Base):
    """What the coder (human or LLM) is given: the taxonomy plus coding guidance."""

    taxonomy: Taxonomy
    instructions: str = Field(
        default="", description="Extra coding guidance appended to the prompt."
    )
    min_evidence_per_dimension: int = Field(default=1, ge=0, le=5)
    max_quote_words: int = Field(default=60, ge=5, le=400)

    @classmethod
    def from_file(cls, path: str | Path) -> Rubric:
        return cls.model_validate(load_structured_file(path))

    def sha256(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()

    def to_prompt(self) -> str:
        """Deterministic plain-text rendering (stable bytes keep the prompt cache warm)."""
        t = self.taxonomy
        lines = [
            f"TAXONOMY: {t.name}",
            f"META-CHARACTERISTIC: {t.meta_characteristic}",
        ]
        if t.description:
            lines.append(f"DESCRIPTION: {t.description}")
        for d in t.dimensions:
            rule = (
                "MUTUALLY EXCLUSIVE - choose exactly one characteristic"
                if d.exclusive
                else "NON-EXCLUSIVE - choose one or more characteristics"
            )
            lines += [
                "",
                f"DIMENSION `{d.id}` ({d.label}) [{rule}]",
                f"  Definition: {d.definition}",
            ]
            for c in d.characteristics:
                lines.append(f"  - `{c.id}` ({c.label}): {c.definition}")
                if c.indicators:
                    lines.append(f"      Indicators: {'; '.join(c.indicators)}")
                if c.exclusions:
                    lines.append(f"      Do NOT use when: {'; '.join(c.exclusions)}")
        if self.instructions.strip():
            lines += ["", "ADDITIONAL CODING INSTRUCTIONS:", self.instructions.strip()]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Nickerson et al. (2013) iteration log and ending conditions
# --------------------------------------------------------------------------- #


class IterationApproach(str, Enum):
    CONCEPTUAL_TO_EMPIRICAL = "conceptual-to-empirical"
    """Deductive: dimensions derived from theory/knowledge, then checked against objects."""
    EMPIRICAL_TO_CONCEPTUAL = "empirical-to-conceptual"
    """Inductive: characteristics derived from a subset of objects, then grouped."""

    @property
    def mode(self) -> str:
        return "deductive" if self is IterationApproach.CONCEPTUAL_TO_EMPIRICAL else "inductive"


class Iteration(_Base):
    number: int = Field(ge=1)
    approach: IterationApproach
    performed_on: date | None = None
    objects_examined: list[str] = Field(
        default_factory=list, description="Ids of papers/objects examined in this iteration."
    )
    dimensions_added: list[str] = Field(default_factory=list)
    dimensions_removed: list[str] = Field(default_factory=list)
    characteristics_added: list[str] = Field(
        default_factory=list, description='Qualified ids, e.g. "dimension.characteristic".'
    )
    characteristics_removed: list[str] = Field(default_factory=list)
    merges: list[str] = Field(default_factory=list, description="Free-text merge records.")
    splits: list[str] = Field(default_factory=list, description="Free-text split records.")
    notes: str = ""

    @property
    def added_anything(self) -> bool:
        return bool(self.dimensions_added or self.characteristics_added)

    @property
    def merged_or_split(self) -> bool:
        return bool(self.merges or self.splits)

    @property
    def changed(self) -> bool:
        return bool(
            self.added_anything
            or self.merged_or_split
            or self.dimensions_removed
            or self.characteristics_removed
        )


SUBJECTIVE_CONDITIONS = ("concise", "robust", "comprehensive", "extendible", "explanatory")


class EndingConditionReport(_Base):
    """Result of checking Nickerson's objective and subjective ending conditions.

    ``None`` means "cannot be decided from the information supplied".
    """

    objective: dict[str, bool | None]
    subjective: dict[str, bool | None]
    messages: list[str] = Field(default_factory=list)

    @property
    def satisfied(self) -> bool:
        return all(v is True for v in self.objective.values()) and all(
            v is True for v in self.subjective.values()
        )


class IterationLog(_Base):
    meta_characteristic: NonEmpty
    iterations: list[Iteration] = Field(default_factory=list)
    subjective_assessment: dict[str, bool] = Field(
        default_factory=dict,
        description=f"Researcher judgement for: {', '.join(SUBJECTIVE_CONDITIONS)}.",
    )

    @model_validator(mode="after")
    def _check(self) -> IterationLog:
        expected = list(range(1, len(self.iterations) + 1))
        if [i.number for i in self.iterations] != expected:
            raise ValueError("Iteration numbers must be consecutive and start at 1.")
        unknown = set(self.subjective_assessment) - set(SUBJECTIVE_CONDITIONS)
        if unknown:
            raise ValueError(f"Unknown subjective conditions: {sorted(unknown)}")
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> IterationLog:
        return cls.model_validate(load_structured_file(path))

    def examined_objects(self) -> set[str]:
        return {o for it in self.iterations for o in it.objects_examined}

    def evaluate_ending_conditions(
        self,
        taxonomy: Taxonomy,
        audit: AuditRun | None = None,
        corpus: Iterable[str] | None = None,
    ) -> EndingConditionReport:
        """Check the eight objective and five subjective ending conditions.

        ``corpus`` is the set of object ids that make up the representative
        sample; ``audit`` supplies the classifications used for the "every
        characteristic is used" condition.
        """
        msgs: list[str] = []
        last = self.iterations[-1] if self.iterations else None
        if last is None:
            msgs.append("The iteration log is empty.")

        if corpus is None:
            sample_ok: bool | None = None
        else:
            missing = set(corpus) - self.examined_objects()
            sample_ok = not missing
            if missing:
                msgs.append(f"{len(missing)} object(s) not yet examined: {sorted(missing)[:10]}")

        if audit is None:
            every_char_used: bool | None = None
        else:
            unused = [k for k, n in audit.characteristic_counts().items() if n == 0]
            every_char_used = not unused
            if unused:
                msgs.append(f"Characteristics with no classified object: {unused}")

        # Conditions 6-8 are guaranteed by the Taxonomy validators; we re-check
        # cross-dimension duplication of characteristic labels as a warning only.
        labels = [c.label.casefold() for d in taxonomy.dimensions for c in d.characteristics]
        dup = sorted({x for x in labels if labels.count(x) > 1})
        if dup:
            msgs.append(f"Characteristic labels repeated across dimensions (check overlap): {dup}")

        objective = {
            "representative_sample_examined": sample_ok,
            "no_object_merged_or_split_last_iteration": (
                None if last is None else not last.merged_or_split
            ),
            "every_characteristic_classifies_an_object": every_char_used,
            "no_new_dimensions_or_characteristics_last_iteration": (
                None if last is None else not last.added_anything
            ),
            "no_dimensions_or_characteristics_merged_or_split_last_iteration": (
                None if last is None else not last.merged_or_split
            ),
            "every_dimension_unique": True,
            "every_characteristic_unique_within_dimension": True,
            "every_cell_unique": True,
        }
        subjective = {k: self.subjective_assessment.get(k) for k in SUBJECTIVE_CONDITIONS}
        if any(v is None for v in subjective.values()):
            msgs.append("Record a researcher judgement for every subjective condition.")
        return EndingConditionReport(objective=objective, subjective=subjective, messages=msgs)


# --------------------------------------------------------------------------- #
# Evidence and audit outputs
# --------------------------------------------------------------------------- #


class PageEvidence(_Base):
    page: int = Field(ge=1, description="1-based page number claimed by the coder.")
    quote: NonEmpty = Field(description="Verbatim text copied from the page.")
    verified: bool = False
    matched_page: int | None = Field(
        default=None, description="Page where the quote was actually found (may differ)."
    )


class Classification(_Base):
    dimension_id: str
    characteristic_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    evidence: list[PageEvidence] = Field(default_factory=list)
    unclassifiable: bool = False
    proposed_characteristic: str | None = Field(
        default=None, description="Suggested new characteristic when nothing fits (inductive cue)."
    )
    issues: list[str] = Field(default_factory=list)

    @property
    def verified(self) -> bool:
        """True when coded, free of issues, and every evidence quote was found in the PDF."""
        return (
            not self.unclassifiable
            and not self.issues
            and bool(self.evidence)
            and all(e.verified for e in self.evidence)
        )

    @property
    def status(self) -> str:
        if self.unclassifiable:
            return "unclassifiable"
        return "verified" if self.verified else "unverified"


class TokenUsage(_Base):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            **{f: getattr(self, f) + getattr(other, f) for f in TokenUsage.model_fields}
        )

    @classmethod
    def from_api(cls, usage: Any) -> TokenUsage:
        if usage is None:
            return cls()
        return cls(**{f: int(getattr(usage, f, 0) or 0) for f in cls.model_fields})


class PaperAudit(_Base):
    paper_id: str
    source_path: str
    sha256: str
    n_pages: int = 0
    pages_sent: int = 0
    truncated: bool = False
    classifications: list[Classification] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    error: str | None = None

    def get(self, dimension_id: str) -> Classification | None:
        return next((c for c in self.classifications if c.dimension_id == dimension_id), None)


class AuditRun(_Base):
    tool_version: str
    model: str
    rubric_sha256: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rubric: Rubric
    papers: list[PaperAudit] = Field(default_factory=list)

    @property
    def taxonomy(self) -> Taxonomy:
        return self.rubric.taxonomy

    def total_usage(self) -> TokenUsage:
        total = TokenUsage()
        for p in self.papers:
            total = total + p.usage
        return total

    def characteristic_counts(self) -> dict[str, int]:
        counts = dict.fromkeys(self.taxonomy.qualified_characteristic_ids(), 0)
        for p in self.papers:
            for c in p.classifications:
                if c.unclassifiable:
                    continue
                for cid in c.characteristic_ids:
                    key = f"{c.dimension_id}.{cid}"
                    if key in counts:
                        counts[key] += 1
        return counts

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> AuditRun:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _ensure_unique(values: Iterable[str], what: str) -> None:
    seen: set[str] = set()
    for v in values:
        if v in seen:
            raise ValueError(f"Duplicate {what}: '{v}'")
        seen.add(v)


def load_structured_file(path: str | Path) -> Any:
    """Load a ``.yaml``/``.yml`` or ``.json`` file."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    if p.suffix.lower() == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported file type '{p.suffix}' (use .yaml, .yml or .json)")
