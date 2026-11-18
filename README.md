# nickerson-lit-synth

[![CI](https://github.com/karthikeyanvcb/nickerson-lit-synth/actions/workflows/ci.yml/badge.svg)](https://github.com/karthikeyanvcb/nickerson-lit-synth/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/nickerson-lit-synth.svg)](https://pypi.org/project/nickerson-lit-synth/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24355538.svg)](https://doi.org/10.5281/zenodo.24355538)

Evidence-verified, LLM-assisted literature coding for **taxonomy** and
**Systematization of Knowledge (SoK)** studies, built around the taxonomy
development method of Nickerson, Varshney & Muntermann (2013).

You define the dimensions and characteristics of your taxonomy in a YAML rubric.
`nickerson-synth` reads a folder of PDFs, asks Claude to code each paper on every
dimension, and **checks every supporting quote against the PDF text**. The output is a
classification matrix you can audit, plus LaTeX tables you can paste into a paper.

It does not replace human coding. Use it for a first coding pass, as a second coder
for inter-rater checks, or to find papers that don't fit your taxonomy yet.

## Features

- **Rubric-as-data.** Pydantic v2 models for dimensions, characteristics, and the
  mutually exclusive (ME) or non-exclusive (NE) rule for each dimension. Duplicate ids
  and labels are rejected.
- **Verbatim evidence.** Each code must cite page-numbered quotes. Quotes are matched
  against PyMuPDF-extracted text after normalising ligatures, quotes, dashes, line-break
  hyphenation and whitespace. A paraphrased quote fails verification. A quote found on
  a different page gets its page number corrected and flagged.
- **Nickerson iteration log.** Record conceptual-to-empirical (deductive) and
  empirical-to-conceptual (inductive) iterations, then check the 8 objective and 5
  subjective ending conditions.
- **Inductive cues.** Papers that fit no characteristic are marked *unclassifiable* and
  can carry a `proposed_characteristic`. Use these to plan your next iteration.
- **Lower cost.** The tool schema and rubric are a shared prefix marked with
  `cache_control: {"type": "ephemeral"}`. One request warms the cache before the batch
  fans out, so later papers pay cache-read prices for the rubric. `--dry-run` estimates
  tokens and cost without calling the API.
- **Exports.** `audit.json`, `matrix.csv`, `evidence.csv`, `coverage.csv`, and LaTeX
  `tabularx` tables (taxonomy overview and a paper × characteristic matrix with `\cite{}`
  keys).
- **Async batches** with bounded concurrency and per-paper error isolation: one bad
  PDF doesn't stop the run.

## Install

```bash
pip install nickerson-lit-synth
export ANTHROPIC_API_KEY=sk-ant-...      # only needed for real (non-dry) runs
```

Requires Python 3.10+.

## Quick start

```bash
nickerson-synth init                      # writes rubric.yaml + iteration_log.yaml
nickerson-synth validate rubric.yaml      # structural checks on your taxonomy
nickerson-synth audit ./pdfs --rubric rubric.yaml --dry-run      # cost estimate
nickerson-synth audit ./pdfs --rubric rubric.yaml --out results  # code the corpus
nickerson-synth export-latex results/audit.json --kind matrix --bibkeys keys.csv -o matrix.tex
nickerson-synth export-latex rubric.yaml --kind taxonomy -o taxonomy.tex
nickerson-synth validate rubric.yaml --log iteration_log.yaml \
    --results results/audit.json --pdfs ./pdfs                   # ending conditions
```

## Rubric format

```yaml
taxonomy:
  name: My taxonomy
  meta_characteristic: The purpose every dimension must serve.
  dimensions:
    - id: evaluation_method          # lower-case slug
      label: Evaluation method
      definition: How the claims are evaluated.
      exclusive: false               # NE: one or more; true = ME: exactly one
      characteristics:
        - id: user_study
          label: User study
          definition: Study with human participants.
          indicators: ["participants", "interviews"]   # optional cues
          exclusions: ["crowdsourced labelling only"]  # optional
        - id: benchmark_experiments
          label: Benchmark experiments
          definition: Quantitative experiments on datasets.
instructions: Optional extra guidance for the coder.
min_evidence_per_dimension: 1
max_quote_words: 60
```

JSON works too. Run `nickerson-synth init` for a full, generic example.

## Outputs

| File | Contents |
|---|---|
| `audit.json` | Full run: rubric, model, per-paper classifications, evidence, token usage |
| `matrix.csv` | One row per paper; for each dimension, the codes and a status (`verified` / `unverified` / `unclassifiable`) |
| `evidence.csv` | One row per quote: claimed page, matched page, verified flag, rationale, issues |
| `coverage.csv` | Papers per characteristic. A zero breaks Nickerson's "every characteristic classifies an object" condition |

A classification is **verified** only if it has the minimum number of quotes, every
quote was found in the PDF, and it broke no rubric rule (unknown characteristic, or the
wrong number of codes on an ME dimension).

`keys.csv` for `--bibkeys` has two columns: `paper_id,bibkey`. The `paper_id` is the
PDF file name without `.pdf`.

## Cost control

```bash
nickerson-synth audit ./pdfs -r rubric.yaml --dry-run --model claude-haiku-5-5
```

- The estimate assumes about 4 characters per token. Real runs print exact usage from
  the API.
- Built-in prices (USD per million tokens) come from Anthropic's pricing page as of
  November 2026 (see `nickerson_synth/pricing.py`). Prices change, so override them
  with `--price-input`, `--price-output`, `--price-cache-write` and
  `--price-cache-read`.
- Caching only applies when the shared prefix is above the model's minimum cacheable
  length (512–4,096 tokens depending on model). Below that, the dry run says the prefix
  won't be cached. Richer definitions and indicators make the rubric both clearer and
  cacheable.
- `--max-chars` caps the paper text sent per PDF (default 180,000 characters, about
  45k tokens). `--limit N` pilots on the first N PDFs.

## Methodological notes

- **Scanned PDFs** have no text layer. Run OCR first (e.g. `ocrmypdf`). They are
  reported as errors, not silently skipped.
- **Prompt injection.** The prompt tells the model to treat paper text as data. Still
  review unverified or low-confidence codes manually.
- **Reproducibility.** `audit.json` stores the tool version, the model id, a SHA-256
  of the rubric and of each PDF. Report these in your methods section.
- **Human oversight.** Report LLM-assisted coding transparently, and validate a sample
  (e.g. Cohen's κ against a human coder) before relying on the results.

## Development

```bash
git clone https://github.com/karthikeyanvcb/nickerson-lit-synth
cd nickerson-lit-synth
pip install -e ".[dev]"
ruff check . && pytest --cov=nickerson_synth
```

The test suite mocks the Anthropic client and generates PDFs on the fly, so it needs
no API key and spends no credits.

## Citation

If you use this tool, please cite the method it implements:

> Nickerson, R. C., Varshney, U., & Muntermann, J. (2013). A method for taxonomy
> development and its application in information systems. *European Journal of
> Information Systems*, 22(3), 336–359. https://doi.org/10.1057/ejis.2012.26

The software itself can be cited using [`CITATION.cff`](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
