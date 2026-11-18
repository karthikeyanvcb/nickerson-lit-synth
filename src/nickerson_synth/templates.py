"""Starter files written by ``nickerson-synth init``.

The example is deliberately generic (how computer-science papers evaluate
their contributions) so it works on any corpus as a smoke test. Replace it
with your own dimensions.
"""

EXAMPLE_RUBRIC = """\
# nickerson-lit-synth rubric
# Each dimension: exclusive: true  -> exactly one characteristic per paper (ME)
#                 exclusive: false -> one or more characteristics (NE)
# Ids must be lower-case slugs (a-z, 0-9, _ and -).

taxonomy:
  name: Evaluation practices in empirical computer-science research
  meta_characteristic: >
    How a paper substantiates its claims, so that readers can judge the strength
    and reproducibility of the evidence.
  description: Example taxonomy shipped with nickerson-lit-synth. Replace with your own.
  dimensions:
    - id: contribution_type
      label: Contribution type
      definition: The primary kind of contribution the paper claims.
      exclusive: true
      characteristics:
        - id: new_method
          label: New method or algorithm
          definition: Proposes a new technique, model, algorithm or protocol.
          indicators: ["we propose", "we introduce a novel"]
        - id: empirical_study
          label: Empirical study
          definition: Measures or characterises an existing phenomenon or system.
        - id: system_or_tool
          label: System or tool
          definition: Presents an implemented system, framework or tool as the main artefact.
        - id: survey_or_sok
          label: Survey or SoK
          definition: Reviews, systematises or classifies prior work.
        - id: theory
          label: Theoretical analysis
          definition: Main contribution is formal - definitions, theorems or proofs.

    - id: evaluation_method
      label: Evaluation method
      definition: How the claims are evaluated. Choose every method that is used.
      exclusive: false
      characteristics:
        - id: benchmark_experiments
          label: Benchmark experiments
          definition: Quantitative experiments on datasets or benchmarks.
        - id: user_study
          label: User study
          definition: Study with human participants (survey, interviews, lab study).
        - id: case_study
          label: Case study
          definition: In-depth analysis of one or a few real-world instances.
        - id: formal_proof
          label: Formal proof
          definition: Mathematical proof or machine-checked verification.
        - id: simulation
          label: Simulation
          definition: Evaluation in a simulated environment or with synthetic workloads.
        - id: no_evaluation
          label: No evaluation
          definition: The paper does not evaluate its claims.

    - id: artifact_availability
      label: Artifact availability
      definition: Whether code and data needed to reproduce results are publicly released.
      exclusive: true
      characteristics:
        - id: code_and_data
          label: Code and data
          definition: Both source code and data are released (link given).
        - id: code_only
          label: Code only
          definition: Source code is released but data is not.
        - id: not_released
          label: Not released
          definition: No artifact link is given, or the paper states they are unavailable.

instructions: |
  Prefer statements from the abstract, contributions list, evaluation section and
  availability statements. Footnotes with repository links count as evidence.
min_evidence_per_dimension: 1
max_quote_words: 60
"""

EXAMPLE_ITERATION_LOG = """\
# Nickerson et al. (2013) iteration log.
# approach: conceptual-to-empirical (deductive) | empirical-to-conceptual (inductive)
meta_characteristic: >
  How a paper substantiates its claims, so that readers can judge the strength
  and reproducibility of the evidence.
iterations:
  - number: 1
    approach: conceptual-to-empirical
    performed_on: 2026-01-15
    dimensions_added: [contribution_type, evaluation_method]
    notes: Initial dimensions derived from prior methodology literature.
  - number: 2
    approach: empirical-to-conceptual
    performed_on: 2026-01-29
    objects_examined: [paper_a, paper_b, paper_c]
    dimensions_added: [artifact_availability]
    characteristics_added: [evaluation_method.simulation]
    notes: Several papers evaluated only in simulation; reproducibility emerged as a theme.
  - number: 3
    approach: empirical-to-conceptual
    performed_on: 2026-02-12
    objects_examined: [paper_d, paper_e]
    notes: No changes - candidate ending iteration.
subjective_assessment:
  concise: true
  robust: true
  comprehensive: true
  extendible: true
  explanatory: true
"""
