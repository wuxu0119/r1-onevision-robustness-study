# Reliability-Signal Requalification Audit

This note records two later engineering studies that were stopped by frozen,
fail-closed validity checks. They are **not model-efficacy results** and are not
used to change the reported Base, SFT, or RL accuracies.

## Stage A: engineering pilot

The immutable Stage-A audit found:

- expected, observed, and unique records: 10,080 / 10,080 / 10,080;
- missing, duplicate, unexpected, and infrastructure-error records: 0;
- multiple-choice answer parseability: 89.83% over 8,064 rows;
- free-response answer parseability: 59.62% over 2,016 rows;
- 123 of 126 parseability cells below their frozen thresholds;
- 5 of 6 verbal-confidence cells below their frozen thresholds.

The authenticated decision was **NO-GO**. Efficacy analysis was disabled, so
Stage A supports only an engineering-integrity claim: the workload completed,
but the candidate signals were not valid enough for accuracy, calibration,
ranking, or intervention-benefit claims.

## Stage B0 v7: scorer-integrity canaries

A later excluded-canary audit compared the proposed compact option scorer with
a full-sequence scoring oracle. It returned **NO-GO** with 53 authenticated
issues:

| Audit level | Base | SFT | RL | Shared | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Role equivalence | 23 | 15 | 8 | 0 | 46 |
| Aggregate equivalence | 0 | 0 | 0 | 5 | 5 |
| Aggregate winner mismatch | 0 | 0 | 0 | 2 | 2 |
| Total | 23 | 15 | 8 | 7 | 53 |

No launch lock was issued and no formal Stage-B0 efficacy run was started.
Changing the tolerance after observing these failures would invalidate the
pre-specified equivalence test.

## Future full-sequence protocol

The frozen oracle-only accounting estimates:

| Workload | Physical calls |
| --- | ---: |
| Generations | 10,848 |
| Full-sequence scoring forwards | 108,198 |
| Total | 119,046 |
| Frozen compact plan | 25,536 |
| Total-call multiplier | 4.66× |

A future run must use a fresh versioned bundle, frozen source and runtime
inventories, documented processor resolution, and excluded canaries that return
GO before any formal efficacy analysis is authorized.

Machine-readable summaries are in
`results/analysis/ifsr_stage_a_audit_summary.json` and
`results/analysis/ifsr_stage_b0_v7_audit_summary.json`.
