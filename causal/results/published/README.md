# Published Qwen summaries

These are the compact, one-seed summaries used in the report:

- `qwen_no_sft_summary.json`: untouched-Qwen baseline.
- `qwen_sft_retention_summary.json`: ordinary SFT plus retention baseline.
- `qwen_oracle_enforced_summary.json`: externally enforced positive control.
- `qwen_learned_multipass_summary.json`: learned `X -> Z -> decision` path and
  predicted/oracle/reversed-Z interventions.

Raw model generations are not committed because they are large and contain
harmful requests and responses. The summaries preserve counts, per-dataset
metrics, aggregate metrics, and provenance.
