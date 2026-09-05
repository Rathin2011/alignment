# Application figures

These figures are generated from the verified three-seed toy-safety summary and
the learned multi-pass Qwen intervention evaluation. PNG files are sized for
copying into Google Docs; SVG files are the vector originals.

- `causal_control_vs_ood`: OOD action accuracy versus causal patching accuracy.
- `ood_and_causal_patching`: the same two metrics as a grouped method comparison.
- `safety_tradeoff`: refusal recall versus benign over-refusal. The previously
  missing over-refusal value for Z supervision is 8.82% (three-seed mean).
- `seed_level_results`: all three seeds plus their means.
- `scaffold_removal_trajectory`: accuracy and causal control during staged
  removal of the graph cut, canonicalization, and Z loss.
- `qwen_z_intervention`: predicted, correct-oracle, and reversed-Z policy
  decisions. This is a single Qwen evaluation, not a three-seed result.

Regenerate them with:

```bash
python3 figures/toy_safety/generate.py \
  --toy-summary toy/safety/results/matched_methods_7445302/summary.json \
  --qwen-summary /path/to/qwen2.5-7b_learned_multipass_shared_eval/summary.json \
  --output figures/toy_safety
```

The plotting script uses only Python's standard library and `gnuplot`.
