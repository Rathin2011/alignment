# Qwen data

Generated JSONL files are ignored by Git. Rebuild them from the released COCA
data using the three commands in `REPRODUCIBILITY.md`.

The pipeline is:

1. sample 500 harmful and 3,000 benign source requests;
2. repeat the harmful requests under deterministic generic wrappers to create
   3,000 harmful and 3,000 benign analyzer records;
3. create one `X -> Z` and one `Z -> decision` example per analyzer record.

The final file therefore contains 12,000 examples. Hashes in
`REPRODUCIBILITY.md` verify that the original experimental data are recovered.
