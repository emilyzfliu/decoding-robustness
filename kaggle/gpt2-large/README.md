# GPT-2 Large Kaggle sweep

These are the two GPU kernel entrypoints used for the remaining GPT-2 Large
robustness sweep. They both run the repository's fixed experiment bundle on a
Tesla T4 and write resumable results under `results_v2/`.

| Kernel | Perturbations |
| --- | --- |
| `split-a` | `char`, `token`, `word` |
| `split-b` | `shuffle`, `typo`, `synonym` |

## Submit

Replace the `id` in the matching metadata file with the Kaggle account that
owns the kernel, then submit with a runtime limit long enough for GPT-2 Large:

```powershell
python -m kaggle kernels push -p .\split-a -t 21600
python -m kaggle kernels push -p .\split-b -t 21600
```

`-t` is the Kaggle kernel runtime limit in seconds. Do not use the default
120-second diagnostic limit for this experiment; it cancels the job while the
first perturbation is still running. Use separate accounts when the account
session limit is unknown.

Each retry removes the previous completion marker and clears both the kernel
and child-driver logs before starting. Existing evaluation files under
`results_v2/` are preserved so interrupted runs can resume. A child-driver
failure prevents `run_complete.json` from being written and causes the kernel
to fail; inspect `five_model_run.log` and `results_v2/run_cross_model.log`.

Do not commit Kaggle credentials, downloaded outputs, or generated `results_v2`
directories. Fetch output only after Kaggle reports `COMPLETE`.
