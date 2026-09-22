# Submissions

One directory per estimator that was packaged and submitted, each containing `estimator.py` and `LICENSE-504aldo`.
Packaged archives are in `packages/`.

| Directory | Submission | Change from the previous one | Adjusted score |
|---|---|---|---|
| `v29/` | #331681 | 504aldo's public V29 chain, verbatim, with an attribution header | 5.418e-9 |
| `v29m8/` | #331686 | `STRASSEN_MIN` 32 → 8: the rank-384/224 families gain Strassen levels | 5.220e-9 |
| `v29m8f5/` | #331732 | `STRASSEN_FIRST` 4 → 5: each grader worker's first call runs at full depth | 5.195e-9 |
| `v31ang/` | #331787 | Spherical-input reparameterisation (κ₄ injection at layer 0 and a₁ rescaling) | **5.137e-9** |
| `v31b/` | #331805 | Re-tuned κ₄ constants — rejected, worse on the grader | 5.194e-9 |

Every variant keeps 504aldo's MIT copyright header; see `../NOTICE`.

## Packaging and submitting

```bash
cd whest-starterkit
python -m uv run whest validate         --estimator ../submissions/v31ang/estimator.py
python -m uv run whest package          --estimator ../submissions/v31ang --output ../submissions/packages/v31ang.tar.gz --yes
python -m uv run whest validate-package ../submissions/packages/v31ang.tar.gz
python -m uv run whest submit           ../submissions/packages/v31ang.tar.gz --yes
python -m uv run python ../experiments/chain/poll_submission.py <submission-id> 180
```

The AIcrowd API key is stored by `whest login` and is never committed. The limit is ten submissions per team per UTC day;
the final ranking uses up to two nominated submissions, re-run on a private suite.
