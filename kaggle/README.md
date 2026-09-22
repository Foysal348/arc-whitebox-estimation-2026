# Kaggle kernels

Heavy runs — 10-network scoring sweeps, Monte-Carlo cumulant teachers, GPU reference chains — run on Kaggle. Each directory
holds a notebook and its `kernel-metadata.json`; estimator variants travel as Kaggle *datasets*.

| Directory | Kernel | What it runs |
|---|---|---|
| `runners/runner_a`, `_b`, `_c` | `whest-runner`, `-b`, `-c` | Score a set of estimator variants on 10 public MLPs against the baked 10⁹-sample truth |
| `estimators/set_a`, `_b`, `_c` | (datasets) | Estimator files plus a `variants.json` mapping tag to file and environment knobs |
| `k4/code`, `run_synthetic`, `run_public`, `local` | `whest-k4teacher`, `-pub` | Monte-Carlo fourth-cumulant teacher: the true (2,2), (3,1) and diagonal slices per layer |
| `dumps/driver`, `code`, `kernel_a`, `kernel_b` | `whest-dump-a`, `-b` | Per-neuron feature dumps of all 100 public MLPs for the residual-structure study |
| `onestep/kernel`, `code` | `whest-onestep` | One-step closure floors and the copula oracle, teacher-forced at 1024×16 |
| `kprop` | `whest-kprop-reference` | ARC's reference K2/K3/K4 chains on real networks: the accuracy ladder |

## Operating rules

* `kprop/local_check.py` executes a notebook's cells locally at small width before it is pushed. **Every kernel is validated
  locally first** — four consecutive unvalidated pushes on 21 Sep 2026 cost about 4.5 hours of Kaggle time.
* Never add a CPU fallback for a CUDA out-of-memory error: the host runs out of memory too and the kernel dies. Catch the
  error and skip that configuration.
* When chaining `python script | tail` in a shell, the exit code belongs to `tail`. Capture the interpreter's own exit code
  before concluding that a local validation passed.
* Two GPU sessions run concurrently at most; CPU kernels are not subject to that limit.

```bash
python -m kaggle kernels push   -p kaggle/runners/runner_a
python -m kaggle kernels status foysalemonshanto/whest-runner
python -m kaggle kernels output foysalemonshanto/whest-runner -p data/kaggle_out/kaggle_out_ang
```
