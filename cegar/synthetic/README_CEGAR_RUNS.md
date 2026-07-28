# Timed synthetic CEGAR runs

`run_synthetic.py` builds or resumes the complete 2-D abstraction, performs a
bounded CEGAR pass over every currently unknown cell, saves the refined cells
and transitions, and computes final recall.

## Models and evaluation data

- `synthetic_cegar_*x*.pkl` is a complete model checkpoint. The `absys` entry
  contains the nested partition, live cell IDs, and successor/predecessor
  transitions.
- `synthetic-v3/synthetic_reach_regions.pkl` is the supplied 100x100 fixed
  reach-region reference used by synthetic-v3. It is not an abstraction model.
- `*.summary.json` is the lightweight final result, including recall, state and
  edge counts, stop reason, and run parameters.

Recall now exactly follows `synthetic-v3/verification_tools.py`: an abstraction
leaf is ground-truth `goal` only if every overlapping fixed reference cell is
`goal`; the numerator and denominator are the verified and total
ground-truth-goal volumes in the initial domain. The configured initial domain,
as in synthetic-v3, is the full `[-10,10]^2` domain.

Transitions also follow synthetic-v3's AABB boundary convention and include
OUT whenever an image partly leaves the domain. Legacy checkpoints are rebuilt
in memory when loaded. Their old completed-candidate markers are not reused
because they were produced with different counterexample semantics.

## Run both requested experiments

From the repository root:

```powershell
.\cegar\synthetic\run_synthetic_models.ps1 -HoursPerModel 3 -Fresh
```

Omit `-Fresh` to resume the standard output checkpoints when they exist. To
seed the 60x60 output from the supplied legacy model:

```powershell
.\cegar\synthetic\run_synthetic_models.ps1 -HoursPerModel 3 -ResumeLegacy60
```

Each model stops when CEGAR finishes the unknown-cell pass or when the first
resource limit is reached:

- 60x60: three hours or `(60+10)^2 = 4,900` live cells.
- 90x90: three hours or `(90+10)^2 = 10,000` live cells.

The absorbing OUT state (`uid=-1`) is excluded from these limits. A split is
not started if it would exceed the cap.

Outputs:

```text
cegar/synthetic/artifacts/synthetic_cegar_60x60.pkl
cegar/synthetic/artifacts/synthetic_cegar_60x60.summary.json
cegar/synthetic/artifacts/synthetic_cegar_90x90.pkl
cegar/synthetic/artifacts/synthetic_cegar_90x90.summary.json
synthetic-v3/synthetic_reach_regions.pkl
```

The runner checkpoints atomically every 15 minutes, at either resource limit,
before evaluation, and after evaluation. Ctrl+C requests a clean checkpoint at
the next CEGAR iteration boundary. When the state cap is reached before the
time limit, the runner prints the total CEGAR-only runtime and records it as
`cegar_runtime_to_state_limit_sec` in the checkpoint metadata and summary.

## Run or evaluate one model

Fresh 60x60:

```powershell
.\.venv\Scripts\python.exe .\cegar\synthetic\run_synthetic.py `
  --grid-size 60 `
  --fresh `
  --time-limit-sec 10800 `
  --max-total-states 4900
```

Evaluate an existing standard checkpoint without changing it:

```powershell
.\.venv\Scripts\python.exe .\cegar\synthetic\run_synthetic.py `
  --grid-size 60 `
  --evaluate-only
```

The default graph counterexample backend is exact for the configured
reach-avoid formula `(!unsafe) U goal` and does not require Spot.
Counterexample validation propagates all four initial-cell vertices together
for up to 10,000 steps, checking OUT before requiring all four vertices to be
simultaneously in the radius-2 goal, matching synthetic-v3's concrete test.

## Inspect a checkpoint

Run from `cegar/synthetic`:

```powershell
..\..\.venv\Scripts\python.exe -c "import pickle; p=pickle.load(open('artifacts/synthetic_cegar_60x60.pkl','rb')); a=p['absys']; uid=next(iter(a.part.leaves)); print(len(a.part.leaves)); print(uid, a.part.leaves[uid].rect, a.tr.succ[uid])"
```

## Evaluate saved models with pyModelChecking

```powershell
.\.venv\Scripts\python.exe `
  .\cegar\synthetic\evaluate_saved_abstractions.py
```

With no positional arguments, this evaluates both standard saved artifacts
using synthetic-v3's CTL property `A F goal` and fixed ground truth. Pass one
or more checkpoint paths to evaluate only those files. Each result is written
beside its checkpoint as `*.pymodelchecking.json`; the
`model_checking_time_sec` field times only `CTL.modelcheck(...)`.
