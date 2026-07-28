# Timed unicycle CEGAR runs

`refine_whole_space_pi.py` is the production runner for the 40³ and 90³
experiments. It builds or resumes a complete abstraction, runs CEGAR for a
bounded amount of time, saves the final partition and transitions, and computes
recall against the matching ground-truth cache.

## What the pickle files contain

- `unicycle_cegar_*x*x*.pkl` is an abstraction checkpoint. Its `absys` entry
  contains the root/refined cell tree (`absys.part`), live cell IDs
  (`absys.part.leaves`), and successor/predecessor transitions (`absys.tr`).
- `gt_safe_unicycle_*x*x*_steps10.pkl` is an evaluation cache. It contains
  simulation-derived safe voxel indices, not an abstraction or transition
  system.
- `*.summary.json` contains the final recall and run/model statistics in a
  lightweight format.

The supplied `unicycle_refinement_model.pkl` is a valid, partially refined 40³
checkpoint. It has 64,000 original roots and 67,748 current leaves. Use it when
you want to continue that earlier run; do not use it for a clean 40³ baseline.

The execution flow is:

1. `main.build_abstraction()` constructs the uniform roots and Taylor
   over-approximation transitions.
2. `classify_all_leaves_once()` computes the globally verified reach-avoid set.
3. CEGAR visits unknown initial cells, validates an abstract counterexample,
   and incrementally splits spurious cells and updates their transitions.
4. The final global classification is evaluated against `gt_safe`.

## Run both experiments

From PowerShell:

```powershell
cd C:\Users\jorda\OneDrive\Desktop\abstraction-training
.\cegar\unicycle\run_cegar_models.ps1 -HoursPerModel 3
```

The default behavior resumes the standard output checkpoint if it already
exists; otherwise it builds a fresh model. To force two clean models:

```powershell
.\cegar\unicycle\run_cegar_models.ps1 -HoursPerModel 3 -Fresh
```

To seed the 40³ output from the supplied partial checkpoint while constructing
90³ normally:

```powershell
.\cegar\unicycle\run_cegar_models.ps1 -HoursPerModel 3 -ResumeLegacy40
```

Each model gets its own three-hour CEGAR budget, so the sequential batch needs
about six hours plus initial construction and final evaluation. The CEGAR timer
starts after model construction and the initial global classification.

Each run also has a live-cell limit of `(m + 10)^3`, where `m` is its initial
grid size. Refinement and finalization therefore begin as soon as either limit
is reached:

- 40³ starts with 64,000 cells and stops at 125,000 live cells or three hours.
- 90³ starts with 729,000 cells and stops at 1,000,000 live cells or three
  hours.

The absorbing OUT state (`uid=-1`) is not a partition cell and is excluded from
these limits. A multiway split is not started if it would exceed the applicable
limit.

Outputs:

```text
cegar/unicycle/artifacts/unicycle_cegar_40x40x40.pkl
cegar/unicycle/artifacts/unicycle_cegar_40x40x40.summary.json
cegar/unicycle/artifacts/unicycle_cegar_90x90x90.pkl
cegar/unicycle/artifacts/unicycle_cegar_90x90x90.summary.json
```

Checkpoints are written atomically every 15 minutes, at the time limit, before
evaluation, and after evaluation. Interrupting with Ctrl+C requests a clean
checkpoint at the next CEGAR iteration boundary.

## Run one model

For example, a fresh 40³ run:

```powershell
.\.venv\Scripts\python.exe .\cegar\unicycle\refine_whole_space_pi.py `
  --grid-size 40 `
  --fresh `
  --time-limit-sec 10800 `
  --max-total-states 125000
```

Omit `--fresh` to resume the standard output checkpoint. To resume the supplied
legacy 40³ checkpoint into the standard output location:

```powershell
.\.venv\Scripts\python.exe .\cegar\unicycle\refine_whole_space_pi.py `
  --grid-size 40 `
  --resume-from .\cegar\unicycle\unicycle_refinement_model.pkl `
  --time-limit-sec 10800
```

## Re-evaluate without refining

```powershell
.\.venv\Scripts\python.exe .\cegar\unicycle\refine_whole_space_pi.py `
  --grid-size 40 `
  --evaluate-only
```

The reported `recall_gt_safe_volume` is the existing codebase's
volume-weighted recall restricted to the configured initial domain
`theta in [-pi/4, pi/4]`. A current abstraction cell contributes only when it
is fully inside that initial domain and all overlapping GT voxels are labeled
safe. The matching 40³ GT cache is used for the 40³ model and the matching 90³
cache for the 90³ model.

The default graph counterexample backend is exact for the configured
reach-avoid formula `(!unsafe) U goal` and does not require the optional Spot
Python package.

## Inspect a saved abstraction

Run this from `cegar/unicycle` so the pickle can resolve the local modules:

```powershell
..\..\.venv\Scripts\python.exe -c "import pickle; p=pickle.load(open('artifacts/unicycle_cegar_40x40x40.pkl','rb')); a=p['absys']; print(len(a.part.leaves)); uid=next(iter(a.part.leaves)); print(uid, a.part.leaves[uid].rect, a.tr.succ[uid])"
```

`a.part.leaves` maps current cell IDs to their nested `CellNode` objects.
`a.tr.succ[uid]["step"]` is the set of successor cell IDs; `a.tr.pred` is the
reverse relation. Cell ID `-1` is the absorbing out-of-domain state.

## Evaluate saved models with pyModelChecking

```powershell
.\.venv\Scripts\python.exe `
  .\cegar\unicycle\evaluate_saved_abstractions.py
```

With no positional arguments, this evaluates both standard saved artifacts
using unicycle-taylor's CTL property `A (safe U goal)`, initial theta region,
and fixed 99x99x99 ground truth. Pass one or more checkpoint paths to
evaluate only those files. Each result is written beside its checkpoint as
`*.pymodelchecking.json`; the `model_checking_time_sec` field times only
`CTL.modelcheck(...)`.
