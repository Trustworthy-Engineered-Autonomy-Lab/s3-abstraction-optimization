# Timed Mountain Car CEGAR runs

`run_mountain_car.py` builds or resumes the complete 2-D Mountain Car
abstraction, using the pretrained DDPG and the existing certified
Taylor/interval reachability code. It performs a bounded CEGAR pass, saves the
nested partition and transition graph, and evaluates recall using
`mountain-car-v3/mc_reach_regions.pkl`.

## Run both requested models

From the repository root:

```powershell
.\cegar\mountain-car\run_mountain_car_models.ps1 -HoursPerModel 3
```

Fresh construction is the default, so an ordinary rerun cannot silently load
an already state-capped model and report a near-zero CEGAR runtime. Use
`-ResumeExisting` to resume the standard output checkpoints.
`-ResumeLegacy` seeds outputs from the supplied `mountaincar_60x60.pkl` and
`mountain_90x90.pkl`; fresh runs are recommended because the supplied files
predate the v3 transition/recall semantics. `-Fresh` remains available as an
explicit synonym for the default behavior.

Each model stops at the first applicable condition:

- 60x60: three hours or `(60+10)^2 = 4,900` live cells.
- 90x90: three hours or `(90+10)^2 = 10,000` live cells.
- completion of the current unknown-cell pass.

OUT (`uid=-1`) is excluded from the state cap. When the state cap wins before
the time limit, the runner prints the CEGAR-only runtime and records
`cegar_runtime_to_state_limit_sec` in the checkpoint and summary.

Outputs:

```text
cegar/mountain-car/artifacts/mountain_car_cegar_60x60.pkl
cegar/mountain-car/artifacts/mountain_car_cegar_60x60.summary.json
cegar/mountain-car/artifacts/mountain_car_cegar_90x90.pkl
cegar/mountain-car/artifacts/mountain_car_cegar_90x90.summary.json
```

The model pickle contains the complete `absys`: root/refined cells, live cell
IDs, Taylor-reachability successors and predecessors, plus final
classification and run metadata. The v3 reach-region pickle is evaluation
data, not an abstraction model.

After every split, the abstraction reruns certified Taylor reachability for
the new children and every predecessor of the replaced parent. Sources outside
that affected set cannot acquire an edge to a child, so rebuilding them would
produce the same edges. The summary records
`cegar_incremental_transition_updates_total` and
`cegar_transition_source_recomputations_total` so this work can be audited.

## One model

```powershell
.\.venv\Scripts\python.exe .\cegar\mountain-car\run_mountain_car.py `
  --grid-size 60 `
  --fresh `
  --time-limit-sec 10800 `
  --max-total-states 4900
```

Evaluate an existing checkpoint without modifying the model:

```powershell
.\.venv\Scripts\python.exe .\cegar\mountain-car\run_mountain_car.py `
  --grid-size 60 `
  --evaluate-only
```

The default graph counterexample backend is exact for universal eventual goal
(`A F goal` / `F goal`) and does not require Spot.

## Evaluate saved models with pyModelChecking

```powershell
.\.venv\Scripts\python.exe `
  .\cegar\mountain-car\evaluate_saved_abstractions.py
```

With no positional arguments, this evaluates both standard saved artifacts
using mountain-car-v3's CTL property `A F goal` and fixed ground truth. Pass
one or more checkpoint paths to evaluate only those files. Each result is
written beside its checkpoint as `*.pymodelchecking.json`; the
`model_checking_time_sec` field times only `CTL.modelcheck(...)`.
