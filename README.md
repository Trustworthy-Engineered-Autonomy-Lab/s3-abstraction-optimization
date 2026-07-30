# $S^3$: A Smooth Simulation Surrogate for Optimizing Discrete Abstractions of Dynamical Systems

This repository contains the optimization and counterexample-guided abstraction
refinement (CEGAR) experiments associated with **$S^3$: A Smooth Simulation
Surrogate for Optimizing Discrete Abstractions of Dynamical Systems**. It
provides three case studies—Spiral, Mountain Car, and Unicycle—and all necessary
code for reproducing the experiments in the paper.

The code builds finite transition-system abstractions of closed-loop dynamical
systems and checks temporal-logic properties with
[pyModelChecking](https://github.com/albertocasagrande/pyModelChecking). The
abstractions are constructed to overapproximate the concrete dynamics; the
optimization and refinement procedures then seek less conservative models with
fewer spurious behaviors and less nondeterminism.

## Context

Let $s = (X, X_0, f)$ be a concrete transition system and let
$\hat{s}_\theta (\hat{X}, \hat{X}_0, \hat{f})$ be the finite abstraction induced by a quantizer $\psi$ with
parameters $\theta$. The abstraction-design problem studied in the paper is

$$
\min_{\theta \in \Theta}
\sigma^{\leftarrow}(s,\widehat{s}_{\theta})
\quad\text{subject to}\quad
\sigma^{\rightarrow}(s,\widehat{s}_{\theta})=0.
$$

The forward-simulation constraint encodes soundness: every concrete behavior
must be represented by the abstraction. The reverse-simulation objective
measures conservatism in the other direction.

The optimization code uses **S3**, a differentiable finite-horizon surrogate
for the reverse-simulation metric. A rectilinear grid is parameterized by
unconstrained gap weights $\omega_{i,j}$, which are converted into positive,
normalized cell widths:

$$\eta_{i,j}=(\overline{x}_i-\underline{x}_i)\frac{\mathrm{Softplus}(\omega_{i,j})}{\sum_{\ell}\mathrm{Softplus}(\omega_{i,\ell})}.
$$

Taylor-model interval reachability is used to overapproximate each cell's
one-step image. Abstract successors are the cells intersecting that reachable
set, so soundness is maintained while the grid is optimized. The S3 loss
smooths the finite-horizon max operations with temperature-scaled
log-sum-exp terms, making the grid parameters amenable to gradient-based
optimization.

The three case studies in the paper are:

- **Spiral:** a stable two-dimensional linear system. The property asks that
  trajectories remain in the modeled domain and eventually reach the goal
  region.
- **Mountain Car:** the two-dimensional `MountainCarContinuous-v0` dynamics
  controlled by a pretrained DDPG policy. The property asks that the car
  eventually reaches the goal position.
- **Unicycle:** three-dimensional planar position and heading dynamics under a
  deterministic obstacle-avoidance/goal-seeking controller. The property asks
  that the system avoid unsafe regions until it reaches the goal.

See the paper and supporting materials for further details.

## Project structure

```text
.
├── Dockerfile                         # Reproducible Python 3.11 environment
├── README.md                          # Project setup and usage
├── requirements.txt                   # Python dependencies
├── optimization/                      # S3 and optimization baselines
│   ├── spiral/                        # Two-dimensional linear case study
│   ├── mountain-car/                  # DDPG Mountain Car case study
│   └── unicycle/                      # Three-dimensional unicycle case study
│       # Each case-study directory contains:
│       #   main_s3.py                 # S3 grid optimization
│       #   weber_baseline.py          # Grid-aspect-ratio baseline
│       #   proxy_validation.py        # S3 proxy validation experiment
│       #   artifacts/                 # Local caches and experiment data
└── cegar/                             # Property-directed local refinement
    ├── spiral/
    │   ├── smoke_test.py              # Build or analyze a saved abstraction
    │   ├── run_synthetic.py           # Full Spiral CEGAR entry point
    │   ├── evaluate_saved_abstractions.py
    │   └── artifacts/                 # Models, caches, and result JSON
    ├── mountain-car/
    │   ├── smoke_test.py
    │   ├── run_mountain_car.py
    │   ├── evaluate_saved_abstractions.py
    │   └── artifacts/
    └── unicycle/
        ├── smoke_test.py
        ├── refine_whole_space_pi.py
        ├── evaluate_saved_abstractions.py
        └── artifacts/
```

The case-study source files remain beside their entry points because some saved
Python artifacts refer to those module names when they are unpickled.

## Requirements and Installation

Python 3.11 is recommended and is the version used by the Docker image.
Dependencies are listed in `requirements.txt`; the main packages include
NumPy, SciPy, JAX with its CPU backend, SymPy, Stable-Baselines3, and
pyModelChecking.

### Windows PowerShell

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script, allow it for the current shell and
then activate the environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

To leave the environment:

```powershell
deactivate
```

### macOS and Linux

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the `python3.11` executable is not available under that exact name, use a
compatible Python 3 installation and verify it with `python3 --version`.

To leave the environment:

```bash
deactivate
```

### Docker

Build the image from the repository root:

```bash
docker build -t abstraction-training .
```

Open an interactive Bash shell in the image:

```bash
docker run --rm -it abstraction-training /bin/bash
```

The repository copied into the image is located at `/app`:

```bash
cd /app
ls
python -u optimization/spiral/main_s3.py --help
python -u cegar/spiral/smoke_test.py --help
```

To edit files on the host and keep generated artifacts after the container
exits, bind-mount the repository over `/app`.

Windows PowerShell:

```powershell
docker run --rm -it -v "${PWD}:/app" abstraction-training /bin/bash
```

macOS or Linux:

```bash
docker run --rm -it -v "$(pwd):/app" abstraction-training /bin/bash
```

If a minimal base image does not provide Bash, substitute `/bin/sh`. When a
host directory is mounted at `/app`, it intentionally replaces the copy of the
repository baked into the image for that container.

## Running the Rxperiments

Choose one of the three case-study directories:

```text
optimization/spiral
optimization/mountain-car
optimization/unicycle
```

Then run an entry point from that directory. For example:

```bash
cd optimization/spiral
python -u main_s3.py --help
python -u weber_baseline.py --help
python -u proxy_validation.py --help
```

On Windows PowerShell, the equivalent navigation command is
`cd optimization\spiral`. Replace `spiral` with `mountain-car` or `unicycle`
to run the other case studies.

### S3 optimization

```bash
python -u main_s3.py --shape 20 --horizon 3 --steps 100
```

`main_s3.py` creates a uniform starting grid, optionally evaluates it, optimizes
the rectilinear gap parameters with the S3 objective, constructs the final
sound abstraction, and reports its model-checking and abstraction-quality
statistics.

| Argument | Meaning |
|---|---|
| `--shape M` | Number of initial cells per state dimension. This gives an \(M \times M\) grid for Spiral/Mountain Car and an \(M^3\) grid for Unicycle. |
| `--horizon H` | Finite rollout horizon used in the S3 objective. |
| `--temp-1 T` | Temperature for the inner, across-time log-sum-exp. |
| `--temp-2 T` | Temperature for the outer, across-cell log-sum-exp. |
| `--sigma S` | Standard deviation of the initial random gap weights. |
| `--lr R` | Gradient-descent learning rate. |
| `--steps K` | Number of optimization iterations. |
| `--eval-init` / `--no-eval-init` | Enable or skip evaluation of the initial uniform abstraction. |

Use `python -u main_s3.py --help` for the defaults encoded by a particular case
study.

### Proxy validation

```bash
python -u proxy_validation.py --shape 20 --samples 100 --rerun
```

The proxy-validation experiment samples quantization parameters, evaluates S3
and the corresponding finite-horizon reverse-simulation quantities, and
reports correlation, approximation error, and evaluation-time statistics.

| Argument | Meaning |
|---|---|
| `--shape M` | Number of grid cells per state dimension. |
| `--samples K` | Number of sampled parameter vectors. |
| `--sigma S` | Standard deviation used to sample those parameter vectors. |
| `--temp-1 T` | Inner log-sum-exp temperature. |
| `--temp-2 T` | Outer log-sum-exp temperature. |
| `--rerun` / `--no-rerun` | Recompute and save samples, or analyze data already stored under `artifacts/proxy-validation-data/`. |

The first Mountain Car run may need to retrieve the pretrained DDPG policy if
it is not already cached. Reachability data, actor derivatives, and proxy
validation samples are stored below the case study's `artifacts/` directory.

### Weber baseline

```bash
python -u weber_baseline.py --shape 20 --horizon 3
```

This baseline optimizes dimension-wise grid widths/aspect ratios while holding
cell volume fixed, then builds and evaluates the resulting abstraction.

| Argument | Meaning |
|---|---|
| `--shape M` | Reference uniform resolution used to determine the fixed cell volume. |
| `--horizon H` | Horizon used when evaluating the finite-horizon simulation metric. |

### Running CEGAR

Each CEGAR case study exposes a common `smoke_test.py` interface. Despite its
name, this is also the simplest entry point for a longer run: it can either
build and save a new abstraction or load and analyze an existing one.

Choose a case study:

```bash
cd cegar/spiral
# or: cd cegar/mountain-car
# or: cd cegar/unicycle
```

For a two-dimensional case study:

```bash
python -u smoke_test.py --shape 60 --time-limit-sec 10800 --max-total-states 4900 --output artifacts/my_cegar_60x60.pkl
```

For a three-dimensional Unicycle case study:

```bash
python -u smoke_test.py --shape 40 --time-limit-sec 10800 --max-total-states 125000 --output artifacts/my_cegar_40x40x40.pkl
```

These examples use the experiment budget convention

$$
N_{\max} =
\begin{cases}
(M+10)^2, & \text{Spiral and Mountain Car},\\
(M+10)^3, & \text{Unicycle},
\end{cases}
$$

and a three-hour wall-clock limit. The run stops when either limit is reached.
The elapsed refinement time is recorded in the summary; the Spiral and
Mountain Car runners also print the total CEGAR runtime when the state limit is
reached first. Checkpoints and the final abstraction contain the refined
cells, stable cell identifiers, and rebuilt transition relation.


```bash
python -u smoke_test.py --load-model artifacts/my_cegar_60x60.pkl
```

For a saved Unicycle model:

```bash
python -u smoke_test.py --load-model artifacts/my_cegar_40x40x40.pkl
```

Loading a model runs the case-study property through pyModelChecking, records
the model-checking time, and computes recall using the same ground-truth
convention as the corresponding optimization case study. The result summary is
written beside the model as a `*.pymodelchecking.json` file unless result
writing is disabled.

#### Common CEGAR arguments

| Argument | Meaning |
|---|---|
| `--shape M` | Uniform resolution of the initial abstraction. |
| `--load-model PATH` | Skip construction and analyze a previously saved abstraction. |
| `--output PATH` | Destination for the newly built abstraction. |
| `--time-limit-sec S` | Wall-clock refinement limit in seconds. Use `10800` for three hours. |
| `--max-total-states N` | Stop after the abstraction reaches this many non-`OUT` states. |
| `--checkpoint-interval-sec S` | Interval between checkpoint writes. |
| `--max-iters K` | Maximum number of CEGAR refinement iterations. |
| `--max-depth D` | Maximum allowed cell-refinement depth. |
| `--min-size X` | Minimum cell side length permitted during splitting. |
| `--split POLICY` | Cell-splitting policy supported by the case-study implementation. |
| `--backend NAME` | Model-checking backend supported by the runner. |
| `--fresh` / `--no-fresh` | Start from a new uniform model or permit available restart data. |
| `--analyze` / `--no-analyze` | Enable or skip final model checking and recall computation. |
| `--write-results` / `--no-write-results` | Enable or disable writing the analysis JSON. |
| `--gt-cache PATH` | Override the case-study ground-truth/reachability cache used for recall. |

Unicycle additionally supports `--gt-steps` and `--analysis-gt` for its
native-size and fixed-reference ground-truth calculations. Run
`python -u smoke_test.py --help` inside a case-study directory for its exact
options and defaults.

The checked properties are:

- Spiral: universal eventual reachability of the goal while respecting the
  modeled domain.
- Mountain Car: universal eventual reachability of the goal position.
- Unicycle: universal safety-until-goal,
  \(\mathrm{A}(\mathit{safe}\ \mathrm{U}\ \mathit{goal})\).

The case-study directories also retain their longer-running Python and
PowerShell entry points for batch experiments:

| Case study | Python entry point | PowerShell batch runner |
|---|---|---|
| Spiral | `run_synthetic.py` | `run_synthetic_models.ps1` |
| Mountain Car | `run_mountain_car.py` | `run_mountain_car_models.ps1` |
| Unicycle | `refine_whole_space_pi.py` | `run_cegar_models.ps1` |
