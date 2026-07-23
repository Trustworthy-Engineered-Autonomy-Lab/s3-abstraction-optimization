# Mountain Car v3 validation

## Conclusion

`upward_proxy` now follows the paper formulation rather than directly
approximating the graph metric.  For each quantization cell with centroid
`x_c`, it computes at every propagated step

```text
s_k = radius(Reach(A_k)) + ||f^k(x_c) - centroid(Reach(A_k))||,
```

then applies an unnormalized temperature-scaled LSE over the horizon and a
second unnormalized LSE over cells.  There is no witness soft minimum,
distance-to-cell replacement, LSE normalization, or mean-cost blend in this
canonical objective.

The horizon is inclusive as written in Equation (13): passing `horizon=H`
produces `H + 1` scores indexed from zero through `H`.

The earlier graph-aligned experiment is retained under the explicit name
`simulation_aligned_proxy` for ablations only.

## What differs from the other case studies

The abstract score and nested smoothing now match `unicycle-taylor`.  Only the
reachable-set operator is specialized:

- Smooth Mountain Car cells retain first-order Taylor reachability.
- Cells crossing actor ReLU, velocity/position clip, or reset boundaries use
  certified interval propagation.  A Hessian remainder is not valid across
  these surfaces.
- Propagated boxes are softly snapped to the learned, generally nonuniform
  grid before their radius and centroid are evaluated.

The original fixed inflation was applied after the current score and stayed
constant while grid widths changed.  Consequently, the scored set was not the
same quantized reachable set represented by the optimized abstraction.

The comparison directories are not exact gold standards.  In `synthetic-v3`,
`upward_proxy` computes the inner-temperature result but returns a global LSE
of raw step values, making `temp_in` ineffective.  In `unicycle-taylor`, large
theta Taylor remainders are explicitly set to zero during abstraction
construction.  These issues were documented but left outside this change.

## Numerical checks

On twelve 20 x 20 grids with horizon 5, the formal interval/soft proxy had:

| Target | Pearson | Spearman |
|---|---:|---:|
| maximum epsilon | 0.977 | 0.965 |
| mean epsilon | 0.956 | 0.972 |

Those figures use a snapping temperature equal to one half of a uniform cell
width.  Smaller temperatures more closely approximate hard snapping but
produce a less useful piecewise gradient.

On a nonuniform 20 x 20 grid, optimizing the formal proxy changed:

| Quantity | Initial | Optimized |
|---|---:|---:|
| formal proxy | 0.8841 | 0.7118 |
| maximum epsilon | 0.6549 | 0.5344 |
| mean epsilon | 0.3543 | 0.2767 |
| verification recall | 0.000586 | 0.000638 |

On the already-strong uniform 50 x 50 grid, the recommended smoother reach
operator and learning rate 0.02 changed maximum epsilon from 0.4690 to 0.4547,
recall from 0.10547 to 0.10555, and transitions from 12,089 to 12,055.  Mean
epsilon increased slightly (0.24408 to 0.24463), illustrating that a smoothed
outer maximum can improve its worst cell without monotonically improving the
cell average on every local run.

## Recommended configuration

```python
temp_in = 0.01
temp_out = 0.03
norm_order = 2.0
propagation = "interval"
inflation_coefs = np.zeros(2)
snap_temperatures = (domain_ub - domain_lb) / (2 * np.asarray(shape))
lr = 0.02
```

Both temperatures are far below the old `temp_out=1.0`, which behaved nearly
like an average across 2,500 Mountain Car cells and caused gradient
cancellation.  If coordinate weights are introduced, use the same metric
convention when interpreting proxy and simulation results.

Verification recall remains a discrete infinite-horizon CTL quantity, whereas
the proxy is a smooth finite-horizon geometric objective.  It should be
reported and used for checkpoint selection, but monotonic improvement is not
implied by the proposition.  Also, `mc_reach_regions.pkl` is a sampled
vertex-rollout benchmark rather than a formal continuous ground truth.

## Tests

The targeted suite checks the controller derivatives, certified interval
images, abstract successor containment, JAX/NumPy proxy equivalence, finite
gradients, and an analytic one-cell/one-step instance of the paper score:

```powershell
..\.venv\Scripts\python.exe -m unittest -v `
  test_mountain_car_system `
  test_mountain_car_objectives `
  test_mountain_car_abstraction
```

The suite currently contains 18 passing tests.
