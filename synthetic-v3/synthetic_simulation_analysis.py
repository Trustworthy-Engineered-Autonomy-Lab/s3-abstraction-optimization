# =====================================================================
# Description: compute the upward simulation metric between the concrete
# and abstract systems
# =====================================================================

# =====================================================================
# Libraries
# =====================================================================

from __future__ import annotations
from array import array
from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, Mapping
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import (
    Bounds,
    LinearConstraint,
    differential_evolution,
    linprog,
    milp,
    minimize,
)

StateId = Hashable
Vector = NDArray[np.float64]
Cell = tuple[ArrayLike, ArrayLike] | None


# =====================================================================
# Metric struct
# =====================================================================

@dataclass
class UpwardMetricResult:
    # ``epsilon`` is the maximum (worst-case) per-state value.
    epsilon: float
    per_abstract_state: dict[StateId, float]
    concrete_witnesses: dict[StateId, Vector]
    horizon: int
    worst_abstract_state: StateId | None = None
    worst_concrete_witness: Vector | None = None
    epsilon_min: float = 0.0
    epsilon_q1: float = 0.0
    epsilon_median: float = 0.0
    epsilon_q3: float = 0.0
    epsilon_mean: float = 0.0


def _epsilon_statistics(values: Vector) -> tuple[float, float, float, float, float]:
    """Return min, first quartile, median, third quartile, and mean."""
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    return (
        float(np.min(values)),
        float(q1),
        float(median),
        float(q3),
        float(np.mean(values)),
    )


# =====================================================================
# Approximation
# =====================================================================

class UpwardSimulationEstimator:
    """
    Finite-horizon approximation of the directed simulation metric

        abstract system  --->  concrete deterministic system.

    cells[q] = (lower_bound, upper_bound)
    successors[q] = iterable of successor abstract state IDs.
    """

    def __init__(
        self,
        f: Callable[[Vector], ArrayLike],
        cells: Mapping[StateId, Cell],
        successors: Mapping[StateId, Iterable[StateId]],
        *,
        loss: str = "box",
        norm_order: float = np.inf,
    ) -> None:
        self.f = f
        self.cells = cells
        self.states = tuple(cells.keys())
        self.index = {q: i for i, q in enumerate(self.states)}

        if not self.states:
            raise ValueError("At least one abstract state is required.")

        bounded_states = [
            q for q in self.states
            if cells[q] is not None
        ]
        oob_states = [
            q for q in self.states
            if cells[q] is None
        ]

        if not bounded_states:
            raise ValueError("At least one bounded cell is required.")

        if len(oob_states) > 1:
            raise ValueError(
                "This implementation assumes at most one out-of-bounds state."
            )

        self.oob_state = oob_states[0] if oob_states else None
        self.oob_index = (
            self.index[self.oob_state]
            if self.oob_state is not None
            else None
        )

        # Infer the concrete dimension from the first ordinary cell.
        first_cell = cells[bounded_states[0]]
        assert first_cell is not None
        self.n = np.asarray(first_cell[0], dtype=float).shape[0]

        number_of_states = len(self.states)

        # NaN rows are used for states without box geometry.
        self.lower = np.full(
            (number_of_states, self.n),
            np.nan,
            dtype=float,
        )
        self.upper = np.full(
            (number_of_states, self.n),
            np.nan,
            dtype=float,
        )

        self.bounded_mask = np.zeros(number_of_states, dtype=bool)

        for q in bounded_states:
            i = self.index[q]
            cell = cells[q]
            assert cell is not None

            lower = np.asarray(cell[0], dtype=float)
            upper = np.asarray(cell[1], dtype=float)

            if lower.shape != (self.n,) or upper.shape != (self.n,):
                raise ValueError(
                    f"Cell {q!r} has inconsistent dimension."
                )

            if np.any(lower > upper):
                raise ValueError(
                    f"Cell {q!r} has lower bounds above upper bounds."
                )

            self.lower[i] = lower
            self.upper[i] = upper
            self.bounded_mask[i] = True

        self.centers = 0.5 * (self.lower + self.upper)

        # Bounding box of the modeled domain.
        #
        # This equals the domain itself when the ordinary cells partition
        # one axis-aligned rectangle.
        self.domain_lower = np.min(
            self.lower[self.bounded_mask],
            axis=0,
        )
        self.domain_upper = np.max(
            self.upper[self.bounded_mask],
            axis=0,
        )

        self.global_lower = self.domain_lower.copy()
        self.global_upper = self.domain_upper.copy()

        # Store the graph in compressed sparse row (CSR) form.  A Python list
        # containing one small ndarray per state has substantial overhead for
        # large abstractions (often more than the edges themselves).
        self.successor_indptr = np.empty(number_of_states + 1, dtype=np.int64)
        self.successor_indptr[0] = 0
        flat_successors = array("q")

        for i, q in enumerate(self.states):
            q_successors = tuple(successors[q])

            if not q_successors:
                raise ValueError(
                    f"Abstract state {q!r} has no successor."
                )

            unknown = [
                r for r in q_successors
                if r not in self.index
            ]

            if unknown:
                raise KeyError(
                    f"State {q!r} has unknown successors: {unknown!r}"
                )

            flat_successors.extend(self.index[r] for r in q_successors)
            self.successor_indptr[i + 1] = len(flat_successors)

        self.successor_indices = np.frombuffer(
            flat_successors,
            dtype=np.int64,
        )

        if loss not in {"box", "center"}:
            raise ValueError("loss must be 'box' or 'center'.")

        if loss == "center" and self.oob_state is not None:
            raise ValueError(
                "Center loss is undefined for the out-of-bounds state. "
                "Use loss='box' or define a custom OOB output."
            )

        self.loss = loss
        self.norm_order = norm_order

    def observation_costs(self, x: ArrayLike) -> Vector:
        """
        Compute ell(q, x) for every abstract state q.

        Ordinary states use distance to their boxes. The OOB state uses
        distance to the complement of the modeled rectangular domain.
        """
        x_arr = np.asarray(x, dtype=float)

        if x_arr.shape != (self.n,):
            raise ValueError(f"x must have shape ({self.n},).")

        return self._observation_costs_at_indices(
            x_arr,
            np.arange(len(self.states), dtype=np.int64),
        )

    def _observation_costs_at_indices(
        self,
        x: ArrayLike,
        indices: NDArray[np.int64],
    ) -> Vector:
        """Compute observation costs only for the requested states."""
        x_arr = np.asarray(x, dtype=float)
        indices = np.asarray(indices, dtype=np.int64)

        if x_arr.shape != (self.n,):
            raise ValueError(f"x must have shape ({self.n},).")

        costs = np.empty(indices.size, dtype=float)
        bounded_positions = self.bounded_mask[indices]
        bounded_indices = indices[bounded_positions]

        if self.loss == "center":
            residual = self.centers[bounded_indices] - x_arr
        else:
            residual = np.maximum(
                np.maximum(
                    self.lower[bounded_indices] - x_arr,
                    x_arr - self.upper[bounded_indices],
                ),
                0.0,
            )

        costs[bounded_positions] = np.linalg.norm(
            residual,
            ord=self.norm_order,
            axis=1,
        )

        oob_positions = ~bounded_positions
        if np.any(oob_positions):
            # Boundary points have zero distance to the complement.
            strictly_inside = np.all(
                (x_arr > self.domain_lower)
                & (x_arr < self.domain_upper)
            )

            if strictly_inside:
                margin_to_each_face = np.minimum(
                    x_arr - self.domain_lower,
                    self.domain_upper - x_arr,
                )
                oob_cost = float(np.min(margin_to_each_face))
            else:
                oob_cost = 0.0
            costs[oob_positions] = oob_cost

        return costs

    def _successors_of_indices(
        self,
        indices: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        """Return the sorted union of successors of ``indices``."""
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 1:
            i = int(indices[0])
            start = self.successor_indptr[i]
            stop = self.successor_indptr[i + 1]
            return self.successor_indices[start:stop]

        pieces = [
            self.successor_indices[
                self.successor_indptr[i]:self.successor_indptr[i + 1]
            ]
            for i in indices
        ]
        return np.unique(np.concatenate(pieces))

    def _reachable_levels(
        self,
        initial_index: int,
        horizon: int,
    ) -> list[NDArray[np.int64]]:
        """Abstract states reachable at each exact time 0, ..., horizon."""
        if horizon < 0:
            raise ValueError("horizon must be nonnegative.")

        levels = [np.asarray([initial_index], dtype=np.int64)]
        for _ in range(horizon):
            levels.append(self._successors_of_indices(levels[-1]))
        return levels

    def rollout(self, x0: ArrayLike, horizon: int) -> NDArray[np.float64]:
        """
        Compute x_t = f^t(x0), t = 0, ..., horizon.
        """
        if horizon < 0:
            raise ValueError("horizon must be nonnegative.")

        trajectory = np.empty((horizon + 1, self.n), dtype=float)
        trajectory[0] = np.asarray(x0, dtype=float)

        for t in range(horizon):
            next_state = np.asarray(self.f(trajectory[t]), dtype=float)

            if next_state.shape != (self.n,):
                raise ValueError(
                    f"f returned shape {next_state.shape}; expected ({self.n},)."
                )
            if not np.all(np.isfinite(next_state)):
                raise FloatingPointError("f produced a non-finite state.")

            trajectory[t + 1] = next_state

        return trajectory

    def finite_horizon_values(
        self,
        x0: ArrayLike,
        horizon: int,
    ) -> Vector:
        """
        Compute V_H(q, x0) simultaneously for all abstract states q.

        The backward pass evaluates

            V_{h+1}(q, x)
              = max(ell(q, x),
                    max_{q' in Succ(q)} V_h(q', f(x))).
        """
        trajectory = self.rollout(x0, horizon)

        # Terminal time cost.
        values = self.observation_costs(trajectory[horizon])

        # Backward dynamic programming over the abstract graph.
        for t in range(horizon - 1, -1, -1):
            worst_successor = np.empty(len(self.states), dtype=float)

            edge_values = values[self.successor_indices]
            worst_successor[:] = np.maximum.reduceat(
                edge_values,
                self.successor_indptr[:-1],
            )

            values = np.maximum(
                self.observation_costs(trajectory[t]),
                worst_successor,
            )

        return values

    def _finite_horizon_value_for_levels(
        self,
        x0: ArrayLike,
        levels: list[NDArray[np.int64]],
    ) -> float:
        """
        Evaluate one initial abstract state using only its reachable subgraph.

        For deterministic concrete dynamics, the backward max over paths is
        equivalent to a max over the states reachable at each exact time.
        """
        trajectory = self.rollout(x0, len(levels) - 1)
        value = 0.0
        for x_t, indices in zip(trajectory, levels):
            costs = self._observation_costs_at_indices(x_t, indices)
            value = max(value, float(np.max(costs, initial=0.0)))
        return value

    def estimate(
        self,
        horizon: int,
        *,
        initial_abstract_states: Iterable[StateId] | None = None,
        restrict_witness_to_own_cell: bool = True,
        maxiter: int = 250,
        popsize: int = 15,
        seed: int = 0,
        verbose: bool = False,
        progress_every: int = 1000,
        store_per_state: bool = True,
        store_witnesses: bool = True,
    ) -> UpwardMetricResult:
        """
        Approximate

            max_q min_x V_H(q, x).

        When restrict_witness_to_own_cell=True, x is constrained to R_q.
        The storage flags can be disabled when only the global worst case is
        needed for a large model.
        """
        if initial_abstract_states is None:
            initial_states = tuple(
                q for q in self.states
                if self.cells[q] is not None
            )
        else:
            initial_states = tuple(initial_abstract_states)

        per_state: dict[StateId, float] = {}
        witnesses: dict[StateId, Vector] = {}
        epsilon = -np.inf
        worst_state: StateId | None = None
        worst_witness: Vector | None = None
        epsilon_values = np.empty(len(initial_states), dtype=float)

        for count, q in enumerate(initial_states):
            if q not in self.index:
                raise KeyError(f"Unknown initial abstract state: {q!r}")

            if self.cells[q] is None:
                raise ValueError(
                    f"Cannot optimize a witness for OOB state {q!r} "
                    "without a bounded concrete search region."
                )

            i = self.index[q]
            levels = self._reachable_levels(i, horizon)

            if restrict_witness_to_own_cell:
                lower = self.lower[i]
                upper = self.upper[i]
            else:
                lower = self.global_lower
                upper = self.global_upper

            bounds = list(zip(lower.tolist(), upper.tolist()))

            def objective(x: Vector) -> float:
                return self._finite_horizon_value_for_levels(x, levels)

            # Global, derivative-free search. The objective contains maxima
            # and is generally nonsmooth.
            global_result = differential_evolution(
                objective,
                bounds=bounds,
                seed=seed + count,
                maxiter=maxiter,
                popsize=popsize,
                polish=False,
                updating="immediate",
                workers=1,
            )

            # A bounded local refinement is often useful.
            local_result = minimize(
                objective,
                global_result.x,
                method="Powell",
                bounds=bounds,
                options={
                    "maxiter": 100,
                    "xtol": 1e-7,
                    "ftol": 1e-7,
                },
            )

            if local_result.fun < global_result.fun:
                best_x = np.asarray(local_result.x, dtype=float)
                best_value = float(local_result.fun)
            else:
                best_x = np.asarray(global_result.x, dtype=float)
                best_value = float(global_result.fun)

            if store_per_state:
                per_state[q] = best_value
            if store_witnesses:
                witnesses[q] = best_x
            if best_value > epsilon:
                epsilon = best_value
                worst_state = q
                worst_witness = best_x.copy()
            epsilon_values[count] = best_value

            if verbose and (
                (count + 1) % progress_every == 0
                or count + 1 == len(initial_states)
            ):
                print(f"Solved {count + 1} / {len(initial_states)} states")

        if not initial_states:
            epsilon = 0.0
        epsilon_min, epsilon_q1, epsilon_median, epsilon_q3, epsilon_mean = (
            _epsilon_statistics(epsilon_values)
        )

        return UpwardMetricResult(
            epsilon=float(epsilon),
            per_abstract_state=per_state,
            concrete_witnesses=witnesses,
            horizon=horizon,
            worst_abstract_state=worst_state,
            worst_concrete_witness=worst_witness,
            epsilon_min=epsilon_min,
            epsilon_q1=epsilon_q1,
            epsilon_median=epsilon_median,
            epsilon_q3=epsilon_q3,
            epsilon_mean=epsilon_mean,
        )

    def estimate_affine(
        self,
        dynamics_matrix: ArrayLike,
        dynamics_offset: ArrayLike,
        horizon: int,
        *,
        initial_abstract_states: Iterable[StateId] | None = None,
        restrict_witness_to_own_cell: bool = True,
        validate_dynamics: bool = True,
        solver_options: Mapping[str, object] | None = None,
        verbose: bool = False,
        progress_every: int = 1000,
        store_per_state: bool = True,
        store_witnesses: bool = True,
    ) -> UpwardMetricResult:
        """Compute the finite-horizon metric exactly for affine dynamics.

        This path applies when ``f(x) = dynamics_matrix @ x +
        dynamics_offset``, the observation loss is ``"box"`` or ``"center"``,
        and the infinity norm is used.  It solves a small LP per initial state;
        states whose reachable set contains OOB use a small LP enumeration for
        one OOB time or a MILP for multiple OOB times.

        Unlike :meth:`estimate`, this is not a stochastic approximation.
        Set ``store_per_state=False`` and ``store_witnesses=False`` for the
        lowest-memory global-metric computation on very large models; the
        worst state and its witness are still returned.
        """
        if horizon < 0:
            raise ValueError("horizon must be nonnegative.")
        if self.norm_order != np.inf:
            raise ValueError(
                "estimate_affine currently requires norm_order=np.inf."
            )

        matrix = np.asarray(dynamics_matrix, dtype=float)
        offset = np.asarray(dynamics_offset, dtype=float)
        if matrix.shape != (self.n, self.n):
            raise ValueError(
                f"dynamics_matrix must have shape ({self.n}, {self.n})."
            )
        if offset.shape != (self.n,):
            raise ValueError(f"dynamics_offset must have shape ({self.n},).")

        if validate_dynamics:
            probes = [0.5 * (self.domain_lower + self.domain_upper)]
            probes.extend(
                probes[0] + np.eye(self.n)[k]
                for k in range(self.n)
            )
            for probe in probes:
                expected = matrix @ probe + offset
                actual = np.asarray(self.f(probe), dtype=float)
                if actual.shape != (self.n,) or not np.allclose(
                    actual,
                    expected,
                    rtol=1e-9,
                    atol=1e-9,
                ):
                    raise ValueError(
                        "The supplied affine dynamics do not match f. Pass "
                        "validate_dynamics=False only if this check is "
                        "intentionally inappropriate."
                    )

        if initial_abstract_states is None:
            initial_states = tuple(
                q for q in self.states if self.cells[q] is not None
            )
        else:
            initial_states = tuple(initial_abstract_states)

        # x_t = trajectory_matrices[t] @ x_0 + trajectory_offsets[t].
        trajectory_matrices = [np.eye(self.n)]
        trajectory_offsets = [np.zeros(self.n)]
        for _ in range(horizon):
            trajectory_matrices.append(matrix @ trajectory_matrices[-1])
            trajectory_offsets.append(
                matrix @ trajectory_offsets[-1] + offset
            )

        (
            aggregate_lowers,
            aggregate_uppers,
            bounded_reachable,
            oob_reachable,
        ) = self._reachable_observation_aggregates(horizon)

        per_state: dict[StateId, float] = {}
        witnesses: dict[StateId, Vector] = {}
        options = dict(solver_options or {})
        epsilon = -np.inf
        worst_state: StateId | None = None
        worst_witness: Vector | None = None
        epsilon_values = np.empty(len(initial_states), dtype=float)

        for count, q in enumerate(initial_states):
            if q not in self.index:
                raise KeyError(f"Unknown initial abstract state: {q!r}")
            if self.cells[q] is None:
                raise ValueError(
                    f"Cannot optimize a witness for OOB state {q!r} "
                    "without a bounded concrete search region."
                )

            i = self.index[q]
            if restrict_witness_to_own_cell:
                search_lower = self.lower[i]
                search_upper = self.upper[i]
            else:
                search_lower = self.global_lower
                search_upper = self.global_upper

            value, witness = self._solve_affine_state(
                i,
                aggregate_lowers,
                aggregate_uppers,
                bounded_reachable,
                oob_reachable,
                trajectory_matrices,
                trajectory_offsets,
                search_lower,
                search_upper,
                options,
            )
            if store_per_state:
                per_state[q] = value
            if store_witnesses:
                witnesses[q] = witness
            if value > epsilon:
                epsilon = value
                worst_state = q
                worst_witness = witness.copy()
            epsilon_values[count] = value

            if verbose and (
                (count + 1) % progress_every == 0
                or count + 1 == len(initial_states)
            ):
                print(f"Solved {count + 1} / {len(initial_states)} states")

        if not initial_states:
            epsilon = 0.0
        epsilon_min, epsilon_q1, epsilon_median, epsilon_q3, epsilon_mean = (
            _epsilon_statistics(epsilon_values)
        )

        return UpwardMetricResult(
            epsilon=float(epsilon),
            per_abstract_state=per_state,
            concrete_witnesses=witnesses,
            horizon=horizon,
            worst_abstract_state=worst_state,
            worst_concrete_witness=worst_witness,
            epsilon_min=epsilon_min,
            epsilon_q1=epsilon_q1,
            epsilon_median=epsilon_median,
            epsilon_q3=epsilon_q3,
            epsilon_mean=epsilon_mean,
        )

    def _reachable_observation_aggregates(
        self,
        horizon: int,
    ) -> tuple[
        list[NDArray[np.float64]],
        list[NDArray[np.float64]],
        list[NDArray[np.bool_]],
        list[NDArray[np.bool_]],
    ]:
        """Propagate coordinate extrema through the graph for all states.

        For the infinity norm, the maximum observation distance over a set of
        boxes depends only on the maximum lower bound and minimum upper bound
        in each coordinate.  Propagating those quantities once per horizon
        step avoids a separate reachable-set expansion for every source.
        """
        if self.loss == "box":
            initial_lower = self.lower.copy()
            initial_upper = self.upper.copy()
        else:
            initial_lower = self.centers.copy()
            initial_upper = self.centers.copy()

        initial_lower[~self.bounded_mask] = -np.inf
        initial_upper[~self.bounded_mask] = np.inf
        aggregate_lowers = [initial_lower]
        aggregate_uppers = [initial_upper]
        bounded_reachable = [self.bounded_mask.copy()]
        oob_reachable = [~self.bounded_mask]

        for _ in range(horizon):
            previous_lower = aggregate_lowers[-1]
            previous_upper = aggregate_uppers[-1]
            next_lower = np.empty_like(previous_lower)
            next_upper = np.empty_like(previous_upper)

            for k in range(self.n):
                next_lower[:, k] = np.maximum.reduceat(
                    previous_lower[self.successor_indices, k],
                    self.successor_indptr[:-1],
                )
                next_upper[:, k] = np.minimum.reduceat(
                    previous_upper[self.successor_indices, k],
                    self.successor_indptr[:-1],
                )

            next_bounded = np.logical_or.reduceat(
                bounded_reachable[-1][self.successor_indices],
                self.successor_indptr[:-1],
            )
            next_oob = np.logical_or.reduceat(
                oob_reachable[-1][self.successor_indices],
                self.successor_indptr[:-1],
            )
            aggregate_lowers.append(next_lower)
            aggregate_uppers.append(next_upper)
            bounded_reachable.append(next_bounded)
            oob_reachable.append(next_oob)

        return (
            aggregate_lowers,
            aggregate_uppers,
            bounded_reachable,
            oob_reachable,
        )

    def _solve_affine_state(
        self,
        initial_index: int,
        aggregate_lowers: list[NDArray[np.float64]],
        aggregate_uppers: list[NDArray[np.float64]],
        bounded_reachable: list[NDArray[np.bool_]],
        oob_reachable: list[NDArray[np.bool_]],
        trajectory_matrices: list[NDArray[np.float64]],
        trajectory_offsets: list[Vector],
        search_lower: Vector,
        search_upper: Vector,
        solver_options: dict[str, object],
    ) -> tuple[float, Vector]:
        """Build and solve the LP/MILP for one initial abstract state."""
        # Variables are [x_0 (n entries), epsilon].
        objective = np.zeros(self.n + 1)
        objective[-1] = 1.0
        rows: list[Vector] = []
        rhs: list[float] = []
        oob_times: list[int] = []

        for t in range(len(trajectory_matrices)):
            transform = trajectory_matrices[t]
            transform_offset = trajectory_offsets[t]

            if bounded_reachable[t][initial_index]:
                target_lower = aggregate_lowers[t][initial_index]
                target_upper = aggregate_uppers[t][initial_index]

                for k in range(self.n):
                    lower_row = np.zeros(self.n + 1)
                    lower_row[:self.n] = -transform[k]
                    lower_row[-1] = -1.0
                    rows.append(lower_row)
                    rhs.append(transform_offset[k] - target_lower[k])

                    upper_row = np.zeros(self.n + 1)
                    upper_row[:self.n] = transform[k]
                    upper_row[-1] = -1.0
                    rows.append(upper_row)
                    rhs.append(target_upper[k] - transform_offset[k])

            if oob_reachable[t][initial_index]:
                oob_times.append(t)

        bounds = list(zip(search_lower, search_upper)) + [(0.0, None)]
        base_rows = np.asarray(rows, dtype=float)
        base_rhs = np.asarray(rhs, dtype=float)

        if not oob_times:
            result = linprog(
                objective,
                A_ub=base_rows,
                b_ub=base_rhs,
                bounds=bounds,
                method="highs",
                options=solver_options,
            )
        elif len(oob_times) == 1:
            result = self._solve_single_oob_time(
                objective,
                base_rows,
                base_rhs,
                bounds,
                trajectory_matrices[oob_times[0]],
                trajectory_offsets[oob_times[0]],
                solver_options,
            )
        else:
            result = self._solve_multiple_oob_times(
                objective,
                base_rows,
                base_rhs,
                bounds,
                oob_times,
                trajectory_matrices,
                trajectory_offsets,
                search_lower,
                search_upper,
                solver_options,
            )

        if not result.success:
            raise RuntimeError(
                f"Affine simulation-metric optimization failed: "
                f"{result.message}"
            )
        return max(0.0, float(result.fun)), np.asarray(result.x[:self.n])

    def _oob_face_rows(
        self,
        transform: NDArray[np.float64],
        transform_offset: Vector,
    ) -> list[tuple[Vector, float]]:
        """Rows for the disjunction distance-to-domain-complement <= eps."""
        faces: list[tuple[Vector, float]] = []
        for k in range(self.n):
            lower_face = np.zeros(self.n + 1)
            lower_face[:self.n] = transform[k]
            lower_face[-1] = -1.0
            faces.append(
                (lower_face, self.domain_lower[k] - transform_offset[k])
            )

            upper_face = np.zeros(self.n + 1)
            upper_face[:self.n] = -transform[k]
            upper_face[-1] = -1.0
            faces.append(
                (upper_face, transform_offset[k] - self.domain_upper[k])
            )
        return faces

    def _solve_single_oob_time(
        self,
        objective: Vector,
        base_rows: NDArray[np.float64],
        base_rhs: Vector,
        bounds: list[tuple[float | None, float | None]],
        transform: NDArray[np.float64],
        transform_offset: Vector,
        solver_options: dict[str, object],
    ):
        """Solve one OOB disjunction by trying its 2n domain faces."""
        best_result = None
        for face_row, face_rhs in self._oob_face_rows(
            transform,
            transform_offset,
        ):
            result = linprog(
                objective,
                A_ub=np.vstack((base_rows, face_row)),
                b_ub=np.append(base_rhs, face_rhs),
                bounds=bounds,
                method="highs",
                options=solver_options,
            )
            if result.success and (
                best_result is None or result.fun < best_result.fun
            ):
                best_result = result

        if best_result is None:
            raise RuntimeError("All OOB face subproblems were infeasible.")
        return best_result

    def _solve_multiple_oob_times(
        self,
        objective: Vector,
        base_rows: NDArray[np.float64],
        base_rhs: Vector,
        bounds: list[tuple[float | None, float | None]],
        oob_times: list[int],
        trajectory_matrices: list[NDArray[np.float64]],
        trajectory_offsets: list[Vector],
        search_lower: Vector,
        search_upper: Vector,
        solver_options: dict[str, object],
    ):
        """Use binary face selectors when OOB occurs at several times."""
        faces_per_time = 2 * self.n
        number_of_binaries = len(oob_times) * faces_per_time
        number_of_variables = self.n + 1 + number_of_binaries
        extended_objective = np.zeros(number_of_variables)
        extended_objective[:self.n + 1] = objective

        constraint_rows = [
            np.pad(row, (0, number_of_binaries)) for row in base_rows
        ]
        constraint_lower = [-np.inf] * len(constraint_rows)
        constraint_upper = base_rhs.tolist()

        binary_offset = self.n + 1
        for time_number, t in enumerate(oob_times):
            face_rows = self._oob_face_rows(
                trajectory_matrices[t],
                trajectory_offsets[t],
            )
            selector_row = np.zeros(number_of_variables)

            for face_number, (face_row, face_rhs) in enumerate(face_rows):
                selector = (
                    binary_offset
                    + time_number * faces_per_time
                    + face_number
                )
                # A safe, tight big-M is the maximum violation over the x0
                # search box at epsilon=0.
                coefficients = face_row[:self.n]
                max_lhs = np.sum(
                    np.where(
                        coefficients >= 0.0,
                        coefficients * search_upper,
                        coefficients * search_lower,
                    )
                )
                big_m = max(0.0, float(max_lhs - face_rhs))

                row = np.zeros(number_of_variables)
                row[:self.n + 1] = face_row
                row[selector] = big_m
                constraint_rows.append(row)
                constraint_lower.append(-np.inf)
                constraint_upper.append(face_rhs + big_m)
                selector_row[selector] = 1.0

            # At least one domain face must be within epsilon.
            constraint_rows.append(selector_row)
            constraint_lower.append(1.0)
            constraint_upper.append(np.inf)

        variable_lower = np.asarray(
            [bound[0] for bound in bounds] + [0.0] * number_of_binaries,
            dtype=float,
        )
        variable_upper = np.asarray(
            [
                np.inf if bound[1] is None else bound[1]
                for bound in bounds
            ]
            + [1.0] * number_of_binaries,
            dtype=float,
        )
        integrality = np.zeros(number_of_variables, dtype=np.uint8)
        integrality[binary_offset:] = 1

        return milp(
            extended_objective,
            integrality=integrality,
            bounds=Bounds(variable_lower, variable_upper),
            constraints=LinearConstraint(
                np.asarray(constraint_rows),
                np.asarray(constraint_lower),
                np.asarray(constraint_upper),
            ),
            options=solver_options,
        )


# =====================================================================
# Helper to simplify evaluation
# =====================================================================

def evaluate_simulation_metric(
        params,
        kripke_components,
        shape,
        domain_lb,
        domain_ub,
        *,
        horizon=1,
        verbose=False
    ):

    import synthetic_abstraction as sa
    import synthetic_objectives as so
    import synthetic_system as ss

    x_edges, y_edges = so.extract_grid_params(params,
                                              shape,
                                              domain_lb,
                                              domain_ub)
    successors, cells = sa.kripke_to_dicts(kripke_components,
                                           x_edges,
                                           y_edges)

    estimator = UpwardSimulationEstimator(
        ss.dynamics_sim,
        cells,
        successors,
        loss="box",
        norm_order=np.inf,
        )

    dynamics_offset = ss.XSTAR - ss.A_GLOBAL @ ss.XSTAR
    result = estimator.estimate_affine(
        ss.A_GLOBAL,
        dynamics_offset,
        horizon=horizon,
        restrict_witness_to_own_cell=True,
        verbose=True,
        store_per_state=False,
        store_witnesses=False,
        )
    
    return result


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    import synthetic_abstraction as sa
    import synthetic_objectives as so
    import synthetic_system as ss

    # Fixed abstraction and environment settings
    abstraction_shape = [70, 70]
    domain_lb = np.array([-10.0, -10.0])
    domain_ub = np.array([10.0, 10.0])

    # Initialize abstraction parameters
    key = jax.random.PRNGKey(1)
    sigma_u = 1.0
    key, k_u1, k_u2 = jax.random.split(key, 3)
    u1 = sigma_u * jax.random.normal(k_u1, (abstraction_shape[0],))
    u2 = sigma_u * jax.random.normal(k_u2, (abstraction_shape[1],))
    params = jnp.concatenate([u1, u2])
    
    # Evaluate
    result = evaluate_simulation_metric(params,
                                        abstraction_shape,
                                        domain_lb,
                                        domain_ub)
    print(result)