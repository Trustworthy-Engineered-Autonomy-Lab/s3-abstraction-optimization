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
from itertools import product
from typing import Callable, Hashable, Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc


StateId = Hashable
Vector = NDArray[np.float64]
Cell = tuple[ArrayLike, ArrayLike] | None


# =====================================================================
# Metric struct
# =====================================================================

@dataclass
class UpwardMetricResult:
    """Upward metric and distribution of optimized per-state values."""

    # Maximum (worst-case) per-state epsilon.
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
    optimization_method: str = "sampling"
    candidates_per_state: int = 0


def _epsilon_statistics(
    values: Vector,
) -> tuple[float, float, float, float, float]:
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
    """Approximate the finite-horizon directed simulation metric.

    ``cells[q]`` is either ``(lower, upper)`` or ``None`` for the OOB state,
    and ``successors[q]`` contains the abstract successor IDs.

    Periodic coordinates are described by ``periodic_dimensions``.  For the
    Dubins state ``(x, y, theta)``, use ``{2: 2*pi}``.  OOB distance defaults
    to the complement of the nonperiodic coordinate domain, so the artificial
    heading cut at ``-pi/pi`` is not treated as out of bounds.
    """

    def __init__(
        self,
        f: Callable[[Vector], ArrayLike],
        cells: Mapping[StateId, Cell],
        successors: Mapping[StateId, Iterable[StateId]],
        *,
        loss: str = "box",
        norm_order: float = np.inf,
        periodic_dimensions: Mapping[int, float] | None = None,
        oob_dimensions: Iterable[int] | None = None,
        coordinate_weights: ArrayLike | None = None,
    ) -> None:
        self.f = f
        self.cells = cells
        self.states = tuple(cells.keys())
        self.index = {q: i for i, q in enumerate(self.states)}

        if not self.states:
            raise ValueError("At least one abstract state is required.")
        if loss not in {"box", "center"}:
            raise ValueError("loss must be 'box' or 'center'.")

        bounded_states = [q for q in self.states if cells[q] is not None]
        oob_states = [q for q in self.states if cells[q] is None]
        if not bounded_states:
            raise ValueError("At least one bounded cell is required.")
        if len(oob_states) > 1:
            raise ValueError("At most one out-of-bounds state is supported.")
        if loss == "center" and oob_states:
            raise ValueError(
                "Center loss is undefined for OOB; use loss='box'."
            )

        self.loss = loss
        self.norm_order = norm_order
        self.oob_state = oob_states[0] if oob_states else None
        self.oob_index = (
            self.index[self.oob_state] if self.oob_state is not None else None
        )

        first_cell = cells[bounded_states[0]]
        assert first_cell is not None
        self.n = np.asarray(first_cell[0], dtype=float).size
        number_of_states = len(self.states)

        self.periodic_dimensions = dict(periodic_dimensions or {})
        for dimension, period in self.periodic_dimensions.items():
            if not 0 <= dimension < self.n:
                raise ValueError(f"Invalid periodic dimension {dimension}.")
            if not np.isfinite(period) or period <= 0.0:
                raise ValueError("Periods must be finite and positive.")

        if oob_dimensions is None:
            self.oob_dimensions = np.asarray(
                [k for k in range(self.n) if k not in self.periodic_dimensions],
                dtype=np.int64,
            )
        else:
            self.oob_dimensions = np.asarray(tuple(oob_dimensions), dtype=np.int64)
        if self.oob_index is not None and self.oob_dimensions.size == 0:
            raise ValueError("OOB requires at least one nonperiodic dimension.")
        if np.any((self.oob_dimensions < 0) | (self.oob_dimensions >= self.n)):
            raise ValueError("oob_dimensions contains an invalid dimension.")

        if coordinate_weights is None:
            self.coordinate_weights = np.ones(self.n, dtype=float)
        else:
            self.coordinate_weights = np.asarray(
                coordinate_weights,
                dtype=float,
            )
        if self.coordinate_weights.shape != (self.n,):
            raise ValueError(
                f"coordinate_weights must have shape ({self.n},)."
            )
        if np.any(~np.isfinite(self.coordinate_weights)) or np.any(
            self.coordinate_weights <= 0.0
        ):
            raise ValueError("coordinate_weights must be finite and positive.")

        self.lower = np.full((number_of_states, self.n), np.nan, dtype=float)
        self.upper = np.full((number_of_states, self.n), np.nan, dtype=float)
        self.bounded_mask = np.zeros(number_of_states, dtype=bool)

        for q in bounded_states:
            i = self.index[q]
            cell = cells[q]
            assert cell is not None
            lower = np.asarray(cell[0], dtype=float)
            upper = np.asarray(cell[1], dtype=float)
            if lower.shape != (self.n,) or upper.shape != (self.n,):
                raise ValueError(f"Cell {q!r} has inconsistent dimension.")
            if np.any(lower > upper):
                raise ValueError(f"Cell {q!r} has lower bounds above upper bounds.")
            self.lower[i] = lower
            self.upper[i] = upper
            self.bounded_mask[i] = True

        self.centers = 0.5 * (self.lower + self.upper)
        self.domain_lower = np.min(self.lower[self.bounded_mask], axis=0)
        self.domain_upper = np.max(self.upper[self.bounded_mask], axis=0)
        self.global_lower = self.domain_lower.copy()
        self.global_upper = self.domain_upper.copy()

        # Compact CSR transition representation.
        self.successor_indptr = np.empty(number_of_states + 1, dtype=np.int64)
        self.successor_indptr[0] = 0
        flat_successors = array("q")
        for i, q in enumerate(self.states):
            if q not in successors:
                raise KeyError(f"State {q!r} is missing from successors.")
            q_successors = tuple(successors[q])
            if not q_successors:
                raise ValueError(f"Abstract state {q!r} has no successor.")
            unknown = [r for r in q_successors if r not in self.index]
            if unknown:
                raise KeyError(f"State {q!r} has unknown successors: {unknown!r}")
            flat_successors.extend(self.index[r] for r in q_successors)
            self.successor_indptr[i + 1] = len(flat_successors)

        self.successor_indices = np.frombuffer(flat_successors, dtype=np.int64)
        self._vectorized_dynamics: bool | None = None

    # -----------------------------------------------------------------
    # Output metric, including circular heading geometry
    # -----------------------------------------------------------------

    @staticmethod
    def _wrap_delta(delta: NDArray[np.float64], period: float):
        return (delta + 0.5 * period) % period - 0.5 * period

    def _periodic_interval_distances(
        self,
        values: NDArray[np.float64],
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
        dimension: int,
    ) -> NDArray[np.float64]:
        """Distances from values to non-wrapping intervals on a circle.

        ``values`` has shape ``(S,)`` and bounds have shape ``(I,)``.  The
        result has shape ``(I, S)``.
        """
        period = self.periodic_dimensions[dimension]
        base = self.domain_lower[dimension]
        normalized = (values - base) % period + base
        lo = lower[:, None]
        hi = upper[:, None]
        span = hi - lo
        inside = (normalized[None, :] >= lo) & (normalized[None, :] <= hi)
        distance_lower = np.abs(
            self._wrap_delta(normalized[None, :] - lo, period)
        )
        distance_upper = np.abs(
            self._wrap_delta(normalized[None, :] - hi, period)
        )
        full_circle = span >= period - 1e-12
        return np.where(
            inside | full_circle,
            0.0,
            np.minimum(distance_lower, distance_upper),
        )

    def _bounded_cost_matrix(
        self,
        concrete_states: NDArray[np.float64],
        bounded_indices: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Observation costs with shape (abstract states, concrete states)."""
        concrete_states = np.asarray(concrete_states, dtype=float)
        bounded_indices = np.asarray(bounded_indices, dtype=np.int64)
        if bounded_indices.size == 0:
            return np.empty((0, concrete_states.shape[0]))

        if self.loss == "box":
            residual = np.maximum(
                np.maximum(
                    self.lower[bounded_indices, None, :] - concrete_states[None, :, :],
                    concrete_states[None, :, :] - self.upper[bounded_indices, None, :],
                ),
                0.0,
            )
            for dimension in self.periodic_dimensions:
                residual[:, :, dimension] = self._periodic_interval_distances(
                    concrete_states[:, dimension],
                    self.lower[bounded_indices, dimension],
                    self.upper[bounded_indices, dimension],
                    dimension,
                )
        else:
            residual = np.abs(
                self.centers[bounded_indices, None, :] - concrete_states[None, :, :]
            )
            for dimension, period in self.periodic_dimensions.items():
                residual[:, :, dimension] = np.abs(
                    self._wrap_delta(
                        concrete_states[None, :, dimension]
                        - self.centers[bounded_indices, None, dimension],
                        period,
                    )
                )

        residual *= self.coordinate_weights[None, None, :]
        return np.linalg.norm(residual, ord=self.norm_order, axis=2)

    def _oob_costs(self, concrete_states: NDArray[np.float64]) -> Vector:
        """Distance to the complement of the modeled spatial domain."""
        concrete_states = np.asarray(concrete_states, dtype=float)
        dims = self.oob_dimensions
        inside = np.all(
            (concrete_states[:, dims] > self.domain_lower[dims])
            & (concrete_states[:, dims] < self.domain_upper[dims]),
            axis=1,
        )
        margins = np.minimum(
            concrete_states[:, dims] - self.domain_lower[dims],
            self.domain_upper[dims] - concrete_states[:, dims],
        )
        weighted_margins = margins * self.coordinate_weights[dims]
        return np.where(inside, np.min(weighted_margins, axis=1), 0.0)

    def _observation_costs_at_indices(
        self,
        x: ArrayLike,
        indices: NDArray[np.int64],
    ) -> Vector:
        x_arr = np.asarray(x, dtype=float)
        indices = np.asarray(indices, dtype=np.int64)
        if x_arr.shape != (self.n,):
            raise ValueError(f"x must have shape ({self.n},).")

        costs = np.empty(indices.size, dtype=float)
        bounded_positions = self.bounded_mask[indices]
        bounded_indices = indices[bounded_positions]
        if bounded_indices.size:
            costs[bounded_positions] = self._bounded_cost_matrix(
                x_arr[None, :],
                bounded_indices,
            )[:, 0]
        if np.any(~bounded_positions):
            costs[~bounded_positions] = self._oob_costs(x_arr[None, :])[0]
        return costs

    def observation_costs(self, x: ArrayLike) -> Vector:
        """Compute ell(q, x) for every abstract state q."""
        return self._observation_costs_at_indices(
            x,
            np.arange(len(self.states), dtype=np.int64),
        )

    def _max_bounded_costs_inf(
        self,
        concrete_states: NDArray[np.float64],
        bounded_indices: NDArray[np.int64],
    ) -> Vector:
        """Exact aggregate for max-box cost under the infinity norm."""
        result = np.zeros(concrete_states.shape[0], dtype=float)
        nonperiodic = [
            k for k in range(self.n) if k not in self.periodic_dimensions
        ]

        for dimension in nonperiodic:
            if self.loss == "box":
                far_lower = np.max(self.lower[bounded_indices, dimension])
                far_upper = np.min(self.upper[bounded_indices, dimension])
            else:
                far_lower = np.max(self.centers[bounded_indices, dimension])
                far_upper = np.min(self.centers[bounded_indices, dimension])
            residual = np.maximum(
                np.maximum(
                    far_lower - concrete_states[:, dimension],
                    concrete_states[:, dimension] - far_upper,
                ),
                0.0,
            )
            result = np.maximum(
                result,
                self.coordinate_weights[dimension] * residual,
            )

        for dimension, period in self.periodic_dimensions.items():
            if self.loss == "box":
                residual = self._periodic_interval_distances(
                    concrete_states[:, dimension],
                    self.lower[bounded_indices, dimension],
                    self.upper[bounded_indices, dimension],
                    dimension,
                )
            else:
                residual = np.abs(
                    self._wrap_delta(
                        concrete_states[None, :, dimension]
                        - self.centers[bounded_indices, None, dimension],
                        period,
                    )
                )
            result = np.maximum(
                result,
                self.coordinate_weights[dimension] * np.max(residual, axis=0),
            )
        return result

    def _max_observation_costs_at_indices(
        self,
        concrete_states: NDArray[np.float64],
        indices: NDArray[np.int64],
        *,
        index_chunk_size: int = 2048,
    ) -> Vector:
        """For each concrete state, return max_q ell(q, x)."""
        concrete_states = np.asarray(concrete_states, dtype=float)
        indices = np.asarray(indices, dtype=np.int64)
        result = np.zeros(concrete_states.shape[0], dtype=float)
        bounded_indices = indices[self.bounded_mask[indices]]

        if bounded_indices.size and self.norm_order == np.inf:
            result = self._max_bounded_costs_inf(
                concrete_states,
                bounded_indices,
            )
        elif bounded_indices.size:
            for start in range(0, bounded_indices.size, index_chunk_size):
                chunk = bounded_indices[start:start + index_chunk_size]
                costs = self._bounded_cost_matrix(concrete_states, chunk)
                result = np.maximum(result, np.max(costs, axis=0))

        if np.any(~self.bounded_mask[indices]):
            result = np.maximum(result, self._oob_costs(concrete_states))
        return result

    # -----------------------------------------------------------------
    # Sparse graph and concrete rollouts
    # -----------------------------------------------------------------

    def _successors_of_indices(self, indices: NDArray[np.int64]) -> NDArray[np.int64]:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 1:
            i = int(indices[0])
            return self.successor_indices[
                self.successor_indptr[i]:self.successor_indptr[i + 1]
            ]
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
        if horizon < 0:
            raise ValueError("horizon must be nonnegative.")
        levels = [np.asarray([initial_index], dtype=np.int64)]
        for _ in range(horizon):
            levels.append(self._successors_of_indices(levels[-1]))
        return levels

    def _apply_dynamics_batch(self, states: NDArray[np.float64]) -> NDArray[np.float64]:
        """Use vectorized dynamics when supported, with a cached fallback."""
        if self._vectorized_dynamics is not False:
            try:
                result = np.asarray(self.f(states), dtype=float)
                if result.shape == states.shape:
                    self._vectorized_dynamics = True
                    if not np.all(np.isfinite(result)):
                        raise FloatingPointError("f produced a non-finite state.")
                    return result
            except (TypeError, ValueError, IndexError):
                if self._vectorized_dynamics is True:
                    raise
            self._vectorized_dynamics = False

        result = np.asarray([self.f(state) for state in states], dtype=float)
        if result.shape != states.shape:
            raise ValueError(
                f"f returned batch shape {result.shape}; expected {states.shape}."
            )
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("f produced a non-finite state.")
        return result

    def rollout(self, x0: ArrayLike, horizon: int) -> NDArray[np.float64]:
        if horizon < 0:
            raise ValueError("horizon must be nonnegative.")
        x_arr = np.asarray(x0, dtype=float)
        if x_arr.shape != (self.n,):
            raise ValueError(f"x0 must have shape ({self.n},).")
        return self.rollout_batch(x_arr[None, :], horizon)[:, 0, :]

    def rollout_batch(
        self,
        initial_states: ArrayLike,
        horizon: int,
    ) -> NDArray[np.float64]:
        if horizon < 0:
            raise ValueError("horizon must be nonnegative.")
        initial = np.asarray(initial_states, dtype=float)
        if initial.ndim != 2 or initial.shape[1] != self.n:
            raise ValueError(f"initial_states must have shape (N, {self.n}).")
        trajectory = np.empty((horizon + 1, *initial.shape), dtype=float)
        trajectory[0] = initial
        for t in range(horizon):
            trajectory[t + 1] = self._apply_dynamics_batch(trajectory[t])
        return trajectory

    def finite_horizon_values(self, x0: ArrayLike, horizon: int) -> Vector:
        """Compute V_H(q, x0) simultaneously for all abstract states."""
        trajectory = self.rollout(x0, horizon)
        values = self.observation_costs(trajectory[horizon])
        for t in range(horizon - 1, -1, -1):
            edge_values = values[self.successor_indices]
            worst_successor = np.maximum.reduceat(
                edge_values,
                self.successor_indptr[:-1],
            )
            values = np.maximum(
                self.observation_costs(trajectory[t]),
                worst_successor,
            )
        return values

    def _values_from_trajectory_for_levels(
        self,
        trajectory: NDArray[np.float64],
        levels: list[NDArray[np.int64]],
    ) -> Vector:
        """Evaluate several candidate witnesses from precomputed rollouts."""
        values = np.zeros(trajectory.shape[1], dtype=float)
        for t, indices in enumerate(levels):
            values = np.maximum(
                values,
                self._max_observation_costs_at_indices(trajectory[t], indices),
            )
        return values

    def _finite_horizon_values_for_levels_batch(
        self,
        initial_states: ArrayLike,
        levels: list[NDArray[np.int64]],
    ) -> Vector:
        trajectory = self.rollout_batch(initial_states, len(levels) - 1)
        return self._values_from_trajectory_for_levels(trajectory, levels)

    def _finite_horizon_value_for_levels(
        self,
        x0: ArrayLike,
        levels: list[NDArray[np.int64]],
    ) -> float:
        values = self._finite_horizon_values_for_levels_batch(
            np.asarray(x0, dtype=float)[None, :],
            levels,
        )
        return float(values[0])

    # -----------------------------------------------------------------
    # Nonlinear inner minimization
    # -----------------------------------------------------------------

    @staticmethod
    def _unit_box_design(dimension: int, num_samples: int, seed: int) -> Vector:
        if num_samples < 0:
            raise ValueError("num_samples must be nonnegative.")
        anchors = [np.full(dimension, 0.5)]
        anchors.extend(np.asarray(corner, dtype=float) for corner in product([0.0, 1.0], repeat=dimension))
        if num_samples:
            sampler = qmc.Sobol(d=dimension, scramble=True, seed=seed)
            exponent = int(np.ceil(np.log2(num_samples)))
            sobol = sampler.random_base2(exponent)[:num_samples]
            anchors.extend(sobol)
        return np.unique(np.asarray(anchors, dtype=float), axis=0)

    def _initial_states_and_indices(
        self,
        initial_abstract_states: Iterable[StateId] | None,
    ) -> tuple[tuple[StateId, ...], NDArray[np.int64]]:
        if initial_abstract_states is None:
            initial_states = tuple(
                q for q in self.states if self.cells[q] is not None
            )
        else:
            initial_states = tuple(initial_abstract_states)

        indices = np.empty(len(initial_states), dtype=np.int64)
        for position, q in enumerate(initial_states):
            if q not in self.index:
                raise KeyError(f"Unknown initial abstract state: {q!r}")
            i = self.index[q]
            if not self.bounded_mask[i]:
                raise ValueError(
                    f"Cannot optimize an OOB witness for state {q!r}."
                )
            indices[position] = i
        return initial_states, indices

    def _build_result(
        self,
        initial_states: tuple[StateId, ...],
        epsilon_values: Vector,
        best_witnesses: NDArray[np.float64],
        horizon: int,
        *,
        store_per_state: bool,
        store_witnesses: bool,
        method: str,
        candidates_per_state: int,
    ) -> UpwardMetricResult:
        if epsilon_values.size:
            worst_position = int(np.argmax(epsilon_values))
            epsilon = float(epsilon_values[worst_position])
            worst_state = initial_states[worst_position]
            worst_witness = best_witnesses[worst_position].copy()
        else:
            epsilon = 0.0
            worst_state = None
            worst_witness = None

        per_state = (
            {q: float(value) for q, value in zip(initial_states, epsilon_values)}
            if store_per_state
            else {}
        )
        witnesses = (
            {q: witness.copy() for q, witness in zip(initial_states, best_witnesses)}
            if store_witnesses
            else {}
        )
        epsilon_min, q1, median, q3, mean = _epsilon_statistics(epsilon_values)
        return UpwardMetricResult(
            epsilon=epsilon,
            per_abstract_state=per_state,
            concrete_witnesses=witnesses,
            horizon=horizon,
            worst_abstract_state=worst_state,
            worst_concrete_witness=worst_witness,
            epsilon_min=epsilon_min,
            epsilon_q1=q1,
            epsilon_median=median,
            epsilon_q3=q3,
            epsilon_mean=mean,
            optimization_method=method,
            candidates_per_state=candidates_per_state,
        )

    def estimate(
        self,
        horizon: int,
        *,
        initial_abstract_states: Iterable[StateId] | None = None,
        restrict_witness_to_own_cell: bool = True,
        num_samples: int = 64,
        batch_size: int = 256,
        seed: int = 0,
        refine: bool = False,
        refine_maxiter: int = 60,
        verbose: bool = False,
        progress_every: int = 1000,
        store_per_state: bool = True,
        store_witnesses: bool = True,
    ) -> UpwardMetricResult:
        """Approximate all inner minima using batched Sobol candidates.

        Every source cell uses its center, all corners, and ``num_samples``
        scrambled Sobol points.  The same normalized design is used in each
        cell, making the dynamics rollout highly vectorizable.  Increasing
        ``num_samples`` tightens the sampled upper bound on the true minimum.
        ``refine=True`` runs bounded Powell refinement from each sampled best
        witness; this is more accurate but should be reserved for smaller
        models or selected initial states.
        """
        if horizon < 0:
            raise ValueError("horizon must be nonnegative.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if progress_every <= 0:
            raise ValueError("progress_every must be positive.")

        initial_states, initial_indices = self._initial_states_and_indices(
            initial_abstract_states
        )
        design = self._unit_box_design(self.n, num_samples, seed)
        candidates_per_state = design.shape[0]
        epsilon_values = np.empty(len(initial_states), dtype=float)
        best_witnesses = np.empty((len(initial_states), self.n), dtype=float)

        for batch_start in range(0, len(initial_states), batch_size):
            batch_stop = min(batch_start + batch_size, len(initial_states))
            batch_indices = initial_indices[batch_start:batch_stop]
            if restrict_witness_to_own_cell:
                search_lower = self.lower[batch_indices]
                search_upper = self.upper[batch_indices]
            else:
                search_lower = np.broadcast_to(
                    self.global_lower,
                    (batch_indices.size, self.n),
                )
                search_upper = np.broadcast_to(
                    self.global_upper,
                    (batch_indices.size, self.n),
                )

            candidates = (
                search_lower[:, None, :]
                + design[None, :, :] * (search_upper - search_lower)[:, None, :]
            )
            flat_trajectory = self.rollout_batch(
                candidates.reshape(-1, self.n),
                horizon,
            )
            trajectory = flat_trajectory.reshape(
                horizon + 1,
                batch_indices.size,
                candidates_per_state,
                self.n,
            )

            for local_position, initial_index in enumerate(batch_indices):
                global_position = batch_start + local_position
                levels = self._reachable_levels(int(initial_index), horizon)
                candidate_values = self._values_from_trajectory_for_levels(
                    trajectory[:, local_position],
                    levels,
                )
                best = int(np.argmin(candidate_values))
                best_value = float(candidate_values[best])
                best_x = candidates[local_position, best].copy()

                if refine:
                    bounds = list(
                        zip(
                            search_lower[local_position],
                            search_upper[local_position],
                        )
                    )

                    def objective(x: Vector) -> float:
                        return self._finite_horizon_value_for_levels(x, levels)

                    local_result = minimize(
                        objective,
                        best_x,
                        method="Powell",
                        bounds=bounds,
                        options={
                            "maxiter": refine_maxiter,
                            "xtol": 1e-7,
                            "ftol": 1e-7,
                        },
                    )
                    if np.isfinite(local_result.fun) and local_result.fun < best_value:
                        best_value = float(local_result.fun)
                        best_x = np.asarray(local_result.x, dtype=float)

                epsilon_values[global_position] = best_value
                best_witnesses[global_position] = best_x

            if verbose:
                solved = batch_stop
                previous = batch_start
                crossed = solved // progress_every > previous // progress_every
                if crossed or solved == len(initial_states):
                    print(f"Solved {solved} / {len(initial_states)} states")

        method = "sobol+powell" if refine else "sobol"
        return self._build_result(
            initial_states,
            epsilon_values,
            best_witnesses,
            horizon,
            store_per_state=store_per_state,
            store_witnesses=store_witnesses,
            method=method,
            candidates_per_state=candidates_per_state,
        )

    def estimate_global(
        self,
        horizon: int,
        *,
        initial_abstract_states: Iterable[StateId] | None = None,
        restrict_witness_to_own_cell: bool = True,
        maxiter: int = 150,
        popsize: int = 12,
        seed: int = 0,
        polish: bool = True,
        verbose: bool = False,
        progress_every: int = 100,
        store_per_state: bool = True,
        store_witnesses: bool = True,
    ) -> UpwardMetricResult:
        """Slower differential-evolution reference for small state subsets."""
        initial_states, initial_indices = self._initial_states_and_indices(
            initial_abstract_states
        )
        epsilon_values = np.empty(len(initial_states), dtype=float)
        best_witnesses = np.empty((len(initial_states), self.n), dtype=float)

        for position, (q, initial_index) in enumerate(
            zip(initial_states, initial_indices)
        ):
            levels = self._reachable_levels(int(initial_index), horizon)
            if restrict_witness_to_own_cell:
                lower = self.lower[initial_index]
                upper = self.upper[initial_index]
            else:
                lower = self.global_lower
                upper = self.global_upper
            bounds = list(zip(lower, upper))

            def objective(x: Vector) -> float:
                return self._finite_horizon_value_for_levels(x, levels)

            result = differential_evolution(
                objective,
                bounds=bounds,
                seed=seed + position,
                maxiter=maxiter,
                popsize=popsize,
                polish=polish,
                updating="immediate",
                workers=1,
            )
            epsilon_values[position] = float(result.fun)
            best_witnesses[position] = np.asarray(result.x, dtype=float)

            if verbose and (
                (position + 1) % progress_every == 0
                or position + 1 == len(initial_states)
            ):
                print(f"Solved {position + 1} / {len(initial_states)} states")

        return self._build_result(
            initial_states,
            epsilon_values,
            best_witnesses,
            horizon,
            store_per_state=store_per_state,
            store_witnesses=store_witnesses,
            method="differential_evolution",
            candidates_per_state=0,
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
    horizon: int = 1,
    num_samples: int = 64,
    batch_size: int = 256,
    seed: int = 0,
    refine: bool = False,
    coordinate_weights: ArrayLike | None = None,
    initial_abstract_states: Iterable[StateId] | None = None,
    verbose: bool = True,
    store_per_state: bool = False,
    store_witnesses: bool = False,
) -> UpwardMetricResult:
    
    import unicycle_abstraction as ua
    import unicycle_objectives as uo
    import unicycle_system as us

    x_edges, y_edges, theta_edges = uo.extract_grid_params(
        params,
        shape,
        domain_lb,
        domain_ub,
    )
    successors, cells = ua.kripke_to_dicts(
        kripke_components,
        x_edges,
        y_edges,
        theta_edges,
    )
    estimator = UpwardSimulationEstimator(
        us.cl_system,
        cells,
        successors,
        loss="box",
        norm_order=np.inf,
        periodic_dimensions={2: 2.0 * np.pi},
        oob_dimensions=(0, 1),
        coordinate_weights=coordinate_weights,
    )
    return estimator.estimate(
        horizon,
        initial_abstract_states=initial_abstract_states,
        num_samples=num_samples,
        batch_size=batch_size,
        seed=seed,
        refine=refine,
        verbose=verbose,
        store_per_state=store_per_state,
        store_witnesses=store_witnesses,
    )


# ================================================================
# Main
# =====================================================================

if __name__ == "__main__":

    import jax
    import jax.numpy as jnp
    import verification_tools as vt

    gt_reach_fname = "unicycle-taylor/unicycle_gt_reach_regions_100.pkl"

    # Fixed abstraction and environment settings
    abstraction_shape = [30, 30, 30]
    domain_lb = np.array([0.0, 0.0, -np.pi])
    domain_ub = np.array([50.0, 50.0, np.pi])

    # Define the initial state subset domain
    init_domain_lb = np.array([0.0, 0.0, -np.pi/4])
    init_domain_ub = np.array([50.0, 50.0, np.pi/4])

    # Initialize abstraction parameters
    key = jax.random.PRNGKey(0)
    sigma_u = 1.0
    key, k_u1, k_u2 = jax.random.split(key, 3)
    u1 = sigma_u * jax.random.normal(k_u1, (abstraction_shape[0],))
    u2 = sigma_u * jax.random.normal(k_u2, (abstraction_shape[1],))
    u3 = sigma_u * jax.random.normal(k_u2, (abstraction_shape[2],))
    params = jnp.concatenate([u1, u2, u3])
    
    # Evaluate
    _, kripke_components = vt.build_and_verify_from_params(params,
                                               abstraction_shape,
                                               domain_lb,
                                               domain_ub,
                                               init_domain_lb,
                                               init_domain_ub,
                                               gt_reach_fname=gt_reach_fname,
                                               verbose=True,
                                               log_time=True)
    
    result = evaluate_simulation_metric(
        params,
        kripke_components,
        abstraction_shape,
        domain_lb,
        domain_ub,
        horizon=1,
        num_samples=64,
        batch_size=256,
        refine=False,
        verbose=True,
    )
    print(f"    > Epsilon = {result.epsilon}")
    print(f"    > Mean epsilon = {result.epsilon_mean}")
    print(f"    > Median epsilon = {result.epsilon_median}")
    print(f"    > Q3 epsilon = {result.epsilon_q3}")