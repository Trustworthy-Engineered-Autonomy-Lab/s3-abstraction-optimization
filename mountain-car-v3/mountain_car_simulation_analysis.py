"""Finite-horizon upward simulation metrics for closed-loop Mountain Car.

For an abstract source q and concrete witness x, this module approximates

    V_H(q, x) = max_{0 <= t <= H} max_{r in Reach_t(q)} ell(r, f^t(x))

and then computes ``max_q min_{x in R_q} V_H(q, x)``.  The concrete dynamics
are nonlinear because the learned controller is in the loop.  Inner minima
are therefore approximated with batched Sobol candidates, optionally followed
by bounded Powell refinement.  Differential evolution is also available as a
slower reference method for selected abstract states.

Both Mountain Car coordinates--position and velocity--are ordinary bounded
real coordinates.  There is no circular/wrapping coordinate.
"""

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


class UpwardSimulationEstimator:
    """Approximate the finite-horizon directed simulation metric.

    ``cells[q]`` is ``(lower, upper)`` for an ordinary state and ``None`` for
    the optional OOB state.  ``successors[q]`` contains abstract successor
    IDs.  Position and velocity are both nonperiodic.

    The default metric is weighted L-infinity distance to a cell.  Since
    position and velocity have very different numerical ranges, callers can
    use ``coordinate_weights`` when a normalized or physically scaled metric
    is desired.  With no weights, both coordinates have weight one.
    """

    def __init__(
        self,
        f: Callable[[Vector], ArrayLike],
        cells: Mapping[StateId, Cell],
        successors: Mapping[StateId, Iterable[StateId]],
        *,
        loss: str = "box",
        norm_order: float = np.inf,
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
            raise ValueError("Center loss is undefined for OOB; use loss='box'.")

        first_cell = cells[bounded_states[0]]
        assert first_cell is not None
        self.n = np.asarray(first_cell[0], dtype=float).size
        if self.n != 2:
            raise ValueError(
                f"Mountain Car states must have dimension 2, received {self.n}."
            )

        self.loss = loss
        self.norm_order = norm_order
        self.oob_state = oob_states[0] if oob_states else None
        self.oob_index = (
            self.index[self.oob_state] if self.oob_state is not None else None
        )

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

        number_of_states = len(self.states)
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

        # Compact compressed-sparse-row transition representation.
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
    # Observation metric
    # -----------------------------------------------------------------

    def _bounded_cost_matrix(
        self,
        concrete_states: NDArray[np.float64],
        bounded_indices: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Return costs with shape (abstract states, concrete states)."""
        concrete_states = np.asarray(concrete_states, dtype=float)
        bounded_indices = np.asarray(bounded_indices, dtype=np.int64)
        if bounded_indices.size == 0:
            return np.empty((0, concrete_states.shape[0]))

        if self.loss == "box":
            residual = np.maximum(
                np.maximum(
                    self.lower[bounded_indices, None, :]
                    - concrete_states[None, :, :],
                    concrete_states[None, :, :]
                    - self.upper[bounded_indices, None, :],
                ),
                0.0,
            )
        else:
            residual = np.abs(
                self.centers[bounded_indices, None, :]
                - concrete_states[None, :, :]
            )
        residual *= self.coordinate_weights[None, None, :]
        return np.linalg.norm(residual, ord=self.norm_order, axis=2)

    def _oob_costs(self, concrete_states: NDArray[np.float64]) -> Vector:
        """Distance to the complement of the position-velocity domain."""
        concrete_states = np.asarray(concrete_states, dtype=float)
        inside = np.all(
            (concrete_states > self.domain_lower)
            & (concrete_states < self.domain_upper),
            axis=1,
        )
        margins = np.minimum(
            concrete_states - self.domain_lower,
            self.domain_upper - concrete_states,
        )
        weighted_margins = margins * self.coordinate_weights
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

    def _max_observation_costs_at_indices(
        self,
        concrete_states: NDArray[np.float64],
        indices: NDArray[np.int64],
        *,
        index_chunk_size: int = 4096,
    ) -> Vector:
        """For every concrete state, return max_q ell(q, x)."""
        concrete_states = np.asarray(concrete_states, dtype=float)
        indices = np.asarray(indices, dtype=np.int64)
        result = np.zeros(concrete_states.shape[0], dtype=float)
        bounded_indices = indices[self.bounded_mask[indices]]

        if bounded_indices.size and self.norm_order == np.inf:
            if self.loss == "box":
                far_lower = np.max(self.lower[bounded_indices], axis=0)
                far_upper = np.min(self.upper[bounded_indices], axis=0)
            else:
                far_lower = np.max(self.centers[bounded_indices], axis=0)
                far_upper = np.min(self.centers[bounded_indices], axis=0)
            residual = np.maximum(
                np.maximum(
                    far_lower - concrete_states,
                    concrete_states - far_upper,
                ),
                0.0,
            )
            result = np.max(residual * self.coordinate_weights, axis=1)
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
        """Use vectorized dynamics when available, with a cached fallback."""
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
        """Compute V_H(q, x0) simultaneously for every abstract state."""
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
        values = np.zeros(trajectory.shape[1], dtype=float)
        for t, indices in enumerate(levels):
            values = np.maximum(
                values,
                self._max_observation_costs_at_indices(trajectory[t], indices),
            )
        return values

    def _finite_horizon_value_for_levels(
        self,
        x0: ArrayLike,
        levels: list[NDArray[np.int64]],
    ) -> float:
        trajectory = self.rollout_batch(
            np.asarray(x0, dtype=float)[None, :],
            len(levels) - 1,
        )
        return float(self._values_from_trajectory_for_levels(trajectory, levels)[0])

    # -----------------------------------------------------------------
    # Graph-wide observation aggregation for weighted L-infinity loss
    # -----------------------------------------------------------------

    def _reachable_observation_aggregates(
        self,
        horizon: int,
    ) -> tuple[
        list[NDArray[np.float64]],
        list[NDArray[np.float64]],
        list[NDArray[np.bool_]],
        list[NDArray[np.bool_]],
    ]:
        """Propagate reachable-cell extrema once for all source states."""
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
            for dimension in range(self.n):
                next_lower[:, dimension] = np.maximum.reduceat(
                    previous_lower[self.successor_indices, dimension],
                    self.successor_indptr[:-1],
                )
                next_upper[:, dimension] = np.minimum.reduceat(
                    previous_upper[self.successor_indices, dimension],
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

    def _values_from_trajectory_aggregates(
        self,
        trajectory: NDArray[np.float64],
        initial_indices: NDArray[np.int64],
        aggregate_lowers: list[NDArray[np.float64]],
        aggregate_uppers: list[NDArray[np.float64]],
        bounded_reachable: list[NDArray[np.bool_]],
        oob_reachable: list[NDArray[np.bool_]],
    ) -> NDArray[np.float64]:
        """Evaluate all sources and candidates in one batch.

        ``trajectory`` has shape ``(H+1, B, S, 2)`` and the result has shape
        ``(B, S)``.
        """
        batch_size, sample_count = trajectory.shape[1:3]
        values = np.zeros((batch_size, sample_count), dtype=float)
        for t in range(trajectory.shape[0]):
            lower = aggregate_lowers[t][initial_indices]
            upper = aggregate_uppers[t][initial_indices]
            residual = np.maximum(
                np.maximum(
                    lower[:, None, :] - trajectory[t],
                    trajectory[t] - upper[:, None, :],
                ),
                0.0,
            )
            bounded_cost = np.max(
                residual * self.coordinate_weights[None, None, :],
                axis=2,
            )
            bounded_cost = np.where(
                bounded_reachable[t][initial_indices, None],
                bounded_cost,
                0.0,
            )
            if np.any(oob_reachable[t][initial_indices]):
                oob_cost = self._oob_costs(
                    trajectory[t].reshape(-1, self.n)
                ).reshape(batch_size, sample_count)
                bounded_cost = np.where(
                    oob_reachable[t][initial_indices, None],
                    np.maximum(bounded_cost, oob_cost),
                    bounded_cost,
                )
            values = np.maximum(values, bounded_cost)
        return values

    # -----------------------------------------------------------------
    # Nonlinear inner minimization
    # -----------------------------------------------------------------

    @staticmethod
    def _unit_box_design(dimension: int, num_samples: int, seed: int) -> Vector:
        if num_samples < 0:
            raise ValueError("num_samples must be nonnegative.")
        anchors = [np.full(dimension, 0.5)]
        anchors.extend(
            np.asarray(corner, dtype=float)
            for corner in product([0.0, 1.0], repeat=dimension)
        )
        if num_samples:
            sampler = qmc.Sobol(d=dimension, scramble=True, seed=seed)
            exponent = int(np.ceil(np.log2(num_samples)))
            anchors.extend(sampler.random_base2(exponent)[:num_samples])
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
                raise ValueError(f"Cannot optimize an OOB witness for state {q!r}.")
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
        progress_every: int = 10_000,
        store_per_state: bool = True,
        store_witnesses: bool = True,
    ) -> UpwardMetricResult:
        """Approximate all inner minima with batched Sobol candidates.

        Each cell uses its center, four corners, and ``num_samples`` scrambled
        Sobol points.  For weighted L-infinity loss, reachable-cell extrema
        are propagated once over the graph and all candidate comparisons are
        vectorized.  The sampled result is a feasible upper approximation of
        the true inner minimum.  ``refine=True`` applies bounded Powell
        refinement to each sampled best witness.
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

        aggregates = None
        if self.norm_order == np.inf:
            aggregates = self._reachable_observation_aggregates(horizon)

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
                + design[None, :, :]
                * (search_upper - search_lower)[:, None, :]
            )
            # After the first full batch, pad the final partial batch so a
            # JIT-compiled controller sees one stable input shape.
            rollout_candidates = candidates
            rollout_batch_size = batch_indices.size
            if batch_start > 0 and batch_indices.size < batch_size:
                padding = batch_size - batch_indices.size
                rollout_candidates = np.concatenate(
                    (
                        candidates,
                        np.repeat(candidates[-1:], padding, axis=0),
                    ),
                    axis=0,
                )
                rollout_batch_size = batch_size

            flat_trajectory = self.rollout_batch(
                rollout_candidates.reshape(-1, self.n),
                horizon,
            )
            trajectory = flat_trajectory.reshape(
                horizon + 1,
                rollout_batch_size,
                candidates_per_state,
                self.n,
            )[:, :batch_indices.size]

            if aggregates is not None:
                candidate_values = self._values_from_trajectory_aggregates(
                    trajectory,
                    batch_indices,
                    *aggregates,
                )
            else:
                candidate_values = np.empty(
                    (batch_indices.size, candidates_per_state),
                    dtype=float,
                )
                for local_position, initial_index in enumerate(batch_indices):
                    levels = self._reachable_levels(int(initial_index), horizon)
                    candidate_values[local_position] = (
                        self._values_from_trajectory_for_levels(
                            trajectory[:, local_position],
                            levels,
                        )
                    )

            best_positions = np.argmin(candidate_values, axis=1)
            batch_rows = np.arange(batch_indices.size)
            batch_values = candidate_values[batch_rows, best_positions]
            batch_witnesses = candidates[batch_rows, best_positions].copy()

            if refine:
                for local_position, initial_index in enumerate(batch_indices):
                    levels = self._reachable_levels(int(initial_index), horizon)
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
                        batch_witnesses[local_position],
                        method="Powell",
                        bounds=bounds,
                        options={
                            "maxiter": refine_maxiter,
                            "xtol": 1e-9,
                            "ftol": 1e-9,
                        },
                    )
                    if (
                        np.isfinite(local_result.fun)
                        and local_result.fun < batch_values[local_position]
                    ):
                        batch_values[local_position] = float(local_result.fun)
                        batch_witnesses[local_position] = np.asarray(
                            local_result.x,
                            dtype=float,
                        )

            epsilon_values[batch_start:batch_stop] = batch_values
            best_witnesses[batch_start:batch_stop] = batch_witnesses

            if verbose:
                crossed = (
                    batch_stop // progress_every > batch_start // progress_every
                )
                if crossed or batch_stop == len(initial_states):
                    print(f"Solved {batch_stop} / {len(initial_states)} states")

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

        for position, initial_index in enumerate(initial_indices):
            levels = self._reachable_levels(int(initial_index), horizon)
            if restrict_witness_to_own_cell:
                lower = self.lower[initial_index]
                upper = self.upper[initial_index]
            else:
                lower = self.global_lower
                upper = self.global_upper

            def objective(x: Vector) -> float:
                return self._finite_horizon_value_for_levels(x, levels)

            result = differential_evolution(
                objective,
                bounds=list(zip(lower, upper)),
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
    """Evaluate a parameterized Mountain Car abstraction."""
    import jax
    import mountain_car_abstraction as mca
    import mountain_car_objectives as mco
    import mountain_car_system as mcs

    position_edges, velocity_edges = mco.extract_grid_params(
        params,
        shape,
        domain_lb,
        domain_ub,
    )
    successors, cells = mca.kripke_to_dicts(
        kripke_components,
        position_edges,
        velocity_edges,
    )

    # The JAX closed-loop function accepts whole candidate batches.  JIT keeps
    # the 400/300-unit actor network from being evaluated one state at a time.
    batched_dynamics = jax.jit(mcs.cl_system_jax)
    estimator = UpwardSimulationEstimator(
        batched_dynamics,
        cells,
        successors,
        loss="box",
        norm_order=np.inf,
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


if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    import verification_tools as vt

    gt_reach_fname = "mountain-car-v3/mc_reach_regions.pkl"
    abstraction_shape = [100, 100]
    domain_lb = np.array([-1.2, -0.07])
    domain_ub = np.array([0.6, 0.07])
    init_domain_lb = domain_lb.copy()
    init_domain_ub = domain_ub.copy()

    key = jax.random.PRNGKey(0)
    key, k_u1, k_u2 = jax.random.split(key, 3)
    u1 = jax.random.normal(k_u1, (abstraction_shape[0],))
    u2 = jax.random.normal(k_u2, (abstraction_shape[1],))
    params = jnp.concatenate([u1, u2])

    _, kripke_components = vt.build_and_verify_from_params(
        params,
        abstraction_shape,
        domain_lb,
        domain_ub,
        init_domain_lb,
        init_domain_ub,
        gt_reach_fname=gt_reach_fname,
        verbose=True,
        log_time=True,
    )
    result = evaluate_simulation_metric(
        params,
        kripke_components,
        abstraction_shape,
        domain_lb,
        domain_ub,
        horizon=5,
        num_samples=64,
        batch_size=256,
        verbose=True,
    )
    print(f"    > Epsilon = {result.epsilon}")
    print(f"    > Mean epsilon = {result.epsilon_mean}")
    print(f"    > Median epsilon = {result.epsilon_median}")
    print(f"    > Q3 epsilon = {result.epsilon_q3}")
