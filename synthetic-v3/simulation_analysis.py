# =====================================================================
# Description: approximate the upward simulation metric between the 
# cocnrete and abstraction system
# =====================================================================

# =====================================================================
# Libraries
# =====================================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, Mapping
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import differential_evolution, minimize
import synthetic_system as ss
import jax
import jax.numpy as jnp
import synthetic_abstraction as sa
import synthetic_objectives as so

StateId = Hashable
Vector = NDArray[np.float64]
Cell = tuple[ArrayLike, ArrayLike] | None


# =====================================================================
# Metric struct
# =====================================================================

@dataclass
class UpwardMetricResult:
    epsilon: float
    per_abstract_state: dict[StateId, float]
    concrete_witnesses: dict[StateId, Vector]
    horizon: int


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
        self.cells = dict(cells)
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

        self.successor_indices: list[NDArray[np.int64]] = []

        for q in self.states:
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

            self.successor_indices.append(
                np.asarray(
                    [self.index[r] for r in q_successors],
                    dtype=np.int64,
                )
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

        costs = np.empty(len(self.states), dtype=float)

        bounded = self.bounded_mask

        if self.loss == "center":
            residual = self.centers[bounded] - x_arr
        else:
            residual = np.maximum(
                np.maximum(
                    self.lower[bounded] - x_arr,
                    x_arr - self.upper[bounded],
                ),
                0.0,
            )

        costs[bounded] = np.linalg.norm(
            residual,
            ord=self.norm_order,
            axis=1,
        )

        if self.oob_index is not None:
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
                costs[self.oob_index] = float(
                    np.min(margin_to_each_face)
                )
            else:
                costs[self.oob_index] = 0.0

        return costs

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

            for i, successor_ids in enumerate(self.successor_indices):
                worst_successor[i] = np.max(values[successor_ids])

            values = np.maximum(
                self.observation_costs(trajectory[t]),
                worst_successor,
            )

        return values

    def estimate(
        self,
        horizon: int,
        *,
        initial_abstract_states: Iterable[StateId] | None = None,
        restrict_witness_to_own_cell: bool = True,
        maxiter: int = 250,
        popsize: int = 15,
        seed: int = 0,
    ) -> UpwardMetricResult:
        """
        Approximate

            max_q min_x V_H(q, x).

        When restrict_witness_to_own_cell=True, x is constrained to R_q.
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

        for count, q in enumerate(initial_states):

            if self.cells[q] is None:
                raise ValueError(
                    f"Cannot optimize a witness for OOB state {q!r} "
                    "without a bounded concrete search region."
                )

            if q not in self.index:
                raise KeyError(f"Unknown initial abstract state: {q!r}")

            i = self.index[q]

            if restrict_witness_to_own_cell:
                lower = self.lower[i]
                upper = self.upper[i]
            else:
                lower = self.global_lower
                upper = self.global_upper

            bounds = list(zip(lower.tolist(), upper.tolist()))

            def objective(x: Vector) -> float:
                return float(self.finite_horizon_values(x, horizon)[i])

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

            per_state[q] = best_value
            witnesses[q] = best_x

            print(f"Solved for {count+1} states")

        epsilon = max(per_state.values(), default=0.0)

        return UpwardMetricResult(
            epsilon=epsilon,
            per_abstract_state=per_state,
            concrete_witnesses=witnesses,
            horizon=horizon,
        )
    

# =====================================================================
# Main
# =====================================================================


if __name__ == "__main__":

    # Fixed abstraction and environment settings
    abstraction_shape = [10, 10]
    domain_lb = np.array([-10.0, -10.0])
    domain_ub = np.array([10.0, 10.0])

    # Define the initial state subset domain
    init_domain_lb = np.array([-10.0, -10.0])
    init_domain_ub = np.array([10.0, 10.0])

    # Initialize abstraction parameters
    key = jax.random.PRNGKey(0)
    sigma_u = 1.0
    key, k_u1, k_u2 = jax.random.split(key, 3)
    u1 = sigma_u * jax.random.normal(k_u1, (abstraction_shape[0],))
    u2 = sigma_u * jax.random.normal(k_u2, (abstraction_shape[1],))
    params = jnp.concatenate([u1, u2])
    x_edges, y_edges = so.extract_grid_params(params,
                                              abstraction_shape,
                                              domain_lb,
                                              domain_ub)

    kripke_components = sa.build_abstraction(x_edges,
                                             y_edges,
                                             verbose=True)
    
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

    result = estimator.estimate(
                    horizon=1,
                    restrict_witness_to_own_cell=True,
                    maxiter=150,
                    popsize=12,
                    seed=10,
                )
    print(result.epsilon)



    # cells = {
    # 0: (np.array([-1.0, -1.0]), np.array([0.0, 0.0])),
    # 1: (np.array([ 0.0, -1.0]), np.array([1.0, 0.0])),
    # 2: (np.array([-1.0,  0.0]), np.array([0.0, 1.0])),
    # 3: (np.array([ 0.0,  0.0]), np.array([1.0, 1.0])),
    # }

    # successors = {
    #     0: {0, 1},
    #     1: {0, 1, 3},
    #     2: {0, 2, 3},
    #     3: {1, 2, 3},
    # }

    # estimator = UpwardSimulationEstimator(
    # f,
    # cells,
    # successors,
    # loss="box",       # Use "center" for the paper-faithful point output metric.
    # norm_order=np.inf,
    # )

    # for horizon in [1, 2, 4, 8, 16]:
    #     result = estimator.estimate(
    #         horizon,
    #         restrict_witness_to_own_cell=True,
    #         maxiter=150,
    #         popsize=12,
    #         seed=10,
    #     )

    #     print(
    #         f"H={horizon:2d}, "
    #         f"estimated upward metric={result.epsilon:.6g}"
    #     )