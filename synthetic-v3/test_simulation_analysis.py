import numpy as np
import pytest

from synthetic_simulation_analysis import UpwardSimulationEstimator


@pytest.fixture
def affine_estimator():
    matrix = np.array([[0.8, -0.3], [0.3, 0.8]])
    offset = np.array([0.2, -0.1])

    def dynamics(state):
        return matrix @ np.asarray(state) + offset

    cells = {
        0: (np.array([-1.0, -1.0]), np.array([0.0, 0.0])),
        1: (np.array([-1.0, 0.0]), np.array([0.0, 1.0])),
        2: (np.array([0.0, -1.0]), np.array([1.0, 0.0])),
        3: (np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        4: None,
    }
    successors = {
        0: {0, 1},
        1: {1, 3, 4},
        2: {0, 2},
        3: {0, 1, 2, 3},
        4: {4},
    }
    estimator = UpwardSimulationEstimator(dynamics, cells, successors)
    return estimator, matrix, offset, cells


def test_local_evaluation_matches_full_dynamic_program(affine_estimator):
    estimator, _, _, _ = affine_estimator
    state = np.array([0.23, -0.41])
    horizon = 3

    full_values = estimator.finite_horizon_values(state, horizon)
    local_values = np.array([
        estimator._finite_horizon_value_for_levels(
            state,
            estimator._reachable_levels(i, horizon),
        )
        for i in range(len(estimator.states))
    ])

    np.testing.assert_allclose(local_values, full_values)


@pytest.mark.parametrize("horizon", [0, 1, 2, 3])
def test_affine_solver_returns_witnesses_with_reported_values(
    affine_estimator,
    horizon,
):
    estimator, matrix, offset, _ = affine_estimator
    result = estimator.estimate_affine(matrix, offset, horizon)

    for q, expected_value in result.per_abstract_state.items():
        levels = estimator._reachable_levels(estimator.index[q], horizon)
        witness_value = estimator._finite_horizon_value_for_levels(
            result.concrete_witnesses[q],
            levels,
        )
        assert witness_value == pytest.approx(expected_value, abs=1e-9)


def test_affine_solver_agrees_with_dense_grid_reference(affine_estimator):
    estimator, matrix, offset, cells = affine_estimator
    horizon = 2
    result = estimator.estimate_affine(matrix, offset, horizon)

    for q in range(4):
        lower, upper = cells[q]
        axes = [
            np.linspace(lower[k], upper[k], 41)
            for k in range(estimator.n)
        ]
        grid = np.stack(
            np.meshgrid(*axes, indexing="ij"),
            axis=-1,
        ).reshape(-1, estimator.n)
        levels = estimator._reachable_levels(estimator.index[q], horizon)
        grid_value = min(
            estimator._finite_horizon_value_for_levels(x, levels)
            for x in grid
        )

        # A grid search is an upper bound on the continuous optimum.  Its
        # resolution is fine enough to independently localize that optimum.
        assert result.per_abstract_state[q] <= grid_value + 1e-10
        assert grid_value - result.per_abstract_state[q] < 0.03


def test_affine_solver_rejects_mismatched_dynamics(affine_estimator):
    estimator, matrix, offset, _ = affine_estimator

    with pytest.raises(ValueError, match="do not match"):
        estimator.estimate_affine(matrix, offset + 1.0, horizon=1)


def test_low_memory_mode_retains_global_worst_case(affine_estimator):
    estimator, matrix, offset, _ = affine_estimator
    full = estimator.estimate_affine(matrix, offset, horizon=2)
    compact = estimator.estimate_affine(
        matrix,
        offset,
        horizon=2,
        store_per_state=False,
        store_witnesses=False,
    )

    assert compact.epsilon == pytest.approx(full.epsilon)
    assert compact.epsilon_min == pytest.approx(full.epsilon_min)
    assert compact.epsilon_q1 == pytest.approx(full.epsilon_q1)
    assert compact.epsilon_median == pytest.approx(full.epsilon_median)
    assert compact.epsilon_q3 == pytest.approx(full.epsilon_q3)
    assert compact.epsilon_mean == pytest.approx(full.epsilon_mean)
    assert compact.per_abstract_state == {}
    assert compact.concrete_witnesses == {}
    assert compact.worst_abstract_state is not None
    levels = estimator._reachable_levels(
        estimator.index[compact.worst_abstract_state],
        compact.horizon,
    )
    value = estimator._finite_horizon_value_for_levels(
        compact.worst_concrete_witness,
        levels,
    )
    assert value == pytest.approx(compact.epsilon)


def test_epsilon_summary_matches_numpy(affine_estimator):
    estimator, matrix, offset, _ = affine_estimator
    result = estimator.estimate_affine(matrix, offset, horizon=2)
    values = np.asarray(list(result.per_abstract_state.values()))

    assert result.epsilon == pytest.approx(np.max(values))
    assert result.epsilon_min == pytest.approx(np.min(values))
    assert result.epsilon_q1 == pytest.approx(np.quantile(values, 0.25))
    assert result.epsilon_median == pytest.approx(np.median(values))
    assert result.epsilon_q3 == pytest.approx(np.quantile(values, 0.75))
    assert result.epsilon_mean == pytest.approx(np.mean(values))
