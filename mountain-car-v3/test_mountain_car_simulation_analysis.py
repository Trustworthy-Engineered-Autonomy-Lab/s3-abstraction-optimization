import numpy as np
import pytest

from mountain_car_simulation_analysis import UpwardSimulationEstimator


def nonlinear_dynamics(states):
    states = np.asarray(states, dtype=float)
    position = np.clip(states[..., 0] + 0.2 * states[..., 1], -1.0, 1.0)
    velocity = np.clip(
        0.8 * states[..., 1] - 0.05 * np.cos(3.0 * states[..., 0]),
        -0.5,
        0.5,
    )
    return np.stack([position, velocity], axis=-1)


@pytest.fixture
def estimator():
    cells = {
        0: (np.array([-1.0, -0.5]), np.array([0.0, 0.0])),
        1: (np.array([-1.0, 0.0]), np.array([0.0, 0.5])),
        2: (np.array([0.0, -0.5]), np.array([1.0, 0.0])),
        3: (np.array([0.0, 0.0]), np.array([1.0, 0.5])),
        4: None,
    }
    successors = {
        0: {0, 1},
        1: {1, 3},
        2: {0, 2},
        3: {0, 1, 2, 3, 4},
        4: {4},
    }
    return UpwardSimulationEstimator(nonlinear_dynamics, cells, successors)


def test_observation_cost_is_ordinary_2d_box_distance(estimator):
    costs = estimator._observation_costs_at_indices(
        np.array([0.25, -0.1]),
        np.array([0, 1, 2, 3]),
    )
    np.testing.assert_allclose(costs, [0.25, 0.25, 0.0, 0.1])


def test_oob_distance_uses_position_and_velocity_faces(estimator):
    interior = estimator._observation_costs_at_indices(
        np.array([0.0, 0.1]),
        np.array([4]),
    )[0]
    boundary = estimator._observation_costs_at_indices(
        np.array([0.0, 0.5]),
        np.array([4]),
    )[0]
    assert interior == pytest.approx(0.4)
    assert boundary == pytest.approx(0.0)


def test_graph_aggregates_match_explicit_reachable_sets(estimator):
    horizon = 3
    initial_indices = np.arange(4, dtype=np.int64)
    rng = np.random.default_rng(8)
    candidates = np.stack(
        [
            rng.uniform(estimator.lower[i], estimator.upper[i], size=(11, 2))
            for i in initial_indices
        ]
    )
    flat = estimator.rollout_batch(candidates.reshape(-1, 2), horizon)
    trajectory = flat.reshape(horizon + 1, 4, 11, 2)
    aggregate_values = estimator._values_from_trajectory_aggregates(
        trajectory,
        initial_indices,
        *estimator._reachable_observation_aggregates(horizon),
    )

    explicit_values = np.empty_like(aggregate_values)
    for i in initial_indices:
        explicit_values[i] = estimator._values_from_trajectory_for_levels(
            trajectory[:, i],
            estimator._reachable_levels(int(i), horizon),
        )
    np.testing.assert_allclose(aggregate_values, explicit_values)


def test_local_values_match_full_dynamic_program(estimator):
    state = np.array([0.23, -0.17])
    horizon = 3
    full = estimator.finite_horizon_values(state, horizon)
    local = np.array([
        estimator._finite_horizon_value_for_levels(
            state,
            estimator._reachable_levels(i, horizon),
        )
        for i in range(len(estimator.states))
    ])
    np.testing.assert_allclose(local, full)


def test_sampling_estimator_returns_reported_witness_values(estimator):
    result = estimator.estimate(
        horizon=2,
        num_samples=16,
        batch_size=3,
        seed=5,
    )
    values = np.asarray(list(result.per_abstract_state.values()))
    assert result.epsilon == pytest.approx(np.max(values))
    assert result.epsilon_min == pytest.approx(np.min(values))
    assert result.epsilon_median == pytest.approx(np.median(values))
    assert result.epsilon_mean == pytest.approx(np.mean(values))
    assert result.candidates_per_state == 21  # 16 Sobol + center + 4 corners

    for q, expected in result.per_abstract_state.items():
        actual = estimator._finite_horizon_value_for_levels(
            result.concrete_witnesses[q],
            estimator._reachable_levels(estimator.index[q], 2),
        )
        assert actual == pytest.approx(expected)


def test_coordinate_weights_scale_velocity_cost():
    cells = {
        0: (np.array([-1.0, -0.5]), np.array([1.0, 0.0])),
        1: (np.array([-1.0, 0.0]), np.array([1.0, 0.5])),
    }
    successors = {0: {0}, 1: {1}}
    weighted = UpwardSimulationEstimator(
        nonlinear_dynamics,
        cells,
        successors,
        coordinate_weights=np.array([1.0, 10.0]),
    )
    cost = weighted._observation_costs_at_indices(
        np.array([0.0, 0.1]),
        np.array([0]),
    )[0]
    assert cost == pytest.approx(1.0)


def test_rejects_non_mountain_car_dimension():
    cells = {0: (np.zeros(3), np.ones(3))}
    with pytest.raises(ValueError, match="dimension 2"):
        UpwardSimulationEstimator(lambda x: x, cells, {0: {0}})
