import numpy as np
import pytest

from unicycle_simulation_analysis import UpwardSimulationEstimator


def identity_dynamics(states):
    states = np.asarray(states, dtype=float)
    result = states.copy()
    result[..., 2] = (result[..., 2] + np.pi) % (2.0 * np.pi) - np.pi
    return result


@pytest.fixture
def circular_estimator():
    cells = {
        0: (
            np.array([0.0, 0.0, -np.pi]),
            np.array([1.0, 1.0, -0.5 * np.pi]),
        ),
        1: (
            np.array([1.0, 0.0, -0.5 * np.pi]),
            np.array([2.0, 1.0, 0.5 * np.pi]),
        ),
        2: (
            np.array([0.0, 0.0, 0.5 * np.pi]),
            np.array([1.0, 1.0, np.pi]),
        ),
        3: None,
    }
    successors = {
        0: {0, 1},
        1: {1, 2},
        2: {0, 2, 3},
        3: {3},
    }
    return UpwardSimulationEstimator(
        identity_dynamics,
        cells,
        successors,
        periodic_dimensions={2: 2.0 * np.pi},
        oob_dimensions=(0, 1),
    )


def test_heading_distance_crosses_pi_seam(circular_estimator):
    # pi - 0.1 is only 0.1 radians from the -pi endpoint of cell 0.
    costs = circular_estimator._observation_costs_at_indices(
        np.array([0.5, 0.5, np.pi - 0.1]),
        np.array([0]),
    )
    assert costs[0] == pytest.approx(0.1)


def test_heading_seam_is_not_out_of_bounds(circular_estimator):
    cost = circular_estimator._observation_costs_at_indices(
        np.array([1.0, 0.5, -np.pi]),
        np.array([3]),
    )[0]
    # Spatial margins are min(1.0, 1.0, 0.5, 0.5) = 0.5.
    assert cost == pytest.approx(0.5)


def test_linf_aggregate_matches_individual_costs(circular_estimator):
    concrete = np.array([
        [0.2, 0.4, -3.0],
        [1.5, 0.7, 0.0],
        [0.8, 0.1, 3.0],
    ])
    indices = np.array([0, 1, 2])
    aggregate = circular_estimator._max_observation_costs_at_indices(
        concrete,
        indices,
    )
    direct = np.max(
        circular_estimator._bounded_cost_matrix(concrete, indices),
        axis=0,
    )
    np.testing.assert_allclose(aggregate, direct)


def test_local_values_match_full_dynamic_program(circular_estimator):
    state = np.array([0.7, 0.4, np.pi - 0.2])
    horizon = 3
    full = circular_estimator.finite_horizon_values(state, horizon)
    local = np.array([
        circular_estimator._finite_horizon_value_for_levels(
            state,
            circular_estimator._reachable_levels(i, horizon),
        )
        for i in range(len(circular_estimator.states))
    ])
    np.testing.assert_allclose(local, full)


def test_sampling_estimator_returns_reproducible_witness_values():
    cells = {
        0: (
            np.array([0.0, 0.0, -np.pi]),
            np.array([1.0, 1.0, np.pi]),
        ),
        1: (
            np.array([2.0, 0.0, -np.pi]),
            np.array([3.0, 1.0, np.pi]),
        ),
    }
    successors = {0: {0, 1}, 1: {1}}
    estimator = UpwardSimulationEstimator(
        identity_dynamics,
        cells,
        successors,
        periodic_dimensions={2: 2.0 * np.pi},
    )
    result = estimator.estimate(
        horizon=1,
        num_samples=16,
        batch_size=2,
        seed=4,
    )

    assert result.epsilon == pytest.approx(1.0)
    assert result.per_abstract_state[0] == pytest.approx(1.0)
    assert result.per_abstract_state[1] == pytest.approx(0.0)
    assert result.epsilon_min == pytest.approx(0.0)
    assert result.epsilon_median == pytest.approx(0.5)
    assert result.epsilon_mean == pytest.approx(0.5)

    for q, expected in result.per_abstract_state.items():
        levels = estimator._reachable_levels(estimator.index[q], 1)
        actual = estimator._finite_horizon_value_for_levels(
            result.concrete_witnesses[q],
            levels,
        )
        assert actual == pytest.approx(expected)


def test_coordinate_weights_scale_heading(circular_estimator):
    weighted = UpwardSimulationEstimator(
        identity_dynamics,
        circular_estimator.cells,
        {
            q: set(
                circular_estimator.states[j]
                for j in circular_estimator.successor_indices[
                    circular_estimator.successor_indptr[i]
                    : circular_estimator.successor_indptr[i + 1]
                ]
            )
            for i, q in enumerate(circular_estimator.states)
        },
        periodic_dimensions={2: 2.0 * np.pi},
        oob_dimensions=(0, 1),
        coordinate_weights=np.array([1.0, 1.0, 2.0]),
    )
    cost = weighted._observation_costs_at_indices(
        np.array([0.5, 0.5, np.pi - 0.1]),
        np.array([0]),
    )[0]
    assert cost == pytest.approx(0.2)
