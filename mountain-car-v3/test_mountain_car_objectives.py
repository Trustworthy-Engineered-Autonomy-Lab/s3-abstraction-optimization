"""Regression tests for the graph-aligned Mountain Car proxy."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

import mountain_car_objectives as objectives
import mountain_car_system as system


class MountainCarObjectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = (3, 4)
        self.params = jnp.linspace(-0.4, 0.5, sum(self.shape))
        self.args = {
            "shape": self.shape,
            "domain_lb": np.array([-1.2, -0.07]),
            "domain_ub": np.array([0.6, 0.07]),
            "horizon": 2,
            "temp_in": 0.02,
            "temp_out": 0.03,
            "temp_witness": 0.01,
            "metric_temp": 0.002,
            "mean_weight": 0.25,
            "inflation_coefs": np.array([0.001, 0.0001]),
            "snap_temperatures": np.array([0.003, 0.0003]),
        }

    def test_batched_proxy_matches_numpy_reference(self) -> None:
        for propagation in ("corners", "interval"):
            args = {**self.args, "propagation": propagation}
            expected = objectives.upward_proxy_bruteforce(
                self.params, args=args
            )
            actual = objectives.upward_proxy(
                self.params, args=args, batch_size=5
            )
            self.assertAlmostEqual(float(actual), float(expected), places=6)

    def test_proxy_gradient_is_finite_and_nontrivial(self) -> None:
        args = {**self.args, "propagation": "interval"}
        objective = lambda params: objectives.upward_proxy(
            params, args=args, batch_size=5
        )
        value, gradient = jax.jit(jax.value_and_grad(objective))(self.params)
        self.assertTrue(bool(jnp.isfinite(value)))
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient))))
        self.assertGreater(float(jnp.linalg.norm(gradient)), 1e-4)

    def test_one_cell_one_step_is_paper_radius_plus_offset(self) -> None:
        args = {
            "shape": (1, 1),
            "domain_lb": np.array([-1.2, -0.07]),
            "domain_ub": np.array([0.6, 0.07]),
            # Equation (13) is inclusive: H=0 contains one scored term.
            "horizon": 0,
            "temp_in": 0.02,
            "temp_out": 0.03,
            "norm_order": 2.0,
            "propagation": "interval",
            "inflation_coefs": np.zeros(2),
            "snap_temperatures": np.zeros(2),
        }
        params = jnp.zeros(2)
        lower = args["domain_lb"]
        upper = args["domain_ub"]
        center = 0.5 * (lower + upper)
        next_center = system.cl_system_numeric(center)
        expected = (
            0.5 * np.linalg.norm(upper - lower)
            + np.linalg.norm(next_center - center)
        )
        actual = objectives.upward_proxy(params, args=args, batch_size=1)
        self.assertAlmostEqual(float(actual), float(expected), places=6)

    def test_hard_snap_selects_containing_cell_faces(self) -> None:
        edges = np.array([-1.0, -0.2, 0.3, 1.0])
        self.assertEqual(
            objectives._soft_cell_faces_numpy(0.1, edges, 0.0),
            (-0.2, 0.3),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
