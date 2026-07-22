"""Verification tests for the cached MountainCar derivative implementation."""

from __future__ import annotations

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch

import mountain_car_system as system


class MountainCarDerivativeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = np.array([-0.4, 0.0], dtype=np.float64)

    def _torch_closed_loop(self, state: torch.Tensor) -> torch.Tensor:
        """Independent differentiable reconstruction of the saved actor/plant."""

        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(value, dtype=state.dtype)

        h1 = torch.relu(tensor(system._W1) @ state + tensor(system._B1))
        h2 = torch.relu(tensor(system._W2) @ h1 + tensor(system._B2))
        z3 = (tensor(system._W3) @ h2 + tensor(system._B3))[0]
        action = system._ACTION_SCALE * torch.tanh(z3) + system._ACTION_BIAS

        position, velocity = state[0], state[1]
        raw_velocity = (
            velocity
            + system.POWER * action
            - system.GRAVITY * torch.cos(3.0 * position)
        )
        next_velocity = torch.clamp(
            raw_velocity, system.MIN_VELOCITY, system.MAX_VELOCITY
        )
        next_position = torch.clamp(
            position + next_velocity, system.MIN_POSITION, system.MAX_POSITION
        )
        next_velocity = torch.where(
            (next_position <= system.MIN_POSITION) & (next_velocity < 0.0),
            torch.zeros_like(next_velocity),
            next_velocity,
        )
        return torch.stack([next_position, next_velocity])

    def _stable_box(self) -> tuple[np.ndarray, np.ndarray]:
        # This actor has densely spaced ReLU boundaries.  Reduce the box until
        # the interval routine certifies one activation/plant branch.
        half_width = 1e-3
        while half_width >= 1e-10:
            lower = self.state - half_width
            upper = self.state + half_width
            try:
                system.interval_hessian(lower, upper)
                return lower, upper
            except system.DerivativeDomainError:
                half_width *= 0.1
        self.fail("Could not find a smooth test box around the sample state.")

    def test_cached_actor_matches_sb3_predict(self) -> None:
        rng = np.random.default_rng(7)
        states = np.column_stack(
            [
                rng.uniform(system.MIN_POSITION, system.MAX_POSITION, 50),
                rng.uniform(system.MIN_VELOCITY, system.MAX_VELOCITY, 50),
            ]
        )
        for state in states:
            sb3_action, _ = system.CONTROLLER.predict(state, deterministic=True)
            sb3_scalar = float(np.asarray(sb3_action).reshape(-1)[0])
            # The derivative path evaluates the stored float32 parameters as a
            # smooth float64 function.  SB3 performs its reductions in
            # float32, so a few ulps accumulate through the 400/300-wide
            # layers.  The resulting transition difference is smaller by the
            # plant's POWER factor (0.0015).
            self.assertAlmostEqual(
                system.controller_action(state), sb3_scalar, delta=3e-5
            )

    def test_closed_loop_matches_sb3_controller_path(self) -> None:
        rng = np.random.default_rng(11)
        for _ in range(50):
            state = np.array(
                [
                    rng.uniform(system.MIN_POSITION, system.MAX_POSITION),
                    rng.uniform(system.MIN_VELOCITY, system.MAX_VELOCITY),
                ]
            )
            sb3_action, _ = system.CONTROLLER.predict(state, deterministic=True)
            expected = system.ol_system(
                state, float(np.asarray(sb3_action).reshape(-1)[0])
            )
            np.testing.assert_allclose(
                system.cl_system_numeric(state), expected, atol=5e-8, rtol=0.0
            )

    def test_jax_closed_loop_matches_numpy_for_single_and_batched_states(self) -> None:
        states = np.array(
            [
                [-0.4, 0.0],
                [-1.2, -0.07],
                [0.6, 0.07],
                [-0.5, 0.02],
            ],
            dtype=np.float32,
        )
        expected = np.stack(
            [system.cl_system_numeric(state) for state in states]
        )

        single = system.cl_system_jax(jnp.asarray(states[0]))
        batched = system.cl_system_jax(jnp.asarray(states))
        jitted = jax.jit(system.cl_system_jax)(jnp.asarray(states))

        self.assertEqual(single.shape, (2,))
        self.assertEqual(batched.shape, states.shape)
        np.testing.assert_allclose(single, expected[0], atol=7e-8, rtol=0.0)
        np.testing.assert_allclose(batched, expected, atol=7e-8, rtol=0.0)
        np.testing.assert_allclose(jitted, expected, atol=7e-8, rtol=0.0)

    def test_interval_closed_loop_contains_samples_across_switches(self) -> None:
        rng = np.random.default_rng(13)
        boxes = [
            (
                np.array([system.MIN_POSITION, system.MIN_VELOCITY]),
                np.array([-1.10, -0.04]),
            ),
            (np.array([-0.55, -0.01]), np.array([-0.35, 0.02])),
            (np.array([0.25, -0.07]), np.array([0.40, -0.055])),
            (np.array([0.45, 0.04]), np.array([system.MAX_POSITION, 0.07])),
        ]
        for lower, upper in boxes:
            image_lower, image_upper = system.interval_cl_system(lower, upper)
            samples = np.vstack(
                [lower, upper, rng.uniform(lower, upper, size=(128, 2))]
            )
            images = np.stack(
                [system.cl_system_numeric(sample) for sample in samples]
            )
            self.assertTrue(np.all(images >= image_lower - 1e-12))
            self.assertTrue(np.all(images <= image_upper + 1e-12))

            jax_lower, jax_upper = system.interval_cl_system_jax(
                jnp.asarray(lower, dtype=jnp.float32),
                jnp.asarray(upper, dtype=jnp.float32),
            )
            np.testing.assert_allclose(jax_lower, image_lower, atol=2e-7, rtol=0.0)
            np.testing.assert_allclose(jax_upper, image_upper, atol=2e-7, rtol=0.0)

    def test_jax_state_jacobian_matches_cached_analytic_jacobian(self) -> None:
        state = jnp.asarray(self.state, dtype=jnp.float32)
        jax_jacobian = jax.jacfwd(system.cl_system_jax)(state)
        expected = system.jacobian(self.state)
        np.testing.assert_allclose(jax_jacobian, expected, atol=2e-6, rtol=2e-6)

    def test_image_area_has_jittable_abstraction_parameter_gradient(self) -> None:
        # Import here to keep the system tests usable independently of the
        # objective module while still exercising the intended integration.
        import mountain_car_objectives as objectives

        shape = (4, 5)
        args = {
            "shape": shape,
            "domain_lb": jnp.array([system.MIN_POSITION, system.MIN_VELOCITY]),
            "domain_ub": jnp.array([system.MAX_POSITION, system.MAX_VELOCITY]),
        }
        params = jnp.zeros(sum(shape), dtype=jnp.float32)

        # Capture args in the closure so the cell counts remain static during
        # JIT tracing; passing a dynamic dict directly to jit would trace the
        # shape entries and make Python slicing invalid.
        objective = lambda values: objectives.image_area(values, args=args)
        value, gradient = jax.jit(jax.value_and_grad(objective))(params)

        self.assertEqual(gradient.shape, params.shape)
        self.assertTrue(bool(jnp.isfinite(value)))
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient))))
        self.assertGreater(float(jnp.linalg.norm(gradient)), 0.0)

    def test_point_jacobian_and_hessian_match_torch_autodiff(self) -> None:
        state = torch.tensor(self.state, dtype=torch.float64, requires_grad=True)
        expected_jacobian = torch.autograd.functional.jacobian(
            self._torch_closed_loop, state
        ).detach().numpy()
        expected_hessian = np.stack(
            [
                torch.autograd.functional.hessian(
                    lambda value, output=output: self._torch_closed_loop(value)[output],
                    state,
                ).detach().numpy()
                for output in range(2)
            ]
        )

        np.testing.assert_allclose(
            system.jacobian(self.state), expected_jacobian, atol=2e-11, rtol=2e-11
        )
        np.testing.assert_allclose(
            system.hessian(self.state), expected_hessian, atol=2e-10, rtol=2e-10
        )

    def test_interval_hessian_contains_sampled_point_hessians(self) -> None:
        lower, upper = self._stable_box()
        hessian_lower, hessian_upper = system.interval_hessian(lower, upper)
        rng = np.random.default_rng(17)
        samples = np.vstack(
            [lower, upper, 0.5 * (lower + upper), rng.uniform(lower, upper, (100, 2))]
        )
        for state in samples:
            point_hessian = system.hessian(state)
            self.assertTrue(np.all(point_hessian >= hessian_lower - 1e-12))
            self.assertTrue(np.all(point_hessian <= hessian_upper + 1e-12))

    def test_taylor_remainder_contains_sampled_linearization_error(self) -> None:
        lower, upper = self._stable_box()
        center = 0.5 * (lower + upper)
        jacobian = system.jacobian(center)
        center_value = system.cl_system_numeric(center)
        remainder_lower, remainder_upper = system.taylor_remainder(lower, upper)

        rng = np.random.default_rng(23)
        samples = np.vstack(
            [lower, upper, center, rng.uniform(lower, upper, (100, 2))]
        )
        for state in samples:
            linear_value = system.linear_cl_system(
                state, center, J=jacobian, f_center=center_value
            )
            error = system.cl_system_numeric(state) - linear_value
            self.assertTrue(np.all(error >= remainder_lower - 2e-15))
            self.assertTrue(np.all(error <= remainder_upper + 2e-15))

    def test_interval_hessian_rejects_a_relu_boundary(self) -> None:
        # Construct a point on a first-layer hyperplane and place a small box
        # around it.  Search for one that lies in the physical state domain.
        boundary_state = None
        for weights, bias in zip(system._W1, system._B1):
            if abs(weights[0]) > 1e-12:
                candidate = np.array([-bias / weights[0], 0.0])
                if system.MIN_POSITION < candidate[0] < system.MAX_POSITION:
                    boundary_state = candidate
                    break
            if abs(weights[1]) > 1e-12:
                candidate = np.array([-0.4, (-bias + 0.4 * weights[0]) / weights[1]])
                if system.MIN_VELOCITY < candidate[1] < system.MAX_VELOCITY:
                    boundary_state = candidate
                    break

        self.assertIsNotNone(boundary_state)
        radius = np.array([1e-6, 1e-6])
        with self.assertRaisesRegex(system.DerivativeDomainError, "layer-1 ReLU"):
            system.interval_hessian(boundary_state - radius, boundary_state + radius)

    def test_tanh_second_interval_includes_critical_extrema(self) -> None:
        critical = math.atanh(1.0 / math.sqrt(3.0))
        lower, upper = system._tanh_second_interval(-2.0 * critical, 2.0 * critical)
        expected = 4.0 / (3.0 * math.sqrt(3.0))
        self.assertAlmostEqual(lower, -expected)
        self.assertAlmostEqual(upper, expected)

    def test_domain_box_inset_changes_only_outer_faces(self) -> None:
        lower = np.array([system.MIN_POSITION, -0.02])
        upper = np.array([-0.8, system.MAX_VELOCITY])
        adjusted_lower, adjusted_upper = system.inset_domain_box(
            lower, upper, inset=1e-8
        )
        np.testing.assert_array_equal(
            adjusted_lower, np.array([system.MIN_POSITION + 1e-8, -0.02])
        )
        np.testing.assert_array_equal(
            adjusted_upper, np.array([-0.8, system.MAX_VELOCITY - 1e-8])
        )

    def test_relaxed_boundary_cell_returns_finite_estimate(self) -> None:
        # This is the lower-left cell of the 100x100 abstraction grid. It
        # touches two physical faces and crosses multiple ReLU boundaries.
        lower = np.array([system.MIN_POSITION, system.MIN_VELOCITY])
        upper = lower + np.array([0.018, 0.0014])

        with self.assertRaises(system.DerivativeDomainError):
            system.interval_hessian(lower, upper)

        hessian_lower, hessian_upper = system.interval_hessian(
            lower,
            upper,
            strict=False,
            boundary_inset=1e-9,
            approximate_resolution=3,
        )
        self.assertTrue(np.all(np.isfinite(hessian_lower)))
        self.assertTrue(np.all(np.isfinite(hessian_upper)))
        self.assertTrue(np.all(hessian_lower <= hessian_upper))

        remainder_lower, remainder_upper = system.taylor_remainder(
            lower,
            upper,
            strict=False,
            boundary_inset=1e-9,
            approximate_resolution=3,
        )
        self.assertTrue(np.all(np.isfinite(remainder_lower)))
        self.assertTrue(np.all(np.isfinite(remainder_upper)))
        self.assertTrue(np.all(remainder_lower <= remainder_upper))


if __name__ == "__main__":
    unittest.main(verbosity=2)
