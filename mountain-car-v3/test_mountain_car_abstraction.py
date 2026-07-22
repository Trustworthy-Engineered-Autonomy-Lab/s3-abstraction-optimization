"""Soundness checks for Mountain Car abstract transitions."""

from __future__ import annotations

import unittest

import numpy as np

import mountain_car_abstraction as abstraction
import mountain_car_system as system


class MountainCarAbstractionTests(unittest.TestCase):
    def test_sampled_concrete_successors_are_abstract_successors(self) -> None:
        x_edges = np.linspace(system.MIN_POSITION, system.MAX_POSITION, 7)
        y_edges = np.linspace(system.MIN_VELOCITY, system.MAX_VELOCITY, 6)
        components = abstraction.build_abstraction(x_edges, y_edges)
        transitions = components["kripke_transitions"]
        rng = np.random.default_rng(29)
        nx = len(x_edges) - 1
        ny = len(y_edges) - 1

        for i in range(nx):
            for j in range(ny):
                lower = np.array([x_edges[i], y_edges[j]])
                upper = np.array([x_edges[i + 1], y_edges[j + 1]])
                source = abstraction.cell_to_id(i, j, nx, ny)
                samples = np.vstack(
                    [lower, upper, rng.uniform(lower, upper, size=(16, 2))]
                )
                for sample in samples:
                    image = system.cl_system_numeric(sample)
                    ip = int(np.clip(
                        np.searchsorted(x_edges, image[0], side="right") - 1,
                        0,
                        nx - 1,
                    ))
                    jp = int(np.clip(
                        np.searchsorted(y_edges, image[1], side="right") - 1,
                        0,
                        ny - 1,
                    ))
                    target = abstraction.cell_to_id(ip, jp, nx, ny)
                    self.assertIn((source, target), transitions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
