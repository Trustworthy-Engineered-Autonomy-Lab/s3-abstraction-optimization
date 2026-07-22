"""Optimizers for Mountain Car abstraction parameters."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def gradient_descent(
    params_init,
    objective_fn,
    *,
    args,
    steps,
    lr,
    grad_clip,
    record_every=10,
    print_every=10,
    return_best=True,
):
    """Run clipped gradient descent and optionally return the best iterate."""

    @jax.jit
    def gd_step(params, lr_value):
        value, gradient = jax.value_and_grad(objective_fn)(params, args=args)
        gradient = jnp.nan_to_num(
            gradient, nan=0.0, posinf=0.0, neginf=0.0
        )
        gradient_norm = jnp.linalg.norm(gradient)
        scale = jnp.minimum(
            1.0, grad_clip / (gradient_norm + 1e-12)
        )
        clipped_gradient = gradient * scale
        updated = params - jnp.asarray(
            lr_value, dtype=params.dtype
        ) * clipped_gradient
        return updated, value, gradient_norm

    params_gd = params_init
    best_params = params_init
    best_value = np.inf
    cost_history = []
    grad_norm_history = []

    for step in range(steps):
        evaluated_params = params_gd
        params_gd, value, gradient_norm = gd_step(params_gd, lr)
        value_float = float(value)
        if value_float < best_value:
            best_value = value_float
            best_params = evaluated_params

        if step % record_every == 0:
            cost_history.append(value_float)
            grad_norm_history.append(float(gradient_norm))

        if step % print_every == 0:
            print(
                f"[{step}] J(p)={value_float:.8f}, "
                f"|grad J(p)|={float(gradient_norm):.8f}"
            )

    final_value = float(objective_fn(params_gd, args=args))
    if final_value < best_value:
        best_params = params_gd

    output_params = best_params if return_best else params_gd
    return (
        output_params,
        np.asarray(cost_history),
        np.asarray(grad_norm_history),
    )
