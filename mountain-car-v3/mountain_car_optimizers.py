# =====================================================================
# Description: contains all optimizers for minimizing differentiable
# cost functions via the abtraction parameters
# =====================================================================

# =====================================================================
# Libraries for the mountain car system
# =====================================================================

import numpy as np
import jax
import jax.numpy as jnp


# =====================================================================
# Gradient descent
# =====================================================================

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
    ):

    @jax.jit
    def gd_step(p, lr_value):
        value, g = jax.value_and_grad(objective_fn)(
            p,
            args=args
        )
        g = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        g_norm = jnp.linalg.norm(g)
        scale = jnp.minimum(1.0, grad_clip / (g_norm + 1e-12))
        g = g * scale
        return p - jnp.asarray(lr_value, dtype=p.dtype) * g, value, g_norm
    
    params_gd = params_init
    cost_history = []
    grad_norm_history = []
    for k in range(steps):
        params_gd, value, g_norm = gd_step(params_gd, lr)

        if k % record_every == 0:
            cost_history.append(float(value))
            grad_norm_history.append(float(g_norm))
        
        if k % print_every == 0:
            print(f"[{k}] J(p)={float(value):.3f}, |∇J(p)|={float(g_norm):.3f}")

    return params_gd, np.array(cost_history), np.array(grad_norm_history)
