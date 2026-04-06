import jax
import jax.numpy as jnp
import numpy as np
from smc_objective_tools import objective, trans_matrix


x1_min, x1_max = -10.0, 10.0
x2_min, x2_max = -10.0, 10.0
x_domain = (x1_min, x1_max, x2_min, x2_max)

x_star = jnp.array([5.0, 5.0])
radius = 2.0

# smaller grid so it runs fast
y1_params_ends = jnp.linspace(-10, 10, 20)
y2_params_ends = jnp.linspace(-10, 10, 20)


def objective_from_params(params):
    theta, a1, a2, h = params
    M = trans_matrix(theta, a1, a2, h)

    return objective(
        x_domain,
        x_star,
        radius,
        y1_params_ends,
        y2_params_ends,
        M
    )


params = jnp.array([0.0, 0.0, 0.0, 0.0])


value_and_grad = jax.value_and_grad(objective_from_params)


lr = 1e-2   

print("Running SGD...")

for step in range(100):
    val, grad = value_and_grad(params)

    params = params - lr * grad

    if step % 10 == 0:
        print(f"Step {step}, Loss: {val}")

print("Final params:", params)
