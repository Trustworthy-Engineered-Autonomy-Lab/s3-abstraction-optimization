# =====================================================================
# Description: main script for training abstraction parameters
# =====================================================================

# =====================================================================
# Libraries
# =====================================================================

from __future__ import annotations

import contextlib
import hashlib
import io
from itertools import product
import jax
import jax.numpy as jnp
import math
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download

with contextlib.redirect_stderr(io.StringIO()):
    from stable_baselines3 import DDPG


# =====================================================================
# Globals
# =====================================================================

CHECKPOINT_FILENAME = "ddpg-MountainCarContinuous-v0-gymnasium.zip"
ACTOR_CACHE_FILENAME = "mountain_car_actor_derivatives.npz"
LOCAL_CHECKPOINT_PATH = Path(__file__).with_name(CHECKPOINT_FILENAME)
ACTOR_CACHE_PATH = Path(__file__).with_name(ACTOR_CACHE_FILENAME)

POWER = 0.0015
GRAVITY = 0.0025
MIN_POSITION = -1.2
MAX_POSITION = 0.6
MIN_VELOCITY = -0.07
MAX_VELOCITY = 0.07

_CACHE_VERSION = 1
_POINT_BOUNDARY_TOL = 1e-12


# =====================================================================
# Specialized errors
# =====================================================================

class DerivativeDomainError(ValueError):
    """Raised when a box crosses a nonsmooth closed-loop boundary."""

class NonDifferentiableStateError(DerivativeDomainError):
    """Raised when a requested point lies on a nonsmooth boundary."""


# =====================================================================
# Controller loading and actor-weight cache
# =====================================================================

def load_controller() -> DDPG:
    """Load the pretrained controller, downloading it only when necessary."""

    print("Importing pretrained DDPG...")
    if LOCAL_CHECKPOINT_PATH.exists():
        return DDPG.load(LOCAL_CHECKPOINT_PATH, device="cpu")

    checkpoint_path = hf_hub_download(
        repo_id="sb3/ddpg-MountainCarContinuous-v0",
        filename="ddpg-MountainCarContinuous-v0.zip",
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You loaded a model that was trained using OpenAI Gym.*",
            category=UserWarning,
        )
        controller = DDPG.load(
            checkpoint_path,
            device="cpu",
            custom_objects={"lr_schedule": lambda _: 0.0},
        )

    controller.save(LOCAL_CHECKPOINT_PATH)
    return controller


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_actor_arrays(controller: DDPG) -> dict[str, np.ndarray]:
    """Extract the three affine layers and action rescaling from SB3."""

    linear_layers = [
        layer for layer in controller.policy.actor.mu
        if hasattr(layer, "weight") and hasattr(layer, "bias")
    ]
    if len(linear_layers) != 3:
        raise RuntimeError(
            "Expected a three-layer DDPG actor, but found "
            f"{len(linear_layers)} affine layers."
        )

    arrays: dict[str, np.ndarray] = {}
    for index, layer in enumerate(linear_layers, start=1):
        arrays[f"W{index}"] = layer.weight.detach().cpu().numpy().astype(
            np.float64, copy=True
        )
        arrays[f"b{index}"] = layer.bias.detach().cpu().numpy().astype(
            np.float64, copy=True
        )

    action_low = np.asarray(controller.action_space.low, dtype=np.float64)
    action_high = np.asarray(controller.action_space.high, dtype=np.float64)
    if action_low.size != 1 or action_high.size != 1:
        raise RuntimeError("This derivative implementation expects one action.")

    # SB3 actors emit normalized actions in [-1, 1], then predict() maps them
    # affinely into the environment's action space.
    arrays["action_scale"] = np.asarray(
        [(action_high.item() - action_low.item()) / 2.0], dtype=np.float64
    )
    arrays["action_bias"] = np.asarray(
        [(action_high.item() + action_low.item()) / 2.0], dtype=np.float64
    )
    return arrays


def _actor_shapes_are_valid(arrays: dict[str, np.ndarray]) -> bool:
    return (
        arrays["W1"].shape == (400, 2)
        and arrays["b1"].shape == (400,)
        and arrays["W2"].shape == (300, 400)
        and arrays["b2"].shape == (300,)
        and arrays["W3"].shape == (1, 300)
        and arrays["b3"].shape == (1,)
        and arrays["action_scale"].shape == (1,)
        and arrays["action_bias"].shape == (1,)
    )


def _load_or_cache_actor_arrays(controller: DDPG) -> dict[str, np.ndarray]:
    """Load actor tensors from a checkpoint-keyed NumPy cache when possible."""

    checkpoint_hash = _checkpoint_sha256(LOCAL_CHECKPOINT_PATH)
    required_keys = {
        "W1", "b1", "W2", "b2", "W3", "b3",
        "action_scale", "action_bias",
    }

    if ACTOR_CACHE_PATH.exists():
        try:
            with np.load(ACTOR_CACHE_PATH, allow_pickle=False) as cached:
                cached_hash = str(cached["checkpoint_sha256"].item())
                cached_version = int(cached["cache_version"].item())
                arrays = {key: np.array(cached[key], copy=True) for key in required_keys}
            if (
                cached_hash == checkpoint_hash
                and cached_version == _CACHE_VERSION
                and _actor_shapes_are_valid(arrays)
            ):
                print(f"Loading cached actor derivative data from {ACTOR_CACHE_PATH}")
                return arrays
        except (KeyError, OSError, ValueError):
            # A stale or partial generated cache is safe to regenerate.
            pass

    arrays = _extract_actor_arrays(controller)
    if not _actor_shapes_are_valid(arrays):
        raise RuntimeError("The cached-derivative code does not match this actor architecture.")

    temp_path = ACTOR_CACHE_PATH.with_suffix(".tmp.npz")
    try:
        np.savez_compressed(
            temp_path,
            checkpoint_sha256=np.asarray(checkpoint_hash),
            cache_version=np.asarray(_CACHE_VERSION, dtype=np.int64),
            **arrays,
        )
        temp_path.replace(ACTOR_CACHE_PATH)
        print(f"Cached actor derivative data to {ACTOR_CACHE_PATH}")
    except OSError as exc:
        warnings.warn(
            f"Could not write actor derivative cache ({exc}); using in-memory weights.",
            RuntimeWarning,
            stacklevel=2,
        )

    return arrays


# =====================================================================
# More global vars...
# =====================================================================

CONTROLLER = load_controller()
_ACTOR_ARRAYS = _load_or_cache_actor_arrays(CONTROLLER)

_W1 = _ACTOR_ARRAYS["W1"]
_B1 = _ACTOR_ARRAYS["b1"]
_W2 = _ACTOR_ARRAYS["W2"]
_B2 = _ACTOR_ARRAYS["b2"]
_W3 = _ACTOR_ARRAYS["W3"]
_B3 = _ACTOR_ARRAYS["b3"]
_ACTION_SCALE = float(_ACTOR_ARRAYS["action_scale"][0])
_ACTION_BIAS = float(_ACTOR_ARRAYS["action_bias"][0])

_JAX_W1 = jnp.asarray(_W1)
_JAX_B1 = jnp.asarray(_B1)
_JAX_W2 = jnp.asarray(_W2)
_JAX_B2 = jnp.asarray(_B2)
_JAX_W3 = jnp.asarray(_W3)
_JAX_B3 = jnp.asarray(_B3)
_JAX_ACTION_SCALE = jnp.asarray(_ACTION_SCALE)
_JAX_ACTION_BIAS = jnp.asarray(_ACTION_BIAS)


# =====================================================================
# Numeric actor and closed-loop dynamics
# =====================================================================

def _validate_state(state: np.ndarray, *, name: str = "state") -> np.ndarray:
    state_array = np.asarray(state, dtype=np.float64)
    if state_array.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {state_array.shape}.")
    if not np.all(np.isfinite(state_array)):
        raise ValueError(f"{name} must contain only finite values.")
    return state_array


def _validate_state_shape_jax(state, *, name: str = "state"):
    state_array = jnp.asarray(state)
    if state_array.ndim == 0 or state_array.shape[-1] != 2:
        raise ValueError(f"{name} must have trailing shape (2,), got {state_array.shape}.")
    return state_array


def controller_action(state: np.ndarray) -> float:
    """Evaluate the cached DDPG actor directly with NumPy."""

    x = _validate_state(state)
    h1 = np.maximum(_W1 @ x + _B1, 0.0)
    h2 = np.maximum(_W2 @ h1 + _B2, 0.0)
    normalized_action = math.tanh(float((_W3 @ h2 + _B3)[0]))
    action = _ACTION_SCALE * normalized_action + _ACTION_BIAS
    return float(np.clip(action, -1.0, 1.0))


def controller_action_jax(state) -> jnp.ndarray:
    """Evaluate the cached DDPG actor with JAX on one state or a batch."""

    x = _validate_state_shape_jax(state)
    # jax.nn.relu uses the standard neural-network derivative convention at
    # zero, unlike spelling ReLU as maximum(x, 0), whose tie subgradient may
    # differ from the pretrained PyTorch actor.
    h1 = jax.nn.relu(jnp.matmul(x, _JAX_W1.T) + _JAX_B1)
    h2 = jax.nn.relu(jnp.matmul(h1, _JAX_W2.T) + _JAX_B2)
    normalized_action = jnp.tanh(
        jnp.squeeze(jnp.matmul(h2, _JAX_W3.T) + _JAX_B3, axis=-1)
    )
    action = _JAX_ACTION_SCALE * normalized_action + _JAX_ACTION_BIAS
    return jnp.clip(action, -1.0, 1.0)


def ol_system(state: np.ndarray, action: float) -> np.ndarray:
    """Open-loop ``MountainCarContinuous-v0`` transition."""

    p, v = _validate_state(state)
    action_scalar = float(np.clip(action, -1.0, 1.0))
    v_next = np.clip(
        v + action_scalar * POWER - GRAVITY * np.cos(3.0 * p),
        MIN_VELOCITY,
        MAX_VELOCITY,
    )
    p_next = np.clip(p + v_next, MIN_POSITION, MAX_POSITION)
    if p_next <= MIN_POSITION and v_next < 0.0:
        v_next = 0.0
    return np.array([p_next, v_next], dtype=np.float64)


def ol_system_jax(state, action) -> jnp.ndarray:
    """JAX-compatible open-loop MountainCar transition for one state or a batch."""

    x = _validate_state_shape_jax(state)
    action_array = jnp.asarray(action)
    action_scalar = jnp.clip(action_array, -1.0, 1.0)
    position = x[..., 0]
    velocity = x[..., 1]
    next_velocity = jnp.clip(
        velocity + action_scalar * POWER - GRAVITY * jnp.cos(3.0 * position),
        MIN_VELOCITY,
        MAX_VELOCITY,
    )
    next_position = jnp.clip(position + next_velocity, MIN_POSITION, MAX_POSITION)
    next_velocity = jnp.where(
        (next_position <= MIN_POSITION) & (next_velocity < 0.0),
        jnp.zeros_like(next_velocity),
        next_velocity,
    )
    return jnp.stack([next_position, next_velocity], axis=-1)


def cl_system_numeric(state: np.ndarray) -> np.ndarray:
    """Evaluate the closed loop using the cached NumPy actor."""

    x = _validate_state(state)
    return ol_system(x, controller_action(x))


def cl_system_jax(state) -> jnp.ndarray:
    """JAX-compatible closed-loop MountainCar transition for one state or a batch."""

    x = _validate_state_shape_jax(state)
    return ol_system_jax(x, controller_action_jax(x))


def jacobian_jax(state) -> jnp.ndarray:
    """Differentiate the JAX closed loop with respect to one state."""

    x = _validate_state_shape_jax(state)
    if x.ndim != 1:
        raise ValueError(
            "jacobian_jax expects one state with shape (2,); "
            "use jax.vmap(jacobian_jax) for a batch."
        )
    return jax.jacfwd(cl_system_jax)(x)


def cl_system(state: np.ndarray) -> np.ndarray:
    """Closed-loop MountainCar transition (fast NumPy implementation)."""

    return cl_system_numeric(state)


# =====================================================================
# Certified interval closed-loop dynamics
# =====================================================================

def _actor_interval(
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[float, float]:
    """Return an interval enclosure of the actor on ``[lower, upper]``.

    This is standard interval-bound propagation (IBP) through the two ReLU
    layers and the monotone tanh output.  Unlike a sampled Hessian, it remains
    valid when the input box crosses neural-network activation boundaries.
    """

    z1_lower, z1_upper = _affine_interval(_W1, _B1, lower, upper)
    h1_lower = np.maximum(z1_lower, 0.0)
    h1_upper = np.maximum(z1_upper, 0.0)

    z2_lower, z2_upper = _affine_interval(
        _W2, _B2, h1_lower, h1_upper
    )
    h2_lower = np.maximum(z2_lower, 0.0)
    h2_upper = np.maximum(z2_upper, 0.0)

    z3_lower, z3_upper = _affine_interval(
        _W3, _B3, h2_lower, h2_upper
    )
    normalized_lower = math.tanh(float(z3_lower[0]))
    normalized_upper = math.tanh(float(z3_upper[0]))
    action_candidates = (
        _ACTION_SCALE * normalized_lower + _ACTION_BIAS,
        _ACTION_SCALE * normalized_upper + _ACTION_BIAS,
    )
    return (
        max(-1.0, min(1.0, min(action_candidates))),
        max(-1.0, min(1.0, max(action_candidates))),
    )


def interval_cl_system(
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Certify a closed-loop post-image AABB for a state-space box.

    The actor is enclosed by IBP.  The cosine range is exact on the supplied
    position interval, and the plant's clips and left-wall reset are handled
    by interval extensions of the corresponding environment operations.
    The result is valid across ReLU, clipping, and reset boundaries.
    """

    lower, upper = _validate_box(lower_bounds, upper_bounds)
    action_lower, action_upper = _actor_interval(lower, upper)
    cos_lower, cos_upper = _cos_interval(3.0 * lower[0], 3.0 * upper[0])

    raw_velocity_lower = (
        lower[1] + POWER * action_lower - GRAVITY * cos_upper
    )
    raw_velocity_upper = (
        upper[1] + POWER * action_upper - GRAVITY * cos_lower
    )
    velocity_lower = float(np.clip(
        raw_velocity_lower, MIN_VELOCITY, MAX_VELOCITY
    ))
    velocity_upper = float(np.clip(
        raw_velocity_upper, MIN_VELOCITY, MAX_VELOCITY
    ))

    raw_position_lower = lower[0] + velocity_lower
    raw_position_upper = upper[0] + velocity_upper
    position_lower = float(np.clip(
        raw_position_lower, MIN_POSITION, MAX_POSITION
    ))
    position_upper = float(np.clip(
        raw_position_upper, MIN_POSITION, MAX_POSITION
    ))

    # A negative velocity is reset to zero only when the unclipped position
    # can hit the left wall.  Including zero in that case encloses both reset
    # and non-reset branches without polluting every velocity interval.
    reset_is_possible = (
        raw_position_lower <= MIN_POSITION and velocity_lower < 0.0
    )
    if reset_is_possible:
        velocity_lower = min(velocity_lower, 0.0)
        velocity_upper = max(velocity_upper, 0.0)

    return (
        np.array([position_lower, velocity_lower], dtype=np.float64),
        np.array([position_upper, velocity_upper], dtype=np.float64),
    )


def _affine_interval_jax(matrix, bias, lower, upper):
    """Batched JAX interval extension of an affine map."""

    positive = jnp.maximum(matrix, 0.0)
    negative = jnp.minimum(matrix, 0.0)
    out_lower = (
        jnp.matmul(lower, positive.T)
        + jnp.matmul(upper, negative.T)
        + bias
    )
    out_upper = (
        jnp.matmul(upper, positive.T)
        + jnp.matmul(lower, negative.T)
        + bias
    )
    return out_lower, out_upper


def _cos_interval_jax(lower, upper):
    """Exact cosine interval with batched, JAX-compatible inputs."""

    first = lower
    second = upper
    lower = jnp.minimum(first, second)
    upper = jnp.maximum(first, second)
    endpoint_lower = jnp.minimum(jnp.cos(lower), jnp.cos(upper))
    endpoint_upper = jnp.maximum(jnp.cos(lower), jnp.cos(upper))
    period = jnp.asarray(2.0 * math.pi, dtype=lower.dtype)

    def contains(phase):
        first = jnp.ceil((lower - phase) / period)
        last = jnp.floor((upper - phase) / period)
        return first <= last

    spans_period = upper - lower >= period
    out_lower = jnp.where(
        spans_period | contains(jnp.asarray(math.pi, dtype=lower.dtype)),
        -jnp.ones_like(endpoint_lower),
        endpoint_lower,
    )
    out_upper = jnp.where(
        spans_period | contains(jnp.asarray(0.0, dtype=lower.dtype)),
        jnp.ones_like(endpoint_upper),
        endpoint_upper,
    )
    return out_lower, out_upper


def interval_cl_system_jax(lower_bounds, upper_bounds):
    """Differentiable batched counterpart of :func:`interval_cl_system`."""

    lower = _validate_state_shape_jax(lower_bounds, name="lower_bounds")
    upper = _validate_state_shape_jax(upper_bounds, name="upper_bounds")

    z1_lower, z1_upper = _affine_interval_jax(
        _JAX_W1, _JAX_B1, lower, upper
    )
    h1_lower = jax.nn.relu(z1_lower)
    h1_upper = jax.nn.relu(z1_upper)
    z2_lower, z2_upper = _affine_interval_jax(
        _JAX_W2, _JAX_B2, h1_lower, h1_upper
    )
    h2_lower = jax.nn.relu(z2_lower)
    h2_upper = jax.nn.relu(z2_upper)
    z3_lower, z3_upper = _affine_interval_jax(
        _JAX_W3, _JAX_B3, h2_lower, h2_upper
    )
    normalized_lower = jnp.tanh(jnp.squeeze(z3_lower, axis=-1))
    normalized_upper = jnp.tanh(jnp.squeeze(z3_upper, axis=-1))
    action_first = _JAX_ACTION_SCALE * normalized_lower + _JAX_ACTION_BIAS
    action_second = _JAX_ACTION_SCALE * normalized_upper + _JAX_ACTION_BIAS
    action_lower = jnp.clip(jnp.minimum(action_first, action_second), -1.0, 1.0)
    action_upper = jnp.clip(jnp.maximum(action_first, action_second), -1.0, 1.0)

    position_lower = lower[..., 0]
    position_upper = upper[..., 0]
    velocity_lower = lower[..., 1]
    velocity_upper = upper[..., 1]
    cos_lower, cos_upper = _cos_interval_jax(
        3.0 * position_lower,
        3.0 * position_upper,
    )
    raw_velocity_lower = (
        velocity_lower + POWER * action_lower - GRAVITY * cos_upper
    )
    raw_velocity_upper = (
        velocity_upper + POWER * action_upper - GRAVITY * cos_lower
    )
    next_velocity_lower = jnp.clip(
        raw_velocity_lower, MIN_VELOCITY, MAX_VELOCITY
    )
    next_velocity_upper = jnp.clip(
        raw_velocity_upper, MIN_VELOCITY, MAX_VELOCITY
    )

    raw_position_lower = position_lower + next_velocity_lower
    raw_position_upper = position_upper + next_velocity_upper
    next_position_lower = jnp.clip(
        raw_position_lower, MIN_POSITION, MAX_POSITION
    )
    next_position_upper = jnp.clip(
        raw_position_upper, MIN_POSITION, MAX_POSITION
    )

    reset_is_possible = (
        (raw_position_lower <= MIN_POSITION) & (next_velocity_lower < 0.0)
    )
    next_velocity_lower = jnp.where(
        reset_is_possible,
        jnp.minimum(next_velocity_lower, 0.0),
        next_velocity_lower,
    )
    next_velocity_upper = jnp.where(
        reset_is_possible,
        jnp.maximum(next_velocity_upper, 0.0),
        next_velocity_upper,
    )

    return (
        jnp.stack([next_position_lower, next_velocity_lower], axis=-1),
        jnp.stack([next_position_upper, next_velocity_upper], axis=-1),
    )


# =====================================================================
# Exact point derivatives within one smooth branch
# =====================================================================

def _actor_point_derivatives(
    state: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return actor value, gradient, and Hessian at a differentiable point."""

    x = _validate_state(state)
    z1 = _W1 @ x + _B1
    if np.any(np.abs(z1) <= _POINT_BOUNDARY_TOL):
        raise NonDifferentiableStateError(
            "The state lies on (or numerically too close to) a layer-1 ReLU boundary."
        )
    mask1 = z1 > 0.0

    h1 = np.where(mask1, z1, 0.0)
    z2 = _W2 @ h1 + _B2
    if np.any(np.abs(z2) <= _POINT_BOUNDARY_TOL):
        raise NonDifferentiableStateError(
            "The state lies on (or numerically too close to) a layer-2 ReLU boundary."
        )
    mask2 = z2 > 0.0

    h2 = np.where(mask2, z2, 0.0)
    z3 = float((_W3 @ h2 + _B3)[0])
    normalized_action = math.tanh(z3)
    action = _ACTION_SCALE * normalized_action + _ACTION_BIAS

    # z3 is affine while mask1 and mask2 remain fixed.
    z3_gradient = (
        (_W3[0] * mask2.astype(np.float64))
        @ _W2
        @ (mask1[:, None].astype(np.float64) * _W1)
    )
    action_gradient = (
        _ACTION_SCALE * (1.0 - normalized_action**2) * z3_gradient
    )
    tanh_second = -2.0 * normalized_action * (1.0 - normalized_action**2)
    action_hessian = (
        _ACTION_SCALE * tanh_second * np.outer(z3_gradient, z3_gradient)
    )

    return float(action), action_gradient, action_hessian

def _closed_loop_point_derivatives(
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return F, J, and H while respecting the active plant branches."""

    x = _validate_state(state)
    p, v = x
    action, action_gradient, action_hessian = _actor_point_derivatives(x)

    raw_velocity = v + POWER * action - GRAVITY * math.cos(3.0 * p)
    if (
        abs(raw_velocity - MIN_VELOCITY) <= _POINT_BOUNDARY_TOL
        or abs(raw_velocity - MAX_VELOCITY) <= _POINT_BOUNDARY_TOL
    ):
        raise NonDifferentiableStateError("The state lies on a velocity-clipping boundary.")

    if MIN_VELOCITY < raw_velocity < MAX_VELOCITY:
        velocity = raw_velocity
        velocity_gradient = POWER * action_gradient + np.array(
            [3.0 * GRAVITY * math.sin(3.0 * p), 1.0]
        )
        velocity_hessian = POWER * action_hessian
        velocity_hessian = np.array(velocity_hessian, copy=True)
        velocity_hessian[0, 0] += 9.0 * GRAVITY * math.cos(3.0 * p)
    else:
        velocity = float(np.clip(raw_velocity, MIN_VELOCITY, MAX_VELOCITY))
        velocity_gradient = np.zeros(2, dtype=np.float64)
        velocity_hessian = np.zeros((2, 2), dtype=np.float64)

    raw_position = p + velocity
    if (
        abs(raw_position - MIN_POSITION) <= _POINT_BOUNDARY_TOL
        or abs(raw_position - MAX_POSITION) <= _POINT_BOUNDARY_TOL
    ):
        raise NonDifferentiableStateError("The state lies on a position-clipping boundary.")

    if MIN_POSITION < raw_position < MAX_POSITION:
        position = raw_position
        position_gradient = np.array([1.0, 0.0]) + velocity_gradient
        position_hessian = np.array(velocity_hessian, copy=True)
    else:
        position = float(np.clip(raw_position, MIN_POSITION, MAX_POSITION))
        position_gradient = np.zeros(2, dtype=np.float64)
        position_hessian = np.zeros((2, 2), dtype=np.float64)

    if position <= MIN_POSITION:
        if abs(velocity) <= _POINT_BOUNDARY_TOL:
            raise NonDifferentiableStateError(
                "The state lies on the left-boundary velocity-reset condition."
            )
        if velocity < 0.0:
            velocity = 0.0
            velocity_gradient = np.zeros(2, dtype=np.float64)
            velocity_hessian = np.zeros((2, 2), dtype=np.float64)

    value = np.array([position, velocity], dtype=np.float64)
    jac = np.stack([position_gradient, velocity_gradient])
    hess = np.stack([position_hessian, velocity_hessian])
    return value, jac, hess


def jacobian(state: np.ndarray) -> np.ndarray:
    """Evaluate the analytic 2-by-2 closed-loop Jacobian at ``state``."""

    return _closed_loop_point_derivatives(state)[1]


def hessian(state: np.ndarray) -> np.ndarray:
    """Evaluate the analytic 2-by-2-by-2 Hessian at ``state``."""

    return _closed_loop_point_derivatives(state)[2]


# =====================================================================
# Interval Hessian
# =====================================================================

def _affine_interval(
    matrix: np.ndarray,
    bias: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positive = np.maximum(matrix, 0.0)
    negative = np.minimum(matrix, 0.0)
    out_lower = positive @ lower + negative @ upper + bias
    out_upper = positive @ upper + negative @ lower + bias
    return out_lower, out_upper


def _fixed_relu_mask(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    layer: int,
) -> np.ndarray:
    ambiguous = (lower <= 0.0) & (upper >= 0.0)
    if np.any(ambiguous):
        raise DerivativeDomainError(
            f"The box crosses or touches {int(np.sum(ambiguous))} layer-{layer} "
            "ReLU boundary/boundaries. Subdivide the box or use a "
            "Jacobian-based nonsmooth remainder bound."
        )
    return lower > 0.0


def _contains_periodic_point(
    lower: float,
    upper: float,
    phase: float,
    period: float,
) -> bool:
    first_index = math.ceil((lower - phase) / period)
    last_index = math.floor((upper - phase) / period)
    return first_index <= last_index


def _cos_interval(lower: float, upper: float) -> tuple[float, float]:
    if lower > upper:
        lower, upper = upper, lower
    if upper - lower >= 2.0 * math.pi:
        return -1.0, 1.0

    endpoint_values = [math.cos(lower), math.cos(upper)]
    out_lower = min(endpoint_values)
    out_upper = max(endpoint_values)
    if _contains_periodic_point(lower, upper, 0.0, 2.0 * math.pi):
        out_upper = 1.0
    if _contains_periodic_point(lower, upper, math.pi, 2.0 * math.pi):
        out_lower = -1.0
    return out_lower, out_upper


def _tanh_second_interval(lower: float, upper: float) -> tuple[float, float]:
    """Tight range of tanh'' on a scalar interval."""

    def tanh_second(value: float) -> float:
        tangent = math.tanh(value)
        return -2.0 * tangent * (1.0 - tangent * tangent)

    critical = math.atanh(1.0 / math.sqrt(3.0))
    candidates = [tanh_second(lower), tanh_second(upper)]
    for point in (-critical, critical):
        if lower <= point <= upper:
            candidates.append(tanh_second(point))
    return min(candidates), max(candidates)


def _scale_interval(lower: float, upper: float, scalar: float) -> tuple[float, float]:
    first = scalar * lower
    second = scalar * upper
    return min(first, second), max(first, second)


def _actor_interval_hessian(
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Certify a fixed actor region and bound its action Hessian."""

    z1_lower, z1_upper = _affine_interval(_W1, _B1, lower, upper)
    mask1 = _fixed_relu_mask(z1_lower, z1_upper, layer=1)

    # With mask1 fixed, layer 1 and therefore layer 2 preactivations are
    # affine functions of the original two-dimensional state.
    h1_matrix = mask1[:, None].astype(np.float64) * _W1
    h1_bias = mask1.astype(np.float64) * _B1
    z2_matrix = _W2 @ h1_matrix
    z2_bias = _W2 @ h1_bias + _B2
    z2_lower, z2_upper = _affine_interval(
        z2_matrix, z2_bias, lower, upper
    )
    mask2 = _fixed_relu_mask(z2_lower, z2_upper, layer=2)

    h2_matrix = mask2[:, None].astype(np.float64) * z2_matrix
    h2_bias = mask2.astype(np.float64) * z2_bias
    z3_gradient = _W3 @ h2_matrix
    z3_bias = _W3 @ h2_bias + _B3
    z3_lower_array, z3_upper_array = _affine_interval(
        z3_gradient, z3_bias, lower, upper
    )
    z3_lower = float(z3_lower_array[0])
    z3_upper = float(z3_upper_array[0])

    normalized_lower = math.tanh(z3_lower)
    normalized_upper = math.tanh(z3_upper)
    action_candidates = (
        _ACTION_SCALE * normalized_lower + _ACTION_BIAS,
        _ACTION_SCALE * normalized_upper + _ACTION_BIAS,
    )
    action_lower = min(action_candidates)
    action_upper = max(action_candidates)

    second_lower, second_upper = _tanh_second_interval(z3_lower, z3_upper)
    second_lower, second_upper = _scale_interval(
        second_lower, second_upper, _ACTION_SCALE
    )
    gradient = z3_gradient[0]
    outer = np.outer(gradient, gradient)
    hessian_lower = np.empty((2, 2), dtype=np.float64)
    hessian_upper = np.empty((2, 2), dtype=np.float64)
    for row, column in np.ndindex(2, 2):
        entry_lower, entry_upper = _scale_interval(
            second_lower, second_upper, float(outer[row, column])
        )
        hessian_lower[row, column] = entry_lower
        hessian_upper[row, column] = entry_upper

    return action_lower, action_upper, hessian_lower, hessian_upper


def _validate_box(
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = _validate_state(lower_bounds, name="lower_bounds")
    upper = _validate_state(upper_bounds, name="upper_bounds")
    if np.any(lower > upper):
        raise ValueError("Every lower bound must be less than or equal to its upper bound.")
    return lower, upper


def _strict_interval_hessian(
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a certified interval enclosure on one smooth branch.
    """

    lower, upper = _validate_box(lower_bounds, upper_bounds)
    action_lower, action_upper, action_h_lower, action_h_upper = (
        _actor_interval_hessian(lower, upper)
    )

    cos_lower, cos_upper = _cos_interval(3.0 * lower[0], 3.0 * upper[0])
    raw_velocity_lower = (
        lower[1] + POWER * action_lower - GRAVITY * cos_upper
    )
    raw_velocity_upper = (
        upper[1] + POWER * action_upper - GRAVITY * cos_lower
    )

    if raw_velocity_upper < MIN_VELOCITY:
        velocity_lower = velocity_upper = MIN_VELOCITY
        velocity_h_lower = np.zeros((2, 2), dtype=np.float64)
        velocity_h_upper = np.zeros((2, 2), dtype=np.float64)
    elif raw_velocity_lower > MAX_VELOCITY:
        velocity_lower = velocity_upper = MAX_VELOCITY
        velocity_h_lower = np.zeros((2, 2), dtype=np.float64)
        velocity_h_upper = np.zeros((2, 2), dtype=np.float64)
    elif (
        raw_velocity_lower > MIN_VELOCITY
        and raw_velocity_upper < MAX_VELOCITY
    ):
        velocity_lower = raw_velocity_lower
        velocity_upper = raw_velocity_upper
        velocity_h_lower = POWER * action_h_lower
        velocity_h_upper = POWER * action_h_upper
        curvature_lower, curvature_upper = _scale_interval(
            cos_lower, cos_upper, 9.0 * GRAVITY
        )
        velocity_h_lower = np.array(velocity_h_lower, copy=True)
        velocity_h_upper = np.array(velocity_h_upper, copy=True)
        velocity_h_lower[0, 0] += curvature_lower
        velocity_h_upper[0, 0] += curvature_upper
    else:
        raise DerivativeDomainError(
            "The box crosses or touches a velocity-clipping boundary. "
            "Subdivide the box before using a Hessian Taylor bound."
        )

    raw_position_lower = lower[0] + velocity_lower
    raw_position_upper = upper[0] + velocity_upper
    if raw_position_upper < MIN_POSITION:
        position_lower = position_upper = MIN_POSITION
        position_h_lower = np.zeros((2, 2), dtype=np.float64)
        position_h_upper = np.zeros((2, 2), dtype=np.float64)
        position_is_left_clipped = True
    elif raw_position_lower > MAX_POSITION:
        position_lower = position_upper = MAX_POSITION
        position_h_lower = np.zeros((2, 2), dtype=np.float64)
        position_h_upper = np.zeros((2, 2), dtype=np.float64)
        position_is_left_clipped = False
    elif raw_position_lower > MIN_POSITION and raw_position_upper < MAX_POSITION:
        position_lower = raw_position_lower
        position_upper = raw_position_upper
        position_h_lower = np.array(velocity_h_lower, copy=True)
        position_h_upper = np.array(velocity_h_upper, copy=True)
        position_is_left_clipped = False
    else:
        raise DerivativeDomainError(
            "The box crosses or touches a position-clipping boundary. "
            "Subdivide the box before using a Hessian Taylor bound."
        )

    if position_is_left_clipped:
        if velocity_upper < 0.0:
            # The environment resets negative velocity at the left wall.
            velocity_h_lower = np.zeros((2, 2), dtype=np.float64)
            velocity_h_upper = np.zeros((2, 2), dtype=np.float64)
        elif velocity_lower > 0.0:
            pass
        else:
            raise DerivativeDomainError(
                "The box crosses or touches the left-wall velocity-reset boundary. "
                "Subdivide the box before using a Hessian Taylor bound."
            )

    _ = (position_lower, position_upper)
    hessian_lower = np.stack([position_h_lower, velocity_h_lower])
    hessian_upper = np.stack([position_h_upper, velocity_h_upper])
    return hessian_lower, hessian_upper


def inset_domain_box(
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    *,
    inset: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Move box faces lying on the physical state boundary slightly inward.

    This helper changes only faces at or beyond MountainCar's physical domain
    limits.  Interior faces are left untouched.  It is intended for relaxed
    derivative estimation, not for a formal certificate over the original
    closed box.
    """

    lower, upper = _validate_box(lower_bounds, upper_bounds)
    inset_value = float(inset)
    if not np.isfinite(inset_value) or inset_value < 0.0:
        raise ValueError("inset must be a finite nonnegative scalar.")

    domain_lower = np.array([MIN_POSITION, MIN_VELOCITY], dtype=np.float64)
    domain_upper = np.array([MAX_POSITION, MAX_VELOCITY], dtype=np.float64)
    domain_width = domain_upper - domain_lower
    if np.any(inset_value * 2.0 >= domain_width):
        raise ValueError("inset is too large for the MountainCar state domain.")

    adjusted_lower = np.array(lower, copy=True)
    adjusted_upper = np.array(upper, copy=True)
    lower_faces = adjusted_lower <= domain_lower
    upper_faces = adjusted_upper >= domain_upper
    adjusted_lower[lower_faces] = domain_lower[lower_faces] + inset_value
    adjusted_upper[upper_faces] = domain_upper[upper_faces] - inset_value

    if np.any(adjusted_lower > adjusted_upper):
        raise ValueError("Insetting collapsed the requested derivative box.")
    return adjusted_lower, adjusted_upper


def _sampled_interval_hessian(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a Hessian range by sampling analytic point Hessians."""

    if isinstance(resolution, bool) or int(resolution) != resolution or resolution < 2:
        raise ValueError("approximate_resolution must be an integer of at least 2.")
    resolution = int(resolution)

    axes = [
        np.linspace(axis_lower, axis_upper, resolution)
        for axis_lower, axis_upper in zip(lower, upper)
    ]
    center = 0.5 * (lower + upper)
    span = upper - lower
    sampled_hessians: list[np.ndarray] = []

    for coordinates in product(*axes):
        point = np.asarray(coordinates, dtype=np.float64)
        candidates = [
            point,
            point + 1e-7 * (center - point),
            np.clip(point + 1e-7 * span, lower, upper),
            np.clip(point - 1e-7 * span, lower, upper),
        ]
        for candidate in candidates:
            try:
                sampled_hessians.append(hessian(candidate))
                break
            except NonDifferentiableStateError:
                continue

    if not sampled_hessians:
        raise DerivativeDomainError(
            "Could not evaluate a relaxed Hessian at any point in the box."
        )

    stacked = np.stack(sampled_hessians)
    hessian_lower = np.min(stacked, axis=0)
    hessian_upper = np.max(stacked, axis=0)

    # Outward padding prevents tiny floating-point reductions from producing
    # an accidentally inverted or zero-width numerical enclosure.
    padding = 1e-12 + 1e-10 * np.maximum(
        np.abs(hessian_lower), np.abs(hessian_upper)
    )
    return hessian_lower - padding, hessian_upper + padding


def interval_hessian(
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    *,
    strict: bool = True,
    boundary_inset: float = 1e-9,
    approximate_resolution: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a strict or relaxed closed-loop interval Hessian.

    In ``strict=True`` mode (the default), this returns a certified enclosure
    and raises :class:`DerivativeDomainError` whenever the box crosses a ReLU,
    clip, or reset boundary.

    In ``strict=False`` mode, physical domain faces are first moved inward by
    ``boundary_inset``.  The certified routine is used if the adjusted box is
    smooth; otherwise point Hessians are sampled on an
    ``approximate_resolution`` square grid and their componentwise range is
    returned.  This is useful for abstraction construction but is not a formal
    Taylor certificate across nonsmooth boundaries.
    """

    lower, upper = _validate_box(lower_bounds, upper_bounds)
    if strict:
        return _strict_interval_hessian(lower, upper)

    adjusted_lower, adjusted_upper = inset_domain_box(
        lower, upper, inset=boundary_inset
    )
    try:
        return _strict_interval_hessian(adjusted_lower, adjusted_upper)
    except DerivativeDomainError:
        return _sampled_interval_hessian(
            adjusted_lower,
            adjusted_upper,
            resolution=approximate_resolution,
        )


# =====================================================================
# Linear model and Taylor remainder
# =====================================================================

def linear_cl_system(
    state: np.ndarray,
    center: np.ndarray,
    *,
    J: np.ndarray | None = None,
    f_center: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the first-order closed-loop model around ``center``."""

    x = _validate_state(state)
    expansion_center = _validate_state(center, name="center")
    if J is None:
        J = jacobian(expansion_center)
    if f_center is None:
        f_center = cl_system_numeric(expansion_center)
    return np.asarray(J, dtype=np.float64) @ (x - expansion_center) + np.asarray(
        f_center, dtype=np.float64
    )


def _interval_mul(
    first_lower: float,
    first_upper: float,
    second_lower: float,
    second_upper: float,
) -> tuple[float, float]:
    products = (
        first_lower * second_lower,
        first_lower * second_upper,
        first_upper * second_lower,
        first_upper * second_upper,
    )
    return min(products), max(products)


def taylor_remainder(
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    *,
    strict: bool = True,
    boundary_inset: float = 1e-9,
    approximate_resolution: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate or certify the first-order remainder around the box midpoint.

    ``strict=False`` forwards to the relaxed sampled-Hessian behavior of
    :func:`interval_hessian`.  The original box half-width is retained even
    when its physical boundary faces are inset for derivative evaluation.
    """

    lower, upper = _validate_box(lower_bounds, upper_bounds)
    hessian_lower, hessian_upper = interval_hessian(
        lower,
        upper,
        strict=strict,
        boundary_inset=boundary_inset,
        approximate_resolution=approximate_resolution,
    )
    half_width = 0.5 * (upper - lower)

    remainder_lower = np.zeros(2, dtype=np.float64)
    remainder_upper = np.zeros(2, dtype=np.float64)
    for output in range(2):
        for first_state in range(2):
            for second_state in range(2):
                if first_state == second_state:
                    displacement_lower = 0.0
                    displacement_upper = half_width[first_state] ** 2
                else:
                    radius = half_width[first_state] * half_width[second_state]
                    displacement_lower = -radius
                    displacement_upper = radius

                term_lower, term_upper = _interval_mul(
                    hessian_lower[output, first_state, second_state],
                    hessian_upper[output, first_state, second_state],
                    displacement_lower,
                    displacement_upper,
                )
                remainder_lower[output] += 0.5 * term_lower
                remainder_upper[output] += 0.5 * term_upper

    return remainder_lower, remainder_upper


# =====================================================================
# Approximate the Lipschitz array
# =====================================================================

def lipschitz_array(
    domain_lb,
    domain_ub,
    points_per_dim=41,
    batch_size=8192
):
    """
    Grid estimate of the componentwise Lipschitz array
    """
    domain_lb = np.asarray(domain_lb, dtype=float)
    domain_ub = np.asarray(domain_ub, dtype=float)

    axes = [
        np.linspace(domain_lb[i], domain_ub[i], points_per_dim)
        for i in range(2)
    ]

    mesh = np.meshgrid(*axes, indexing="ij")
    states = np.stack(mesh, axis=-1).reshape(-1, 2)

    batched_jacobian = jax.jit(jax.vmap(jacobian_jax))

    L = np.zeros((2, 2), dtype=float)

    for start in range(0, len(states), batch_size):
        state_batch = jnp.asarray(states[start:start + batch_size])

        J = np.asarray(batched_jacobian(state_batch))

        L_batch = np.max(np.abs(J), axis=0)
        L = np.maximum(L, L_batch)

    return L

if __name__ == "__main__":

    domain_lb = np.array([-1.2, -0.07])
    domain_ub = np.array([0.6, 0.07])
    goal_position = 0.45
    position_values = np.linspace(domain_lb[0], domain_ub[0], 25)
    velocity_values = np.linspace(domain_lb[1], domain_ub[1], 17)
    grid_position, grid_velocity = np.meshgrid(position_values, velocity_values)

    states = np.column_stack([grid_position.ravel(), grid_velocity.ravel()])
    next_states = np.stack([cl_system(state) for state in states])
    displacement = next_states - states

    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.axvspan(
        goal_position,
        MAX_POSITION,
        facecolor="#2a9d8f",
        alpha=0.25,
        label=r"Goal: $x_1 \geq 0.45$",
        zorder=1,
    )
    axis.quiver(
        grid_position,
        grid_velocity,
        displacement[:, 0].reshape(grid_position.shape),
        displacement[:, 1].reshape(grid_velocity.shape),
        color="#264653",
        angles="xy",
        scale_units="xy",
        scale=0.5,
        width=0.003,
        headwidth=4,
        headlength=5,
        zorder=2,
    )
    axis.set_xlabel(r"$x_1$ (position)")
    axis.set_ylabel(r"$x_2$ (velocity)")
    axis.set_xlim(domain_lb[0], domain_ub[0])
    axis.set_ylim(domain_lb[1], domain_ub[1])
    # axis.legend(loc="upper left", frameon=True)
    plt.show()
