# Libraries
import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle

CENTER = np.array([5.0, 5.0])

def order_vertices_ccw(verts):
    """Order 2D polygon vertices counter-clockwise to avoid self-intersecting plots."""
    verts = np.asarray(verts, dtype=float)
    c = verts.mean(axis=0)
    angles = np.arctan2(verts[:, 1] - c[1], verts[:, 0] - c[0])
    return verts[np.argsort(angles)]

# Compute bound in y-space
def get_yspace_bounds(
    M,
    x1_min, x1_max,
    x2_min, x2_max
):
    """
    Given invertible M in x = M y, compute bounds on y1,y2 when
    x lies in the axis-aligned rectangle:
        X = [x1_min, x1_max] x [x2_min, x2_max].

    Returns:
        bounds: dict {"y1": (min,max), "y2": (min,max)}
        verts_y: (4,2) array of the 4 vertices in y-space (preimages of x-corners)
    """
    M = np.asarray(M, dtype=float)

    B = np.linalg.inv(M)

    # Four corners of the x-rectangle
    corners_x = np.array([
        [c1, c2]
        for c1, c2 in itertools.product([x1_min, x1_max], [x2_min, x2_max])
    ], dtype=float)

    # Map corners back to y-space: y = M^{-1} x
    verts_y = (B @ corners_x.T).T  # shape (4,2)
    verts_y = order_vertices_ccw(verts_y)

    y1_min, y1_max = float(verts_y[:, 0].min()), float(verts_y[:, 0].max())
    y2_min, y2_max = float(verts_y[:, 1].min()), float(verts_y[:, 1].max())

    bounds = {"y1": (y1_min, y1_max), "y2": (y2_min, y2_max)}
    return bounds, verts_y


# Helper function to find parametric equation intersection with domain boundary
def line_segment_in_rect(p, d, x1_min, x1_max, x2_min, x2_max):
    """Return (t_lo, t_hi) such that p + t d stays inside the rectangle, or None if no intersection."""
    t_lo, t_hi = -np.inf, np.inf

    # x1 constraints
    if abs(d[0]) < 1e-12:
        if not (x1_min <= p[0] <= x1_max):
            return None
    else:
        t1 = (x1_min - p[0]) / d[0]
        t2 = (x1_max - p[0]) / d[0]
        t_lo, t_hi = max(t_lo, min(t1, t2)), min(t_hi, max(t1, t2))

    # x2 constraints
    if abs(d[1]) < 1e-12:
        if not (x2_min <= p[1] <= x2_max):
            return None
    else:
        t1 = (x2_min - p[1]) / d[1]
        t2 = (x2_max - p[1]) / d[1]
        t_lo, t_hi = max(t_lo, min(t1, t2)), min(t_hi, max(t1, t2))

    if t_lo > t_hi:
        return None
    return t_lo, t_hi

# Plotting function for grids in X and Y spaces
def grid_plotter(M, y1_vals, y2_vals, x1_min, x1_max, x2_min, x2_max):

    fig, (ax_y, ax_x) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # Define the X-domain
    verts_x = np.array([
        [x1_min, x2_min],
        [x1_min, x2_max],
        [x1_max, x2_max],
        [x1_max, x2_min],
    ], dtype=float)

    # Define the induced Y-domain
    bounds_y, verts_y = get_yspace_bounds(
        M,
        x1_min, x1_max,
        x2_min, x2_max
    )

    # Vertical family: y1 = c, y2 varies
    for c in y1_vals:
        a = M[:, 0]
        d = M[:, 1]
        p = float(c) * a

        seg = line_segment_in_rect(p, d, x1_min, x1_max, x2_min, x2_max)
        if seg is not None:
            t_lo, t_hi = seg
            t = np.linspace(t_lo, t_hi, 200)
            X = p[None, :] + t[:, None] * d[None, :]
            ax_x.plot(X[:, 0], X[:, 1], linewidth=1)

        y_line_x = float(c) * np.ones(200)
        y_line_y = np.linspace(bounds_y["y2"][0], bounds_y["y2"][1], 200)
        ax_y.plot(y_line_x, y_line_y, linewidth=1)

    # Horizontal family: y2 = c, y1 varies
    for c in y2_vals:
        a = M[:, 1]
        d = M[:, 0]
        p = float(c) * a

        seg = line_segment_in_rect(p, d, x1_min, x1_max, x2_min, x2_max)
        if seg is not None:
            t_lo, t_hi = seg
            t = np.linspace(t_lo, t_hi, 200)
            X = p[None, :] + t[:, None] * d[None, :]
            ax_x.plot(X[:, 0], X[:, 1], linewidth=1)

        y_line_y = float(c) * np.ones(200)
        y_line_x = np.linspace(bounds_y["y1"][0], bounds_y["y1"][1], 200)
        ax_y.plot(y_line_x, y_line_y, linewidth=1)

    # Overlay the X-domain boundary
    poly_x = np.vstack([verts_x, verts_x[0]])
    ax_x.plot(poly_x[:, 0], poly_x[:, 1], marker='o', color='red')
    ax_x.set_title("X-space domain")
    ax_x.set_xlabel("x1")
    ax_x.set_ylabel("x2")
    ax_y.set_aspect("auto")
    ax_x.set_xlim(x1_min-1, x1_max+1)
    ax_x.set_ylim(x2_min-1, x2_max+1)
    ax_x.grid(True, alpha=0.3)

    # Overlay the Y-domain boundary
    poly_y = np.vstack([verts_y, verts_y[0]])
    ax_y.plot(poly_y[:, 0], poly_y[:, 1], marker='o', color='red')
    ax_y.set_title("Y-space domain")
    ax_y.set_xlabel("y1")
    ax_y.set_ylabel("y2")
    ax_y.set_aspect("auto")
    ax_y.set_xlim(bounds_y["y1"][0]-1, bounds_y["y1"][1]+1)
    ax_y.set_ylim(bounds_y["y2"][0]-1, bounds_y["y2"][1]+1)
    ax_y.grid(True, alpha=0.3)

    plt.show()


def plot_x_grid_satisfaction(
    M,
    y1_params,
    y2_params,
    *,
    kripke_structure,
    sat,
    x_star=None,
    radius=None,
    x_domain=None,
    goal_ap="g",
    figsize=(7, 7),
    alpha=0.6,
):
    """Plot the affine grid in X-space and color each cell by satisfaction.

    Parameters
    ----------
    M : array-like shape (2,2)
        Linear map x = M y.
    y1_params, y2_params : array-like
        Monotone grid line coordinates in y-space. Cells are the rectangles
        [y1[i],y1[i+1]] x [y2[j],y2[j+1]].
    kripke_structure : pyModelChecking.kripke.Kripke
        Used only for goal cell indication via its labelling function.
    sat : set[int]
        Set of Kripke state IDs that satisfy the checked state formula.
    x_domain : tuple (x1_min, x1_max, x2_min, x2_max) or None
        If provided, draws the X-domain rectangle.
    goal_ap : str
        Atomic proposition that marks goal cells (default: "g").
    x_star : array-like shape (2,), optional
        Center of goal circle in X-space.
    radius : float, optional
        Radius of goal circle in X-space. If provided with x_star, a circle is drawn.
    """
    M = np.asarray(M, dtype=float)
    y1_params = np.asarray(y1_params, dtype=float)
    y2_params = np.asarray(y2_params, dtype=float)
    sat = set(sat)

    n1 = len(y1_params) - 1
    n2 = len(y2_params) - 1

    # Grab state->AP-set mapping from pyModelChecking
    try:
        labels_by_state = kripke_structure.labelling_function()
    except Exception:
        labels_by_state = getattr(kripke_structure, "_labels", {})

    def state_id(i, j):
        return i * n2 + j

    polys = []
    facecolors = []
    edgecolors = []
    linewidths = []

    for i in range(n1):
        y1_lo, y1_hi = y1_params[i], y1_params[i + 1]
        for j in range(n2):
            y2_lo, y2_hi = y2_params[j], y2_params[j + 1]
            corners_y = np.array(
                [
                    [y1_lo, y2_lo],
                    [y1_lo, y2_hi],
                    [y1_hi, y2_hi],
                    [y1_hi, y2_lo],
                ],
                dtype=float,
            )
            corners_x = (M @ corners_y.T).T
            polys.append(corners_x)

            sid = state_id(i, j)
            is_sat = sid in sat
            aps = labels_by_state.get(sid, set())
            is_goal = goal_ap in aps

            facecolors.append((0.2, 0.7, 0.2, alpha) if is_sat else (0.85, 0.2, 0.2, alpha))
            if is_goal:
                edgecolors.append((0.0, 0.0, 0.0, 1.0))
                linewidths.append(2.0)
            else:
                edgecolors.append((0.0, 0.0, 0.0, 0.25))
                linewidths.append(0.5)

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    coll = PolyCollection(polys, facecolors=facecolors, edgecolors=edgecolors, linewidths=linewidths, closed=True)
    ax.add_collection(coll)

    if x_star is not None and radius is not None:
        x_star = np.asarray(x_star, dtype=float).reshape(2,)
        circ = Circle((float(x_star[0]), float(x_star[1])), float(radius), fill=False, color="k", linewidth=2.0)
        ax.add_patch(circ)

    if x_domain is not None:
        x1_min, x1_max, x2_min, x2_max = map(float, x_domain)
        verts_x = np.array(
            [
                [x1_min, x2_min],
                [x1_min, x2_max],
                [x1_max, x2_max],
                [x1_max, x2_min],
                [x1_min, x2_min],
            ],
            dtype=float,
        )
        ax.plot(verts_x[:, 0], verts_x[:, 1], color="k", linewidth=1.5)
        ax.set_xlim(x1_min, x1_max)
        ax.set_ylim(x2_min, x2_max)
    else:
        all_pts = np.vstack(polys)
        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
        pad = 0.05 * float(np.max(maxs - mins) + 1e-12)
        ax.set_xlim(mins[0] - pad, maxs[0] + pad)
        ax.set_ylim(mins[1] - pad, maxs[1] + pad)

    ax.set_title("x-space grid (green=satisfies, red=fails")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    plt.show()
