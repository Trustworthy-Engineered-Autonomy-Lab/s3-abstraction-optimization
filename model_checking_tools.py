# Libraries
import numpy as np
import pyModelChecking as pmc
import grid_plot_tools as gpt
import pyModelChecking.CTL as CTL
from pyModelChecking.CTL import A, E, G, F, Imply
import time


XSTAR = np.array([5.0, 5.0])
RADIUS = 2.0

def dynamics(state):
    A = np.array([[0.8, -0.3],
                  [0.3,  0.8]])
    x_star = XSTAR
    state = np.asarray(state, dtype=float)
    return (state - x_star) @ A.T + x_star


# Helper to find intersecting cells from AABB in y-space
def intersecting_cells_from_y_aabb(y1_params, y2_params, y1_min, y1_max, y2_min, y2_max):
    """Sound over-approx: return all (i,j) whose Y-cell rectangle overlaps the given AABB.
    """
    y1_params = np.asarray(y1_params, dtype=float)
    y2_params = np.asarray(y2_params, dtype=float)
    n1 = len(y1_params) - 1
    n2 = len(y2_params) - 1

    # If completely outside the overall grid bounds, no in-grid successors.
    grid_y1_min, grid_y1_max = float(y1_params[0]), float(y1_params[-1])
    grid_y2_min, grid_y2_max = float(y2_params[0]), float(y2_params[-1])
    if y1_max < grid_y1_min or y1_min > grid_y1_max or y2_max < grid_y2_min or y2_min > grid_y2_max:
        return []

    # Use side="right" for both ends.
    # This avoids the empty-set edge case when y_min == y_max equals a grid line:
    #   side="left" would assign the max to the lower cell, potentially giving i_hi < i_lo.
    i_lo = int(np.searchsorted(y1_params, y1_min, side="right") - 1)
    i_hi = int(np.searchsorted(y1_params, y1_max, side="right") - 1)
    j_lo = int(np.searchsorted(y2_params, y2_min, side="right") - 1)
    j_hi = int(np.searchsorted(y2_params, y2_max, side="right") - 1)

    i_lo = max(0, min(n1 - 1, i_lo))
    i_hi = max(0, min(n1 - 1, i_hi))
    j_lo = max(0, min(n2 - 1, j_lo))
    j_hi = max(0, min(n2 - 1, j_hi))

    if i_hi < i_lo or j_hi < j_lo:
        return []

    return [(i, j) for i in range(i_lo, i_hi + 1) for j in range(j_lo, j_hi + 1)]


# Helper to check if point is in valid y domain
def point_in_parallelogram(p, verts, tol=1e-9):
    """Return True iff point p is inside/on the parallelogram defined by verts (CCW order)."""
    verts = np.asarray(verts, dtype=float)
    p = np.asarray(p, dtype=float)

    v0 = verts[0]
    e1 = verts[1] - v0
    e2 = verts[-1] - v0
    E = np.column_stack([e1, e2])  # 2x2

    # Solve p = v0 + a*e1 + b*e2
    ab = np.linalg.solve(E, p - v0)
    a, b = float(ab[0]), float(ab[1])
    return (-tol <= a <= 1.0 + tol) and (-tol <= b <= 1.0 + tol)


# Goal state labeling function
def is_goal_state(vertices):

    x_star = XSTAR
    radius = RADIUS

    # Return True iff all corners of the x-space cell are within the goal circle
    for v in vertices:
        if np.linalg.norm(v - x_star) > radius:
            return False
    return True

# Out-of-bounds state labeling function
def is_oob_state(vertices, x_bounds, y_bounds):

    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    # Return True iff any vertex is out of bounds
    for v in vertices:
        if (v[0] < x_min) or (v[0] > x_max) or (v[1] < y_min) or (v[1] > y_max):
            return True
    return False




def make_kripke(x1_params, x2_params, allow_self_loops=True, advanced_metrics=False):

    # Initialize Kripke structure parameters
    nstates_1 = len(x1_params) - 1
    nstates_2 = len(x2_params) - 1
    n_kripke_states = nstates_1 * nstates_2 + 1 # includes out of bounds state
    oob_state_id = n_kripke_states - 1

    # Initialize Kripike structure components
    kripke_states = list(range(n_kripke_states))  # last state is out-of-bounds
    kripke_transitions = set()
    kripke_labels = {}

    # ID labeling function
    def cell_state_id(i, j):
        return i * nstates_2 + j

    # Determine cell area
    if advanced_metrics:

        # Cell area using average cell wall lengths
        dy1 = np.mean(np.diff(x1_params))
        dy2 = np.mean(np.diff(x2_params))
        cell_area = dy1 * dy2

    # Loop through each abstract state
    print("Labeling states and building transitions...")
    count = 0
    succ_count = 0
    self_loop_count = 0
    image_area = 0.0
    displacement = 0.0
    intersection_over_area = 0.0
    intersection_over_union = 0.0
    for i in range(nstates_1):
        x1_lo, x1_hi = x1_params[i], x1_params[i+1]
        for j in range(nstates_2):
            x2_lo, x2_hi = x2_params[j], x2_params[j+1]
            corners = np.array([
                [x1_lo, x2_lo],
                [x1_lo, x2_hi],
                [x1_hi, x2_hi],
                [x1_hi, x2_lo]])

            # Compute y-space image
            x_next = dynamics(corners)

            # Check label of current cell
            label = ['goal'] if is_goal_state(corners) else ['safe']

            # Allocate image area if requested (approximated as AABB of image)
            if advanced_metrics:

                # Compute AABB side lengths of parent and image
                x1_lo_pre, x1_hi_pre = float(corners[:, 0].min()), float(corners[:, 0].max())
                x2_lo_pre, x2_hi_pre = float(corners[:, 1].min()), float(corners[:, 1].max())
                x1_lo_post, x1_hi_post = float(x_next[:, 0].min()), float(x_next[:, 0].max())
                x2_lo_post, x2_hi_post = float(x_next[:, 1].min()), float(x_next[:, 1].max())

                # Compute image AABB area
                img_area = (x1_hi_post - x1_lo_post) * (x2_hi_post - x2_lo_post)
                image_area += img_area

                # Compute parent AABB area
                parent_area = (x1_hi_pre - x1_lo_pre) * (x2_hi_pre - x2_lo_pre)

                # Compute displacement between centroids of parent and image AABB
                dists = np.linalg.norm(x_next - corners, axis=1)
                displacement += np.mean(dists)

                # Compute the proportion of the parent cell contained in the image AABB
                l1 = max(0, min(x1_hi_pre, x1_hi_post) - max(x1_lo_pre, x1_lo_post))
                l2 = max(0, min(x2_hi_pre, x2_hi_post) - max(x2_lo_pre, x2_lo_post))
                int_area = (l1 * l2)
                intersection_over_area += int_area / parent_area

                # Compute the proportion of intersection area to union area
                union_area = parent_area + image_area - int_area
                intersection_over_union += int_area / union_area
            
            # Identify successor cells
            x1_min, x1_max = float(x_next[:, 0].min()), float(x_next[:, 0].max())
            x2_min, x2_max = float(x_next[:, 1].min()), float(x_next[:, 1].max())
            succ_cells = intersecting_cells_from_y_aabb(x1_params, x2_params, x1_min, x1_max, x2_min, x2_max)

            # Indicate if any successor goes out of bounds
            hits_oob = is_oob_state(x_next, (float(x1_params[0]), float(x1_params[-1])),
                                             (float(x2_params[0]), float(x2_params[-1])))
            # hits_oob = any(not point_in_parallelogram(pt, verts_y_domain) for pt in y_next)

            # Allocate relations to Kripke structure components
            src = cell_state_id(i, j)
            for (ip, jp) in succ_cells:
                dst = cell_state_id(ip, jp)
                if not allow_self_loops and dst == src:
                        continue
                edge = (src, dst)
                if edge not in kripke_transitions:
                    kripke_transitions.add(edge)
                    succ_count += 1
                    if dst == src:
                        self_loop_count += 1
            if hits_oob:
                edge = (src, oob_state_id)
                if edge not in kripke_transitions:
                    kripke_transitions.add(edge)  # transitions oob
                    succ_count += 1

            # # Force self loop if no in-grid successors and no OOB transition
            # if (not succ_cells) and (not hits_oob):
            #     kripke_transitions.add((src, src))

            kripke_labels[src] = label
            if count % 10000 == 0:
                    print(f"    > Processed {count} / {n_kripke_states - 1} states...")
            count += 1

    # Label the out-of-bounds state; add self-loop
    kripke_labels[oob_state_id] = ['fail']
    kripke_transitions.add((oob_state_id, oob_state_id))

    # Define initial states (in bounds, non-goal states)
    initial_states = [s for s in kripke_states if s != oob_state_id]

    # Output stats
    print(f"Average successors per state: {succ_count / (n_kripke_states - 1):.2f}")
    print(f"Average self-loops per state: {self_loop_count / (n_kripke_states - 1):.4f}")
    print(f"Average cell area: {cell_area:.4f}")
    print(f"Average image area: {image_area / (n_kripke_states - 1):.4f}")
    print(f"Average image area / cell area: {image_area / ((n_kripke_states - 1) * cell_area):.2f}")
    print(f"Average centroid displacement: {displacement / (n_kripke_states - 1):.4f}")
    print(f"Average IoA: {intersection_over_area / (n_kripke_states - 1):.4f}")
    print(f"Average IoU: {intersection_over_union / (n_kripke_states - 1):.4f}")

    # Make the Kripke structure
    print("Constructing Kripke structure...")
    kripke_structure = pmc.Kripke(S=kripke_states,
                                  S0=initial_states,
                                  R=list(kripke_transitions),
                                  L=kripke_labels)
    
    return kripke_structure




def make_kripke_from_params(y1_params, y2_params, M, allow_self_loops=True, advanced_metrics=False):

    # # Unpack domain parameters
    # x1_min, x1_max, x2_min, x2_max = x_domain
    # _, verts_y_domain = gpt.get_yspace_bounds(M, x1_min, x1_max, x2_min, x2_max)

    # Initialize Kripke structure parameters
    nstates_1 = len(y1_params) - 1
    nstates_2 = len(y2_params) - 1
    n_kripke_states = nstates_1 * nstates_2 + 1 # includes out of bounds state
    oob_state_id = n_kripke_states - 1

    # Initialize Kripike structure components
    kripke_states = list(range(n_kripke_states))  # last state is out-of-bounds
    kripke_transitions = set()
    kripke_labels = {}

    # ID labeling function
    def cell_state_id(i, j):
        return i * nstates_2 + j

    # Precompute transformation inverse
    M = np.asarray(M, dtype=float)
    invM = np.linalg.inv(M)

    # Determine cell area
    if advanced_metrics:

        # Cell area using average cell wall lengths
        dy1 = np.mean(np.diff(y1_params))
        dy2 = np.mean(np.diff(y2_params))
        cell_area = dy1 * dy2

    # Loop through each abstract state
    print("Labeling states and building transitions...")
    count = 0
    succ_count = 0
    self_loop_count = 0
    image_area = 0.0
    displacement = 0.0
    intersection_over_area = 0.0
    for i in range(nstates_1):
        y1_lo, y1_hi = y1_params[i], y1_params[i+1]
        for j in range(nstates_2):
            y2_lo, y2_hi = y2_params[j], y2_params[j+1]
            corners = np.array([
                [y1_lo, y2_lo],
                [y1_lo, y2_hi],
                [y1_hi, y2_hi],
                [y1_hi, y2_lo]])

            # Compute y-space image
            x_corners = (M @ corners.T).T
            x_next = dynamics(x_corners)
            y_next = (invM @ x_next.T).T

            # Check label of current cell
            label = ['goal'] if is_goal_state(x_corners) else ['safe']

            # Allocate image area if requested (approximated as AABB of image)
            if advanced_metrics:

                # Compute AABB area
                x_min_img, x_max_img = float(x_next[:, 0].min()), float(x_next[:, 0].max())
                y_min_img, y_max_img = float(x_next[:, 1].min()), float(x_next[:, 1].max())
                img_area = (x_max_img - x_min_img) * (y_max_img - y_min_img)
                image_area += img_area

                # Compute average push distance of vertices
                dists = np.linalg.norm(x_next - x_corners, axis=1)
                displacement += np.mean(dists)

                # Compute the proportion of the parent cell contained in the image AABB
                y1_min_img, y1_max_img = float(y_next[:, 0].min()), float(y_next[:, 0].max())
                y2_min_img, y2_max_img = float(y_next[:, 1].min()), float(y_next[:, 1].max())
                overlap_y1 = max(0.0, min(y1_hi, y1_max_img) - max(y1_lo, y1_min_img))
                overlap_y2 = max(0.0, min(y2_hi, y2_max_img) - max(y2_lo, y2_min_img))
                parent_area = (y1_hi - y1_lo) * (y2_hi - y2_lo)
                intersection_over_area += (overlap_y1 * overlap_y2) / parent_area if parent_area > 0.0 else 0.0
            
            # Identify successor cells
            y1_min, y1_max = float(y_next[:, 0].min()), float(y_next[:, 0].max())
            y2_min, y2_max = float(y_next[:, 1].min()), float(y_next[:, 1].max())
            succ_cells = intersecting_cells_from_y_aabb(y1_params, y2_params, y1_min, y1_max, y2_min, y2_max)

            # Indicate if any successor goes out of bounds
            hits_oob = is_oob_state(x_next, (float(y1_params[0]), float(y1_params[-1])),
                                             (float(y2_params[0]), float(y2_params[-1])))
            # hits_oob = any(not point_in_parallelogram(pt, verts_y_domain) for pt in y_next)

            # Allocate relations to Kripke structure components
            src = cell_state_id(i, j)
            for (ip, jp) in succ_cells:
                dst = cell_state_id(ip, jp)
                if not allow_self_loops and dst == src:
                        continue
                edge = (src, dst)
                if edge not in kripke_transitions:
                    kripke_transitions.add(edge)
                    succ_count += 1
                    if dst == src:
                        self_loop_count += 1
            if hits_oob:
                edge = (src, oob_state_id)
                if edge not in kripke_transitions:
                    kripke_transitions.add(edge)  # transitions oob
                    succ_count += 1

            # # Force self loop if no in-grid successors and no OOB transition
            # if (not succ_cells) and (not hits_oob):
            #     kripke_transitions.add((src, src))

            kripke_labels[src] = label
            if count % 10000 == 0:
                    print(f"    > Processed {count} / {n_kripke_states - 1} states...")
            count += 1

    # Label the out-of-bounds state; add self-loop
    kripke_labels[oob_state_id] = ['fail']
    kripke_transitions.add((oob_state_id, oob_state_id))

    # Define initial states (in bounds, non-goal states)
    initial_states = [s for s in kripke_states if s != oob_state_id]

    # Output stats
    print(f"Average successors per state: {succ_count / (n_kripke_states - 1):.2f}")
    print(f"Average self-loops per state: {self_loop_count / (n_kripke_states - 1):.4f}")
    print(f"Average cell area: {cell_area:.4f}")
    print(f"Average image area: {image_area / (n_kripke_states - 1):.4f}")
    print(f"Average image area / cell area: {image_area / ((n_kripke_states - 1) * cell_area):.2f}")
    print(f"Average centroid displacement: {displacement / (n_kripke_states - 1):.4f}")
    print(f"Average IoA: {intersection_over_area / (n_kripke_states - 1):.4f}")

    # Make the Kripke structure
    print("Constructing Kripke structure...")
    kripke_structure = pmc.Kripke(S=kripke_states,
                                  S0=initial_states,
                                  R=list(kripke_transitions),
                                  L=kripke_labels)
    
    return kripke_structure


def model_check_kripke(kripke_structure):

    start_cpu = time.process_time()
    phi = 'A (safe U goal)'  # Eventually reach goal
    sat = CTL.modelcheck(kripke_structure, phi)
    end_cpu = time.process_time()
    cpu_time = end_cpu - start_cpu
    print(f"Model checking CPU time (s): {cpu_time:.2f}")
    
    return sat


def fixed_check_ground_truth(x_domain, x_star, radius, y1_params, y2_params, M, grid_resolution=100):

    # Unpack domain parameters
    x1_min, x1_max, x2_min, x2_max = map(float, x_domain)
    x_star = np.asarray(x_star, dtype=float)

    # Initialize a uniform grid over the x-domain
    x1_vals = np.linspace(x1_min, x1_max, grid_resolution)
    x2_vals = np.linspace(x2_min, x2_max, grid_resolution)

    # Iterate through each grid cell; classify as 'fail', 'goal', or 'unk'
    fixed_check = {}
    for i in range(grid_resolution - 1):
        for j in range(grid_resolution - 1):
            x1_lo, x1_hi = x1_vals[i], x1_vals[i+1]
            x2_lo, x2_hi = x2_vals[j], x2_vals[j+1]

            # Define cell corners
            corners = np.array([
                [x1_lo, x2_lo],
                [x1_lo, x2_hi],
                [x1_hi, x2_hi],
                [x1_hi, x2_lo]])
            
            # Push vertices through dynamics until any fail or all succeed
            max_steps = 10_000
            any_oob = False
            all_goal = False
            reached_terminal = False
            for _ in range(max_steps):
                any_oob = any(
                    (v[0] < x1_min) or (v[0] > x1_max) or (v[1] < x2_min) or (v[1] > x2_max)
                    for v in corners
                )
                if any_oob:
                    fixed_check[(i, j)] = 'fail'
                    reached_terminal = True
                    break

                all_goal = all(np.linalg.norm(v - x_star) <= radius for v in corners)
                if all_goal:
                    fixed_check[(i, j)] = 'goal'
                    reached_terminal = True
                    break

                corners = dynamics(corners)

            if not reached_terminal:
                fixed_check[(i, j)] = 'unk'

    # --- Now classify the user's affine (y-grid) cells using the fixed x-grid labels ---
    nstates_1 = len(y1_params) - 1
    nstates_2 = len(y2_params) - 1

    # ID labeling function for user abstraction cells
    def cell_state_id(i, j):
        return i * nstates_2 + j

    M = np.asarray(M, dtype=float)

    # Helper: over-approx candidate x-grid cells from an x-space AABB
    def _intersecting_fixed_cells_from_x_aabb(x1_min_aabb, x1_max_aabb, x2_min_aabb, x2_max_aabb):
        n1 = grid_resolution - 1
        n2 = grid_resolution - 1

        # If completely outside x-domain, return empty and let caller decide label.
        if x1_max_aabb < x1_min or x1_min_aabb > x1_max or x2_max_aabb < x2_min or x2_min_aabb > x2_max:
            return []

        # Map AABB limits to grid-cell indices.
        i_lo = int(np.searchsorted(x1_vals, x1_min_aabb, side="right") - 1)
        i_hi = int(np.searchsorted(x1_vals, x1_max_aabb, side="right") - 1)
        j_lo = int(np.searchsorted(x2_vals, x2_min_aabb, side="right") - 1)
        j_hi = int(np.searchsorted(x2_vals, x2_max_aabb, side="right") - 1)

        i_lo = max(0, min(n1 - 1, i_lo))
        i_hi = max(0, min(n1 - 1, i_hi))
        j_lo = max(0, min(n2 - 1, j_lo))
        j_hi = max(0, min(n2 - 1, j_hi))

        if i_hi < i_lo or j_hi < j_lo:
            return []

        return [(ii, jj) for ii in range(i_lo, i_hi + 1) for jj in range(j_lo, j_hi + 1)]

    # Helper: convex polygon intersection via Separating Axis Theorem (SAT)
    def _poly_intersects_poly(poly_a, poly_b, tol=1e-12):
        poly_a = np.asarray(poly_a, dtype=float)
        poly_b = np.asarray(poly_b, dtype=float)

        def _edges(poly):
            return poly[(np.arange(len(poly)) + 1) % len(poly)] - poly

        def _axes_from_edges(edges):
            # Perpendicular normals for each edge
            return np.column_stack([-edges[:, 1], edges[:, 0]])

        def _project(poly, axis):
            axis = np.asarray(axis, dtype=float)
            norm = np.linalg.norm(axis)
            if norm < tol:
                return None
            axis = axis / norm
            dots = poly @ axis
            return float(dots.min()), float(dots.max())

        axes = np.vstack([
            _axes_from_edges(_edges(poly_a)),
            _axes_from_edges(_edges(poly_b)),
        ])

        for axis in axes:
            pa = _project(poly_a, axis)
            pb = _project(poly_b, axis)
            if pa is None or pb is None:
                continue
            a_min, a_max = pa
            b_min, b_max = pb
            if a_max < b_min - tol or b_max < a_min - tol:
                return False
        return True

    # Helper: parallelogram (x-space) intersects x-rectangle cell (axis-aligned)
    def _parallelogram_intersects_cell(par_verts, cell_i, cell_j):
        x1_lo, x1_hi = x1_vals[cell_i], x1_vals[cell_i + 1]
        x2_lo, x2_hi = x2_vals[cell_j], x2_vals[cell_j + 1]
        rect = np.array([
            [x1_lo, x2_lo],
            [x1_lo, x2_hi],
            [x1_hi, x2_hi],
            [x1_hi, x2_lo],
        ], dtype=float)
        return _poly_intersects_poly(par_verts, rect)

    ground_truth_check = {}

    for i in range(nstates_1):
        y1_lo, y1_hi = y1_params[i], y1_params[i + 1]
        for j in range(nstates_2):
            y2_lo, y2_hi = y2_params[j], y2_params[j + 1]

            # y-cell corners (axis-aligned in y-space)
            y_corners = np.array([
                [y1_lo, y2_lo],
                [y1_lo, y2_hi],
                [y1_hi, y2_hi],
                [y1_hi, y2_lo],
            ], dtype=float)

            # Map to x-space: x = M y (affine cell becomes a parallelogram in x)
            x_par = (M @ y_corners.T).T

            src = cell_state_id(i, j)

            # If any vertex is outside the x-domain, conservatively mark as fail
            if np.any(x_par[:, 0] < x1_min) or np.any(x_par[:, 0] > x1_max) or np.any(x_par[:, 1] < x2_min) or np.any(x_par[:, 1] > x2_max):
                ground_truth_check[src] = 'fail'
                continue

            # Candidate fixed cells from AABB over-approx
            x1_min_aabb, x1_max_aabb = float(x_par[:, 0].min()), float(x_par[:, 0].max())
            x2_min_aabb, x2_max_aabb = float(x_par[:, 1].min()), float(x_par[:, 1].max())
            candidates = _intersecting_fixed_cells_from_x_aabb(x1_min_aabb, x1_max_aabb, x2_min_aabb, x2_max_aabb)

            if not candidates:
                ground_truth_check[src] = 'unk'
                continue

            # Check actual intersections and aggregate labels
            seen_fail = False
            seen_unk = False
            seen_goal = False

            for (ii, jj) in candidates:
                if not _parallelogram_intersects_cell(x_par, ii, jj):
                    continue
                lab = fixed_check.get((ii, jj), 'unk')
                if lab == 'fail':
                    seen_fail = True
                    break
                if lab == 'unk':
                    seen_unk = True
                elif lab == 'goal':
                    seen_goal = True

            if seen_fail:
                ground_truth_check[src] = 'fail'
            elif seen_unk:
                ground_truth_check[src] = 'unk'
            else:
                # If it intersects only goal-labeled fixed cells, declare goal. If it
                # didn't intersect anything due to numerical edge cases, keep unk.
                ground_truth_check[src] = 'goal' if seen_goal else 'unk'

    return ground_truth_check




def check_ground_truth(x_domain, x_star, radius, y1_params, y2_params, M):

    # Unpack domain parameters
    x1_min, x1_max, x2_min, x2_max = map(float, x_domain)
    x_star = np.asarray(x_star, dtype=float)

    # Initialize ground truth dictionary
    nstates_1 = len(y1_params) - 1
    nstates_2 = len(y2_params) - 1
    ground_truth_check = {}

    # ID labeling function
    def cell_state_id(i, j):
        return i * nstates_2 + j

    # Precompute transformation matrix
    M = np.asarray(M, dtype=float)

    # Loop through each abstract state
    for i in range(nstates_1):
        y1_lo, y1_hi = y1_params[i], y1_params[i+1]
        for j in range(nstates_2):
            y2_lo, y2_hi = y2_params[j], y2_params[j+1]

            # Transform corners to x-space
            corners = np.array([
                [y1_lo, y2_lo],
                [y1_lo, y2_hi],
                [y1_hi, y2_hi],
                [y1_hi, y2_lo]])
            x_verts = (M @ corners.T).T

            # Push vertices through dynamics until any fail or all succeed
            src = cell_state_id(i, j)
            max_steps = 10_000
            any_oob = False
            all_goal = False
            reached_terminal = False
            for _ in range(max_steps):
                any_oob = any(
                    (v[0] < x1_min) or (v[0] > x1_max) or (v[1] < x2_min) or (v[1] > x2_max)
                    for v in x_verts
                )
                if any_oob:
                    ground_truth_check[src] = 'fail'
                    reached_terminal = True
                    break

                all_goal = all(np.linalg.norm(v - x_star) <= radius for v in x_verts)
                if all_goal:
                    ground_truth_check[src] = 'goal'
                    reached_terminal = True
                    break

                x_verts = dynamics(x_verts)

            if not reached_terminal:
                ground_truth_check[src] = 'unk'

            # print(f"Cell ({i},{j}): " +
            #       ("OOB reached" if any_oob else
            #        "All goal reached" if all_goal else
            #        "Neither OOB nor all goal reached"))

    return ground_truth_check

if __name__ == "__main__":

    K = pmc.Kripke(S=[0, 1, 3],
                R=[(0, 2), (2, 2), (0, 1), (1, 0), (3, 2)],
                L={1: ['p', 'q'], 2: ['p', 'q'], 3: ['q']})

    print(K.states())
    print(K.transitions())

def false_negative_rate(true_safe_states, checked_safe_states):
    false_negative_states = {s for s in true_safe_states if s not in checked_safe_states}
    denom = len(true_safe_states)
    fnr = (len(false_negative_states) / denom) if denom > 0 else float('nan')
    return fnr, false_negative_states


