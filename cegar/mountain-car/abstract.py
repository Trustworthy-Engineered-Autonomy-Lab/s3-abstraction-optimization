# Libraries
from plant import dynamics
import numpy as np
from abstraction import AdaptiveGrid2D


# Instantiate adaptive grid
state_bounds = [(-10, 10), (-10, 10)]
grid = AdaptiveGrid2D(state_bounds, shape=(2, 2))
x_star = np.array([5.0, 5.0])

# Cut first tile into equal quadrants
tile_id = (0, 0)
xmin_xmax, ymin_ymax = grid.tile(tile_id).bounds.as_tuples()
x_split = 0.5 * (xmin_xmax[0] + xmin_xmax[1])
y_split = 0.5 * (ymin_ymax[0] + ymin_ymax[1])
grid.split(tile_id, x_split=x_split, y_split=y_split)
# for leaf in grid._tiles[tile_id].iter_leaves():
#     print("Leaf", leaf.id, leaf.bounds.as_tuples())

def f(x: np.ndarray) -> np.ndarray:
    return dynamics(x, x_star)


# Iterate over the *current partition* (base tiles + leaves)
for cell in grid.iter_cells():
    print("cell", cell.id, cell.bounds.as_tuples())
    image_poly = grid.image_polygon(cell, f)
    post_cells = grid.cells_intersecting_polygon(image_poly, exact=True)
    print("  post:", [p.id for p in post_cells])


# Or compute the full abstract transition relation
rel = grid.transition_relation(f, exact=True)
print("transition relation:")
for src, dsts in rel.items():
    print(" ", src, "->", sorted(dsts))
    






# # print("    future corners:", future)
#     if tile.is_split:
#         for leaf in tile.iter_leaves():
#             print("  leaf", leaf.id, leaf.bounds.as_tuples())





