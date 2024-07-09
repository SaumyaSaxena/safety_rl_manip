import numpy as np

def signed_dist_fn_rectangle(grid_x, x_target_min, x_target_max, obstacle=False, plot=False):
    # Compute distances to each edge of the rectangle
    dist_from_walls = np.maximum(x_target_min - grid_x, grid_x - x_target_max)
    signed_distance_grid = np.max(dist_from_walls, axis=-1)
    if obstacle:
        signed_distance_grid = -1*signed_distance_grid

    return signed_distance_grid

def create_grid(x_min, x_max, N_x):
    X = [np.linspace(x_min[i], x_max[i], N_x[i]) for i in range(len(x_min))]
    grid = np.meshgrid(*X, indexing='ij')
    return np.stack(grid, axis=-1)