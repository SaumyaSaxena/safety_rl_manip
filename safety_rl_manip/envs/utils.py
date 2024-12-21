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

def create_centered_polygon_with_halfsize(size_x: float, size_y: float):
    return np.array([
        [-size_x, -size_y],
        [-size_x, size_y],
        [size_x, size_y],
        [size_x, -size_y],
    ])

import mujoco
from robosuite.models.base import MujocoModel
def get_contacts(env, model):
    """
    Checks for any contacts with @model (as defined by @model's contact_geoms) and returns the set of
    geom names currently in contact with that model (excluding the geoms that are part of the model itself).
    Args:
        sim (MjSim): Current simulation model
        model (MujocoModel): Model to check contacts for.
    Returns:
        set: Unique geoms that are actively in contact with this model.
    Raises:
        AssertionError: [Invalid input type]
    """
    # Make sure model is MujocoModel type
    assert isinstance(model, MujocoModel), "Inputted model must be of type MujocoModel; got type {} instead!".format(
        type(model)
    )
    contact_set = set()
    for contact in env.data.contact[: env.data.ncon]:
        # check contact geom in geoms; add to contact set if match is found
        # g1, g2 = env.model.geom_id2name(contact.geom1), env.model.geom_id2name(contact.geom2)

        g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

        if g1 in model.contact_geoms and g2 not in model.contact_geoms:
            contact_set.add(g2)
        elif g2 in model.contact_geoms and g1 not in model.contact_geoms:
            contact_set.add(g1)
    return contact_set

def check_contact(env, geoms_1, geoms_2=None):
    """
    Finds contact between two geom groups.
    Args:
        geoms_1 (str or list of str or MujocoModel): an individual geom name or list of geom names or a model. If
            a MujocoModel is specified, the geoms checked will be its contact_geoms
        geoms_2 (str or list of str or MujocoModel or None): another individual geom name or list of geom names.
            If a MujocoModel is specified, the geoms checked will be its contact_geoms. If None, will check
            any collision with @geoms_1 to any other geom in the environment
    Returns:
        bool: True if any geom in @geoms_1 is in contact with any geom in @geoms_2.
    """
    # Check if either geoms_1 or geoms_2 is a string, convert to list if so
    if type(geoms_1) is str:
        geoms_1 = [geoms_1]
    elif isinstance(geoms_1, MujocoModel):
        geoms_1 = geoms_1.contact_geoms
    if type(geoms_2) is str:
        geoms_2 = [geoms_2]
    elif isinstance(geoms_2, MujocoModel):
        geoms_2 = geoms_2.contact_geoms
    for i in range(env.data.ncon):
        contact = env.data.contact[i]
        # check contact geom in geoms
        c1_in_g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) in geoms_1
        c2_in_g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) in geoms_2 if geoms_2 is not None else True
        # check contact geom in geoms (flipped)
        c2_in_g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) in geoms_1
        c1_in_g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) in geoms_2 if geoms_2 is not None else True
        if (c1_in_g1 and c2_in_g2) or (c1_in_g2 and c2_in_g1):
            return True
    return False