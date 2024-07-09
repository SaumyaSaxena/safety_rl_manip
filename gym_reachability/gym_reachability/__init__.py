"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
         Vicenc Rubies Royo ( vrubies@berkeley.edu )
"""

from gym.envs.registration import register

register(
    id="multi_player_lunar_lander_reachability-v0", entry_point=(
        "gym_reachability.gym_reachability.envs:"
        + "MultiPlayerLunarLanderReachability"
    )
)

register(
    id="one_player_reach_avoid_lunar_lander-v0", entry_point=(
        "gym_reachability.gym_reachability.envs:"
        + "OnePlayerReachAvoidLunarLander"
    )
)

register(
    id="dubins_car-v1",
    entry_point="gym_reachability.gym_reachability.envs:DubinsCarOneEnv"
)

register(
    id="dubins_car_pe-v0",
    entry_point="gym_reachability.gym_reachability.envs:DubinsCarPEEnv"
)

register(
    id="point_mass-v0",
    entry_point="gym_reachability.gym_reachability.envs:PointMassEnv"
)

register(
    id="zermelo_show-v0",
    entry_point="gym_reachability.gym_reachability.envs:ZermeloShowEnv"
)

register(
    id="pickup_1D_env-v0",
    entry_point="gym_reachability.gym_reachability.envs:Pickup1DEnv"
)

register(
    id="point_mass_1D_cont_env-v0",
    entry_point="gym_reachability.gym_reachability.envs:PointMass1DContEnv"
)
