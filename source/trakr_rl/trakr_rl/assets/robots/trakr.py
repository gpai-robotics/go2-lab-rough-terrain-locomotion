# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

TRAKR_USD_PATH = os.path.join(os.path.dirname(__file__), "usd", "trakr_imu.usd")


@configclass
class TrakrArticulationCfg(ArticulationCfg):
    """Configuration for Trakr articulations."""

    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.9


#Rigid body properties were taken from the trakr_legged_rl repository
#Articulation Properties were kept the same as unitree_go2
@configclass
class TrakrUsdFileCfg(sim_utils.UsdFileCfg):
    activate_contact_sensors = True
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.1,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    )

TRAKR_CFG = TrakrArticulationCfg(
    # spawn=TrakrUrdfFileCfg(
    #     asset_path=f"{UNITREE_ROS_DIR}/robots/go2_description/urdf/go2_description.urdf",
    # ),
    spawn=TrakrUsdFileCfg(
        usd_path=TRAKR_USD_PATH,
    ),
    #Joint Positions and Default Standing Positionswere taken from the trakr_legged_rl repository
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.255),
        joint_pos={
            ".*_adduction": 0.0,
            ".*F_hip": 0.0,
            ".*B_hip": 0.0,
            "L[F,B]_knee": 0.0,
            "R[F,B]_knee": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    #Actuator parameters were taken from the trakr_legged_rl repository
    #Joint Names were taken from the trakr_imu.usd file by inspecting the Stage in IsaacSim
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_adduction", ".*_hip", ".*_knee"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=20.0,
            damping=1.0,
            friction=1.0,
        ),
    },
    #Joint Names were taken from the trakr_imu.usd file by inspecting the Stage in IsaacSim
        joint_sdk_names=[
        "LB_adduction", "LB_hip", "LB_knee",
        "LF_adduction", "LF_hip", "LF_knee",
        "RB_adduction", "RB_hip", "RB_knee",
        "RF_adduction", "RF_hip", "RF_knee",
    ],
)
