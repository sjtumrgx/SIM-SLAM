import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

from Robocon2026.utils.utils import euler2quaternion

JETBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"assets/Jetbot/jetbot.usd",
    ),
    actuators={
        "wheel_acts": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            damping=None,
            stiffness=None,
        )
    },
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(5.45, 2.925, 0.0),
        rot=euler2quaternion([-85, 0, 0]),
    ),
)
