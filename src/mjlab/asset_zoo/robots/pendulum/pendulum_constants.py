from pathlib import Path
import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg
from mjlab.utils.spec_config import ActuatorCfg
from mjlab.utils.actuator import ElectricActuator

PENDULUM_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "pendulum" / "xmls" / "pendulum.xml"
)
assert PENDULUM_XML.exists(), f"XML not found: {PENDULUM_XML}"

def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(PENDULUM_XML))

PENDULUM_ACTUATOR_PARAM = ElectricActuator(
  reflected_inertia=0.0,
  velocity_limit=100.0,
  effort_limit=10.0,
)

PENDULUM_ACTUATOR = ActuatorCfg(
  joint_names_expr=["pivot"],
  effort_limit=PENDULUM_ACTUATOR_PARAM.effort_limit,
  stiffness=10.0,
  damping=0.1,
)

# PENDULUM_ARTICULATION = EntityArticulationInfoCfg(
#   actuators=(PENDULUM_ACTUATOR,), # use XML defined actuators
# )

PENDULUM_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(PENDULUM_ACTUATOR, ), # use XML defined actuators
)

PENDULUM_ROBOT_CFG = EntityCfg(
  spec_fn=get_spec,
  articulation=PENDULUM_ARTICULATION,
  init_state=EntityCfg.InitialStateCfg(
        pos=(0, 0.0, 1.5),  # <- I want to set initial position
        rot=(0, 0, 0, 1),  # <- and rotation
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
),
)

if __name__ == "__main__":
  import mujoco.viewer as viewer
  robot = Entity(PENDULUM_ROBOT_CFG)
  viewer.launch(robot.spec.compile())