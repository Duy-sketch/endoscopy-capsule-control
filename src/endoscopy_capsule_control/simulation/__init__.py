"""
Simulation utilities for the DEMA-MCE system.
"""

from .mujoco_system import (
    MujocoSystem,
    Pose,
    default_model_path,
)

from .dema_geometry import (
    DEMAGeometry,
    ElectromagnetGeometry,
    get_dema_geometry,
    lcs_to_world_position,
    lcs_to_world_vector,
    world_to_lcs_position,
    world_to_lcs_vector,
)

from .capsule_state import (
    CapsuleState,
    get_capsule_state,
)

from .capsule_wrench import (
    CapsuleExternalWrench,
    apply_capsule_external_wrench,
    compute_capsule_external_wrench,
)

from .initialization import (
    AUBO_ACTUATOR_NAMES,
    AUBO_JOINT_NAMES,
    AUBO_WORK_Q_DEG,
    AUBO_WORK_Q_RAD,
    initialize_aubo_work_pose,
    initialize_hover_operating_point,
    initialize_mce_hover_pose,
    set_aubo_configuration,
    set_free_joint_pose,
)

from .perturbation import (
    apply_initial_perturbation,
    rotation_matrix_to_quaternion_wxyz,
    rotation_y,
)

from .diagnostics import (
    PostSettlingDiagnostics,
    PostSettlingMetrics,
)

from .viewer import (
    configure_capsule_tracking_camera,
    configure_dema_overview_camera,
)

from .hover_simulation import (
    HoverSimulation,
    HoverSimulationResult,
)


__all__ = [
    "MujocoSystem",
    "Pose",
    "default_model_path",

    "DEMAGeometry",
    "ElectromagnetGeometry",
    "get_dema_geometry",
    "lcs_to_world_position",
    "lcs_to_world_vector",
    "world_to_lcs_position",
    "world_to_lcs_vector",

    "CapsuleState",
    "get_capsule_state",

    "CapsuleExternalWrench",
    "apply_capsule_external_wrench",
    "compute_capsule_external_wrench",

    "AUBO_ACTUATOR_NAMES",
    "AUBO_JOINT_NAMES",
    "AUBO_WORK_Q_DEG",
    "AUBO_WORK_Q_RAD",
    "initialize_aubo_work_pose",
    "initialize_hover_operating_point",
    "initialize_mce_hover_pose",
    "set_aubo_configuration",
    "set_free_joint_pose",

    "apply_initial_perturbation",
    "rotation_matrix_to_quaternion_wxyz",
    "rotation_y",

    "PostSettlingDiagnostics",
    "PostSettlingMetrics",

    "configure_capsule_tracking_camera",
    "configure_dema_overview_camera",

    "HoverSimulation",
    "HoverSimulationResult",
]