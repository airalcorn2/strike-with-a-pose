import numpy as np

STEP_SIZES = {
    "x": 0.001,
    "y": 0.001,
    "z": 0.001,
    "yaw_obj_x": 0.001,
    "yaw_obj_y": 0.001,
    "pitch_obj_x": 0.001,
    "pitch_obj_y": 0.001,
    "roll_obj_x": 0.001,
    "roll_obj_y": 0.001,
    "yaw_light_x": 0.001,
    "yaw_light_y": 0.001,
    "pitch_light_x": 0.001,
    "pitch_light_y": 0.001,
    "roll_light_x": 0.001,
    "roll_light_y": 0.001,
}
LRS = {
    "x": 0.001,
    "y": 0.001,
    "z": 0.001,
    "yaw_obj_x": 0.001,
    "yaw_obj_y": 0.001,
    "pitch_obj_x": 0.001,
    "pitch_obj_y": 0.001,
    "roll_obj_x": 0.001,
    "roll_obj_y": 0.001,
    "yaw_light_x": 0.01,
    "yaw_light_y": 0.01,
    "pitch_light_x": 0.01,
    "pitch_light_y": 0.01,
    "roll_light_x": 0.01,
    "roll_light_y": 0.01,
}
BUMP = 0.1
CAMERA_DISTANCE = 2.0
TOO_CLOSE = 0.0
TOO_FAR = CAMERA_DISTANCE - 30.0
MIN_Z = min(TOO_CLOSE, TOO_FAR)
MAX_Z = max(TOO_CLOSE, TOO_FAR)
ANGLE_OF_VIEW = 16.426
TAN_ANGLE = np.tan(ANGLE_OF_VIEW / 2 * np.pi / 180.0)
TRANS_PARAMS = {"x", "y", "z"}
ROT_AXES = {"yaw", "pitch", "roll"}
ROTS = {"obj", "light"}
UPDATE_PARAMS = list(TRANS_PARAMS)
for ROT_AXIS in ROT_AXES:
    UPDATE_PARAMS += [ROT_AXIS + "_obj_x", ROT_AXIS + "_obj_y"]

(OBJ_PATH, MTL_PATH) = ("objects/Jeep/Jeep.obj", "objects/Jeep/Jeep.mtl")
(TRUE_CLASS, TARGET_CLASS) = (609, 635)
OPTIM = "fd"  # "fd", "zrs", or "rs"
GIF_F = "test.gif"
INITIAL_PARAMS = {
    "x": 0,
    "y": 0,
    "z": -4,
    "yaw_obj": np.pi / 4,
    "pitch_obj": 0,
    "roll_obj": 0,
    "yaw_light": 0,
    "pitch_light": 0,
    "roll_light": 0,
}
