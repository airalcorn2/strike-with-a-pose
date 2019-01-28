from strike_with_a_pose.image_classifier import ImageClassifier
from strike_with_a_pose.object_detector import ObjectDetector
from strike_with_a_pose.image_captioner import ImageCaptioner
from strike_with_a_pose.class_activation_mapper import ClassActivationMapper

MODELS = {
    "classifier": ImageClassifier,
    "detector": ObjectDetector,
    "captioner": ImageCaptioner,
    "mapper": ClassActivationMapper,
}
# Order matters.
MODEL_OBJ_AND_MTL_FS = {
    "classifier": ("Jeep.obj", "Jeep.mtl"),
    "detector": ("Jeep.obj", "Jeep.mtl"),
    "captioner": ("bird.obj", "bird.mtl"),
    "mapper": ("Jeep.obj", "Jeep.mtl"),
}
MODEL_INITIAL_PARAMS = {
    "classifier": {
        "x_delta": -0.1090,
        "y_delta": -0.2109,
        "z_delta": -3.8000,
        "yaw": 178.5066,
        "pitch": -4.6715,
        "roll": 12.8242,
        "amb_int": 0.0,
        "dif_int": 1.0,
        "DirLight": (0.0000, 0.7071, 0.7071),
        "USE_BACKGROUND": True,
    },
    "detector": {
        "x_delta": 0.3552,
        "y_delta": -0.2539,
        "z_delta": -6.3783,
        "yaw": -159.4024,
        "pitch": -3.9317,
        "roll": -1.2106,
        "amb_int": 0.7000,
        "dif_int": 0.7000,
        "DirLight": (0.0000, 1.0000, 0.000),
        "USE_BACKGROUND": True,
    },
    "captioner": {
        "x_delta": -0.2401,
        "y_delta": 0.4551,
        "z_delta": -12.7916,
        "yaw": 100.6933,
        "pitch": -7.4264,
        "roll": 7.3828,
        "amb_int": 0.7000,
        "dif_int": 0.7000,
        "DirLight": (0.0000, 1.0000, 0.000),
        "USE_BACKGROUND": True,
    },
    "mapper": {
        "x_delta": -0.1090,
        "y_delta": -0.2109,
        "z_delta": -3.8000,
        "yaw": 178.5066,
        "pitch": -4.6715,
        "roll": 12.8242,
        "amb_int": 0.0,
        "dif_int": 1.0,
        "DirLight": (0.0000, 0.7071, 0.7071),
        "USE_BACKGROUND": True,
    },
}

MODEL_TYPE = "classifier"
MODEL = MODELS[MODEL_TYPE]
BACKGROUND_F = "background_{0}.jpg".format(MODEL_TYPE)
(OBJ_F, MTL_F) = MODEL_OBJ_AND_MTL_FS[MODEL_TYPE]
INITIAL_PARAMS = MODEL_INITIAL_PARAMS[MODEL_TYPE]
