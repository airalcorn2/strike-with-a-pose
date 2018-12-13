from strike_with_a_pose.ImageClassifier import ImageClassifier
from strike_with_a_pose.ObjectDetector import ObjectDetector
from strike_with_a_pose.ImageCaptioner import ImageCaptioner

MODELS = {"classifier": ImageClassifier, "detector": ObjectDetector,
          "captioner": ImageCaptioner}
# Order matters.
MODEL_OBJ_AND_TEXTURE_FS = {
    "classifier": [("interior.obj", "interior.tga"),
                   ("exterior.obj", "exterior.tga"),
                   ("glass.obj", "glass.tga")],
    "detector": [("interior.obj", "interior.tga"),
                 ("exterior.obj", "exterior.tga"),
                 ("glass.obj", "glass.tga")],
    "captioner": [("bird.obj", "bird.tga")]
}
MODEL_INITIAL_PARAMS = {
    "classifier": {
        "x_delta": -0.3005,
        "y_delta": -0.2227,
        "z_delta": -9.6000,
        "yaw": 178.5066,
        "pitch": -4.6715,
        "roll": 12.8242,
        "amb_int": 0.7000,
        "dir_int": 0.7000,
        "DirLight": (0.0000, 1.0000, 0.000),
        "USE_BACKGROUND": False
    },
    "detector": {
        "x_delta": -0.4260,
        "y_delta": -0.1446,
        "z_delta": -12.9783,
        "yaw": -159.4024,
        "pitch": -3.9317,
        "roll": -1.2106,
        "amb_int": 0.7000,
        "dir_int": 0.7000,
        "DirLight": (0.0000, 1.0000, 0.000),
        "USE_BACKGROUND": True
    },
    "captioner": {
        "x_delta": -0.2401,
        "y_delta": 0.4551,
        "z_delta": -12.7916,
        "yaw": 100.6933,
        "pitch": -7.4264,
        "roll": 7.3828,
        "amb_int": 0.7000,
        "dir_int": 0.7000,
        "DirLight": (0.0000, 1.0000, 0.000),
        "USE_BACKGROUND": True
    }
}

MODEL_TYPE = "classifier"
MODEL = MODELS[MODEL_TYPE]
BACKGROUND_F = "background_{0}.jpg".format(MODEL_TYPE)
OBJ_AND_TEXTURE_FS = MODEL_OBJ_AND_TEXTURE_FS[MODEL_TYPE]
INITIAL_PARAMS = MODEL_INITIAL_PARAMS[MODEL_TYPE]
