from strike_with_a_pose.Classifier import Classifier
from strike_with_a_pose.ObjectDetector import ObjectDetector
from strike_with_a_pose.ImgCaptioner import ImgCaptioner


MODELS = {"classifier": Classifier, "object_detector": ObjectDetector,"image_captioner": ImgCaptioner}
# Order matters.
MODEL_OBJ_AND_TEXTURE_FS = {
    "classifier": [("interior.obj", "interior.tga"),
                   ("exterior.obj", "exterior.tga"),
                   ("glass.obj", "glass.tga")],
    "object_detector": [("interior.obj", "interior.tga"),
                        ("exterior.obj", "exterior.tga"),
                        ("glass.obj", "glass.tga")],
    "image_captioner": [("bird.obj", "bird.tga")]
}
MODEL_INITIAL_PARAMS = {
    "classifier": {
        "x_delta": -0.3865,
        "y_delta": 0.6952,
        "z_delta": -9.6000,
        "yaw": -138.0867,
        "pitch": -3.8813,
        "roll": -2.8028,
        "amb_int": 0.7000,
        "dir_int": 0.7000,
        "DirLight": (0.0000, 1.0000, 0.000)
    },
    "object_detector": {
        "x_delta": -0.3865,
        "y_delta": 0.6952,
        "z_delta": -9.6000,
        "yaw": -138.0867,
        "pitch": -3.8813,
        "roll": -2.8028,
        "amb_int": 0.7000,
        "dir_int": 0.7000,
        "DirLight": (0.0000, 1.0000, 0.000)
    },
    "image_captioner": {
        "x_delta": -0.3865,
        "y_delta": 0.6952,
        "z_delta": -9.6000,
        "yaw": -138.0867,
        "pitch": -3.8813,
        "roll": -2.8028,
        "amb_int": 0.7000,
        "dir_int": 0.7000,
        "DirLight": (0.0000, 1.0000, 0.000)
    }
}

MODEL_TYPE = "image_captioner"
MODEL = MODELS[MODEL_TYPE]
BACKGROUND_F = "background_{0}.jpg".format(MODEL_TYPE)
OBJ_AND_TEXTURE_FS = MODEL_OBJ_AND_TEXTURE_FS[MODEL_TYPE]
INITIAL_PARAMS = MODEL_INITIAL_PARAMS[MODEL_TYPE]
