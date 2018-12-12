from strike_with_a_pose.Classifier import Classifier
from strike_with_a_pose.ObjectDetector import ObjectDetector
from strike_with_a_pose.ImgCaptioning import ImgCaptioning


MODEL_TYPE = "image_captioning"


MODELS = {"classifier": Classifier, "object_detector": ObjectDetector,"image_captioning": ImgCaptioning}
MODEL = MODELS[MODEL_TYPE]

BACKGROUND_F = "background_{0}.jpg".format(MODEL_TYPE)

TEXTURE_FS = ["Deer_horns_D.tga", "Deer_body_D.tga"]

# Order matters.
OBJ_FS = ["deer_0.obj","deer_1.obj"]

INITIAL_PARAMS = {
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