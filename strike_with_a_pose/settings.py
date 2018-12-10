from PyQt5 import QtCore
from PyQt5.QtWidgets import QLabel, QPushButton
from strike_with_a_pose.Classifier import Classifier
from strike_with_a_pose.ObjectDetector import ObjectDetector


def get_classifier_gui_comps():
    # Prediction text.
    PRED_TEXT = QLabel("<strong>Top Label</strong>: <br>"
                       "<strong>Top Probability</strong>: <br><br>"
                       "<strong>True Label</strong>: <br>"
                       "<strong>True Probability</strong>: ")
    PRED_TEXT.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
    # Predict button.
    PREDICT = QPushButton("Predict")
    # Order matters. Prediction button must be named "predict" in tuple.
    return [("pred_text", PRED_TEXT), ("predict", PREDICT)]


def get_object_detector_gui_comps():
    # Detect button.
    DETECT = QPushButton("Detect")
    # Order matters. Prediction button must be named "predict" in tuple.
    return [("predict", DETECT)]


def GET_GUI_COMPS():
    return GUI_COMP_FUNCS[MODEL_TYPE]()


MODEL_TYPE = "object_detector"

MODELS = {"classifier": Classifier, "object_detector": ObjectDetector}
MODEL = MODELS[MODEL_TYPE]

BACKGROUND_F = "background_{0}.jpg".format(MODEL_TYPE)

TEXTURE_FS = ["interior.tga", "exterior.tga", "glass.tga"]

# Order matters.
OBJ_FS = ["interior.obj", "exterior.obj", "glass.obj"]

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

GUI_COMP_FUNCS = {"classifier": get_classifier_gui_comps,
                  "object_detector": get_object_detector_gui_comps}