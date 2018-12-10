from strike_with_a_pose.Classifier import Classifier
# from strike_with_a_pose.ObjectDetector import ObjectDetector
from PyQt5 import QtCore
from PyQt5.QtWidgets import QLabel, QPushButton

MODEL = Classifier
BACKGROUND_F = "background_classifier.jpg"
# BACKGROUND_F = "background_object_detector.jpg"
TEXTURE_FS = ["interior.tga", "exterior.tga", "glass.tga"]
YOLO_CLASSES_F = "yolo_classes.png"
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


def GET_GUI_COMPS():
    # Prediction text.
    PRED_TEXT = QLabel("<strong>Top Label</strong>: <br>"
                       "<strong>Top Probability</strong>: <br><br>"
                       "<strong>True Label</strong>: <br>"
                       "<strong>True Probability</strong>: ")
    PRED_TEXT.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
    # Predict button.
    PREDICT = QPushButton("Predict")
    # Detect button.
    # DETECT = QPushButton("Detect")
    # Order matters. Prediction button must be named "predict" in tuple.
    GUI_COMPS = [("pred_text", PRED_TEXT), ("predict", PREDICT)]
    # GUI_COMPS = [("predict", DETECT)]
    return GUI_COMPS