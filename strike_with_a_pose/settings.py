MODEL_TYPE = "object detector"
USE_INCEPTION = True
BACKGROUND_F = "background_object_detector.jpg"
TEXTURE_FS = ["interior.tga", "exterior.tga", "glass.tga"]
YOLO_CLASSES_F = "yolo_classes.png"
# Order matters.
OBJ_FS = ["interior.obj", "exterior.obj", "glass.obj"]
CLASS_F = "obj.cls"
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