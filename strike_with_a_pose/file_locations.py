import pkg_resources

IMAGENET_F = pkg_resources.resource_filename(
    "strike_with_a_pose", "data/imagenet_classes.txt"
)
SCENE_DIR = pkg_resources.resource_filename("strike_with_a_pose", "scene_files/")

YOLO_CLASSES = pkg_resources.resource_filename("strike_with_a_pose", "data/yolov3.txt")
YOLO_WEIGHTS = pkg_resources.resource_filename(
    "strike_with_a_pose", "data/yolov3.weights"
)
YOLO_CONFIG = pkg_resources.resource_filename("strike_with_a_pose", "data/yolov3.cfg")
YOLO_CLASSES_F = "yolo_classes.png"
