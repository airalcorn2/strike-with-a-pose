import cv2
import numpy as np
import os
import pkg_resources
import urllib

YOLO_CLASSES = pkg_resources.resource_filename("strike_with_a_pose", "yolov3.txt")
YOLO_WEIGHTS = pkg_resources.resource_filename("strike_with_a_pose", "yolov3.weights")
YOLO_CONFIG = pkg_resources.resource_filename("strike_with_a_pose", "yolov3.cfg")


class Model:
    def __init__(self):
        classes = len(open(YOLO_CLASSES, "r").readlines())
        self.yolo_rgbs = np.random.uniform(0, 255, size=(classes, 3)) / 255.0

        if not os.path.isfile(YOLO_WEIGHTS):
            print("Downloading YOLOv3 weights.")
            url = "https://pjreddie.com/media/files/yolov3.weights"
            urllib.request.urlretrieve(url, YOLO_WEIGHTS)

        self.net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
        layer_names = self.net.getLayerNames()
        self.yolo_output_layers = [layer_names[i[0] - 1] for i in
                                   self.net.getUnconnectedOutLayers()]

    def detect(self, image):
        boxes = []
        class_ids = []

        bboxes = []
        confidences = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        image = np.array(image)
        width = image.shape[1]
        height = image.shape[0]
        scale = 0.00392

        # Magic number.
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True,
                                     crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.yolo_output_layers)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    orig_center_x = int(detection[0] * width)
                    orig_center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = orig_center_x - w / 2
                    y = orig_center_y - h / 2
                    bboxes.append([x, y, w, h])

                    center_x = detection[0] * 2 - 1
                    center_y = 1 - detection[1] * 2
                    half_w = detection[2]
                    half_h = detection[3]

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append(
                        np.array([[center_x - half_w, center_y + half_h],
                                  [center_x + half_w, center_y + half_h],
                                  [center_x - half_w, center_y + half_h],
                                  [center_x - half_w, center_y - half_h],
                                  [center_x + half_w, center_y + half_h],
                                  [center_x + half_w, center_y - half_h],
                                  [center_x + half_w, center_y - half_h],
                                  [center_x - half_w, center_y - half_h]]))

        return self.create_box_and_label_arrays(cv2.dnn.NMSBoxes(bboxes,
                                                                 confidences,
                                                                 conf_threshold,
                                                                 nms_threshold),
                                                class_ids, boxes)

    def create_box_and_label_arrays(self, indices, class_ids, boxes):
        box_arrays = []
        label_arrays = []
        box_rgbs = []
        for index in indices:

            idx = index[0]
            class_id = class_ids[idx]
            box = boxes[idx]
            num_rows = class_id % 20  # Magic number.
            num_cols = int(class_id / 20)
            box_x = box[0][0]
            box_y = box[0][1]

            box_array = np.zeros((8, 8))
            box_array[:, :2] = box
            box_arrays.append(box_array)

            vertices_yolo = np.array([[box_x, box_y, 0.0],
                                      [box_x, box_y - 30.0 / 299, 0.0],
                                      [box_x + 150.0 / 299, box_y, 0.0],
                                      [box_x + 150.0 / 299, box_y - 30.0 / 299,
                                       0.0],
                                      [box_x + 150.0 / 299, box_y, 0.0],
                                      [box_x, box_y - 30.0 / 299, 0.0]])

            normals = np.repeat([[0.0, 0.0, 1.0]], len(vertices_yolo), axis=0)

            # Magic numbers.
            yolo_coords = np.array(
                [[0.25 * num_cols, 1.0 - num_rows * (48.96 / 1024)],
                 [0.25 * num_cols, 1.0 - (num_rows + 1) * (48.96 / 1024) + 1.0 / 1024],
                 [0.25 * (num_cols + 1), 1.0 - num_rows * (48.96 / 1024)],
                 [0.25 * (num_cols + 1), 1.0 - (num_rows + 1) * (48.96 / 1024) + 1.0 / 1024],
                 [0.25 * (num_cols + 1), 1.0 - num_rows * (48.96 / 1024)],
                 [0.25 * num_cols, 1.0 - (num_rows + 1) * (48.96 / 1024) + 1.0 / 1024]])
            label_array = np.hstack((vertices_yolo, normals, yolo_coords))
            label_arrays.append(label_array)

            box_rgbs.append(self.yolo_rgbs[class_id])

        return (box_arrays, label_arrays, box_rgbs)