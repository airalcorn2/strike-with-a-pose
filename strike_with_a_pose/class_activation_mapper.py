import cv2
import moderngl
import numpy as np
import pkg_resources
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from PyQt5 import QtCore
from PyQt5.QtWidgets import QLabel, QPushButton

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGENET_F = pkg_resources.resource_filename(
    "strike_with_a_pose", "data/imagenet_classes.txt"
)
SCENE_DIR = pkg_resources.resource_filename("strike_with_a_pose", "scene_files/")
TRUE_CLASS = 609


class ClassActivationMapper(nn.Module):
    def __init__(self):
        super(ClassActivationMapper, self).__init__()
        self.net = models.inception_v3(pretrained=True)
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False

        self.preprocess = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.label_map = self.load_imagenet_label_map()
        self.true_class = TRUE_CLASS
        self.true_label = self.label_map[self.true_class]

        feature_params = list(self.net.parameters())[-2]
        self.weight_softmax = np.squeeze(feature_params.data.numpy())
        # Get features from Mixed_7c layer.
        self.features = torch.zeros((1, 2048, 8, 8))

        def copy_data(module, input, output):
            self.features.data.copy_(output.data)

        self.net._modules.get("Mixed_7c").register_forward_hook(copy_data)
        (bn, feat_nc, feat_h, feat_w) = self.features.shape
        self.feat_shape = (feat_nc, feat_h * feat_w)
        self.cam_shape = (feat_h, feat_w)

        self.out_shape = (299, 299)

    def load_imagenet_label_map(self):
        input_f = open(IMAGENET_F)
        label_map = {}
        for line in input_f:
            parts = line.strip().split(": ")
            (num, label) = (int(parts[0]), parts[1].replace('"', ""))
            label_map[num] = label

        input_f.close()
        return label_map

    @staticmethod
    def get_gui_comps():
        # Prediction text.
        pred_text = QLabel(
            "<strong>Top Label</strong>: <br>"
            "<strong>Top Probability</strong>: <br><br>"
            "<strong>True Label</strong>: <br>"
            "<strong>True Probability</strong>: "
        )
        pred_text.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)

        # Class activation map button.
        activation_map = QPushButton("Map Class Activations")
        # Order matters. Prediction button must be named "predict" in tuple.
        return [("pred_text", pred_text), ("predict", activation_map)]

    def init_scene_comps(self):
        self.CAM_VAO = None
        self.CAM_VBO = None
        self.CAM_MAP = None

    def forward(self, input_image):
        image_normalized = self.preprocess(input_image)
        out = self.net(image_normalized[None, :, :, :])
        return out

    def get_CAM(self, class_idx):
        cam = self.weight_softmax[class_idx].dot(self.features.reshape(self.feat_shape))
        cam = cam.reshape(*self.cam_shape)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam = cv2.resize(cam_img, self.out_shape)
        return output_cam

    def predict(self, image):
        with torch.no_grad():
            image_tensor = torch.Tensor(np.array(image) / 255.0).to(DEVICE)
            input_image = image_tensor.permute(2, 0, 1)
            out = self.forward(input_image)
            probs = torch.nn.functional.softmax(out, dim=1)
            probs_np = probs[0].detach().cpu().numpy()
            top_label = self.label_map[probs_np.argmax()]
            top_prob = probs_np.max()
            true_label = self.true_label
            true_prob = probs_np[self.true_class]
            self.pred_text.setText(
                "<strong>Top Label</strong>: {0}<br>"
                "<strong>Top Label Probability</strong>: {1:.4f}<br><br>"
                "<strong>True Label</strong>: {2}<br>"
                "<strong>True Label Probability</strong>: {3:.4f}".format(
                    top_label, top_prob, true_label, true_prob
                )
            )

            CAM = self.get_CAM(probs_np.argmax())
            heatmap = cv2.applyColorMap(
                cv2.resize(CAM, self.out_shape), cv2.COLORMAP_JET
            )
            cam_array = heatmap * 0.3 + np.array(image) * 0.5
            (cam_array[:, :, 0], cam_array[:, :, 2]) = (
                cam_array[:, :, 2],
                cam_array[:, :, 0].copy(),
            )
            cam_img = (
                Image.fromarray(np.uint8(cam_array))
                .transpose(Image.FLIP_TOP_BOTTOM)
                .convert("RGBA")
            )
            self.CAM_MAP = self.CTX.texture(cam_img.size, 4, cam_img.tobytes())
            self.CAM_MAP.build_mipmaps()
            vertices = np.array(
                [
                    [-1.0, -1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ]
            )
            normals = np.repeat([[0.0, 0.0, 1.0]], len(vertices), axis=0)
            texture_coords = np.array(
                [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
            )
            CAM_ARRAY = np.hstack((vertices, normals, texture_coords))
            self.CAM_VBO = self.CTX.buffer(CAM_ARRAY.flatten().astype("f4").tobytes())
            self.CAM_VAO = self.CTX.simple_vertex_array(
                self.PROG, self.CAM_VBO, "in_vert", "in_norm", "in_text"
            )

    def render(self):
        if self.CAM_MAP is not None:
            self.CTX.disable(moderngl.DEPTH_TEST)
            self.PROG["mode"].value = 1
            self.CAM_MAP.use()
            self.CAM_VAO.render()
            self.CTX.enable(moderngl.DEPTH_TEST)
            self.PROG["mode"].value = 0

    def clear(self):
        if self.CAM_MAP is not None:
            self.CAM_MAP.release()
            self.CAM_VAO.release()
            self.CAM_VBO.release()

            self.CAM_MAP = None
            self.CAM_VAO = None
            self.CAM_VBO = None
