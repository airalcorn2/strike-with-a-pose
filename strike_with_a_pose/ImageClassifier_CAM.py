import numpy as np
import pkg_resources
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import moderngl
import cv2
from PIL import Image

from PyQt5 import QtCore
from PyQt5.QtWidgets import QLabel, QPushButton

USE_INCEPTION = True
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGENET_F = pkg_resources.resource_filename("strike_with_a_pose", "data/imagenet_classes.txt")
SCENE_DIR = pkg_resources.resource_filename("strike_with_a_pose", "scene_files/")
CAM_F = pkg_resources.resource_filename("strike_with_a_pose", "scene_files/cam.jpg")

TRUE_CLASS = 609


class ImageClassifier_CAM(nn.Module):
    def __init__(self):
        super(ImageClassifier_CAM, self).__init__()
        if USE_INCEPTION:
            self.net = models.inception_v3(pretrained=True)
        else:
            self.net = models.alexnet(pretrained=True)

        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False

        self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])

        self.label_map = self.load_imagenet_label_map()

        self.true_class = TRUE_CLASS
        self.true_label = self.label_map[self.true_class]

        self.finalconv_name = 'Mixed_7c'
        self.features_blobs = []
        self.CAM_VAO = []
        self.CAM_VBO = []
        self.OPEN_CAM = 0



    def load_imagenet_label_map(self):
        input_f = open(IMAGENET_F)
        label_map = {}
        for line in input_f:
            parts = line.strip().split(": ")
            (num, label) = (int(parts[0]), parts[1].replace("\"", ""))
            label_map[num] = label

        input_f.close()
        return label_map

    @staticmethod
    def get_gui_comps():

        # Prediction text.
        pred_text = QLabel("<strong>Top Label</strong>: <br>"
                           "<strong>Top Probability</strong>: <br><br>"
                           "<strong>True Label</strong>: <br>"
                           "<strong>True Probability</strong>: ")
        pred_text.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)

        # Classify button.

        classify = QPushButton("Classify with CAM")



        # Order matters. Prediction button must be named "predict" in tuple.
        return [("pred_text", pred_text), ("predict", classify)]

    def init_scene_comps(self):
        pass

    def forward(self, input_image):
        image_normalized = self.preprocess(input_image)
        out = self.net(image_normalized[None, :, :, :])
        return out

    def get_weight_softmax(self, net):
        # hook the feature extractor
        features_blobs = []
        finalconv_name = self.finalconv_name

        def hook_feature(module, input, output):
            features_blobs.append(output.data.cpu().numpy())

        self.net._modules.get(finalconv_name).register_forward_hook(hook_feature)
        # get the softmax weight
        params = list(net.parameters())
        weight_softmax = np.squeeze(params[-2].data.numpy())

        return weight_softmax, features_blobs

    def returnCAM(self,feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 512x512
        size_upsample = (299, 299)
        bz, nc, h, w = feature_conv.shape

        output_cam = []
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

    def predict(self, image):


        with torch.no_grad():

            self.OPEN_CAM = 1
            weight_softmax, features_blobs = self.get_weight_softmax(self.net)

            image_tensor = torch.Tensor(np.array(image) / 255.0).to(DEVICE)
            input_image = image_tensor.permute(2, 0, 1)
            out = self.forward(input_image)
            probs = torch.nn.functional.softmax(out, dim=1)
            probs_np = probs[0].detach().cpu().numpy()
            top_label = self.label_map[probs_np.argmax()]
            top_prob = probs_np.max()
            true_label = self.true_label
            true_prob = probs_np[self.true_class]

            CAMs = self.returnCAM(features_blobs[0], weight_softmax, [probs_np.argmax()])
            heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (299, 299)), cv2.COLORMAP_JET)
            cv2.imwrite(CAM_F, heatmap * 0.3 + np.array(image) * 0.5)

            self.cam_img = Image.open(CAM_F).transpose(Image.FLIP_TOP_BOTTOM).convert("RGBA")

            self.CAM_MAP = self.CTX.texture(self.cam_img.size, 4,
                                            self.cam_img.tobytes())
            self.CAM_MAP.build_mipmaps()
            vertices = np.array([[-1.0, -1.0, 0.0],
                                 [-1.0, 1.0, 0.0],
                                 [1.0, 1.0, 0.0],
                                 [-1.0, -1.0, 0.0],
                                 [1.0, -1.0, 0.0],
                                 [1.0, 1.0, 0.0]])
            # Not used for the background, but the vertex shader expects a normal.
            normals = np.repeat([[0.0, 0.0, 1.0]], len(vertices), axis=0)
            # Image coordinates are [0, 1].
            texture_coords = np.array([[0.0, 0.0],
                                       [0.0, 1.0],
                                       [1.0, 1.0],
                                       [0.0, 0.0],
                                       [1.0, 0.0],
                                       [1.0, 1.0]])

            CAM_ARRAY = np.hstack((vertices, normals, texture_coords))
            self.CAM_VBO = self.CTX.buffer(CAM_ARRAY.flatten().astype("f4").tobytes())
            self.CAM_VAO = self.CTX.simple_vertex_array(self.PROG, self.CAM_VBO,
                                                        "in_vert", "in_norm",
                                                        "in_text")



            self.pred_text.setText("<strong>Top Label</strong>: {0}<br>"
                                   "<strong>Top Label Probability</strong>: {1:.4f}<br><br>"
                                   "<strong>True Label</strong>: {2}<br>"
                                   "<strong>True Label Probability</strong>: {3:.4f}".format(top_label, top_prob, true_label, true_prob))





    def render(self):
        '''
        pass
        '''
        if self.OPEN_CAM == 1:
            self.CTX.disable(moderngl.DEPTH_TEST)
            self.PROG["mode"].value = 1
            self.CAM_MAP.use()
            self.CAM_VAO.render()
            self.CTX.enable(moderngl.DEPTH_TEST)
            self.PROG["mode"].value = 0



    def clear(self):
        '''
        pass
        '''
        if self.OPEN_CAM == 1:
            self.OPEN_CAM = 0
            self.CAM_MAP.release()
            self.CAM_VAO.release()
            self.CAM_VBO.release()

            self.CAM_VAO = []
            self.CAM_VBO = []


