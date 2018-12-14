import numpy as np
import pkg_resources
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PyQt5 import QtCore
from PyQt5.QtWidgets import QLabel, QPushButton

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGENET_F = pkg_resources.resource_filename("strike_with_a_pose", "data/imagenet_classes.txt")
SCENE_DIR = pkg_resources.resource_filename("strike_with_a_pose", "scene_files/")
TRUE_CLASS = 609


class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.net = models.inception_v3(pretrained=True)
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False

        self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])

        self.label_map = self.load_imagenet_label_map()

        self.true_class = TRUE_CLASS
        self.true_label = self.label_map[self.true_class]

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
        classify = QPushButton("Classify")
        # Order matters. Prediction button must be named "predict" in tuple.
        return [("pred_text", pred_text), ("predict", classify)]

    def init_scene_comps(self):
        pass

    def forward(self, input_image):
        image_normalized = self.preprocess(input_image)
        out = self.net(image_normalized[None, :, :, :])
        return out

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

            self.pred_text.setText("<strong>Top Label</strong>: {0}<br>"
                                   "<strong>Top Label Probability</strong>: {1:.4f}<br><br>"
                                   "<strong>True Label</strong>: {2}<br>"
                                   "<strong>True Label Probability</strong>: {3:.4f}".format(top_label, top_prob, true_label, true_prob))

    def render(self):
        pass

    def clear(self):
        pass