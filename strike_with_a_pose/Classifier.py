import numpy as np
import pkg_resources
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from strike_with_a_pose.settings import CLASS_F, USE_INCEPTION

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGENET_F = pkg_resources.resource_filename("strike_with_a_pose", "imagenet_classes.txt")
SCENE_DIR = pkg_resources.resource_filename("strike_with_a_pose", "scene_files/")


def load_imagenet_label_map():
    """Map ImageNet integer indexes to labels.

    :return:
    """
    input_f = open(IMAGENET_F)
    label_map = {}
    for line in input_f:
        parts = line.strip().split(": ")
        (num, label) = (int(parts[0]), parts[1].replace("\"", ""))
        label_map[num] = label

    input_f.close()
    return label_map


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        if USE_INCEPTION:
            self.net = models.inception_v3(pretrained=True)
        else:
            self.net = models.alexnet(pretrained=True)

        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False

        # Set up preprocessor.
        self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])

        self.label_map = load_imagenet_label_map()

        # Object attributes.
        self.true_class = int(open("{0}{1}".format(SCENE_DIR, CLASS_F)).read())
        self.true_label = self.label_map[self.true_class]

    def forward(self, input_image):
        image_normalized = self.preprocess(input_image)

        # Generate predictions.
        out = self.net(image_normalized[None, :, :, :])
        return out

    def classify(self, image):
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
            return (top_label, top_prob, true_label, true_prob)