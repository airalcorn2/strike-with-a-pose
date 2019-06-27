import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image, ImageOps


class Model(nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()
        self.device = device

        self.net = models.inception_v3(pretrained=True)

        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False

        # Set up preprocessor.
        self.preprocess = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        (self.width, self.height) = (299, 299)

    def forward(self, input_image):
        (width, height) = (self.width, self.height)

        # Resize image to work with neural network.
        if height < input_image.height < input_image.width:
            new_height = height
            new_width = new_height * input_image.width // input_image.height
        else:
            new_width = width
            new_height = new_width * input_image.height // input_image.width

        image = input_image.resize((new_width, new_height), Image.ANTIALIAS)
        image = ImageOps.fit(image, (width, height), Image.ANTIALIAS)

        # Generate predictions.
        image_tensor = torch.Tensor(np.array(image) / 255.0)
        input_image = image_tensor.permute(2, 0, 1)
        image_normalized = self.preprocess(input_image)
        out = self.net(image_normalized[None, :, :, :].to(self.device))
        return out


def gen_rotation_matrix(yaw=0.0, pitch=0.0, roll=0.0):
    """Generate a rotation matrix from yaw, pitch, and roll angles (in radians).

    :param yaw:
    :param pitch:
    :param roll:
    :return:
    """
    R_yaw = np.eye(3)
    R_yaw[0, 0] = np.cos(yaw)
    R_yaw[0, 2] = np.sin(yaw)
    R_yaw[2, 0] = -np.sin(yaw)
    R_yaw[2, 2] = np.cos(yaw)

    R_pitch = np.eye(3)
    R_pitch[1, 1] = np.cos(pitch)
    R_pitch[1, 2] = -np.sin(pitch)
    R_pitch[2, 1] = np.sin(pitch)
    R_pitch[2, 2] = np.cos(pitch)

    R_roll = np.eye(3)
    R_roll[0, 0] = np.cos(roll)
    R_roll[0, 1] = -np.sin(roll)
    R_roll[1, 0] = np.sin(roll)
    R_roll[1, 1] = np.cos(roll)

    return np.dot(R_yaw, np.dot(R_pitch, R_roll))


def gen_rotation_matrix_from_azim_elev(azimuth=0.0, elevation=0.0):
    """Generate a rotation matrix from azimuth and elevation angles (in radians).

    :param azimuth:
    :param elevation:
    :return:
    """
    y = np.sin(elevation)
    radius = np.cos(elevation)
    x = radius * np.sin(azimuth)
    z = radius * np.cos(azimuth)

    EYE = np.array([x, y, z])
    TARGET = np.zeros(3)
    UP = np.array([0.0, 1.0, 0.0])

    diff = EYE - TARGET
    zaxis = diff / np.linalg.norm(diff)
    crossed = np.cross(UP, zaxis)
    xaxis = crossed / np.linalg.norm(crossed)
    yaxis = np.cross(zaxis, xaxis)

    return np.stack([xaxis, yaxis, zaxis])


def load_imagenet_label_map():
    """Load ImageNet label dictionary.

    :return:
    """
    input_f = open("imagenet_classes.txt")
    label_map = {}
    for line in input_f:
        parts = line.strip().split(": ")
        (num, label) = (int(parts[0]), parts[1].replace('"', ""))
        label_map[num] = label

    input_f.close()
    return label_map
