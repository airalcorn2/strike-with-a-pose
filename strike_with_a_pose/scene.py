import moderngl
import numpy as np
import pkg_resources
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from objloader import Obj
from PIL import Image, ImageOps
from pyrr import Matrix44
from strike_with_a_pose.settings import *

# Package resources.
SCENE_DIR = pkg_resources.resource_filename("strike_with_a_pose", "scene_files/")
IMAGENET_F = pkg_resources.resource_filename("strike_with_a_pose", "imagenet_classes.txt")

# PyTorch stuff.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Average ImageNet pixel.
(R, G, B) = (0.485, 0.456, 0.406)

# Camera stuff.
CAMERA_DISTANCE = 2.0

EYE = np.array([0.0, 0.0, CAMERA_DISTANCE])
TARGET = np.zeros(3)
UP = np.array([0.0, 1.0, 0.0])
LOOK_AT = Matrix44.look_at(EYE, TARGET, UP)

(WIDTH, HEIGHT) = (299, 299) if USE_INCEPTION else (224, 224)
RATIO = float(WIDTH) / float(HEIGHT)
VIEWING_ANGLE = 8.213
ANGLE = 2 * 8.213
perspective = Matrix44.perspective_projection(ANGLE, RATIO, 0.1, 1000.0)


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

    def forward(self, input_image):
        image_normalized = self.preprocess(input_image)

        # Generate predictions.
        out = self.net(image_normalized[None, :, :, :])
        return out


class Scene:
    WINDOW_SIZE = (WIDTH, HEIGHT)
    wnd = None

    def __init__(self):
        self.MODEL = Model().to(DEVICE)
        self.LABEL_MAP = load_imagenet_label_map()
        self.CTX = moderngl.create_context()

        self.PROG = self.CTX.program(
            vertex_shader="""
                #version 330

                uniform vec2 Pan;
                uniform float Zoom;

                uniform mat3 R;
                uniform mat3 L;
                uniform vec3 DirLight;
                uniform mat4 Mvp;
                uniform bool is_background;

                in vec3 in_vert;
                in vec3 in_norm;
                in vec2 in_text;

                out vec3 v_norm;
                out vec2 v_text;
                out vec3 v_light;

                void main() {
                    if (!is_background) {
                        gl_Position = Mvp * vec4((R * in_vert) + vec3(Pan, Zoom), 1.0);
                        v_norm = R * in_norm;
                        v_text = in_text;
                        v_light = L * DirLight;
                    } else {
                        gl_Position = vec4(in_vert, 1.0);
                        v_norm = in_norm;
                        v_text = in_text;
                    }
                }
            """,
            fragment_shader="""
                #version 330

                uniform float dir_int;
                uniform float amb_int;
                uniform sampler2D Texture;
                uniform bool is_background;
                uniform bool use_texture;

                in vec3 v_norm;
                in vec2 v_text;
                in vec3 v_light;

                out vec4 f_color;

                void main() {
                    if (!is_background) {
                        float lum = clamp(dot(v_light, v_norm), 0.0, 1.0) * dir_int + amb_int;
                        if (use_texture) {
                            f_color = vec4(texture(Texture, v_text).rgb * lum, texture(Texture, v_text).a);
                        } else {
                            f_color = vec4(vec3(1.0, 1.0, 1.0) * lum, 1.0);
                        }
                    } else {
                        f_color = vec4(texture(Texture, v_text).rgba);
                    }
                }
            """
        )

        self.CTX.enable(moderngl.DEPTH_TEST)
        self.CTX.enable(moderngl.BLEND)
        self.PROG["is_background"].value = False
        self.PROG["use_texture"].value = True
        self.USE_BACKGROUND = False
        self.PROG["use_texture"].value = False
        self.PROG["Pan"].value = (0, 0)
        self.PROG["Zoom"].value = 0
        self.PROG["DirLight"].value = (0, 1, 0)
        self.PROG["dir_int"].value = 0.7
        self.PROG["amb_int"].value = 0.5
        self.PROG["Mvp"].write((perspective * LOOK_AT).astype("f4").tobytes())
        self.R = np.eye(3)
        self.PROG["R"].write(self.R.astype("f4").tobytes())
        self.L = np.eye(3)
        self.PROG["L"].write(self.L.astype("f4").tobytes())
        self.CAMERA_DISTANCE = CAMERA_DISTANCE
        self.TOO_CLOSE = self.CAMERA_DISTANCE - 2.0
        self.TOO_FAR = self.CAMERA_DISTANCE - 30.0
        self.TAN_ANGLE = np.tan(VIEWING_ANGLE * np.pi / 180.0)

        # Load background.
        if BACKGROUND_F is not None:
            background_f = "{0}{1}".format(SCENE_DIR, BACKGROUND_F)
            background_img = Image.open(background_f).transpose(Image.FLIP_TOP_BOTTOM).convert("RGBA")
            (width, height) = (WIDTH, HEIGHT)

            # Resize background image to work with neural network.
            if height < background_img.height < background_img.width:
                new_height = height
                new_width = new_height * background_img.width // background_img.height
            else:
                new_width = width
                new_height = new_width * background_img.height // background_img.width

            background_img = background_img.resize((new_width, new_height), Image.ANTIALIAS)
            background_img = ImageOps.fit(background_img, (width, height), Image.ANTIALIAS)

            # Convert background image to ModernGL texture.
            self.BACKGROUND = self.CTX.texture(background_img.size, 4, background_img.tobytes())
            self.BACKGROUND.build_mipmaps()

            # Create background 3D object consisting of two triangles forming a
            # rectangle.
            # Screen coordinates are [-1, 1].
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

            BACKGROUND_ARRAY = np.hstack((vertices, normals, texture_coords))
            BACKGROUND_VBO = self.CTX.buffer(BACKGROUND_ARRAY.flatten().astype("f4").tobytes())
            self.BACKGROUND_VAO = self.CTX.simple_vertex_array(self.PROG, BACKGROUND_VBO,
                                                               "in_vert", "in_norm",
                                                               "in_text")

        # Set up object.
        self.TRUE_CLASS = int(open("{0}{1}".format(SCENE_DIR, CLASS_F)).read())
        self.TRUE_LABEL = self.LABEL_MAP[self.TRUE_CLASS]

        # Load textures.
        TEXTURES = []
        for TEXTURE_F in TEXTURE_FS:
            texture_f = SCENE_DIR + "{1}".format(SCENE_DIR, TEXTURE_F)
            texture_img = Image.open(texture_f).transpose(Image.FLIP_TOP_BOTTOM).convert("RGBA")
            TEXTURE = self.CTX.texture(texture_img.size, 4, texture_img.tobytes())
            TEXTURE.build_mipmaps()
            TEXTURES.append(TEXTURE)

        self.TEXTURES = TEXTURES

        # Load vertices.
        VAOS = []
        (min_val, abs_max_val, max_val) = (None, None, None)
        for OBJ_F in OBJ_FS:
            input_obj = SCENE_DIR + OBJ_F
            obj = Obj.open(input_obj)
            packed_array = obj.to_array()[:, :-1]

            # Normalize vertices into a unit cube centered at zero.
            original_vertices = packed_array[:, :3].copy()

            if min_val is None:
                min_val = original_vertices.min(axis=0)

            original_vertices -= min_val

            if abs_max_val is None:
                abs_max_val = np.abs(original_vertices).max()

            original_vertices /= abs_max_val
            original_vertices *= 2

            if max_val is None:
                max_val = original_vertices.max(axis=0)

            original_vertices -= max_val / 2
            packed_array[:, :3] = original_vertices

            vbo = self.CTX.buffer(packed_array.flatten().astype("f4").tobytes())
            vao = self.CTX.simple_vertex_array(self.PROG, vbo, "in_vert",
                                               "in_norm", "in_text")
            VAOS.append(vao)

        self.VAOS = VAOS

    def render(self):
        self.CTX.viewport = self.wnd.viewport

        if self.USE_BACKGROUND and BACKGROUND_F is not None:

            self.CTX.disable(moderngl.DEPTH_TEST)
            self.PROG["is_background"].value = True
            self.BACKGROUND.use()
            self.BACKGROUND_VAO.render()

            self.CTX.enable(moderngl.DEPTH_TEST)
            self.PROG["is_background"].value = False

        else:
            self.CTX.clear(R, G, B)

        for (i, VAO) in enumerate(self.VAOS):
            if self.PROG["use_texture"].value:
                self.TEXTURES[i].use()

            VAO.render()

    def pan(self, deltas):
        self.PROG["Pan"].value = deltas

    def zoom(self, delta):
        self.PROG["Zoom"].value = delta

    def set_amb(self, new_int):
        self.PROG["amb_int"].value = new_int

    def set_dir(self, new_int):
        self.PROG["dir_int"].value = new_int

    def gen_rotation_matrix_from_angle_axis(self, theta, axis):
        # See: https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle.
        c = np.cos(theta)
        s = np.sin(theta)
        (ux, uy, uz) = (axis[0], axis[1], axis[2])
        x_col = np.array([[c + ux ** 2 * (1 - c)],
                          [uy * ux * (1 - c) + uz * s],
                          [uz * ux * (1 - c) - uy * s]])
        y_col = np.array([[ux * uy * (1 - c) - uz * s],
                          [c + uy ** 2 * (1 - c)],
                          [uz * uy * (1 - c) + ux * s]])
        z_col = np.array([[ux * uz * (1 - c) + uy * s],
                          [uy * uz * (1 - c) - ux * s],
                          [c + uz ** 2 * (1 - c)]])
        return np.hstack((x_col, y_col, z_col))

    def gen_rotation_matrix(self, yaw=0.0, pitch=0.0, roll=0.0):
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

    def get_angles_from_R(self):
        R = self.R.T
        yaw = np.arctan2(R[0, 2], R[2, 2])
        roll = np.arctan2(R[1, 0], R[1, 1])
        pitch = np.arctan2(-R[1, 2], np.sqrt(R[1, 0] ** 2 + R[1, 1] ** 2))
        return(yaw, pitch, roll)

    def rotate(self, angles):
        # Update rotation matrix.
        R_yaw = self.gen_rotation_matrix_from_angle_axis(angles[0], self.R[:, 1])
        R_pitch = self.gen_rotation_matrix_from_angle_axis(angles[1], self.R[:, 0])
        R_roll = self.gen_rotation_matrix_from_angle_axis(angles[2], self.R[:, 2])
        self.R = np.dot(np.dot(R_yaw, np.dot(R_pitch, R_roll)), self.R)
        self.PROG["R"].write(self.R.astype("f4").tobytes())

    def rotate_light(self, angles):
        L_yaw = self.gen_rotation_matrix_from_angle_axis(angles[0], self.L[:, 1])
        L_pitch = self.gen_rotation_matrix_from_angle_axis(angles[1], self.L[:, 0])
        L_roll = self.gen_rotation_matrix_from_angle_axis(angles[2], self.L[:, 2])
        self.L = np.dot(np.dot(L_yaw, np.dot(L_pitch, L_roll)), self.L)
        self.PROG["L"].write(self.L.astype("f4").tobytes())

    def predict(self, image):
        with torch.no_grad():
            image_tensor = torch.Tensor(np.array(image) / 255.0).to(DEVICE)
            input_image = image_tensor.permute(2, 0, 1)
            out = self.MODEL(input_image)
            probs = torch.nn.functional.softmax(out, dim=1)
            probs_np = probs[0].detach().cpu().numpy()
            top_label = self.LABEL_MAP[probs_np.argmax()]
            top_prob = probs_np.max()
            true_label = self.TRUE_LABEL
            true_prob = probs_np[self.TRUE_CLASS]
            return (top_label, top_prob, true_label, true_prob)

    def get_params(self):
        (x, y) = self.PROG["Pan"].value
        z = self.PROG["Zoom"].value
        (yaw, pitch, roll) = self.get_angles_from_R()

        params = [("x_delta", x), ("y_delta", y), ("z_delta", z),
                  ("yaw", np.degrees(yaw)), ("pitch", np.degrees(pitch)), ("roll", np.degrees(roll)),
                  ("amb_int", self.PROG["amb_int"].value),
                  ("dir_int", self.PROG["dir_int"].value),
                  ("DirLight", tuple(np.dot(self.L.T, np.array(self.PROG["DirLight"].value))))]
        return params

    def set_params(self, params):
        self.PROG["Pan"].value = (params["x_delta"], params["y_delta"])
        self.PROG["Zoom"].value = params["z_delta"]
        for rot in ["yaw", "pitch", "roll"]:
            rads = np.radians(params[rot])
            rads_x = np.cos(rads)
            rads_y = np.sin(rads)
            params[rot] = np.arctan2(rads_y, rads_x)

        self.R = self.gen_rotation_matrix(params["yaw"], params["pitch"],
                                          params["roll"]).T
        self.PROG["R"].write(self.R.astype("f4").tobytes())
        self.PROG["amb_int"].value = params["amb_int"]
        self.PROG["dir_int"].value = params["dir_int"]
        self.PROG["DirLight"].value = params["DirLight"]
        self.L = np.eye(3)
        self.PROG["L"].write(self.L.astype("f4").tobytes())