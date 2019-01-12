import moderngl
import numpy as np
import pkg_resources

from PIL import Image, ImageOps
from pyrr import Matrix44
from strike_with_a_pose.settings import *

# Package resources.
SCENE_DIR = pkg_resources.resource_filename("strike_with_a_pose", "scene_files/")

# Average ImageNet pixel.
(R, G, B) = (0.485, 0.456, 0.406)

# Camera stuff.
CAMERA_DISTANCE = 2.0

EYE = np.array([0.0, 0.0, CAMERA_DISTANCE])
TARGET = np.zeros(3)
UP = np.array([0.0, 1.0, 0.0])
LOOK_AT = Matrix44.look_at(EYE, TARGET, UP)

(WIDTH, HEIGHT) = (299, 299)
RATIO = float(WIDTH) / float(HEIGHT)
VIEWING_ANGLE = 8.213
ANGLE = 2 * 8.213
perspective = Matrix44.perspective_projection(ANGLE, RATIO, 0.1, 1000.0)


def parse_obj_file(input_obj):
    """Parse wavefront .obj file.

    :param input_obj:
    :return: NumPy array with shape (3 * num_faces, 8). Each row contains (1) the
    coordinates of a vertex of a face, (2) the vertex's normal vector, and (3) the texture
    coordinates for the vertex.
    """
    data = {"v": [], "vn": [], "vt": []}
    packed_arrays = {}
    obj_f = open(input_obj)
    current_mtl = None
    min_vec = np.full(3, np.inf)
    max_vec = np.full(3, -np.inf)
    empty_vt = np.array([0.0, 0.0, 0.0])
    for line in obj_f:
        line = line.strip()
        if line == "":
            continue

        parts = line.split()
        elem_type = parts[0]
        if elem_type in data:
            vals = np.array(parts[1:4], dtype=np.float)
            if elem_type == "v":
                min_vec = np.minimum(min_vec, vals)
                max_vec = np.maximum(max_vec, vals)
            elif elem_type == "vn":
                vals /= np.linalg.norm(vals)
            elif elem_type == "vt":
                if len(vals) < 3:
                    vals = np.array(list(vals) + [0.0], dtype=np.float)

            data[elem_type].append(vals)
        elif elem_type == "f":
            f = parts[1:4]
            for fv in f:
                (v, vt, vn) = fv.split("/")

                # Convert to zero-based indexing.
                v = int(v) - 1
                vn = int(vn) - 1
                vt = int(vt) - 1 if vt else -1

                if vt == -1:
                    row = np.concatenate((data["v"][v], data["vn"][vn], empty_vt))
                else:
                    row = np.concatenate((data["v"][v], data["vn"][vn], data["vt"][vt]))
                packed_arrays[current_mtl].append(row)
        elif elem_type == "usemtl":
            current_mtl = parts[1]
            if current_mtl not in packed_arrays:
                packed_arrays[current_mtl] = []
        elif elem_type == "l":
            if current_mtl in packed_arrays:
                packed_arrays.pop(current_mtl)

    max_pos_vec = max_vec - min_vec
    max_pos_val = max(max_pos_vec)
    max_pos_vec_norm = max_pos_vec / max_pos_val
    for (sub_obj, packed_array) in packed_arrays.items():
        # z-coordinate of texture is always zero (if present).
        packed_array = np.stack(packed_array)[:, :8]
        original_vertices = packed_array[:, :3].copy()

        # All coordinates greater than or equal to zero.
        original_vertices -= min_vec
        # All coordinates between zero and one.
        original_vertices /= max_pos_val
        # All coordinates between zero and two.
        original_vertices *= 2
        # All coordinates between negative one and positive one with the center of object
        # at (0, 0, 0).
        original_vertices -= max_pos_vec_norm

        packed_array[:, :3] = original_vertices
        packed_arrays[sub_obj] = packed_array

    return packed_arrays


def parse_mtl_file(input_mtl):
    """Parse Wavefront .mtl file.

    :param input_mtl:
    :return:
    """
    vector_elems = {"Ka", "Kd", "Ks"}
    float_elems = {"Ns", "Ni", "d"}
    int_elems = {"illum"}
    current_mtl = None
    mtl_infos = {}
    mtl_f = open(input_mtl)
    sub_objs = []
    for line in mtl_f:
        line = line.strip()
        if line == "":
            continue

        parts = line.split()
        elem_type = parts[0]
        if elem_type in vector_elems:
            vals = np.array(parts[1:4], dtype=np.float)
            mtl_infos[current_mtl][elem_type] = vals
        elif elem_type in float_elems:
            mtl_infos[current_mtl][elem_type] = float(parts[1])
        elif elem_type in int_elems:
            mtl_infos[current_mtl][elem_type] = int(parts[1])
        elif elem_type == "newmtl":
            current_mtl = parts[1]
            sub_objs.append(current_mtl)
            mtl_infos[current_mtl] = {"d": 1.0}
        elif elem_type == "map_Kd":
            mtl_infos[current_mtl]["map_Kd"] = parts[1]

    sub_objs.sort()
    sub_objs.reverse()
    non_trans = [sub_obj for sub_obj in sub_objs if mtl_infos[sub_obj]["d"] == 1.0]
    trans = [
        (sub_obj, mtl_infos[sub_obj]["d"])
        for sub_obj in sub_objs
        if mtl_infos[sub_obj]["d"] < 1.0
    ]
    trans.sort(key=lambda x: x[1], reverse=True)
    sub_objs = non_trans + [sub_obj for (sub_obj, d) in trans]
    return (mtl_infos, sub_objs)


class Scene:
    WINDOW_SIZE = (WIDTH, HEIGHT)
    wnd = None

    def __init__(self):
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
                uniform int mode;

                in vec3 in_vert;
                in vec3 in_norm;
                in vec2 in_text;

                out vec3 v_pos;
                out vec3 v_norm;
                out vec2 v_text;
                out vec3 v_light;

                void main() {
                    if (mode == 0) {
                        gl_Position = Mvp * vec4((R * in_vert) + vec3(Pan, Zoom), 1.0);
                        v_pos = in_vert;
                        v_norm = R * in_norm;
                        v_text = in_text;
                        v_light = L * DirLight;
                    } else {
                        gl_Position = vec4(in_vert, 1.0);
                        v_pos = in_vert;
                        v_norm = in_norm;
                        v_text = in_text;
                    }
                }
            """,
            fragment_shader="""
                #version 330

                uniform float amb_int;
                uniform float dif_int;
                uniform vec3 cam_pos;

                uniform sampler2D Texture;
                uniform int mode;
                uniform bool use_texture;
                uniform bool has_image;

                uniform vec3 box_rgb;

                uniform vec3 amb_rgb;
                uniform vec3 dif_rgb;
                uniform vec3 spc_rgb;
                uniform float spec_exp;
                uniform float trans;

                in vec3 v_pos;
                in vec3 v_norm;
                in vec2 v_text;
                in vec3 v_light;

                out vec4 f_color;

                void main() {
                    if (mode == 0) {
                        float dif = clamp(dot(v_light, v_norm), 0.0, 1.0) * dif_int;
                        if (use_texture) {
                            vec3 surface_rgb = dif_rgb;
                            if (has_image) {
                                surface_rgb = texture(Texture, v_text).rgb;
                            }
                            vec3 ambient = amb_int * amb_rgb * surface_rgb;
                            vec3 diffuse = dif * dif_rgb * surface_rgb;
                            float spec = 0.0;
                            if (dif > 0.0) {
                                vec3 reflected = reflect(-v_light, v_norm);
                                vec3 surface_to_camera = normalize(cam_pos - v_pos);
                                spec = pow(clamp(dot(surface_to_camera, reflected), 0.0, 1.0), spec_exp);
                            }
                            vec3 specular = spec * spc_rgb * surface_rgb;
                            vec3 linear = ambient + diffuse + specular;
                            f_color = vec4(linear, trans);
                        } else {
                            f_color = vec4(vec3(1.0, 1.0, 1.0) * dif + amb_int, 1.0);
                        }
                    } else if (mode == 1) {
                        f_color = vec4(texture(Texture, v_text).rgba);
                    } else {
                        f_color = vec4(box_rgb, 1.0);
                    }
                }
            """,
        )

        self.CTX.enable(moderngl.DEPTH_TEST)
        self.CTX.enable(moderngl.BLEND)
        self.PROG["mode"].value = 0
        self.PROG["use_texture"].value = True
        self.PROG["has_image"].value = False
        self.PROG["Pan"].value = (0, 0)
        self.PROG["Zoom"].value = 0
        self.PROG["DirLight"].value = (0, 1, 0)
        self.PROG["dif_int"].value = 0.7
        self.PROG["amb_int"].value = 0.5
        self.PROG["cam_pos"].value = tuple(EYE)
        self.PROG["Mvp"].write((perspective * LOOK_AT).astype("f4").tobytes())
        self.R = np.eye(3)
        self.PROG["R"].write(self.R.astype("f4").tobytes())
        self.L = np.eye(3)
        self.PROG["L"].write(self.L.astype("f4").tobytes())
        self.PROG["amb_rgb"].value = (1.0, 1.0, 1.0)
        self.PROG["dif_rgb"].value = (1.0, 1.0, 1.0)
        self.PROG["spc_rgb"].value = (1.0, 1.0, 1.0)
        self.PROG["spec_exp"].value = 0.0
        self.use_spec = True

        self.CAMERA_DISTANCE = CAMERA_DISTANCE
        self.TOO_CLOSE = self.CAMERA_DISTANCE - 2.0
        self.TOO_FAR = self.CAMERA_DISTANCE - 30.0
        self.TAN_ANGLE = np.tan(VIEWING_ANGLE * np.pi / 180.0)

        # Load background.
        self.USE_BACKGROUND = False
        if BACKGROUND_F is not None:
            background_f = "{0}{1}".format(SCENE_DIR, BACKGROUND_F)
            background_img = (
                Image.open(background_f)
                .transpose(Image.FLIP_TOP_BOTTOM)
                .convert("RGBA")
            )
            (width, height) = (WIDTH, HEIGHT)

            # Resize background image to work with neural network.
            if height < background_img.height < background_img.width:
                new_height = height
                new_width = new_height * background_img.width // background_img.height
            else:
                new_width = width
                new_height = new_width * background_img.height // background_img.width

            background_img = background_img.resize(
                (new_width, new_height), Image.ANTIALIAS
            )
            background_img = ImageOps.fit(
                background_img, (width, height), Image.ANTIALIAS
            )

            # Convert background image to ModernGL texture.
            self.BACKGROUND = self.CTX.texture(
                background_img.size, 4, background_img.tobytes()
            )
            self.BACKGROUND.build_mipmaps()

            # Create background 3D object consisting of two triangles forming a
            # rectangle.
            # Screen coordinates are [-1, 1].
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
            # Not used for the background, but the vertex shader expects a normal.
            normals = np.repeat([[0.0, 0.0, 1.0]], len(vertices), axis=0)
            # Image coordinates are [0, 1].
            texture_coords = np.array(
                [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
            )

            BACKGROUND_ARRAY = np.hstack((vertices, normals, texture_coords))
            BACKGROUND_VBO = self.CTX.buffer(
                BACKGROUND_ARRAY.flatten().astype("f4").tobytes()
            )
            self.BACKGROUND_VAO = self.CTX.simple_vertex_array(
                self.PROG, BACKGROUND_VBO, "in_vert", "in_norm", "in_text"
            )

        # Load vertices and textures.
        VAOS = {}
        TEXTURES = {}
        packed_arrays = parse_obj_file(SCENE_DIR + OBJ_F)
        (mtl_infos, sub_objs) = parse_mtl_file(SCENE_DIR + MTL_F)
        SUB_OBJS = []
        for sub_obj in sub_objs:
            if sub_obj not in packed_arrays:
                print("Skipping {0}.".format(sub_obj))
                continue

            SUB_OBJS.append(sub_obj)
            packed_array = packed_arrays[sub_obj]
            vbo = self.CTX.buffer(packed_array.flatten().astype("f4").tobytes())
            vao = self.CTX.simple_vertex_array(
                self.PROG, vbo, "in_vert", "in_norm", "in_text"
            )
            VAOS[sub_obj] = vao

            if "map_Kd" in mtl_infos[sub_obj]:
                texture_f = SCENE_DIR + mtl_infos[sub_obj]["map_Kd"]
                texture_img = (
                    Image.open(texture_f)
                    .transpose(Image.FLIP_TOP_BOTTOM)
                    .convert("RGBA")
                )
                TEXTURE = self.CTX.texture(texture_img.size, 4, texture_img.tobytes())
                TEXTURE.build_mipmaps()
                TEXTURES[sub_obj] = TEXTURE

        self.SUB_OBJS = SUB_OBJS
        self.RENDER_OBJS = self.SUB_OBJS
        self.VAOS = VAOS
        self.TEXTURES = TEXTURES
        self.MTL_INFO = mtl_infos

    def render(self):
        self.CTX.viewport = self.wnd.viewport

        if self.USE_BACKGROUND and BACKGROUND_F is not None:

            self.CTX.disable(moderngl.DEPTH_TEST)
            self.PROG["mode"].value = 1
            self.BACKGROUND.use()
            self.BACKGROUND_VAO.render()

            self.CTX.enable(moderngl.DEPTH_TEST)
            self.PROG["mode"].value = 0

        else:
            self.CTX.clear(R, G, B)

        for SUB_OBJ in self.RENDER_OBJS:
            if self.PROG["use_texture"].value:
                self.PROG["amb_rgb"].value = tuple(self.MTL_INFO[SUB_OBJ]["Ka"])
                self.PROG["dif_rgb"].value = tuple(self.MTL_INFO[SUB_OBJ]["Kd"])
                if self.use_spec:
                    self.PROG["spc_rgb"].value = tuple(self.MTL_INFO[SUB_OBJ]["Ks"])
                    self.PROG["spec_exp"].value = self.MTL_INFO[SUB_OBJ]["Ns"]
                else:
                    self.PROG["spc_rgb"].value = (0.0, 0.0, 0.0)

                self.PROG["trans"].value = self.MTL_INFO[SUB_OBJ]["d"]
                if SUB_OBJ in self.TEXTURES:
                    self.PROG["has_image"].value = True
                    self.TEXTURES[SUB_OBJ].use()

            self.VAOS[SUB_OBJ].render()
            self.PROG["has_image"].value = False

        self.MODEL.render()

    def pan(self, deltas):
        self.PROG["Pan"].value = deltas

    def zoom(self, delta):
        self.PROG["Zoom"].value = delta

    def set_amb(self, new_int):
        self.PROG["amb_int"].value = new_int

    def set_dir(self, new_int):
        self.PROG["dif_int"].value = new_int

    def gen_rotation_matrix_from_angle_axis(self, theta, axis):
        # See: https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle.
        c = np.cos(theta)
        s = np.sin(theta)
        (ux, uy, uz) = (axis[0], axis[1], axis[2])
        x_col = np.array(
            [
                [c + ux ** 2 * (1 - c)],
                [uy * ux * (1 - c) + uz * s],
                [uz * ux * (1 - c) - uy * s],
            ]
        )
        y_col = np.array(
            [
                [ux * uy * (1 - c) - uz * s],
                [c + uy ** 2 * (1 - c)],
                [uz * uy * (1 - c) + ux * s],
            ]
        )
        z_col = np.array(
            [
                [ux * uz * (1 - c) + uy * s],
                [uy * uz * (1 - c) - ux * s],
                [c + uz ** 2 * (1 - c)],
            ]
        )
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

    def get_angles_from_matrix(self, R):
        R = R.T
        yaw = np.arctan2(R[0, 2], R[2, 2])
        roll = np.arctan2(R[1, 0], R[1, 1])
        pitch = np.arctan2(-R[1, 2], np.sqrt(R[1, 0] ** 2 + R[1, 1] ** 2))
        return (yaw, pitch, roll)

    def rotate(self, angles):
        R_yaw = self.gen_rotation_matrix_from_angle_axis(angles[0], self.R[:, 1])
        R_pitch = self.gen_rotation_matrix_from_angle_axis(angles[1], self.R[:, 0])
        R_roll = self.gen_rotation_matrix_from_angle_axis(angles[2], self.R[:, 2])
        R = np.dot(np.dot(R_yaw, np.dot(R_pitch, R_roll)), self.R)
        (yaw, pitch, roll) = self.get_angles_from_matrix(R)
        self.R = self.gen_rotation_matrix(yaw, pitch, roll).T
        self.PROG["R"].write(self.R.astype("f4").tobytes())

    def rotate_light(self, angles):
        L_yaw = self.gen_rotation_matrix_from_angle_axis(angles[0], self.L[:, 1])
        L_pitch = self.gen_rotation_matrix_from_angle_axis(angles[1], self.L[:, 0])
        L_roll = self.gen_rotation_matrix_from_angle_axis(angles[2], self.L[:, 2])
        L = np.dot(np.dot(L_yaw, np.dot(L_pitch, L_roll)), self.L)
        (yaw, pitch, roll) = self.get_angles_from_matrix(L)
        self.L = self.gen_rotation_matrix(yaw, pitch, roll).T
        self.PROG["L"].write(self.L.astype("f4").tobytes())

    def get_params(self):
        (x, y) = self.PROG["Pan"].value
        z = self.PROG["Zoom"].value
        (yaw, pitch, roll) = self.get_angles_from_matrix(self.R)

        params = [
            ("x_delta", x),
            ("y_delta", y),
            ("z_delta", z),
            ("yaw", np.degrees(yaw)),
            ("pitch", np.degrees(pitch)),
            ("roll", np.degrees(roll)),
            ("amb_int", self.PROG["amb_int"].value),
            ("dif_int", self.PROG["dif_int"].value),
            (
                "DirLight",
                tuple(np.dot(self.L.T, np.array(self.PROG["DirLight"].value))),
            ),
        ]
        return params

    def set_params(self, params):
        self.PROG["Pan"].value = (params["x_delta"], params["y_delta"])
        self.PROG["Zoom"].value = params["z_delta"]
        for rot in ["yaw", "pitch", "roll"]:
            rads = np.radians(params[rot])
            rads_x = np.cos(rads)
            rads_y = np.sin(rads)
            params[rot] = np.arctan2(rads_y, rads_x)

        self.R = self.gen_rotation_matrix(
            params["yaw"], params["pitch"], params["roll"]
        ).T
        self.PROG["R"].write(self.R.astype("f4").tobytes())
        self.PROG["amb_int"].value = params["amb_int"]
        self.PROG["dif_int"].value = params["dif_int"]
        self.PROG["DirLight"].value = params["DirLight"]
        self.L = np.eye(3)
        self.PROG["L"].write(self.L.astype("f4").tobytes())
