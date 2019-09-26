import moderngl
import numpy as np
import os

from PIL import Image, ImageOps
from pyrr import Matrix44
from strike_with_a_pose.file_locations import *
from strike_with_a_pose.settings import *

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


def gen_file_with_vertex_normals(input_f):
    """If the .obj file is missing vertex normals, calculate and add them.

    :param input_f:
    :return:
    """
    vertices = []
    faces = []
    v_lines = []
    f_lines = []
    with open(input_f) as obj_f:
        for line in obj_f:
            line = line.strip()
            parts = line.split()
            if parts[0] == "v":
                vertices.append(np.array(parts[1:4], dtype=np.float))
                v_lines.append(line)
            elif parts[0] == "f":
                # Check if file already contains vertex normals.
                f_v1_parts = parts[1].split("/")
                if len(f_v1_parts) == 3:
                    if f_v1_parts[2]:
                        return False

                alt_f_parts = ["f"]
                face = []
                for part in parts[1:4]:
                    f_v_parts = part.split("/")
                    v = f_v_parts[0]
                    face.append(v)
                    vt = ""
                    if len(f_v_parts) >= 2:
                        vt = f_v_parts[1]

                    alt_f_parts.append(f"{v}/{vt}/{v}")

                faces.append(np.array(face, dtype=np.int) - 1)
                f_lines.append(" ".join(alt_f_parts))
            elif parts[0] != "mtllib":
                f_lines.append(line)
            else:
                v_lines.append(line)

    vertices = np.stack(vertices)
    # See: https://computergraphics.stackexchange.com/questions/1562/how-to-calculate-surface-normals-for-generated-geometry/1576#1576
    # and: http://www.iquilezles.org/www/articles/normals/normals.htm.
    vertex_normals = np.zeros(vertices.shape)
    for face in faces:
        (ic, ib, ia) = tuple(face)

        e1 = vertices[ia] - vertices[ib]
        e2 = vertices[ic] - vertices[ib]
        normal = np.cross(e1, e2)

        vertex_normals[ia] += normal
        vertex_normals[ib] += normal
        vertex_normals[ic] += normal

    vertex_normals = vertex_normals / np.linalg.norm(vertex_normals, axis=1)[None].T

    obj_name = input_f.split("/")[-1].split(".")[0]
    output_f = "/".join(input_f.split("/")[:-1]) + f"{obj_name}_alt.obj"
    with open(output_f, "w") as alt_obj_f:
        for v_line in v_lines:
            print(v_line, file=alt_obj_f)

        for vertex_normal in vertex_normals:
            print("vn " + " ".join([str(val) for val in vertex_normal]), file=alt_obj_f)

        for f_line in f_lines:
            print(f_line, file=alt_obj_f)

    return output_f


def parse_obj_file(input_obj):
    """Parse Wavefront .obj file.

    :param input_obj:
    :return: dictionary of NumPy arrays with shape (3 * num_faces, 8). Each row contains:
    (1) the coordinates of a vertex of a face, (2) the vertex's normal vector, and (3) the
    texture coordinates for the vertex. Every three rows correspond to a face.
    """
    data = {"v": [], "vn": [], "vt": []}
    packed_arrays = {}
    alt_obj_f = gen_file_with_vertex_normals(input_obj)
    if alt_obj_f:
        input_obj = alt_obj_f

    obj_f = open(input_obj)
    current_mtl = None
    min_vec = np.full(3, np.inf)
    max_vec = np.full(3, -np.inf)
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
            if len(parts[1:]) == 4:
                triangles = [parts[1:4], [parts[1], parts[3], parts[4]]]
            else:
                triangles = [parts[1:4]]

            for f in triangles:
                for fv in f:
                    v_parts = fv.split("/")
                    (v, vt, vn) = (v_parts[0], None, None)
                    if len(v_parts) == 3:
                        (vt, vn) = v_parts[1:]
                    elif len(v_parts) == 2:
                        vt = v_parts[1]

                    # Convert to zero-based indexing.
                    v = int(v) - 1
                    vt = int(vt) - 1 if vt else -1
                    vn = int(vn) - 1 if vn else -1

                    data_v = data["v"][v]

                    if vn != -1:
                        data_vn = data["vn"][vn]
                    else:
                        data_vn = np.zeros(3)

                    if vt != -1:
                        data_vt = data["vt"][vt][:2]
                    else:
                        data_vt = np.zeros(2)

                    row = np.concatenate((data_v, data_vn, data_vt))

                    packed_arrays[current_mtl].append(row)
        elif elem_type == "usemtl":
            current_mtl = parts[1]
            if current_mtl not in packed_arrays:
                packed_arrays[current_mtl] = []
        elif elem_type == "l":
            if current_mtl in packed_arrays:
                packed_arrays.pop(current_mtl)

    obj_f.close()
    if alt_obj_f:
        os.remove(alt_obj_f)

    max_pos_vec = max_vec - min_vec
    max_pos_val = max(max_pos_vec)
    max_pos_vec_norm = max_pos_vec / max_pos_val
    for (sub_obj, packed_array) in packed_arrays.items():
        packed_array = np.stack(packed_array)
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
            mtl_infos[current_mtl][elem_type] = tuple(vals)
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

    mtl_f.close()
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

                uniform float x;
                uniform float y;
                uniform float z;

                uniform mat3 R_obj;
                uniform mat3 R_light;
                uniform vec3 DirLight;
                uniform mat4 VP;
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
                        v_pos = R_obj * in_vert + vec3(x, y, z);
                        gl_Position = VP * vec4(v_pos, 1.0);
                        v_norm = R_obj * in_norm;
                        v_text = in_text;
                        v_light = R_light * DirLight;
                    } else {
                        gl_Position = vec4(in_vert, 1.0);
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
                            vec3 diffuse = dif * surface_rgb;
                            if (has_image) {
                                surface_rgb = texture(Texture, v_text).rgb;
                                diffuse = dif * dif_rgb * surface_rgb;
                            }
                            vec3 ambient = amb_int * amb_rgb * surface_rgb;
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
        self.PROG["x"].value = 0
        self.PROG["y"].value = 0
        self.PROG["z"].value = 0
        self.PROG["DirLight"].value = (0, 1, 0)
        self.PROG["dif_int"].value = 0.7
        self.PROG["amb_int"].value = 0.5
        self.PROG["cam_pos"].value = tuple(EYE)
        self.angle_of_view = 16.426
        self.TAN_ANGLE = np.tan(self.angle_of_view / 2 * np.pi / 180.0)
        perspective = Matrix44.perspective_projection(
            self.angle_of_view, RATIO, 0.1, 1000.0
        )
        self.PROG["VP"].write((perspective * LOOK_AT).astype("f4").tobytes())
        self.PROG["R_obj"].write(np.eye(3).astype("f4").tobytes())
        self.PROG["R_light"].write(np.eye(3).astype("f4").tobytes())
        self.PROG["amb_rgb"].value = (1.0, 1.0, 1.0)
        self.PROG["dif_rgb"].value = (1.0, 1.0, 1.0)
        self.PROG["spc_rgb"].value = (1.0, 1.0, 1.0)
        self.PROG["spec_exp"].value = 0.0
        self.use_spec = True

        self.CAMERA_DISTANCE = CAMERA_DISTANCE
        self.TOO_CLOSE = self.CAMERA_DISTANCE
        self.TOO_FAR = self.CAMERA_DISTANCE - 30.0

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
            # Image coordinates are [0, 1].
            texture_coords = np.array(
                [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
            )

            BACKGROUND_ARRAY = np.hstack((vertices, texture_coords))
            BACKGROUND_VBO = self.CTX.buffer(
                BACKGROUND_ARRAY.flatten().astype("f4").tobytes()
            )
            self.BACKGROUND_VAO = self.CTX.simple_vertex_array(
                self.PROG, BACKGROUND_VBO, "in_vert", "in_text"
            )

        # Load vertices and textures.
        VAOS = {}
        TEXTURES = {}
        packed_arrays = parse_obj_file(SCENE_DIR + OBJ_F)
        (MTL_INFOS, render_objs) = parse_mtl_file(SCENE_DIR + MTL_F)
        RENDER_OBJS = []
        for render_obj in render_objs:
            if render_obj not in packed_arrays:
                print("Skipping {0}.".format(render_obj))
                continue

            RENDER_OBJS.append(render_obj)
            packed_array = packed_arrays[render_obj]
            vbo = self.CTX.buffer(packed_array.flatten().astype("f4").tobytes())
            vao = self.CTX.simple_vertex_array(
                self.PROG, vbo, "in_vert", "in_norm", "in_text"
            )
            VAOS[render_obj] = vao

            if "map_Kd" in MTL_INFOS[render_obj]:
                texture_f = SCENE_DIR + MTL_INFOS[render_obj]["map_Kd"]
                texture_img = (
                    Image.open(texture_f)
                    .transpose(Image.FLIP_TOP_BOTTOM)
                    .convert("RGBA")
                )
                TEXTURE = self.CTX.texture(texture_img.size, 4, texture_img.tobytes())
                TEXTURE.build_mipmaps()
                TEXTURES[render_obj] = TEXTURE

        self.OG_RENDER_OBJS = RENDER_OBJS
        self.RENDER_OBJS = RENDER_OBJS
        self.RENDER_OBJ = True
        self.CULL_FACES = True
        self.VAOS = VAOS
        self.TEXTURES = TEXTURES
        self.MTL_INFOS = MTL_INFOS

        self.param_names = [
            "x",
            "y",
            "z",
            "yaw",
            "pitch",
            "roll",
            "amb_int",
            "dif_int",
            "DirLight",
            "angle_of_view",
        ]
        self.prog_vals = {"x", "y", "z", "amb_int", "dif_int"}
        self.angle2func = {
            "yaw": self.get_yaw_from_matrix,
            "pitch": self.get_pitch_from_matrix,
            "roll": self.get_roll_from_matrix,
        }

    def render(self):
        if self.USE_BACKGROUND and BACKGROUND_F is not None:

            self.CTX.disable(moderngl.DEPTH_TEST)
            self.PROG["mode"].value = 1
            self.BACKGROUND.use()
            self.BACKGROUND_VAO.render()

            self.CTX.enable(moderngl.DEPTH_TEST)
            self.PROG["mode"].value = 0

        else:
            self.CTX.clear(R, G, B)

        if self.RENDER_OBJ:
            if self.CULL_FACES:
                self.CTX.enable(moderngl.CULL_FACE)

            for RENDER_OBJ in self.RENDER_OBJS:
                if self.PROG["use_texture"].value:
                    self.PROG["amb_rgb"].value = self.MTL_INFOS[RENDER_OBJ]["Ka"]
                    self.PROG["dif_rgb"].value = self.MTL_INFOS[RENDER_OBJ]["Kd"]
                    if self.use_spec:
                        self.PROG["spc_rgb"].value = self.MTL_INFOS[RENDER_OBJ]["Ks"]
                        self.PROG["spec_exp"].value = self.MTL_INFOS[RENDER_OBJ]["Ns"]
                    else:
                        self.PROG["spc_rgb"].value = (0.0, 0.0, 0.0)

                    self.PROG["trans"].value = self.MTL_INFOS[RENDER_OBJ]["d"]
                    if RENDER_OBJ in self.TEXTURES:
                        self.PROG["has_image"].value = True
                        self.TEXTURES[RENDER_OBJ].use()

                self.VAOS[RENDER_OBJ].render()
                self.PROG["has_image"].value = False

            self.CTX.disable(moderngl.CULL_FACE)

    def set_xy(self, xy):
        self.PROG["x"].value = xy[0]
        self.PROG["y"].value = xy[1]

    def adjust_angle_of_view(self, angle_of_view):
        self.angle_of_view = angle_of_view
        self.TAN_ANGLE = np.tan(np.radians(self.angle_of_view / 2))
        perspective = Matrix44.perspective_projection(
            self.angle_of_view, RATIO, 0.1, 1000.0
        )
        self.PROG["VP"].write((perspective * LOOK_AT).astype("f4").tobytes())

    def gen_rot_matrix_yaw(self, yaw):
        R_yaw = np.eye(3)
        R_yaw[0, 0] = np.cos(yaw)
        R_yaw[0, 2] = np.sin(yaw)
        R_yaw[2, 0] = -np.sin(yaw)
        R_yaw[2, 2] = np.cos(yaw)
        return R_yaw

    def gen_rot_matrix_pitch(self, pitch):
        R_pitch = np.eye(3)
        R_pitch[1, 1] = np.cos(pitch)
        R_pitch[1, 2] = -np.sin(pitch)
        R_pitch[2, 1] = np.sin(pitch)
        R_pitch[2, 2] = np.cos(pitch)
        return R_pitch

    def gen_rot_matrix_roll(self, roll):
        R_roll = np.eye(3)
        R_roll[0, 0] = np.cos(roll)
        R_roll[0, 1] = -np.sin(roll)
        R_roll[1, 0] = np.sin(roll)
        R_roll[1, 1] = np.cos(roll)
        return R_roll

    def gen_rotation_matrix(self, yaw=0.0, pitch=0.0, roll=0.0):
        R_yaw = self.gen_rot_matrix_yaw(yaw)
        R_pitch = self.gen_rot_matrix_pitch(pitch)
        R_roll = self.gen_rot_matrix_roll(roll)
        return np.dot(R_yaw, np.dot(R_pitch, R_roll))

    def get_yaw_from_matrix(self, R_mat):
        return np.arctan2(R_mat[0, 2], R_mat[2, 2])

    def get_pitch_from_matrix(self, R_mat):
        return np.arctan2(-R_mat[1, 2], np.sqrt(R_mat[1, 0] ** 2 + R_mat[1, 1] ** 2))

    def get_roll_from_matrix(self, R_mat):
        return np.arctan2(R_mat[1, 0], R_mat[1, 1])

    def get_angles_from_matrix(self, R_mat):
        yaw = self.get_yaw_from_matrix(R_mat)
        pitch = self.get_pitch_from_matrix(R_mat)
        roll = self.get_roll_from_matrix(R_mat)
        return {"yaw": yaw, "pitch": pitch, "roll": roll}

    def rotate(self, angles, which):
        R_yaw = self.gen_rot_matrix_yaw(angles[0])
        R_pitch = self.gen_rot_matrix_pitch(angles[1])
        R_roll = self.gen_rot_matrix_roll(angles[2])
        R_mat = np.array(self.PROG[which].value).reshape((3, 3)).T
        R_mat = np.dot(np.dot(R_yaw, np.dot(R_pitch, R_roll)), R_mat)
        angles = self.get_angles_from_matrix(R_mat)
        R_mat = self.gen_rotation_matrix(**angles)
        self.PROG[which].write(R_mat.T.astype("f4").tobytes())

    def get_params(self):
        params = [
            (param_name, self.get_param(param_name)) for param_name in self.param_names
        ]
        return params

    def get_param(self, name):
        if name in self.prog_vals:
            return self.PROG[name].value
        elif name in self.angle2func:
            R_obj = np.array(self.PROG["R_obj"].value).reshape((3, 3)).T
            rads = self.angle2func[name](R_obj)
            return np.degrees(rads)
        elif name == "DirLight":
            R_light = np.array(self.PROG["R_light"].value).reshape((3, 3)).T
            return tuple(np.dot(R_light, np.array(self.PROG["DirLight"].value)))
        elif name == "angle_of_view":
            return self.angle_of_view

    def set_params(self, params):
        for (name, value) in params.items():
            self.set_param(name, value)

    def set_param(self, name, value):
        if name in self.angle2func:
            rads = np.radians(value)
            rads_x = np.cos(rads)
            rads_y = np.sin(rads)
            value = np.arctan2(rads_y, rads_x)

            R_obj = np.array(self.PROG["R_obj"].value).reshape((3, 3)).T
            angles = self.get_angles_from_matrix(R_obj)
            angles[name] = value
            R_obj = self.gen_rotation_matrix(**angles)
            self.PROG["R_obj"].write(R_obj.T.astype("f4").tobytes())
        elif name in self.prog_vals:
            self.PROG[name].value = value
        elif name == "angle_of_view":
            self.adjust_angle_of_view(value)
        elif name == "DirLight":
            self.PROG["DirLight"].value = value
            self.PROG["R_light"].write(np.eye(3).astype("f4").tobytes())
