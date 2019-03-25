# PYTHONPATH=strike_with_a_pose python3 -m strike_with_a_pose.app

import io
import numpy as np
import pkg_resources
import time

from OpenGL import GL
from PIL import Image
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QSurfaceFormat
from PyQt5.QtWidgets import QFormLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit
from PyQt5.QtWidgets import QMessageBox, QOpenGLWidget, QPushButton, QVBoxLayout
from PyQt5.QtWidgets import QWidget
from strike_with_a_pose.scene import Scene
from strike_with_a_pose.settings import INITIAL_PARAMS, MODEL, TRUE_CLASS

INSTRUCTIONS_F = pkg_resources.resource_filename(
    "strike_with_a_pose", "instructions.html"
)
ABOUT_F = pkg_resources.resource_filename("strike_with_a_pose", "about.html")

fmt = QSurfaceFormat()
fmt.setVersion(3, 3)
fmt.setProfile(QSurfaceFormat.CoreProfile)
fmt.setSwapInterval(1)
fmt.setDepthBufferSize(24)
fmt.setSamples(3)
QSurfaceFormat.setDefaultFormat(fmt)


class WindowInfo:
    def __init__(self):
        self.size = (0, 0)
        self.mouse = (0, 0)
        self.wheel = 0
        self.time = 0
        self.ratio = 1.0
        self.viewport = (0, 0, 0, 0)
        self.keys = np.full(256, False)
        self.old_keys = np.copy(self.keys)

    def key_down(self, key):
        return self.keys[key]

    def key_pressed(self, key):
        return self.keys[key] and not self.old_keys[key]

    def key_released(self, key):
        return not self.keys[key] and self.old_keys[key]


class WheelTool:
    def __init__(self, amb_int, dif_int, too_close, too_far, angle_of_view):
        self.total_z = 0.0
        self.too_close = too_close
        self.too_far = too_far
        self.amb_int = amb_int
        self.dif_int = dif_int
        self.angle_of_view = angle_of_view
        self.min_angle_of_view = angle_of_view
        self.step_scaling = 10
        self.param2getter = {
            "z": self.get_z,
            "amb_int": self.get_amb,
            "dif_int": self.get_dif,
            "angle_of_view": self.get_angle_of_view,
        }
        self.param2setter = {
            "z": self.set_z,
            "amb_int": self.set_amb,
            "dif_int": self.set_dif,
            "angle_of_view": self.set_angle_of_view,
        }

    def get_z(self):
        return self.total_z

    def get_amb(self):
        return self.amb_int

    def get_dif(self):
        return self.dif_int

    def get_angle_of_view(self):
        return self.angle_of_view

    def change_z(self, step):
        self.total_z -= step / self.step_scaling
        self.total_z = max(self.too_far, min(self.too_close, self.total_z))

    def change_amb(self, step):
        self.amb_int += step / self.step_scaling
        self.amb_int = max(0, self.amb_int)

    def change_dif(self, step):
        self.dif_int += step / self.step_scaling
        self.dif_int = max(0, self.dif_int)

    def change_angle_of_view(self, step):
        self.angle_of_view += step / self.step_scaling
        self.angle_of_view = max(self.min_angle_of_view, min(self.angle_of_view, 90))

    def set_z(self, z):
        self.total_z = max(self.too_far, min(self.too_close, z))

    def set_amb(self, amb):
        self.amb_int = max(0, amb)

    def set_dif(self, dif):
        self.dif_int = max(0, dif)

    def set_angle_of_view(self, angle):
        self.angle_of_view = max(self.min_angle_of_view, min(angle, 90))


class RotateTool:
    # See: https://en.wikibooks.org/wiki/OpenGL_Programming/Modern_OpenGL_Tutorial_Arcball.
    def __init__(self, screen_width, screen_height):
        self.arcball_on = False
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.obj = {
            "last_mx": 0.0,
            "last_my": 0.0,
            "cur_mx": 0.0,
            "cur_my": 0.0,
            "yaw": 0,
            "pitch": 0,
            "roll": 0,
        }
        self.light = {
            "last_mx": 0.0,
            "last_my": 0.0,
            "cur_mx": 0.0,
            "cur_my": 0.0,
            "yaw": 0,
            "pitch": 0,
            "roll": 0,
        }
        self.data = {"R_obj": self.obj, "R_light": self.light}

    def start_drag(self, x, y, which):
        data = self.data[which]
        data["last_mx"] = data["cur_mx"] = x
        data["last_my"] = data["cur_my"] = y
        self.arcball_on = True

    def dragging(self, x, y, which):
        if self.arcball_on:
            data = self.data[which]
            data["last_mx"] = data["cur_mx"]
            data["last_my"] = data["cur_my"]
            data["cur_mx"] = x
            data["cur_my"] = y
            self.update_angles(which)

    def get_arcball_vector(self, x, y, screen_width, screen_height):
        P = [2 * x / screen_width - 1.0, 2 * y / screen_height - 1.0, 0]
        P[1] = -P[1]
        OP_squared = P[0] ** 2 + P[1] ** 2
        if OP_squared <= 1:
            P[2] = np.sqrt(1 - OP_squared)
        else:
            P[0] /= np.sqrt(OP_squared)
            P[1] /= np.sqrt(OP_squared)
        return P

    def update_angles(self, which):
        data = self.data[which]
        if (data["cur_mx"] != data["last_mx"]) or (data["cur_my"] != data["last_my"]):
            va = self.get_arcball_vector(
                data["last_mx"], data["last_my"], self.screen_width, self.screen_height
            )
            vb = self.get_arcball_vector(
                data["cur_mx"], data["cur_my"], self.screen_width, self.screen_height
            )
            theta = 3.0 * np.arccos(min((1, np.dot(va, vb))))
            axis = np.cross(va, vb)
            c = np.cos(theta)
            s = np.sin(theta)
            # See: https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle.
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
            # See: http://www.songho.ca/opengl/gl_anglestoaxes.html. "RyRxRz" --> (yaw, pitch, roll)
            R = np.hstack((x_col, y_col, z_col))

            # See: http://planning.cs.uiuc.edu/node103.html
            # and: http://www.gregslabaugh.net/publications/euler.pdf.
            data["yaw"] = np.arctan2(R[0, 2], R[2, 2])
            data["roll"] = np.arctan2(R[1, 0], R[1, 1])
            data["pitch"] = np.arctan2(-R[1, 2], np.sqrt(R[1, 0] ** 2 + R[1, 1] ** 2))

    def stop_drag(self, x, y, which):
        if self.arcball_on:
            self.dragging(x, y, which)
            self.arcball_on = False

    def get_value(self, which):
        data = self.data[which]
        return (data["yaw"], data["pitch"], data["roll"])


class XYTool:
    def __init__(self, tan_angle):
        self.total_x = 0.0
        self.total_y = 0.0
        self.start_x = 0.0
        self.start_y = 0.0
        self.delta_x = 0.0
        self.delta_y = 0.0
        self.tan_angle = tan_angle
        self.drag = False

    def start_drag(self, x, y):
        self.start_x = x
        self.start_y = y
        self.drag = True

    def dragging(self, x, y):
        if self.drag:
            self.delta_x = (x - self.start_x) * 2.0
            self.delta_y = (y - self.start_y) * 2.0

    def stop_drag(self, x, y, dist):
        if self.drag:
            self.dragging(x, y)
            self.total_x += self.delta_x
            self.total_y -= self.delta_y

            max_trans = dist * self.tan_angle
            self.total_x = max(-max_trans, min(max_trans, self.total_x))
            self.total_y = max(-max_trans, min(max_trans, self.total_y))

            self.delta_x = 0.0
            self.delta_y = 0.0
            self.drag = False

    def get_value(self, dist):
        (tmp_x, tmp_y) = (self.total_x + self.delta_x, self.total_y - self.delta_y)
        max_trans = dist * self.tan_angle
        tmp_x = max(-max_trans, min(max_trans, tmp_x))
        tmp_y = max(-max_trans, min(max_trans, tmp_y))
        return (tmp_x, tmp_y)

    def set_value(self, x, y, dist):
        max_trans = dist * self.tan_angle
        self.total_x = max(-max_trans, min(max_trans, x))
        self.total_y = max(-max_trans, min(max_trans, y))


class SceneWindow(QOpenGLWidget):
    def __init__(self):
        super(SceneWindow, self).__init__(None)

        self.setFocusPolicy(QtCore.Qt.ClickFocus)

        (width, height) = Scene.WINDOW_SIZE
        self.setFixedSize(width, height)

        self.start_time = time.clock()
        self.scene_class = lambda: None
        self.scene = None
        self.entry_fields = None
        self.mode_text = None
        self.about_txt = open(ABOUT_F).read()

        self.wnd = WindowInfo()
        # For high DPI displays.
        pixel_ratio = QtWidgets.QDesktopWidget().devicePixelRatio()
        self.wnd.viewport = (0, 0) + (pixel_ratio * width, pixel_ratio * height)
        self.wnd.ratio = width / height
        self.wnd.size = (width, height)

        self.wheel_tool = None
        self.rotate_tool = None
        self.xy_tool = None

        self.live = False

        self.TRANS = 0
        self.ROT = 1
        self.AMB = 2
        self.DIF = 3
        self.VIEW = 4
        self.mode_names = [
            "Translate",
            "Rotate",
            "Ambient Light",
            "Directional Light",
            "Angle of View",
        ]
        self.mode_keys = {
            QtCore.Qt.Key_T: self.TRANS,
            QtCore.Qt.Key_R: self.ROT,
            QtCore.Qt.Key_A: self.AMB,
            QtCore.Qt.Key_D: self.DIF,
            QtCore.Qt.Key_V: self.VIEW,
        }
        self.mode_things = [None, "R_obj", None, "R_light", None]
        self.mode = self.TRANS

        self.screenshot = 0
        self.make_prediction = True

    def fill_entry_form(self):
        params = self.scene.get_params()
        for (name, value) in params:
            edit = self.entry_fields[name]
            if name != "DirLight":
                edit.setText("{0:.5f}".format(value)[:-1])
            else:
                edit.setText("({0:.4f}, {1:.4f}, {2:.4f})".format(*value))

    def set_entry_form_values(self):
        params = {}
        for (name, field) in self.entry_fields.items():
            field_val = field.text()
            if name != "DirLight":
                try:
                    params[name] = float(field_val)
                except ValueError:
                    error = QMessageBox()
                    error.setText("{0} is not a float.".format(field_val))
                    error.setWindowTitle("Bad Value")
                    error.show()
                    self.error = error
                    return
            else:
                try:
                    params[name] = np.array(eval(field.text()), dtype="float32")
                    params[name] /= np.linalg.norm(params[name])
                    params[name] = tuple(params[name])
                except:
                    error = QMessageBox()
                    error.setText("{0} is not a triple of floats.".format(field_val))
                    error.setWindowTitle("Bad Value")
                    error.show()
                    self.error = error
                    return

            if name == "z" and not (
                self.scene.TOO_FAR <= params[name] <= self.scene.TOO_CLOSE
            ):
                z_info = QMessageBox()
                z_info.setText(
                    "z is capped between {0} and {1}.".format(
                        self.scene.TOO_FAR, self.scene.TOO_CLOSE
                    )
                )
                z_info.setWindowTitle("z")
                z_info.show()
                self.z_info = z_info
                return

        dist = np.abs(self.scene.CAMERA_DISTANCE - params["z"])
        new_tan = np.tan(params["angle_of_view"] * np.pi / 180.0)
        max_trans = dist * new_tan
        for trans in ["x", "y"]:
            if not (-max_trans <= params[trans] <= max_trans):
                trans_info = QMessageBox()
                trans_info.setText(
                    "{0} is capped between -{1:.4f} and {1:.4f} for a z of {2:.4f} and a angle_of_view of {3:.4f}.".format(
                        trans, max_trans, params["z"], params["angle_of_view"]
                    )
                )
                trans_info.setWindowTitle(trans)
                trans_info.show()
                self.trans_info = trans_info
                return

        for param in self.wheel_tool.param2setter:
            self.wheel_tool.param2setter[param](params[param])
            params[param] = self.wheel_tool.param2getter[param]()

        self.xy_tool.set_value(params["x"], params["y"], dist)
        params["x"] = self.xy_tool.total_x
        params["y"] = self.xy_tool.total_y
        self.xy_tool.tan_angle = new_tan
        self.set_too_close()
        self.set_min_angle_of_view()

        self.scene.set_params(params)
        self.fill_entry_form()

    def set_too_close(self):
        too_close_x = np.abs(self.xy_tool.total_x) / self.xy_tool.tan_angle
        too_close_y = np.abs(self.xy_tool.total_y) / self.xy_tool.tan_angle
        self.wheel_tool.too_close = self.scene.CAMERA_DISTANCE - max(
            too_close_x, too_close_y
        )

    def set_min_angle_of_view(self):
        min_angle_x = np.arctan(
            np.abs(self.xy_tool.total_x)
            / np.abs(self.scene.CAMERA_DISTANCE - self.wheel_tool.total_z)
        )
        min_angle_y = np.arctan(
            np.abs(self.xy_tool.total_y)
            / np.abs(self.scene.CAMERA_DISTANCE - self.wheel_tool.total_z)
        )
        self.wheel_tool.min_angle_of_view = np.degrees(max(min_angle_x, min_angle_y))

    def keyPressEvent(self, event):

        key = event.key()

        if key == QtCore.Qt.Key_F:
            self.scene.CULL_FACES = not self.scene.CULL_FACES

        # Quit when ESC is pressed
        if key == QtCore.Qt.Key_Escape:
            QtCore.QCoreApplication.instance().quit()

        self.wnd.keys[event.nativeVirtualKey() & 0xFF] = True

        if key == QtCore.Qt.Key_G:
            QMessageBox.about(self, "About", self.about_txt)

        if key == QtCore.Qt.Key_C:
            self.capture_screenshot()

        if key == QtCore.Qt.Key_O:
            self.scene.RENDER_OBJ = not self.scene.RENDER_OBJ
            self.model.clear()

        if key == QtCore.Qt.Key_I:
            (render_obj_idx, _) = QInputDialog.getInt(
                self,
                "Select Individual Component",
                "Component Index",
                len(self.scene.OG_RENDER_OBJS),
            )
            if render_obj_idx >= len(self.scene.OG_RENDER_OBJS):
                self.scene.RENDER_OBJS = self.scene.OG_RENDER_OBJS
            else:
                self.scene.RENDER_OBJS = self.scene.OG_RENDER_OBJS[
                    render_obj_idx : render_obj_idx + 1
                ]
                render_obj = self.scene.OG_RENDER_OBJS[render_obj_idx]
                print(render_obj)
                print(self.scene.MTL_INFOS[render_obj])

        if key == QtCore.Qt.Key_S:
            self.scene.use_spec = not self.scene.use_spec

        if key == QtCore.Qt.Key_Q:
            self.make_prediction = True

        if key == QtCore.Qt.Key_L:
            self.live = not self.live

        if key == QtCore.Qt.Key_B:
            self.scene.USE_BACKGROUND = not self.scene.USE_BACKGROUND
            self.model.clear()

        if key == QtCore.Qt.Key_X:
            self.scene.PROG["use_texture"].value = not self.scene.PROG[
                "use_texture"
            ].value

        if key in self.mode_keys:
            self.mode = self.mode_keys[key]
            self.mode_text.setText(
                """<p align="center"><strong>{0}</strong></p>""".format(
                    self.mode_names[self.mode]
                )
            )

    def get_prediction(self):
        # See: https://stackoverflow.com/questions/1733096/convert-pyqt-to-pil-image.
        buffer = QtCore.QBuffer()
        buffer.open(QtCore.QIODevice.ReadWrite)
        qimage = self.grabFramebuffer()
        qimage.save(buffer, "PNG")

        strio = io.BytesIO()
        strio.write(buffer.data())
        buffer.close()
        strio.seek(0)
        pil_im = Image.open(strio)
        pil_im = pil_im.resize(self.scene.WINDOW_SIZE)

        self.model.predict(pil_im)

    def capture_screenshot(self):
        buffer = QtCore.QBuffer()
        buffer.open(QtCore.QIODevice.ReadWrite)
        qimage = self.grabFramebuffer()
        result = qimage.save("screenshot_{0}.png".format(str(self.screenshot)).zfill(2))
        self.screenshot += 1

    def keyReleaseEvent(self, event):
        self.wnd.keys[event.nativeVirtualKey() & 0xFF] = False

    def wheelEvent(self, evt):
        # See: http://doc.qt.io/qt-5/qwheelevent.html#angleDelta
        # and: https://wiki.qt.io/Smooth_Zoom_In_QGraphicsView.
        mode = self.mode
        if mode in self.wheel_update_funcs:
            steps = evt.angleDelta().y() / (8 * 15)
            self.wheel_update_funcs[mode](steps)
            self.scene.set_param(
                self.wheel_scene_params[mode], self.wheel_tool_vals[mode]()
            )

        if mode != self.ROT:
            self.model.clear()

        if mode == self.VIEW:
            self.xy_tool.tan_angle = np.tan(
                self.wheel_tool.angle_of_view * np.pi / 180.0
            )

        self.update()
        self.fill_entry_form()

    def mouseClickDragEvent(self, evt, state):
        mode = self.mode
        if mode == self.TRANS:
            (x, y) = (evt.x() / 512, evt.y() / 512)
            if state != "stop":
                self.drag_update_funcs[state][mode](x, y)
            else:
                self.drag_update_funcs[state][mode](
                    x, y, np.abs(self.scene.CAMERA_DISTANCE - self.wheel_tool.total_z)
                )
                self.set_too_close()
                self.set_min_angle_of_view()

            (x, y) = self.xy_tool.get_value(
                np.abs(self.scene.CAMERA_DISTANCE - self.wheel_tool.total_z)
            )
            self.scene.set_param("x", x)
            self.scene.set_param("y", y)
        elif mode == self.ROT or mode == self.DIF:
            (x, y) = (evt.x(), evt.y())
            thing = self.mode_things[mode]
            self.drag_update_funcs[state][mode](x, y, thing)
            self.scene.rotate(self.rotate_tool.get_value(thing), thing)

        self.update()
        self.fill_entry_form()

    def mousePressEvent(self, evt):
        self.model.clear()
        self.mouseClickDragEvent(evt, "start")

    def mouseMoveEvent(self, evt):
        self.mouseClickDragEvent(evt, "dragging")

    def mouseReleaseEvent(self, evt):
        self.mouseClickDragEvent(evt, "stop")

    def initialize_scene(self):
        self.scene = self.scene_class()
        self.scene.CTX.viewport = self.wnd.viewport
        self.scene.set_params(INITIAL_PARAMS)
        self.scene.USE_BACKGROUND = INITIAL_PARAMS["USE_BACKGROUND"]
        self.fill_entry_form()

    def initialize_model(self):
        self.model = MODEL(TRUE_CLASS)
        for (name, comp) in self.model_gui_comps.items():
            if name != "predict":
                setattr(self.model, name, comp)

        self.model.CTX = self.scene.CTX
        self.model.PROG = self.scene.PROG
        self.model.init_scene_comps()

    def initialize_wheel_tool(self):
        (amb, dif, z, view) = (self.AMB, self.DIF, self.TRANS, self.VIEW)
        self.wheel_tool = WheelTool(
            self.scene.PROG["amb_int"].value,
            self.scene.PROG["dif_int"].value,
            self.scene.TOO_CLOSE,
            self.scene.TOO_FAR,
            self.scene.angle_of_view,
        )
        self.wheel_update_funcs = {
            amb: self.wheel_tool.change_amb,
            dif: self.wheel_tool.change_dif,
            z: self.wheel_tool.change_z,
            view: self.wheel_tool.change_angle_of_view,
        }
        self.wheel_scene_params = {
            amb: "amb_int",
            dif: "dif_int",
            z: "z",
            view: "angle_of_view",
        }
        self.wheel_tool_vals = {
            amb: self.wheel_tool.get_amb,
            dif: self.wheel_tool.get_dif,
            z: self.wheel_tool.get_z,
            view: self.wheel_tool.get_angle_of_view,
        }
        self.wheel_tool.total_z = INITIAL_PARAMS["z"]

    def initialize_drag_tools(self):
        (rot, dif, xy) = (self.ROT, self.DIF, self.TRANS)
        self.rotate_tool = RotateTool(self.wnd.size[0], self.wnd.size[1])
        self.xy_tool = XYTool(self.scene.TAN_ANGLE)
        self.drag_update_funcs = {
            "start": {
                rot: self.rotate_tool.start_drag,
                dif: self.rotate_tool.start_drag,
                xy: self.xy_tool.start_drag,
            },
            "dragging": {
                rot: self.rotate_tool.dragging,
                dif: self.rotate_tool.dragging,
                xy: self.xy_tool.dragging,
            },
            "stop": {
                rot: self.rotate_tool.stop_drag,
                dif: self.rotate_tool.stop_drag,
                xy: self.xy_tool.stop_drag,
            },
        }
        self.drag_modes = {rot, dif, xy}
        self.xy_tool.total_x = INITIAL_PARAMS["x"]
        self.xy_tool.total_y = INITIAL_PARAMS["y"]

    def paintGL(self):
        if self.scene is None:
            self.initialize_scene()
            self.initialize_model()
            self.initialize_wheel_tool()
            self.initialize_drag_tools()

        self.wnd.time = time.clock() - self.start_time

        if self.make_prediction or self.live:
            self.model.clear()
            self.scene.render()
            self.get_prediction()
            self.make_prediction = False
        else:
            self.scene.render()

        self.model.render()

        self.wnd.old_keys = np.copy(self.wnd.keys)
        self.wnd.wheel = 0
        self.update()


class Window(QWidget):
    def __init__(self, title):
        super(Window, self).__init__(None)

        comp_width = 299
        col_width = 350
        header_height = 25

        # OpenGL window.
        self.scene_window = SceneWindow()

        # GUI consists of three components laid out horizontally.

        # Component #1: parameter entry.
        # Parameter entry components are laid out vertically.
        pvlo = QVBoxLayout()

        param_text = QLabel(
            """<p align="center"><strong>Scene Parameters</strong></p>"""
        )
        param_text.setFixedSize(col_width, header_height)
        pvlo.addWidget(param_text)

        flo_widget = QWidget()

        # See: https://www.tutorialspoint.com/pyqt/pyqt_qpushbutton_widget.htm.
        submit = QPushButton("Set Parameters")
        submit.setFixedWidth(comp_width)
        submit.clicked.connect(self.scene_window.set_entry_form_values)

        flo = QFormLayout()
        entry_fields = {}
        params = [
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
        for name in params:
            # See: https://www.tutorialspoint.com/pyqt/pyqt_qlineedit_widget.htm.
            edit = QLineEdit()
            edit.setFixedWidth(200)
            edit.returnPressed.connect(submit.click)
            entry_fields[name] = edit
            if name in {"yaw", "pitch", "roll"}:
                name += " (deg.)"

            flo.addRow(name, edit)

        self.scene_window.entry_fields = entry_fields
        flo.setContentsMargins(0, 0, 0, 0)

        flo_widget.setLayout(flo)
        flo_widget.setFixedWidth(comp_width)

        pvlo.addWidget(flo_widget)
        pvlo.setAlignment(flo_widget, QtCore.Qt.AlignHCenter)
        pvlo.addWidget(submit)
        pvlo.setAlignment(submit, QtCore.Qt.AlignHCenter)

        # Component #2: scene and neural network predictions.
        vlo = QVBoxLayout()

        # Mode text.
        mode_text = QLabel("""<p align="center"><strong>Translate</strong></p>""")
        mode_text.setFixedSize(col_width, header_height)
        self.scene_window.mode_text = mode_text

        vlo.addWidget(mode_text)

        vlo.addWidget(self.scene_window)
        vlo.setAlignment(self.scene_window, QtCore.Qt.AlignHCenter)

        # Model-specific GUI components.
        model_gui_comps = {}
        for (name, comp) in MODEL.get_gui_comps():
            if name == "predict":
                comp.clicked.connect(self.scene_window.get_prediction)

            comp.setFixedWidth(comp_width)
            vlo.addWidget(comp)
            vlo.setAlignment(comp, QtCore.Qt.AlignHCenter)

            model_gui_comps[name] = comp

        self.scene_window.model_gui_comps = model_gui_comps

        # Component #3: instructions.
        ivlo = QVBoxLayout()

        instructions_header = QLabel(
            """<p align="center"><strong>Instructions</strong></p>"""
        )
        instructions_header.setFixedHeight(header_height)
        ivlo.addWidget(instructions_header)

        instructions_txt = open(INSTRUCTIONS_F).read()
        instructions = QLabel(instructions_txt)
        instructions.setAlignment(QtCore.Qt.AlignTop)
        instructions.setFixedWidth(col_width)
        instructions.setWordWrap(True)
        instructions.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
        instructions.setOpenExternalLinks(True)

        ivlo.addWidget(instructions)

        # Combine three components.
        hlo = QHBoxLayout()
        hlo.addLayout(pvlo)
        hlo.setAlignment(pvlo, QtCore.Qt.AlignTop)
        hlo.addLayout(vlo)
        hlo.setAlignment(vlo, QtCore.Qt.AlignTop)
        hlo.addLayout(ivlo)
        hlo.setAlignment(ivlo, QtCore.Qt.AlignTop)

        self.setLayout(hlo)

        self.title = title
        self.setWindowTitle(self.title)
        self.setFixedHeight(495)

        self.scene_window.setFocus()


def run_gui():
    app = QtWidgets.QApplication([])
    widget = Window("Strike (With) A Pose")
    Scene.wnd = widget.scene_window.wnd
    widget.scene_window.scene_class = Scene
    widget.show()
    widget.move(QtWidgets.QDesktopWidget().rect().center() - widget.rect().center())
    app.exec_()
    del app


def main():
    run_gui()


if __name__ == "__main__":
    main()
