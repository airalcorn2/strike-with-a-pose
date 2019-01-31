# PYTHONPATH=strike_with_a_pose python -m strike_with_a_pose.app

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
from strike_with_a_pose.settings import INITIAL_PARAMS, MODEL

INSTRUCTIONS_F = pkg_resources.resource_filename(
    "strike_with_a_pose", "instructions.html"
)

fmt = QSurfaceFormat()
fmt.setVersion(3, 3)
fmt.setProfile(QSurfaceFormat.CoreProfile)
fmt.setSwapInterval(1)
fmt.setDepthBufferSize(24)
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


class PanTool:
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


class WheelTool:
    def __init__(self, amb_int, dif_int, too_close, too_far):
        self.total_z = 0.0
        self.too_close = too_close
        self.too_far = too_far
        self.amb_int = amb_int
        self.dif_int = dif_int

    def zooming(self, step):
        self.total_z -= step / 10
        self.total_z = max(self.too_far, min(self.too_close, self.total_z))

    def change_amb(self, step):
        self.amb_int += step / 10
        self.amb_int = max(0, self.amb_int)

    def change_dif(self, step):
        self.dif_int += step / 10
        self.dif_int = max(0, self.dif_int)

    def set_z(self, z):
        self.total_z = max(self.too_far, min(self.too_close, z))


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
        self.data = {"obj": self.obj, "light": self.light}

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
            data["yaw"] = -np.arctan2(R[0, 2], R[2, 2])
            data["roll"] = -np.arctan2(R[1, 0], R[1, 1])
            data["pitch"] = -np.arctan2(-R[1, 2], np.sqrt(R[1, 0] ** 2 + R[1, 1] ** 2))

    def stop_drag(self, x, y, which):
        if self.arcball_on:
            self.dragging(x, y, which)
            self.arcball_on = False

    def get_value(self, which):
        data = self.data[which]
        return (data["yaw"], data["pitch"], data["roll"])


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

        self.wnd = WindowInfo()
        # For high DPI displays.
        pixel_ratio = QtWidgets.QDesktopWidget().devicePixelRatio()
        self.wnd.viewport = (0, 0) + (pixel_ratio * width, pixel_ratio * height)
        self.wnd.ratio = width / height
        self.wnd.size = (width, height)

        self.pan_tool = None
        self.wheel_tool = None
        self.rotate_tool = RotateTool(self.wnd.size[0], self.wnd.size[1])

        self.rotate = False
        self.ambient = False
        self.directional = False
        self.live = False

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
            if name == "z_delta" and not (
                self.scene.TOO_FAR <= params[name] <= self.scene.TOO_CLOSE
            ):
                z_info = QMessageBox()
                z_info.setText(
                    "z_delta is capped between {0} and {1}.".format(
                        self.scene.TOO_FAR, self.scene.TOO_CLOSE
                    )
                )
                z_info.setWindowTitle("z_delta")
                z_info.show()
                self.z_info = z_info

        dist = np.abs(self.scene.CAMERA_DISTANCE - params["z_delta"])
        max_trans = dist * self.pan_tool.tan_angle
        for trans in ["x_delta", "y_delta"]:
            if not (-max_trans <= params[trans] <= max_trans):
                trans_info = QMessageBox()
                trans_info.setText(
                    "{0} is capped between -{1:.4f} and {1:.4f} for a z_delta of {2:.4f}.".format(
                        trans, max_trans, params["z_delta"]
                    )
                )
                trans_info.setWindowTitle(trans)
                trans_info.show()
                self.trans_info = trans_info

        self.wheel_tool.set_z(params["z_delta"])
        params["z_delta"] = self.wheel_tool.total_z
        self.pan_tool.set_value(params["x_delta"], params["y_delta"], dist)
        params["x_delta"] = self.pan_tool.total_x
        params["y_delta"] = self.pan_tool.total_y

        self.scene.set_params(params)
        self.fill_entry_form()

    def keyPressEvent(self, event):
        # Quit when ESC is pressed
        if event.key() == QtCore.Qt.Key_Escape:
            QtCore.QCoreApplication.instance().quit()

        self.wnd.keys[event.nativeVirtualKey() & 0xFF] = True

        if event.key() == QtCore.Qt.Key_O:
            self.scene.RENDER_OBJ = not self.scene.RENDER_OBJ

        if event.key() == QtCore.Qt.Key_E:
            (sub_obj_idx, _) = QInputDialog.getInt(
                self, "Select Sub Object", "Object Index", len(self.scene.SUB_OBJS)
            )
            if sub_obj_idx >= len(self.scene.SUB_OBJS):
                self.scene.RENDER_OBJS = self.scene.SUB_OBJS
            else:
                self.scene.RENDER_OBJS = self.scene.SUB_OBJS[
                    sub_obj_idx : sub_obj_idx + 1
                ]
                sub_obj = self.scene.SUB_OBJS[sub_obj_idx]
                print(sub_obj)
                print(self.scene.MTL_INFO[sub_obj])

        if event.key() == QtCore.Qt.Key_S:
            self.scene.use_spec = not self.scene.use_spec

        if event.key() == QtCore.Qt.Key_Q:
            self.get_prediction()

        if event.key() == QtCore.Qt.Key_L:
            self.live = not self.live

        if event.key() == QtCore.Qt.Key_B:
            self.scene.USE_BACKGROUND = not self.scene.USE_BACKGROUND
            self.model.clear()

        if event.key() == QtCore.Qt.Key_X:
            self.scene.PROG["use_texture"].value = not self.scene.PROG[
                "use_texture"
            ].value

        if event.key() == QtCore.Qt.Key_T:
            self.rotate = False
            self.ambient = False
            self.directional = False
            self.mode_text.setText(
                """<p align="center"><strong>Translate</strong></p>"""
            )

        if event.key() == QtCore.Qt.Key_R:
            if self.rotate:
                self.rotate = False
                self.mode_text.setText(
                    """<p align="center"><strong>Translate</strong></p>"""
                )
            else:
                self.rotate = True
                self.mode_text.setText(
                    """<p align="center"><strong>Rotate</strong></p>"""
                )
            self.ambient = False
            self.directional = False

        if event.key() == QtCore.Qt.Key_A:
            if self.ambient:
                self.ambient = False
                self.mode_text.setText(
                    """<p align="center"><strong>Translate</strong></p>"""
                )
            else:
                self.ambient = True
                self.mode_text.setText(
                    """<p align="center"><strong>Ambient Light</strong></p>"""
                )

            self.rotate = False
            self.directional = False

        if event.key() == QtCore.Qt.Key_D:
            if self.directional:
                self.directional = False
                self.mode_text.setText(
                    """<p align="center"><strong>Translate</strong></p>"""
                )
            else:
                self.directional = True
                self.mode_text.setText(
                    """<p align="center"><strong>Directional Light</strong></p>"""
                )
            self.rotate = False
            self.ambient = False

    def get_prediction(self):
        # See: https://stackoverflow.com/questions/1733096/convert-pyqt-to-pil-image.
        self.model.clear()

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

    def keyReleaseEvent(self, event):
        self.wnd.keys[event.nativeVirtualKey() & 0xFF] = False

    def wheelEvent(self, evt):
        # See: http://doc.qt.io/qt-5/qwheelevent.html#angleDelta
        # and: https://wiki.qt.io/Smooth_Zoom_In_QGraphicsView.
        steps = evt.angleDelta().y() / (8 * 15)
        if self.ambient:
            self.wheel_tool.change_amb(steps)
            self.scene.set_amb(self.wheel_tool.amb_int)
        elif self.directional:
            self.wheel_tool.change_dif(steps)
            self.scene.set_dir(self.wheel_tool.dif_int)
        elif not self.rotate:
            self.wheel_tool.zooming(steps)
            self.scene.zoom(self.wheel_tool.total_z)

        if not self.rotate:
            self.model.clear()

        self.update()
        self.fill_entry_form()

    def mousePressEvent(self, evt):
        if self.rotate:
            self.rotate_tool.start_drag(evt.x(), evt.y(), "obj")
            self.scene.rotate(self.rotate_tool.get_value("obj"))
        elif self.directional:
            self.rotate_tool.start_drag(evt.x(), evt.y(), "light")
            self.scene.rotate_light(self.rotate_tool.get_value("light"))
        elif not self.ambient:
            self.pan_tool.start_drag(evt.x() / 512, evt.y() / 512)
            self.scene.pan(
                self.pan_tool.get_value(
                    np.abs(self.scene.CAMERA_DISTANCE - self.wheel_tool.total_z)
                )
            )
        self.update()
        self.fill_entry_form()
        self.model.clear()

    def mouseMoveEvent(self, evt):
        if self.rotate:
            self.rotate_tool.dragging(evt.x(), evt.y(), "obj")
            self.scene.rotate(self.rotate_tool.get_value("obj"))
        elif self.directional:
            self.rotate_tool.dragging(evt.x(), evt.y(), "light")
            self.scene.rotate_light(self.rotate_tool.get_value("light"))
        elif not self.ambient:
            self.pan_tool.dragging(evt.x() / 512, evt.y() / 512)
            self.scene.pan(
                self.pan_tool.get_value(
                    np.abs(self.scene.CAMERA_DISTANCE - self.wheel_tool.total_z)
                )
            )
        self.update()
        self.fill_entry_form()

    def mouseReleaseEvent(self, evt):
        if self.rotate:
            self.rotate_tool.stop_drag(evt.x(), evt.y(), "obj")
            self.scene.rotate(self.rotate_tool.get_value("obj"))
        elif self.directional:
            self.rotate_tool.stop_drag(evt.x(), evt.y(), "light")
            self.scene.rotate_light(self.rotate_tool.get_value("light"))
        elif not (self.ambient or self.directional):
            self.pan_tool.stop_drag(
                evt.x() / 512,
                evt.y() / 512,
                np.abs(self.scene.CAMERA_DISTANCE - self.wheel_tool.total_z),
            )
            self.scene.pan(
                self.pan_tool.get_value(
                    np.abs(self.scene.CAMERA_DISTANCE - self.wheel_tool.total_z)
                )
            )
        self.update()
        self.fill_entry_form()

    def paintGL(self):
        if self.scene is None:
            self.scene = self.scene_class()
            self.wheel_tool = WheelTool(
                self.scene.PROG["amb_int"].value,
                self.scene.PROG["dif_int"].value,
                self.scene.TOO_CLOSE,
                self.scene.TOO_FAR,
            )
            self.pan_tool = PanTool(self.scene.TAN_ANGLE)

            self.pan_tool.total_x = INITIAL_PARAMS["x_delta"]
            self.pan_tool.total_y = INITIAL_PARAMS["y_delta"]
            self.wheel_tool.total_z = INITIAL_PARAMS["z_delta"]
            self.scene.set_params(INITIAL_PARAMS)
            self.fill_entry_form()
            self.scene.USE_BACKGROUND = INITIAL_PARAMS["USE_BACKGROUND"]

            self.model = MODEL()
            for (name, comp) in self.model_gui_comps.items():
                if name != "predict":
                    setattr(self.model, name, comp)

            self.model.CTX = self.scene.CTX
            self.model.PROG = self.scene.PROG
            self.model.init_scene_comps()
            self.scene.MODEL = self.model

            self.scene.render()

            self.get_prediction()

        self.wnd.time = time.clock() - self.start_time
        self.scene.render()
        if self.live:
            self.get_prediction()

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

        flo = QFormLayout()
        entry_fields = {}
        params = [
            "x_delta",
            "y_delta",
            "z_delta",
            "yaw",
            "pitch",
            "roll",
            "amb_int",
            "dif_int",
            "DirLight",
        ]
        for name in params:
            # See: https://www.tutorialspoint.com/pyqt/pyqt_qlineedit_widget.htm.
            edit = QLineEdit()
            edit.setFixedWidth(200)
            entry_fields[name] = edit
            if name in {"yaw", "pitch", "roll"}:
                name += " (deg.)"
            flo.addRow(name, edit)

        self.scene_window.entry_fields = entry_fields
        flo.setContentsMargins(0, 0, 0, 0)

        # See: https://www.tutorialspoint.com/pyqt/pyqt_qpushbutton_widget.htm.
        submit = QPushButton("Set Parameters")
        submit.setFixedWidth(comp_width)
        submit.clicked.connect(self.scene_window.set_entry_form_values)

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
        self.setFixedHeight(525)

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
