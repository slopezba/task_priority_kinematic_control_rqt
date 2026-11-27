"""Manual client for the bimanual end-effectors facing planner, sharing the panel's ROS node."""

import math
import os

from ament_index_python.packages import get_package_share_directory
from action_msgs.msg import GoalStatus
from python_qt_binding.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
    QAction, QApplication, QLabel, QLineEdit, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)
from python_qt_binding.QtCore import Qt
from python_qt_binding.QtGui import QKeySequence
from rclpy.action import ActionClient
from rclpy.time import Time
from std_srvs.srv import Trigger
from sura_manipulation_actions.action import PlanBimanualCartesianTrajectory
from tf2_ros import Buffer, TransformException, TransformListener

from .marker_geometry import AXIS_NAMES, axis_vector, facing_measure
from .planning_markers import PlanningMarkerEditor, pose_from_transform


DEFAULT_ACTION = "/cirtesub/manipulation/plan_bimanual_cartesian_trajectory"
DEFAULT_SERVER = "/plan_bimanual_cartesian_trajectory_lifecycle_action_server"


def numeric_vector(text, size, label):
    try:
        values = [float(value.strip()) for value in text.split(",")]
    except ValueError as error:
        raise ValueError(f"{label} must contain comma-separated numbers") from error
    if len(values) != size or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} must contain exactly {size} finite numbers")
    return values


def arm_frames(config_filename="bimanual_cartesian_planner.yaml"):
    """Use the installed example as defaults; all identifiers remain editable."""
    fallback = {
        "alpha_left": "cirtesub/alpha_left/standard_jaws_tool",
        "alpha_right": "cirtesub/alpha_right/standard_jaws_tool",
    }
    try:
        import yaml

        path = os.path.join(get_package_share_directory("sura_manipulation_actions"),
                            "config", config_filename)
        with open(path, encoding="utf-8") as stream:
            parameters = yaml.safe_load(stream)["/**"]["ros__parameters"]
        namespace = parameters.get("robot_namespace", "cirtesub")
        if not isinstance(namespace, str) or not isinstance(parameters["arm_names"], list):
            return fallback
        namespace = namespace.strip("/")
        frames = {}
        for name in parameters["arm_names"]:
            frame = parameters.get(f"arms.{name}.tip_frame", "")
            if not isinstance(name, str) or not name or not isinstance(frame, str) or not frame:
                continue
            if not namespace:
                frame = frame.replace("${robot_namespace}/", "")
            frames[name] = frame.replace("${robot_namespace}", namespace)
        return frames or fallback
    except Exception:
        return fallback


def configured_reference_frame(config_filename="bimanual_cartesian_planner.yaml", parameter="relative_reference_frame"):
    try:
        import yaml

        path = os.path.join(get_package_share_directory("sura_manipulation_actions"),
                            "config", config_filename)
        with open(path, encoding="utf-8") as stream:
            parameters = yaml.safe_load(stream)["/**"]["ros__parameters"]
        namespace = parameters.get("robot_namespace", "cirtesub").strip("/")
        frame = parameters.get(parameter, "${robot_namespace}/base_link")
        if not namespace:
            frame = frame.replace("${robot_namespace}/", "")
        return frame.replace("${robot_namespace}", namespace)
    except Exception:
        return "cirtesub/base_link"


def relative_reference_frame():
    return configured_reference_frame()


def roll_value(text):
    """Empty or 'nan' keeps the measured roll."""
    text = text.strip()
    if not text or text.lower() == "nan":
        return float("nan")
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError("Roll must be a number, or empty to keep the measured roll") from error
    if not math.isfinite(value):
        raise ValueError("Roll must be finite, or empty to keep the measured roll")
    return value


class BimanualCartesianTab(QWidget):
    _action_type = PlanBimanualCartesianTrajectory

    def _initialize_client(self, node):
        self._node = node
        self._client = None
        self._client_name = None
        self._goal_handle = None
        self._send_future = None
        self._cancel_future = None
        self._execute_client = None
        self._execute_future = None
        self._cancel_requested = False
        self._cancel_acknowledged = False
        self._busy = False
        self._waiting = False
        self._closed = False
        self._generation = 0
        self._tf_buffer = Buffer(node=node)
        self._tf_listener = TransformListener(self._tf_buffer, node, spin_thread=False)

    def __init__(self, node):
        super().__init__()
        self._initialize_client(node)
        layout = QVBoxLayout(self)
        self.form_scroll = QScrollArea()
        self.form_scroll.setWidgetResizable(True)
        self.form_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        content = QWidget()
        form = QFormLayout(content)
        self.form_scroll.setWidget(content)
        layout.addWidget(self.form_scroll, 1)
        note = QLabel(
            "End-effectors facing: each tool axis points at the other tool, with the requested "
            "relative roll around the line between them and distance. Only the arm joints move. "
            "Angles are radians; zero tolerances remain strict; zero speed factors use server "
            "defaults. Left/right arms and planning.weights are configured in the planner YAML. "
            "Backend activation is manual."
        )
        note.setWordWrap(True)
        form.addRow(note)
        server_note = QLabel(f"Lifecycle server: {DEFAULT_SERVER}")
        server_note.setWordWrap(True)
        form.addRow(server_note)

        self.action_edit = QLineEdit(DEFAULT_ACTION)
        self.execute_service_edit = QLineEdit(DEFAULT_SERVER + "/execute_pending_trajectory")
        for edit in (self.action_edit, self.execute_service_edit):
            edit.setCursorPosition(0)
            edit.setToolTip(edit.text())
            edit.textChanged.connect(edit.setToolTip)
        frames = arm_frames()
        names = list(frames)[:2] if len(frames) > 1 else ["alpha_left", "alpha_right"]
        self.arm_names = names
        self.left_tip_edit = QLineEdit(frames.get(names[0], ""))
        self.right_tip_edit = QLineEdit(frames.get(names[1], ""))
        self.reference_frame_edit = QLineEdit(relative_reference_frame())
        self.reference_frame_edit.setToolTip("Frame used to read both tools and draw the RViz target")
        self.left_tip_edit.setToolTip("TF tool link; must match the left tip configured in the planner")
        self.right_tip_edit.setToolTip("TF tool link; must match the right tip configured in the planner")
        self.left_axis_combo, self.right_axis_combo = QComboBox(), QComboBox()
        for combo, side in ((self.left_axis_combo, "left"), (self.right_axis_combo, "right")):
            combo.addItems(list(AXIS_NAMES))
            combo.setCurrentText("+z" if side == "left" else "-z")
            combo.setToolTip(f"Axis of the {side} tool, in its own frame, that points at the other tool")
        self.roll_edit = QLineEdit(f"{math.pi:.12g}")  # 180 degrees
        self.roll_edit.setPlaceholderText("keep measured roll")
        self.roll_edit.setToolTip("Relative roll around the left->right line [rad]; empty keeps the measured one")
        self.distance_spin = self._spin(0.3, 100.0, 0.01)
        self.distance_spin.setToolTip("Distance between both tools [m]")
        self.read_pose_button = QPushButton("Read Current Facing (TF)")
        self.read_pose_button.clicked.connect(self._read_pose)
        self.pointing_tolerance_spin = self._spin(0.05, math.pi, 0.01)
        self.roll_tolerance_spin = self._spin(0.05, math.pi, 0.01)
        self.distance_tolerance_spin = self._spin(0.005, 100.0, 0.001)
        self.step_spin = self._spin(0.005, 1.0, 0.001)
        self.step_spin.setToolTip("Metres. Zero selects the planner's configured default")
        self.planning_time_spin = self._spin(10.0, 3600.0, 1.0)
        self.planning_time_spin.setDecimals(2)
        self.planning_time_spin.setToolTip("Seconds. Zero selects the configured default")
        self.velocity_scaling_spin = self._spin(0.0, 1.0, 0.01)
        self.acceleration_scaling_spin = self._spin(0.0, 1.0, 0.01)
        for spin in (self.velocity_scaling_spin, self.acceleration_scaling_spin):
            spin.setDecimals(3)
            spin.setSpecialValueText("Server default")
            spin.setToolTip("Joint limit factor: 0.1 = 10%, 0.05 = 5%. Zero uses the server default")
        self.velocity_scaling_spin.setValue(0.1)
        self.acceleration_scaling_spin.setValue(0.05)
        # The minimum is the "server default" entry; every value from 0 on is sent as is.
        self.orientation_fraction_spin = QDoubleSpinBox()
        self.orientation_fraction_spin.setDecimals(2)
        self.orientation_fraction_spin.setRange(-0.05, 0.99)
        self.orientation_fraction_spin.setSingleStep(0.05)
        self.orientation_fraction_spin.setSpecialValueText("Server default")
        self.orientation_fraction_spin.setValue(0.4)
        self.orientation_fraction_spin.setToolTip(
            "Share of the trajectory used to correct the pointing axes and the roll before the "
            "distance changes: 0.4 = the first 40%. Zero changes everything simultaneously")
        self.backend_combo = QComboBox()
        for label, value in (("Server default", ""), ("Controller", "controller"),
                             ("Task", "task"), ("Facing", "facing")):
            self.backend_combo.addItem(label, value)
        self.backend_combo.setToolTip(
            "Facing streams the facing reference to EndEffectors_Facing and the arm joints "
            "to robot_nominal (base masked); Server default uses the planner's execution_backend parameter")
        self.backend_combo.setCurrentIndex(self.backend_combo.findData("facing"))
        self.execute_checkbox = QCheckBox("Execute after planning")
        self.execute_checkbox.setChecked(False)
        self.optimize_checkbox = QCheckBox("Optimize whole trajectory (robot_nominal)")
        self.optimize_checkbox.setToolTip(
            "Plan the whole facing trajectory with the controller's Task Priority hierarchy "
            "(the same global planner as the single-arm mode). Requires the Facing backend; "
            "tracking tolerances come from the planner's execution.facing_* parameters.")
        self.stpik_checkbox = QCheckBox("Use STPIK")
        self.stpik_checkbox.setChecked(True)
        self.stpik_checkbox.setToolTip(
            "Facing execution: simulate the facing hierarchy with STPIK before streaming. "
            "Unchecked skips offline validation; execution still monitors tracking.")
        self.backend_combo.currentIndexChanged.connect(lambda _index: self._update_controls())

        form.addRow("Action", self.action_edit)
        form.addRow("Execute Pending Service", self.execute_service_edit)
        form.addRow("Left Tool Frame (TF)", self.left_tip_edit)
        form.addRow("Right Tool Frame (TF)", self.right_tip_edit)
        form.addRow("Reference Frame", self.reference_frame_edit)
        form.addRow("Left Pointing Axis", self.left_axis_combo)
        form.addRow("Right Pointing Axis", self.right_axis_combo)
        form.addRow("Roll [rad]", self.roll_edit)
        form.addRow("Distance [m]", self.distance_spin)
        form.addRow("Pointing Tolerance [rad]", self.pointing_tolerance_spin)
        form.addRow("Roll Tolerance [rad]", self.roll_tolerance_spin)
        form.addRow("Distance Tolerance [m]", self.distance_tolerance_spin)
        form.addRow("Orientation Phase [0..1)", self.orientation_fraction_spin)
        form.addRow("Cartesian Step [m]", self.step_spin)
        form.addRow("Planning Time [s]", self.planning_time_spin)
        form.addRow("Velocity Scaling [0..1]", self.velocity_scaling_spin)
        form.addRow("Acceleration Scaling [0..1]", self.acceleration_scaling_spin)
        form.addRow("Execution Backend", self.backend_combo)
        form.addRow(self.optimize_checkbox)
        form.addRow(self.stpik_checkbox)
        form.addRow(self.execute_checkbox)

        buttons = QHBoxLayout()
        buttons.addWidget(self.read_pose_button)
        self.send_button = QPushButton("Send Goal")
        self.send_button.clicked.connect(self._send_goal)
        self.execute_button = QPushButton("Execute Pending")
        self.execute_button.clicked.connect(self._execute_pending)
        self.cancel_button = QPushButton("Cancel Goal")
        self.cancel_button.clicked.connect(self._cancel_goal)
        for button in (self.send_button, self.execute_button, self.cancel_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        footer = QFormLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.feedback_label = QLabel("Read the current facing or enter a target, then send a goal")
        self.result_label = QLabel("")
        self.feedback_label.setWordWrap(True)
        self.result_label.setWordWrap(True)
        footer.addRow("Progress", self.progress)
        footer.addRow("Feedback", self.feedback_label)
        footer.addRow("Result", self.result_label)
        self._configure_diagnostic_copy(footer)
        layout.addLayout(footer)
        self._inputs = [
            self.action_edit, self.execute_service_edit, self.left_tip_edit, self.right_tip_edit,
            self.reference_frame_edit, self.left_axis_combo, self.right_axis_combo, self.roll_edit,
            self.distance_spin, self.pointing_tolerance_spin, self.roll_tolerance_spin,
            self.distance_tolerance_spin, self.read_pose_button, self.step_spin,
            self.planning_time_spin, self.velocity_scaling_spin, self.acceleration_scaling_spin,
            self.backend_combo, self.execute_checkbox, self.optimize_checkbox,
            self.orientation_fraction_spin,
        ]
        self._markers = PlanningMarkerEditor(self, form, bimanual=True)
        self._update_controls()

    @staticmethod
    def _spin(value, maximum, step):
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(0.0, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _configure_diagnostic_copy(self, footer):
        for name in ("proposal_label", "feedback_label", "errors_label", "result_label"):
            label = getattr(self, name, None)
            if label is not None:
                label.setTextFormat(Qt.PlainText)
                label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
                label.setFocusPolicy(Qt.StrongFocus)
        self.progress.setFocusPolicy(Qt.StrongFocus)
        self.progress.setContextMenuPolicy(Qt.ActionsContextMenu)
        self.copy_progress_action = QAction("Copy progress", self.progress)
        self.copy_progress_action.setShortcut(QKeySequence.Copy)
        self.copy_progress_action.setShortcutContext(Qt.WidgetShortcut)
        self.copy_progress_action.triggered.connect(
            lambda: QApplication.clipboard().setText(self.progress.text()))
        self.progress.addAction(self.copy_progress_action)
        self.copy_diagnostics_button = QPushButton("Copy diagnostics")
        self.copy_diagnostics_button.clicked.connect(self._copy_diagnostics)
        footer.addRow(self.copy_diagnostics_button)

    def _copy_diagnostics(self):
        lines = [f"Progress: {self.progress.text()}"]
        for title, name in (("Proposed posture", "proposal_label"), ("Feedback", "feedback_label"),
                            ("Tracking Errors", "errors_label"), ("Result", "result_label")):
            label = getattr(self, name, None)
            if label is not None:
                lines.append(f"{title}: {label.text()}")
        QApplication.clipboard().setText("\n".join(lines))

    def _vector_row(self, value, maximum, labels=("X", "Y", "Z")):
        row = QHBoxLayout()
        spins = []
        for label in labels:
            row.addWidget(QLabel(label))
            spin = self._spin(value, maximum, 0.001)
            spins.append(spin)
            row.addWidget(spin)
        return spins, row

    def _update_controls(self):
        if self._closed:
            return
        editable = not self._busy and self._execute_future is None
        for widget in self._inputs:
            widget.setEnabled(editable)
        self.send_button.setEnabled(editable)
        if hasattr(self, "stpik_checkbox"):
            self.stpik_checkbox.setEnabled(editable and self.backend_combo.currentData() in ("", "facing"))
        self.execute_button.setEnabled(
            self._waiting and self._goal_handle is not None and
            self._execute_future is None and not self._cancel_requested and not self._closed)
        self.cancel_button.setEnabled(self._busy and not self._cancel_requested and not self._closed)
        if hasattr(self, "_markers"):
            self._markers.set_editable(editable)

    def _error(self, message):
        if not self._closed:
            self.result_label.setText(f"Error: {message}")

    def facing_fields(self):
        return (self.left_axis_combo.currentText(), self.right_axis_combo.currentText(),
                roll_value(self.roll_edit.text()), self.distance_spin.value())

    def set_facing_fields(self, left, right, roll, distance):
        widgets = (self.left_axis_combo, self.right_axis_combo, self.roll_edit, self.distance_spin)
        blocked = [widget.blockSignals(True) for widget in widgets]
        self.left_axis_combo.setCurrentText(left)
        self.right_axis_combo.setCurrentText(right)
        self.roll_edit.setText(f"{roll:.12g}" if math.isfinite(roll) else "")
        self.distance_spin.setValue(distance)
        for widget, previous in zip(widgets, blocked):
            widget.blockSignals(previous)

    def _read_pose(self):
        if self._markers.read_pose():
            return
        reference = self.reference_frame_edit.text().strip()
        left = self.left_tip_edit.text().strip()
        right = self.right_tip_edit.text().strip()
        if not reference or not left or not right or left == right:
            self._error("Enter a reference frame and distinct left/right tool frames")
            return
        try:
            left_tf = self._tf_buffer.lookup_transform(reference, left, Time())
            right_tf = self._tf_buffer.lookup_transform(
                reference, right, Time.from_msg(left_tf.header.stamp))
            poses = [pose_from_transform(left_tf.transform), pose_from_transform(right_tf.transform)]
            axes = (self.left_axis_combo.currentText(), self.right_axis_combo.currentText())
            distance, roll = facing_measure(*poses, *axes)
        except (TransformException, ValueError) as error:
            self._error(f"Facing TF is unavailable: {error}")
            return
        self.set_facing_fields(*axes, float("nan") if roll is None else roll, distance)
        self._markers.set_poses(poses, reference)
        self.result_label.setText("Current distance and roll read; edit them to set the target")

    def _goal_from_fields(self):
        left, right, roll, distance = self.facing_fields()
        if distance <= 0.0:
            raise ValueError("Distance must be positive")
        goal = PlanBimanualCartesianTrajectory.Goal()
        for vector, name in ((goal.left_pointing_axis, left), (goal.right_pointing_axis, right)):
            vector.x, vector.y, vector.z = axis_vector(name)
        goal.roll_angle, goal.distance = roll, distance
        goal.pointing_tolerance = self.pointing_tolerance_spin.value()
        goal.roll_tolerance = self.roll_tolerance_spin.value()
        goal.distance_tolerance = self.distance_tolerance_spin.value()
        fraction = self.orientation_fraction_spin.value()
        goal.facing_orientation_fraction = fraction if fraction >= 0.0 else -1.0
        goal.cartesian_step = self.step_spin.value()
        goal.planning_time = self.planning_time_spin.value()
        goal.velocity_scaling = self.velocity_scaling_spin.value()
        goal.acceleration_scaling = self.acceleration_scaling_spin.value()
        goal.execution_backend = self.backend_combo.currentData() or ""
        goal.use_stpik = self.stpik_checkbox.isChecked()
        goal.execute = self.execute_checkbox.isChecked()
        goal.optimize_trajectory_nominal = self.optimize_checkbox.isChecked()
        if goal.optimize_trajectory_nominal and goal.execution_backend not in ("", "facing"):
            raise ValueError("Optimize whole trajectory requires the Facing backend")
        return goal

    def _send_goal(self):
        if self._busy or self._closed or self._execute_future is not None:
            return
        try:
            goal = self._goal_from_fields()
            self._markers.commit_fields()
            name = self.action_edit.text().strip()
            if not name:
                raise ValueError("Enter the Cartesian action endpoint")
            if self._client_name != name:
                if self._client is not None:
                    self._client.destroy()
                self._client = None
                self._client_name = None
                self._client = ActionClient(self._node, self._action_type, name)
                self._client_name = name
            if (not self._client.server_is_ready() and
                    not self._client.wait_for_server(timeout_sec=0.5)):
                raise ValueError(
                    f"Action server unavailable: {name}. Configure/activate the Cartesian lifecycle server.")
            self._generation += 1
            generation = self._generation
            self._busy = True
            self._waiting = self._cancel_requested = False
            self._cancel_acknowledged = False
            self._cancel_future = None
            self.progress.setValue(0)
            self.feedback_label.setText("Sending Cartesian goal...")
            self.result_label.clear()
            self._update_controls()
            self._markers.prepare_goal(goal,
                lambda target: self._dispatch_goal(target, generation),
                lambda error: self._target_preparation_failed(error, generation))
        except Exception as error:
            self._busy = False
            self._error(str(error))
        self._update_controls()

    def _target_preparation_failed(self, error, generation):
        if generation != self._generation:
            return
        self._busy = self._waiting = False
        self._error(error)
        self._update_controls()

    def _dispatch_goal(self, goal, generation):
        if generation != self._generation or self._closed or self._cancel_requested:
            self._busy = False
            self._cancel_acknowledged = True
            self._update_controls()
            return
        try:
            self._send_future = self._client.send_goal_async(
                goal, feedback_callback=lambda msg: self._feedback(msg, generation))
            self._send_future.add_done_callback(lambda future: self._accepted(future, generation))
        except Exception as error:
            self._target_preparation_failed(str(error), generation)

    def _accepted(self, future, generation):
        if generation != self._generation:
            return
        self._send_future = None
        try:
            handle = future.result()
            if handle is None or not handle.accepted:
                self._busy = False
                self._error("Cartesian goal rejected")
                return
            self._goal_handle = handle
            self._markers.bind_goal(handle)
            handle.get_result_async().add_done_callback(
                lambda result: self._result(result, generation))
            if self._cancel_requested or self._closed:
                self._cancel_goal()
            elif self._busy and not self._waiting:
                self.feedback_label.setText("Goal accepted; planning")
        except Exception as error:
            self._busy = False
            self._error(str(error))
        finally:
            self._update_controls()

    def _feedback(self, message, generation):
        if self._closed or generation != self._generation or not self._busy:
            return
        feedback = message.feedback
        self._waiting = feedback.current_stage == "waiting_for_execute"
        self.progress.setValue(round(max(0.0, min(1.0, feedback.progress)) * 100))
        self.feedback_label.setText(f"{feedback.current_stage}: {feedback.message}")
        self._update_controls()

    def _result(self, future, generation):
        if generation != self._generation:
            return
        self._busy = self._waiting = False
        self._goal_handle = None
        try:
            wrapped = future.result()
            result = wrapped.result
            if not self._closed:
                trajectory = result.joint_trajectory
                duration = 0.0
                if trajectory.points:
                    end = trajectory.points[-1].time_from_start
                    duration = end.sec + end.nanosec * 1e-9
                if wrapped.status == GoalStatus.STATUS_CANCELED:
                    status = "canceled"
                elif wrapped.status == GoalStatus.STATUS_SUCCEEDED and result.success:
                    status = "success"
                else:
                    status = "failed"
                self.result_label.setText(
                    f"{status}: {result.message} | {len(trajectory.joint_names)} joints, "
                    f"{len(trajectory.points)} points, {duration:.3f} s")
                self.feedback_label.setText(f"{status}: {result.message}")
                if status == "success":
                    self.progress.setValue(100)
        except Exception as error:
            self._error(str(error))
        self._update_controls()

    def _cancel_goal(self):
        if not self._busy:
            return
        self._cancel_requested = True
        if not self._closed:
            self.feedback_label.setText("Cancel requested")
        if self._goal_handle is not None and self._cancel_future is None:
            try:
                self._cancel_future = self._goal_handle.cancel_goal_async()
                generation = self._generation
                self._cancel_future.add_done_callback(
                    lambda future: self._canceled(future, generation))
            except Exception as error:
                self._cancel_requested = False
                self._error(str(error))
        self._update_controls()

    def _canceled(self, future, generation):
        if generation != self._generation:
            return
        self._cancel_future = None
        try:
            self._cancel_acknowledged = bool(future.result().goals_canceling)
            if not self._cancel_acknowledged and self._busy:
                self._cancel_requested = False
                self._error("Server did not accept cancellation; the goal remains active")
        except Exception as error:
            self._cancel_requested = False
            self._error(str(error))
        self._update_controls()

    def _execute_pending(self):
        if (self._closed or not self._waiting or self._goal_handle is None or
                self._cancel_requested or self._execute_future is not None):
            return
        name = self.execute_service_edit.text().strip()
        try:
            if not name:
                raise ValueError("Enter the Execute Pending service endpoint")
            self._execute_client = self._node.create_client(Trigger, name)
            if (not self._execute_client.service_is_ready() and
                    not self._execute_client.wait_for_service(timeout_sec=0.5)):
                raise ValueError(f"Execute Pending service unavailable: {name}")
            generation = self._generation
            self._execute_future = self._execute_client.call_async(Trigger.Request())
            self._execute_future.add_done_callback(
                lambda future: self._executed(future, generation))
        except Exception as error:
            self._clear_execute_client()
            self._error(str(error))
        self._update_controls()

    def _executed(self, future, generation):
        try:
            response = future.result()
            if generation == self._generation and not self._closed and self._busy:
                status = "Execution requested" if response.success else "Execution request rejected"
                self.result_label.setText(f"{status}: {response.message}")
        except Exception as error:
            if generation == self._generation:
                self._error(str(error))
        finally:
            self._clear_execute_client()
            self._update_controls()

    def _clear_execute_client(self):
        self._execute_future = None
        if self._execute_client is not None:
            self._node.destroy_client(self._execute_client)
            self._execute_client = None

    def request_shutdown(self):
        self._closed = True
        self._markers.close()
        self._cancel_goal()
        self._update_controls()

    def shutdown_complete(self):
        return not self._busy or self._cancel_acknowledged

    def destroy_clients(self):
        self._markers.close()
        self._tf_listener.unregister()
        self._clear_execute_client()
        if self._client is not None:
            self._client.destroy()
            self._client = None
