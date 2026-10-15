"""Manual client for the bimanual Cartesian planner, sharing the panel's ROS node."""

import math
import os

from ament_index_python.packages import get_package_share_directory
from action_msgs.msg import GoalStatus
from python_qt_binding.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)
from rclpy.action import ActionClient
from rclpy.time import Time
from std_srvs.srv import Trigger
from sura_manipulation_actions.action import PlanBimanualCartesianTrajectory
from tf2_ros import Buffer, TransformException, TransformListener

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


def relative_pose_components(arm_a, arm_b):
    """Both TF transforms share reference axes: dp=pB-pA, dq=qB*conjugate(qA)."""
    position = [getattr(arm_b.translation, axis) - getattr(arm_a.translation, axis)
                for axis in ("x", "y", "z")]
    quaternions = []
    for transform in (arm_a, arm_b):
        values = [getattr(transform.rotation, axis) for axis in ("x", "y", "z", "w")]
        norm = math.hypot(*values)
        if not math.isfinite(norm) or norm < 1e-12:
            raise ValueError("TF orientation must have a finite, nonzero norm")
        quaternions.append([value / norm for value in values])
    lx, ly, lz, lw = quaternions[0]
    fx, fy, fz, fw = quaternions[1]
    quaternion = [fx * lw - fw * lx - fy * lz + fz * ly,
                  fy * lw - fw * ly - fz * lx + fx * lz,
                  fz * lw - fw * lz - fx * ly + fy * lx,
                  fw * lw + fx * lx + fy * ly + fz * lz]
    if quaternion[3] < 0.0:
        quaternion = [-value for value in quaternion]
    if not all(math.isfinite(value) for value in position):
        raise ValueError("TF position must be finite")
    return position, quaternion


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
            "Relative target: B minus A position, and B/A "
            "rotation difference, expressed in reference (base) axes. "
            "Positions are metres; angular constraints are radians. "
            "Zero relative tolerances remain strict; zero speed factors use server defaults. "
            "Joint movement weights are configured with planning.weights in the planner YAML. "
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
        self.arm_a_combo = QComboBox()
        self.arm_b_combo = QComboBox()
        self._arm_frames = arm_frames()
        for combo in (self.arm_a_combo, self.arm_b_combo):
            combo.setEditable(True)
            combo.addItems(list(self._arm_frames))
        if self.arm_b_combo.count() > 1:
            self.arm_b_combo.setCurrentIndex(1)
        self.arm_a_tip_edit = QLineEdit(self._arm_frames[self.arm_a_combo.currentText()])
        self.arm_b_tip_edit = QLineEdit(self._arm_frames[self.arm_b_combo.currentText()])
        self.reference_frame_edit = QLineEdit(relative_reference_frame())
        self.reference_frame_edit.setToolTip("Must match relative_reference_frame configured in the planner")
        self.arm_a_tip_edit.setToolTip("TF tool link; must match arm A tip configured in the planner")
        self.arm_b_tip_edit.setToolTip("TF tool link; must match arm B tip configured in the planner")
        self.arm_a_combo.currentTextChanged.connect(
            lambda name: self.arm_a_tip_edit.setText(self._arm_frames.get(name.strip(), "")))
        self.arm_b_combo.currentTextChanged.connect(
            lambda name: self.arm_b_tip_edit.setText(self._arm_frames.get(name.strip(), "")))

        self.xyz_edit = QLineEdit("0.0,0.0,0.0")
        self.xyz_edit.setToolTip("Comma-separated B-minus-A X,Y,Z in reference axes, in metres")
        self.quaternion_edit = QLineEdit("0.0,0.0,0.0,1.0")
        self.quaternion_edit.setToolTip("Relative rotation R_arm_b * R_arm_a^T in reference axes; quaternion XYZW")
        self.read_pose_button = QPushButton("Read Relative Pose (TF)")
        self.read_pose_button.clicked.connect(self._read_pose)
        self.position_tolerances, position_row = self._vector_row(0.002, 100.0)
        self.orientation_tolerances, orientation_row = self._vector_row(0.03, math.pi)
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
        self.execute_checkbox = QCheckBox("Execute after planning")
        self.execute_checkbox.setChecked(False)

        form.addRow("Action", self.action_edit)
        form.addRow("Execute Pending Service", self.execute_service_edit)
        form.addRow("Arm A", self.arm_a_combo)
        form.addRow("Arm B", self.arm_b_combo)
        form.addRow("Relative Reference Frame", self.reference_frame_edit)
        form.addRow("Arm A Tool Frame", self.arm_a_tip_edit)
        form.addRow("Arm B Tool Frame (TF)", self.arm_b_tip_edit)
        form.addRow("Relative XYZ [m]", self.xyz_edit)
        form.addRow("Relative Quaternion XYZW", self.quaternion_edit)
        form.addRow("Relative Position Tolerances [m]", position_row)
        form.addRow("Relative Orientation Tolerances [rad]", orientation_row)
        form.addRow("Cartesian Step [m]", self.step_spin)
        form.addRow("Planning Time [s]", self.planning_time_spin)
        form.addRow("Velocity Scaling [0..1]", self.velocity_scaling_spin)
        form.addRow("Acceleration Scaling [0..1]", self.acceleration_scaling_spin)
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
        self.feedback_label = QLabel("Read the current pose or enter a relative target, then send a goal")
        self.result_label = QLabel("")
        self.feedback_label.setWordWrap(True)
        self.result_label.setWordWrap(True)
        footer.addRow("Progress", self.progress)
        footer.addRow("Feedback", self.feedback_label)
        footer.addRow("Result", self.result_label)
        layout.addLayout(footer)
        self._inputs = [
            self.action_edit, self.execute_service_edit, self.arm_a_combo, self.arm_b_combo,
            self.arm_a_tip_edit, self.arm_b_tip_edit, self.xyz_edit, self.quaternion_edit,
            self.reference_frame_edit,
            self.read_pose_button, self.step_spin, self.planning_time_spin,
            self.velocity_scaling_spin, self.acceleration_scaling_spin,
            self.execute_checkbox, *self.position_tolerances,
            *self.orientation_tolerances,
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
        self.execute_button.setEnabled(
            self._waiting and self._goal_handle is not None and
            self._execute_future is None and not self._cancel_requested and not self._closed)
        self.cancel_button.setEnabled(self._busy and not self._cancel_requested and not self._closed)
        if hasattr(self, "_markers"):
            self._markers.set_editable(editable)

    def _error(self, message):
        if not self._closed:
            self.result_label.setText(f"Error: {message}")

    def _read_pose(self):
        if self._markers.read_pose():
            return
        reference = self.reference_frame_edit.text().strip()
        arm_a = self.arm_a_tip_edit.text().strip()
        arm_b = self.arm_b_tip_edit.text().strip()
        if not reference or not arm_a or not arm_b or arm_a == arm_b or reference in (arm_a, arm_b):
            self._error("Enter a reference frame and distinct arm A/B tool frames")
            return
        try:
            arm_a_tf = self._tf_buffer.lookup_transform(reference, arm_a, Time())
            arm_b_tf = self._tf_buffer.lookup_transform(
                reference, arm_b, Time.from_msg(arm_a_tf.header.stamp))
            position, quaternion = relative_pose_components(arm_a_tf.transform, arm_b_tf.transform)
        except (TransformException, ValueError) as error:
            self._error(f"Relative TF is unavailable: {error}")
            return
        self.xyz_edit.setText(",".join(f"{value:.12g}" for value in position))
        self.quaternion_edit.setText(",".join(f"{value:.12g}" for value in quaternion))
        self._markers.set_poses([pose_from_transform(arm_a_tf.transform),
                                pose_from_transform(arm_b_tf.transform)], reference)
        self.result_label.setText(f"Current relative pose read in {reference}; edit XYZ to set an offset")

    def _goal_from_fields(self):
        arm_a = self.arm_a_combo.currentText().strip()
        arm_b = self.arm_b_combo.currentText().strip()
        frame = self.reference_frame_edit.text().strip()
        if not arm_a or not arm_b or arm_a == arm_b:
            raise ValueError("Arm A and arm B must be distinct configured arm names")
        if not frame:
            raise ValueError("Enter the configured relative reference frame")
        xyz = numeric_vector(self.xyz_edit.text(), 3, "Relative XYZ")
        quaternion = numeric_vector(self.quaternion_edit.text(), 4, "Quaternion XYZW")
        norm = math.hypot(*quaternion)
        if norm < 1e-12 or not math.isfinite(norm):
            raise ValueError("Quaternion must have a finite, nonzero norm")
        goal = PlanBimanualCartesianTrajectory.Goal()
        goal.arm_a, goal.arm_b = arm_a, arm_b
        goal.relative_target.header.frame_id = frame
        goal.relative_target.header.stamp = self._node.get_clock().now().to_msg()
        for axis, value in zip(("x", "y", "z"), xyz):
            setattr(goal.relative_target.pose.position, axis, value)
        for axis, value in zip(("x", "y", "z", "w"), quaternion):
            setattr(goal.relative_target.pose.orientation, axis, value / norm)
        for vector, spins in (
            (goal.relative_position_tolerance, self.position_tolerances),
            (goal.relative_orientation_tolerance, self.orientation_tolerances),
        ):
            for axis, spin in zip(("x", "y", "z"), spins):
                setattr(vector, axis, spin.value())
        goal.cartesian_step = self.step_spin.value()
        goal.planning_time = self.planning_time_spin.value()
        goal.velocity_scaling = self.velocity_scaling_spin.value()
        goal.acceleration_scaling = self.acceleration_scaling_spin.value()
        goal.execute = self.execute_checkbox.isChecked()
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
