"""Cartesian vehicle/arm goals with per-goal mobility and tracking controls."""
import math

from python_qt_binding.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.time import Time
from std_srvs.srv import Trigger
from sura_manipulation_actions.action import PlanCartesianTrajectory

from .bimanual_cartesian_tab import (
    BimanualCartesianTab, arm_frames, numeric_vector, TransformException,
)
from .planning_markers import PlanningMarkerEditor, pose_components, pose_from_components, transform_pose

DEFAULT_ACTION = "/cirtesub/manipulation/plan_cartesian_trajectory"
DEFAULT_SERVER = "/plan_cartesian_trajectory_lifecycle_action_server"
DEFAULT_RELEASE_SERVICE = "/cirtesub/controller/task_priority/release_cartesian_trajectory"
BASE_AXES = ("X", "Y", "Z", "Roll", "Pitch", "Yaw")


def transformed_pose(position, quaternion, transform):
    """Express a pose in the target frame of a TF transform."""
    return pose_components(transform_pose(pose_from_components(position, quaternion), transform))


class CartesianTab(BimanualCartesianTab):
    _action_type = PlanCartesianTrajectory

    def __init__(self, node):
        QWidget.__init__(self)
        self._initialize_client(node)
        self._joint_names, self._joint_widgets, self._joint_choices = [], [], {}
        self._catalog_arm, self._catalog_client, self._catalog_future = None, None, None
        self._catalog_generation = 0
        self._release_client, self._release_future = None, None
        layout = QVBoxLayout(self)
        self.form_scroll = QScrollArea()
        self.form_scroll.setWidgetResizable(True)
        self.form_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        content = QWidget()
        form = QFormLayout(content)
        self.form_scroll.setWidget(content)
        layout.addWidget(self.form_scroll, 1)
        note = QLabel("Absolute tool target in any TF frame. Changing the reference frame preserves "
                      "the physical target. Pose streams tool targets to the configured task hierarchy. "
                      "For joint/base planning, base axes use world NED and unchecked DOFs stay fixed. "
                      "Backend activation is manual.")
        note.setWordWrap(True)
        form.addRow(note)
        self.action_edit = QLineEdit(DEFAULT_ACTION)
        self.execute_service_edit = QLineEdit(DEFAULT_SERVER + "/execute_pending_trajectory")
        self.release_service_edit = QLineEdit(DEFAULT_RELEASE_SERVICE)
        for edit in (self.action_edit, self.execute_service_edit, self.release_service_edit):
            edit.setCursorPosition(0)
            edit.setToolTip(edit.text())
            edit.textChanged.connect(edit.setToolTip)
        self.execute_service_edit.editingFinished.connect(self._refresh_joint_catalog)
        self._arm_frames = arm_frames("single_arm_cartesian_planner.yaml")
        self.arm_combo = QComboBox()
        self.arm_combo.setEditable(True)
        self.arm_combo.addItems(list(self._arm_frames))
        self.tool_frame_edit = QLineEdit(self._arm_frames.get(self.arm_combo.currentText(), ""))
        self.arm_combo.currentTextChanged.connect(self._arm_changed)
        self.reference_frame_edit = QLineEdit("world_ned")
        self._target_frame = "world_ned"
        self.reference_frame_edit.setToolTip("Any TF frame; changing it transforms the current target")
        self.reference_frame_edit.editingFinished.connect(self._change_reference_frame)
        self.tool_frame_edit.setToolTip("Used for TF reading; must match the configured arm tip")
        self.backend_combo = QComboBox()
        for label, value in (("Server default", ""), ("Controller", "controller"),
                             ("Task", "task"), ("Pose", "pose")):
            self.backend_combo.addItem(label, value)
        self.backend_combo.currentIndexChanged.connect(self._base_dofs_changed)
        self.xyz_edit = QLineEdit("0.0,0.0,0.0")
        self.xyz_edit.setToolTip("Absolute X,Y,Z in metres in the selected target frame")
        self.quaternion_edit = QLineEdit("0.0,0.0,0.0,1.0")
        self.quaternion_edit.setToolTip("Absolute tool orientation in target frame; quaternion XYZW")
        self.position_tolerances, position_row = self._vector_row(0.02, 100.0)
        self.orientation_tolerances, orientation_row = self._vector_row(0.2, math.pi)
        self.step_spin = self._spin(0.02, 1.0, 0.001)
        self.planning_time_spin = self._spin(10.0, 3600.0, 1.0)
        self.planning_time_spin.setDecimals(2)
        self.velocity_scaling_spin = self._spin(0.8, 1.0, 0.01)
        self.acceleration_scaling_spin = self._spin(0.4, 1.0, 0.01)
        for spin in (self.velocity_scaling_spin, self.acceleration_scaling_spin):
            spin.setDecimals(3)
            spin.setSpecialValueText("Server default")
            spin.setToolTip("Timing factor: 0.1 = 10%; zero uses server default. "
                            "Pose scales Cartesian limits; other backends scale joint/base timing.")
        self.execute_checkbox = QCheckBox("Execute after planning")
        for label, widget in (
            ("Action", self.action_edit), ("Execute Pending Service", self.execute_service_edit),
            ("Release Trajectory Service", self.release_service_edit),
            ("Arm", self.arm_combo), ("Execution Backend", self.backend_combo),
            ("Reference Frame", self.reference_frame_edit), ("Tool Frame (TF)", self.tool_frame_edit),
            ("Target XYZ [m]", self.xyz_edit), ("Target Quaternion XYZW", self.quaternion_edit),
            ("Planning Position Tolerances [m]", position_row),
            ("Planning Orientation Tolerances [rad]", orientation_row),
            ("Cartesian Step [m]", self.step_spin), ("Planning Time [s]", self.planning_time_spin),
            ("Velocity Scaling [0..1]", self.velocity_scaling_spin),
            ("Acceleration Scaling [0..1]", self.acceleration_scaling_spin),
        ):
            form.addRow(label, widget)
        self.base_bounds_checkbox = QCheckBox("Override base position/orientation bounds")
        self.base_bounds_checkbox.setToolTip(
            "XYZ bounds: offsets from initial world position [m]; RPY bounds: absolute world angles [rad]")
        self.base_bounds_checkbox.toggled.connect(self._update_controls)
        form.addRow(self.base_bounds_checkbox)
        base_grid = QGridLayout()
        for col, label in enumerate(("Base DOF", "Lower", "Upper", "Max velocity", "Max acceleration",
                                     "Path tolerance", "Goal tolerance")):
            base_grid.addWidget(QLabel(label), 0, col)
        self.base_dof_checks = []
        self.base_lower_spins, self.base_upper_spins = [], []
        self.base_velocity_spins, self.base_acceleration_spins = [], []
        self.base_path_spins, self.base_goal_spins = [], []
        for row, axis in enumerate(BASE_AXES, 1):
            linear = row <= 3
            angle_bound = 1.56 if axis == "Pitch" else math.pi
            check = QCheckBox(axis)
            check.toggled.connect(self._base_dofs_changed)
            self.base_dof_checks.append(check)
            base_grid.addWidget(check, row, 0)
            for col, collection, value, signed in (
                (1, self.base_lower_spins, -1.0 if linear else -angle_bound, True),
                (2, self.base_upper_spins, 1.0 if linear else angle_bound, True),
                (3, self.base_velocity_spins, 1.0 if linear else 0.5, False),
                (4, self.base_acceleration_spins, 0.4 if linear else 0.2, False),
                (5, self.base_path_spins, 0.05 if linear else 0.3, False),
                (6, self.base_goal_spins, 0.05 if linear else 0.3, False),
            ):
                spin = self._spin(abs(value), 1e6, 0.01)
                if signed:
                    spin.setRange(-1e6, 1e6)
                    if axis == "Pitch":
                        spin.setRange(-1.56, 1.56)
                elif col in (3, 4):
                    spin.setMinimum(0.000001)
                spin.setValue(value)
                spin.setToolTip("Metres and seconds" if linear else "Radians and seconds; world RPY")
                collection.append(spin)
                base_grid.addWidget(spin, row, col)
        form.addRow(base_grid)
        self.joint_catalog_label = QLabel("Read the configured server's ordered joint names")
        self.joint_catalog_label.setWordWrap(True)
        self.refresh_joints_button = QPushButton("Refresh Arm Joints")
        self.refresh_joints_button.clicked.connect(self._refresh_joint_catalog)
        form.addRow(self.joint_catalog_label, self.refresh_joints_button)
        self.joint_grid = QGridLayout()
        form.addRow(self.joint_grid)
        self.tip_path_spins, tip_path_row = self._six_axis_row([0.02] * 3 + [0.3] * 3)
        self.tip_goal_spins, tip_goal_row = self._six_axis_row([0.02] * 3 + [0.3] * 3)
        form.addRow("Tool Path Tracking Tolerance [m, rad]", tip_path_row)
        form.addRow("Tool Goal Tracking Tolerance [m, rad]", tip_goal_row)
        self.goal_time_spin = self._spin(0.5, 3600.0, 0.1)
        form.addRow("Goal Time Tolerance [s]", self.goal_time_spin)
        form.addRow(self.execute_checkbox)
        self.read_pose_button = QPushButton("Read Current Pose (TF)")
        self.send_button = QPushButton("Send Goal")
        self.execute_button = QPushButton("Execute Pending")
        self.cancel_button = QPushButton("Cancel Goal")
        self.release_button = QPushButton("Release Trajectory Hold")
        self.release_button.setToolTip("Release trajectory ownership after completion/cancellation")
        buttons = QHBoxLayout()
        for button, callback in ((self.read_pose_button, self._read_pose),
                                 (self.send_button, self._send_goal),
                                 (self.execute_button, self._execute_pending),
                                 (self.cancel_button, self._cancel_goal),
                                 (self.release_button, self._release_hold)):
            button.clicked.connect(callback)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        footer = QFormLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.feedback_label = QLabel("Read the tool pose and refresh arm joints before sending a goal")
        self.result_label, self.errors_label = QLabel(""), QLabel("")
        for label in (self.feedback_label, self.result_label, self.errors_label):
            label.setWordWrap(True)
        footer.addRow("Progress", self.progress)
        footer.addRow("Feedback", self.feedback_label)
        footer.addRow("Tracking Errors", self.errors_label)
        footer.addRow("Result", self.result_label)
        layout.addLayout(footer)
        self._inputs = [self.action_edit, self.execute_service_edit, self.release_service_edit,
                        self.arm_combo, self.backend_combo, self.reference_frame_edit, self.tool_frame_edit,
                        self.xyz_edit, self.quaternion_edit, self.step_spin, self.planning_time_spin,
                        self.velocity_scaling_spin, self.acceleration_scaling_spin,
                        self.execute_checkbox, self.read_pose_button, self.base_bounds_checkbox,
                        self.refresh_joints_button, self.goal_time_spin,
                        *self.position_tolerances, *self.orientation_tolerances,
                        *self.base_dof_checks, *self.base_lower_spins, *self.base_upper_spins,
                        *self.base_velocity_spins, *self.base_acceleration_spins,
                        *self.base_path_spins, *self.base_goal_spins,
                        *self.tip_path_spins, *self.tip_goal_spins]
        self._markers = PlanningMarkerEditor(self, form)
        self._update_controls()
        self._refresh_joint_catalog()

    def _six_axis_row(self, values):
        row, spins = QHBoxLayout(), []
        for axis, value in zip(BASE_AXES, values):
            row.addWidget(QLabel(axis))
            spin = self._spin(value, 1e6, 0.001)
            spins.append(spin)
            row.addWidget(spin)
        return spins, row

    def _arm_changed(self, arm):
        self.tool_frame_edit.setText(self._arm_frames.get(arm.strip(), ""))
        self._refresh_joint_catalog()

    def _clear_catalog_client(self):
        self._catalog_future = None
        if self._catalog_client is not None:
            self._node.destroy_client(self._catalog_client)
            self._catalog_client = None

    def _refresh_joint_catalog(self):
        if self._busy or self._closed:
            return
        self._catalog_generation += 1
        generation = self._catalog_generation
        self._clear_catalog_client()
        arm = self.arm_combo.currentText().strip()
        self._catalog_arm = None
        server = self.execute_service_edit.text().strip().rsplit("/", 1)[0]
        try:
            if not arm or not server:
                raise ValueError("Enter the arm and Execute Pending service before refreshing joints")
            self._catalog_client = self._node.create_client(GetParameters, server + "/get_parameters")
            if not self._catalog_client.service_is_ready():
                raise ValueError("Configure the Cartesian server, then refresh arm joints")
            request = GetParameters.Request()
            request.names = [f"arms.{arm}.joint_names"]
            self._catalog_future = self._catalog_client.call_async(request)
            self.joint_catalog_label.setText(f"Reading joint order for {arm}...")
            self._catalog_future.add_done_callback(
                lambda future: self._catalog_received(future, arm, generation))
        except Exception as error:
            self._clear_catalog_client()
            self.joint_catalog_label.setText(str(error))
        self._update_controls()

    def _catalog_received(self, future, arm, generation):
        if self._closed or generation != self._catalog_generation:
            return
        try:
            response = future.result()
            if len(response.values) != 1 or response.values[0].type != ParameterType.PARAMETER_STRING_ARRAY:
                raise ValueError("Server did not provide the arm joint order; configure it and refresh")
            names = list(response.values[0].string_array_value)
            if not names or any(not name for name in names) or len(set(names)) != len(names):
                raise ValueError("Server joint catalog must contain distinct, nonempty joint names")
            self._replace_joint_widgets(names)
            self._catalog_arm = arm
            self.joint_catalog_label.setText(f"{arm}: {len(names)} joints in server order")
        except Exception as error:
            self._catalog_arm = None
            self.joint_catalog_label.setText(str(error))
        finally:
            self._clear_catalog_client()
            self._update_controls()

    def _replace_joint_widgets(self, names):
        for name, widgets in zip(self._joint_names, self._joint_widgets):
            self._joint_choices[name] = (widgets[0].isChecked(), widgets[1].value(), widgets[2].value())
            for widget in widgets:
                self._inputs.remove(widget)
        while self.joint_grid.count():
            item = self.joint_grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._joint_names, self._joint_widgets = names, []
        for col, label in enumerate(("Available arm joint", "Path tolerance [rad or m]", "Goal tolerance [rad or m]")):
            self.joint_grid.addWidget(QLabel(label), 0, col)
        for row, name in enumerate(names, 1):
            enabled, path, goal = self._joint_choices.get(name, (True, 0.05, 0.05))
            check = QCheckBox(name)
            check.setChecked(enabled)
            widgets = (check, self._spin(path, 1e6, 0.001), self._spin(goal, 1e6, 0.001))
            self._joint_widgets.append(widgets)
            self._inputs.extend(widgets)
            for col, widget in enumerate(widgets):
                self.joint_grid.addWidget(widget, row, col)

    def _base_dofs_changed(self):
        enabled = any(check.isChecked() for check in self.base_dof_checks)
        pose_mode = self.backend_combo.currentData() == "pose"
        index = self.backend_combo.findData("controller")
        self.backend_combo.model().item(index).setEnabled(not enabled)
        if enabled and not pose_mode and self.backend_combo.currentData() != "task":
            self.backend_combo.setCurrentIndex(self.backend_combo.findData("task"))
        self.backend_combo.setToolTip(
            "Pose uses the configured task hierarchy to resolve tool targets" if pose_mode else
            "Base motion requires the task backend" if enabled else "")
        self._update_controls()

    def _change_reference_frame(self):
        frame = self.reference_frame_edit.text().strip()
        if frame == self._target_frame:
            return True
        try:
            if not frame:
                raise ValueError("Reference frame must not be empty")
            xyz = numeric_vector(self.xyz_edit.text(), 3, "Target XYZ")
            quaternion = numeric_vector(self.quaternion_edit.text(), 4, "Quaternion XYZW")
            tf = self._tf_buffer.lookup_transform(frame, self._target_frame, Time()).transform
            xyz, quaternion = transformed_pose(xyz, quaternion, tf)
            self.xyz_edit.setText(",".join(f"{value:.12g}" for value in xyz))
            self.quaternion_edit.setText(",".join(f"{value:.12g}" for value in quaternion))
            self._target_frame = frame
            self._markers.commit_fields()
            return True
        except (TransformException, ValueError) as error:
            self.reference_frame_edit.setText(self._target_frame)
            self._error(f"Reference frame change rejected: {error}")
            return False

    def _read_pose(self):
        if self._markers.read_pose():
            return
        if not self._change_reference_frame():
            return
        reference, tool = self._target_frame, self.tool_frame_edit.text().strip()
        if not tool or reference == tool:
            self._error("Enter distinct reference and tool frames")
            return
        try:
            pose = self._tf_buffer.lookup_transform(reference, tool, Time()).transform
            position = [getattr(pose.translation, axis) for axis in ("x", "y", "z")]
            quaternion = [getattr(pose.rotation, axis) for axis in ("x", "y", "z", "w")]
            norm = math.hypot(*quaternion)
            if not all(math.isfinite(value) for value in position) or not math.isfinite(norm) or norm < 1e-12:
                raise ValueError("TF must contain finite coordinates and a nonzero quaternion")
            quaternion = [value / norm for value in quaternion]
        except (TransformException, ValueError) as error:
            self._error(f"Tool TF is unavailable: {error}")
            return
        self.xyz_edit.setText(",".join(f"{value:.12g}" for value in position))
        self.quaternion_edit.setText(",".join(f"{value:.12g}" for value in quaternion))
        self._markers.set_poses([pose_from_components(position, quaternion)], reference)
        self.result_label.setText(f"Current tool pose read in {reference}; edit XYZ to set the target")

    def _goal_from_fields(self):
        arm = self.arm_combo.currentText().strip()
        pose_mode = self.backend_combo.currentData() == "pose"
        if not arm:
            raise ValueError("Enter the arm name before sending a goal")
        if not pose_mode and (self._catalog_arm != arm or not self._joint_names):
            raise ValueError("Refresh arm joints from the configured server before sending a goal")
        if not self._change_reference_frame():
            raise ValueError("Target frame could not be transformed; the previous frame was restored")
        xyz = numeric_vector(self.xyz_edit.text(), 3, "Target XYZ")
        quaternion = numeric_vector(self.quaternion_edit.text(), 4, "Quaternion XYZW")
        norm = math.hypot(*quaternion)
        if norm < 1e-12 or not math.isfinite(norm):
            raise ValueError("Quaternion must have a finite, nonzero norm")
        goal = self._action_type.Goal()
        goal.arm = arm
        goal.target_pose.header.frame_id = self._target_frame
        goal.target_pose.header.stamp = Time().to_msg()
        for axis, value in zip(("x", "y", "z"), xyz):
            setattr(goal.target_pose.pose.position, axis, value)
        for axis, value in zip(("x", "y", "z", "w"), quaternion):
            setattr(goal.target_pose.pose.orientation, axis, value / norm)
        for vector, spins in ((goal.position_tolerance, self.position_tolerances),
                              (goal.orientation_tolerance, self.orientation_tolerances)):
            for axis, spin in zip(("x", "y", "z"), spins):
                setattr(vector, axis, spin.value())
        tracking = goal.tracking_tolerances
        if not pose_mode:
            goal.available_dofs = [check.isChecked() for check in self.base_dof_checks] + [
                widgets[0].isChecked() for widgets in self._joint_widgets]
            goal.base_limits.max_velocity = [spin.value() for spin in self.base_velocity_spins]
            goal.base_limits.max_acceleration = [spin.value() for spin in self.base_acceleration_spins]
            if self.base_bounds_checkbox.isChecked():
                lower = [spin.value() for spin in self.base_lower_spins]
                upper = [spin.value() for spin in self.base_upper_spins]
                if any(lo > hi for lo, hi in zip(lower, upper)):
                    raise ValueError("Each base lower bound must be less than or equal to its upper bound")
                goal.base_limits.lower_limits, goal.base_limits.upper_limits = lower, upper
            tracking.base_path = [spin.value() for spin in self.base_path_spins]
            tracking.base_goal = [spin.value() for spin in self.base_goal_spins]
            tracking.joint_path = [widgets[1].value() for widgets in self._joint_widgets]
            tracking.joint_goal = [widgets[2].value() for widgets in self._joint_widgets]
        tracking.tip_path = [spin.value() for spin in self.tip_path_spins]
        tracking.tip_goal = [spin.value() for spin in self.tip_goal_spins]
        duration_ns = round(self.goal_time_spin.value() * 1e9)
        tracking.goal_time_tolerance.sec = duration_ns // 1000000000
        tracking.goal_time_tolerance.nanosec = duration_ns % 1000000000
        goal.cartesian_step, goal.planning_time = self.step_spin.value(), self.planning_time_spin.value()
        goal.velocity_scaling = self.velocity_scaling_spin.value()
        goal.acceleration_scaling = self.acceleration_scaling_spin.value()
        goal.execution_backend = self.backend_combo.currentData()
        if any(goal.available_dofs[:6]) and goal.execution_backend != "task":
            raise ValueError("Base motion requires the task backend")
        goal.execute = self.execute_checkbox.isChecked()
        return goal

    def _update_controls(self):
        super()._update_controls()
        if self._closed:
            return
        editable = not self._busy and self._execute_future is None and self._release_future is None
        if self._release_future is not None:
            for widget in self._inputs:
                widget.setEnabled(False)
        pose_mode = self.backend_combo.currentData() == "pose"
        arm = self.arm_combo.currentText().strip()
        self.send_button.setEnabled(editable and bool(arm) and (pose_mode or self._catalog_arm == arm))
        # Refresh can replace a lost/unanswered metadata request.
        self.refresh_joints_button.setEnabled(editable and not pose_mode)
        mobility_controls = [self.base_bounds_checkbox, *self.base_dof_checks,
                             *self.base_velocity_spins, *self.base_acceleration_spins,
                             *self.base_path_spins, *self.base_goal_spins,
                             *(widget for widgets in self._joint_widgets for widget in widgets)]
        for widget in mobility_controls:
            widget.setEnabled(editable and not pose_mode)
        for widget in (*self.base_lower_spins, *self.base_upper_spins):
            widget.setEnabled(editable and not pose_mode and self.base_bounds_checkbox.isChecked())
        self.release_button.setEnabled(editable and not pose_mode)
        self.release_service_edit.setEnabled(editable and not pose_mode)
        self._markers.set_editable(editable)

    def _feedback(self, message, generation):
        super()._feedback(message, generation)
        if self._closed or generation != self._generation or not self._busy:
            return
        feedback = message.feedback
        def values(items):
            return ", ".join(f"{float(value):.4g}" for value in items)
        tool_errors = f"Tool XYZ/rotation: [{values(feedback.tip_error)}]"
        self.errors_label.setText(tool_errors if self.backend_combo.currentData() == "pose" else
                                 f"Base XYZ/RPY: [{values(feedback.base_error)}] | "
                                 f"Arm: [{values(feedback.joint_error)}] | {tool_errors}")

    def _result(self, future, generation):
        super()._result(future, generation)
        if self._closed or generation != self._generation:
            return
        try:
            result = future.result().result
            if self.backend_combo.currentData() == "pose":
                self.result_label.setText(self.result_label.text().split(" | ", 1)[0] +
                                          f" | tool poses: {len(result.desired_path.poses)}")
            else:
                self.result_label.setText(self.result_label.text() +
                                          f" | base: {len(result.base_trajectory.points)} points")
        except Exception:
            pass  # The common result handler already reports transport errors.

    def _release_hold(self):
        if (self._closed or self._busy or self._release_future is not None or
                self.backend_combo.currentData() == "pose"):
            return
        try:
            name = self.release_service_edit.text().strip()
            if not name:
                raise ValueError("Enter the trajectory release service endpoint")
            self._release_client = self._node.create_client(Trigger, name)
            if not self._release_client.service_is_ready():
                raise ValueError(f"Trajectory release service unavailable: {name}")
            self._release_future = self._release_client.call_async(Trigger.Request())
            self._release_future.add_done_callback(self._released_hold)
        except Exception as error:
            self._clear_release_client()
            self._error(str(error))
        self._update_controls()

    def _released_hold(self, future):
        try:
            response = future.result()
            if not self._closed:
                status = "Trajectory hold released" if response.success else "Release rejected"
                self.result_label.setText(f"{status}: {response.message}")
        except Exception as error:
            self._error(str(error))
        finally:
            self._clear_release_client()
            self._update_controls()

    def _clear_release_client(self):
        self._release_future = None
        if self._release_client is not None:
            self._node.destroy_client(self._release_client)
            self._release_client = None

    def destroy_clients(self):
        self._catalog_generation += 1
        self._clear_catalog_client()
        self._clear_release_client()
        super().destroy_clients()
