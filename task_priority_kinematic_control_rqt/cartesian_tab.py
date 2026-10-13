"""Single-arm Cartesian goals, using the common pending/cancel client protocol."""
import math

from python_qt_binding.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)
from rclpy.time import Time
from sura_manipulation_actions.action import PlanCartesianTrajectory

from .bimanual_cartesian_tab import (
    BimanualCartesianTab, arm_frames, configured_reference_frame, numeric_vector,
    TransformException,
)

DEFAULT_ACTION = "/cirtesub/manipulation/plan_cartesian_trajectory"
DEFAULT_SERVER = "/plan_cartesian_trajectory_lifecycle_action_server"


class CartesianTab(BimanualCartesianTab):
    _action_type = PlanCartesianTrajectory

    def __init__(self, node):
        QWidget.__init__(self)
        self._initialize_client(node)
        layout = QVBoxLayout(self)
        self.form_scroll = QScrollArea()
        self.form_scroll.setWidgetResizable(True)
        self.form_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        content = QWidget()
        form = QFormLayout(content)
        self.form_scroll.setWidget(content)
        layout.addWidget(self.form_scroll, 1)
        note = QLabel("Single-arm Cartesian target: absolute tool pose in the reference/base frame. "
                      "Only the selected arm is commanded. Backend activation is manual.")
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
        self._arm_frames = arm_frames("single_arm_cartesian_planner.yaml")
        self.arm_combo = QComboBox()
        self.arm_combo.setEditable(True)
        self.arm_combo.addItems(list(self._arm_frames))
        self.tool_frame_edit = QLineEdit(self._arm_frames[self.arm_combo.currentText()])
        self.arm_combo.currentTextChanged.connect(
            lambda name: self.tool_frame_edit.setText(self._arm_frames.get(name.strip(), "")))
        self.reference_frame_edit = QLineEdit(configured_reference_frame(
            "single_arm_cartesian_planner.yaml", "reference_frame"))
        self.reference_frame_edit.setToolTip("Must match reference_frame configured in the single-arm server")
        self.tool_frame_edit.setToolTip("Used for TF reading; must match the configured arm tip")
        self.backend_combo = QComboBox()
        for label, value in (("Server default", ""), ("Controller", "controller"), ("Task", "task")):
            self.backend_combo.addItem(label, value)
        self.xyz_edit = QLineEdit("0.0,0.0,0.0")
        self.xyz_edit.setToolTip("Absolute X,Y,Z in metres in the reference frame. Read TF, then edit the target.")
        self.quaternion_edit = QLineEdit("0.0,0.0,0.0,1.0")
        self.quaternion_edit.setToolTip("Absolute tool orientation in reference axes; quaternion XYZW")
        self.position_tolerances, position_row = self._vector_row(0.002, 100.0)
        self.orientation_tolerances, orientation_row = self._vector_row(0.03, math.pi)
        self.step_spin = self._spin(0.005, 1.0, 0.001)
        self.planning_time_spin = self._spin(10.0, 3600.0, 1.0)
        self.planning_time_spin.setDecimals(2)
        self.velocity_scaling_spin = self._spin(0.0, 1.0, 0.01)
        self.acceleration_scaling_spin = self._spin(0.0, 1.0, 0.01)
        for spin in (self.velocity_scaling_spin, self.acceleration_scaling_spin):
            spin.setDecimals(3)
            spin.setSpecialValueText("Server default")
            spin.setToolTip("Joint limit factor: 0.1 = 10%, 0.05 = 5%; zero uses server default")
        self.execute_checkbox = QCheckBox("Execute after planning")
        self.execute_checkbox.setChecked(False)
        for label, widget in (
            ("Action", self.action_edit), ("Execute Pending Service", self.execute_service_edit),
            ("Arm", self.arm_combo), ("Execution Backend", self.backend_combo),
            ("Reference Frame", self.reference_frame_edit), ("Tool Frame (TF)", self.tool_frame_edit),
            ("Target XYZ [m]", self.xyz_edit), ("Target Quaternion XYZW", self.quaternion_edit),
            ("Position Tolerances [m]", position_row), ("Orientation Tolerances [rad]", orientation_row),
            ("Cartesian Step [m]", self.step_spin), ("Planning Time [s]", self.planning_time_spin),
            ("Velocity Scaling [0..1]", self.velocity_scaling_spin),
            ("Acceleration Scaling [0..1]", self.acceleration_scaling_spin),
        ):
            form.addRow(label, widget)
        form.addRow(self.execute_checkbox)
        self.read_pose_button = QPushButton("Read Current Pose (TF)")
        self.send_button = QPushButton("Send Goal")
        self.execute_button = QPushButton("Execute Pending")
        self.cancel_button = QPushButton("Cancel Goal")
        buttons = QHBoxLayout()
        for button, callback in ((self.read_pose_button, self._read_pose),
                                 (self.send_button, self._send_goal),
                                 (self.execute_button, self._execute_pending),
                                 (self.cancel_button, self._cancel_goal)):
            button.clicked.connect(callback)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        footer = QFormLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.feedback_label = QLabel("Read the current tool pose, then enter the Cartesian target")
        self.result_label = QLabel("")
        self.feedback_label.setWordWrap(True)
        self.result_label.setWordWrap(True)
        footer.addRow("Progress", self.progress)
        footer.addRow("Feedback", self.feedback_label)
        footer.addRow("Result", self.result_label)
        layout.addLayout(footer)
        self._inputs = [self.action_edit, self.execute_service_edit, self.arm_combo,
                        self.backend_combo, self.reference_frame_edit, self.tool_frame_edit,
                        self.xyz_edit, self.quaternion_edit, self.step_spin, self.planning_time_spin,
                        self.velocity_scaling_spin, self.acceleration_scaling_spin,
                        self.execute_checkbox, self.read_pose_button,
                        *self.position_tolerances, *self.orientation_tolerances]
        self._update_controls()

    def _read_pose(self):
        reference = self.reference_frame_edit.text().strip()
        tool = self.tool_frame_edit.text().strip()
        if not reference or not tool or reference == tool:
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
        self.result_label.setText(f"Current tool pose read in {reference}; edit XYZ to set the target")

    def _goal_from_fields(self):
        arm = self.arm_combo.currentText().strip()
        frame = self.reference_frame_edit.text().strip()
        if not arm or not frame:
            raise ValueError("Enter a configured arm and reference frame")
        xyz = numeric_vector(self.xyz_edit.text(), 3, "Target XYZ")
        quaternion = numeric_vector(self.quaternion_edit.text(), 4, "Quaternion XYZW")
        norm = math.hypot(*quaternion)
        if norm < 1e-12 or not math.isfinite(norm):
            raise ValueError("Quaternion must have a finite, nonzero norm")
        goal = self._action_type.Goal()
        goal.arm = arm
        goal.target_pose.header.frame_id = frame
        goal.target_pose.header.stamp = self._node.get_clock().now().to_msg()
        for axis, value in zip(("x", "y", "z"), xyz):
            setattr(goal.target_pose.pose.position, axis, value)
        for axis, value in zip(("x", "y", "z", "w"), quaternion):
            setattr(goal.target_pose.pose.orientation, axis, value / norm)
        for vector, spins in ((goal.position_tolerance, self.position_tolerances),
                              (goal.orientation_tolerance, self.orientation_tolerances)):
            for axis, spin in zip(("x", "y", "z"), spins):
                setattr(vector, axis, spin.value())
        goal.cartesian_step = self.step_spin.value()
        goal.planning_time = self.planning_time_spin.value()
        goal.velocity_scaling = self.velocity_scaling_spin.value()
        goal.acceleration_scaling = self.acceleration_scaling_spin.value()
        goal.execution_backend = self.backend_combo.currentData()
        goal.execute = self.execute_checkbox.isChecked()
        return goal
