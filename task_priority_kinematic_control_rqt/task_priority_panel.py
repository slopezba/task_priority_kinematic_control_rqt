import functools

from geometry_msgs.msg import PoseStamped
from python_qt_binding.QtCore import Qt
from python_qt_binding.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from rqt_gui_py.plugin import Plugin
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from task_priority_kinematic_control.msg import HierarchyState
from task_priority_kinematic_control.srv import (
    ListTasks,
    ReorderTasks,
    SetTaskEnabled,
)


class TaskPriorityPanel(Plugin):
    def __init__(self, context):
        super().__init__(context)
        self.setObjectName("TaskPriorityPanel")

        self._owns_rclpy_context = False
        if not rclpy.get_default_context().ok():
            rclpy.init(args=None)
            self._owns_rclpy_context = True
        self._node = Node("task_priority_rqt_panel")

        self._widget = QWidget()
        self._widget.setWindowTitle("Task Priority Control")
        self._layout = QVBoxLayout(self._widget)

        self._status_label = QLabel("Waiting for hierarchy state...")
        self._status_label.setAlignment(Qt.AlignLeft)
        self._layout.addWidget(self._status_label)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["Task", "Plugin", "Group", "Priority", "Enabled", "Status"]
        )
        self._layout.addWidget(self._table)

        self._refresh_button = QPushButton("Refresh")
        self._refresh_button.clicked.connect(self._refresh_tasks)
        self._layout.addWidget(self._refresh_button)

        reorder_box = QGroupBox("Reorder Tasks")
        reorder_layout = QHBoxLayout(reorder_box)
        self._reorder_edit = QLineEdit()
        self._reorder_edit.setPlaceholderText("left_pose,right_pose,joint_limits,...")
        self._reorder_button = QPushButton("Apply Order")
        self._reorder_button.clicked.connect(self._apply_order)
        reorder_layout.addWidget(self._reorder_edit)
        reorder_layout.addWidget(self._reorder_button)
        self._layout.addWidget(reorder_box)

        goal_box = QGroupBox("Pose Goal")
        goal_layout = QFormLayout(goal_box)
        self._goal_task_combo = QComboBox()
        self._goal_task_combo.currentIndexChanged.connect(self._on_goal_task_changed)
        self._goal_target_type = QLabel("No active target tasks")
        self._goal_xyz = QLineEdit("0.0,0.0,0.0")
        self._goal_quat = QLineEdit("0.0,0.0,0.0,1.0")
        self._goal_joints = QLineEdit()
        self._goal_joint_info = QLabel("")
        self._goal_joint_info.setWordWrap(True)
        self._pose_target_widget = QWidget()
        pose_target_layout = QFormLayout(self._pose_target_widget)
        pose_target_layout.addRow("XYZ", self._goal_xyz)
        pose_target_layout.addRow("Quaternion XYZW", self._goal_quat)
        self._joint_target_widget = QWidget()
        joint_target_layout = QFormLayout(self._joint_target_widget)
        joint_target_layout.addRow("Joint Names", self._goal_joint_info)
        joint_target_layout.addRow("Float64MultiArray", self._goal_joints)
        self._goal_button = QPushButton("Send Target")
        self._goal_button.clicked.connect(self._send_target)
        goal_layout.addRow("Task", self._goal_task_combo)
        goal_layout.addRow("Target Type", self._goal_target_type)
        goal_layout.addRow(self._pose_target_widget)
        goal_layout.addRow(self._joint_target_widget)
        goal_layout.addRow(self._goal_button)
        self._layout.addWidget(goal_box)

        context.add_widget(self._widget)

        self._list_client = self._node.create_client(ListTasks, "/list_tasks")
        self._enable_client = self._node.create_client(SetTaskEnabled, "/set_task_enabled")
        self._reorder_client = self._node.create_client(ReorderTasks, "/reorder_tasks")
        self._pose_goal_pubs = {}
        self._joint_target_pubs = {}
        self._targetable_tasks = {}
        self._state_sub = self._node.create_subscription(
            HierarchyState, "/hierarchy_state", self._on_hierarchy_state, 10
        )

        self._refresh_tasks()

    def shutdown_plugin(self):
        self._node.destroy_node()
        if self._owns_rclpy_context and rclpy.get_default_context().ok():
            rclpy.shutdown()

    def _spin_until_complete(self, future):
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self._node, timeout_sec=0.1)
        return future.result()

    def _show_error(self, message):
        QMessageBox.warning(self._widget, "Task Priority Control", message)

    def _on_hierarchy_state(self, msg):
        self._status_label.setText(
            f"Backend: {msg.backend_name} | Solver: {msg.solver_method} | Ready: {msg.ready}"
        )

    def _refresh_tasks(self):
        if not self._list_client.wait_for_service(timeout_sec=0.2):
            self._show_error("list_tasks service is not available")
            return
        req = ListTasks.Request()
        response = self._spin_until_complete(self._list_client.call_async(req))
        if response is None or not response.success:
            self._show_error("Failed to list tasks")
            return

        self._table.setRowCount(len(response.tasks))
        targetable_tasks = {}
        for row, task in enumerate(response.tasks):
            self._table.setItem(row, 0, QTableWidgetItem(task.id))
            self._table.setItem(row, 1, QTableWidgetItem(task.plugin))
            self._table.setItem(row, 2, QTableWidgetItem(task.group))
            self._table.setItem(row, 3, QTableWidgetItem(str(task.priority)))
            checkbox = QCheckBox()
            checkbox.setChecked(task.enabled)
            checkbox.stateChanged.connect(
                functools.partial(self._toggle_task, task.id)
            )
            self._table.setCellWidget(row, 4, checkbox)
            self._table.setItem(row, 5, QTableWidgetItem(task.status_message))
            if task.enabled and task.target_type in ("pose", "joint_array"):
                targetable_tasks[task.id] = task

        previous_task_id = self._goal_task_combo.currentData()
        self._targetable_tasks = targetable_tasks
        self._goal_task_combo.blockSignals(True)
        self._goal_task_combo.clear()
        for task_id, task in self._targetable_tasks.items():
            self._goal_task_combo.addItem(task_id, task_id)
        self._goal_task_combo.blockSignals(False)

        if previous_task_id in self._targetable_tasks:
            self._goal_task_combo.setCurrentIndex(
                self._goal_task_combo.findData(previous_task_id)
            )
        elif self._goal_task_combo.count() > 0:
            self._goal_task_combo.setCurrentIndex(0)

        self._on_goal_task_changed()

    def _toggle_task(self, task_id, state):
        if not self._enable_client.wait_for_service(timeout_sec=0.2):
            self._show_error("set_task_enabled service is not available")
            return
        req = SetTaskEnabled.Request()
        req.task_id = task_id
        req.enabled = state == Qt.Checked
        response = self._spin_until_complete(self._enable_client.call_async(req))
        if response is None or not response.success:
            self._show_error(response.message if response else "Failed to update task state")
            return
        self._refresh_tasks()

    def _apply_order(self):
        ordered_ids = [item.strip() for item in self._reorder_edit.text().split(",") if item.strip()]
        if not ordered_ids:
            self._show_error("Please provide at least one task id")
            return
        if not self._reorder_client.wait_for_service(timeout_sec=0.2):
            self._show_error("reorder_tasks service is not available")
            return
        req = ReorderTasks.Request()
        req.ordered_task_ids = ordered_ids
        response = self._spin_until_complete(self._reorder_client.call_async(req))
        if response is None or not response.success:
            self._show_error(response.message if response else "Failed to reorder tasks")
            return
        self._refresh_tasks()

    def _on_goal_task_changed(self):
        task_id = self._goal_task_combo.currentData()
        task = self._targetable_tasks.get(task_id)
        if task is None:
            self._goal_target_type.setText("No active target tasks")
            self._goal_joint_info.setText("")
            self._pose_target_widget.setVisible(False)
            self._joint_target_widget.setVisible(False)
            self._goal_button.setEnabled(False)
            return

        if task.target_type == "pose":
            self._goal_target_type.setText("PoseStamped")
            self._goal_joint_info.setText("")
            self._pose_target_widget.setVisible(True)
            self._joint_target_widget.setVisible(False)
            self._goal_button.setEnabled(True)
            return

        if task.target_type == "joint_array":
            joint_names = ", ".join(task.joint_names)
            self._goal_target_type.setText("Float64MultiArray")
            self._goal_joint_info.setText(joint_names)
            self._goal_joints.setPlaceholderText(
                ",".join(["0.0"] * len(task.joint_names))
            )
            self._pose_target_widget.setVisible(False)
            self._joint_target_widget.setVisible(True)
            self._goal_button.setEnabled(True)
            return

        self._goal_target_type.setText("Unsupported")
        self._goal_joint_info.setText("")
        self._pose_target_widget.setVisible(False)
        self._joint_target_widget.setVisible(False)
        self._goal_button.setEnabled(False)

    def _send_target(self):
        task_id = self._goal_task_combo.currentData()
        task = self._targetable_tasks.get(task_id)
        if task is None:
            self._show_error("Please select an active task")
            return

        if task.target_type == "pose":
            try:
                xyz = [float(value.strip()) for value in self._goal_xyz.text().split(",")]
                quat = [float(value.strip()) for value in self._goal_quat.text().split(",")]
            except ValueError:
                self._show_error("Pose target fields must contain comma-separated numbers")
                return

            if len(xyz) != 3 or len(quat) != 4:
                self._show_error("XYZ must have 3 values and quaternion must have 4 values")
                return

            topic_name = f"/task_priority_controller/tasks/{task_id}/target"
            if task_id not in self._pose_goal_pubs:
                self._pose_goal_pubs[task_id] = self._node.create_publisher(
                    PoseStamped, topic_name, 10
                )

            pose_msg = PoseStamped()
            pose_msg.header.stamp = self._node.get_clock().now().to_msg()
            pose_msg.header.frame_id = "world_ned"
            pose_msg.pose.position.x = xyz[0]
            pose_msg.pose.position.y = xyz[1]
            pose_msg.pose.position.z = xyz[2]
            pose_msg.pose.orientation.x = quat[0]
            pose_msg.pose.orientation.y = quat[1]
            pose_msg.pose.orientation.z = quat[2]
            pose_msg.pose.orientation.w = quat[3]
            self._pose_goal_pubs[task_id].publish(pose_msg)
            return

        if task.target_type == "joint_array":
            try:
                joint_target = [float(value.strip()) for value in self._goal_joints.text().split(",")]
            except ValueError:
                self._show_error("Joint target field must contain comma-separated numbers")
                return

            if len(joint_target) != len(task.joint_names):
                self._show_error(
                    f"Joint target must contain {len(task.joint_names)} values for the selected task"
                )
                return

            topic_name = f"/task_priority_controller/tasks/{task_id}/joint_target"
            if task_id not in self._joint_target_pubs:
                self._joint_target_pubs[task_id] = self._node.create_publisher(
                    Float64MultiArray, topic_name, 10
                )

            joint_msg = Float64MultiArray()
            joint_msg.data = joint_target
            self._joint_target_pubs[task_id].publish(joint_msg)
            return

        self._show_error("Selected task does not accept runtime targets")
