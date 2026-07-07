import functools

from geometry_msgs.msg import PoseStamped
from python_qt_binding.QtCore import QMimeData, QPoint, Qt, Signal
from python_qt_binding.QtGui import QDrag
from python_qt_binding.QtWidgets import (
    QAbstractItemView,
    QApplication,
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

TASK_TOPIC_PREFIX = "/cirtesub/controller/task_priority/tasks"


class TaskTableWidget(QTableWidget):
    row_dropped = Signal(int, int)

    def __init__(self, rows, columns):
        super().__init__(rows, columns)
        self._drag_start_row = -1
        self._drag_start_pos = QPoint()
        self.setDragEnabled(False)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropOverwriteMode(False)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start_pos = event.pos()
            self._drag_start_row = self.indexAt(event.pos()).row()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not event.buttons() & Qt.LeftButton:
            return
        if self._drag_start_row < 0:
            return
        if (
            event.pos() - self._drag_start_pos
        ).manhattanLength() < QApplication.startDragDistance():
            return

        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(str(self._drag_start_row))
        drag.setMimeData(mime_data)
        drag.exec_(Qt.MoveAction)
        self._drag_start_row = -1

    def dragEnterEvent(self, event):
        if event.source() is self:
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event):
        if event.source() is self:
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event):
        source_row = self._drag_start_row
        target_row = self.indexAt(event.pos()).row()
        if target_row < 0:
            target_row = self.rowCount() - 1

        self._drag_start_row = -1
        if (
            source_row < 0
            or target_row < 0
            or source_row >= self.rowCount()
            or target_row >= self.rowCount()
        ):
            event.ignore()
            return

        event.acceptProposedAction()
        if source_row != target_row:
            self.row_dropped.emit(source_row, target_row)


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

        self._table = TaskTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["Task", "Plugin", "Group", "Priority", "Enabled", "Status"]
        )
        self._table.row_dropped.connect(self._apply_order_from_table_move)
        self._table.itemSelectionChanged.connect(self._update_priority_buttons)
        self._layout.addWidget(self._table)

        table_buttons = QHBoxLayout()
        self._refresh_button = QPushButton("Refresh")
        self._refresh_button.clicked.connect(lambda: self._refresh_tasks())
        self._move_up_button = QPushButton("Up")
        self._move_up_button.clicked.connect(functools.partial(self._move_selected_task, -1))
        self._move_down_button = QPushButton("Down")
        self._move_down_button.clicked.connect(functools.partial(self._move_selected_task, 1))
        table_buttons.addWidget(self._refresh_button)
        table_buttons.addWidget(self._move_up_button)
        table_buttons.addWidget(self._move_down_button)
        self._layout.addLayout(table_buttons)

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
        self._last_ordered_task_ids = []
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

    def _refresh_tasks(self, selected_task_id=None):
        if not self._list_client.wait_for_service(timeout_sec=0.2):
            self._show_error("list_tasks service is not available")
            return
        req = ListTasks.Request()
        response = self._spin_until_complete(self._list_client.call_async(req))
        if response is None or not response.success:
            self._show_error("Failed to list tasks")
            return

        if selected_task_id is None:
            selected_task_id = self._selected_task_id()

        self._table.setRowCount(len(response.tasks))
        targetable_tasks = {}
        self._last_ordered_task_ids = []
        for row, task in enumerate(response.tasks):
            self._last_ordered_task_ids.append(task.id)
            task_item = QTableWidgetItem(task.id)
            task_item.setData(Qt.UserRole, task.id)
            self._table.setItem(row, 0, task_item)
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

        self._select_task_row(selected_task_id)
        self._update_priority_buttons()

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

    def _selected_task_id(self):
        current_row = self._table.currentRow()
        if current_row < 0:
            return None
        item = self._table.item(current_row, 0)
        if item is None:
            return None
        return item.data(Qt.UserRole)

    def _select_task_row(self, task_id):
        if task_id is None:
            return
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is not None and item.data(Qt.UserRole) == task_id:
                self._table.selectRow(row)
                return

    def _update_priority_buttons(self):
        current_row = self._table.currentRow()
        self._move_up_button.setEnabled(current_row > 0)
        self._move_down_button.setEnabled(
            current_row >= 0 and current_row < self._table.rowCount() - 1
        )

    def _move_selected_task(self, offset):
        source_row = self._table.currentRow()
        target_row = source_row + offset
        if source_row < 0 or target_row < 0 or target_row >= len(self._last_ordered_task_ids):
            return
        self._apply_order_from_table_move(source_row, target_row)

    def _apply_order_from_table_move(self, source_row, target_row):
        if source_row == target_row:
            return

        if (
            source_row < 0
            or target_row < 0
            or source_row >= len(self._last_ordered_task_ids)
            or target_row >= len(self._last_ordered_task_ids)
        ):
            self._refresh_tasks()
            return

        ordered_ids = list(self._last_ordered_task_ids)
        task_id = ordered_ids.pop(source_row)
        ordered_ids.insert(target_row, task_id)

        self._apply_order(ordered_ids, task_id)

    def _apply_order(self, ordered_ids, selected_task_id):
        if ordered_ids == self._last_ordered_task_ids:
            return

        if not self._reorder_client.wait_for_service(timeout_sec=0.2):
            self._show_error("reorder_tasks service is not available")
            self._refresh_tasks(selected_task_id)
            return

        req = ReorderTasks.Request()
        req.ordered_task_ids = ordered_ids
        response = self._spin_until_complete(self._reorder_client.call_async(req))
        if response is None or not response.success:
            self._show_error(response.message if response else "Failed to reorder tasks")
            self._refresh_tasks(selected_task_id)
            return

        self._last_ordered_task_ids = ordered_ids
        self._refresh_tasks(selected_task_id)

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

            topic_name = f"{TASK_TOPIC_PREFIX}/{task_id}/target"
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

            topic_name = f"{TASK_TOPIC_PREFIX}/{task_id}/joint_target"
            if task_id not in self._joint_target_pubs:
                self._joint_target_pubs[task_id] = self._node.create_publisher(
                    Float64MultiArray, topic_name, 10
                )

            joint_msg = Float64MultiArray()
            joint_msg.data = joint_target
            self._joint_target_pubs[task_id].publish(joint_msg)
            return

        self._show_error("Selected task does not accept runtime targets")
