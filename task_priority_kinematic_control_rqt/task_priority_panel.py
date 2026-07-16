import functools
import time

from geometry_msgs.msg import PoseStamped
from python_qt_binding.QtCore import QMimeData, QPoint, Qt, QTimer, Signal
from python_qt_binding.QtGui import QDrag
from python_qt_binding.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from rqt_gui_py.plugin import Plugin
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from task_priority_kinematic_control.msg import ControllerOutput, HierarchyState, TaskGainUpdate, TaskState
from task_priority_kinematic_control.srv import (
    ListTasks,
    ReorderTasks,
    SetSolverConfig,
    SetTaskGains,
    SetTaskEnabled,
)

TASK_TOPIC_PREFIX = "/cirtesub/controller/task_priority/tasks"
CONTROLLER_OUTPUT_TOPIC = "/cirtesub/controller/task_priority/output"
DEFAULT_CONTROLLER_NODE = "/cirtesub/controller/task_priority_controller"
LAMBDA_SCALE = 10000
LAMBDA_MIN = 0.0001
LAMBDA_MAX = 1.0
GAIN_MIN = 0.0
GAIN_MAX = 5.0
DOF_WEIGHT_MIN = 0.001
DOF_WEIGHT_MAX = 20.0
JOINT_TARGET_SCALE = 1000
DEFAULT_JOINT_MIN = 0.0
DEFAULT_JOINT_MAX = 6.1


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

        self._pose_goal_pubs = {}
        self._joint_target_pubs = {}
        self._targetable_tasks = {}
        self._task_statuses = {}
        self._task_state_subs = {}
        self._latest_task_states = {}
        self._latest_controller_output = None
        self._last_ordered_task_ids = []
        self._gain_entries = {}
        self._dof_weight_entries = []
        self._joint_target_controls = []

        self._widget = QWidget()
        self._widget.setWindowTitle("Task Priority Control")
        self._layout = QVBoxLayout(self._widget)
        self._joint_target_live_timer = QTimer(self._widget)
        self._joint_target_live_timer.setSingleShot(True)
        self._joint_target_live_timer.setInterval(120)
        self._joint_target_live_timer.timeout.connect(self._publish_live_joint_target)

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
        self._stop_button = QPushButton("Stop")
        self._stop_button.clicked.connect(self._stop_task_priority)
        table_buttons.addWidget(self._refresh_button)
        table_buttons.addWidget(self._move_up_button)
        table_buttons.addWidget(self._move_down_button)
        table_buttons.addWidget(self._stop_button)
        self._layout.addLayout(table_buttons)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_solver_tab(), "Solver")
        self._tabs.addTab(self._build_gains_tab(), "Gains")
        self._tabs.addTab(self._build_goal_tab(), "Pose Goal")
        self._tabs.addTab(self._build_errors_tab(), "Errors")
        self._layout.addWidget(self._tabs)

        context.add_widget(self._widget)

        self._list_client = self._node.create_client(ListTasks, "/list_tasks")
        self._enable_client = self._node.create_client(SetTaskEnabled, "/set_task_enabled")
        self._set_solver_config_client = self._node.create_client(SetSolverConfig, "/set_solver_config")
        self._set_task_gains_client = self._node.create_client(SetTaskGains, "/set_task_gains")
        self._reorder_client = self._node.create_client(ReorderTasks, "/reorder_tasks")
        self._stop_client = self._node.create_client(Trigger, "/stop_task_priority")
        self._state_sub = self._node.create_subscription(
            HierarchyState, "/hierarchy_state", self._on_hierarchy_state, 10
        )
        self._controller_output_sub = self._node.create_subscription(
            ControllerOutput, CONTROLLER_OUTPUT_TOPIC, self._on_controller_output, 10
        )
        self._spin_timer = QTimer(self._widget)
        self._spin_timer.timeout.connect(self._spin_ros_once)
        self._spin_timer.start(50)

        self._refresh_tasks()
        self._read_solver_parameters(show_errors=False)
        self._read_gain_parameters(show_errors=False)

    def _build_solver_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)

        self._controller_node_edit = QLineEdit(DEFAULT_CONTROLLER_NODE)
        self._solver_method_combo = QComboBox()
        for method in ("dls", "pinv", "svd"):
            self._solver_method_combo.addItem(method, method)
        self._solver_method_combo.currentIndexChanged.connect(self._on_solver_method_changed)

        lambda_row = QHBoxLayout()
        self._dls_lambda_slider = QSlider(Qt.Horizontal)
        self._dls_lambda_slider.setRange(int(LAMBDA_MIN * LAMBDA_SCALE), int(LAMBDA_MAX * LAMBDA_SCALE))
        self._dls_lambda_slider.setSingleStep(1)
        self._dls_lambda_slider.setPageStep(100)
        self._dls_lambda_slider.setValue(int(0.05 * LAMBDA_SCALE))
        self._dls_lambda_slider.valueChanged.connect(self._on_dls_lambda_slider_changed)
        self._dls_lambda_value_label = QLabel(self._format_float(self._dls_lambda_value(), 4))
        lambda_row.addWidget(self._dls_lambda_slider)
        lambda_row.addWidget(self._dls_lambda_value_label)

        solver_buttons = QHBoxLayout()
        self._read_solver_button = QPushButton("Read")
        self._read_solver_button.clicked.connect(self._read_solver_parameters)
        self._apply_solver_button = QPushButton("Apply")
        self._apply_solver_button.clicked.connect(self._apply_solver_parameters)
        solver_buttons.addWidget(self._read_solver_button)
        solver_buttons.addWidget(self._apply_solver_button)

        self._dof_weights_box = QGroupBox("DOF Weights")
        self._dof_weights_layout = QGridLayout(self._dof_weights_box)
        self._dof_weights_layout.setHorizontalSpacing(6)
        self._dof_weights_layout.setVerticalSpacing(4)
        self._dof_weights_empty_label = QLabel("No DOF weights loaded")
        self._dof_weights_layout.addWidget(self._dof_weights_empty_label, 0, 0)

        layout.addRow("Controller Node", self._controller_node_edit)
        layout.addRow("Method", self._solver_method_combo)
        layout.addRow("DLS Lambda", lambda_row)
        layout.addRow(self._dof_weights_box)
        layout.addRow(solver_buttons)
        return tab

    def _build_gains_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        buttons = QHBoxLayout()
        self._read_gains_button = QPushButton("Read")
        self._read_gains_button.clicked.connect(self._read_gain_parameters)
        self._apply_gains_button = QPushButton("Apply")
        self._apply_gains_button.clicked.connect(self._apply_gain_parameters)
        buttons.addWidget(self._read_gains_button)
        buttons.addWidget(self._apply_gains_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self._gains_scroll = QScrollArea()
        self._gains_scroll.setWidgetResizable(True)
        self._gains_container = QWidget()
        self._gains_layout = QVBoxLayout(self._gains_container)
        self._gains_empty_label = QLabel("No gain parameters loaded")
        self._gains_layout.addWidget(self._gains_empty_label)
        self._gains_layout.addStretch(1)
        self._gains_scroll.setWidget(self._gains_container)
        layout.addWidget(self._gains_scroll)
        return tab

    def _build_goal_tab(self):
        tab = QWidget()
        goal_layout = QFormLayout(tab)
        self._goal_task_combo = QComboBox()
        self._goal_task_combo.currentIndexChanged.connect(self._on_goal_task_changed)
        self._goal_target_type = QLabel("No active target tasks")
        self._goal_xyz = QLineEdit("0.0,0.0,0.0")
        self._goal_quat = QLineEdit("0.0,0.0,0.0,1.0")
        self._goal_joint_info = QLabel("")
        self._goal_joint_info.setWordWrap(True)
        self._pose_target_widget = QWidget()
        pose_target_layout = QFormLayout(self._pose_target_widget)
        pose_target_layout.addRow("XYZ", self._goal_xyz)
        pose_target_layout.addRow("Quaternion XYZW", self._goal_quat)
        self._joint_target_widget = QWidget()
        joint_target_layout = QVBoxLayout(self._joint_target_widget)
        joint_target_layout.addWidget(self._goal_joint_info)
        self._joint_target_scroll = QScrollArea()
        self._joint_target_scroll.setWidgetResizable(True)
        self._joint_target_container = QWidget()
        self._joint_target_layout = QGridLayout(self._joint_target_container)
        self._joint_target_scroll.setWidget(self._joint_target_container)
        joint_target_layout.addWidget(self._joint_target_scroll)
        self._live_joint_target_checkbox = QCheckBox("Live")
        self._live_joint_target_checkbox.setVisible(False)
        self._goal_button = QPushButton("Send Target")
        self._goal_button.clicked.connect(self._send_target)
        target_buttons = QHBoxLayout()
        target_buttons.addWidget(self._live_joint_target_checkbox)
        target_buttons.addWidget(self._goal_button)
        target_buttons.addStretch(1)
        goal_layout.addRow("Task", self._goal_task_combo)
        goal_layout.addRow("Target Type", self._goal_target_type)
        goal_layout.addRow(self._pose_target_widget)
        goal_layout.addRow(self._joint_target_widget)
        goal_layout.addRow(target_buttons)
        return tab

    def _build_errors_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self._errors_table = QTableWidget(0, 4)
        self._errors_table.setHorizontalHeaderLabels(
            ["Task", "Active", "Error", "Command/Velocity"]
        )
        self._errors_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._errors_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self._errors_table)

        self._controller_output_table = QTableWidget(3, 3)
        self._controller_output_table.setHorizontalHeaderLabels(
            ["Output", "Names", "Velocity"]
        )
        self._controller_output_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._controller_output_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        for row, name in enumerate(("Base", "Left Arm", "Right Arm")):
            self._controller_output_table.setItem(row, 0, QTableWidgetItem(name))
            self._controller_output_table.setItem(row, 1, QTableWidgetItem(""))
            self._controller_output_table.setItem(row, 2, QTableWidgetItem(""))
        layout.addWidget(self._controller_output_table)
        return tab

    def shutdown_plugin(self):
        self._spin_timer.stop()
        self._node.destroy_node()
        if self._owns_rclpy_context and rclpy.get_default_context().ok():
            rclpy.shutdown()

    def _spin_ros_once(self):
        if rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.0)

    def _spin_until_complete(self, future, timeout_sec=2.0):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done():
            if time.monotonic() >= deadline:
                return None
            rclpy.spin_once(self._node, timeout_sec=0.02)
            QApplication.processEvents()
        return future.result()

    def _show_error(self, message):
        QMessageBox.warning(self._widget, "Task Priority Control", message)

    def _controller_node_name(self):
        node_name = self._controller_node_edit.text().strip()
        return node_name.rstrip("/") if node_name else DEFAULT_CONTROLLER_NODE

    def _format_float(self, value, decimals=3):
        return f"{value:.{decimals}f}"

    def _format_values(self, values):
        if not values:
            return ""
        return ", ".join(self._format_float(value) for value in values)

    def _format_vector(self, values):
        if not values:
            return "[]"
        return "[" + " ".join(self._format_float(value) for value in values) + "]"

    def _slider_value_from_double(self, value):
        return int(round(value * JOINT_TARGET_SCALE))

    def _double_from_slider_value(self, value):
        return value / float(JOINT_TARGET_SCALE)

    def _dls_lambda_value(self):
        return max(LAMBDA_MIN, self._dls_lambda_slider.value() / float(LAMBDA_SCALE))

    def _set_dls_lambda_value(self, value):
        clipped = min(max(value, LAMBDA_MIN), LAMBDA_MAX)
        self._dls_lambda_slider.blockSignals(True)
        self._dls_lambda_slider.setValue(int(round(clipped * LAMBDA_SCALE)))
        self._dls_lambda_slider.blockSignals(False)
        self._dls_lambda_value_label.setText(self._format_float(self._dls_lambda_value(), 4))

    def _labels_for_dof_weights(self, values):
        base_labels = ["base x", "base y", "base z", "base roll", "base pitch", "base yaw"]
        if len(values) <= len(base_labels):
            return base_labels[: len(values)]

        labels = list(base_labels)
        arm_value_count = len(values) - len(base_labels)
        if arm_value_count == 10:
            labels.extend([f"left axis {axis}" for axis in ("a", "b", "c", "d", "e")])
            labels.extend([f"right axis {axis}" for axis in ("a", "b", "c", "d", "e")])
        else:
            labels.extend([f"dof {index}" for index in range(len(base_labels), len(values))])
        return labels[: len(values)]

    def _rebuild_dof_weight_controls(self, values):
        self._clear_layout(self._dof_weights_layout)
        self._dof_weight_entries = []

        if not values:
            self._dof_weights_empty_label = QLabel("No DOF weights loaded")
            self._dof_weights_layout.addWidget(self._dof_weights_empty_label, 0, 0)
            return

        labels = self._labels_for_dof_weights(values)
        for index, value in enumerate(values):
            label = QLabel(labels[index] if index < len(labels) else str(index))
            label.setMaximumWidth(82)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            spin = QDoubleSpinBox()
            spin.setRange(DOF_WEIGHT_MIN, DOF_WEIGHT_MAX)
            spin.setDecimals(3)
            spin.setSingleStep(0.05)
            spin.setMaximumWidth(90)
            spin.setValue(min(max(value, DOF_WEIGHT_MIN), DOF_WEIGHT_MAX))
            row = index // 3
            col = (index % 3) * 2
            self._dof_weights_layout.setColumnMinimumWidth(col, 78)
            self._dof_weights_layout.setColumnMinimumWidth(col + 1, 92)
            self._dof_weights_layout.addWidget(label, row, col)
            self._dof_weights_layout.addWidget(spin, row, col + 1)
            self._dof_weight_entries.append(spin)

    def _read_solver_parameters(self, show_errors=True):
        node_name = self._controller_node_name()
        client = self._node.create_client(GetParameters, f"{node_name}/get_parameters")
        try:
            if not client.wait_for_service(timeout_sec=0.2):
                if show_errors:
                    self._show_error(f"get_parameters service is not available for {node_name}")
                return

            req = GetParameters.Request()
            req.names = ["solver_method", "dls_lambda", "dof_weights"]
            response = self._spin_until_complete(client.call_async(req))
            if response is None or len(response.values) != 3:
                if show_errors:
                    self._show_error("Failed to read solver parameters")
                return

            method_value = response.values[0]
            lambda_value = response.values[1]
            dof_weights_value = response.values[2]
            if method_value.type != ParameterType.PARAMETER_STRING:
                if show_errors:
                    self._show_error("solver_method is not a string parameter")
                return
            if lambda_value.type != ParameterType.PARAMETER_DOUBLE:
                if show_errors:
                    self._show_error("dls_lambda is not a double parameter")
                return
            if dof_weights_value.type != ParameterType.PARAMETER_DOUBLE_ARRAY:
                if show_errors:
                    self._show_error("dof_weights is not a double array parameter")
                return

            method_index = self._solver_method_combo.findData(method_value.string_value)
            if method_index >= 0:
                self._solver_method_combo.blockSignals(True)
                self._solver_method_combo.setCurrentIndex(method_index)
                self._solver_method_combo.blockSignals(False)

            self._set_dls_lambda_value(lambda_value.double_value)
            self._rebuild_dof_weight_controls(list(dof_weights_value.double_array_value))
        finally:
            self._node.destroy_client(client)

    def _on_solver_method_changed(self):
        pass

    def _on_dls_lambda_slider_changed(self):
        self._dls_lambda_value_label.setText(self._format_float(self._dls_lambda_value(), 4))

    def _apply_solver_parameters(self):
        if not self._set_solver_config_client.wait_for_service(timeout_sec=0.2):
            self._show_error("set_solver_config service is not available")
            return

        req = SetSolverConfig.Request()
        req.solver_method = self._solver_method_combo.currentData()
        req.dls_lambda = self._dls_lambda_value()
        req.update_dof_weights = bool(self._dof_weight_entries)
        req.dof_weights = [spin.value() for spin in self._dof_weight_entries]

        response = self._spin_until_complete(self._set_solver_config_client.call_async(req))
        if response is None:
            self._show_error("Timed out setting solver config")
            return
        if not response.success:
            self._show_error(response.message or "Controller rejected solver config")

    def _read_gain_parameters(self, show_errors=True):
        if not self._last_ordered_task_ids:
            self._refresh_tasks()
        if not self._last_ordered_task_ids:
            return

        names = []
        for task_id in self._last_ordered_task_ids:
            names.extend([f"tasks.{task_id}.gain", f"tasks.{task_id}.gain_scalar"])

        node_name = self._controller_node_name()
        client = self._node.create_client(GetParameters, f"{node_name}/get_parameters")
        try:
            if not client.wait_for_service(timeout_sec=0.2):
                if show_errors:
                    self._show_error(f"get_parameters service is not available for {node_name}")
                return

            req = GetParameters.Request()
            req.names = names
            response = self._spin_until_complete(client.call_async(req))
            if response is None or len(response.values) != len(names):
                if show_errors:
                    self._show_error("Failed to read gain parameters")
                return

            gain_specs = []
            for task_id in self._last_ordered_task_ids:
                gain_name = f"tasks.{task_id}.gain"
                scalar_name = f"tasks.{task_id}.gain_scalar"
                gain_value = response.values[names.index(gain_name)]
                scalar_value = response.values[names.index(scalar_name)]
                task = self._task_statuses.get(task_id)
                if gain_value.type == ParameterType.PARAMETER_DOUBLE_ARRAY and gain_value.double_array_value:
                    gain_specs.append(
                        (task_id, "gain", self._labels_for_gain(task, gain_value.double_array_value), list(gain_value.double_array_value))
                    )
                if scalar_value.type == ParameterType.PARAMETER_DOUBLE:
                    plugin = task.plugin if task is not None else ""
                    if "JointLimitsTask" in plugin or "BaseYawTask" in plugin:
                        gain_specs.append((task_id, "gain_scalar", ["gain_scalar"], [scalar_value.double_value]))

            self._rebuild_gain_controls(gain_specs)
        finally:
            self._node.destroy_client(client)

    def _labels_for_gain(self, task, values):
        if task is not None and task.joint_names and len(task.joint_names) == len(values):
            return list(task.joint_names)
        if len(values) == 6:
            return ["x", "y", "z", "roll", "pitch", "yaw"]
        if len(values) == 3:
            return ["x", "y", "z"]
        return [str(i) for i in range(len(values))]

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)

    def _rebuild_gain_controls(self, gain_specs):
        self._clear_layout(self._gains_layout)
        self._gain_entries = {}

        if not gain_specs:
            self._gains_empty_label = QLabel("No gain parameters loaded")
            self._gains_layout.addWidget(self._gains_empty_label)
            self._gains_layout.addStretch(1)
            return

        for task_id, field, labels, values in gain_specs:
            box = QGroupBox(f"{task_id} / {field}")
            grid = QGridLayout(box)
            grid.setHorizontalSpacing(6)
            grid.setVerticalSpacing(4)
            spinboxes = []
            for index, value in enumerate(values):
                label = QLabel(labels[index] if index < len(labels) else str(index))
                label.setMaximumWidth(72)
                label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                spin = QDoubleSpinBox()
                spin.setRange(GAIN_MIN, GAIN_MAX)
                spin.setDecimals(3)
                spin.setSingleStep(0.05)
                spin.setMaximumWidth(90)
                spin.setValue(min(max(value, GAIN_MIN), GAIN_MAX))
                row = index // 3
                col = (index % 3) * 2
                grid.setColumnMinimumWidth(col, 64)
                grid.setColumnMinimumWidth(col + 1, 92)
                grid.setColumnStretch(col, 0)
                grid.setColumnStretch(col + 1, 0)
                grid.addWidget(label, row, col)
                grid.addWidget(spin, row, col + 1)
                spinboxes.append(spin)
            self._gain_entries[(task_id, field)] = spinboxes
            self._gains_layout.addWidget(box)
        self._gains_layout.addStretch(1)

    def _apply_gain_parameters(self):
        if not self._gain_entries:
            return

        updates = []
        for (task_id, field), spinboxes in self._gain_entries.items():
            update = TaskGainUpdate()
            update.task_id = task_id
            update.field = field
            values = [spin.value() for spin in spinboxes]
            if field == "gain_scalar":
                update.values = [values[0]]
            else:
                update.values = values
            updates.append(update)

        if not self._set_task_gains_client.wait_for_service(timeout_sec=0.2):
            self._show_error("set_task_gains service is not available")
            return

        req = SetTaskGains.Request()
        req.updates = updates
        response = self._spin_until_complete(self._set_task_gains_client.call_async(req))
        if response is None:
            self._show_error("Timed out setting task gains")
            return
        if not response.success:
            self._show_error(response.message or "Controller rejected gain parameters")

    def _on_hierarchy_state(self, msg):
        self._status_label.setText(
            f"Backend: {msg.backend_name} | Solver: {msg.solver_method} | Ready: {msg.ready}"
        )
        self._task_statuses = {task.id: task for task in msg.tasks}
        self._update_task_rows_from_statuses()
        self._refresh_errors_table()

    def _on_task_state(self, task_id, msg):
        self._latest_task_states[task_id] = msg
        self._refresh_errors_table()

    def _on_controller_output(self, msg):
        self._latest_controller_output = msg
        self._refresh_controller_output_table()

    def _ensure_task_state_subscription(self, task_id):
        if task_id in self._task_state_subs:
            return
        topic_name = f"{TASK_TOPIC_PREFIX}/{task_id}/state"
        self._task_state_subs[task_id] = self._node.create_subscription(
            TaskState,
            topic_name,
            functools.partial(self._on_task_state, task_id),
            10,
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
        self._task_statuses = {task.id: task for task in response.tasks}
        self._last_ordered_task_ids = []
        for row, task in enumerate(response.tasks):
            self._last_ordered_task_ids.append(task.id)
            self._ensure_task_state_subscription(task.id)
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
        self._update_goal_task_combo(targetable_tasks)
        self._refresh_errors_table()

    def _update_goal_task_combo(self, targetable_tasks):
        previous_task_id = self._goal_task_combo.currentData()
        self._targetable_tasks = targetable_tasks
        self._goal_task_combo.blockSignals(True)
        self._goal_task_combo.clear()
        for task_id in self._last_ordered_task_ids:
            if task_id in self._targetable_tasks:
                self._goal_task_combo.addItem(task_id, task_id)
        self._goal_task_combo.blockSignals(False)

        if previous_task_id in self._targetable_tasks:
            self._goal_task_combo.setCurrentIndex(
                self._goal_task_combo.findData(previous_task_id)
            )
        elif self._goal_task_combo.count() > 0:
            self._goal_task_combo.setCurrentIndex(0)

        self._on_goal_task_changed()

    def _update_task_rows_from_statuses(self):
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is None:
                continue
            task_id = item.data(Qt.UserRole)
            task = self._task_statuses.get(task_id)
            if task is None:
                continue
            status_item = self._table.item(row, 5)
            if status_item is not None:
                status_item.setText(task.status_message)

    def _refresh_errors_table(self):
        task_ids = list(self._last_ordered_task_ids)
        self._errors_table.setRowCount(len(task_ids))
        for row, task_id in enumerate(task_ids):
            status = self._task_statuses.get(task_id)
            state = self._latest_task_states.get(task_id)
            active = state.active if state is not None else (status.active if status is not None else False)
            error = state.error if state is not None else (status.error if status is not None else [])
            command = state.velocity if state is not None else (status.command if status is not None else [])
            values = [
                task_id,
                "yes" if active else "no",
                self._format_values(error),
                self._format_values(command),
            ]
            for col, value in enumerate(values):
                self._errors_table.setItem(row, col, QTableWidgetItem(value))

    def _refresh_controller_output_table(self):
        msg = self._latest_controller_output
        if msg is None:
            return

        rows = (
            ("Base", msg.base_velocity_names, msg.base_velocity),
            ("Left Arm", msg.left_joint_names, msg.left_arm_velocity),
            ("Right Arm", msg.right_joint_names, msg.right_arm_velocity),
        )
        for row, (name, names, values) in enumerate(rows):
            self._controller_output_table.setItem(row, 0, QTableWidgetItem(name))
            self._controller_output_table.setItem(row, 1, QTableWidgetItem(", ".join(names)))
            self._controller_output_table.setItem(
                row,
                2,
                QTableWidgetItem(self._format_vector(values)),
            )

    def _read_controller_parameters(self, names, show_errors=False):
        node_name = self._controller_node_name()
        client = self._node.create_client(GetParameters, f"{node_name}/get_parameters")
        try:
            if not client.wait_for_service(timeout_sec=0.2):
                if show_errors:
                    self._show_error(f"get_parameters service is not available for {node_name}")
                return None

            req = GetParameters.Request()
            req.names = names
            response = self._spin_until_complete(client.call_async(req))
            if response is None or len(response.values) != len(names):
                if show_errors:
                    self._show_error("Failed to read controller parameters")
                return None
            return response.values
        finally:
            self._node.destroy_client(client)

    def _joint_limit_ranges_for_task(self, task):
        ranges = {joint_name: (DEFAULT_JOINT_MIN, DEFAULT_JOINT_MAX) for joint_name in task.joint_names}
        values = self._read_controller_parameters(
            [
                "left_arm_joints",
                "right_arm_joints",
                "tasks.joint_limits.lower_limits",
                "tasks.joint_limits.upper_limits",
            ]
        )
        if values is None:
            return ranges

        if (
            values[0].type != ParameterType.PARAMETER_STRING_ARRAY
            or values[1].type != ParameterType.PARAMETER_STRING_ARRAY
            or values[2].type != ParameterType.PARAMETER_DOUBLE_ARRAY
            or values[3].type != ParameterType.PARAMETER_DOUBLE_ARRAY
        ):
            return ranges

        joint_order = list(values[0].string_array_value) + list(values[1].string_array_value)
        lower_limits = list(values[2].double_array_value)
        upper_limits = list(values[3].double_array_value)
        for index, joint_name in enumerate(joint_order):
            if index >= len(lower_limits) or index >= len(upper_limits):
                continue
            if joint_name in ranges and upper_limits[index] > lower_limits[index]:
                ranges[joint_name] = (lower_limits[index], upper_limits[index])
        return ranges

    def _target_values_for_task(self, task_id, task):
        state = self._latest_task_states.get(task_id)
        if state is not None and len(state.target) == len(task.joint_names):
            return list(state.target)

        values = self._read_controller_parameters([f"tasks.{task_id}.target"])
        if values and values[0].type == ParameterType.PARAMETER_DOUBLE_ARRAY:
            target = list(values[0].double_array_value)
            if len(target) == len(task.joint_names):
                return target

        return [0.0] * len(task.joint_names)

    def _rebuild_joint_target_controls(self, task_id, task):
        self._clear_layout(self._joint_target_layout)
        self._joint_target_controls = []
        ranges = self._joint_limit_ranges_for_task(task)
        values = self._target_values_for_task(task_id, task)

        for row, joint_name in enumerate(task.joint_names):
            lower, upper = ranges.get(joint_name, (DEFAULT_JOINT_MIN, DEFAULT_JOINT_MAX))
            if upper <= lower:
                lower, upper = DEFAULT_JOINT_MIN, DEFAULT_JOINT_MAX
            value = min(max(values[row], lower), upper)

            name_label = QLabel(joint_name)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(self._slider_value_from_double(lower), self._slider_value_from_double(upper))
            slider.setValue(self._slider_value_from_double(value))
            slider.setSingleStep(1)
            slider.setPageStep(100)

            spin = QDoubleSpinBox()
            spin.setRange(lower, upper)
            spin.setDecimals(3)
            spin.setSingleStep(0.01)
            spin.setValue(value)

            slider.valueChanged.connect(functools.partial(self._on_joint_slider_changed, row))
            slider.sliderReleased.connect(self._publish_live_joint_target)
            spin.valueChanged.connect(functools.partial(self._on_joint_spin_changed, row))

            self._joint_target_layout.addWidget(name_label, row, 0)
            self._joint_target_layout.addWidget(slider, row, 1)
            self._joint_target_layout.addWidget(spin, row, 2)
            self._joint_target_controls.append((slider, spin))

    def _on_joint_slider_changed(self, row, value):
        if row >= len(self._joint_target_controls):
            return
        _, spin = self._joint_target_controls[row]
        spin.blockSignals(True)
        spin.setValue(self._double_from_slider_value(value))
        spin.blockSignals(False)
        self._schedule_live_joint_target()

    def _on_joint_spin_changed(self, row, value):
        if row >= len(self._joint_target_controls):
            return
        slider, _ = self._joint_target_controls[row]
        slider.blockSignals(True)
        slider.setValue(self._slider_value_from_double(value))
        slider.blockSignals(False)
        self._schedule_live_joint_target()

    def _joint_target_values(self):
        return [spin.value() for _, spin in self._joint_target_controls]

    def _schedule_live_joint_target(self):
        if self._live_joint_target_checkbox.isVisible() and self._live_joint_target_checkbox.isChecked():
            self._joint_target_live_timer.start()

    def _publish_live_joint_target(self):
        self._joint_target_live_timer.stop()
        if not self._live_joint_target_checkbox.isVisible() or not self._live_joint_target_checkbox.isChecked():
            return
        task_id = self._goal_task_combo.currentData()
        task = self._targetable_tasks.get(task_id)
        if task is None or task.target_type != "joint_array":
            return
        self._publish_joint_target(task_id, task, show_errors=False)

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

    def _stop_task_priority(self):
        if not self._stop_client.wait_for_service(timeout_sec=0.2):
            self._show_error("stop_task_priority service is not available")
            return

        response = self._spin_until_complete(self._stop_client.call_async(Trigger.Request()))
        if response is None or not response.success:
            self._show_error(response.message if response else "Failed to stop task priority")
            return

        self._refresh_tasks()
        self._read_gain_parameters(show_errors=False)

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
        self._read_gain_parameters(show_errors=False)

    def _on_goal_task_changed(self):
        task_id = self._goal_task_combo.currentData()
        task = self._targetable_tasks.get(task_id)
        if task is None:
            self._goal_target_type.setText("No active target tasks")
            self._goal_joint_info.setText("")
            self._pose_target_widget.setVisible(False)
            self._joint_target_widget.setVisible(False)
            self._clear_layout(self._joint_target_layout)
            self._joint_target_controls = []
            self._live_joint_target_checkbox.setVisible(False)
            self._goal_button.setEnabled(False)
            return

        if task.target_type == "pose":
            self._goal_target_type.setText("PoseStamped")
            self._goal_joint_info.setText("")
            self._pose_target_widget.setVisible(True)
            self._joint_target_widget.setVisible(False)
            self._clear_layout(self._joint_target_layout)
            self._joint_target_controls = []
            self._live_joint_target_checkbox.setVisible(False)
            self._goal_button.setEnabled(True)
            return

        if task.target_type == "joint_array":
            joint_names = ", ".join(task.joint_names)
            self._goal_target_type.setText("Float64MultiArray")
            self._goal_joint_info.setText(joint_names)
            self._pose_target_widget.setVisible(False)
            self._joint_target_widget.setVisible(True)
            self._rebuild_joint_target_controls(task_id, task)
            self._live_joint_target_checkbox.setVisible(True)
            self._goal_button.setEnabled(True)
            return

        self._goal_target_type.setText("Unsupported")
        self._goal_joint_info.setText("")
        self._pose_target_widget.setVisible(False)
        self._joint_target_widget.setVisible(False)
        self._clear_layout(self._joint_target_layout)
        self._joint_target_controls = []
        self._live_joint_target_checkbox.setVisible(False)
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
            self._publish_joint_target(task_id, task, show_errors=True)
            return

        self._show_error("Selected task does not accept runtime targets")

    def _publish_joint_target(self, task_id, task, show_errors=True):
        joint_target = self._joint_target_values()
        if len(joint_target) != len(task.joint_names):
            if show_errors:
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
