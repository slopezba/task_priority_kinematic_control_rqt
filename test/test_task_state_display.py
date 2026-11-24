"""Task telemetry must reach the errors table despite busy ROS subscriptions."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace
from unittest.mock import Mock, patch

from python_qt_binding.QtWidgets import QApplication, QTableWidget

from task_priority_kinematic_control_rqt import task_priority_panel as module


def test_nominal_telemetry_after_other_ready_callbacks():
    app = QApplication.instance() or QApplication([])
    panel_type = module.TaskPriorityPanel
    # A nominal state can be behind every other task and the hierarchy callback.
    pending = [lambda: None for _ in range(24)]
    table = QTableWidget(0, 4)
    panel = SimpleNamespace(
        _node=Mock(), _task_state_subs=dict.fromkeys(range(24)),
        _latest_task_states={}, _task_statuses={},
        _last_ordered_task_ids=["robot_nominal"], _errors_table=table,
        _update_task_rows_from_statuses=Mock(),
        _refresh_controller_output_table=Mock(),
        _format_float=lambda value: f"{value:.3f}",
    )
    panel._format_values = lambda values: panel_type._format_values(panel, values)
    panel._refresh_errors_table = Mock(
        side_effect=lambda: panel_type._refresh_errors_table(panel))
    state = SimpleNamespace(active=True, error=[0.012, -0.003], velocity=[0.025, -0.01])
    pending.append(lambda: panel_type._on_task_state(panel, "robot_nominal", state))

    def spin(*args, **kwargs):
        if pending:
            pending.pop(0)()

    with patch.object(module.rclpy, "ok", return_value=True), \
            patch.object(module.rclpy, "spin_once", side_effect=spin), \
            patch.object(module.time, "monotonic", return_value=0.0):
        panel_type._spin_ros_once(panel)

    assert table.item(0, 1).text() == "yes"
    assert table.item(0, 2).text() == "0.012, -0.003"
    assert table.item(0, 3).text() == "0.025, -0.010"
    panel._refresh_errors_table.assert_called_once()


def test_busy_ros_stream_leaves_time_for_qt():
    panel = SimpleNamespace(
        _node=Mock(), _task_state_subs=dict.fromkeys(range(30)),
        _update_task_rows_from_statuses=Mock(), _refresh_errors_table=Mock(),
        _refresh_controller_output_table=Mock(),
    )
    with patch.object(module.rclpy, "ok", return_value=True), \
            patch.object(module.rclpy, "spin_once") as spin, \
            patch.object(module.time, "monotonic", side_effect=[0.0, 0.006, 0.012]):
        module.TaskPriorityPanel._spin_ros_once(panel)
    assert spin.call_count == 2
    panel._refresh_errors_table.assert_called_once()
