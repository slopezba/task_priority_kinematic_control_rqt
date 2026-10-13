"""Exercise Qt and ROS messages with fake action/services; never dispatch to a robot."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from concurrent.futures import Future
import math
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from python_qt_binding.QtWidgets import QApplication
from python_qt_binding.QtCore import QObject
from std_srvs.srv import Trigger
from sura_manipulation_actions.action import PlanBimanualCartesianTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from task_priority_kinematic_control_rqt import bimanual_cartesian_tab as module


class FakeHandle:
    def __init__(self, accepted=True):
        self.accepted = accepted
        self.result = Future()
        self.cancellation = Future()
        self.cancel_calls = 0

    def get_result_async(self):
        return self.result

    def cancel_goal_async(self):
        self.cancel_calls += 1
        return self.cancellation


class CartesianTabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.node = MagicMock()
        self.node.get_clock.return_value.now.return_value.to_msg.return_value = Time()
        self.client = MagicMock()
        self.client.server_is_ready.return_value = True
        self.client.wait_for_server.return_value = False
        self.acceptance = Future()
        self.client.send_goal_async.return_value = self.acceptance
        self.service = MagicMock()
        self.service.service_is_ready.return_value = True
        self.service.wait_for_service.return_value = False
        self.service_reply = Future()
        self.service.call_async.return_value = self.service_reply
        self.node.create_client.return_value = self.service
        self.buffer = MagicMock()
        for name, replacement in (
            ("Buffer", MagicMock(return_value=self.buffer)),
            ("TransformListener", MagicMock()),
            ("arm_frames", lambda: {"arm_a": "tool_a", "arm_b": "tool_b"}),
            ("relative_reference_frame", lambda: "robot_base"),
            ("ActionClient", MagicMock(return_value=self.client)),
        ):
            patcher = patch.object(module, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.tab = module.BimanualCartesianTab(self.node)

    def tearDown(self):
        self.tab.request_shutdown()
        self.tab.destroy_clients()
        self.tab.deleteLater()

    def send_and_accept(self):
        self.tab.send_button.click()
        handle = FakeHandle()
        self.acceptance.set_result(handle)
        return handle

    def feedback(self, stage="waiting_for_execute", generation=None):
        feedback = PlanBimanualCartesianTrajectory.Feedback()
        feedback.current_stage = stage
        feedback.message = "plan ready"
        feedback.progress = 0.8
        self.tab._feedback(SimpleNamespace(feedback=feedback),
                           self.tab._generation if generation is None else generation)

    @staticmethod
    def complete(handle, success=True, status=GoalStatus.STATUS_SUCCEEDED):
        result = PlanBimanualCartesianTrajectory.Result()
        result.success = success
        result.message = "completed" if success else "canceled by operator"
        result.joint_trajectory.joint_names = ["joint_a", "joint_b", "joint_c"]
        point = JointTrajectoryPoint()
        point.time_from_start.sec = 2
        result.joint_trajectory.points = [point]
        handle.result.set_result(SimpleNamespace(status=status, result=result))

    def test_goal_preserves_zero_constraints_and_normalizes_orientation(self):
        self.tab.xyz_edit.setText("0.1,-0.2,0.3")
        self.tab.quaternion_edit.setText("0,0,0,2")
        for spin in self.tab.position_tolerances + self.tab.orientation_tolerances:
            spin.setValue(0.0)
        goal = self.tab._goal_from_fields()
        self.assertEqual((goal.arm_a, goal.arm_b), ("arm_a", "arm_b"))
        self.assertEqual(goal.relative_target.header.frame_id, "robot_base")
        self.assertEqual(goal.relative_target.pose.position.y, -0.2)
        self.assertEqual(goal.relative_target.pose.orientation.w, 1.0)
        self.assertFalse(hasattr(self.tab, "radius_spin"))
        self.assertFalse(hasattr(self.tab, "arm_a_angles"))
        self.assertEqual(goal.relative_position_tolerance.x, 0.0)
        self.assertEqual(goal.relative_orientation_tolerance.y, 0.0)
        self.assertEqual(goal.cartesian_step, 0.005)
        self.assertEqual(goal.planning_time, 10.0)
        self.assertEqual(goal.velocity_scaling, 0.0)
        self.assertEqual(goal.acceleration_scaling, 0.0)
        self.assertFalse(goal.execute)

    def test_speed_factors_are_sent_and_locked_with_pending_plan(self):
        self.tab.velocity_scaling_spin.setValue(0.05)
        self.tab.acceleration_scaling_spin.setValue(0.2)
        handle = self.send_and_accept()
        sent = self.client.send_goal_async.call_args.args[0]
        self.assertEqual(sent.velocity_scaling, 0.05)
        self.assertEqual(sent.acceleration_scaling, 0.2)
        self.feedback()
        self.assertFalse(self.tab.velocity_scaling_spin.isEnabled())
        self.assertFalse(self.tab.acceleration_scaling_spin.isEnabled())
        self.complete(handle)
        self.assertTrue(self.tab.velocity_scaling_spin.isEnabled())
        self.assertTrue(self.tab.acceleration_scaling_spin.isEnabled())

    def test_each_speed_factor_can_independently_use_server_default(self):
        self.assertEqual(self.tab.velocity_scaling_spin.minimum(), 0.0)
        self.assertEqual(self.tab.velocity_scaling_spin.maximum(), 1.0)
        self.assertEqual(self.tab.acceleration_scaling_spin.maximum(), 1.0)
        self.assertEqual(self.tab.velocity_scaling_spin.text(), "Server default")
        self.tab.velocity_scaling_spin.setValue(1.0)
        goal = self.tab._goal_from_fields()
        self.assertEqual(goal.velocity_scaling, 1.0)
        self.assertEqual(goal.acceleration_scaling, 0.0)
        self.tab.velocity_scaling_spin.setValue(0.0)
        self.tab.acceleration_scaling_spin.setValue(0.05)
        goal = self.tab._goal_from_fields()
        self.assertEqual(goal.velocity_scaling, 0.0)
        self.assertEqual(goal.acceleration_scaling, 0.05)

    def test_custom_arms_frames_and_action_endpoint_are_used(self):
        self.tab.arm_a_combo.setEditText("different_arm_a")
        self.tab.arm_b_combo.setEditText("different_arm_b")
        self.tab.arm_a_tip_edit.setText("new_arm_a_tool")
        self.tab.reference_frame_edit.setText("other_robot/base")
        self.tab.action_edit.setText("/other_robot/cartesian_plan")
        self.tab.execute_checkbox.setChecked(True)
        self.tab.send_button.click()
        module.ActionClient.assert_called_once_with(
            self.node, PlanBimanualCartesianTrajectory, "/other_robot/cartesian_plan")
        goal = self.client.send_goal_async.call_args.args[0]
        self.assertEqual(goal.arm_a, "different_arm_a")
        self.assertEqual(goal.relative_target.header.frame_id, "other_robot/base")
        self.assertTrue(goal.execute)
        self.node.create_client.assert_not_called()

    def test_invalid_input_never_creates_action_client(self):
        for text in ("nan,0,0", "1,2", "1,inf,3", "not a number"):
            self.tab.xyz_edit.setText(text)
            self.tab.send_button.click()
            self.assertIn("Error:", self.tab.result_label.text())
        self.tab.xyz_edit.setText("0,0,0")
        self.tab.quaternion_edit.setText("0,0,0,0")
        self.tab.send_button.click()
        self.tab.quaternion_edit.setText("0,0,0,1")
        self.tab.arm_b_combo.setEditText("arm_a")
        self.tab.send_button.click()
        module.ActionClient.assert_not_called()
        self.assertTrue(self.tab.send_button.isEnabled())

    def test_current_relative_pose_reads_both_tools_in_base_at_same_time(self):
        arm_a = TransformStamped()
        arm_a.header.stamp.sec = 42
        arm_a.transform.translation.x = 0.2
        arm_a.transform.rotation.z = math.sin(math.pi / 4)
        arm_a.transform.rotation.w = math.cos(math.pi / 4)
        arm_b = TransformStamped()
        arm_b.transform.translation.x = 0.62
        arm_b.transform.translation.z = -0.03
        arm_b.transform.rotation.w = 1.0
        self.buffer.lookup_transform.side_effect = [arm_a, arm_b]
        self.tab.read_pose_button.click()
        calls = self.buffer.lookup_transform.call_args_list
        self.assertEqual(calls[0].args[:2], ("robot_base", "tool_a"))
        self.assertEqual(calls[1].args[:2], ("robot_base", "tool_b"))
        self.assertEqual(calls[1].args[2].nanoseconds, 42_000_000_000)
        actual = module.numeric_vector(self.tab.xyz_edit.text(), 3, "XYZ")
        for value, expected in zip(actual, (0.42, 0.0, -0.03)):
            self.assertAlmostEqual(value, expected)
        quaternion = module.numeric_vector(self.tab.quaternion_edit.text(), 4, "Quaternion")
        self.assertAlmostEqual(quaternion[2], -math.sin(math.pi / 4))
        self.assertAlmostEqual(quaternion[3], math.cos(math.pi / 4))
        module.ActionClient.assert_not_called()

    def test_swapping_roles_keeps_reference_and_inverts_relative_components(self):
        arm_a = TransformStamped().transform
        arm_b = TransformStamped().transform
        arm_a.rotation.y, arm_a.rotation.w = math.sin(0.3), math.cos(0.3)
        arm_b.rotation.x, arm_b.rotation.w = math.sin(-0.2), math.cos(-0.2)
        arm_b.translation.x = 0.5
        first_p, first_q = module.relative_pose_components(arm_a, arm_b)
        swapped_p, swapped_q = module.relative_pose_components(arm_b, arm_a)
        self.assertEqual(swapped_p, [-value for value in first_p])
        for value, expected in zip(swapped_q, [-first_q[0], -first_q[1], -first_q[2], first_q[3]]):
            self.assertAlmostEqual(value, expected)
        self.tab.arm_a_combo.setEditText("arm_b")
        self.tab.arm_b_combo.setEditText("arm_a")
        self.assertEqual(self.tab._goal_from_fields().relative_target.header.frame_id, "robot_base")

    def test_tf_failure_does_not_change_target_or_send_goal(self):
        self.tab.xyz_edit.setText("1,2,3")
        self.buffer.lookup_transform.side_effect = module.TransformException("missing TF")
        self.tab.read_pose_button.click()
        self.assertEqual(self.tab.xyz_edit.text(), "1,2,3")
        self.assertIn("TF is unavailable", self.tab.result_label.text())
        module.ActionClient.assert_not_called()

    def test_pending_goal_waits_for_explicit_execute_and_rejection_is_retryable(self):
        handle = self.send_and_accept()
        self.tab._send_goal()
        self.client.send_goal_async.assert_called_once()
        self.feedback()
        self.assertEqual(self.tab.progress.value(), 80)
        self.assertTrue(self.tab.execute_button.isEnabled())
        self.node.create_client.assert_not_called()
        self.tab.execute_button.click()
        self.node.create_client.assert_called_once_with(
            Trigger, module.DEFAULT_SERVER + "/execute_pending_trajectory")
        self.assertFalse(self.tab.execute_button.isEnabled())
        response = Trigger.Response(success=False, message="backend inactive")
        self.service_reply.set_result(response)
        self.assertTrue(self.tab.execute_button.isEnabled())
        self.assertIn("backend inactive", self.tab.result_label.text())
        self.assertIs(self.tab._goal_handle, handle)
        self.node.destroy_client.assert_called_once_with(self.service)

    def test_cancel_before_acceptance_cancels_late_accepted_goal(self):
        self.tab.send_button.click()
        self.tab.cancel_button.click()
        handle = FakeHandle()
        self.acceptance.set_result(handle)
        self.assertEqual(handle.cancel_calls, 1)
        handle.cancellation.set_result(SimpleNamespace(goals_canceling=[object()]))
        self.complete(handle, False, GoalStatus.STATUS_CANCELED)
        self.assertTrue(self.tab.send_button.isEnabled())
        self.assertFalse(self.tab.execute_button.isEnabled())
        self.assertIn("canceled:", self.tab.result_label.text())

    def test_goal_rejection_restores_editing(self):
        self.tab.send_button.click()
        self.acceptance.set_result(FakeHandle(accepted=False))
        self.assertTrue(self.tab.send_button.isEnabled())
        self.assertTrue(self.tab.action_edit.isEnabled())
        self.assertIn("rejected", self.tab.result_label.text())

    def test_terminal_result_is_not_overwritten_by_late_execute_reply(self):
        handle = self.send_and_accept()
        self.feedback()
        self.tab.execute_button.click()
        self.complete(handle)
        summary = self.tab.result_label.text()
        self.assertIn("3 joints, 1 points, 2.000 s", summary)
        self.assertFalse(self.tab.send_button.isEnabled())
        self.service_reply.set_result(Trigger.Response(success=True, message="requested"))
        self.assertEqual(self.tab.result_label.text(), summary)
        self.assertTrue(self.tab.send_button.isEnabled())

    def test_stale_feedback_and_cancel_reply_do_not_change_new_goal(self):
        handle = self.send_and_accept()
        first_generation = self.tab._generation
        self.tab.cancel_button.click()
        self.complete(handle)
        self.client.send_goal_async.return_value = Future()
        self.tab.send_button.click()
        self.feedback(generation=first_generation)
        handle.cancellation.set_result(SimpleNamespace(goals_canceling=[]))
        self.assertFalse(self.tab._cancel_requested)
        self.assertFalse(self.tab._waiting)
        self.assertEqual(self.tab.feedback_label.text(), "Sending Cartesian goal...")

    def test_closing_panel_cancels_even_if_acceptance_is_pending(self):
        self.tab.send_button.click()
        self.tab.request_shutdown()
        self.assertFalse(self.tab.shutdown_complete())
        handle = FakeHandle()
        self.acceptance.set_result(handle)
        self.assertEqual(handle.cancel_calls, 1)
        handle.cancellation.set_result(SimpleNamespace(goals_canceling=[object()]))
        self.assertTrue(self.tab.shutdown_complete())

    def test_unavailable_server_does_not_send(self):
        self.client.server_is_ready.return_value = False
        self.tab.send_button.click()
        self.client.send_goal_async.assert_not_called()
        self.assertTrue(self.tab.send_button.isEnabled())
        self.assertIn("Cartesian lifecycle server", self.tab.result_label.text())

    def test_initial_action_and_service_discovery_can_complete_before_dispatch(self):
        self.client.server_is_ready.return_value = False
        self.client.wait_for_server.return_value = True
        self.send_and_accept()
        self.client.wait_for_server.assert_called_once_with(timeout_sec=0.5)
        self.feedback()
        self.service.service_is_ready.return_value = False
        self.service.wait_for_service.return_value = True
        self.tab.execute_button.click()
        self.service.wait_for_service.assert_called_once_with(timeout_sec=0.5)
        self.service.call_async.assert_called_once()


class PanelIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_existing_tabs_and_shared_ros_node_are_preserved(self):
        from task_priority_kinematic_control_rqt import task_priority_panel as panel

        class Context(QObject):
            def add_widget(self, widget):
                self.widget = widget

        context = Context()
        node = MagicMock()
        with patch.object(panel, "Node", return_value=node), \
                patch.object(panel, "ActionClient"), \
                patch.object(panel.rclpy, "get_default_context") as ros_context, \
                patch.object(panel.TaskPriorityPanel, "_refresh_tasks"), \
                patch.object(panel.TaskPriorityPanel, "_read_solver_parameters"), \
                patch.object(panel.TaskPriorityPanel, "_read_gain_parameters"), \
                patch.object(module, "Buffer"), patch.object(module, "TransformListener"):
            ros_context.return_value.ok.return_value = True
            plugin = panel.TaskPriorityPanel(context)
            plugin._spin_timer.stop()
            names = [plugin._tabs.tabText(i) for i in range(plugin._tabs.count())]
            self.assertEqual(names, ["Solver", "Gains", "Pose Goal", "Errors",
                                     "Predefined Poses", "Cartesian", "Bimanual Cartesian"])
            self.assertIs(plugin._cartesian_tab._node, node)
            self.assertIs(plugin._bimanual_cartesian_tab._node, node)
            self.assertEqual(module.TransformListener.call_count, 2)
            module.TransformListener.assert_any_call(
                plugin._bimanual_cartesian_tab._tf_buffer, node, spin_thread=False)
            plugin.shutdown_plugin()
            node.destroy_node.assert_called_once()
            context.widget.close()


if __name__ == "__main__":
    unittest.main()
