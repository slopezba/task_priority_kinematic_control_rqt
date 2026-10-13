"""Single-arm UI goals and pending flow with fake ROS endpoints."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from concurrent.futures import Future
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from python_qt_binding.QtWidgets import QApplication
from sura_manipulation_actions.action import PlanCartesianTrajectory

from task_priority_kinematic_control_rqt import cartesian_tab as module
from task_priority_kinematic_control_rqt import bimanual_cartesian_tab as common
from test_bimanual_cartesian_tab import FakeHandle


class CartesianTabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.node = MagicMock()
        self.node.get_clock.return_value.now.return_value.to_msg.return_value = Time()
        self.client = MagicMock()
        self.client.server_is_ready.return_value = True
        self.acceptance = Future()
        self.client.send_goal_async.return_value = self.acceptance
        self.buffer = MagicMock()
        for target, name, replacement in (
            (common, "Buffer", MagicMock(return_value=self.buffer)),
            (common, "TransformListener", MagicMock()),
            (common, "ActionClient", MagicMock(return_value=self.client)),
            (module, "arm_frames", MagicMock(return_value={"arm_a": "tool_a", "arm_b": "tool_b"})),
            (module, "configured_reference_frame", MagicMock(return_value="base")),
        ):
            patcher = patch.object(target, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.tab = module.CartesianTab(self.node)

    def tearDown(self):
        self.tab.request_shutdown()
        self.tab.destroy_clients()
        self.tab.deleteLater()

    def test_absolute_pose_goal_and_independent_timing_factors(self):
        self.tab.xyz_edit.setText("0.4,-0.2,0.1")
        self.tab.quaternion_edit.setText("0,0,0,2")
        self.tab.velocity_scaling_spin.setValue(0.05)
        self.tab.acceleration_scaling_spin.setValue(0.2)
        goal = self.tab._goal_from_fields()
        self.assertEqual(goal.arm, "arm_a")
        self.assertEqual(goal.target_pose.header.frame_id, "base")
        self.assertEqual(goal.target_pose.pose.position.x, 0.4)
        self.assertEqual(goal.target_pose.pose.orientation.w, 1.0)
        self.assertEqual(goal.velocity_scaling, 0.05)
        self.assertEqual(goal.acceleration_scaling, 0.2)
        self.assertEqual(goal.execution_backend, "")
        self.assertFalse(goal.execute)

    def test_read_tf_uses_base_to_selected_tool_without_relative_subtraction(self):
        tf = TransformStamped()
        tf.transform.translation.x = 0.42
        tf.transform.rotation.w = 2.0
        self.buffer.lookup_transform.return_value = tf
        self.tab.arm_combo.setCurrentText("arm_b")
        self.tab.read_pose_button.click()
        self.assertEqual(self.tab.tool_frame_edit.text(), "tool_b")
        self.assertEqual(self.buffer.lookup_transform.call_args.args[:2], ("base", "tool_b"))
        self.assertEqual(self.tab.xyz_edit.text(), "0.42,0,0")
        self.assertEqual(self.tab.quaternion_edit.text(), "0,0,0,1")
        common.ActionClient.assert_not_called()

    def test_task_backend_goal_uses_single_action_and_waits_for_manual_execute(self):
        self.tab.arm_combo.setCurrentText("arm_b")
        self.tab.backend_combo.setCurrentIndex(2)
        self.tab.send_button.click()
        common.ActionClient.assert_called_once_with(self.node, PlanCartesianTrajectory, module.DEFAULT_ACTION)
        goal = self.client.send_goal_async.call_args.args[0]
        self.assertEqual(goal.arm, "arm_b")
        self.assertEqual(goal.execution_backend, "task")
        handle = FakeHandle()
        self.acceptance.set_result(handle)
        feedback = PlanCartesianTrajectory.Feedback()
        feedback.current_stage = "waiting_for_execute"
        feedback.progress = 0.8
        self.tab._feedback(SimpleNamespace(feedback=feedback), self.tab._generation)
        self.assertFalse(self.tab.backend_combo.isEnabled())
        self.assertTrue(self.tab.execute_button.isEnabled())
        self.node.create_client.assert_not_called()
        self.tab.execute_button.click()
        self.assertEqual(self.node.create_client.call_args.args[1], module.DEFAULT_SERVER + "/execute_pending_trajectory")

    def test_controller_selection_and_strict_zero_constraints(self):
        self.tab.backend_combo.setCurrentIndex(1)
        for spin in self.tab.position_tolerances + self.tab.orientation_tolerances:
            spin.setValue(0.0)
        goal = self.tab._goal_from_fields()
        self.assertEqual(goal.execution_backend, "controller")
        self.assertEqual(goal.position_tolerance.x, 0.0)
        self.assertEqual(goal.orientation_tolerance.z, 0.0)

    def test_invalid_pose_never_dispatches(self):
        self.tab.xyz_edit.setText("nan,0,0")
        self.tab.send_button.click()
        common.ActionClient.assert_not_called()
        self.assertIn("Error:", self.tab.result_label.text())

    def test_cancel_before_acceptance_cancels_single_goal(self):
        self.tab.send_button.click()
        self.tab.cancel_button.click()
        handle = FakeHandle()
        self.acceptance.set_result(handle)
        self.assertEqual(handle.cancel_calls, 1)


if __name__ == "__main__":
    unittest.main()
