"""Cartesian UI behavior with isolated actions, metadata and TF endpoints."""
import math
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from concurrent.futures import Future
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration, Time
from geometry_msgs.msg import PoseStamped, TransformStamped, Vector3
from python_qt_binding.QtWidgets import QApplication
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters

from task_priority_kinematic_control_rqt import cartesian_tab as module
from task_priority_kinematic_control_rqt import bimanual_cartesian_tab as common
from test_bimanual_cartesian_tab import FakeHandle


def ui_goal():
    """Mock the new wire fields so UI tests do not require interface regeneration."""
    return SimpleNamespace(
        arm="", target_pose=PoseStamped(), position_tolerance=Vector3(), orientation_tolerance=Vector3(),
        available_dofs=[], base_limits=SimpleNamespace(lower_limits=[], upper_limits=[],
                                                     max_velocity=[], max_acceleration=[]),
        tracking_tolerances=SimpleNamespace(base_path=[], base_goal=[], joint_path=[], joint_goal=[],
                                           tip_path=[], tip_goal=[], goal_time_tolerance=Duration()),
        cartesian_step=0.0, planning_time=0.0, velocity_scaling=0.0, acceleration_scaling=0.0,
        execution_backend="", execute=False)


class CartesianTabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.node, self.client, self.buffer, self.service = (MagicMock() for _ in range(4))
        self.node.get_clock.return_value.now.return_value.to_msg.return_value = Time(sec=42)
        current_tf = TransformStamped()
        current_tf.header.stamp.sec = 40
        current_tf.transform.rotation.w = 1.0
        self.buffer.lookup_transform.return_value = current_tf
        self.client.server_is_ready.return_value = True
        self.acceptance, self.service_response = Future(), Future()
        self.client.send_goal_async.return_value = self.acceptance
        self.service.service_is_ready.return_value = True
        self.service.call_async.return_value = self.service_response
        self.joint_catalog = {
            "arm_a": ["arm_a/c", "arm_a/b", "arm_a/e"],
            "arm_b": [f"arm_b/j{i}" for i in range(5)],
        }
        self.auto_catalog, self.catalog_requests = True, []
        self.node.create_client.side_effect = self.create_service
        self.action_type = SimpleNamespace(Goal=ui_goal)
        for target, name, replacement in (
            (common, "Buffer", MagicMock(return_value=self.buffer)),
            (common, "TransformListener", MagicMock()),
            (common, "ActionClient", MagicMock(return_value=self.client)),
            (module, "arm_frames", MagicMock(return_value={"arm_a": "tool_a", "arm_b": "tool_b"})),
            (module.CartesianTab, "_action_type", self.action_type),
        ):
            patcher = patch.object(target, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch('task_priority_kinematic_control_rqt.planning_markers.target_interfaces',
                        side_effect=ImportError('isolated manual UI test'))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tab = module.CartesianTab(self.node)
        self.tab._markers.show.setChecked(False)

    def create_service(self, service_type, name):
        if service_type != GetParameters:
            return self.service
        client = MagicMock()
        client.service_is_ready.return_value = True
        def request(req):
            arm = req.names[0].split(".")[1]
            future = Future()
            self.catalog_requests.append((arm, future))
            if self.auto_catalog:
                future.set_result(self.catalog_response(self.joint_catalog[arm]))
            return future
        client.call_async.side_effect = request
        return client

    @staticmethod
    def catalog_response(names):
        return SimpleNamespace(values=[SimpleNamespace(
            type=ParameterType.PARAMETER_STRING_ARRAY, string_array_value=names)])

    @staticmethod
    def feedback(stage="waiting_for_execute"):
        return SimpleNamespace(feedback=SimpleNamespace(current_stage=stage, progress=0.8, message="ready",
                              base_error=[0.01] * 6, joint_error=[0.02] * 3, tip_error=[0.001] * 6))

    def tearDown(self):
        self.tab.request_shutdown()
        self.tab.destroy_clients()
        self.tab.deleteLater()

    def test_default_mask_uses_server_order_and_world_frame(self):
        self.tab.xyz_edit.setText("0.4,-0.2,0.1")
        self.tab.quaternion_edit.setText("0,0,0,2")
        self.tab.velocity_scaling_spin.setValue(0.05)
        self.tab.acceleration_scaling_spin.setValue(0.2)
        goal = self.tab._goal_from_fields()
        self.assertEqual(self.tab._joint_names, ["arm_a/c", "arm_a/b", "arm_a/e"])
        self.assertEqual(goal.available_dofs, [False] * 6 + [True] * 3)
        self.assertEqual(goal.arm, "arm_a")
        self.assertEqual(goal.target_pose.header.frame_id, "world_ned")
        self.assertEqual(goal.target_pose.header.stamp.sec, 0)
        self.assertEqual(goal.target_pose.header.stamp.nanosec, 0)
        self.assertEqual(goal.target_pose.pose.position.x, 0.4)
        self.assertEqual(goal.target_pose.pose.orientation.w, 1.0)
        self.assertEqual(goal.velocity_scaling, 0.05)
        self.assertEqual(goal.acceleration_scaling, 0.2)
        self.assertEqual(goal.execution_backend, "")
        self.assertFalse(goal.execute)

    def test_base_mask_selects_task_and_disables_controller(self):
        self.tab.backend_combo.setCurrentIndex(1)
        self.tab.base_dof_checks[0].setChecked(True)
        self.tab.base_dof_checks[5].setChecked(True)
        self.tab._joint_widgets[1][0].setChecked(False)
        goal = self.tab._goal_from_fields()
        self.assertEqual(goal.available_dofs, [True, False, False, False, False, True, True, False, True])
        self.assertEqual(goal.execution_backend, "task")
        self.assertFalse(self.tab.backend_combo.model().item(1).isEnabled())
        self.tab.base_dof_checks[0].setChecked(False)
        self.tab.base_dof_checks[5].setChecked(False)
        self.assertTrue(self.tab.backend_combo.model().item(1).isEnabled())

    def test_pose_backend_ignores_saved_mobility_and_joint_tracking(self):
        self.tab.base_dof_checks[0].setChecked(True)
        self.tab._joint_widgets[1][0].setChecked(False)
        self.tab.base_bounds_checkbox.setChecked(True)
        self.tab.base_lower_spins[0].setValue(2.0)
        self.tab.base_upper_spins[0].setValue(1.0)
        self.tab.backend_combo.setCurrentIndex(self.tab.backend_combo.findData("pose"))
        self.tab.velocity_scaling_spin.setValue(0.2)
        self.tab.tip_path_spins[0].setValue(0.025)
        goal = self.tab._goal_from_fields()
        self.assertEqual(goal.execution_backend, "pose")
        self.assertEqual(goal.available_dofs, [])
        for field in ("lower_limits", "upper_limits", "max_velocity", "max_acceleration"):
            self.assertEqual(getattr(goal.base_limits, field), [])
        for field in ("base_path", "base_goal", "joint_path", "joint_goal"):
            self.assertEqual(getattr(goal.tracking_tolerances, field), [])
        self.assertEqual(goal.tracking_tolerances.tip_path[0], 0.025)
        self.assertEqual(goal.velocity_scaling, 0.2)
        controls = [self.tab.base_bounds_checkbox, self.tab.refresh_joints_button,
                    self.tab.release_button, self.tab.release_service_edit,
                    *self.tab.base_dof_checks, *self.tab.base_lower_spins, *self.tab.base_upper_spins,
                    *self.tab.base_velocity_spins, *self.tab.base_acceleration_spins,
                    *self.tab.base_path_spins, *self.tab.base_goal_spins,
                    *(widget for widgets in self.tab._joint_widgets for widget in widgets)]
        self.assertTrue(all(not widget.isEnabled() for widget in controls))
        self.assertTrue(all(spin.isEnabled() for spin in self.tab.tip_path_spins + self.tab.tip_goal_spins))
        self.assertTrue(self.tab.velocity_scaling_spin.isEnabled())
        self.tab._release_hold()
        self.assertIsNone(self.tab._release_future)

    def test_pose_backend_needs_no_joint_catalog_and_preserves_joint_choices(self):
        self.tab._joint_widgets[1][0].setChecked(False)
        self.tab.backend_combo.setCurrentIndex(self.tab.backend_combo.findData("pose"))
        self.tab._catalog_arm = None
        self.tab._update_controls()
        self.assertTrue(self.tab.send_button.isEnabled())
        self.assertEqual(self.tab._goal_from_fields().execution_backend, "pose")
        self.tab.backend_combo.setCurrentIndex(self.tab.backend_combo.findData("task"))
        self.assertFalse(self.tab.send_button.isEnabled())
        self.tab._refresh_joint_catalog()
        self.assertTrue(self.tab.send_button.isEnabled())
        self.assertEqual(self.tab._goal_from_fields().available_dofs[-3:], [True, False, True])
        self.assertTrue(all(widget.isEnabled() for widgets in self.tab._joint_widgets for widget in widgets))

    def test_pose_pending_execution_feedback_and_path_result(self):
        self.tab.backend_combo.setCurrentIndex(self.tab.backend_combo.findData("pose"))
        self.tab.send_button.click()
        self.assertEqual(self.client.send_goal_async.call_args.args[0].execution_backend, "pose")
        handle = FakeHandle()
        self.acceptance.set_result(handle)
        self.tab._feedback(self.feedback(), self.tab._generation)
        self.assertTrue(self.tab.execute_button.isEnabled())
        self.assertTrue(all(not widget.isEnabled() for widget in self.tab._inputs))
        self.assertTrue(self.tab.errors_label.text().startswith("Tool XYZ/rotation:"))
        self.assertNotIn("Arm:", self.tab.errors_label.text())
        self.tab.execute_button.click()
        self.service_response.set_result(SimpleNamespace(success=True, message="executing"))
        result = SimpleNamespace(success=True, message="completed",
                                 joint_trajectory=SimpleNamespace(joint_names=[], points=[]),
                                 base_trajectory=SimpleNamespace(points=[]),
                                 desired_path=SimpleNamespace(poses=[PoseStamped(), PoseStamped()]))
        handle.result.set_result(SimpleNamespace(status=GoalStatus.STATUS_SUCCEEDED, result=result))
        self.assertIn("tool poses: 2", self.tab.result_label.text())
        self.assertNotIn("0 joints", self.tab.result_label.text())
        self.assertFalse(self.tab.base_dof_checks[0].isEnabled())
        self.assertTrue(self.tab.send_button.isEnabled())

    def test_joint_choices_follow_names_across_arm_switches(self):
        self.tab._joint_widgets[1][0].setChecked(False)
        self.tab._joint_widgets[1][1].setValue(0.12)
        self.tab.arm_combo.setCurrentText("arm_b")
        self.assertEqual(len(self.tab._goal_from_fields().available_dofs), 11)
        self.tab.arm_combo.setCurrentText("arm_a")
        self.assertFalse(self.tab._joint_widgets[1][0].isChecked())
        self.assertEqual(self.tab._goal_from_fields().tracking_tolerances.joint_path, [0.05, 0.12, 0.05])

    def test_limits_and_tracking_tolerances_are_independent_of_planning(self):
        goal = self.tab._goal_from_fields()
        self.assertEqual(goal.base_limits.lower_limits, [])
        self.assertEqual(goal.base_limits.max_velocity, [1.0] * 3 + [0.5] * 3)
        self.assertEqual(goal.base_limits.max_acceleration, [0.4] * 3 + [0.2] * 3)
        self.assertFalse(self.tab.base_lower_spins[0].isEnabled())
        self.tab.base_bounds_checkbox.setChecked(True)
        self.tab.base_lower_spins[0].setValue(-0.4)
        self.tab.base_upper_spins[0].setValue(0.6)
        self.tab.base_path_spins[0].setValue(0.08)
        self.tab.tip_goal_spins[3].setValue(0.04)
        self.tab.goal_time_spin.setValue(1.25)
        goal = self.tab._goal_from_fields()
        self.assertEqual(goal.base_limits.lower_limits[0], -0.4)
        self.assertEqual(goal.base_limits.upper_limits[0], 0.6)
        self.assertEqual(goal.base_limits.lower_limits[4], -1.56)
        self.assertEqual(goal.base_limits.upper_limits[4], 1.56)
        self.assertEqual(goal.tracking_tolerances.base_path[0], 0.08)
        self.assertEqual(goal.tracking_tolerances.tip_goal[3], 0.04)
        self.assertEqual(goal.tracking_tolerances.goal_time_tolerance.sec, 1)
        self.assertEqual(goal.tracking_tolerances.goal_time_tolerance.nanosec, 250000000)
        self.assertEqual(goal.orientation_tolerance.x, 0.2)
        self.tab.base_lower_spins[0].setValue(0.8)
        with self.assertRaisesRegex(ValueError, "lower bound"):
            self.tab._goal_from_fields()

    def test_read_tf_uses_selected_tool_and_reference(self):
        tf = TransformStamped()
        tf.transform.translation.x = 0.42
        tf.transform.rotation.w = 2.0
        self.buffer.lookup_transform.return_value = tf
        self.tab.arm_combo.setCurrentText("arm_b")
        self.tab.read_pose_button.click()
        self.assertEqual(self.tab.tool_frame_edit.text(), "tool_b")
        self.assertEqual(self.buffer.lookup_transform.call_args.args[:2], ("world_ned", "tool_b"))
        self.assertEqual(self.tab.xyz_edit.text(), "0.42,0,0")
        self.assertEqual(self.tab.quaternion_edit.text(), "0,0,0,1")
        common.ActionClient.assert_not_called()

    def test_changing_frame_transforms_position_and_orientation(self):
        tf = TransformStamped()
        tf.transform.translation.x = 10.0
        tf.transform.rotation.z = math.sqrt(0.5)
        tf.transform.rotation.w = math.sqrt(0.5)
        self.buffer.lookup_transform.return_value = tf
        self.tab.xyz_edit.setText("1,2,3")
        self.tab.reference_frame_edit.setText("base_link")
        self.assertTrue(self.tab._change_reference_frame())
        self.assertEqual(self.buffer.lookup_transform.call_args.args[:2], ("base_link", "world_ned"))
        goal = self.tab._goal_from_fields()
        self.assertEqual(goal.target_pose.header.frame_id, "base_link")
        self.assertAlmostEqual(goal.target_pose.pose.position.x, 8.0)
        self.assertAlmostEqual(goal.target_pose.pose.position.y, 1.0)
        self.assertAlmostEqual(goal.target_pose.pose.position.z, 3.0)
        self.assertAlmostEqual(goal.target_pose.pose.orientation.z, math.sqrt(0.5))
        self.assertEqual(goal.available_dofs[:6], [False] * 6)

    def test_unavailable_frame_restores_previous_target(self):
        self.buffer.lookup_transform.side_effect = module.TransformException("missing frame")
        self.tab.xyz_edit.setText("1,2,3")
        self.tab.reference_frame_edit.setText("missing")
        self.assertFalse(self.tab._change_reference_frame())
        self.assertEqual(self.tab.reference_frame_edit.text(), "world_ned")
        self.assertEqual(self.tab.xyz_edit.text(), "1,2,3")
        self.assertIn("missing frame", self.tab.result_label.text())

    def test_metadata_pending_disables_send_and_late_old_arm_is_ignored(self):
        self.auto_catalog = False
        self.tab._refresh_joint_catalog()
        arm_a_future = self.catalog_requests[-1][1]
        self.assertFalse(self.tab.send_button.isEnabled())
        self.tab.arm_combo.setCurrentText("arm_b")
        arm_b_future = self.catalog_requests[-1][1]
        arm_a_future.set_result(self.catalog_response(["wrong_arm_joint"]))
        self.assertIsNone(self.tab._catalog_arm)
        arm_b_future.set_result(self.catalog_response(self.joint_catalog["arm_b"]))
        self.assertEqual(self.tab._joint_names, self.joint_catalog["arm_b"])
        self.assertTrue(self.tab.send_button.isEnabled())

    def test_invalid_catalog_blocks_dispatch(self):
        self.joint_catalog["arm_a"] = ["duplicate", "duplicate"]
        self.tab._refresh_joint_catalog()
        self.assertFalse(self.tab.send_button.isEnabled())
        with self.assertRaisesRegex(ValueError, "Refresh arm joints"):
            self.tab._goal_from_fields()
        common.ActionClient.assert_not_called()

    def test_task_goal_waits_for_manual_execute_and_locks_all_controls(self):
        self.tab.arm_combo.setCurrentText("arm_b")
        self.tab.base_dof_checks[0].setChecked(True)
        self.tab.send_button.click()
        common.ActionClient.assert_called_once_with(self.node, self.action_type, module.DEFAULT_ACTION)
        goal = self.client.send_goal_async.call_args.args[0]
        self.assertEqual(goal.arm, "arm_b")
        self.assertEqual(goal.execution_backend, "task")
        self.acceptance.set_result(FakeHandle())
        self.tab._feedback(self.feedback(), self.tab._generation)
        self.assertTrue(all(not widget.isEnabled() for widget in self.tab._inputs))
        self.assertTrue(self.tab.execute_button.isEnabled())
        self.assertFalse(self.tab.release_button.isEnabled())
        self.assertIn("0.01", self.tab.errors_label.text())
        self.assertFalse(any(call.args[1].endswith("execute_pending_trajectory")
                             for call in self.node.create_client.call_args_list))
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

    def test_release_is_explicit_and_service_result_restores_controls(self):
        self.tab.release_button.click()
        self.assertEqual(self.node.create_client.call_args.args[1], module.DEFAULT_RELEASE_SERVICE)
        self.assertFalse(self.tab.release_button.isEnabled())
        self.assertFalse(self.tab.send_button.isEnabled())
        self.service_response.set_result(SimpleNamespace(success=True, message="restored"))
        self.assertTrue(self.tab.release_button.isEnabled())
        self.assertTrue(self.tab.send_button.isEnabled())
        self.assertIn("Trajectory hold released", self.tab.result_label.text())
        common.ActionClient.assert_not_called()



if __name__ == "__main__":
    unittest.main()
