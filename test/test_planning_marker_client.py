"""RQT communicates with the external authority; no server or robot is hosted here."""
from concurrent.futures import Future
from copy import deepcopy
import os
import threading
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from action_msgs.msg import GoalStatus
from python_qt_binding.QtWidgets import QApplication, QWidget, QFormLayout, QLineEdit, QComboBox, QPushButton

from task_priority_kinematic_control_rqt import planning_markers as module
from task_priority_kinematic_control_rqt.planning_marker_node import Target
from task_priority_kinematic_control_rqt.marker_geometry import pose_from_components
from test_planning_marker_server import API, Request, initial_target


class MarkerClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tab = QWidget()
        self.tab._node = MagicMock()
        self.tab._busy = False
        self.tab._inputs = []
        self.tab._target_frame = 'world_ned'
        self.tab.xyz_edit, self.tab.quaternion_edit = QLineEdit('0,0,0'), QLineEdit('0,0,0,1')
        self.tab.reference_frame_edit = QLineEdit('world_ned')
        self.tab.action_edit = QLineEdit('/test/cartesian_plan')
        self.tab.tool_frame_edit = QLineEdit('left_tip')
        self.tab.arm_combo = QComboBox()
        self.tab.arm_combo.setEditable(True)
        self.tab.arm_combo.addItems(['left', 'right'])
        self.tab.send_button = QPushButton()
        self.tab._refresh_joint_catalog = MagicMock()
        self.tab._update_controls = MagicMock()
        self.requests = []
        self.client = MagicMock()
        self.client.service_is_ready.return_value = True
        def send(request):
            future = Future()
            self.requests.append((request, future))
            return future
        self.client.call_async.side_effect = send
        with patch.object(module, 'target_interfaces', return_value=(Target, API)), \
                patch.object(module, 'MarkerTransport', return_value=self.client) as transport:
            self.editor = module.PlanningMarkerEditor(self.tab, QFormLayout(self.tab))
            self.transport_args = transport.call_args.args
        self.editor._on_state(initial_target())

    def tearDown(self):
        self.editor.close()
        self.tab.deleteLater()

    def reply(self, state=None, success=True):
        request, future = self.requests[-1]
        response_state = deepcopy(state or request.target)
        if state is None:
            response_state.revision = request.expected_revision + 1
        future.set_result(SimpleNamespace(success=success, message='accepted' if success else 'target changed',
                                          state=response_state))
        self.app.processEvents()

    def test_panel_only_subscribes_and_calls_services(self):
        self.tab._node.create_service.assert_not_called()
        self.tab._node.create_publisher.assert_not_called()
        self.assertFalse(hasattr(self.editor, 'server'))
        self.tab._node.create_subscription.assert_not_called()
        self.tab._node.create_client.assert_not_called()
        self.assertTrue(self.transport_args[3].endswith('/cartesian/update'))
        self.assertEqual(self.tab.xyz_edit.text(), '0.2,0.1,0')

    def test_numeric_edit_waits_for_authoritative_state(self):
        self.tab.xyz_edit.setText('1,2,3')
        self.tab.xyz_edit.editingFinished.emit()
        self.assertEqual(self.requests[-1][0].operation, Request.EDIT)
        self.assertEqual(self.requests[-1][0].target.target_pose.pose.position.x, 1.0)
        self.assertAlmostEqual(self.editor.poses[0].position.x, 0.2)
        self.reply()
        self.assertEqual(self.tab.xyz_edit.text(), '1,2,3')
        self.assertEqual(self.editor.confirmed.target_pose.pose.position.x, 1.0)

    def test_marker_state_changes_fields_without_echo_requests(self):
        state = initial_target()
        state.revision = 5
        state.target_pose.pose = pose_from_components([4,5,6], [0,0,0,2])
        state.tool_poses = [state.target_pose.pose]
        self.editor._on_state(state)
        self.assertEqual(self.tab.xyz_edit.text(), '4,5,6')
        self.assertEqual(self.requests, [])

    def test_newer_drag_wins_over_delayed_edit_response(self):
        self.tab.xyz_edit.setText('1,2,3')
        self.editor.commit_fields()
        delayed = deepcopy(self.requests[-1][0].target)
        delayed.revision = 1
        newer = deepcopy(delayed)
        newer.revision = 2
        newer.target_pose.pose = pose_from_components([9,8,7], [0,0,0,1])
        newer.tool_poses = [newer.target_pose.pose]
        self.editor._on_state(newer)
        self.reply(delayed)
        self.assertEqual(self.editor.confirmed.revision, 2)
        self.assertEqual(self.tab.xyz_edit.text(), '9,8,7')

    def test_read_tf_is_a_remote_operation(self):
        self.assertTrue(self.editor.read_pose())
        self.assertEqual(self.requests[-1][0].operation, Request.READ_TF)
        self.assertFalse(hasattr(self.editor, '_tf_buffer'))

    def test_goal_is_not_sent_until_frozen_target_is_acknowledged(self):
        goal = SimpleNamespace(arm='left', target_pose=None)
        done, failed = MagicMock(), MagicMock()
        self.tab._busy = True
        self.editor.prepare_goal(goal, done, failed)
        self.assertEqual(self.requests[-1][0].operation, Request.LOCK)
        done.assert_not_called()
        locked = deepcopy(self.requests[-1][0].target)
        locked.locked, locked.owner_id = True, self.editor.client_id
        locked.target_pose.header.stamp.sec = 40
        locked.revision = 1
        self.reply(locked)
        done.assert_called_once_with(goal)
        self.assertEqual(goal.target_pose.header.stamp.sec, 40)
        failed.assert_not_called()

    def test_failed_reservation_never_dispatches(self):
        done, failed = MagicMock(), MagicMock()
        self.tab._busy = True
        self.editor.prepare_goal(SimpleNamespace(), done, failed)
        self.reply(success=False)
        done.assert_not_called()
        failed.assert_called_once_with('target changed')

    def test_bind_goal_accepts_the_real_ros_uuid_and_target_message(self):
        try:
            from sura_manipulation_actions.msg import PlanningTargetState
        except ImportError:
            self.skipTest('PlanningTargetState interface has not been generated')
        # ROS fixed arrays yield numpy.uint8 elements, unlike list-based mocks.
        self.tab._busy = True
        state = initial_target().message(PlanningTargetState)
        state.locked, state.owner_id = True, self.editor.client_id
        self.editor._on_state(state)
        goal_id = GoalStatus().goal_info.goal_id
        goal_id.uuid = list(range(16))
        self.editor.bind_goal(SimpleNamespace(goal_id=goal_id))
        request, _ = self.requests[-1]
        self.assertEqual(request.operation, Request.BIND_GOAL)
        self.assertEqual(bytes(request.target.goal_id), bytes(goal_id.uuid))
        self.reply()
        self.assertTrue(self.editor.confirmed.locked)

    def test_stale_server_revision_is_replaced_after_restart(self):
        previous = initial_target()
        previous.revision = 100
        self.editor._on_state(previous)
        restarted = initial_target()
        restarted.revision = 1
        self.editor._on_state(restarted)
        self.assertEqual(self.editor.confirmed.server_id, restarted.server_id)
        self.assertEqual(self.editor.confirmed.revision, 1)

    def test_foreign_lock_is_preserved_when_panel_refreshes_controls(self):
        locked = initial_target()
        locked.locked, locked.owner_id = True, 'another-editor'
        self.editor._on_state(locked)
        self.tab.send_button.setEnabled(True)
        self.editor.set_editable(True)
        self.assertFalse(self.tab.send_button.isEnabled())

    def test_closing_panel_does_not_destroy_external_markers(self):
        self.editor.close()
        self.client.close.assert_called_once()
        self.tab._node.destroy_client.assert_not_called()
        self.tab._node.destroy_subscription.assert_not_called()
        self.tab._node.destroy_service.assert_not_called()

    def test_manual_mode_remains_available_without_marker_server(self):
        self.editor.show.setChecked(False)
        self.requests.clear()
        goal, done, failed = SimpleNamespace(), MagicMock(), MagicMock()
        self.editor.prepare_goal(goal, done, failed)
        done.assert_called_once_with(goal)
        failed.assert_not_called()
        self.assertEqual(self.requests, [])

    def test_queued_goal_is_failed_when_the_prior_edit_times_out(self):
        self.tab.xyz_edit.setText('1,2,3')
        self.editor.commit_fields()
        self.tab._busy = True
        done, failed = MagicMock(), MagicMock()
        self.editor.prepare_goal(SimpleNamespace(), done, failed)
        future, _, callback = self.editor._inflight
        self.editor._inflight = (future, 0.0, callback)
        self.editor._tick()
        done.assert_not_called()
        failed.assert_called_once()

    def test_queued_goal_is_not_dispatched_after_an_edit_conflict(self):
        self.tab.xyz_edit.setText('1,2,3')
        self.editor.commit_fields()
        self.tab._busy = True
        done, failed = MagicMock(), MagicMock()
        self.editor.prepare_goal(SimpleNamespace(), done, failed)
        self.reply(success=False)
        done.assert_not_called()
        failed.assert_called_once_with('target changed')
        self.assertEqual(len(self.requests), 1)

    def test_worker_reply_waits_for_qt_and_is_not_falsely_timed_out(self):
        done, failed = MagicMock(), MagicMock()
        self.tab._busy = True
        self.editor.prepare_goal(SimpleNamespace(), done, failed)
        request, future = self.requests[-1]
        locked = deepcopy(request.target)
        locked.locked, locked.owner_id = True, self.editor.client_id
        future_inflight, _, callback = self.editor._inflight
        self.editor._inflight = (future_inflight, 0.0, callback)
        worker = threading.Thread(target=lambda: future.set_result(
            SimpleNamespace(success=True, message='accepted', state=locked)))
        worker.start()
        worker.join()
        done.assert_not_called()
        self.editor._tick()  # Qt has not delivered the already received response.
        failed.assert_not_called()
        self.app.processEvents()
        done.assert_called_once()
        failed.assert_not_called()

    def test_worker_state_is_applied_only_on_qt_thread(self):
        state = initial_target()
        state.target_pose.pose.position.x = 8.0
        worker = threading.Thread(target=lambda: self.transport_args[4](state))
        worker.start()
        worker.join()
        self.assertEqual(self.tab.xyz_edit.text(), '0.2,0.1,0')
        self.app.processEvents()
        self.assertEqual(self.tab.xyz_edit.text(), '8,0.1,0')

    def test_old_connection_state_is_ignored_after_reconnect(self):
        old_callback = self.transport_args[4]
        with patch.object(module, 'MarkerTransport', return_value=MagicMock()):
            self.editor._connect()
        old_callback(initial_target())
        self.app.processEvents()
        self.assertIsNone(self.editor.confirmed)
