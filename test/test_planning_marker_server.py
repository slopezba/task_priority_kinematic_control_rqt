"""The marker authority runs without Qt; RQT communicates only via ROS interfaces."""
from concurrent.futures import Future
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import fields
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import uuid
import xml.etree.ElementTree as ET

from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import TransformStamped
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from visualization_msgs.msg import InteractiveMarkerFeedback
from visualization_msgs.srv import GetInteractiveMarkers

from task_priority_kinematic_control_rqt.marker_geometry import (
    pose_components, pose_from_components, reconstruct_b, relative_pose, transform_pose,
)
from task_priority_kinematic_control_rqt.planning_marker_node import MarkerSession, PlanningMarkerNode, Target


class Request:
    EDIT, READ_TF, LOCK, RELEASE, BIND_GOAL, HEARTBEAT = range(6)
    def __init__(self):
        self.operation, self.client_id, self.expected_revision = 0, '', 0
        self.target = Target()


API = SimpleNamespace(Request=Request)


def initial_target(bimanual=False):
    poses = [pose_from_components([0.2, 0.1, 0.0], [0, 0, 0.7071067811865476, 0.7071067811865476])]
    if bimanual:
        poses.append(pose_from_components([0.7, -0.2, 0.3], [0.7071067811865476, 0, 0, 0.7071067811865476]))
    target = Target(bimanual=bimanual, marker_namespace='/test/pair' if bimanual else '/test/cartesian',
                    frame_id='base' if bimanual else 'world_ned',
                    arm_names=['left', 'right'] if bimanual else ['left'],
                    tool_frames=['left_tip', 'right_tip'] if bimanual else ['left_tip'], tool_poses=poses)
    target.target_pose.header.frame_id = target.frame_id
    target.target_pose.pose = relative_pose(*poses) if bimanual else poses[0]
    return target


class SessionTest(unittest.TestCase):
    def setUp(self):
        self.node, self.buffer = MagicMock(), MagicMock()
        self.node.get_clock.return_value.now.return_value = Time(seconds=42)
        tf = TransformStamped()
        tf.header.stamp.sec = 40
        tf.transform.rotation.w = 1.0
        self.buffer.lookup_transform.return_value = tf
        self.sessions = []

    def tearDown(self):
        for session in self.sessions:
            session.close()

    def session(self, bimanual=False):
        session = MarkerSession(self.node, self.buffer, initial_target(bimanual))
        self.sessions.append(session)
        return session

    def operation(self, session, operation, target=None, client='editor'):
        request = Request()
        request.operation, request.client_id = operation, client
        request.expected_revision = session.target.revision
        request.target = deepcopy(target or session.target)
        response = SimpleNamespace(success=False, message='', state=None)
        host = SimpleNamespace(edit_type=API, marker_state_type=Target)
        return PlanningMarkerNode.update_target(host, session, request, response)

    def test_markers_exist_and_answer_without_an_editor(self):
        session = self.session(True)
        response = session.server.getInteractiveMarkersCallback(None, GetInteractiveMarkers.Response())
        self.assertEqual(len(response.markers), 2)
        self.assertTrue(all(len(m.controls) == 7 for m in response.markers))
        self.assertTrue(all(len(m.controls[0].markers) == 3 for m in response.markers))
        self.assertNotIn('python_qt_binding', sys.modules.get(MarkerSession.__module__).__dict__)

    def test_read_tf_aligns_gripper_meshes_without_changing_tool_targets(self):
        # Derive the expected geometry from the robot description, including
        # tool_joint's rotation/translation relative to the mesh origin.
        urdf = ET.parse(Path(__file__).resolve().parents[2] /
                        'alpha/alpha_description/xacro/end_effectors/standard_jaws.urdf.xacro')
        def joint_pose(name):
            origin = urdf.find(f'.//joint[@name="${{prefix}}{name}"]/origin')
            position = [float(v) for v in origin.attrib['xyz'].split()]
            roll, pitch, yaw = [float(v) for v in origin.attrib['rpy'].split()]
            self.assertEqual((roll, yaw), (0.0, 0.0))
            return pose_from_components(position, [0, math.sin(pitch / 2), 0, math.cos(pitch / 2)])
        tool_in_base = joint_pose('tool_joint')
        meshes_in_base = [pose_from_components([0, 0, 0], [0, 0, 0, 1]),
                          joint_pose('rs1_130_joint'), joint_pose('rs1_139_joint')]
        for bimanual in (False, True):
            with self.subTest(bimanual=bimanual):
                session = self.session(bimanual)
                bases, tools, transforms = [], [], {}
                for i, tool in enumerate(session.target.tool_frames):
                    base = TransformStamped()
                    base.transform.translation.x = 0.8 + i
                    base.transform.translation.y = -0.2
                    base.transform.translation.z = 0.4
                    rotation = [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)] if i == 0 else \
                        [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
                    for axis, value in zip('xyzw', rotation):
                        setattr(base.transform.rotation, axis, value)
                    tool_pose = transform_pose(tool_in_base, base.transform)
                    tf = TransformStamped()
                    tf.header.frame_id, tf.child_frame_id = session.target.frame_id, tool
                    tf.header.stamp.sec = 40
                    for axis in 'xyz':
                        setattr(tf.transform.translation, axis, getattr(tool_pose.position, axis))
                    tf.transform.rotation = deepcopy(tool_pose.orientation)
                    bases.append(base)
                    tools.append(tool_pose)
                    transforms[tool] = tf
                self.buffer.lookup_transform.side_effect = lambda frame, tool, stamp: transforms[tool]
                self.assertTrue(self.operation(session, Request.READ_TF).success)
                self.assertEqual(session.target.tool_poses, tools)
                expected = relative_pose(*tools) if bimanual else tools[0]
                self.assertEqual(session.target.target_pose.pose, expected)
                response = session.server.getInteractiveMarkersCallback(None, GetInteractiveMarkers.Response())
                for marker in response.markers:
                    i = int(marker.name.rsplit('_', 1)[1])
                    self.assertEqual(marker.pose, tools[i])
                    for mesh, mesh_in_base in zip(marker.controls[0].markers, meshes_in_base):
                        actual = transform_pose(mesh.pose, transforms[session.target.tool_frames[i]].transform)
                        expected = transform_pose(mesh_in_base, bases[i].transform)
                        for got, want in zip(sum(pose_components(actual), []), sum(pose_components(expected), [])):
                            self.assertAlmostEqual(got, want)

    def test_relative_marker_drag_and_numeric_edit_use_reference_axes(self):
        session = self.session(True)
        target = session.target
        for got, expected in zip(pose_components(target.target_pose.pose)[1], [0.5, 0.5, -0.5, 0.5]):
            self.assertAlmostEqual(got, expected)
        feedback = InteractiveMarkerFeedback()
        feedback.marker_name = f'target_{session.epoch}_1'
        feedback.header.frame_id = 'base'
        feedback.event_type = InteractiveMarkerFeedback.POSE_UPDATE
        feedback.pose = deepcopy(target.tool_poses[1])
        feedback.pose.position.x += 0.1
        session.feedback(feedback)
        self.assertAlmostEqual(target.target_pose.pose.position.x, 0.6)
        before = deepcopy(target.tool_poses[0])
        edit = deepcopy(target)
        edit.target_pose.pose = pose_from_components([0.4, 0.2, -0.1], [0, 0, 0, 2])
        self.assertTrue(self.operation(session, Request.EDIT, edit).success)
        self.assertEqual(session.target.tool_poses[0], before)
        for got, expected in zip(pose_components(session.target.tool_poses[1])[0], [0.6, 0.3, -0.1]):
            self.assertAlmostEqual(got, expected)

    def test_swap_roles_preserves_physical_poses(self):
        session = self.session(True)
        old = deepcopy(session.target.tool_poses)
        edit = deepcopy(session.target)
        edit.arm_names.reverse()
        edit.tool_frames.reverse()
        self.assertTrue(self.operation(session, Request.EDIT, edit).success)
        self.assertEqual(session.target.tool_poses, old[::-1])
        self.assertAlmostEqual(session.target.target_pose.pose.position.x, -0.5)

    def test_invalid_edit_and_stale_revision_do_not_replace_target(self):
        session = self.session()
        old = deepcopy(session.target)
        edit = deepcopy(old)
        edit.target_pose.pose.orientation.w = edit.target_pose.pose.orientation.z = 0.0
        response = self.operation(session, Request.EDIT, edit)
        self.assertFalse(response.success)
        self.assertEqual(session.target, old)
        request = Request()
        request.expected_revision = old.revision + 1
        request.target = deepcopy(old)
        host = SimpleNamespace(edit_type=API, marker_state_type=Target)
        response = PlanningMarkerNode.update_target(host, session, request, SimpleNamespace())
        self.assertFalse(response.success)
        self.assertEqual(session.target, old)

    def test_lock_freezes_world_and_rejects_edits_until_terminal_action(self):
        session = self.session()
        edit = deepcopy(session.target)
        edit.action_name = '/test/plan'
        self.assertTrue(self.operation(session, Request.LOCK, edit).success)
        self.assertTrue(session.target.locked)
        self.assertEqual(session.target.target_pose.header.stamp.sec, 40)
        self.assertFalse(self.operation(session, Request.EDIT).success)
        response = session.server.getInteractiveMarkersCallback(None, GetInteractiveMarkers.Response())
        self.assertEqual(len(response.markers[0].controls), 1)
        self.assertEqual(response.markers[0].header.frame_id, 'world_ned')
        edit = deepcopy(session.target)
        edit.goal_id = list(range(16))
        self.assertTrue(self.operation(session, Request.BIND_GOAL, edit).success)
        status = GoalStatus()
        status.goal_info.goal_id.uuid = edit.goal_id
        status.status = GoalStatus.STATUS_EXECUTING
        session.action_status(GoalStatusArray(status_list=[status]))
        session.owner_seen = time.monotonic() - 10.0
        session.tick()
        self.assertTrue(session.target.locked)
        status.status = GoalStatus.STATUS_SUCCEEDED
        session.action_status(GoalStatusArray(status_list=[status]))
        self.assertFalse(session.target.locked)

    def test_another_editor_cannot_release_a_reserved_goal(self):
        session = self.session()
        edit = deepcopy(session.target)
        edit.action_name = '/test/plan'
        self.assertTrue(self.operation(session, Request.LOCK, edit).success)
        self.assertFalse(self.operation(session, Request.RELEASE, client='other').success)
        self.assertTrue(session.target.locked)
        self.assertTrue(self.operation(session, Request.RELEASE).success)

    def test_action_status_recovers_late_acceptance_after_editor_disappears(self):
        session = self.session()
        edit = deepcopy(session.target)
        edit.action_name = '/test/plan'
        self.assertTrue(self.operation(session, Request.LOCK, edit).success)
        status = GoalStatus()
        status.goal_info.goal_id.uuid = list(range(16))
        status.goal_info.stamp.sec = 43  # Reservation was at ROS time 42.
        status.status = GoalStatus.STATUS_EXECUTING
        session.action_status(GoalStatusArray(status_list=[status]))
        self.assertEqual(list(session.target.goal_id), list(range(16)))
        session.owner_seen = time.monotonic() - 10
        session.tick()
        self.assertTrue(session.target.locked)
        status.status = GoalStatus.STATUS_CANCELED
        session.action_status(GoalStatusArray(status_list=[status]))
        self.assertFalse(session.target.locked)

    def test_real_ros_goal_binding_can_be_published_and_returned(self):
        try:
            from sura_manipulation_actions.msg import PlanningTargetState
        except ImportError:
            self.skipTest('PlanningTargetState interface has not been generated')
        for bimanual in (False, True):
            with self.subTest(bimanual=bimanual):
                session = self.session(bimanual)
                session.freeze('editor', '/test/plan')
                request = Request()
                request.operation, request.client_id = Request.BIND_GOAL, 'editor'
                request.target = session.target.message(PlanningTargetState)
                request.target.goal_id = list(range(16))
                host = SimpleNamespace(edit_type=API, marker_state_type=PlanningTargetState)
                response = PlanningMarkerNode.update_target(host, session, request, SimpleNamespace())
                self.assertTrue(response.success)
                self.assertEqual(bytes(response.state.goal_id), bytes(range(16)))
                self.assertTrue(all(type(value) is int for value in session.target.goal_id))

    def test_recovered_ros_uuid_can_be_published_without_the_editor(self):
        try:
            from sura_manipulation_actions.msg import PlanningTargetState
        except ImportError:
            self.skipTest('PlanningTargetState interface has not been generated')
        session = self.session()
        session.freeze('editor', '/test/plan')
        published = []
        session.changed = lambda target: published.append(target.message(PlanningTargetState))
        status = GoalStatus()
        status.goal_info.goal_id.uuid = list(range(16))
        status.goal_info.stamp.sec = 43
        status.status = GoalStatus.STATUS_EXECUTING
        session.action_status(GoalStatusArray(status_list=[status]))
        self.assertEqual(bytes(published[-1].goal_id), bytes(range(16)))
        self.assertTrue(session.target.locked)

    def test_disconnected_unidentified_goal_keeps_reservation_for_recovery(self):
        session = self.session()
        edit = deepcopy(session.target)
        edit.action_name = '/test/plan'
        self.assertTrue(self.operation(session, Request.LOCK, edit).success)
        session.owner_seen = time.monotonic() - 10
        session.action_status(GoalStatusArray())
        session.tick()
        self.assertTrue(session.target.locked)
        self.assertIn('Editor disconnected', session.target.status)

    def test_missing_tf_does_not_remove_get_markers_service(self):
        target = initial_target()
        target.tool_poses = []
        self.buffer.lookup_transform.side_effect = __import__('tf2_ros').TransformException('missing TF')
        session = MarkerSession(self.node, self.buffer, target)
        self.sessions.append(session)
        session.tick()
        response = session.server.getInteractiveMarkersCallback(None, GetInteractiveMarkers.Response())
        self.assertEqual(response.markers, [])
        self.assertIn('missing TF', session.target.status)

    def test_namespace_change_replaces_only_marker_server(self):
        session = self.session()
        previous = session.server
        edit = deepcopy(session.target)
        edit.marker_namespace = '/other/cartesian'
        self.assertTrue(self.operation(session, Request.EDIT, edit).success)
        self.assertNotEqual(session.server, previous)
        self.assertEqual(session.server.namespace, '/other/cartesian')

    def test_interfaces_match_python_state_without_running_codegen(self):
        from pathlib import Path
        schema = Path(__file__).resolve().parents[2] / 'sura_manipulation_actions/msg/PlanningTargetState.msg'
        names = {line.split()[1] for line in schema.read_text().splitlines() if line and not line.startswith('#')}
        self.assertEqual(names, {f.name for f in fields(Target)})


class NodeConstructionTest(unittest.TestCase):
    def test_node_constructs_both_sessions_without_overwriting_node_properties(self):
        # Exercise the actual constructor before codegen using message-compatible
        # DTOs and mocked ROS entity creation; Node's read-only properties remain.
        import sura_manipulation_actions.msg as messages
        from task_priority_kinematic_control_rqt import planning_marker_node as module
        def init(node, *args, **kwargs):
            node._context = MagicMock()
        with ExitStack() as stack:
            stack.enter_context(patch.object(messages, 'PlanningTargetState', Target, create=True))
            stack.enter_context(patch.dict(sys.modules, {
                'sura_manipulation_actions.srv': SimpleNamespace(UpdatePlanningTarget=API)}))
            stack.enter_context(patch.object(Node, '__init__', init))
            stack.enter_context(patch.object(Node, 'destroy_node', return_value=True))
            stack.enter_context(patch.object(Node, 'declare_parameter',
                lambda self, name, default: SimpleNamespace(value=default)))
            stack.enter_context(patch.object(Node, 'get_clock', return_value=SimpleNamespace(now=lambda: Time(seconds=42))))
            stack.enter_context(patch.object(module, 'Buffer'))
            stack.enter_context(patch.object(module, 'TransformListener'))
            for name in ('create_publisher', 'create_subscription', 'create_service', 'create_timer',
                         'destroy_publisher', 'destroy_subscription', 'destroy_service'):
                stack.enter_context(patch.object(Node, name))
            node = PlanningMarkerNode()
            self.assertEqual(len(node.sessions), 2)
            self.assertEqual(len(node.state_publishers), 2)
            self.assertEqual(len(node.editor_services), 2)
            self.assertTrue(node.destroy_node())


# A separate process owns the real interactive-marker service. It deliberately
# receives high-rate state traffic while the client never loads a Qt application.
SERVER_CODE = '''
import signal
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Float64
from task_priority_kinematic_control_rqt.planning_marker_node import MarkerSession, Target
from task_priority_kinematic_control_rqt.marker_geometry import pose_from_components
assert not any(name.startswith('python_qt_binding') for name in __import__('sys').modules)
rclpy.init(args=[])
node = Node('independent_marker_test_server')
pose = pose_from_components([1,2,3], [0,0,0,1])
target = Target(marker_namespace=__import__('sys').argv[1], arm_names=['arm'], tool_frames=['tool'], tool_poses=[pose])
session = MarkerSession(node, None, target)
pub = node.create_publisher(Float64, target.marker_namespace + '/load', 10)
node.create_subscription(Float64, target.marker_namespace + '/load', lambda msg: None, 10)
timer = node.create_timer(0.002, lambda: pub.publish(Float64(data=1.0)))
stopping = [False]
signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
executor = SingleThreadedExecutor()
executor.add_node(node)
try:
    while not stopping[0] and rclpy.ok():
        executor.spin_once(timeout_sec=0.1)
finally:
    timer.cancel()
    executor.shutdown()
    session.close()
    node.destroy_node()
    rclpy.shutdown()
'''


class IndependentServiceIntegrationTest(unittest.TestCase):
    def test_real_get_markers_responds_under_load_without_rqt(self):
        namespace = '/marker_regression_' + uuid.uuid4().hex[:10]
        with tempfile.TemporaryDirectory(prefix='marker_ros_test_') as log_dir, patch.dict(os.environ, {
                'ROS_DOMAIN_ID': str(150 + os.getpid() % 50), 'ROS_LOCALHOST_ONLY': '1',
                'ROS_LOG_DIR': log_dir}):
            child = subprocess.Popen(['/usr/bin/python3', '-c', SERVER_CODE, namespace],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            context = Context()
            node = None
            executor = None
            try:
                rclpy.init(args=[], context=context)
                node = Node('marker_test_client_' + uuid.uuid4().hex[:8], context=context)
                executor = SingleThreadedExecutor(context=context)
                executor.add_node(node)
                client = node.create_client(GetInteractiveMarkers, namespace + '/get_interactive_markers')
                self.assertTrue(client.wait_for_service(timeout_sec=6.0), 'Independent marker service did not start')
                for _ in range(3):
                    future = client.call_async(GetInteractiveMarkers.Request())
                    executor.spin_until_future_complete(future, timeout_sec=2.0)
                    self.assertTrue(future.done(), 'GetInteractiveMarkers did not respond')
                    self.assertEqual(len(future.result().markers), 1)
                    self.assertEqual(future.result().markers[0].pose.position.x, 1.0)
            finally:
                if executor is not None:
                    executor.shutdown()
                if node is not None:
                    node.destroy_node()
                if context.ok():
                    context.shutdown()
                child.terminate()
                try:
                    _, errors = child.communicate(timeout=3.0)
                except subprocess.TimeoutExpired:
                    child.kill()
                    _, errors = child.communicate()
                if child.returncode not in (0, -15):
                    raise AssertionError(errors)
