"""Independent planning target server. No Qt imports and no robot commands."""
from copy import deepcopy
from dataclasses import dataclass, field, fields
import time
import uuid

from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import PoseStamped
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.time import Time
from rclpy.validate_full_topic_name import validate_full_topic_name
from tf2_ros import Buffer, TransformListener, TransformException
from visualization_msgs.msg import InteractiveMarkerFeedback

from .marker_geometry import (
    gripper_marker, normalized_pose, pose_from_transform, relative_pose,
    reconstruct_b, transform_pose,
)


@dataclass
class Target:
    bimanual: bool = False
    server_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    revision: int = 0
    marker_namespace: str = ""
    frame_id: str = "world_ned"
    arm_names: list = field(default_factory=list)
    tool_frames: list = field(default_factory=list)
    tool_poses: list = field(default_factory=list)
    target_pose: PoseStamped = field(default_factory=PoseStamped)
    visible: bool = True
    locked: bool = False
    owner_id: str = ""
    action_name: str = ""
    goal_id: list = field(default_factory=lambda: [0] * 16)
    status: str = "Waiting for tool TF"

    @classmethod
    def from_message(cls, message):
        return cls(**{f.name: deepcopy(getattr(message, f.name)) for f in fields(cls)})

    def message(self, message_type):
        result = message_type()
        for f in fields(self):
            setattr(result, f.name, deepcopy(getattr(self, f.name)))
        return result


def common_tool_poses(buffer, frame, tools):
    latest = [buffer.lookup_transform(frame, tool, Time()) for tool in tools]
    stamps = [tf.header.stamp for tf in latest if tf.header.stamp.sec or tf.header.stamp.nanosec]
    stamp = min(stamps, key=lambda s: (s.sec, s.nanosec)) if stamps else None
    return [pose_from_transform(buffer.lookup_transform(frame, tool, Time.from_msg(stamp)).transform)
            if stamp is not None and tf.header.stamp != stamp and
            (tf.header.stamp.sec or tf.header.stamp.nanosec) else pose_from_transform(tf.transform)
            for tool, tf in zip(tools, latest)]


class MarkerSession:
    """Single-threaded target authority; also usable without the editor wire API."""
    def __init__(self, node, buffer, target, changed=lambda state: None):
        self.node, self.buffer, self.target, self.changed = node, buffer, target, changed
        self.server = InteractiveMarkerServer(node, target.marker_namespace)
        self.frozen = None
        self.epoch = 0
        self.owner_seen = time.monotonic()
        self.status_subscription = None
        self.status_seen = False
        self.active_goals = set()
        self.goal_statuses = {}
        self.reserved_ros_ns = 0
        self.render()

    def publish(self, status=None):
        if status is not None:
            self.target.status = status
        self.target.revision += 1
        self.changed(self.target)

    def validate(self, incoming):
        count = 2 if self.target.bimanual else 1
        if incoming.bimanual != self.target.bimanual or not incoming.frame_id or \
                len(incoming.arm_names) != count or len(incoming.tool_frames) != count or \
                any(not s for s in incoming.arm_names + incoming.tool_frames) or \
                len(set(incoming.arm_names)) != count or len(set(incoming.tool_frames)) != count:
            raise ValueError("Select distinct arms/tool frames and a reference frame")
        validate_full_topic_name(incoming.marker_namespace + '/update')
        if not incoming.marker_namespace.startswith('/') or incoming.marker_namespace.endswith('/'):
            raise ValueError("Use an absolute marker namespace without a trailing slash")

    def edit(self, incoming, read_tf=False, for_goal=False):
        self.validate(incoming)
        old = self.target
        if old.bimanual and incoming.frame_id != old.frame_id:
            raise ValueError('Bimanual reference frame must match the configured marker/planner frame')
        selection_changed = (incoming.arm_names, incoming.tool_frames) != (old.arm_names, old.tool_frames)
        swapped = self.target.bimanual and incoming.arm_names == old.arm_names[::-1] and \
            incoming.tool_frames == old.tool_frames[::-1] and incoming.frame_id == old.frame_id
        if read_tf:
            poses = common_tool_poses(self.buffer, incoming.frame_id, incoming.tool_frames)
        elif swapped and old.tool_poses:
            poses = deepcopy(old.tool_poses[::-1])
        elif selection_changed and old.tool_poses:
            poses = common_tool_poses(self.buffer, incoming.frame_id, incoming.tool_frames)
        elif incoming.tool_poses:
            if len(incoming.tool_poses) != (2 if old.bimanual else 1):
                raise ValueError("Incomplete tool pose list")
            poses = [normalized_pose(p) for p in incoming.tool_poses]
        elif old.tool_poses:
            poses = deepcopy(old.tool_poses)
            if incoming.frame_id != old.frame_id:
                tf = self.buffer.lookup_transform(incoming.frame_id, old.frame_id, Time())
                poses = [transform_pose(p, tf.transform) for p in poses]
        else:
            poses = common_tool_poses(self.buffer, incoming.frame_id, incoming.tool_frames)
        # A relative field edit anchors A; markers explicitly supply both poses.
        if not read_tf and (not selection_changed or for_goal):
            desired = normalized_pose(incoming.target_pose.pose)
            poses = [poses[0], reconstruct_b(poses[0], desired)] if old.bimanual else [desired]
        target = relative_pose(*poses) if old.bimanual else poses[0]
        if incoming.marker_namespace != old.marker_namespace:
            replacement = InteractiveMarkerServer(self.node, incoming.marker_namespace)
            self.server.shutdown()
            self.server = replacement
        self.target = Target.from_message(incoming)
        self.target.server_id = old.server_id
        self.target.tool_poses = poses
        self.target.target_pose = PoseStamped()
        self.target.target_pose.header.frame_id = incoming.frame_id
        self.target.target_pose.pose = target
        self.target.revision = old.revision
        self.target.locked, self.target.owner_id = False, ""
        self.target.goal_id = [0] * 16
        self.frozen = None
        if selection_changed or incoming.frame_id != old.frame_id:
            self.epoch += 1
        self.publish("Relative planning target" if old.bimanual else "Absolute planning target")
        self.render()

    def freeze(self, owner, action_name):
        if not owner or not action_name:
            raise ValueError("A client and action endpoint are required to reserve a target")
        if not self.target.tool_poses:
            raise ValueError("Initialize the tool poses before planning")
        frozen = None
        stamp = self.target.target_pose.header.stamp
        if not self.target.bimanual:
            sources = set(self.target.tool_frames)
            if self.target.frame_id != 'world_ned':
                sources.add(self.target.frame_id)
            latest = [self.buffer.lookup_transform('world_ned', s, Time()) for s in sources if s != 'world_ned']
            stamps = [tf.header.stamp for tf in latest if tf.header.stamp.sec or tf.header.stamp.nanosec]
            stamp = min(stamps, key=lambda s: (s.sec, s.nanosec)) if stamps else self.node.get_clock().now().to_msg()
            pose = self.target.target_pose.pose
            if self.target.frame_id != 'world_ned':
                tf = self.buffer.lookup_transform('world_ned', self.target.frame_id, Time.from_msg(stamp))
                pose = transform_pose(pose, tf.transform)
            frozen = ('world_ned', [deepcopy(pose)])
        if self.status_subscription is not None:
            self.node.destroy_subscription(self.status_subscription)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.status_seen, self.active_goals = False, set()
        self.goal_statuses = {}
        self.status_subscription = self.node.create_subscription(
            GoalStatusArray, action_name.rstrip('/') + '/_action/status', self.action_status, qos)
        self.frozen = frozen
        self.target.target_pose.header.stamp = stamp
        self.target.locked, self.target.owner_id, self.target.action_name = True, owner, action_name
        self.reserved_ros_ns = self.node.get_clock().now().nanoseconds
        self.owner_seen = time.monotonic()
        self.epoch += 1
        self.publish("Reserved for planning/execution")
        self.render()

    def release(self):
        if self.frozen is not None:
            frame, poses = self.frozen
            pose = poses[0]
            if frame != self.target.frame_id:
                try:
                    tf = self.buffer.lookup_transform(self.target.frame_id, frame, Time())
                    pose = transform_pose(pose, tf.transform)
                except (TransformException, ValueError):
                    self.target.frame_id = frame
            self.target.tool_poses = [deepcopy(pose)]
            self.target.target_pose.pose = deepcopy(pose)
            self.target.target_pose.header.frame_id = self.target.frame_id
        self.frozen = None
        self.target.locked, self.target.owner_id = False, ""
        self.target.goal_id = [0] * 16
        self.target.target_pose.header.stamp = Time().to_msg()
        self.epoch += 1
        self.publish("Target editable")
        self.render()

    def action_status(self, message):
        self.status_seen = True
        self.goal_statuses = {bytes(s.goal_info.goal_id.uuid): s.status for s in message.status_list}
        active = (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING, GoalStatus.STATUS_CANCELING)
        self.active_goals = {bytes(s.goal_info.goal_id.uuid) for s in message.status_list if s.status in active}
        # Covers a panel closing after dispatch but before its acceptance
        # callback can bind the UUID. Only a unique newly accepted goal qualifies.
        if self.target.locked and not any(self.target.goal_id):
            candidates = [s for s in message.status_list if
                          s.goal_info.stamp.sec * 1000000000 + s.goal_info.stamp.nanosec >= self.reserved_ros_ns]
            if len(candidates) == 1:
                self.target.goal_id = [int(value) for value in candidates[0].goal_info.goal_id.uuid]
                self.publish('Following action status')
        goal_id = bytes(self.target.goal_id)
        if self.target.locked and any(goal_id):
            for status in message.status_list:
                if bytes(status.goal_info.goal_id.uuid) == goal_id and status.status in (
                        GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_CANCELED):
                    self.release()
                    break

    def tick(self):
        if not self.target.locked and not self.target.tool_poses:
            try:
                self.edit(deepcopy(self.target), read_tf=True)
            except (ValueError, TransformException) as error:
                if self.target.status != str(error):
                    self.publish(str(error))
        if self.target.locked and time.monotonic() - self.owner_seen > 3.0:
            if not self.target.status.startswith('Editor disconnected'):
                self.publish('Editor disconnected; target remains locked until action completion or explicit release')

    def feedback(self, feedback):
        if self.target.locked or feedback.event_type not in (
                InteractiveMarkerFeedback.POSE_UPDATE, InteractiveMarkerFeedback.MOUSE_UP):
            self.render()
            return
        try:
            index = int(feedback.marker_name.rsplit('_', 1)[1])
            if feedback.marker_name != f'target_{self.epoch}_{index}' or feedback.header.frame_id != self.target.frame_id:
                return
            poses = deepcopy(self.target.tool_poses)
            poses[index] = normalized_pose(feedback.pose)
            self.target.tool_poses = poses
            self.target.target_pose.pose = relative_pose(*poses) if self.target.bimanual else poses[0]
            self.publish()
        except (ValueError, IndexError) as error:
            self.publish(str(error))
        self.render()

    def render(self):
        target = self.target
        frame, poses = self.frozen if self.frozen is not None else (target.frame_id, target.tool_poses)
        self.server.clear()
        if target.visible:
            for i, pose in enumerate(poses):
                label = (("A" if i == 0 else "B") + " — " if target.bimanual else "Cartesian — ") + target.arm_names[i]
                color = ((0.2, 0.8, 1.0) if i == 0 else (1.0, 0.55, 0.15)) if target.bimanual else (0.65, 0.35, 1.0)
                self.server.insert(gripper_marker(f'target_{self.epoch}_{i}', label, frame, pose, color,
                                                  not target.locked), feedback_callback=self.feedback)
        self.server.applyChanges()

    def close(self):
        if self.status_subscription is not None:
            self.node.destroy_subscription(self.status_subscription)
        self.server.shutdown()


class PlanningMarkerNode(Node):
    def __init__(self):
        super().__init__('planning_marker_server')
        # Imported at runtime so geometry/core tests do not need interface codegen.
        from sura_manipulation_actions.msg import PlanningTargetState
        from sura_manipulation_actions.srv import UpdatePlanningTarget
        self.sessions, self.state_publishers, self.editor_services = [], [], []
        self.buffer = Buffer(node=self)
        self.listener = TransformListener(self.buffer, self, spin_thread=False)
        robot = self.declare_parameter('robot_namespace', 'cirtesub').value.strip('/')
        prefix = '/' + robot if robot else ''
        arms = self.declare_parameter('arm_names', ['alpha_left', 'alpha_right']).value
        tools = [self.declare_parameter(f'arms.{arm}.tip_frame',
                 f'{robot}/{arm}/standard_jaws_tool'.lstrip('/')).value for arm in arms]
        if not arms:
            raise ValueError('Configure at least one arm')
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.marker_state_type = PlanningTargetState
        self.edit_type = UpdatePlanningTarget
        for bimanual, mode in ((False, 'cartesian'), (True, 'bimanual')):
            if bimanual and len(arms) < 2:
                continue
            suffix = 'bimanual_cartesian_goal_markers' if bimanual else 'cartesian_goal_markers'
            namespace = self.declare_parameter(f'{mode}.marker_namespace', f'{prefix}/manipulation/{suffix}').value
            frame = self.declare_parameter(f'{mode}.reference_frame', f'{robot}/base_link'.lstrip('/') if bimanual else 'world_ned').value
            count = 2 if bimanual else 1
            selected_arms = self.declare_parameter(f'{mode}.arm_names', list(arms[:count])).value
            selected_tools = self.declare_parameter(f'{mode}.tool_frames', list(tools[:count])).value
            target = Target(bimanual=bimanual, marker_namespace=namespace, frame_id=frame,
                            arm_names=list(selected_arms), tool_frames=list(selected_tools))
            endpoint = f'{prefix}/manipulation/planning_targets/{mode}'
            publisher = self.create_publisher(PlanningTargetState, endpoint + '/state', qos)
            session = MarkerSession(self, self.buffer, target,
                changed=lambda state, pub=publisher: pub.publish(state.message(PlanningTargetState)))
            self.sessions.append(session)
            self.state_publishers.append(publisher)
            self.editor_services.append(self.create_service(UpdatePlanningTarget, endpoint + '/update',
                lambda req, res, s=session: self.update_target(s, req, res)))
            session.publish()
        self.watchdog = self.create_timer(0.5, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def update_target(self, session, request, response):
        operation, target = request.operation, session.target
        api = self.edit_type.Request
        try:
            if operation in (api.EDIT, api.READ_TF, api.LOCK):
                if target.locked:
                    raise ValueError('Planning target is already reserved')
                if request.expected_revision != target.revision:
                    raise ValueError('Target changed; use the confirmed state and retry')
                if request.target.server_id != target.server_id:
                    raise ValueError('Marker server restarted; refresh the target state')
                session.edit(Target.from_message(request.target), read_tf=operation == api.READ_TF,
                             for_goal=operation == api.LOCK)
                if operation == api.LOCK:
                    session.freeze(request.client_id, request.target.action_name)
            else:
                if target.locked and request.client_id != target.owner_id:
                    raise ValueError('Another editor owns the planning target')
                if operation == api.RELEASE:
                    session.release()
                elif operation == api.BIND_GOAL:
                    if not target.locked or not any(request.target.goal_id):
                        raise ValueError('A reserved target and accepted goal UUID are required')
                    if any(target.goal_id) and list(request.target.goal_id) != list(target.goal_id):
                        raise ValueError('A different action goal is already bound to this target')
                    target.goal_id = [int(value) for value in request.target.goal_id]
                    session.owner_seen = time.monotonic()
                    session.publish('Following action status')
                    if session.goal_statuses.get(bytes(target.goal_id)) in (
                            GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_CANCELED):
                        session.release()
                elif operation == api.HEARTBEAT:
                    session.owner_seen = time.monotonic()
                else:
                    raise ValueError('Unknown target operation')
            response.success, response.message = True, session.target.status
        except (ValueError, TransformException) as error:
            response.success, response.message = False, str(error)
        response.state = session.target.message(self.marker_state_type)
        return response

    def tick(self):
        for session in self.sessions:
            session.tick()

    def destroy_node(self):
        for session in self.sessions:
            session.close()
        self.listener.unregister()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PlanningMarkerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
