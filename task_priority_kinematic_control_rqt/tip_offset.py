"""Controlled point (tip) of a pose task as seen by the planning markers.

When the pose task controls an offset point (task parameter use_offset, e.g. the tip of a zip tie held by
the gripper), planning targets and the current pose are poses of that point: tool pose * offset. The
offset and the use_offset state are read from the same topics the controller publishes/consumes.
"""
import math
import time

from geometry_msgs.msg import PoseStamped
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

from .marker_geometry import compose_pose, normalized_pose


class TipOffset:
    def __init__(self, node, tool_frame, offset_topic, use_topic, max_age=1.0, on_change=None):
        self.tool_frame, self.max_age, self.on_change = tool_frame, float(max_age), on_change
        self.use, self.pose, self.seen = False, None, 0.0
        self.node = node
        self.subscriptions = [
            node.create_subscription(PoseStamped, offset_topic, self._offset_received,
                                     QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)),
            node.create_subscription(Bool, use_topic, self._use_received,
                                     QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                                reliability=ReliabilityPolicy.RELIABLE))]

    def _offset_received(self, message):
        # Same checks as the controller: a rejected message counts as not received.
        if message.header.frame_id and message.header.frame_id != self.tool_frame:
            return
        try:
            pose = normalized_pose(message.pose)
        except ValueError:
            return
        if not all(math.isfinite(v) for v in (pose.position.x, pose.position.y, pose.position.z)):
            return
        self.pose, self.seen = pose, time.monotonic()

    def _use_received(self, message):
        changed = bool(message.data) != self.use
        self.use = bool(message.data)
        if changed and self.on_change is not None:
            self.on_change(self)

    def fresh_offset(self):
        if self.pose is None:
            raise ValueError(f"use_offset is on for {self.tool_frame} but no offset has been received")
        if time.monotonic() - self.seen > self.max_age:
            raise ValueError(f"use_offset is on for {self.tool_frame} but its offset is stale")
        return self.pose

    def apply(self, pose):
        """Tool pose -> pose of the controlled point (unchanged when the task does not use an offset)."""
        return compose_pose(pose, self.fresh_offset()) if self.use else pose
