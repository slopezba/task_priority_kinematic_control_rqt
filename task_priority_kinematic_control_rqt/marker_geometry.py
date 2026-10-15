"""Pose math and marker geometry shared by the independent server and RQT."""
from copy import deepcopy
import math
from geometry_msgs.msg import Pose, Transform
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker


def quaternion_product(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz]


def pose_from_components(position, quaternion):
    norm = math.hypot(*quaternion)
    if len(position) != 3 or len(quaternion) != 4 or not all(math.isfinite(v) for v in position) or \
            not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("Pose needs finite XYZ and a finite, nonzero quaternion")
    pose = Pose()
    for axis, value in zip("xyz", position):
        setattr(pose.position, axis, float(value))
    for axis, value in zip("xyzw", quaternion):
        setattr(pose.orientation, axis, float(value / norm))
    return pose


def pose_components(pose):
    return ([getattr(pose.position, axis) for axis in "xyz"],
            [getattr(pose.orientation, axis) for axis in "xyzw"])


def normalized_pose(pose):
    return pose_from_components(*pose_components(pose))


def transform_pose(pose, transform):
    position, quaternion = pose_components(normalized_pose(pose))
    rotation = [getattr(transform.rotation, axis) for axis in "xyzw"]
    translation = [getattr(transform.translation, axis) for axis in "xyz"]
    rotation = pose_components(pose_from_components(translation, rotation))[1]
    conjugate = [-rotation[0], -rotation[1], -rotation[2], rotation[3]]
    rotated = quaternion_product(quaternion_product(rotation, [*position, 0.0]), conjugate)
    return pose_from_components([v + t for v, t in zip(rotated[:3], translation)],
                                quaternion_product(rotation, quaternion))


def pose_from_transform(transform):
    return pose_from_components([getattr(transform.translation, axis) for axis in "xyz"],
                                [getattr(transform.rotation, axis) for axis in "xyzw"])


def relative_pose(a, b):
    pa, qa = pose_components(normalized_pose(a))
    pb, qb = pose_components(normalized_pose(b))
    return pose_from_components([y - x for x, y in zip(pa, pb)],
                                quaternion_product(qb, [-qa[0], -qa[1], -qa[2], qa[3]]))


def reconstruct_b(a, relative):
    pa, qa = pose_components(normalized_pose(a))
    dp, dq = pose_components(normalized_pose(relative))
    return pose_from_components([x + y for x, y in zip(pa, dp)], quaternion_product(dq, qa))


def gripper_marker(name, label, frame, pose, color, editable):
    marker = InteractiveMarker()
    marker.name, marker.header.frame_id = name, frame
    marker.description = label + (" — target" if editable else " — locked target")
    # Match end_effector_marker_scale in task_priority_target_markers.yaml.
    marker.pose, marker.scale = deepcopy(pose), 0.10
    visual = InteractiveMarkerControl()
    visual.name, visual.always_visible = "gripper", True
    visual.orientation.w = 1.0
    # The meshes use standard_jaws_base_link coordinates; targets use
    # standard_jaws_tool. Invert tool_joint from standard_jaws.urdf.xacro
    # (xyz="0 0 0.04", rpy="0 -1.571 0") for the visual only.
    base_in_tool = Transform()
    base_in_tool.rotation.y = math.sin(1.571 / 2.0)
    base_in_tool.rotation.w = math.cos(1.571 / 2.0)
    base_in_tool.translation.x = -0.04 * math.sin(1.571)
    base_in_tool.translation.z = -0.04 * math.cos(1.571)
    for resource, offset in (("RS1-124.stl", None), ("RS1-130.stl", 0.0155), ("RS1-139.stl", -0.0155)):
        mesh = Marker()
        mesh.type = Marker.MESH_RESOURCE
        mesh.mesh_resource = "package://alpha_description/meshes/end_effectors/" + resource
        mesh.pose.orientation.w = 1.0
        if offset is not None:
            mesh.pose.position.y, mesh.pose.position.z = offset, 0.0069
        mesh.pose = transform_pose(mesh.pose, base_in_tool)
        mesh.scale.x = mesh.scale.y = mesh.scale.z = 1.0
        mesh.color.r, mesh.color.g, mesh.color.b, mesh.color.a = *color, 0.65
        visual.markers.append(mesh)
    marker.controls.append(visual)
    if editable:
        for axis in "xyz":
            for mode, prefix in ((InteractiveMarkerControl.MOVE_AXIS, "move_"),
                                 (InteractiveMarkerControl.ROTATE_AXIS, "rotate_")):
                control = InteractiveMarkerControl()
                control.name, control.interaction_mode = prefix + axis, mode
                control.orientation.w = math.sqrt(0.5)
                # InteractiveMarker axis controls act along their local X axis.
                setattr(control.orientation, {"x": "x", "y": "z", "z": "y"}[axis], math.sqrt(0.5))
                control.orientation_mode = InteractiveMarkerControl.FIXED
                marker.controls.append(control)
    return marker

