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


def compose_pose(a, b):
    """Pose of frame b (given in a's frame) expressed in the frame a is expressed in: a * b."""
    (pa, qa), (pb, qb) = pose_components(normalized_pose(a)), pose_components(normalized_pose(b))
    conjugate = [-qa[0], -qa[1], -qa[2], qa[3]]
    rotated = quaternion_product(quaternion_product(qa, [*pb, 0.0]), conjugate)
    return pose_from_components([x + y for x, y in zip(pa, rotated[:3])], quaternion_product(qa, qb))


def pose_from_transform(transform):
    return pose_from_components([getattr(transform.translation, axis) for axis in "xyz"],
                                [getattr(transform.rotation, axis) for axis in "xyzw"])


# End-effectors facing: same conventions as sura_manipulation_actions/facing_geometry.hpp.
AXIS_NAMES = ("+x", "-x", "+y", "-y", "+z", "-z")


def axis_vector(name):
    if name not in AXIS_NAMES:
        raise ValueError("Pointing axes must be one of " + ", ".join(AXIS_NAMES))
    sign = 1.0 if name[0] == "+" else -1.0
    return [sign if axis == name[1] else 0.0 for axis in "xyz"]


def axis_name(vector):
    values = [float(v) for v in vector]
    for name in AXIS_NAMES:
        if all(abs(a - b) < 1e-3 for a, b in zip(values, axis_vector(name))):
            return name
    raise ValueError("Pointing axes must be pure signed axes (+-x, +-y or +-z)")


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _rotate(quaternion, vector):
    conjugate = [-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]]
    return quaternion_product(quaternion_product(quaternion, [*vector, 0.0]), conjugate)[:3]


def _axis_angle(axis, angle):
    return [value * math.sin(angle / 2.0) for value in axis] + [math.cos(angle / 2.0)]


def _align(a, b):
    """Shortest rotation taking unit vector a onto unit vector b."""
    normal = _cross(a, b)
    sine = math.hypot(*normal)
    angle = math.atan2(sine, _dot(a, b))
    if sine < 1e-9:
        if angle < 1.0:
            return [0.0, 0.0, 0.0, 1.0]
        normal = _cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
        sine = math.hypot(*normal)
    return _axis_angle([value / sine for value in normal], angle)


def _facing_direction(a, b):
    delta = [y - x for x, y in zip(pose_components(a)[0], pose_components(b)[0])]
    distance = math.hypot(*delta)
    if distance < 1e-9:
        raise ValueError("Facing needs two separated tools")
    return [value / distance for value in delta], distance


def _facing_roll(d, qa, qb, left, right):
    def reference(quaternion, name):
        sign = 1.0 if name[0] == "+" else -1.0
        index = "xyz".index(name[1])
        return _rotate(quaternion, [sign if i == (index + 1) % 3 else 0.0 for i in range(3)])
    bl, br = reference(qa, left), reference(qb, right)
    y = _dot(d, _cross(bl, br))
    x = _dot(bl, br) - _dot(d, bl) * _dot(d, br)
    return math.atan2(y, x) if x * x + y * y > 1e-4 else None


def facing_measure(a, b, left, right):
    """(distance, roll) of two tool poses; roll is None where it is undefined."""
    d, distance = _facing_direction(a, b)
    return distance, _facing_roll(d, pose_components(normalized_pose(a))[1],
                                  pose_components(normalized_pose(b))[1], left, right)


def facing_poses(a, b, left, right, roll=None, distance=None):
    """Both tools facing each other, anchored at the position of a along the line to b."""
    d, measured = _facing_direction(a, b)
    distance = measured if distance is None else distance
    (pa, qa), (_, qb) = pose_components(normalized_pose(a)), pose_components(normalized_pose(b))
    opposite = [-value for value in d]
    qa = quaternion_product(_align(_rotate(qa, axis_vector(left)), d), qa)
    qb = quaternion_product(_align(_rotate(qb, axis_vector(right)), opposite), qb)
    if roll is not None and math.isfinite(roll):
        current = _facing_roll(d, qa, qb, left, right)
        if current is not None:
            delta = math.atan2(math.sin(roll - current), math.cos(roll - current))
            qb = quaternion_product(_axis_angle(d, delta), qb)
    return (pose_from_components(pa, qa),
            pose_from_components([p + distance * value for p, value in zip(pa, d)], qb))


def frame_visual(length=0.12):
    """Axes (x red, y green, z blue) of the controlled point, drawn instead of the gripper."""
    visual = InteractiveMarkerControl()
    visual.name, visual.always_visible = "frame", True
    visual.orientation.w = 1.0
    half = math.sqrt(0.5)
    # Arrows point along their local +x: y needs a +90 deg turn about z, z a -90 deg turn about y.
    for color, (qy, qz, qw) in (((1.0, 0.1, 0.1), (0.0, 0.0, 1.0)), ((0.1, 1.0, 0.1), (0.0, half, half)),
                                ((0.2, 0.4, 1.0), (-half, 0.0, half))):
        arrow = Marker()
        arrow.type = Marker.ARROW
        arrow.pose.orientation.y, arrow.pose.orientation.z, arrow.pose.orientation.w = qy, qz, qw
        arrow.scale.x, arrow.scale.y, arrow.scale.z = length, 0.08 * length, 0.16 * length
        arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = *color, 0.9
        visual.markers.append(arrow)
    return visual


def gripper_marker(name, label, frame, pose, color, editable, tip=False):
    """Interactive target marker. With tip=True the pose is the controlled point (offset) of the task:
    a frame is drawn instead of the gripper meshes."""
    marker = InteractiveMarker()
    marker.name, marker.header.frame_id = name, frame
    marker.description = label + (" — target" if editable else " — locked target") + (" (tip)" if tip else "")
    # Match end_effector_marker_scale in task_priority_target_markers.yaml.
    marker.pose, marker.scale = deepcopy(pose), 0.10
    if tip:
        marker.controls.append(frame_visual())
        return _with_axis_controls(marker, editable)
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
    return _with_axis_controls(marker, editable)


def _with_axis_controls(marker, editable):
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

