# task_priority_kinematic_control_rqt

ROS 2 RQT plugin for monitoring and interacting with the task-priority kinematic controller.

## Contents

- Task state table
- Enable and disable task controls
- Task reordering controls
- Runtime solver method, `dls_lambda`, and `dof_weights` tuning through ROS parameters
- Runtime target publication for supported tasks
- Single-arm Cartesian planning with absolute tool target, arm selection,
  per-goal task/controller backend and velocity/acceleration controls
- Bimanual Cartesian planning with relative-pose input, leader constraints,
  progress feedback, explicit pending-plan execution, and cancellation

Drag the horizontal divider between the task table and tabs to change their
heights. All tabs support scrolling when the panel is shortened, so long forms
do not impose a large minimum window height.

## Cartesian tab

**Cartesian** calls `/cirtesub/manipulation/plan_cartesian_trajectory` with
`PlanCartesianTrajectory`. Select the arm and **Execution Backend** (server
default, controller or task). **Read Current Pose (TF)** reads the selected
`standard_jaws_tool` pose in **Reference Frame**, normally `cirtesub/base_link`.
XYZ and quaternion are absolute tool coordinates in that frame; edit them to
request a straight Cartesian path from the measured configuration.

The selected task/controller endpoint comes from the server's arm YAML. Only
that arm's joints are sent. Planning tolerances and velocity/acceleration factors
work as in the bimanual tab. Execution is unchecked by default and waits for
**Execute Pending**; the panel never enables tasks or switches controllers.
The common client code handles cancellation, late acceptance and panel shutdown.

Launch all action servers and optionally the GUI with one command:

```bash
ros2 launch sura_manipulation_actions manipulation_actions.launch.py \
  execution_backend:=task use_rviz:=true use_rqt:=true
ros2 lifecycle set /plan_cartesian_trajectory_lifecycle_action_server configure
ros2 lifecycle set /plan_cartesian_trajectory_lifecycle_action_server activate
```

Robot bringup and backend activation stay with the operator. Build both the
actions and RQT packages before using this tab, then restart the server and RQT.

## Bimanual Cartesian tab

The **Bimanual Cartesian** tab calls
`/cirtesub/manipulation/plan_bimanual_cartesian_trajectory`
(`sura_manipulation_actions/action/PlanBimanualCartesianTrajectory`). It uses the
Cartesian lifecycle server, independently of the bimanual joint server:

```bash
ros2 lifecycle set /plan_bimanual_cartesian_trajectory_lifecycle_action_server configure
ros2 lifecycle set /plan_bimanual_cartesian_trajectory_lifecycle_action_server activate
```

Launch that server first, with the robot/TF/joint states already available:

```bash
ros2 launch sura_manipulation_actions manipulation_actions.launch.py \
  execution_backend:=controller use_rviz:=true
```

Choose the leader and follower. **Read Relative Pose (TF)** reads both tool poses
in **Relative Reference Frame**, at a common timestamp, and fills XYZ with
`p_follower - p_leader`, and the quaternion with `R_follower * R_leader^T`.
Both are expressed in reference/base axes. Editing X by `0.01` requests one
centimetre more separation along base X even if the leader is rotated.
Identity quaternion requests equal tool orientations. This is a relative
relationship, not the follower's absolute pose.

The reference must match `relative_reference_frame` in the planner (a model
link independent of the commanded arms); it defaults to `cirtesub/base_link`.
Tool frames must match the configured tips and are used for TF reading.
Arm names, reference/tool frames, action and execute-service endpoints remain
editable for other robots. Defaults are read from the installed planner YAML.

The Alpha example uses `cirtesub/alpha_left/standard_jaws_tool` and
`cirtesub/alpha_right/standard_jaws_tool`. Selecting the leader selects its
corresponding tool frame; the reference/base frame stays the same when roles are
swapped. Read the pose again after swapping to obtain the reversed relationship.

Set the leader radius in metres, roll/pitch/yaw limits in radians, and relative
position/orientation tolerances per component. Zero is a strict constraint.
Quaternions are entered as XYZW and normalized before sending. Zero planning
time or Cartesian step selects the server's configured default.

**Velocity Scaling** and **Acceleration Scaling** select the joint limit factors
for each goal: `0.1` is 10 %, `0.05` is 5 %. Zero displays **Server default** and
uses the corresponding YAML value. Both controls default to zero. These factors
change the timing shown in RViz and sent to either backend; **Planning Time**
is only the computation budget. Inputs are locked while a goal is active:
cancel the active goal, choose new factors and send another goal to change speed.
The action interface now includes these two fields: rebuild both the action and
RQT packages, then restart the server and clients after updating.

**Send Goal** plans and publishes the robot trajectory in RViz. Execution is
unchecked by default: the goal remains in `waiting_for_execute`. Review the plan
and click **Execute Pending**, or explicitly check **Execute after planning**
before sending. The panel never activates tasks or switches controllers. If the
backend is inactive, enable it manually and request execution again. **Cancel
Goal** cancels the owned goal, including an acceptance that arrives after the
cancel click. Closing the panel also requests cancellation.

The tab displays progress, planner/execution diagnostics and, when the action
finishes, the joint count, waypoint count and duration. Goal inputs stay locked
while a goal is active; cancel it before entering a replacement. It shares the
existing panel ROS timer and does not create another spinning thread.

Offscreen Qt tests use fake actions/services and do not command a robot:

```bash
QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s test -v
```

## Dependencies

This package depends on `task_priority_kinematic_control` and standard ROS 2 RQT packages.

## Build

Build it inside a ROS 2 workspace with `colcon`.

## Notes

This repository is intended to contain the `task_priority_kinematic_control_rqt` package only.
