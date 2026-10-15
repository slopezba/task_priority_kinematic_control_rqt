# task_priority_kinematic_control_rqt

ROS 2 RQT plugin for monitoring and interacting with the task-priority kinematic controller.

## Contents

- Task state table
- Enable and disable task controls
- Task reordering controls
- Runtime solver method, `dls_lambda`, and `dof_weights` tuning through ROS parameters
- Runtime target publication for supported tasks
- Cartesian planning of the selected arm and optional vehicle DOFs, with TF
  target frames, per-goal limits and trajectory tracking tolerances
- Bimanual Cartesian planning with relative-pose input, per-joint movement weights,
  progress feedback, explicit pending-plan execution, and cancellation

Drag the horizontal divider between the task table and tabs to change their
heights. All tabs support scrolling when the panel is shortened, so long forms
do not impose a large minimum window height.

## Cartesian tab

**Cartesian** calls `/cirtesub/manipulation/plan_cartesian_trajectory` with
`PlanCartesianTrajectory`. **Reference Frame** defaults to `world_ned` and accepts
any frame available through TF. **Read Current Pose (TF)** reads the selected
tool in that frame. XYZ and quaternion are absolute tool coordinates; changing
the reference frame transforms both to preserve the physical target. Missing TF
restores the previous frame and leaves the coordinates intact. RQT sends a zero
timestamp so the server resolves one current TF snapshot and fixes the target
in the world before planning, even when it was entered in the moving base frame.

Configure the server, select the arm and use **Refresh Arm Joints** if the catalog
has not loaded. RQT reads `arms.<arm>.joint_names` from the server's parameter
service; it displays the actual names and order, including arms with different
joint counts. Sending is blocked until the catalog is valid. The parameter
service is derived from **Execute Pending Service**, so update that endpoint
when using a renamed server. Refresh can retry a pending metadata request.
The **Pose** backend does not require a joint catalog.

The six base checkboxes default to off and the arm joint checkboxes default to
on. The goal's `available_dofs` is `[x, y, z, roll, pitch, yaw, joints...]` in the
server's joint order. Unchecked variables are locked to their initial values
throughout the plan. Base axes always refer to the planning world frame; choosing
a different target frame does not change the mask. Joint choices and tracking
tolerances are retained by joint name when switching arms.

Enabling a base DOF automatically selects **Task** and disables **Controller**.
The task backend follows the base and arm trajectories with a shared start time.
When all base DOFs are off, arm-only goals can use either backend. The panel never
switches controllers; start the task-priority controller before execution.

Select **Pose** to plan only the end-effector path and stream Cartesian pose
references to the selected arm's `left_pose` or `right_pose` task. The configured
task hierarchy resolves base and arm motion. These tasks and the planner use
`cirtesub/alpha_left/standard_jaws_tool` and
`cirtesub/alpha_right/standard_jaws_tool`, with references expressed in `world_ned`.
Base/joint mobility controls, base limits and base/joint tracking tolerances are
disabled and sent as empty arrays. Their saved values are restored when switching
back to another backend. Tool path/goal tracking tolerances and the final settling
allowance remain editable. Velocity/acceleration scaling applies to the server's
Cartesian limits in this mode. Start the task-priority controller and enable the
required tasks before executing. The geometric path does not check reachability
or collisions in advance.

Base maximum velocities and accelerations are editable for each axis (metres or
radians per second and per second squared), initially `0.2` for XYZ and `0.3` for
RPY. Bounds use the server defaults until **Override base position/orientation
bounds** is checked. XYZ lower/upper bounds are offsets from the initial world
position in metres; RPY bounds are absolute world angles in radians. The UI checks
that every lower bound is no larger than its upper bound before dispatch. Pitch
bounds stay inside `[-1.56, 1.56] rad` to avoid Euler singularities.

**Planning Position/Orientation Tolerances** constrain the planned tool path.
Separate path and goal tracking tolerances cover base, each joint and the tool,
and **Goal Time Tolerance** controls the final settling allowance. Their initial
values are base `0.05 m / 0.1 rad`, joints `0.05 rad` (or metres for prismatic
joints), tool `0.002 m / 0.03 rad`, and `0.5 s`. Velocity/acceleration scaling
applies to the jointly timed plan; base maximum limits remain separately editable.

Execution is unchecked by default and waits for **Execute Pending**. All goal
controls are locked while planning, waiting or executing. Feedback includes
base, joint and tool tracking errors, and the result also reports base waypoint
count. Cancellation uses the shared action client, including late acceptance
and panel shutdown. **Release Trajectory Hold** is an explicit controller
service call to release trajectory ownership after the goal ends and restore
manual tasks; it stays disabled while this tab has an active goal. Its endpoint
defaults to `/cirtesub/controller/task_priority/release_cartesian_trajectory`.
In **Pose**, pending execution and cancellation use the same action/services;
feedback reports tool error and the result reports tool pose count. Cancellation
holds the measured tool pose and restores the task's temporary settings. The
trajectory release button is disabled because pose execution does not acquire
joint/base trajectory ownership.

Launch all action servers and optionally the GUI with one command:

```bash
ros2 launch sura_manipulation_actions manipulation_actions.launch.py \
  execution_backend:=task use_rviz:=true use_rqt:=true
ros2 lifecycle set /plan_cartesian_trajectory_lifecycle_action_server configure
ros2 lifecycle set /plan_cartesian_trajectory_lifecycle_action_server activate
```

Robot bringup and backend activation stay with the operator. Rebuild the actions,
controller and RQT packages before using this extended interface, then restart
the server, controller and RQT.

## Independent planning markers

The `planning_marker_server` executable owns the Cartesian and bimanual interactive
markers in a separate process, with its own ROS executor. It imports no Qt code.
RQT only subscribes to confirmed target states and calls edit/reservation services.
Those client operations use a dedicated ROS node and executor thread, rather than
the panel's Qt-driven spinner. Only confirmed states and responses are queued to
the Qt thread; a response already received cannot time out while waiting for Qt.
RViz's `get_interactive_markers` requests and dragging work with RQT closed.
The error/state tables and their update policy have not been changed.

The unified manipulation launch now starts this node automatically, independently
of `use_rqt`. Start the normal planning UI with:

```bash
ros2 launch sura_manipulation_actions manipulation_actions.launch.py \
  execution_backend:=task use_rviz:=true use_rqt:=true
```

Alternatively, start only the marker authority:

```bash
ros2 launch task_priority_kinematic_control_rqt planning_markers.launch.py \
  robot_namespace:=cirtesub use_sim_time:=false
```

Run one authority per robot. When using the standalone launch alongside the unified
launch, set `launch_planning_markers:=false` in the latter. After upgrading from the
old embedded implementation, close the old RQT process before starting the new node
so duplicate services do not remain on the same namespaces.

RViz's property in this ROS version is **Interactive Markers Namespace**, not
Update Topic. Both supplied RViz configurations now use the correct property:

| Editor | Interactive Markers Namespace | RQT Planning target service |
| --- | --- | --- |
| Cartesian | `/cirtesub/manipulation/cartesian_goal_markers` | `/cirtesub/manipulation/planning_targets/cartesian/update` |
| Bimanual | `/cirtesub/manipulation/bimanual_cartesian_goal_markers` | `/cirtesub/manipulation/planning_targets/bimanual/update` |

Do not append `/update` in the RViz namespace field. That suffix is used only by
the transport topic. In RQT, **Planning marker namespace** edits the marker namespace
through the authority; **Planning target service** selects its editor endpoint.
For another robot namespace, update the service endpoint in each tab. Confirmed
configuration and poses then arrive from that node. **Mostrar objetivos en RViz**
controls visibility. Disabling it also enables the existing manual goal path,
which can be used without the marker node.

Use RViz's **Interact** tool to drag targets. **Cartesian** has one purple pin;
**Bimanual Cartesian** has blue A and orange B pins. Field edits are acknowledged
by the node; marker dragging publishes updated state back to RQT. The bimanual
markers define only `pB-pA` and `RB*RA^T` in the configured reference frame. Their
absolute positions are illustrative, and the base is fixed. Editing the relative
fields anchors A and reconstructs B; swapping roles preserves physical tool poses.
**Read Current Pose / Read Relative Pose** asks the node to obtain a common TF
snapshot. Missing TF retains the last valid target. The services exist and answer
even before TF is ready (RViz initially receives an empty marker list).
The rings and arrows use a scale of `0.10`, matching the configured `left_pose`
marker. Gripper meshes include the inverse of the Alpha `standard_jaws_tool`
fixed joint so their orientation and position match the robot when reading TF.
The target pose and drag controls remain expressed at the tool frame.

**Send Goal** reserves the authoritative target before dispatch. Cartesian targets
are frozen in `world_ned` at a common recent TF time, while keeping the original
frame and timestamp in the action goal for tolerance axes. Edits and competing
reservations are rejected while locked. RQT binds the accepted action UUID; the
node observes the action's status topic and unlocks after success, abort or cancel,
even if the UI disappears. Closing RQT follows its existing action-cancellation
policy but does not destroy the external marker servers.

If the editor disappears before an action can be identified, the node conservatively
retains the lock and publishes an explanation rather than guessing that execution
ended. An operator can cancel the relevant action, read `owner_id` from the state,
and explicitly release that reservation using operation `3` (RELEASE) and that
`client_id` on its update service. Releasing a marker reservation never sends robot
commands or alters a pending plan in the planner.

The new `sura_manipulation_actions/msg/PlanningTargetState` is published with reliable,
transient-local depth-one QoS at the corresponding `/state` endpoint. It includes
poses, relative/absolute target, configuration, revision, server instance ID and
reservation state. `UpdatePlanningTarget` supports EDIT, READ_TF, LOCK, RELEASE,
BIND_GOAL and HEARTBEAT. Revision checks prevent stale edits overwriting a newer
marker drag; server instance IDs handle node restarts. These new interfaces require
rebuilding `sura_manipulation_actions` and the RQT package; planning actions are unchanged.

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

Choose arm A and arm B; neither has priority in the planner. **Read Relative Pose (TF)** reads both tool poses
in **Relative Reference Frame**, at a common timestamp, and fills XYZ with
`p_arm_b - p_arm_a`, and the quaternion with `R_arm_b * R_arm_a^T`.
Both are expressed in reference/base axes. Editing X by `0.01` requests one
centimetre more separation along base X even if arm A is rotated.
Identity quaternion requests equal tool orientations. This is a relative
relationship, not arm B's absolute pose.

The reference must match `relative_reference_frame` in the planner (a model
link independent of the commanded arms); it defaults to `cirtesub/base_link`.
Tool frames must match the configured tips and are used for TF reading.
Arm names, reference/tool frames, action and execute-service endpoints remain
editable for other robots. Defaults are read from the installed planner YAML.

The Alpha example uses `cirtesub/alpha_left/standard_jaws_tool` and
`cirtesub/alpha_right/standard_jaws_tool`. Selecting either arm selects its
corresponding tool frame; the reference/base frame stays the same when A/B are
swapped. Read the pose again after swapping to obtain the reversed relationship.

Set relative position/orientation tolerances per component. Zero is a strict constraint.
Quaternions are entered as XYZW and normalized before sending. Zero planning
time or Cartesian step selects the server's configured default.

**Velocity Scaling** and **Acceleration Scaling** select the joint limit factors
for each goal: `0.1` is 10 %, `0.05` is 5 %. Zero displays **Server default** and
uses the corresponding YAML value. Both controls default to zero. These factors
change the timing shown in RViz and sent to either backend; **Planning Time**
is only the computation budget. Inputs are locked while a goal is active:
cancel the active goal, choose new factors and send another goal to change speed.
Configure joint movement penalties in the server YAML with `planning.weights`.
In Alpha the order is left b,c,d,e, then right b,c,d,e. A larger value discourages
moving that joint away from its measured initial position; zero adds no penalty
and does not lock it. Weights stay on the same joints when A/B are swapped.
Restart the server after editing its YAML and configure it again, or use cleanup,
update the ROS parameter and configure. This discards any pending plan.
The action now uses `arm_a`/`arm_b` and `arm_a_path`/`arm_b_path`, and has no arm
radius/orientation fields. Rebuild the action and RQT packages and restart clients
to use this interface.

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
