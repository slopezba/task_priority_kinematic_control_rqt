# task_priority_kinematic_control_rqt

ROS 2 RQT plugin for monitoring and interacting with the task-priority kinematic controller.

## Contents

- Task state table
- Enable and disable task controls
- Task reordering controls
- Runtime solver method, `dls_lambda`, and `dof_weights` tuning through ROS parameters
- Runtime target publication for supported tasks

## Dependencies

This package depends on `task_priority_kinematic_control` and standard ROS 2 RQT packages.

## Build

Build it inside a ROS 2 workspace with `colcon`.

## Notes

This repository is intended to contain the `task_priority_kinematic_control_rqt` package only.
