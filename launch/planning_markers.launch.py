"""Standalone marker authority; RQT and robot controllers are optional clients."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot_namespace', default_value='cirtesub'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        Node(package='task_priority_kinematic_control_rqt', executable='planning_marker_server',
             name='planning_marker_server', output='screen', parameters=[{
                 'robot_namespace': LaunchConfiguration('robot_namespace'),
                 'use_sim_time': LaunchConfiguration('use_sim_time'),
             }]),
    ])
