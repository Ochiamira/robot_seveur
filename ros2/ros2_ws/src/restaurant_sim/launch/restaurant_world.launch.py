import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    restaurant_share = get_package_share_directory('restaurant_sim')
    turtlebot_share = get_package_share_directory('turtlebot3_gazebo')
    ros_gz_sim_share = get_package_share_directory('ros_gz_sim')

    turtlebot_model = os.environ.get('TURTLEBOT3_MODEL', 'waffle_pi')
    turtlebot_folder = f'turtlebot3_{turtlebot_model}'

    world_file = os.path.join(
        restaurant_share,
        'worlds',
        'restaurant.sdf',
    )

    robot_sdf = os.path.join(
        turtlebot_share,
        'models',
        turtlebot_folder,
        'model.sdf',
    )

    bridge_config = os.path.join(restaurant_share, 'config', 'turtlebot3_waffle_pi_bridge.yaml')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                ros_gz_sim_share,
                'launch',
                'gz_sim.launch.py',
            )
        ),
        launch_arguments={
            'gz_args': f'-r -s -v 2 {world_file}',
            'on_exit_shutdown': 'true',
        }.items(),
    )

    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                turtlebot_share,
                'launch',
                'robot_state_publisher.launch.py',
            )
        ),
        launch_arguments={
            'use_sim_time': 'true',
        }.items(),
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-world', 'restaurant',
            '-name', 'server_robot',
            '-file', robot_sdf,
            '-x', '0.0',
            '-y', '2.35',
            '-z', '0.01',
            '-Y', '-1.5708',
        ],
        output='screen',
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '--ros-args',
            '-p',
            f'config_file:={bridge_config}',
        ],
        output='screen',
    )

    image_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=['/camera/image_raw'],
        output='screen',
    )

    return LaunchDescription([
        AppendEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            os.path.join(restaurant_share, 'models'),
        ),
        AppendEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            os.path.join(turtlebot_share, 'models'),
        ),
        gazebo,
        robot_state_publisher,
        spawn_robot,
        bridge,
        image_bridge,
    ])