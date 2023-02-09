"""Mirobot_Controller controller."""

import sys
from controller import Supervisor
import math
import os
import torch
from stable_baselines3.common.logger import configure
import time

try:
    import gym
    import numpy as np
    from stable_baselines3 import SAC
    from stable_baselines3.common.buffers import DictReplayBuffer
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
except ImportError:
    sys.exit(
        'Please make sure you have all dependencies installed. '
        'Run: "pip3 install numpy gym stable_baselines3"'
    )

try:
    import ikpy
    from ikpy.chain import Chain
except ImportError:
    sys.exit('The "ikpy" Python module is not installed. '
             'To run this sample, please upgrade "pip" and install ikpy with this command: "pip install ikpy"')

if ikpy.__version__[0] < '3':
    sys.exit('The "ikpy" Python module version is too old. '
             'Please upgrade "ikpy" Python module to version "3.0" or newer with this command: "pip install --upgrade ikpy"')


IKPY_MAX_ITERATIONS = 32

# Initialize the Webots Supervisor.
supervisor = Supervisor()
timeStep = int(1024 * supervisor.getBasicTimeStep())

# Create the arm chain from the URDF
base_path = 'C:/Users/Administrator/Documents/GitHub/mirobot_rl/'
urdf_file = base_path + 'Mirobot.urdf'
armChain = Chain.from_urdf_file(urdf_file)
for i in [0, 7]:
    armChain.active_links_mask[i] = False

# Initialize the arm motors and encoders.
motors = []
for link in armChain.links:
    if 'motor' in link.name:
        motor = supervisor.getDevice(link.name)
        motor.setVelocity(1.0)
        position_sensor = motor.getPositionSensor()
        position_sensor.enable(timeStep)
        motors.append(motor)

# Get the arm and target nodes.
target = supervisor.getFromDef('target')
end_effector = supervisor.getFromDef('ee_tool')
arm = supervisor.getSelf()
robot = supervisor.getFromDef('robot')

while supervisor.step(timeStep) != -1:

    # Get the absolute postion of the target and the arm base.
    targetPosition = target.getPosition()
    armPosition = arm.getPosition()
    eePosition = end_effector.getPosition()

    # Compute the position of the target relatively to the arm.
    # x and y axis are inverted because the arm is not aligned with the Webots global axes.
    x = targetPosition[0] - armPosition[0]
    y = targetPosition[1] - armPosition[1]
    z = targetPosition[2] - armPosition[2]
    
    distance_x = targetPosition[0] - eePosition[0]
    distance_y = targetPosition[1] - eePosition[1]
    distance_z = targetPosition[2] - eePosition[2]
    
    total_distance = math.sqrt(distance_x**2 + distance_y**2 + distance_z**2)    
    # Call "ikpy" to compute the inverse kinematics of the arm.
    initial_position = [0] + [m.getPositionSensor().getValue() for m in motors] + [0]
    ikResults = armChain.inverse_kinematics([x, y, z], [0, 0, -1], orientation_mode="X", max_iter=IKPY_MAX_ITERATIONS, initial_position=initial_position)

    # Recalculate the inverse kinematics of the arm if necessary.
    position = armChain.forward_kinematics(ikResults)
    squared_distance = (position[0, 3] - x)**2 + (position[1, 3] - y)**2 + (position[2, 3] - z)**2
    if math.sqrt(squared_distance) > 1e-5:
        ikResults = armChain.inverse_kinematics([x, y, z], [0, 0, -1], orientation_mode="X")
    
    # Actuate the arm motors with the IK results.
    if total_distance > 1e-5:
        for i in range(len(motors)):
            motors[i].setPosition(ikResults[i + 1])
    
    print(f'Total Distance: {total_distance}')

    
