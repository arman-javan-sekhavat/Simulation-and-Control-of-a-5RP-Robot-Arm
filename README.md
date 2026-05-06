# Simulation and Control of a 5RP Robot Arm

This repository includes 3D models, MJCF files and the code for simulating and controlling a 6-DoF robot arm
using the MuJoCo and JAX libraries. The following algorithms are implemented:

1) Forward Kinematics
2) Inverse Kinematics
3) Inverse Dynamics (utilizing both Lagrangian and Newton-Euler formulations)
4) Trajectory Generation (Cubic and Quintic methods)
5) Linear IJC (Independent Joint Control)
6) CTM (Computed Torque Method)
7) TJ (Transpose Jacobian)
8) MTJ (Modified Transpose Jacobian)

## Simulation Preview
<img width="563" height="524" alt="image" src="https://github.com/user-attachments/assets/e4db93ef-778e-45f7-9412-86e3b9ae3c8b" />

## ForwardKinematics.py
This file includes several functions required for computing the forward-kinematics-related quantities of the manipulator, such as the end-effector's position and orientation relative to the base frame. These functions are used in some of the notebooks.

## forward_kinematics.mlx
This MATLAB Livescript file includes symbolic computations related to the manipulator's forward kinematics.

## InverseKinematics.py
This file includes functions required for computing the inverse kinematics of the manipulator.

## InverseDynamics.py
This file provides functions for computing the manipulator's inverse dynamics using both the Recursive Newton-Euler (RNEA) algorithm, and the Lagrangian formulation. It also provides dedicated functions to evaluate the total kinetic and potential energies of the manipulator.
