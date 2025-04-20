import time
import math
import numpy as np
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import roboticstoolbox as rt
from roboticstoolbox import models
from spatialmath import SE3
from spatialmath.base import angvec2tr

def axis_angle_to_transform(tcp_pose):
    """
    Convert a UR axis-angle TCP pose to a 4x4 transformation matrix.
    UR TCP pose is [x, y, z, Rx, Ry, Rz], where (Rx, Ry, Rz) is axis-angle.
    """
    x, y, z, Rx, Ry, Rz = tcp_pose
    angle = np.linalg.norm([Rx, Ry, Rz])
    
    # Build the rotation matrix from axis-angle
    if angle < 1e-12:
        # Very small angle => no rotation
        rot_mat = np.eye(4)
    else:
        axis = [Rx/angle, Ry/angle, Rz/angle]
        rot_mat = angvec2tr(angle, axis)
    
    # Set the translation part
    rot_mat[0, 3] = x
    rot_mat[1, 3] = y
    rot_mat[2, 3] = z
    
    return rot_mat

def main():
    # RTDE connection parameters
    ROBOT_IP = "192.168.200.20"  # Replace with your UR5's IP address
    PORT = 30004                 # Default RTDE port for UR robots

    # Load the RTDE configuration (ensure your XML outputs joint angles & TCP pose)
    conf = rtde_config.ConfigFile("./RTDE_Python_Client_Library/examples/record_configuration.xml")
    output_names, output_types = conf.get_recipe("out")

    # Establish RTDE connection to the UR5
    con = rtde.RTDE(ROBOT_IP, PORT)
    con.connect()
    con.send_output_setup(output_names, output_types)
    con.send_start()

    # Use the built-in UR5 model from the Robotics Toolbox for Python
    # ur5e = rt.models.UR5e()
    ur5 = models.DH.UR5()

    # Example tool offset: Suppose the tool is 0.15 m along the Z-axis from the UR5 flange.
    # Adjust these values to match your actual tool geometry.
    # tool_offset = SE3(0, 0, 0.15)
    # ur5.tool = tool_offset  # Incorporate this offset into forward kinematics

    print("Connected to UR5. Retrieving joint angles and computing forward kinematics...\n")

    # Receive one state from the RTDE
    state = con.receive()
    if state:
        # 1) Joint angles (in radians)
        q = state.actual_q
        print("Joint angles:", q)

        # 2) Robot-reported TCP pose ([x, y, z, Rx, Ry, Rz] in axis-angle)
        tcp_pose = state.actual_TCP_pose
        print("Robot-reported TCP pose (axis-angle):", tcp_pose)

        # Convert to a 4x4 transformation matrix
        T_robot_reported = axis_angle_to_transform(tcp_pose)
        print("\nRobot-reported Transformation Matrix (Base to TCP):")
        print(T_robot_reported)

        # 3) Compute forward kinematics with tool offset included
        T_dh = ur5.fkine(q)  # This now includes the tool transform (ur5.tool)
        print("\nDH-based FK (with tool offset) from joint angles:")
        print(T_dh)

    # Cleanup
    con.disconnect()

if __name__ == "__main__":
    main()
