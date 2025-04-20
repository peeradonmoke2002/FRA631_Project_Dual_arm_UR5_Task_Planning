import numpy as np
import time
from spatialmath import SE3
from classrobot.UR5e_DH import UR5eDH
from rtde_control import RTDEControlInterface

def moveL_real_rtde_speedJ_with_invJ(robot, T_start, T_goal, q_start, duration=3.0, frequency=100):
    ROBOT_IP = "192.168.0.100"  # Replace with your robot's IP
    rtde_control = RTDEControlInterface(ROBOT_IP)

    dt = 1.0 / frequency
    steps = int(duration * frequency)
    q = np.array(q_start)

    try:
        for step in range(steps + 1):
            t = step * dt
            tau = t / duration

            # Cubic scalar + derivative (velocity scaling)
            s = 3 * tau**2 - 2 * tau**3
            s_dot = (6 * tau - 6 * tau**2) / duration

            # Desired pose and next pose
            T_desired = T_start.interp(T_goal, s)
            T_next = T_start.interp(T_goal, s + s_dot * dt)

            # Approximate task-space velocity (twist)
            delta_T = T_desired.inv() @ T_next
            x_dot = delta_T.twist() / dt  # 6x1 Cartesian velocity

            # Compute Jacobian at current q
            J = robot.jacob0(q)

            # Compute joint velocities using pseudo-inverse of J
            q_dot = np.linalg.pinv(J) @ x_dot

            # Send joint velocities via speedJ
            rtde_control.speedJ(q_dot.tolist(), acceleration=1.0, time=dt)

            # Update joint state (only for simulation; UR handles this internally)
            q = q + q_dot * dt

            print(f"[{step:03}] q_dot = {np.round(q_dot, 3)}")

            time.sleep(dt)

        rtde_control.stopJ()
    finally:
        rtde_control.disconnect()
