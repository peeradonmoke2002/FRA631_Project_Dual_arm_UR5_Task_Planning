import rtde_control
import rtde_receive
import rtde_io
import time
import math
from typing import Optional, List
import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
from spatialmath.base import trotx, troty, trotz
from spatialmath import SO3
from math import pi, cos, sin
from .planning import Planning

class RobotControl:
    def __init__(self):
        self._ROBOT_CON_ = None
        self._ROBOT_RECV_ = None
        self._ROBOT_IO_ = None

    def robot_init(self, host: str) -> None:
        self._ROBOT_CON_ = rtde_control.RTDEControlInterface(host)
        self._ROBOT_RECV_ = rtde_receive.RTDEReceiveInterface(host)
        self._ROBOT_IO_ = rtde_io.RTDEIOInterface(host)

    def robot_release(self) -> None:
        if self._ROBOT_CON_ is not None:
            self._ROBOT_CON_.stopScript()
            self._ROBOT_CON_.disconnect()

        if self._ROBOT_RECV_ is not None:
            self._ROBOT_RECV_.disconnect()

        if self._ROBOT_IO_ is not None:
            self._ROBOT_IO_.disconnect()

    # --------------------------
    # Data Acquisition Methods
    # --------------------------
    def robot_get_joint_deg(self) -> list:
        """Return the actual joint positions in degrees."""
        res = self._ROBOT_RECV_.getActualQ()
        return [math.degrees(rad) for rad in res]

    def robot_get_joint_rad(self) -> list:
        """Return the actual joint positions in degrees."""
        res = self._ROBOT_RECV_.getActualQ()
        return res
    
    def robot_get_position(self):
        """Return the current TCP pose."""
        return self._ROBOT_RECV_.getActualTCPPose()
    
    def robot_get_TCP_offset(self):
        """Return the current TCP offset."""
        return self._ROBOT_CON_.getTCPOffset()
    
    def robot_get_fk(self):
        return self._ROBOT_CON_.getForwardKinematics()
    
    def robot_get_test_fk(self, joint_pos, TCP_offset):
        """
        Calculate forward kinematics for the robot.
        
        Parameters:
            joint_pos: Joint positions as a list of floats [q1, q2, q3, q4, q5, q6].
            TCP_offset: TCP offset as a list of floats [x, y, z, Rx, Ry, Rz].
        
        Returns:
            The pose computed from FK.
        """
        pose = self._ROBOT_CON_.getForwardKinematics(joint_pos,TCP_offset)
        return pose

    def robot_get_ik(self, x: List[float]):
        """
        Calculate inverse kinematics for the robot.
        
        Parameters:
            x: Target pose as a list of floats [x, y, z, Rx, Ry, Rz].
        
        Returns:
            The joint configuration computed from IK or raises an error if IK fails.
        """
        q_robot = self.robot_get_joint_rad()  # Current joint configuration as seed
        joint_robot = self._ROBOT_CON_.getInverseKinematics(x, q_robot)
        
        if joint_robot is None:
            raise ValueError("Inverse kinematics failed to find a solution for the given pose.")
        
        return joint_robot
     
    # --------------------------
    # Movement Methods
    # --------------------------
    def robot_move_j(self, joint_rad=None, speed=0.01, acceleration=0.05, asynchronous=False) -> None:
        """Move robot joints to specified positions (in degrees)."""
        # if joint_degree is None:
        #     joint_degree = [0] * 6
        # joint_rad = [math.radians(deg) for deg in joint_degree]
        self._ROBOT_CON_.moveJ(q=joint_rad, speed=speed, acceleration=acceleration, asynchronous=asynchronous)

    def robot_move_jik(self, joint_rad=None, speed=0.01, acceleration=0.05, asynchronous=False) -> None:
        """Move robot joints to specified positions (in degrees)."""
        # if joint_degree is None:
        #     joint_degree = [0] * 6
        # joint_rad = [math.radians(deg) for deg in joint_degree]
        self._ROBOT_CON_.moveJ_IK(pose=joint_rad, speed=speed, acceleration=acceleration, asynchronous=asynchronous)

    def robot_move_j_stop(self, a=2.0, asynchronous=False) -> None:
        """Stop joint movement."""
        self._ROBOT_CON_.stopJ(a, asynchronous)

    def robot_move_speed(self, velocity, acceleration, time) -> None:
        """Move robot with a linear speed."""
        self._ROBOT_CON_.speedL(xd=velocity, acceleration=acceleration, time=time)

    def robot_move_speed_stop(self, acceleration=0.1) -> None:
        """Stop linear speed movement."""
        self._ROBOT_CON_.speedStop(a=acceleration)

    def robot_is_joint_move(self) -> bool:
        """Check if any joint is moving based on joint velocities."""
        res = self._ROBOT_RECV_.getActualQd()
        vel_max = max(res)
        print(f"getActualQd = {res}, vel_max = {vel_max}")
        return abs(vel_max) > 0.0001

    def robot_io_digital_set(self, id: int, signal: bool):
        """Set a digital output signal."""
        return self._ROBOT_IO_.setStandardDigitalOut(id, signal)

    def robot_moveL(self, pose: list, speed: float = 0.25, acceleration: float = 1.2, asynchronous=False) -> None:
        """Move robot linearly to the given pose."""
        self._ROBOT_CON_.moveL(pose, speed, acceleration, asynchronous=asynchronous)

    def my_robot_moveL(self,
                        robotDH,
                        pose: list,
                        dt: float   ,
                        speed: float     ,
                        acceleration: float ,
                        visualize: bool = False
                        ) -> None:
            """
            Move the robot linearly to the given pose using a Cartesian cubic
            trajectory and resolved-rate IK with speedJ.
            """
            # 1) CURRENT & TARGET SE3
            pos_current = self.robot_get_position()
            T_current = SE3(pos_current[0], pos_current[1], pos_current[2]) @ SE3.RPY(pos_current[3], pos_current[4], pos_current[5], unit='rad')

            T_goal =  SE3(pose[0], pose[1], pose[2]) @ SE3.RPY(pose[3], pose[4], pose[5], unit='rad')

            # 2) COMPUTE CARTESIAN TRAJECTORY
            planning = Planning(dt)
            pos_start = T_current.t  # This is a (3,) numpy array [x, y, z]
            pos_end   = T_goal.t  # This is a (3,) numpy array [x, y, z]
            dist = np.linalg.norm(pos_end - pos_start)
            T_total, profile = planning.compute_traj_time(dist, speed, acceleration)

            # time vector
            num_steps = int(np.ceil(T_total / dt)) + 1  # Use np.ceil for consistency
            t_vec = np.linspace(0, T_total, num_steps)

            # preallocate
            pos_traj = np.zeros((len(t_vec), 3))
            speed_traj = np.zeros((len(t_vec), 3))  # Initialize speed trajectory array
            acc_traj = np.zeros((len(t_vec), 3))  # Initialize acceleration trajectory array

            v0, v1 = 0.0, 0.0
            # per-axis cubic

            for axis in range(3):
                t_arr, p_arr, v_arr, a_arr = planning.cubic_trajectory(
                    pos_start[axis], pos_end[axis], v0, v1, T_total
                )
                pos_traj[:, axis] = p_arr.flatten()
                speed_traj[:, axis] = v_arr.flatten()
                acc_traj[:, axis] = a_arr.flatten()

            traj_T = []
            for j, t in enumerate(t_vec):
                s = t / T_total  # normalized time [0, 1]
                T_interp = T_current.interp(T_goal, s)  # Interpolate orientation
                # Replace the translation with the cubic trajectory value for consistency:
                T_interp = SE3.Rt(T_interp.R[:3, :3], pos_traj[j, :])
                traj_T.append(T_interp)
            # build SE3 waypoints (orientation by interp, position from pos_traj)
            SE3_waypoints = []
            for T in traj_T:
                SE3_waypoints.append(T)

            # 3) INVERSE KINEMATICS FOR ALL WAYPOINTS
            # Note: if use this it will take too mcuh time to run so skip it!
            # joint_traj = []
            # for idx, T_pose in enumerate(traj_T):
            #     pos = T_pose.t.tolist()
            #     orientation = T_pose.rpy()
            #     tcp_pose_list = pos + list(orientation)
            #     q_joint = self.robot_get_ik(tcp_pose_list)
            #     if q_joint is None:
            #         print(f"IK failed for waypoint {idx}.")
            #         break  # or handle the error as needed
            #     joint_traj.append(q_joint)
            # if len(joint_traj) != len(traj_T):
            #     raise RuntimeError("Incomplete joint trajectory. Please check IK solutions for all waypoints.")

            print("Successfully computed joint trajectory for all waypoints.")
            # 4) EXECUTE WITH RESOLVED-RATE IK (speedJ)
            joint_vels = []   # for visualization
            timestamps = []
            start_time = time.time()
            joint_traj = []

            Kp = 0.5  # Proportional gain for position error

            for i in range(num_steps):
                current_time = time.time() - start_time
                timestamps.append(current_time)
                # 1) feed‑forward Cartesian speed
                v_ff = speed_traj[i, :] 
                ang_vel = np.zeros(3)  # Initialize angular velocity
                v_ff = np.concatenate((v_ff, ang_vel))  # Shape: (6,)
                v_ff = np.array(v_ff, dtype=float)  # Ensure 
                # # 2) Cartesian position error
                tcp_pose = self.robot_get_position()
                tcp_pose = [tcp_pose[0],tcp_pose[1],tcp_pose[2]] 
                Pe    = tcp_pose
                p_des = traj_T[i].t                         # desired [x,y,z]
                e_vec = p_des - Pe 
                v_fb   = Kp * e_vec
                kp_max = speed / np.where(e_vec==0, np.inf, e_vec)
                print(f"[{i:03d}] t={current_time:.3f}s  error={e_vec}  kp_max={kp_max}")
                v_total = np.hstack((v_ff[:3] + v_fb, np.zeros(3)))  # shape (6,)


                # 5) map to joint velocities
                q_current = self.robot_get_joint_rad()  # Current joint configuration
                joint_traj.append(q_current)
                # compute analytic Jacobian
                J = robotDH.jacob0(q_current)
                dq = np.linalg.inv(J) @ v_total

                if np.linalg.cond(J) < 1e-09:
                    print("Jacobian is singular, cannot compute joint velocities.")
                joint_vels.append(dq)



                # send the speedJ command
                self.robot_speed_J(dq.tolist(), acceleration=acceleration, time=dt)

                # wait for the next step
                time.sleep(dt)


            # if user asked for visualization data, return it
            if visualize:
                return SE3_waypoints, np.array(joint_traj), np.array(joint_vels), timestamps

            # otherwise, we’re done
            time.sleep(0.5)  # wait for last command to finish
            self.robot_move_speed_stop()
            print(">> my_robot_moveL: trajectory executed successfully")
            return True


    def robot_moveL_stop(self, a=10.0, asynchronous=False) -> None:
        """Stop linear movement."""
        self._ROBOT_CON_.stopL(a, asynchronous)

    def robot_speed_J(self, velocity: list, acceleration: float, time: Optional[float] = None) -> None:
        """Set joint velocity.
        qd - joint speeds [rad/s]
        acceleration - joint acceleration [rad/s^2] (of leading axis)
        time - time [s] before the function returns (optional)
        """
        if time is not None:
            self._ROBOT_CON_.speedJ(velocity, acceleration, time)
        else:
            self._ROBOT_CON_.speedJ(velocity, acceleration)

    def robot_move_home(self):
        HOME_POS = [0.701172053107018, 0.184272460738082, 0.1721568294843568,
                    -1.7318488600590023, 0.686830145115122, -1.731258978679887]
        SPEED = 0.05
        ACC = 0.25
        print("Moving to home position...")
        self.robot_moveL(HOME_POS, SPEED, ACC)
        print("Arrived at home position.")
        time.sleep(1)


    # def my_convert_position_from_left_to_avatar(self,position: list[float]) -> list[float]:
    #     '''
    #     Convert TCP Position from Robot (Left) Ref to Avatar Ref
    #     '''
        
    #     # swap axis z    y   x
    #     res = [-position[2], -position[1], -position[0]]

    #     # translation
    #     res[0] -= 0.055
    #     res[1] += 0.400

    #     return res
    
    # def my_convert_position_from_avatar_to_left(self, position: list[float]) -> list[float]:
    #     '''
    #     Convert TCP Position from Avatar Reference back to Robot (Left) Reference.
    #     This is the inverse of my_convert_position_from_left_to_avatar.
        
    #     Given (from the original conversion):
    #     Avatar[0] = -Robot[2] - 0.055
    #     Avatar[1] = -Robot[1] + 0.400
    #     Avatar[2] = -Robot[0]
        
    #     Then the inverse conversion is:
    #     Robot[0] = -Avatar[2]
    #     Robot[1] = 0.400 - Avatar[1]
    #     Robot[2] = -(Avatar[0] + 0.055)
    #     '''
    #     res = [-position[2], 0.400 - position[1], -(position[0] + 0.055)]
    #     return res
    
    # def convert_gripper_to_maker(self, position: list[float]) -> list[float]:
    #     '''
    #     Convert TCP Position from Gripper Ref to Marker Ref
    #     '''
    
    #     res = [position[0], position[1], position[2]]

    #     # translation
    #     res[0] += 0.18
    #     res[1] += 0.18

    #     return res
       
    # def my_transform_position_to_world_ref(self, position: list[float]) -> list[float]:
    #     """
    #     Convert position from local robot reference to world (avatar) reference.
    #     Applies:
    #     - Rotation about Z by -pi/2
    #     - Rotation about Y by pi
    #     - Translation by (0.75, 0, 1.51)
        
    #     The conversion is done as:
    #         p_world = R * p_robot + t
    #     where R = Ry @ Rz and t = [0.75, 0.0, 1.51].
    #     """
    #     from math import cos, sin, pi
    #     import numpy as np
        
    #     # Rotation about Z by -pi/2:
    #     Rz = np.array([
    #         [cos(-pi/2), -sin(-pi/2), 0],
    #         [sin(-pi/2),  cos(-pi/2), 0],
    #         [0,           0,          1]
    #     ])
    #     # Rotation about Y by pi:
    #     Ry = np.array([
    #         [ cos(pi), 0, sin(pi)],
    #         [ 0,       1, 0      ],
    #         [-sin(pi), 0, cos(pi)]
    #     ])
    #     # Combined rotation:
    #     R = Ry @ Rz
    #     # Translation vector:
    #     t = np.array([0.75, 0.0, 1.51])
        
    #     # Compute final world position:
    #     pos_final = R @ np.array(position) + t
    #     return pos_final.tolist()

    

