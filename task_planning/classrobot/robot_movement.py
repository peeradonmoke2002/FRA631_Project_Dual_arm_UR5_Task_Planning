import rtde_control
import rtde_receive
import rtde_io
import time
import math
from typing import Optional, List
import numpy as np
from math import pi, cos, sin

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
    
    def robot_get_fk(self, q: Optional[List[float]] = None, tcp_offset: Optional[List[float]] = None):
         if q is None and tcp_offset is None:
             return self._ROBOT_RECV_.getForwardKinematics()
         else:
            return self._ROBOT_RECV_.getForwardKinematics(q, tcp_offset)


    def robot_get_ik(self, 
                     x: List[float], 
                     qnear: Optional[List[float]] = None, 
                     maxPositionError: Optional[float] = None, 
                     maxOrientationError: Optional[float] = None):
        return self._ROBOT_CON_.getInverseKinematics(x)

        
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

    def robot_move_speed(self, velocity) -> None:
        """Move robot with a linear speed."""
        self._ROBOT_CON_.speedL(xd=velocity, acceleration=0.1, time=0)

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



    def my_convert_position_from_left_to_avatar(self,position: list[float]) -> list[float]:
        '''
        Convert TCP Position from Robot (Left) Ref to Avatar Ref
        '''
        
        # swap axis z    y   x
        res = [-position[2], -position[1], -position[0]]

        # translation
        res[0] -= 0.055
        res[1] += 0.400

        return res
    
    def my_convert_position_from_avatar_to_left(self, position: list[float]) -> list[float]:
        '''
        Convert TCP Position from Avatar Reference back to Robot (Left) Reference.
        This is the inverse of my_convert_position_from_left_to_avatar.
        
        Given (from the original conversion):
        Avatar[0] = -Robot[2] - 0.055
        Avatar[1] = -Robot[1] + 0.400
        Avatar[2] = -Robot[0]
        
        Then the inverse conversion is:
        Robot[0] = -Avatar[2]
        Robot[1] = 0.400 - Avatar[1]
        Robot[2] = -(Avatar[0] + 0.055)
        '''
        res = [-position[2], 0.400 - position[1], -(position[0] + 0.055)]
        return res
    
    def convert_gripper_to_maker(self, position: list[float]) -> list[float]:
        '''
        Convert TCP Position from Gripper Ref to Marker Ref
        '''
    
        res = [position[0], position[1], position[2]]

        # translation
        res[0] += 0.18
        res[1] += 0.18

        return res

            
    def my_transform_position_to_world_ref(self, position: list[float]) -> list[float]:
        """
        Convert position from local robot reference to world (avatar) reference.
        Applies:
        - Rotation about Z by -pi/2
        - Rotation about Y by pi
        - Translation by (0.75, 0, 1.51)
        
        The conversion is done as:
            p_world = R * p_robot + t
        where R = Ry @ Rz and t = [0.75, 0.0, 1.51].
        """
        from math import cos, sin, pi
        import numpy as np
        
        # Rotation about Z by -pi/2:
        Rz = np.array([
            [cos(-pi/2), -sin(-pi/2), 0],
            [sin(-pi/2),  cos(-pi/2), 0],
            [0,           0,          1]
        ])
        # Rotation about Y by pi:
        Ry = np.array([
            [ cos(pi), 0, sin(pi)],
            [ 0,       1, 0      ],
            [-sin(pi), 0, cos(pi)]
        ])
        # Combined rotation:
        R = Ry @ Rz
        # Translation vector:
        t = np.array([0.75, 0.0, 1.51])
        
        # Compute final world position:
        pos_final = R @ np.array(position) + t
        return pos_final.tolist()

    

