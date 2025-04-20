import time
import rtde_control
import rtde_receive
import rtde.rtde as rtde

def main():
    # Replace with your robot's IP address
    robot_ip = "192.168.200.10"

    # Create RTDE control and receive interfaces
    rtde_c = rtde_control.RTDEControlInterface(robot_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

    # Define a target pose for moveL
    # Format: [X, Y, Z, Rx, Ry, Rz]
    # X, Y, Z in meters; Rx, Ry, Rz in radians (axis-angle rotation)
    target_pose = [0.5502667757508777, 0.20303105602786847, 0.18085563973956334, -1.7556926715044847, 0.71148022042343, -1.7637475489007919]

    # Perform a linear move (moveL) with specified acceleration and velocity
    acceleration = 0.01  # m/s^2
    velocity = 0.01      # m/s
    rtde_c.moveL(target_pose, acceleration, velocity)
    
    # Pause briefly to let the move finish
    time.sleep(1)

    # Read and print the actual TCP pose to verify
    actual_pose = rtde_r.getActualTCPPose()
    print("Current TCP pose:", actual_pose)

    # (Optional) Disconnect from the robot
    # For some versions of ur_rtde, an explicit disconnect is not strictly necessary
    rtde_c.disconnect()

if __name__ == "__main__":
    main()
