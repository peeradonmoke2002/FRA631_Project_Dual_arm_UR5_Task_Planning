import time
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from classrobot.robot_movement import RobotControl

def main():
    robot_ip = "192.168.200.10"
    robot = RobotControl()
    robot.robot_release()  # Release any previous connections
    print("Connecting to robot...")
    robot.robot_init(robot_ip)

    # Define a joint velocity command: here only the first joint moves at 0.1 rad/s
    joint_velocity = [0, 0, 0, 0, 0, 0.1]
    acceleration = 1.0   # rad/s^2
    time_interval = 0.1  # seconds per command cycle
    duration = 2.0       # total duration of motion in seconds

    try:
        start_time = time.time()
        while time.time() - start_time < duration:
            # Call the method to send joint velocity via speedJ
            robot.robot_speed_J(joint_velocity, acceleration, time_interval)
            time.sleep(time_interval)
        
        # Stop the robot after executing the commands
        robot.robot_release()
        print("Robot stopped successfully.")
        
    except Exception as e:
        robot.robot_release()
        print(f"An error occurred: {e}")
        print("Robot stopped due to error.")

    




if __name__ == "__main__":
    main()
