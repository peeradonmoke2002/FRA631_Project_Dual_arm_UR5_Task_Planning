import time
import sys
import pathlib
import numpy as np
import cv2 as cv
from spatialmath import SE3

# Robotics and device imports
from classrobot.UR5e_DH import UR5eDH
from classrobot import robot_movement, realsense_cam, gripper
from classrobot.point3d import Point3D

# Default positions and orientations
HOME_POS = [
    0.701172053107018, 0.184272460738082, 0.1721568294843568,
    -1.7318488600590023, 0.686830145115122, -1.731258978679887
]
GRIP_RPY = [-1.7224438319206319, 0.13545161633255984, -1.2975236351897372]
PLACE_RPY = [-1.7318443587261685, 0.686842056802218, -1.7312759524010408]

class PickPlaceBox:
    """
    Handles pick-and-place operations using a UR5e arm, RealSense camera, and 3-finger gripper.
    """
    def __init__(self, robot_ip="192.168.200.10", gripper_ip="192.168.200.11", dt=0.01):
        # Robot setup
        self.robot = robot_movement.RobotControl()
        self.robot.robot_release()
        self.robot.robot_init(robot_ip)

        # Kinematics
        self.robotDH = UR5eDH()
        self.robotDH.tool = SE3(0, 0, 0.200)

        # Camera setup
        self.cam = realsense_cam.RealsenseCam()

        # Gripper setup
        self.gripper = gripper.MyGripper3Finger()
        self._init_gripper(gripper_ip)

        # Control parameters
        self.speed = 0.1
        self.acceleration = 0.25
        self.dt = dt
        self.FIXED_Y = 0.18427318897339476

        # Marker tracking
        self.used_empty_markers = []  # IDs 100, 101, 102

    def _init_gripper(self, host, port=502):
        print(f"Connecting to gripper at {host}:{port}...", end="")
        if self.gripper.my_init(host=host, port=port):
            print("SUCCESS")
            # test open/close
            self.close_gripper()
            time.sleep(1)
            self.open_gripper()
            time.sleep(1)
        else:
            print("FAILURE")
            self.gripper.my_release()
            sys.exit(1)

    def stop_all(self):
        """Release all devices."""
        self.robot.robot_release()
        self.cam.stop()
        self.gripper.my_release()

    def close_gripper(self):
        """Close the 3-finger gripper."""
        self.gripper.my_hand_close()

    def open_gripper(self):
        """Open the 3-finger gripper."""
        self.gripper.my_hand_open()

    def capture_markers(self):
        """Capture and return ArUco marker detections."""
        return self.cam.cam_capture_marker(cv.aruco.DICT_5X5_1000)

    def transform_marker_points(self, markers):
        """Convert camera-frame markers to robot-world coordinates."""
        return self.cam.transform_marker_points(markers)

    def get_robot_tcp(self):
        """Get current end-effector pose."""
        pose = self.robot.robot_get_position()
        print("TCP:", pose)
        return pose

    def move_home(self):
        """Move robot to predefined home pose."""
        print("Moving home...")
        self.robot.robot_moveL(HOME_POS, speed=self.speed)

    def box_distances(self, markers):
        """Print X-Z distances between all real-box markers (ID < 100)."""
        real = [m for m in markers if m["id"] < 100]
        for i in range(len(real)):
            for j in range(i+1, len(real)):
                p1, p2 = real[i]["point"], real[j]["point"]
                dx, dz = p1.x - p2.x, p1.z - p2.z
                print(f"IDs {real[i]['id']}–{real[j]['id']}: Δx={dx:.3f}, Δz={dz:.3f}")

    def detect_overlaps(self, markers, y_thresh=0.34):
        """Return list of real-box markers below y_thresh (considered in a stack)."""
        return [m for m in markers if m["id"] < 100 and m["point"].y < y_thresh]

    def pick_box(self, marker):
        """Approach and pick a box at the given marker."""
        p = marker["point"]
        approach = [p.x, p.y + 0.20, p.z] + PLACE_RPY
        pick_pose = [p.x, p.y, p.z] + GRIP_RPY

        print(f"[PICK] ID {marker['id']} at ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})")
        self.robot.my_robot_moveL(self.robotDH, approach, self.dt, self.speed, self.acceleration)
        time.sleep(1)
        self.robot.my_robot_moveL(self.robotDH, pick_pose, self.dt, self.speed, self.acceleration)
        self.close_gripper()
        time.sleep(1)
        self.robot.my_robot_moveL(self.robotDH, approach, self.dt, self.speed, self.acceleration)
        time.sleep(1)

    def place_box(self, point):
        """Approach and place a box at the given point."""
        approach = [point.x, self.FIXED_Y, point.z] + PLACE_RPY
        place_pose = [point.x, point.y - 0.06, point.z] + GRIP_RPY

        print(f"[PLACE] at ({point.x:.3f}, {point.y:.3f}, {point.z:.3f})")
        self.robot.my_robot_moveL(self.robotDH, approach, self.dt, self.speed, self.acceleration)
        self.robot.my_robot_moveL(self.robotDH, place_pose, self.dt, self.speed, self.acceleration)
        self.open_gripper()
        time.sleep(1)
        self.robot.my_robot_moveL(self.robotDH, approach, self.dt, self.speed, self.acceleration)
        time.sleep(1)

    def phase_destack(self):
        """De-stack all piles onto empty markers 100–102."""
        while True:
            raw = self.capture_markers()
            trans = self.transform_marker_points(raw)
            overlaps = self.detect_overlaps(trans)
            if not overlaps:
                print("No more stacks.")
                break

            # pick lowest box
            lowest = min(overlaps, key=lambda m: m["point"].y)
            empties = [m for m in trans if m["id"] in {100,101,102} and m["id"] not in self.used_empty_markers]
            if not empties:
                print("No empty slots left.")
                break

            spot = empties[0]
            self.used_empty_markers.append(spot["id"])
            self.pick_box(lowest)
            self.place_box(spot["point"])
            self.move_home()

    def phase_chain_stack(self, target_id=104, count=3):
        """Chain-stack `count` boxes onto `target_id` and then onto each other."""
        raw_all = self.capture_markers()
        all_pts = self.transform_marker_points(raw_all)
        boxes = sorted([m for m in all_pts if m["id"] < 100], key=lambda m: m["id"])[:count]

        prev_id = target_id
        for m in boxes:
            raw = self.capture_markers()
            updated = {pt["id"]: pt["point"] for pt in self.transform_marker_points(raw)}
            if prev_id not in updated:
                print(f"Missing marker {prev_id}, aborting.")
                return

            print(f"Stacking box {m['id']} onto {prev_id}")
            self.pick_box(m)
            self.place_box(updated[prev_id])
            self.move_home()
            prev_id = m["id"]

if __name__ == "__main__":
    picker = PickPlaceBox()
    picker.move_home()
    time.sleep(1)

    # Example: run de-stack then chain-stack
    picker.phase_destack()
    picker.phase_chain_stack(target_id=104, count=3)

    picker.move_home()
    picker.stop_all()
    print("Done.")
