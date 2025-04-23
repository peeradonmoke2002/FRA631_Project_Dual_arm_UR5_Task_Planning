# sortbox.py
import time
import cv2 as cv
import sys
import pathlib
import numpy as np
import json
from spatialmath import SE3
# Append the parent directory to sys.path to allow relative module imports.
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from classrobot import robot_movement, realsense_cam
from classrobot.point3d import Point3D
from classrobot import gripper
from classrobot.UR5e_DH import UR5eDH

HOME_POS = [0.701172053107018, 0.184272460738082, 0.1721568294843568, 
            -1.7318488600590023, 0.686830145115122, -1.731258978679887]
GRAP_RPY = [-1.7224438319206319, 0.13545161633255984, -1.2975236351897372]
RPY = [-1.7318443587261685, 0.686842056802218, -1.7312759524010408]

safe_pos_right_hand = [-0.46091229133782063, -0.03561778716957069, 0.5597582676128492,
                        -0.026107352593298303, -1.6156259386208272, 0.011223536317220348]
HOME_POS_right_hand =    [-0.7000166613125681, 0.17996458960475226, 0.17004862107257468,
                           -0.014724582993124808, -1.5742326761027705, -0.016407326458784333] 
class SortBox:
    def __init__(self):
        self.robot_ip = "192.168.200.10"
        self.speed = 0.025
        self.acceleration = 1.0
        self.FIX_Y = 0.184273188973394
        self.recive_pos = [ ]
        # Initialize the robot connection once.
        self.getrobotDH()
        self.robot = robot_movement.RobotControl()
        self.robot.robot_release()
        self.robot.robot_init(self.robot_ip)
        self.cam = realsense_cam.RealsenseCam()
        self._GRIPPER_LEFT_ = gripper.MyGripper3Finger()
        self.init_gripper()
        # Track which empty markers have been used (markers with IDs 100, 101, 102).
        self.used_empty_markers = []
        self.dt = 0.01

    def init_gripper(self):
       
        # Initialize the gripper connection
        host = "192.168.200.11"  # Replace with your gripper's IP address
        port = 502              # Typically the default Modbus TCP port
        print(f"Connecting to 3-Finger {host}:{port}", end="")

        res = self._GRIPPER_LEFT_.my_init(host=host, port=port)
        if res:
            print("[SUCCESS]")
        else:
            print("[FAILURE]")
            self._GRIPPER_LEFT_.my_release()
            exit()

        # Delay slightly longer than the TIME_PROTECTION (0.5 s)
        print("Testing gripper ...", end="")
        self.close_gripper()  # Now this should actuate the close command
        self.open_gripper()    # Test open command
        self._GRIPPER_LEFT_.my_release()



    def init_gripper_before_grap(self):
       
        # Initialize the gripper connection
        host = "192.168.200.11"  # Replace with your gripper's IP address
        port = 502              # Typically the default Modbus TCP port
        print(f"Connecting to 3-Finger {host}:{port}", end="")

        res = self._GRIPPER_LEFT_.my_init(host=host, port=port)
        if res:
            print("[SUCCESS]")
        else:
            print("[FAILURE]")
            self._GRIPPER_LEFT_.my_release()
            exit()



    def stop_all(self):
        self.robot.robot_release()
        self.cam.stop()
        self._GRIPPER_LEFT_.my_release()

    def close_gripper(self):
        """
        Closes the gripper.
        """
        self.init_gripper_before_grap()
        time.sleep(0.6)
        print("Closing gripper...")
        self._GRIPPER_LEFT_.my_hand_close()
        time.sleep(2)
        self._GRIPPER_LEFT_.my_release()

    def open_gripper(self):
        """
        Opens the gripper.
        """
        self.init_gripper_before_grap()
        print("Opening gripper...")
        time.sleep(0.6)
        self._GRIPPER_LEFT_.my_hand_open()
        time.sleep(2)
        self._GRIPPER_LEFT_.my_release()


    def getrobotDH(self):
        self.robotDH = UR5eDH()
        tool_offset = SE3(0, 0, 0.200)
        self.robotDH.tool = tool_offset


    def cam_relasense(self):
        aruco_dict_type = cv.aruco.DICT_5X5_1000
        point3d = self.cam.cam_capture_marker(aruco_dict_type)
        return point3d



    def get_robot_TCP(self):
        """
        Connects to the robot and retrieves the current TCP (end-effector) position.
        Returns a 3-element list: [x, y, z].
        """
        pos = self.robot.robot_get_position()
        print("Robot TCP position:", pos)
        return pos
    
    def move_home(self):
        print("Moving to home position...")
        time.sleep(2)
        # self.robot.robot_moveL(HOME_POS, self.speed)
        self.robot.my_robot_moveL(self.robotDH, HOME_POS, self.dt, self.speed, self.acceleration, False)


    def detect_overlaps(self, pts, y_tol=0.45):
        """
        Group markers whose Y-coordinates differ by more than y_tol.
        Returns list of groups (each a list of markers).
        """
        groups = []
        used = set()
        n = len(pts)

        for i in range(n):
            if i in used:
                continue

            base = pts[i]
            group = [base]

            for j in range(i+1, n):
                if j in used:
                    continue

                other = pts[j]
                dy = abs(base["point"].y - other["point"].y)

                # if y > tol = stack
                if dy > y_tol:
                    group.append(other)
                    used.add(j)

            # ถ้าพบอย่างน้อย 2 ตัว → เก็บเป็นกลุ่ม
            if len(group) > 1:
                used.add(i)
                groups.append(group)

        return groups


    def max_marker(self, transformed_points):
        """
        Finds the marker with the maximum (highest) z-coordinate among all captured markers.
        Prints the marker ID and its z value, then returns that marker.
        """
        if not transformed_points:
            print("No markers found.")
            return None

        max_marker = transformed_points[0]
        # Loop over each marker and update the max_marker if a marker with a higher z is found.
        for marker in transformed_points:
            if marker["point"].z > max_marker["point"].z:
                max_marker = marker

        print(f"Marker with highest z: ID {max_marker['id']} with z = {max_marker['point'].z}")
        return max_marker

    def pick_box(self, marker):
        """
        1) Approach over the box: move in X–Z, keep Y fixed.
        2) Descend to the actual Y of the box and close gripper.
        3) Retract back to the approach pose.
        """
        p = marker["point"]
     
        approach = [p.x + 0.05, self.FIX_Y, p.z] + GRAP_RPY
        pick_pos = [p.x + 0.05, p.y,p.z] + GRAP_RPY

        print(f"[PICK] Marker ID {marker['id']} at (x={p.x}, y={p.y}, z={p.z})")
        # move above in X–Z
        self.robot.my_robot_moveL(self.robotDH, approach, self.dt, self.speed, self.acceleration, False)

        time.sleep(3)
        # descend in Y
        self.robot.my_robot_moveL(self.robotDH, pick_pos, self.dt, self.speed, self.acceleration, False)
        # grip
        self.close_gripper()
        time.sleep(3)
        # retract back to approach pose
        self.robot.my_robot_moveL(self.robotDH, approach, self.dt, self.speed, self.acceleration, False)
        time.sleep(3)

    def place_box(self, marker_point):
        """
        1) Approach over the place point in X–Z, Y fixed.
        2) Descend to actual Y of the place marker and open gripper.
        3) Retract back to the approach pose.
        """

                   
            # ----- change orientation ur code  ----- 
            # 
            #
            # 
            #  ----- move as linear our code  ----- 
            # self.robot.my_robot_moveL(self.robotDH, target_pose_up, self.dt, self.speed, self.acceleration, False)

        p = marker_point
        # approach pose (X–Z move, Y fixed)
        approach = [p.x+0.05, self.FIX_Y, p.z] + GRAP_RPY
        # actual place pose
        place_pos = [p.x+0.05, p.y-0.06,p.z] + GRAP_RPY

        tcp_pose_goal = self.get_robot_TCP()

        print(f"[PLACE] at marker pos (x={p.x}, y={p.y}, z={p.z})")
        # move above in X–Z
        self.robot.my_robot_moveL(self.robotDH, approach, self.dt, self.speed, self.acceleration, False)
        time.sleep(3)
        pos_current_goal = tcp_pose_goal[:3]+ GRAP_RPY
        time.sleep(3)
        self.robot.robot_moveL(pos_current_goal, speed=self.speed, acceleration=self.acceleration)

        # descend in Y
        self.robot.my_robot_moveL(self.robotDH, place_pos, self.dt, self.speed, self.acceleration, False)
        time.sleep(0.6)
        self.open_gripper()
        time.sleep(1)
        # retract
        self.robot.my_robot_moveL(self.robotDH, approach, self.dt, self.speed, self.acceleration, False)
        time.sleep(1)
   

    def find_next_stack_position(self, current_id, sorted_markers):
        """Finds the next marker that has a higher ID."""
        for marker in sorted_markers:
            if marker["id"] > current_id:
                return marker
        return None
    
    def transform_marker_points(self, maker_point):
        transformed_points = self.cam.transform_marker_points(maker_point)
        return transformed_points



    def sort_pick_and_place(self):
        """
        Main function: pick boxes by sorted ID and stack them at the next higher marker position.
        """

        marker_points = self.cam_relasense()
        transformed_points = self.transform_marker_points(marker_points)
        sorted_markers = sorted(transformed_points, key=lambda m: m["id"])

        for marker in sorted_markers:
            current_id = marker["id"]
            self.pick_box(marker)
            next_marker = self.find_next_stack_position(current_id, sorted_markers)
            if next_marker:
                self.place_box(next_marker["point"])
            else:
                print(f"[SKIP STACK] No higher marker found for marker ID {current_id}")

    def get_next_empty_marker(self, transformed_points):
        """
        Searches for an available empty marker among those with IDs 100, 101, or 102.
        If a marker has already been used for placement, it is skipped.
        Returns the marker dictionary if found; otherwise, returns None.
        """
        empty_ids = {100, 101}
        for marker in transformed_points:
            if marker["id"] in empty_ids and marker["id"] not in self.used_empty_markers:
                self.used_empty_markers.append(marker["id"])
                print(f"[EMPTY] Using empty marker ID {marker['id']}")
                return marker
        print("[EMPTY] No available empty marker found.")
        return None
    
    def stack_chain_boxes(self, target_id=104, count=3):
        """
        Stack the lowest-ID boxes in sequence:
        1) First box goes to the position of marker `target_id`.
        2) Each subsequent box goes to the original position of the previously stacked box’s marker.
        """
        raw_pts     = self.cam_relasense()
        transformed = self.transform_marker_points(raw_pts)
        # Build a map from marker ID → Point3D
        point_map = {m['id']: m['point'] for m in transformed}

        # Gather all box markers (IDs <100), sort by ascending ID, and take the first `count`
        box_markers = sorted([m for m in transformed if m['id'] < 100],
                             key=lambda m: m['id'])[:count]
        # Chain‐stack 
        prev_id = target_id
        for marker in box_markers:
            box_id = marker['id']

            # Make sure we know the placement point
            if prev_id not in point_map:
                print(f"[STACK] Cannot find position for marker {prev_id}, abort chain.")
                return

            place_pt = point_map[prev_id]
            print(f"[STACK] Picking box {box_id} → placing at marker {prev_id} position")
            self.pick_box(marker)
            self.place_box(place_pt)
            self.move_home()
            time.sleep(1)
            #Now the next box uses this box’s original marker ID
            prev_id = box_id 
 
def main():
    sortbox = SortBox()
    sortbox.move_home()
    time.sleep(2)

    # --- Destack loop ---
    while True:
        raw_pts     = sortbox.cam_relasense()
        transformed = sortbox.transform_marker_points(raw_pts)
        overlaps    = sortbox.detect_overlaps(transformed, y_tol=0.5)

        if not overlaps:
            print("[DESTACK] No stacked boxes detected. Now stacking 3 boxes at marker 104.")
            # Stack the three lowest‐ID boxes onto marker 104
            sortbox.stack_chain_boxes(target_id=104, num_boxes=3)
            break

        top   = sortbox.max_marker(transformed)
        empty = sortbox.get_next_empty_marker(transformed)
        if top is None or empty is None:
            print("[DESTACK] Cannot proceed, exiting loop.")
            break

        sortbox.pick_box(top)
        sortbox.place_box(empty["point"])
        sortbox.move_home()
        time.sleep(1) 

    # Final cleanup
    sortbox.move_home()
    time.sleep(1)
    sortbox.stop_all()
    print("Program completed. Robot and camera released.")


if __name__ == "__main__":
    main()
