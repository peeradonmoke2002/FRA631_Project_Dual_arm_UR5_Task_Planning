# move2object.py
import time
import cv2 as cv
import sys
import pathlib
import numpy as np
import json
from spatialmath import SE3
from classrobot.UR5e_DH import UR5eDH

# Append the parent directory to sys.path to allow relative module imports.
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from classrobot import robot_movement, realsense_cam
from classrobot.point3d import Point3D
from classrobot import gripper



HOME_POS = [0.701172053107018, 0.184272460738082, 0.1721568294843568, 
            -1.7318488600590023, 0.686830145115122, -1.731258978679887]
GRAP_RPY = [-1.7224438319206319, 0.13545161633255984, -1.2975236351897372]
RPY = [-1.7318443587261685, 0.686842056802218, -1.7312759524010408]

HOME_POS_GRAP = [0.701172053107018, 0.184272460738082, 0.1721568294843568, 
            -1.7224438319206319, 0.13545161633255984, -1.2975236351897372]  


class Move2Object:

    def __init__(self):

        self.HOME_POS = [0.701172053107018, 0.184272460738082, 0.1721568294843568,
                         -1.7318488600590023, 0.686830145115122, -1.731258978679887]
        self.HOME_POS_GRAP = [0.701172053107018, 0.184272460738082, 0.1721568294843568, 
            -1.7224438319206319, 0.13545161633255984, -1.2975236351897372]  
        self.robot_ip = "192.168.200.10"
        self.speed = 0.06
        self.acceleration = 0.20
        self.FIX_Y = 0.18427318897339476
        self.RPY = [-1.7318443587261685, 0.686842056802218, -1.7312759524010408]
        self.medium_RPY = [-1.7224438319206319, 0.13545161633255984, -1.2975236351897372]
        self.high_RPY =  [-1.7224438319206319, 0.13545161633255984, -1.2975236351897372]
        self.robot = robot_movement.RobotControl()
        self.robot.robot_release()
        self.robot.robot_init(self.robot_ip)
        self.cam = realsense_cam.RealsenseCam()
        self.dt = 0.01
 
        self.recive_pos = [ ]
        self.getrobotDH()
    
    
        self._GRIPPER_LEFT_ = gripper.MyGripper3Finger()
        self.init_gripper()
        # Track which empty markers have been used (markers with IDs 100, 101, 102).
        self.used_empty_markers = []
        self.group_left = set(range(100, 104))
        self.group_right = set(range(105, 108))
        

    def init_gripper(self):
        # Initialize the gripper connection.
        host = "192.168.200.11"  # Replace with your gripper's IP address.
        port = 502               # Typically the default Modbus TCP port.
        print(f"Connecting to 3-Finger {host}:{port}", end="")

        res = self._GRIPPER_LEFT_.my_init(host=host, port=port)
        if res:
            print("[SUCCESS]")
        else:
            print("[FAILURE]")
            self._GRIPPER_LEFT_.my_release()
            exit()

        time.sleep(0.6)  # Slight delay.
        print("Testing gripper ...", end="")
        self.close_gripper()  # Test the close command.
        time.sleep(2)
        self.open_gripper()   # Test the open command.
        time.sleep(2)

    def stop_all(self):
        self.robot.robot_release()
        self.cam.stop()
        self._GRIPPER_LEFT_.my_release()

    def close_gripper(self):
        gr = gripper.MyGripper3Finger()
        # Initialize the gripper
        gr.my_init(host="192.168.200.11", port=502)
        time.sleep(0.6)
        # Test closing
        gr.my_hand_close()
        time.sleep(2)
      
        

    def open_gripper(self):
        gr = gripper.MyGripper3Finger()
        # Initialize the gripper
        gr.my_init(host="192.168.200.11", port=502)
        time.sleep(0.6)
        # Test closing
        gr.my_hand_open()
        time.sleep(2)

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
        self.robot.robot_moveL(self.HOME_POS,self.speed)

    def move_home_rpy(self):
        print("Moving to home position...")
        self.robot.robot_moveL(self.HOME_POS_GRAP,self.speed)

    def transform_marker_points(self, maker_point):
        transformed_points = self.cam.transform_marker_points(maker_point)
        return transformed_points


    def min_marker(self, transformed_points):
        """
        Find the marker with the lowest Z *only* among real boxes (ID < 100).
        """
        # keep only real‑box IDs
        candidates = [m for m in transformed_points if m["id"] < 100]
        if not candidates:
            print("No box markers found.")
            return None

        # pick the one with min Z
        bottom = min(candidates, key=lambda m: m["point"].y)
        print(f"Marker with lowest z: ID {bottom['id']} (z={bottom['point'].y})")
        return bottom

    def max_marker(self, transformed_points):
        # keep only real-box IDs
        candidates = [m for m in transformed_points if m["id"] < 100]
        if not candidates:
            print("No box markers found.")
            return None

        top = max(candidates, key=lambda m: m["point"].y)
        print(f"Marker with highest z: ID {top['id']} (z={top['point'].y})")
        return top

    def box_distances(self, markers):
        real_boxes = [m for m in markers if m["id"] < 100]

        for i in range(len(real_boxes)):
            for j in range(i + 1, len(real_boxes)):
                id1 = real_boxes[i]["id"]
                id2 = real_boxes[j]["id"]
                p1 = real_boxes[i]["point"]
                p2 = real_boxes[j]["point"]
                dx = p1.x - p2.x
                dz = p1.z - p2.z
                distance = (dx**2 + dz**2)**0.5
                print(f"ID {id1}-{id2}: x{dx:+.3f} z{dz:+.3f}")
    
    def getrobotDH(self):
        self.robotDH = UR5eDH()
        tool_offset = SE3(0, 0, 0.200)
        self.robotDH.tool = tool_offset

    def pick_box(self, marker):
 
        p = marker["point"]
        approach = [p.x , p.y - 0.20, p.z] + RPY
        pick_pos = [p.x , p.y-0.01 ,p.z] + GRAP_RPY

        print(f"[PICK] Marker ID {marker['id']} at (x={p.x}, y={p.y}, z={p.z})")
        # move above in X–Z
        self.robot.my_robot_moveL(self.robotDH, approach, self.dt, self.speed, self.acceleration, False)
        time.sleep(3) 
        tcp_pose_goal = self.get_robot_TCP()
        pos_current_goal = tcp_pose_goal[:3]+ GRAP_RPY
        self.robot.robot_moveL(pos_current_goal, speed=self.speed, acceleration=self.acceleration)
        self.robot.my_robot_moveL(self.robotDH, pick_pos, self.dt, self.speed, self.acceleration, False)
        # grip
        self.close_gripper()
        time.sleep(3)
        # retract back to approach pose
        self.robot.my_robot_moveL(self.robotDH, approach, self.dt, self.speed, self.acceleration, False)
        time.sleep(3)
    


    def place_box(self, marker_point):
        p = marker_point
        # approach pose (X–Z move, Y fixed)
        approach = [p.x, p.y - 0.20, p.z] + RPY
        # actual place pose
        place_pos = [p.x, p.y-0.08,p.z] + GRAP_RPY
        print(f"[PLACE] at marker pos (x={p.x}, y={p.y}, z={p.z})")
        # move above in X–Z
        print("move linear")

        self.robot.my_robot_moveL(self.robotDH, approach, self.dt, self.speed, self.acceleration, False)
        self.robot.my_robot_moveL(self.robotDH, place_pos, self.dt, self.speed, self.acceleration, False)
        # grip
        
        self.open_gripper()

        self.robot.my_robot_moveL(self.robotDH, approach, self.dt, self.speed, self.acceleration, False)
        time.sleep(3)

    

    def find_next_stack_position(self, current_id, sorted_markers):
        """Finds the next marker that has a higher ID."""
        for marker in sorted_markers:
            if marker["id"] > current_id:
                return marker
        return None


    def sort_pick_and_place(self):

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

    
    def stack_chain_boxes(self, target_id=104, count=3):

        # Gather the first N box markers by ascending ID
        raw_pts_all  = self.cam_relasense()
        transformed_all = self.transform_marker_points(raw_pts_all)
        box_markers = sorted([m for m in transformed_all if m['id'] < 100],
                            key=lambda m: m['id'])[:count]

        prev_id = target_id
        for marker in box_markers:
            # Refresh marker map right before placing
            raw_pts     = self.cam_relasense()
            transformed = self.transform_marker_points(raw_pts)
            point_map   = {m['id']: m['point'] for m in transformed}

            if prev_id not in point_map:
                print(f"[STACK] Cannot find updated position for marker {prev_id}, abort chain.")
                return

            place_pt = point_map[prev_id]
            print(f"[STACK] Picking box {marker['id']} → placing at marker {prev_id} position")

            # pick the box
            self.pick_box(marker)
            # place it at the freshly read position
            self.place_box(place_pt)
            # go home
            self.move_home()
            time.sleep(1)

            # next time, we’ll stack onto the just-picked box’s ID
            prev_id = marker['id']

    def detect_overlaps(self, pts):

        STACK_Y_THRESHOLD = 0.34

        # 1) Filter down to just the real boxes
        real_boxes = [m for m in pts if m["id"] < 100]

        # 2) Collect those whose Y exceeds the threshold
        stacked = [m for m in real_boxes if m["point"].y < STACK_Y_THRESHOLD]

        # 3) Return as one group if non‑empty, else no groups
        return [stacked] if stacked else []
    
    def destack_within_zone(self, group_ids):
        """
        Destack only within the specified group of marker IDs (e.g., left or right).
        """
        while True:
            raw_pts = self.cam_relasense()
            transformed = self.transform_marker_points(raw_pts)
            self.box_distances(transformed)

            overlaps = self.detect_overlaps(transformed)
            if not overlaps:
                print("[DESTACK] No more stacks in this zone.")
                break

            lowest = self.min_marker(transformed)
            if lowest is None:
                print("[DESTACK] No boxes found.")
                break

            candidates = [
                m for m in transformed
                if m["id"] in group_ids
                and m["id"] not in self.used_empty_markers
            ]
            if not candidates:
                print("[DESTACK] No empty spots in specified group.")
                break

            spot = candidates[0]
            self.used_empty_markers.append(spot["id"])
            print(f"[DESTACK] Moving box {lowest['id']} → marker {spot['id']}")
            self.pick_box(lowest)
            self.place_box(spot["point"])
            self.move_home()
            time.sleep(1)
        
        
def main():
    mover = Move2Object()
    mover.move_home()
    time.sleep(2)
    while True:
        # ----- Phase 1: Destack only within the left or right zone -----
        print("Choose an operation:")
        print("  1) move all box on LEFT zone (100–104)")
        print("  2) move all box on RIGHT zone (104–108)")
        print("  3) Custom stacking")

        op_choice = input("Enter 1/2/3: ").strip()

        if op_choice == "1":
            mover.destack_within_zone(mover.group_left)
            again = input("Do you want to perform another operation? (y/n): ").strip().lower()
            if again != "y":
                break
            continue
        elif op_choice == "2":
            mover.destack_within_zone(mover.group_right)
            again = input("Do you want to perform another operation? (y/n): ").strip().lower()
            if again != "y":
                break
            continue
        elif op_choice != "3":
            print("Invalid choice. Exiting.")
            break
        elif op_choice == "3":
            # ----- Phase 2: Arrange remaining boxes: CLI menu -----
            raw_pts     = mover.cam_relasense()
            transformed = mover.transform_marker_points(raw_pts)
            box_ids   = sorted(m["id"] for m in transformed if m["id"] < 100)
            empty_ids = sorted(m["id"] for m in transformed if m["id"] >= 100)

            # Check if blocks are currently stacked
            overlaps = mover.detect_overlaps(transformed)
            is_stacked = bool(overlaps)

            print("\nHow would you like to stack the remaining boxes?")
            # Only allow sequential stacking options if blocks are not stacked
            if not is_stacked:
                print("  1) min→max IDs:", box_ids)
                print("  2) max→min IDs:", list(reversed(box_ids)))
            print("  3) custom sequence")

            choice = input("Enter 1/2/3: ").strip()

            if choice == "1":
                seq = box_ids
            elif choice == "2":
                seq = list(reversed(box_ids))
            else:
                # custom: show both lists and read a space‐separated list
                print("Available boxes:", box_ids)
                print("Available empty spots:", empty_ids)
                tokens = input("Type your sequence (e.g. '3 7 8'): ").split()
                seq = [int(t) for t in tokens]

            # Always prompt user to select destination marker from available empty spots
            print("Available empty spots for placement:", empty_ids)
            while True:
                try:
                    dest = int(input("Select the destination marker ID from above: ").strip())
                    if dest in empty_ids:
                        break
                    else:
                        print("Please select a valid empty marker ID from the list.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            print(f"\n[ARRANGE] Stacking {seq} onto marker {dest}")
            # Now walk through your sequence:
            for bid in seq:
                # find its current 3D point
                pt = next(m["point"] for m in transformed if m["id"] == bid)
                print(f"[STACK] Picking box {bid} → placing at marker {dest}")
                mover.pick_box({"id": bid, "point": pt})
                # re‐read dest’s pose just in case:
                mover.move_home_rpy()
                raw_pts2  = mover.cam_relasense()
                xf2       = mover.transform_marker_points(raw_pts2)
                
                dest_pt   = next(m["point"] for m in xf2 if m["id"] == dest)
                # Append GRAP_RPY to the destination point
            
                mover.place_box(dest_pt)
                mover.move_home()
                time.sleep(1)
                dest = bid    # next time we stack onto *this* box’s original spot

            # ----- Cleanup -----
            mover.move_home()
            time.sleep(1)
            again = input("Do you want to perform another operation? (y/n): ").strip().lower()
            if again != "y":
                break
            continue
    mover.stop_all()
    print("Program completed. Robot and camera released.")

if __name__ == "__main__":
    main()
