# print_pose_cam.py
import cv2 as cv
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from classrobot import realsense_cam

class PrintPoseCam():
    def __init__(self):
        self.cam = realsense_cam.RealsenseCam()

    def stop_all(self):
        self.cam.stop()

    def cam_relasense(self):
        aruco_dict_type = cv.aruco.DICT_5X5_1000
        point3d = self.cam.cam_capture_marker(aruco_dict_type)
        return point3d

    def printPoseCam(self):

        maker_point = self.cam_relasense()
        print(maker_point)
        transfrom_point = self.cam.transform_marker_points(maker_point)
        # print(transfrom_point)
        # Sort the markers by their id in ascending order.
        sorted_markers = sorted(transfrom_point, key=lambda m: m["id"])
    
        for marker in sorted_markers:
            marker_id = marker["id"]
            point = marker["point"]  # This is an instance of Point3D.
            print(marker_id)
            print(point.x, point.y, point.z)    
         
        
def main():
    try:
        printPoseCam = PrintPoseCam()
        printPoseCam.printPoseCam()
        printPoseCam.stop_all()
        print("Finished")

    except Exception as e:
        print(f"Error initializing Move2Object: {e}")
        return


if __name__ == "__main__":
    main()
