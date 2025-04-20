import pyrealsense2 as rs
import cv2
import numpy as np
import math
from .point3d import Point3D
import json
import pathlib
import sys
from scipy.spatial.transform import Rotation as R
import math 

config_path = pathlib.Path(__file__).parent.parent / "FRA631_Project_Dual_arm_UR5_Calibration/caribration/config/cam.json"
print("Loading camera configuration from:", config_path)
jsonObj = json.load(open(config_path))
json_string = str(jsonObj).replace("'", '\"')



class RealsenseCam:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.config = None
        self.profile = None
        self.align = None
        self.align_depth = None 
        self.imu_pipe = None
        self.imu_config = None
        self.init_cam()
        print("RealsenseCam initialized with width: {}, height: {}, fps: {}".format(width, height, fps))
  
    def init_cam(self):
        """
        Initialize the RealSense camera pipeline with color, depth, and motion streams.
        This method is called in the constructor.
        """
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.imu_pipe = rs.pipeline()
            self.imu_config = rs.config()

            if self.pipeline is None or self.config is None:
                raise RuntimeError("Failed to create RealSense pipeline or config.")

            # Enable color and depth streams before starting the pipeline.
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            
            # Enable motion streams without specifying fps; device defaults will be used.
            self.imu_config.enable_stream(rs.stream.gyro)
            self.imu_config.enable_stream(rs.stream.accel)

            # Start the pipeline and get the profile.
            self.profile = self.pipeline.start(self.config)
            self.imu_pipe.start(self.imu_config)

            # Align depth to color.
            self.align = rs.align(rs.stream.color)
            self.align_depth = rs.align(rs.stream.depth)

            # Get device and enable advanced mode.
            dev = self.profile.get_device()
            if not dev:
                raise RuntimeError("Failed to get RealSense device.")

            advnc_mode = rs.rs400_advanced_mode(dev)
            if not advnc_mode.is_enabled():
                print("Enabling advanced mode...")
                advnc_mode.toggle_advanced_mode(True)

            # Load settings from config JSON.
            advnc_mode.load_json(json_string)

            print(f"RealSense camera started with aligned color and depth streams "
                f"({self.width}x{self.height}@{self.fps}fps)")
        except Exception as e:
            print(f"[ERROR] Failed to initialize RealSense camera: {e}")
            raise  # Or use sys.exit(1) if running as a standalone script


    def get_color_frame(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            return None
        image = np.asanyarray(color_frame.get_data())
        return image

    def get_depth_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame:
            return None
        # Return the depth frame (do not convert to numpy array so get_distance() remains available)
        return depth_frame

    def get_color_and_depth_frames(self) -> tuple:
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None, None
        
        # Get intrinsics from the depth stream.
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        color_image = np.asanyarray(color_frame.get_data())
        # Return the color image, the raw depth frame, and the intrinsics.
        return color_image, depth_frame, depth_intrinsics


    def get_color_intrinsics(self, depth_intrinsics) -> tuple:
        """
        Retrieve the color camera's intrinsic matrix and distortion coefficients.

        Returns:
            camera_matrix (np.ndarray): 3x3 intrinsic matrix.
            dist_coeffs (np.ndarray): Distortion coefficients array.
        """
        # Reuse our get_color_and_depth_frames to get the intrinsics
        
        if depth_intrinsics is None:
            return None, None
        # Build the 3x3 camera matrix from the intrinsics.
        camera_matrix = np.array([[depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                                  [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                                  [0, 0, 1]], dtype=np.float32)
        # Get the distortion coefficients.
        dist_coeffs = np.array(depth_intrinsics.coeffs, dtype=np.float32)
        return camera_matrix, dist_coeffs

    def get_depth_intrinsics(self, depth_frame) -> tuple:
        """
        Retrieve the depth camera's intrinsic matrix and distortion coefficients.

        Returns:
            camera_matrix (np.ndarray): 3x3 intrinsic matrix.
            dist_coeffs (np.ndarray): Distortion coefficients array.
        """
        if not depth_frame:
            return None, None
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        camera_matrix = np.array([[depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                                  [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                                  [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array(depth_intrinsics.coeffs, dtype=np.float32)
        return camera_matrix, dist_coeffs


    def get_all_board_pose(self, aruco_dict):
        """
        Detects Aruco markers in the color image, computes the center pixel of each marker,
        deprojects the center pixel using RealSense depth data to obtain a 3D point, and returns
        a list of markers with their IDs and corresponding 3D positions.
        
        Returns:
        - output_image: The color image with detected markers and center points drawn.
        - marker_points: A list of dictionaries. Each dictionary contains:
                {"id": marker_id, "point": Point3D(...)}
        """
        color_image, depth_frame, depth_intrinsics = self.get_color_and_depth_frames()
        if color_image is None or depth_frame is None:
            print("Failed to capture color/depth.")
            return None, []
        
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
        output_image = color_image.copy()

        marker_points = []
        
        if ids is not None and len(ids) > 0:
            # Draw detected markers on the image.
            cv2.aruco.drawDetectedMarkers(output_image, corners, ids)
            
            # Process each detected marker.
            for i in range(len(ids)):
                c = corners[i][0]  # corners for marker i; shape (4,2)
                cx = int(np.mean(c[:, 0]))
                cy = int(np.mean(c[:, 1]))
                # Draw the center for visualization.
                cv2.circle(output_image, (cx, cy), 4, (0, 0, 255), -1)
                
                # Get depth at the marker center.
                depth = depth_frame.get_distance(cx, cy)
                if depth > 0:
                    point_coords = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cx, cy], depth)
                    # Depending on your coordinate conventions, you may adjust the axes.
                    x = point_coords[0]
                    y = point_coords[1]
                    z = point_coords[2]
                    point3d = Point3D(x, y, z)
                else:
                    print(f"Invalid depth at marker {i} (pixel: {cx},{cy}).")
                    point3d = Point3D(0, 0, 0)
                
                # Extract the marker id.
                marker_id = int(ids[i][0])
                marker_points.append({"id": marker_id, "point": point3d})
                # print(f"Marker id {marker_id} at pixel ({cx}, {cy}) -> 3D: {point3d}")
                
            return output_image, marker_points
        else:
            print("No markers detected.")
            return output_image, []


    def get_board_pose(self, aruco_dict):
        color_image, depth_frame, depth_intrinsics = self.get_color_and_depth_frames()
        if color_image is None or depth_frame is None:
            print("Failed to capture color/depth.")
            return None, Point3D(0, 0, 0)  # Use Point3D, not Points3D

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
        output_image = color_image.copy()

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

            all_cx = []
            all_cy = []
            for i in range(len(ids)):
                c = corners[i][0]
                cx = int(np.mean(c[:, 0]))
                cy = int(np.mean(c[:, 1]))
                all_cx.append(cx)
                all_cy.append(cy)
                cv2.circle(output_image, (cx, cy), 4, (0, 0, 255), -1)

            cx = int(np.mean(all_cx))
            cy = int(np.mean(all_cy))
            point2d = [cx, cy]
            print(cx,cy)
            depth = depth_frame.get_distance(cx, cy)
            if depth > 0:
                point_coords = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cx, cy], depth)
                # Convert to Point3D 
                # x = result[2]
                # y = -point_coords[1]
                # z = -point_coords[2]
                x = point_coords[0]
                y = point_coords[1]
                z = point_coords[2]
                point3d = Point3D(x, y, z)
                print(f"Detected board center at: {point3d}")
                return output_image, point3d
            else:
                print("Invalid depth.")
                return output_image, Point3D(0, 0, 0)
        else:
            print("No markers detected.")
            return output_image, Point3D(0, 0, 0)

    def get_board_pose_estimate(self, aruco_dict, camera_matrix, dist_coeffs, marker_length=0.05):
        """
        Estimates the 6DoF pose of the detected Aruco marker board using a hybrid approach:
        - The rotation (rvec) is obtained from OpenCV’s estimatePoseSingleMarkers.
        - The translation (tvec) is derived from the RealSense depth data.
        
        Parameters:
        - aruco_dict: The dictionary used for marker detection 
            (e.g., cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)).
        - camera_matrix: The camera intrinsic matrix (3x3 numpy array).
        - dist_coeffs: Distortion coefficients (numpy array).
        - marker_length: The side length of the marker in meters (default is 0.05).
        
        Returns:
        - output_image: The color image with detected markers and coordinate axes drawn.
        - rvec: The rotation vector (from OpenCV) of the first detected marker, or None.
        - tvec: The translation (as a Point3D) computed from the RealSense depth data, or a Point3D at (0,0,0).
        """
        # Capture color and depth frames.
        color_image, depth_frame, depth_intrinsics = self.get_color_and_depth_frames()
        if color_image is None or depth_frame is None:
            print("Failed to capture color/depth.")
            return None, None, Point3D(0, 0, 0)
        
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # Create an Aruco detector and detect markers.
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
        output_image = color_image.copy()
        
        if ids is not None and len(ids) > 0:
            # Draw detected markers for visualization.
            cv2.aruco.drawDetectedMarkers(output_image, corners, ids)
            
            # Use estimatePoseSingleMarkers to obtain rvecs and tvecs for each marker.
            rvecs, tvecs_dummy, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                markerLength=marker_length,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs
            )
            
            if rvecs is not None and len(rvecs) > 0:
                # For the first detected marker, compute its center pixel.
                pts = corners[0][0]  # 4 corners for marker 0
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                
                # Use the RealSense depth data to compute a translation.
                depth = depth_frame.get_distance(cx, cy)
                if depth > 0:
                    # Deproject the center pixel to 3D using the depth intrinsics.
                    point_coords = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cx, cy], depth)
                    tvec_depth = np.array([point_coords[0], point_coords[1], point_coords[2]]).reshape(1, 3)
                    point3d = Point3D(tvec_depth[0][0], tvec_depth[0][1], tvec_depth[0][2])
                else:
                    print("Invalid depth at marker center.")
                    return output_image, None, Point3D(0, 0, 0)
                
                # Draw coordinate axes for each detected marker using the computed tvec_depth.
                for marker_index in range(len(rvecs)):
                    cv2.drawFrameAxes(output_image, camera_matrix, dist_coeffs, 
                                    rvecs[marker_index], tvec_depth.reshape(1, 3), marker_length)
                    
                # Use the first marker's rotation vector as the estimated rotation.
                rvec = rvecs[0]
                # Convert the rotation vector to a rotation matrix.
                rot_matrix, _ = cv2.Rodrigues(rvec)
                # Convert rotation matrix to quaternion and Euler angles via scipy.
                r = R.from_matrix(rot_matrix)
                quat = r.as_quat()   # in (x, y, z, w) order
                roll, pitch, yaw = self.euler_from_quaternion(quat[0], quat[1], quat[2], quat[3])
                print("Roll (deg):", math.degrees(roll))
                print("Pitch (deg):", math.degrees(pitch))
                print("Yaw (deg):", math.degrees(yaw))
                print("Estimated translation from depth:", point3d)
                print("Estimated rotation vector:", rvec.ravel())
                
                return output_image, rvec, point3d
            else:
                print("estimatePoseSingleMarkers returned no rotation data.")
                return output_image, None, Point3D(0, 0, 0)
        else:
            print("No markers detected.")
            return output_image, None, Point3D(0, 0, 0)




    # def get_board_pose_estimate(self, aruco_dict, camera_matrix, dist_coeffs, marker_length=0.05):
    #     """
    #     Estimates the 6DoF pose of the detected Aruco marker board using estimatePoseSingleMarkers.
        
    #     Parameters:
    #     - aruco_dict: The dictionary used for marker detection 
    #         (e.g., cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)).
    #     - camera_matrix: The camera intrinsic matrix (3x3 numpy array).
    #     - dist_coeffs: Distortion coefficients (numpy array, e.g., np.zeros((5,1), dtype=np.float32)).
    #     - marker_length: The actual side length of the marker in meters (default is 0.05).
        
    #     Returns:
    #     - output_image: The color image with detected markers and the coordinate axes drawn.
    #     - rvec: The rotation vector for the first detected marker (if available) or None.
    #     - tvec: The translation (as a Point3D) for the first detected marker (if available) or a Point3D at (0,0,0).
    #     """
    #     # Capture color and depth frames.
    #     color_image, depth_frame, _ = self.get_color_and_depth_frames()
    #     if color_image is None or depth_frame is None:
    #         print("Failed to capture color/depth.")
    #         return None, None, Point3D(0, 0, 0)
        
    #     gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
    #     # Create an Aruco detector and detect markers.
    #     parameters = cv2.aruco.DetectorParameters()
    #     detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    #     corners, ids, rejected = detector.detectMarkers(gray)
    #     output_image = color_image.copy()
        
    #     if ids is not None and len(ids) > 0:
    #         # Draw detected markers for visualization.
    #         cv2.aruco.drawDetectedMarkers(output_image, corners, ids)
            
    #         # Use estimatePoseSingleMarkers to obtain rvecs and tvecs for each marker.
    #         rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
    #             corners,
    #             markerLength=marker_length,
    #             cameraMatrix=camera_matrix,
    #             distCoeffs=dist_coeffs
    #         )
            
    #         # Check if pose estimation returned any data.
    #         if rvecs is not None and len(rvecs) > 0:
    #             # For each detected marker, draw coordinate axes on the image.
    #             for marker_index in range(len(rvecs)):
    #                 cv2.drawFrameAxes(output_image, camera_matrix, dist_coeffs, 
    #                                 rvecs[marker_index], tvecs[marker_index], marker_length)
                
    #             # Use the first marker's pose as an example.
    #             # tvecs is of shape (num_markers, 1, 3); extract the first one.
    #             tvec = tvecs[0]
    #             rvec = rvecs[0]
    #             # Convert the translation to a Point3D.
    #             point3d = Point3D(tvec[0][0], tvec[0][1], tvec[0][2])
                
    #             # Convert the rotation vector to a rotation matrix.
    #             rot_matrix, _ = cv2.Rodrigues(rvec)
    #             # If you want to work with quaternions and Euler angles, you can do so. For example,
    #             # using the scipy.spatial.transform.Rotation package:
    #             #
    #             # from scipy.spatial.transform import Rotation as R
    #             r = R.from_matrix(rot_matrix)
    #             quat = r.as_quat()   # returns in (x, y, z, w) format
    #             roll, pitch, yaw = self.euler_from_quaternion(quat[0], quat[1], quat[2], quat[3])
    #             print("Roll (deg):", math.degrees(roll))
    #             print("Pitch (deg):", math.degrees(pitch))
    #             print("Yaw (deg):", math.degrees(yaw))
    #             #
    #             # For now, we simply print the raw rotation vector.
    #             print("Estimated translation (tvec):", point3d)
    #             print("Estimated rotation vector (rvec):", rvec.ravel())
                
    #             return output_image, rvec, point3d
    #         else:
    #             print("estimatePoseSingleMarkers returned no pose data.")
    #             return output_image, None, Point3D(0, 0, 0)
    #     else:
    #         print("No markers detected.")
    #         return output_image, None, Point3D(0, 0, 0)



    def euler_from_quaternion(self,x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
            
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
            
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
            
        return roll_x, pitch_y, yaw_z # in radians
    

    def get_imu_frames(self):
        """
        Retrieves accelerometer and gyroscope data.
        
        Returns:
            accel_data: A namedtuple with fields x, y, z from the accelerometer.
            gyro_data: A namedtuple with fields x, y, z from the gyroscope.
        """
        # self.imu_pipe.start(self.imu_config)
        mot_frames = self.imu_pipe.wait_for_frames()
        if mot_frames:
            accel_data = mot_frames[0].as_motion_frame().get_motion_data()
            gyro_data = mot_frames[1].as_motion_frame().get_motion_data()
            return accel_data, gyro_data
        else:
            print("Failed to retrieve IMU data.")
            return None, None


    def stop(self):
        """
        Stop the RealSense pipeline.
        """
        if self.pipeline is None and self.imu_pipe is None:
            print("Pipeline is not initialized.")
            return
        if self.pipeline and self.imu_pipe:
            print("Stopping RealSense camera...")
            self.pipeline.stop()
            self.imu_pipe.stop()
            self.imu_pipe = None
            self.pipeline = None
            self.config = None
            self.profile = None
            self.align = None
            self.align_depth = None 
            print("RealSense camera stopped.")
        else:
            print("Pipeline is already stopped or not initialized.")
 

    def restart(self):
        """
        Restart the RealSense pipeline.
        """
        if self.pipeline is None:
            print("Pipeline is not initialized. Cannot restart.")
            return
        print("Restarting RealSense camera...")
        # Stop the pipeline before reinitializing
        self.stop()
        self.init_cam()
        print("RealSense camera restarted with aligned color and depth streams.")

    def __del__(self):
        self.stop()



