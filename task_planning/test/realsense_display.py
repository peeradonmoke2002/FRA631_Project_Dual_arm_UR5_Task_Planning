import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # Configure the RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable the depth and color streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        while True:
            # Wait for frames (blocking call)
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            # If either frame is not available, skip
            if not depth_frame or not color_frame:
                continue

            # Convert frames to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply a color map to the depth image for easier viewing
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # Display both images
            cv2.imshow("RealSense Color", color_image)
            cv2.imshow("RealSense Depth", depth_colormap)

            # Press 'ESC' to exit
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        # Stop streaming and close windows
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
