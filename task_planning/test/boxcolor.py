import cv2
import numpy as np

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- Define 4 corner points manually ---
src_pts = np.array([
    [274, 527],   # bottom-left
    [280, 288],    # top-left
    [870, 290],   # top-right
    [875, 527]   # bottom-right
], dtype="float32")

# --- Define destination size ---
width, height = 1280, 720
dst_pts = np.array([
    [0, height - 1],      # bottom-left
    [0, 0],               # top-left
    [width - 1, 0],       # top-right
    [width - 1, height - 1]  # bottom-right
], dtype="float32")

# Perspective transform matrix
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# HSV color ranges
color_ranges = {
    'red1':    ([0, 100, 100], [10, 255, 255]),
    'red2':    ([160, 100, 100], [180, 255, 255]),
    'yellow':  ([20, 100, 100], [30, 255, 255]),
    'blue':    ([100, 150, 0], [140, 255, 255])
}

draw_colors = {
    'red': (0, 0, 255),
    'yellow': (0, 255, 255),
    'blue': (255, 0, 0)
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror view

    # Warp the frame to the defined workspace
    warped = cv2.warpPerspective(frame, M, (width, height))

    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

    # --- RED: merge both masks ---
    mask_red1 = cv2.inRange(hsv, np.array(color_ranges['red1'][0]), np.array(color_ranges['red1'][1]))
    mask_red2 = cv2.inRange(hsv, np.array(color_ranges['red2'][0]), np.array(color_ranges['red2'][1]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Detect red
    for mask, label in [(mask_red, 'red')]:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:
                x, y, w, h = cv2.boundingRect(cnt)
                cx = x + w // 2
                cy = y + h // 2
                cv2.rectangle(warped, (x, y), (x + w, y + h), draw_colors[label], 2)
                cv2.circle(warped, (cx, cy), 5, draw_colors[label], -1)
                cv2.putText(warped, f"{label} ({cx},{cy})", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_colors[label], 2)

    # Detect yellow and blue
    for label in ['yellow', 'blue']:
        lower = np.array(color_ranges[label][0])
        upper = np.array(color_ranges[label][1])
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:
                x, y, w, h = cv2.boundingRect(cnt)
                cx = x + w // 2
                cy = y + h // 2
                cv2.rectangle(warped, (x, y), (x + w, y + h), draw_colors[label], 2)
                cv2.circle(warped, (cx, cy), 5, draw_colors[label], -1)
                cv2.putText(warped, f"{label} ({cx},{cy})", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_colors[label], 2)

    # Show warped workspace only
    cv2.imshow("Warped Workspace", warped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
