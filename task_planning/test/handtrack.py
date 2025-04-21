import cv2
import mediapipe as mp

# Initialize webcam
webcam = cv2.VideoCapture(1)  # Change to 1 if needed
if not webcam.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

# Main loop
while True:
    success, frame = webcam.read()
    if not success:
        print("⚠️ Failed to grab frame.")
        break

    # Flip frame horizontally for a mirror view
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with MediaPipe
    results = hands.process(image_rgb)
    
    # Draw landmarks if any hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Display the frame
    cv2.imshow("Hand Tracking", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
