import cv2
# from cv2 import destroyAllWindows
import mediapipe as mp
import numpy as np
import tkinter as tk

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])

    angle = radians * 180.0 / np.pi

    return angle - 180

# PROGRAM START ----------------------------------------------------------------------------------

# webcam input
# OPEN WINDOW
window = tk.Tk()
output_label = tk.Label(window, text="Initializing...", font=("Arial", 50))
output_label.pack()

cap = cv2.VideoCapture(0)


# Create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video = cv2.VideoWriter("output_video.mp4", fourcc, 30.0, (1920, 1080), isColor=True)


# Curl counter variables
counter = 0
stage = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Create a blank transparent
        output = np.zeros((1080, 1920, 3), dtype=np.uint8)
        output[:, :, 1] = 255  # Set green channel to 255
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]

            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]

            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]

            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Calculate angle

            left_angle_hip = calculate_angle(left_shoulder, left_hip, left_knee)
            left_angle_hip = round(left_angle_hip, 2)
            right_angle_hip = calculate_angle(right_shoulder, right_hip, right_knee)
            right_angle_hip = round(right_angle_hip, 2)

            left_angle_knee = calculate_angle(left_hip, left_knee, left_ankle)  # Knee joint angle
            left_angle_knee = round(left_angle_knee, 2)
            right_angle_knee = calculate_angle(right_hip, right_knee, right_ankle)
            right_angle_knee = round(right_angle_knee, 2)




            # Render detections
            mp_drawing.draw_landmarks(output, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            output_video.write(output)



            # ---------------------------------------------
            # Visualize angle
        except:
            pass
        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

        # Rep data
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(image, 'STAGE', (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage,
                    (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # DEBUG
        #print(angles)

cap.release()
output_video.release()
# out.release()
cv2.destroyAllWindows()


# destroyAllWindows()


