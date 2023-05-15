import cv2
# from cv2 import destroyAllWindows
import mediapipe as mp
import numpy as np
import tkinter as tk

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def update_output(label, angles):
    # Update the output
    output = '\n'.join([f"{key}: {value}" for key, value in angles.items()])
    label.config(text=output)

    # Schedule the next update
    #label.after(1000, lambda: update_output(label, angles))
    label.master.update()


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle




# PROGRAM START ----------------------------------------------------------------------------------

angles = {}

# OPEN WINDOW
window = tk.Tk()
output_label = tk.Label(window, text="Initializing...", font=("Arial", 50))
output_label.pack()

# webcam input
# angle_min = []
# angle_min_hip = []
cap = cv2.VideoCapture(0)
# Curl counter variables
counter = 0
stage = None

"""width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('your_video.mp4', fourcc, 10.0, size, True)"""

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Calculate angle
            # angle = calculate_angle(shoulder, elbow, wrist)

            left_angle_hip = calculate_angle(left_shoulder, left_hip, left_knee)
            left_angle_hip = round(left_angle_hip, 2)
            right_angle_hip = calculate_angle(right_shoulder, right_hip, right_knee)
            right_angle_hip = round(right_angle_hip, 2)

            left_angle_knee = calculate_angle(left_hip, left_knee, left_ankle)  # Knee joint angle
            left_angle_knee = round(left_angle_knee, 2)
            right_angle_knee = calculate_angle(right_hip, right_knee, right_ankle)
            right_angle_knee = round(right_angle_knee, 2)

            left_hip_angle = 180 - left_angle_hip
            left_knee_angle = 180 - left_angle_knee
            right_hip_angle = 180 - right_angle_hip
            right_knee_angle = 180 - right_angle_knee

            # angle_min.append(angle_knee)
            # angle_min_hip.append(angle_hip)

            angles['left_angle_hip'] = left_angle_hip
            angles['left_angle_knee'] = left_angle_knee
            angles['right_angle_hip'] = right_angle_hip
            angles['right_angle_knee'] = right_angle_knee

            # SAVE TXT FILE -------------------------------

            with open('angles.txt', 'w') as f:
                f.write('\n'.join([str(value) for value in angles.values()]))

            # print(f'{left_hip_angle}\n{right_hip_angle}\n{left_knee_angle}\n{right_knee_angle}\n')
            update_output(output_label, angles)

            # ---------------------------------------------
            # Visualize angle
            """cv2.putText(image, str(angle), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )"""

            '''cv2.putText(image, str(angle_knee), 
                           tuple(np.multiply(knee, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (79, 121, 66), 2, cv2.LINE_AA
                                )'''

            """cv2.putText(image, str(angle_hip), 
                           tuple(np.multiply(hip, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )"""
        except:
            pass

        # Render squat counter
        # Setup status box
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

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        # out.write(image)
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # DEBUG
        #print(angles)

cap.release()
# out.release()
cv2.destroyAllWindows()

window.mainloop()

# destroyAllWindows()


