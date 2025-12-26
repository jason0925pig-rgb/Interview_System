import cv2
import numpy as np
import json
import time
from ultralytics import YOLO


# Load YOLO pose model
model = YOLO("yolo11m-pose.pt")


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


# Variables for fidget detection
prev_left_wrist = None
prev_right_wrist = None

# Log dictionary to store actions with timestamps
actions_log = {}


def detect_custom_actions(kp):
    """
    kp: pose keypoints, shape (17,2)
    """

    global prev_left_wrist, prev_right_wrist

    nose = kp[0]
    left_eye, right_eye = kp[1], kp[2]
    left_ear, right_ear = kp[3], kp[4]

    l_shoulder, r_shoulder = kp[5], kp[6]
    l_elbow, r_elbow = kp[7], kp[8]
    l_wrist, r_wrist = kp[9], kp[10]

    l_hip, r_hip = kp[11], kp[12]

    actions = []

    # Body center points
    shoulder_center = ((l_shoulder[0] + r_shoulder[0]) / 2,
                       (l_shoulder[1] + r_shoulder[1]) / 2)
    hip_center = ((l_hip[0] + r_hip[0]) / 2,
                  (l_hip[1] + r_hip[1]) / 2)

    # 1. Arms crossed
    if (
        distance(l_wrist, r_elbow) < 80 and
        distance(r_wrist, l_elbow) < 80
    ):
        actions.append("arms_crossed")

    # 2. Hands clasped
    if distance(l_wrist, r_wrist) < 60:
        actions.append("hands_clasped")

    # 3. Chin rest
    if distance(l_wrist, nose) < 70 or distance(r_wrist, nose) < 70:
        actions.append("chin_rest")

    # 4. Lean forward
    torso_height = abs(shoulder_center[1] - hip_center[1])
    if torso_height < 120:
        actions.append("lean_forward")

    # 5. Lean back
    if torso_height > 200:
        actions.append("lean_back")

    # 6. Head down
    if nose[1] > shoulder_center[1] + 40:
        actions.append("head_down")

    # 7. Touch face
    face_center = ((left_eye[0] + right_eye[0]) / 2,
                   (left_eye[1] + right_eye[1]) / 2)
    if distance(l_wrist, face_center) < 70 or distance(r_wrist, face_center) < 70:
        actions.append("touch_face")

    # 8. Touch nose
    if distance(l_wrist, nose) < 40 or distance(r_wrist, nose) < 40:
        actions.append("touch_nose")

    # 9. Fix hair
    if (
        distance(l_wrist, left_ear) < 60 or distance(r_wrist, right_ear) < 60 or
        distance(l_wrist, right_ear) < 60 or distance(r_wrist, left_ear) < 60
    ):
        actions.append("fix_hair")

    # 10. Fidget hands (fast movement)
    fidget_detected = False
    if prev_left_wrist is not None:
        if distance(prev_left_wrist, l_wrist) > 25:
            fidget_detected = True

    if prev_right_wrist is not None:
        if distance(prev_right_wrist, r_wrist) > 25:
            fidget_detected = True

    if fidget_detected:
        actions.append("fidget_hands")

    prev_left_wrist = l_wrist
    prev_right_wrist = r_wrist

    return list(set(actions))  # remove duplicates


# Open webcam
cap = cv2.VideoCapture(0)

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Use CPU mode
    results = model(frame, device="cpu", verbose=False)

    # Compute timestamp
    elapsed = int(time.time() - start_time)
    timestamp = time.strftime("%H:%M:%S", time.gmtime(elapsed))

    frame_actions = []

    for r in results:
        if r.keypoints is None:
            continue

        for person in r.keypoints.xy:
            kp = person.cpu().numpy()

            actions = detect_custom_actions(kp)

            # Add actions to frame-level list
            frame_actions.extend(actions)

            # Display on screen
            y = 30
            for act in actions:
                cv2.putText(frame, f"ACTION: {act}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y += 30

    # Save actions into log if any detected
    if len(frame_actions) > 0:
        actions_log[timestamp] = frame_actions

    # Show frame
    cv2.imshow("Interview Action Detector - 10 Actions", frame)

    # Quit when 'q' is pressed → also write JSON
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera
cap.release()
cv2.destroyAllWindows()

# Save JSON log
with open("actions_log.json", "w") as f:
    json.dump(actions_log, f, indent=4)

print("Action log saved to actions_log.json")
