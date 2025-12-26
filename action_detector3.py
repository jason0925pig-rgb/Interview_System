import cv2
import numpy as np
import time
import json
from ultralytics import YOLO


# Load YOLO pose model
model = YOLO("yolo11m-pose.pt")


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


# Previous wrist positions for detecting fidgeting
prev_left_wrist = None
prev_right_wrist = None


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

    # Center points
    shoulder_center = (
        (l_shoulder[0] + r_shoulder[0]) / 2,
        (l_shoulder[1] + r_shoulder[1]) / 2
    )
    hip_center = (
        (l_hip[0] + r_hip[0]) / 2,
        (l_hip[1] + r_hip[1]) / 2
    )

    # -----------------------------
    # 1. Arms Crossed
    # -----------------------------
    if (
        distance(l_wrist, r_elbow) < 80 and
        distance(r_wrist, l_elbow) < 80
    ):
        actions.append("arms_crossed")

    # -----------------------------
    # 2. Hands Clasped
    # -----------------------------
    if distance(l_wrist, r_wrist) < 60:
        actions.append("hands_clasped")

    # -----------------------------
    # 3. Chin Rest
    # -----------------------------
    if distance(l_wrist, nose) < 70 or distance(r_wrist, nose) < 70:
        actions.append("chin_rest")

    # -----------------------------
    # 4. Lean Forward
    # -----------------------------
    torso_height = abs(shoulder_center[1] - hip_center[1])
    if torso_height < 120:
        actions.append("lean_forward")

    # -----------------------------
    # 5. Lean Back
    # -----------------------------
    if torso_height > 200:
        actions.append("lean_back")

    # -----------------------------
    # 6. Head Down
    # -----------------------------
    if nose[1] > shoulder_center[1] + 40:
        actions.append("head_down")

    # -----------------------------
    # 7. Touch Face
    # -----------------------------
    face_center = (
        (left_eye[0] + right_eye[0]) / 2,
        (left_eye[1] + right_eye[1]) / 2
    )

    if distance(l_wrist, face_center) < 70 or distance(r_wrist, face_center) < 70:
        actions.append("touch_face")

    # -----------------------------
    # 8. Touch Nose
    # -----------------------------
    if distance(l_wrist, nose) < 40 or distance(r_wrist, nose) < 40:
        actions.append("touch_nose")

    # -----------------------------
    # 9. Fix Hair
    # -----------------------------
    if (
        distance(l_wrist, left_ear) < 60 or distance(r_wrist, right_ear) < 60 or
        distance(l_wrist, right_ear) < 60 or distance(r_wrist, left_ear) < 60
    ):
        actions.append("fix_hair")

    # -----------------------------
    # 10. Fidget Hands (fast wrist movement)
    # -----------------------------
    fidget_detected = False

    if prev_left_wrist is not None:
        if distance(prev_left_wrist, l_wrist) > 25:
            fidget_detected = True

    if prev_right_wrist is not None:
        if distance(prev_right_wrist, r_wrist) > 25:
            fidget_detected = True

    if fidget_detected:
        actions.append("fidget_hands")

    # Save last wrist positions
    prev_left_wrist = l_wrist
    prev_right_wrist = r_wrist

    return list(set(actions))



# ============================================================
# Main program
# ============================================================

cap = cv2.VideoCapture(0)

start_time = None   # Will be set on first successful frame
event_logs = []     # All action logs
frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Set timestamp start at the first frame successfully captured
    if start_time is None:
        start_time = time.time()

    current_time = time.time() - start_time  # seconds since first frame

    results = model(frame, device="cpu", verbose=False)

    for r in results:
        if r.keypoints is None:
            continue

        for person in r.keypoints.xy:
            kp = person.cpu().numpy()

            actions = detect_custom_actions(kp)

            # Draw detected actions on screen
            y = 30
            for act in actions:
                cv2.putText(frame, f"ACTION: {act}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y += 30

            # Save actions to log
            if len(actions) > 0:
                mm = int(current_time // 60)
                ss = int(current_time % 60)

                event_logs.append({
                    "time": f"{mm:02d}:{ss:02d}",
                    "timestamp_seconds": round(current_time, 2),
                    "actions": actions
                })

    cv2.imshow("Interview Action Detector - 10 Actions", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# ============================================================
# Save event logs to JSON file
# ============================================================

with open("action_log.json", "w", encoding="utf-8") as f:
    json.dump(event_logs, f, indent=4)

print("\nSaved action_log.json successfully!")
