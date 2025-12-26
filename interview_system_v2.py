import cv2
import numpy as np
import json
import time
from ultralytics import YOLO
from deepface import DeepFace

# ------------------------------------------------------
# Pose-based action detection utilities
# ------------------------------------------------------

def distance(p1, p2):
    """Euclidean distance between two points (x, y)."""
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

# For fidget detection across frames
prev_left_wrist = None
prev_right_wrist = None


def detect_custom_actions(kp):
    """
    Detect interview-related body actions from pose keypoints.

    kp: numpy array with shape (17, 2) -> (x, y) for each keypoint.
    Returns: list of action strings.
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
        (l_shoulder[1] + r_shoulder[1]) / 2,
    )
    hip_center = (
        (l_hip[0] + r_hip[0]) / 2,
        (l_hip[1] + r_hip[1]) / 2,
    )

    # 1. arms_crossed (双手抱胸)
    if distance(l_wrist, r_elbow) < 80 and distance(r_wrist, l_elbow) < 80:
        actions.append("arms_crossed")

    # 2. hands_clasped (双手扣在一起)
    if distance(l_wrist, r_wrist) < 60:
        actions.append("hands_clasped")

    # 3. chin_rest (托下巴 / 手撑脸)
    if distance(l_wrist, nose) < 70 or distance(r_wrist, nose) < 70:
        actions.append("chin_rest")

    # 4. lean_forward (身体前倾)
    torso_height = abs(shoulder_center[1] - hip_center[1])
    if torso_height < 120:
        actions.append("lean_forward")

    # 5. lean_back (身体后仰)
    if torso_height > 200:
        actions.append("lean_back")

    # 6. head_down (低头)
    if nose[1] > shoulder_center[1] + 40:
        actions.append("head_down")

    # 7. touch_face (手碰脸的区域)
    face_center = (
        (left_eye[0] + right_eye[0]) / 2,
        (left_eye[1] + right_eye[1]) / 2,
    )
    if distance(l_wrist, face_center) < 70 or distance(r_wrist, face_center) < 70:
        actions.append("touch_face")

    # 8. touch_nose（摸鼻子）
    if distance(l_wrist, nose) < 40 or distance(r_wrist, nose) < 40:
        actions.append("touch_nose")

    # 9. fix_hair（整理头发，手靠近耳朵）
    if (
        distance(l_wrist, left_ear) < 60
        or distance(r_wrist, right_ear) < 60
        or distance(l_wrist, right_ear) < 60
        or distance(r_wrist, left_ear) < 60
    ):
        actions.append("fix_hair")

    # 10. fidget_hands（手的小动作 / 抖动）
    fidget_detected = False
    if prev_left_wrist is not None and distance(prev_left_wrist, l_wrist) > 25:
        fidget_detected = True
    if prev_right_wrist is not None and distance(prev_right_wrist, r_wrist) > 25:
        fidget_detected = True
    if fidget_detected:
        actions.append("fidget_hands")

    # Update previous wrist positions
    prev_left_wrist = l_wrist
    prev_right_wrist = r_wrist

    # Remove duplicates
    return list(set(actions))


# ------------------------------------------------------
# Main real-time interview system with DeepFace
# ------------------------------------------------------

def main():
    # 1) Load models
    pose_model = YOLO("yolo11m-pose.pt")  # make sure this file is in the same folder

    print("Loading DeepFace emotion model (first time can be slow)...")
    # DeepFace will internally load face detector + emotion model on first call

    # 2) Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    # Logs for JSON
    combined_log = []

    # Start time (from first successful frame)
    start_time = None

    print("Press 'q' to stop and save combined_log.json")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        if start_time is None:
            start_time = time.time()

        elapsed = time.time() - start_time  # seconds since first frame

        # ---------------- Pose / action detection ----------------
        results = pose_model(frame, device="cpu", verbose=False)

        all_actions = []
        for r in results:
            if r.keypoints is None:
                continue
            for person in r.keypoints.xy:
                kp = person.cpu().numpy()
                actions = detect_custom_actions(kp)
                all_actions.extend(actions)

        # Remove duplicates for this frame
        all_actions = list(set(all_actions))

        # ---------------- Emotion detection with DeepFace ----------------
        dominant_emotion = None
        try:
            # enforce_detection=False -> do not crash if face is not clearly found
            emo_result = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="retinaface",  # you can also try 'opencv' if too slow
            )

            # DeepFace may return a list; handle both cases
            if isinstance(emo_result, list):
                emo_result = emo_result[0]

            dominant_emotion = emo_result.get("dominant_emotion", None)

        except Exception as e:
            # If DeepFace fails on this frame, just skip emotion
            print(f"[DeepFace warning] {e}")
            dominant_emotion = None

        # ---------------- Draw results on the frame ----------------
        # Actions on the left
        y = 30
        for act in all_actions:
            cv2.putText(
                frame,
                f"ACTION: {act}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            y += 25

        # Emotion at bottom-left
        if dominant_emotion is not None:
            cv2.putText(
                frame,
                f"Emotion: {dominant_emotion}",
                (10, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )

        # Time stamp overlay
        cv2.putText(
            frame,
            f"t = {elapsed:.1f}s",
            (10, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
        )

        cv2.imshow("Interview System v2 (Actions + Emotion)", frame)

        # ------------- Save log for this frame -------------
        combined_log.append(
            {
                "time_seconds": elapsed,
                "actions": all_actions,
                "emotion": dominant_emotion,
            }
        )

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Save JSON log
    with open("combined_log.json", "w", encoding="utf-8") as f:
        json.dump(combined_log, f, indent=4, ensure_ascii=False)

    print("Saved combined_log.json")


if __name__ == "__main__":
    main()
