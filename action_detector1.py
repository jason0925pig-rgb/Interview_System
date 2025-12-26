import cv2
import numpy as np
from ultralytics import YOLO


# 加载 YOLO pose
model = YOLO("yolo11m-pose.pt")


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


# 用于手部抖动检测（fidget）
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

    # 中心点计算
    shoulder_center = ( (l_shoulder[0] + r_shoulder[0]) / 2,
                        (l_shoulder[1] + r_shoulder[1]) / 2 )
    hip_center = ( (l_hip[0] + r_hip[0]) / 2,
                   (l_hip[1] + r_hip[1]) / 2 )


    # ────────────────────────────────
    # 1. arms_crossed (双手抱胸)
    # ────────────────────────────────
    if (
        distance(l_wrist, r_elbow) < 80 and
        distance(r_wrist, l_elbow) < 80
    ):
        actions.append("arms_crossed")


    # ────────────────────────────────
    # 2. hands_clasped (双手扣在一起)
    # ────────────────────────────────
    if distance(l_wrist, r_wrist) < 60:
        actions.append("hands_clasped")


    # ────────────────────────────────
    # 3. chin_rest (托下巴)
    # ────────────────────────────────
    if distance(l_wrist, nose) < 70 or distance(r_wrist, nose) < 70:
        actions.append("chin_rest")


    # ────────────────────────────────
    # 4. lean_forward (身体前倾)
    # ────────────────────────────────
    torso_height = abs(shoulder_center[1] - hip_center[1])
    if torso_height < 120:
        actions.append("lean_forward")


    # ────────────────────────────────
    # 5. lean_back (身体后仰)
    # ────────────────────────────────
    if torso_height > 200:
        actions.append("lean_back")


    # ────────────────────────────────
    # 6. head_down (低头)
    # ────────────────────────────────
    if nose[1] > shoulder_center[1] + 40:
        actions.append("head_down")


    # ────────────────────────────────
    # 7. touch_face (手碰脸)
    # ────────────────────────────────
    face_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    if distance(l_wrist, face_center) < 70 or distance(r_wrist, face_center) < 70:
        actions.append("touch_face")


    # ────────────────────────────────
    # 8. touch_nose（摸鼻子）
    # ────────────────────────────────
    if distance(l_wrist, nose) < 40 or distance(r_wrist, nose) < 40:
        actions.append("touch_nose")


    # ────────────────────────────────
    # 9. fix_hair（整理头发）
    # 条件：手靠近耳朵
    # ────────────────────────────────
    if (
        distance(l_wrist, left_ear) < 60 or distance(r_wrist, right_ear) < 60 or
        distance(l_wrist, right_ear) < 60 or distance(r_wrist, left_ear) < 60
    ):
        actions.append("fix_hair")


    # ────────────────────────────────
    # 10. fidget_hands（手的小动作）
    # 条件：手连续快速移动
    # ────────────────────────────────
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


    return list(set(actions))  # 去重


# ====================================================================
# 主程序循环
# ====================================================================

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device="cpu", verbose=False)

    for r in results:
        if r.keypoints is None:
            continue

        for person in r.keypoints.xy:
            kp = person.cpu().numpy()

            actions = detect_custom_actions(kp)

            # 显示检测结果
            y = 30
            for act in actions:
                cv2.putText(frame, f"ACTION: {act}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                y += 30

    cv2.imshow("Interview Action Detector - 10 Actions", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
