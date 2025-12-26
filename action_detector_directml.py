import cv2
import numpy as np
import onnxruntime as ort


# =============================
# DirectML Session
# =============================
sess = ort.InferenceSession(
    "yolo11m-pose.onnx",
    providers=["DmlExecutionProvider"]  # DirectML GPU 加速
)


# 输入名称
input_name = sess.get_inputs()[0].name


def preprocess(image):
    img = cv2.resize(image, (640, 640))
    img = img[:, :, ::-1]  # BGR → RGB
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (0, 1, 2))  
    img = np.transpose(img, (2, 0, 1))  
    img = np.expand_dims(img, 0)
    return img


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


prev_left_wrist = None
prev_right_wrist = None


# =============================
# 动作识别
# =============================
def detect_actions(kp):
    global prev_left_wrist, prev_right_wrist

    nose = kp[0]
    left_eye, right_eye = kp[1], kp[2]
    left_ear, right_ear = kp[3], kp[4]

    l_shoulder, r_shoulder = kp[5], kp[6]
    l_elbow, r_elbow = kp[7], kp[8]
    l_wrist, r_wrist = kp[9], kp[10]

    l_hip, r_hip = kp[11], kp[12]

    actions = []

    shoulder_center = ((l_shoulder[0] + r_shoulder[0]) / 2,
                       (l_shoulder[1] + r_shoulder[1]) / 2)
    hip_center = ((l_hip[0] + r_hip[0]) / 2,
                  (l_hip[1] + r_hip[1]) / 2)

    # 1 arms crossed
    if distance(l_wrist, r_elbow) < 80 and distance(r_wrist, l_elbow) < 80:
        actions.append("arms_crossed")

    # 2 hands clasped
    if distance(l_wrist, r_wrist) < 60:
        actions.append("hands_clasped")

    # 3 chin rest
    if distance(l_wrist, nose) < 70 or distance(r_wrist, nose) < 70:
        actions.append("chin_rest")

    # 4 lean forward
    torso_h = abs(shoulder_center[1] - hip_center[1])
    if torso_h < 120:
        actions.append("lean_forward")

    # 5 lean back
    if torso_h > 200:
        actions.append("lean_back")

    # 6 head down
    if nose[1] > shoulder_center[1] + 40:
        actions.append("head_down")

    # 7 touch_face
    face_center = ((left_eye[0] + right_eye[0]) / 2,
                   (left_eye[1] + right_eye[1]) / 2)
    if distance(l_wrist, face_center) < 70 or distance(r_wrist, face_center) < 70:
        actions.append("touch_face")

    # 8 touch_nose
    if distance(l_wrist, nose) < 40 or distance(r_wrist, nose) < 40:
        actions.append("touch_nose")

    # 9 fix_hair
    if distance(l_wrist, left_ear) < 60 or distance(r_wrist, right_ear) < 60:
        actions.append("fix_hair")

    # 10 fidget hands
    fidget = False
    if prev_left_wrist is not None and distance(prev_left_wrist, l_wrist) > 25:
        fidget = True
    if prev_right_wrist is not None and distance(prev_right_wrist, r_wrist) > 25:
        fidget = True

    if fidget:
        actions.append("fidget_hands")

    prev_left_wrist = l_wrist
    prev_right_wrist = r_wrist

    return list(set(actions))


# =============================
# 主循环
# =============================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    inp = preprocess(frame)

    out = sess.run(None, {input_name: inp})[0]

    # YOLO pose 输出格式需要你自己解析
    # 这里假设 out[0] = [17, 3]  (x, y, conf)
    keypoints = out[0][:, :2]

    actions = detect_actions(keypoints)

    y = 30
    for act in actions:
        cv2.putText(frame, f"ACTION: {act}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += 30

    cv2.imshow("DirectML Pose Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
