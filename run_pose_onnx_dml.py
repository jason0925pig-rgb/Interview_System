import cv2
import numpy as np
import onnxruntime as ort

# -----------------------------
# 1. 使用 DirectML 加速
# -----------------------------
providers = [
    ("DmlExecutionProvider", { "device_id": 0 }),
    "CPUExecutionProvider"
]
session = ort.InferenceSession("yolo11m-pose.onnx", providers=providers)
input_name = session.get_inputs()[0].name

print("Using providers:", session.get_providers())

# -----------------------------
# 2. 打开摄像头
# -----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (640, 640))
    img_input = img[:, :, ::-1] / 255.0  # BGR → RGB
    img_input = img_input.transpose(2, 0, 1).astype(np.float32)
    img_input = np.expand_dims(img_input, axis=0)

    # -------------------------
    # 3. ONNX 推理
    # -------------------------
    outputs = session.run(None, {input_name: img_input})[0]  # shape: (1, 56, 8400)
    outputs = outputs[0]

    # -------------------------
    # 4. 解析 YOLO11 Pose 输出
    # -------------------------
    # 每个目标 56 维：
    # 0-3: bbox
    # 4: score
    # 5: class
    # 6-55: 25 keypoints (x,y,conf)
    scores = outputs[4]
    mask = scores > 0.4

    filtered = outputs[:, mask].T  # (N, 56)

    # -------------------------
    # 5. 可视化
    # -------------------------
    for det in filtered:
        x, y, w, h = det[:4].astype(int)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # Keypoints
        kpts = det[6:].reshape(25, 2)
        for (kx, ky) in kpts.astype(int):
            cv2.circle(frame, (kx, ky), 3, (0,0,255), -1)

    cv2.imshow("ONNX Pose (DirectML)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
