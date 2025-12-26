import cv2
import numpy as np
import time
import json
import threading
import sounddevice as sd
from faster_whisper import WhisperModel
from ultralytics import YOLO
from deepface import DeepFace

# ==============================
# Pose / action detection part
# ==============================

model = YOLO("yolo11m-pose.pt")  # make sure this file is in the same folder


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


# Previous wrist positions for fidget detection
prev_left_wrist = None
prev_right_wrist = None

# Logs
action_logs = []       # list of {time, timestamp_seconds, actions}
speech_logs = []       # list of {time, timestamp_seconds, text}
emotion_logs = []      # list of {time, timestamp_seconds, emotion}

# Shared state
start_time = None      # will be set when first frame is read
stop_flag = False      # used to stop STT thread
current_subtitle = ""  # latest recognized text for on-screen display
current_emotion = None # latest detected emotion for on-screen display


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
    face_center = (
        (left_eye[0] + right_eye[0]) / 2,
        (left_eye[1] + right_eye[1]) / 2
    )
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


# ==============================
# Speech-to-text (STT) thread
# ==============================

def stt_worker():
    """
    Record short chunks from microphone, transcribe with faster-whisper,
    and append results into speech_logs.
    """
    global start_time, stop_flag, speech_logs, current_subtitle

    SAMPLE_RATE = 16000
    CHUNK_SECONDS = 4  # each chunk length

    # Load Whisper tiny model on CPU
    print("[STT] Loading Whisper tiny model on CPU...")
    model_stt = WhisperModel("tiny", device="cpu")

    # Wait until video has started and start_time is set
    while start_time is None and not stop_flag:
        time.sleep(0.1)

    print("[STT] Start listening...")

    while not stop_flag:
        # Record a short chunk of audio
        duration = CHUNK_SECONDS
        audio = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )
        sd.wait()

        if stop_flag:
            break

        # Flatten to 1D
        audio_mono = audio.flatten()

        # Transcribe
        segments, _ = model_stt.transcribe(
            audio_mono,
            beam_size=1,
            language="en"  # or "zh" / None for auto-detect
        )

        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue

            # Use current global time as approximate timestamp
            ts = time.time() - start_time
            mm = int(ts // 60)
            ss = int(ts % 60)

            current_subtitle = text  # for on-screen display
            print(f"[STT {mm:02d}:{ss:02d}] {text}")

            speech_logs.append({
                "time": f"{mm:02d}:{ss:02d}",
                "timestamp_seconds": round(ts, 2),
                "text": text
            })

    print("[STT] Stopped.")


# ==============================
# Main video + pose + emotion
# ==============================

def main():
    global start_time, stop_flag, action_logs, current_subtitle, current_emotion, emotion_logs

    # Start STT thread
    stt_thread = threading.Thread(target=stt_worker, daemon=True)
    stt_thread.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open camera.")
        stop_flag = True
        return

    print("[MAIN] Press 'q' to quit and save logs.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: cannot read frame.")
            break

        # Set start_time when we get the first frame
        if start_time is None:
            start_time = time.time()
            print("[MAIN] Timing started from first frame.")

        # Current elapsed time
        ts = time.time() - start_time
        mm = int(ts // 60)
        ss = int(ts % 60)

        # Pose inference on CPU
        results = model(frame, device="cpu", verbose=False)

        frame_actions = []

        for r in results:
            if r.keypoints is None:
                continue

            for person in r.keypoints.xy:
                kp = person.cpu().numpy()
                actions = detect_custom_actions(kp)
                frame_actions.extend(actions)

        # Log actions for this moment
        if frame_actions:
            action_logs.append({
                "time": f"{mm:02d}:{ss:02d}",
                "timestamp_seconds": round(ts, 2),
                "actions": list(set(frame_actions))
            })

        # ------------- Emotion detection with DeepFace -------------
        dominant_emotion = None
        try:
            # enforce_detection=False: do not crash if no face
            emo_result = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="retinaface",  # if too slow, you can change to "opencv"
            )

            # DeepFace may return a list -> take first
            if isinstance(emo_result, list):
                emo_result = emo_result[0]

            dominant_emotion = emo_result.get("dominant_emotion", None)

        except Exception as e:
            # You can print this if you want to debug DeepFace
            # print("[DeepFace warning]", e)
            dominant_emotion = None

        current_emotion = dominant_emotion

        if dominant_emotion is not None:
            emotion_logs.append({
                "time": f"{mm:02d}:{ss:02d}",
                "timestamp_seconds": round(ts, 2),
                "emotion": dominant_emotion
            })

        # ----------------------- Draw overlays -----------------------
        # Draw actions on frame
        y = 30
        for act in set(frame_actions):
            cv2.putText(frame, f"ACTION: {act}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y += 30

        # Draw current emotion
        if current_emotion:
            cv2.putText(frame, f"Emotion: {current_emotion}", (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Draw current subtitle (speech) at bottom of screen
        if current_subtitle:
            h, w, _ = frame.shape
            cv2.putText(frame, current_subtitle, (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw time
        cv2.putText(frame, f"t = {ts:.1f}s", (10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Show frame
        cv2.imshow("Interview Monitor (Action + Emotion + Speech)", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Stop STT thread
    stop_flag = True
    time.sleep(0.5)

    # Save separate logs
    with open("action_log.json", "w", encoding="utf-8") as f:
        json.dump(action_logs, f, ensure_ascii=False, indent=2)

    with open("transcription_log.json", "w", encoding="utf-8") as f:
        json.dump(speech_logs, f, ensure_ascii=False, indent=2)

    with open("emotion_log.json", "w", encoding="utf-8") as f:
        json.dump(emotion_logs, f, ensure_ascii=False, indent=2)

    # ---------------- Build combined log per second ----------------
    combined = {}
    # Merge actions
    for entry in action_logs:
        sec = int(entry["timestamp_seconds"])
        if sec not in combined:
            combined[sec] = {
                "time": entry["time"],
                "timestamp_seconds": float(sec),
                "actions": [],
                "texts": [],
                "emotions": []
            }
        combined[sec]["actions"].extend(entry["actions"])

    # Merge speech
    for entry in speech_logs:
        sec = int(entry["timestamp_seconds"])
        if sec not in combined:
            combined[sec] = {
                "time": entry["time"],
                "timestamp_seconds": float(sec),
                "actions": [],
                "texts": [],
                "emotions": []
            }
        combined[sec]["texts"].append(entry["text"])

    # Merge emotions
    for entry in emotion_logs:
        sec = int(entry["timestamp_seconds"])
        if sec not in combined:
            combined[sec] = {
                "time": entry["time"],
                "timestamp_seconds": float(sec),
                "actions": [],
                "texts": [],
                "emotions": []
            }
        combined[sec]["emotions"].append(entry["emotion"])

    # Deduplicate actions / emotions
    combined_list = []
    for sec in sorted(combined.keys()):
        item = combined[sec]
        item["actions"] = sorted(list(set(item["actions"])))
        item["emotions"] = sorted(list(set(item["emotions"])))
        combined_list.append(item)

    with open("combined_log.json", "w", encoding="utf-8") as f:
        json.dump(combined_list, f, ensure_ascii=False, indent=2)

    print("Saved action_log.json, transcription_log.json, emotion_log.json, combined_log.json")


if __name__ == "__main__":
    main()
