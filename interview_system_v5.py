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
# Configuration
# ==============================

POSE_MODEL = "yolo11m-pose.pt"

# Setting the audio
SAMPLE_RATE = 16000          # 16 kHz
VOICE_CHUNK_SECONDS = 4      # record every 4 second

# Shared globals
start_time = None
stop_flag = False

prev_left_wrist = None
prev_right_wrist = None

action_logs = []
speech_logs = []
voice_emotion_logs = []
facial_emotion_logs = []  # New log for facial emotions

current_subtitle = ""
current_voice_emotion = ""
current_facial_emotion = ""


# ================================================================
# Utility
# ================================================================
def distance(p1, p2):
    # Calculate Euclidean distance between two points
    return np.linalg.norm(np.array(p1) - np.array(p2))


# ================================================================
# Pose / Action Detection
# ================================================================
model_pose = YOLO(POSE_MODEL)


def detect_custom_actions(kp):
    # Detect custom actions based on keypoints
    global prev_left_wrist, prev_right_wrist

    nose = kp[0]
    left_eye, right_eye = kp[1], kp[2]
    left_ear, right_ear = kp[3], kp[4]

    l_shoulder, r_shoulder = kp[5], kp[6]
    l_elbow, r_elbow = kp[7], kp[8]
    l_wrist, r_wrist = kp[9], kp[10]

    l_hip, r_hip = kp[11], kp[12]

    actions = []

    shoulder_center = (
        (l_shoulder[0] + r_shoulder[0]) / 2,
        (l_shoulder[1] + r_shoulder[1]) / 2
    )
    hip_center = (
        (l_hip[0] + r_hip[0]) / 2,
        (l_hip[1] + r_hip[1]) / 2
    )

    # 1. Arm Crossed
    if distance(l_wrist, r_elbow) < 80 and distance(r_wrist, l_elbow) < 80:
        actions.append("arms_crossed")

    # 2. Hands Clasped
    if distance(l_wrist, r_wrist) < 60:
        actions.append("hands_clasped")

    # 3. Chin Rest
    if distance(l_wrist, nose) < 70 or distance(r_wrist, nose) < 70:
        actions.append("chin_rest")

    # 4. Lean forward or backward
    torso_height = abs(shoulder_center[1] - hip_center[1])
    if torso_height < 120:
        actions.append("lean_forward")
    if torso_height > 200:
        actions.append("lean_back")

    # 5. Head Down
    if nose[1] > shoulder_center[1] + 40:
        actions.append("head_down")

    # 6. Touch Face
    face_center = (
        (left_eye[0] + right_eye[0]) / 2,
        (left_eye[1] + right_eye[1]) / 2
    )
    if distance(l_wrist, face_center) < 70 or distance(r_wrist, face_center) < 70:
        actions.append("touch_face")

    # 7. Touch Nose
    if distance(l_wrist, nose) < 40 or distance(r_wrist, nose) < 40:
        actions.append("touch_nose")

    # 8. Fix Hair
    if (
        distance(l_wrist, left_ear) < 60 or
        distance(r_wrist, right_ear) < 60 or
        distance(l_wrist, right_ear) < 60 or
        distance(r_wrist, left_ear) < 60
    ):
        actions.append("fix_hair")

    # 9. Fidget Hand
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

    return list(set(actions))


# ================================================================
# Facial Emotion Detection (DeepFace)
# ================================================================
def detect_facial_emotion(frame):
    """
    Using DeepFace recognize your emotion in real time, return dominant emotion
    """
    global current_facial_emotion

    try:
        res = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )
        emo = res[0]["dominant_emotion"]
        current_facial_emotion = emo
        return emo
    except Exception as e:
        # Ignore the failure test
        # print("[FACE] Error:", e)
        return None


# ================================================================
# Voice Emotion Detection (simple energy-based)
# ================================================================
def detect_voice_emotion_simple(audio_chunk):
    """
    An easy but accurate rule to recognize your voice emotion:
    - Bigger Energy → agitated
    - Normal Energy → neutral
    - Small Energy → calm
    """
    global current_voice_emotion

    # audio_chunk is 1D float32/64, from -1 to 1
    energy = float(np.mean(audio_chunk ** 2))

    if energy > 0.03:
        emotion = "agitated"
    elif energy > 0.005:
        emotion = "neutral"
    else:
        emotion = "calm"

    current_voice_emotion = emotion
    return emotion


# ================================================================
# Speech-to-Text (Whisper)
# ================================================================
def stt_worker():
    global stop_flag, start_time, speech_logs, current_subtitle

    print("[STT] Loading Whisper tiny on CPU...")
    model_stt = WhisperModel("tiny", device="cpu")

    # Wait for the video start_time
    while start_time is None and not stop_flag:
        time.sleep(0.1)

    print("[STT] STT thread active.")

    while not stop_flag:
        # Record your audio
        duration = VOICE_CHUNK_SECONDS
        audio = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )
        sd.wait()

        if stop_flag:
            break

        audio_mono = audio.flatten()

        # ========== 1) STT ==========
        try:
            segments, _ = model_stt.transcribe(
                audio_mono,
                beam_size=1,
                language="en"  # Here can change the language to Chinese by "zh" or ANY LANGUAGE by "None"
            )
        except Exception as e:
            print("[STT] Transcribe error:", e)
            continue

        text_full = ""
        for seg in segments:
            text_full += seg.text.strip() + " "

        if text_full.strip():
            ts = time.time() - start_time
            mm = int(ts // 60)
            ss = int(ts % 60)

            speech_logs.append({
                "time": f"{mm:02d}:{ss:02d}",
                "timestamp_seconds": round(ts, 2),
                "text": text_full.strip()
            })

            current_subtitle = text_full.strip()
            print(f"[STT {mm:02d}:{ss:02d}] {text_full.strip()}")

        # ========== 2) Voice emotion ==========
        emotion = detect_voice_emotion_simple(audio_mono)
        ts = time.time() - start_time
        mm = int(ts // 60)
        ss = int(ts % 60)

        voice_emotion_logs.append({
            "time": f"{mm:02d}:{ss:02d}",
            "timestamp_seconds": round(ts, 2),
            "voice_emotion": emotion
        })

    print("[STT] Thread stopped.")


# ================================================================
# Main
# ================================================================
def main():
    global start_time, stop_flag

    # Start STT 
    threading.Thread(target=stt_worker, daemon=True).start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot access camera.")
        stop_flag = True
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if start_time is None:
            start_time = time.time()
            print("[SYSTEM] Timer started from first frame.")

        ts = time.time() - start_time
        mm = int(ts // 60)
        ss = int(ts % 60)

        # 1) Recognize Action
        results = model_pose(frame, device="cpu", verbose=False)
        frame_actions = []

        for r in results:
            if r.keypoints is None:
                continue
            for person in r.keypoints.xy:
                kp = person.cpu().numpy()
                actions = detect_custom_actions(kp)
                frame_actions.extend(actions)

        if frame_actions:
            action_logs.append({
                "time": f"{mm:02d}:{ss:02d}",
                "timestamp_seconds": round(ts, 2),
                "actions": list(set(frame_actions))
            })

        # 2) Recognize Face
        facial_emo = detect_facial_emotion(frame)
        if facial_emo:
            facial_emotion_logs.append({
                "time": f"{mm:02d}:{ss:02d}",
                "timestamp_seconds": round(ts, 2),
                "facial_emotion": facial_emo
            })

        # ========== Show in the screen ==========
        y = 30
        # Action
        for act in set(frame_actions):
            cv2.putText(frame, f"ACTION: {act}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y += 30

        # Emotion
        if facial_emo:
            cv2.putText(frame, f"FACE: {facial_emo}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y += 30

        # Voice Emotion
        if current_voice_emotion:
            cv2.putText(frame, f"VOICE: {current_voice_emotion}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2)
            y += 30

        # Text（STT）
        if current_subtitle:
            h, w, _ = frame.shape
            cv2.putText(frame, current_subtitle, (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Interview V4 (Pose + Face + STT + Voice Emotion)", frame)

        # q to stop the programme
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_flag = True
    time.sleep(0.3)

    cap.release()
    cv2.destroyAllWindows()

    # Save the individual logs to json files
    with open("action_log.json", "w", encoding="utf-8") as f:
        json.dump(action_logs, f, ensure_ascii=False, indent=2)

    with open("transcription_log.json", "w", encoding="utf-8") as f:
        json.dump(speech_logs, f, ensure_ascii=False, indent=2)

    with open("voice_emotion_log.json", "w", encoding="utf-8") as f:
        json.dump(voice_emotion_logs, f, ensure_ascii=False, indent=2)

    with open("facial_emotion_log.json", "w", encoding="utf-8") as f:
        json.dump(facial_emotion_logs, f, ensure_ascii=False, indent=2)

    # Build combined log per second
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
                "facial_emotions": [],
                "voice_emotions": []
            }
        combined[sec]["actions"].extend(entry["actions"])
    # Merge transcriptions (texts)
    for entry in speech_logs:
        sec = int(entry["timestamp_seconds"])
        if sec not in combined:
            combined[sec] = {
                "time": entry["time"],
                "timestamp_seconds": float(sec),
                "actions": [],
                "texts": [],
                "facial_emotions": [],
                "voice_emotions": []
            }
        combined[sec]["texts"].append(entry["text"])
    # Merge facial emotions
    for entry in facial_emotion_logs:
        sec = int(entry["timestamp_seconds"])
        if sec not in combined:
            combined[sec] = {
                "time": entry["time"],
                "timestamp_seconds": float(sec),
                "actions": [],
                "texts": [],
                "facial_emotions": [],
                "voice_emotions": []
            }
        combined[sec]["facial_emotions"].append(entry["facial_emotion"])
    # Merge voice emotions
    for entry in voice_emotion_logs:
        sec = int(entry["timestamp_seconds"])
        if sec not in combined:
            combined[sec] = {
                "time": entry["time"],
                "timestamp_seconds": float(sec),
                "actions": [],
                "texts": [],
                "facial_emotions": [],
                "voice_emotions": []
            }
        combined[sec]["voice_emotions"].append(entry["voice_emotion"])
    # Deduplicate actions / emotions
    combined_list = []
    for sec in sorted(combined.keys()):
        item = combined[sec]
        item["actions"] = sorted(list(set(item["actions"])))
        item["facial_emotions"] = sorted(list(set(item["facial_emotions"])))
        item["voice_emotions"] = sorted(list(set(item["voice_emotions"])))
        combined_list.append(item)
    with open("combined_log.json", "w", encoding="utf-8") as f:
        json.dump(combined_list, f, ensure_ascii=False, indent=2)

    print("Saved action_log.json, transcription_log.json, voice_emotion_log.json, facial_emotion_log.json, combined_log.json")


if __name__ == "__main__":
    main()