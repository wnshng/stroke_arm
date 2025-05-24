# 실시간 MLP 기반 양팔 대칭 측정 프로그램 (최종 버전)
import cv2
import numpy as np
import pandas as pd
import joblib
import time
from collections import Counter
import mediapipe as mp
import pyttsx3
from PIL import ImageFont, ImageDraw, Image

# 모델 및 스케일러 로딩
model = joblib.load("model/mlp_arm_model.joblib")
scaler = joblib.load("model/scaler_arm.joblib")

# Mediapipe 초기화
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# TTS 초기화
tts = pyttsx3.init()
def speak(text):
    tts.say(text)
    tts.runAndWait()

# 이미지에 텍스트 출력
def draw_text(img, text, pos=(30, 40), size=26, color=(255, 255, 255)):
    try:
        font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
        font = ImageFont.truetype(font_path, size)
    except:
        font = ImageFont.load_default()
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 특징 추출 함수
def extract_arm_features(landmarks, hand_directions):
    def to_np(idx):
        return np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
    l_upper = to_np(13) - to_np(11)
    r_upper = to_np(14) - to_np(12)
    def vector_angle(v1, v2):
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    angle_diff = vector_angle(l_upper, r_upper)
    depth_diff = abs(to_np(15)[2] - to_np(16)[2])
    height_diff = abs(to_np(15)[1] - to_np(16)[1])
    hand_dir = np.mean(hand_directions) if hand_directions else 0.0
    return [angle_diff, depth_diff, height_diff, hand_dir]

# 1초간 정상 유지 판단을 최대 10초 동안 2회 시도
def wait_for_stable_pose_with_retry(cap, pose, hands, model, scaler, max_attempts=2, max_duration=10.0, interval=0.1, require_normal_count=10):
    for attempt in range(max_attempts):
        speak("양팔을 들어주세요")
        print(f"\n🕑 시도 {attempt + 1}: 팔을 들어주세요")
        stable_count = 0
        start_time = time.perf_counter()

        while time.perf_counter() - start_time < max_duration:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = pose.process(rgb)
            hands_result = hands.process(rgb)

            hand_dirs = []
            lh_finger_coords = [0.0] * 63
            rh_finger_coords = [0.0] * 63

            if hands_result.multi_hand_landmarks and hands_result.multi_handedness:
                for idx, hand_landmarks in enumerate(hands_result.multi_hand_landmarks):
                    handedness = hands_result.multi_handedness[idx].classification[0].label
                    wrist = hand_landmarks.landmark[0]
                    index_tip = hand_landmarks.landmark[8]
                    dx = index_tip.x - wrist.x
                    dy = index_tip.y - wrist.y
                    direction = np.arctan2(dy, dx)
                    hand_dirs.append(direction)
                    coords = [v for lm in hand_landmarks.landmark for v in (lm.x, lm.y, lm.z)]
                    if handedness == 'Left':
                        lh_finger_coords = coords
                    else:
                        rh_finger_coords = coords

            if pose_result.pose_landmarks:
                try:
                    landmarks = pose_result.pose_landmarks.landmark
                    features = extract_arm_features(landmarks, hand_dirs)
                    input_features = features + lh_finger_coords + rh_finger_coords
                    X_input = pd.DataFrame([input_features], columns=scaler.feature_names_in_)
                    X_scaled = scaler.transform(X_input)
                    pred = model.predict(X_scaled)[0]
                    if pred == 0:
                        stable_count += 1
                        if stable_count >= require_normal_count:
                            speak("팔이 안정적으로 인식되었습니다. 측정을 시작합니다.")
                            return True
                    else:
                        stable_count = 0
                except:
                    stable_count = 0

            frame = draw_text(frame, f"정상 유지 중: {stable_count}/10", color=(0, 255, 0))
            cv2.imshow("실시간 예측", frame)
            if cv2.waitKey(1) == 27:
                return False
            time.sleep(interval)

        if attempt < max_attempts - 1:
            speak("팔이 안정적으로 인식되지 않았습니다. 다시 팔을 들어주세요.")

    speak("측정에 실패하였습니다. 프로그램을 종료합니다.")
    return False

# 5초간 50프레임 측정

def perform_measurement(cap, pose, hands, model, scaler, duration=5.0, interval=0.1):
    results = []
    for _ in range(50):  # 50회로 제한
        iter_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = pose.process(rgb)
        hands_result = hands.process(rgb)

        hand_dirs = []
        lh_finger_coords = [0.0] * 63
        rh_finger_coords = [0.0] * 63

        if hands_result.multi_hand_landmarks and hands_result.multi_handedness:
            for idx, hand_landmarks in enumerate(hands_result.multi_hand_landmarks):
                handedness = hands_result.multi_handedness[idx].classification[0].label
                wrist = hand_landmarks.landmark[0]
                index_tip = hand_landmarks.landmark[8]
                dx = index_tip.x - wrist.x
                dy = index_tip.y - wrist.y
                direction = np.arctan2(dy, dx)
                hand_dirs.append(direction)
                coords = [v for lm in hand_landmarks.landmark for v in (lm.x, lm.y, lm.z)]
                if handedness == 'Left':
                    lh_finger_coords = coords
                else:
                    rh_finger_coords = coords

        if pose_result.pose_landmarks:
            try:
                landmarks = pose_result.pose_landmarks.landmark
                features = extract_arm_features(landmarks, hand_dirs)
                input_features = features + lh_finger_coords + rh_finger_coords
                X_input = pd.DataFrame([input_features], columns=scaler.feature_names_in_)
                X_scaled = scaler.transform(X_input)
                pred = model.predict(X_scaled)[0]
                results.append(pred)
                frame = draw_text(frame, f"측정 중... {len(results)}/50", color=(0, 255, 255))
            except:
                pass

        cv2.imshow("실시간 예측", frame)
        if cv2.waitKey(1) == 27:
            break
        elapsed = time.perf_counter() - iter_start
        time.sleep(max(0.0, interval - elapsed))
    return results

# 측정 평가
def evaluate_and_report(results, round_num=1):
    counter = Counter(results)
    normal_count = counter[0]
    total = len(results)
    rate = (normal_count / total) * 100
    print(f"\n📊 [{round_num}차 측정] 팔 대칭률: {rate:.2f}% (정상 {normal_count} / 총 {total})")
    speak(f"팔 대칭률은 {int(rate)} 퍼센트입니다.")
    return rate

# 메인 루프
def main():
    cap = cv2.VideoCapture(0)
    if wait_for_stable_pose_with_retry(cap, pose, hands, model, scaler):
        results1 = perform_measurement(cap, pose, hands, model, scaler)
        rate1 = evaluate_and_report(results1)
        if rate1 >= 80:
            speak("팔이 정상적으로 대칭입니다. 프로그램을 종료합니다.")
            cap.release()
            cv2.destroyAllWindows()
            return
        else:
            speak("팔이 비대칭이 의심됩니다. 다시 한번 측정을 시작합니다.")
            if wait_for_stable_pose_with_retry(cap, pose, hands, model, scaler):
                results2 = perform_measurement(cap, pose, hands, model, scaler)
                rate2 = evaluate_and_report(results2, round_num=2)
                if rate2 >= 80:
                    speak("두 번째 측정 결과, 팔이 정상적으로 대칭입니다. 종료합니다.")
                else:
                    speak("팔이 비대칭이 의심되므로 가까운 병원에 내원 부탁드립니다.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
