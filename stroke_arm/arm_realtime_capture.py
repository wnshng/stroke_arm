# --- 실시간 예측 + 저장 + 학습 통합 코드 (arm_full_pipeline.py with both hands) ---
import cv2
import numpy as np
import mediapipe as mp
import joblib
import pandas as pd
from datetime import datetime
from PIL import ImageFont, ImageDraw, Image
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# --- 경로 설정 ---
DATA_DIR = "data"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "mlp_arm_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_arm.joblib")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 컬럼 정의 ---
base_columns = ['angle_diff', 'depth_diff', 'height_diff', 'hand_dir']
finger_columns = [f'lh_finger_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']] + \
                  [f'rh_finger_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
all_columns = base_columns + finger_columns + ['label']

# --- CSV 파일 이름 미리 생성 ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"{DATA_DIR}/arm_feature_data_{timestamp}.csv"
pd.DataFrame(columns=all_columns).to_csv(csv_filename, index=False)

# --- Mediapipe 초기화 ---
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# --- 관절 인덱스 정의 ---
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16

data = []

# --- 특징 추출 ---
def extract_arm_features(landmarks, hand_directions):
    def to_np(idx):
        return np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
    l_upper = to_np(LEFT_ELBOW) - to_np(LEFT_SHOULDER)
    r_upper = to_np(RIGHT_ELBOW) - to_np(RIGHT_SHOULDER)
    def vector_angle(v1, v2):
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    angle_diff = vector_angle(l_upper, r_upper)
    depth_diff = abs(to_np(LEFT_WRIST)[2] - to_np(RIGHT_WRIST)[2])
    height_diff = abs(to_np(LEFT_WRIST)[1] - to_np(RIGHT_WRIST)[1])
    hand_dir = np.mean(hand_directions) if hand_directions else 0.0
    return [angle_diff, depth_diff, height_diff, hand_dir]

# --- 한글 출력 ---
def draw_text(img, text, pos=(30, 40), size=26, color=(255, 255, 255)):
    try:
        font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
        font = ImageFont.truetype(font_path, size)
    except OSError:
        font = ImageFont.load_default()
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- 모델 로드 (있다면) ---
model, scaler = None, None
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

# --- 웹캠 실행 ---
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_result = pose.process(rgb)
    hands_result = hands.process(rgb)

    msg, color = "Y: 정상 저장 / N: 비정상 저장", (255, 255, 255)
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
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks.landmark
        for idx in [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]:
            x = int(landmarks[idx].x * frame.shape[1])
            y = int(landmarks[idx].y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
        features = extract_arm_features(landmarks, hand_dirs)
        msg = f"입력 저장 대기 | 각도차: {features[0]:.1f}, 깊이차: {features[1]:.2f}, 높이차: {features[2]:.2f}"

        # 실시간 예측
        if model and scaler:
            input_features = features + lh_finger_coords + rh_finger_coords
            try:
                X_input = pd.DataFrame([input_features], columns=scaler.feature_names_in_)
                X_scaled = scaler.transform(X_input)
                pred = model.predict(X_scaled)[0]
                if pred == 0:
                    msg += " | ✅ 예측: 정상"
                    color = (0, 255, 0)
                else:
                    msg += " | ❌ 예측: 비대칭"
                    color = (0, 0, 255)
            except Exception as e:
                msg = f"에러: {e}"
                color = (0, 0, 255)
    else:
        features = None
        msg = "팔 인식 실패"
        color = (120, 120, 120)

    frame = draw_text(frame, msg, color=color)
    cv2.imshow("팔 + 양손 비대칭 측정 및 저장", frame)

    key = cv2.waitKey(5)
    if key == 27:
        break
    elif key == ord('y') and features:
        row = features + lh_finger_coords + rh_finger_coords + [0]
        pd.DataFrame([row], columns=all_columns).to_csv(csv_filename, mode='a', header=False, index=False)
        print("✅ 정상 샘플 저장")
    elif key == ord('n') and features:
        row = features + lh_finger_coords + rh_finger_coords + [1]
        pd.DataFrame([row], columns=all_columns).to_csv(csv_filename, mode='a', header=False, index=False)
        print("❌ 비정상 샘플 저장")

cap.release()
cv2.destroyAllWindows()

# --- 모델 학습 ---
print("\n📊 모델 학습 시작...")
files = sorted(glob.glob(f"{DATA_DIR}/arm_feature_data_*.csv"))
df_list = [pd.read_csv(f) for f in files if 'label' in pd.read_csv(f).columns]
df = pd.concat(df_list, ignore_index=True)
X = df.drop("label", axis=1)
y = df["label"].astype(int)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42, early_stopping=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"✅ 모델 저장 완료: {MODEL_PATH}, {SCALER_PATH}")