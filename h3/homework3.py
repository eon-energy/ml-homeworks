import cv2
import mediapipe as mp
import os
from deepface import DeepFace

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

your_image_path = "me.jpg"
print(os.path.exists(your_image_path))

def count_fingers(landmarks):
    finger_tips = [4, 8, 12, 16, 20]
    count = 0

    if landmarks[4].x < landmarks[3].x:
        count += 1

    for tip in finger_tips[1:]:
        if landmarks[tip].y < landmarks[tip - 2].y:
            count += 1

    return count

# Инициализация камеры
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Конвертация изображения в RGB (для медиапайпа)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb_frame)
    fingers_count = 0

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers_count = count_fingers(hand_landmarks.landmark)

    cv2.putText(frame, f"Fingers: {fingers_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Поиск лиц на кадре
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        try:
            emotion = DeepFace.analyze(frame[y:y+h, x:x+w], actions=['emotion'], enforce_detection=False)
            dominant_emotion = emotion[0]['dominant_emotion']
        except Exception:
            dominant_emotion = "Неизвестная эмоция"

        try:
            result = DeepFace.verify(
                img1_path=your_image_path,
                img2_path=frame[y:y+h, x:x+w],
                model_name="Facenet",
                distance_metric="cosine"
            )
            is_owner = result["verified"]
        except Exception as e:
            is_owner = False

        name = "unknown"
        print("is_owner =", is_owner)

        if is_owner:
            print("fingers_count =", fingers_count)

            if fingers_count == 1:
                print("зашло в условие 1")
                name = "ION"
            elif fingers_count == 2:
                print("зашло в условие 2")
                name = "ЦИЦКИШВИЛИ"
            elif fingers_count == 3:
                print("зашло в условие 3")
                name = dominant_emotion  # Показать эмоцию
            else:
                name = ""
        else:
            name = "unknown"

        if name:
            cv2.putText(frame, name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face and Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
