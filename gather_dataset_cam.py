import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
fps = 30


class WebCamData:
    def __init__(self, video_path):
        self.video_path = video_path

    def split_video(self):
        cv2.destroyAllWindows()
        cap = cv2.VideoCapture(self.video_path)
        if cap.isOpened():
            cnt = 0
            current_dist = 0
            gesture_cnt = 0
            data = []

            ret, img = cap.read()
            cv2.imshow('webcam', img)

            while True:
                cnt += 1
                ret, img = cap.read()
                if not ret:
                    break

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if result.multi_hand_landmarks is not None:
                    for res in result.multi_hand_landmarks:
                        joint = np.zeros((21, 3))
                        for j, lm in enumerate(res.landmark):
                            joint[j] = [lm.x, lm.y, lm.z]

                        dist = np.linalg.norm(joint[8] - joint[4])
                        if current_dist == 0:
                            print('start')
                        elif current_dist > 0.1 >= dist:
                            gesture_cnt += 1
                            print(f'down{gesture_cnt}, {cnt}')
                        elif current_dist <= 0.1 < dist:
                            print(f'up{gesture_cnt}, {cnt}')
                        current_dist = dist
                        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                cv2.imshow('webcam', img)
                if cv2.waitKey(1) == ord('q'):
                    break

    def show_skeleton(self):
        cv2.destroyAllWindows()
        cap = cv2.VideoCapture(self.video_path)
        if cap.isOpened():
            cnt = 0
            current_dist = 0
            gesture_cnt = 0
            data = []

            ret, img = cap.read()
            cv2.imshow('webcam', img)

            while True:
                cnt += 1
                ret, img = cap.read()
                if not ret:
                    break

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if result.multi_hand_landmarks is not None:
                    for res in result.multi_hand_landmarks:
                        joint = np.zeros((21, 3))
                        for j, lm in enumerate(res.landmark):
                            joint[j] = [lm.x, lm.y, lm.z]

                        dist = np.linalg.norm(joint[8] - joint[4])
                        if current_dist == 0:
                            print('start')
                        elif current_dist > 0.1 >= dist:
                            gesture_cnt += 1
                            print(f'down{gesture_cnt}, {cnt}')
                        elif current_dist <= 0.1 < dist:
                            print(f'up{gesture_cnt}, {cnt}')
                        current_dist = dist
                        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                cv2.imshow('webcam', img)
                if cv2.waitKey(1) == ord('q'):
                    break
