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
        if video_path.find('doubleclick'):  # 동작 라벨링
            self.action = 'doubleclick'
            self.action_label = 2
        else:
            self.action = 'click'
            self.action_label = 1

    def record_index(self, action_count, nothing_count):
        # 마지막 index 기록
        f_w = open(f'dataset/{self.action}/last_idx.txt', 'w')
        f_w.write(str(action_count))
        f_w.close()
        f_w = open(f'dataset/nothing/last_idx.txt', 'w')
        f_w.write(str(nothing_count))
        f_w.close()

    def split_video(self):
        cv2.destroyAllWindows()
        cap = cv2.VideoCapture(self.video_path)  # opencv로 녹화 영상 틀기
        os.makedirs(f'dataset/{self.action}', exist_ok=True)
        os.makedirs('dataset/nothing', exist_ok=True)

        action_count = 0
        if os.path.isfile(f'dataset/{self.action}/last_idx.txt'):
            f_r = open(f'dataset/{self.action}/last_idx.txt', 'r')
            action_count = int(f_r.readline())
            f_r.close()

        nothing_count = 0
        if os.path.isfile(f'dataset/nothing/last_idx.txt'):
            f_r = open(f'dataset/nothing/last_idx.txt', 'r')
            nothing_count = int(f_r.readline())
            f_r.close()

        if cap.isOpened():
            cnt = 0  # 0보다 클 때 데이터 수집중
            current_dist = 0  # 최근 엄지와 검지 사이의 거리
            doubleclick_det = 0  # doubleclick의 경우 검지와 엄지 사이의 거리가 두번 왔다 갔다함

            ret, img = cap.read()
            cv2.imshow('webcam', img)

            data = []  # 기본 데이터 배열
            action_data = []  # 제스처 데이터 배열

            gesture_cnt = 0
            while True:
                if cnt == 50:  # 제스처 데이터 수집 완료
                    gesture_cnt += 1
                    cnt = 0
                    np.save(os.path.join(f'dataset/{self.action}', f'seq_{action_count}'), action_data)
                    action_data = []
                    doubleclick_det = 0
                    if gesture_cnt == 8:
                        self.record_index(action_count, nothing_count)
                        break
                elif cnt > 0:  # 제스처 데이터 수집중
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

                        d = np.append(joint.flatten(), 0)
                        if cnt > 0:
                            action_data = np.append(action_data, d.reshape(1, 64), axis=0)
                        else:
                            data.append(d)

                        dist = np.linalg.norm(joint[8] - joint[4])
                        if current_dist - dist > 0.03 and doubleclick_det == 0:
                            action_count += 1
                            doubleclick_det = 1
                            print(f'{self.action}_{action_count}')
                            data = np.array(data)

                            # 동작의 sequence를 -5프레임부터 시작
                            action_data = data[-5:, :-1]
                            labels = np.reshape([self.action_label] * 5, (5, 1))
                            action_data = np.concatenate([action_data, labels], axis=1)  # 라벨 붙이기
                            action_data = np.reshape(action_data, (5, 64))

                            # nothing data 분류
                            if data.size >= 50 * 64:
                                nothing_count += 1
                                nothing_data = data[:50, :]
                                np.save(os.path.join('dataset/nothing', f'seq_{nothing_count}'), nothing_data)
                            data = []
                            cnt = 5

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
