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
        if video_path.find('doubleclick') != -1:  # 동작 라벨링
            self.action = 'doubleclick'
            self.action_label = 2
        # elif video_path.find('rclick') != -1:
        #     self.action = 'rclick'
        #     self.action_label = 6
        elif video_path.find('click') != -1:
            self.action = 'click'
            self.action_label = 1
        elif video_path.find('cap') != -1:
            self.action = 'cap'
            self.action_label = 3
        elif video_path.find('altf') != -1:
            self.action = 'altf'
            self.action_label = 5
        elif video_path.find('alt') != -1:
            self.action = 'alt'
            self.action_label = 4
        else:
            self.action = 'nothing'
            self.action_label = 0

    def record_index(self, action_count, nothing_count):
        # 마지막 index 기록
        f_w = open(f'dataset/{self.action}/last_idx.txt', 'w')
        f_w.write(str(action_count))
        f_w.close()
        if nothing_count != 0:
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
                if cnt == 50:  # 제스처 데이터 수집 완료 (0.04초씩 50프레임)
                    gesture_cnt += 1
                    cnt = 0
                    print(action_data.shape)
                    np.save(os.path.join(f'dataset/{self.action}', f'seq_{action_count}'), action_data)
                    action_data = []
                    doubleclick_det = 0
                    if gesture_cnt == 8:
                        self.record_index(action_count, nothing_count)
                        break
                ret, img = cap.read()
                if not ret:
                    break

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if result.multi_hand_landmarks is not None:
                    for res in result.multi_hand_landmarks:
                        joint = np.zeros((21, 4))
                        for j, lm in enumerate(res.landmark):
                            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                        # Compute angles between joints
                        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
                             :3]  # Parent joint
                        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                             :3]  # Child joint
                        v = v2 - v1  # [20, 3]
                        # Normalize v
                        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                        # Get angle using arcos of dot product
                        angle = np.arccos(np.einsum('nt,nt->n',
                                                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],
                                                    :]))  # [15,]

                        angle = np.degrees(angle)  # Convert radian to degree

                        angle_label = np.array([angle], dtype=np.float32)

                        if cnt > 0:
                            angle_label = np.append(angle_label, self.action_label)
                            d = np.concatenate([joint.flatten(), angle_label])
                            action_data = np.append(action_data, d.reshape(1, 100), axis=0)
                            cnt += 1
                        else:
                            angle_label = np.append(angle_label, 0)
                            d = np.append(joint.flatten(), angle_label)
                            data.append(d)

                        dist = np.linalg.norm(joint[8] - joint[4])
                        if current_dist - dist > 0.02 and dist < 0.1 and doubleclick_det == 0:
                            action_count += 1
                            doubleclick_det = 1
                            print(f'{self.action}_{action_count}')
                            data = np.array(data)

                            # 동작의 sequence를 -5프레임부터 시작
                            action_data = data[-5:, :-1]
                            labels = np.reshape([self.action_label] * 5, (5, 1))
                            action_data = np.concatenate([action_data, labels], axis=1)  # 라벨 붙이기
                            action_data = np.reshape(action_data, (5, 100))

                            # nothing data 분류
                            if data.size >= 50 * 100:  # data가 50프레임 이상 존재할 때만
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

    def split_video_rclick(self):
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
            current_dist = 0  # 최근 엄지와 중지 사이의 거리

            ret, img = cap.read()
            cv2.imshow('webcam', img)

            data = []  # 기본 데이터 배열
            action_data = []  # 제스처 데이터 배열

            gesture_cnt = 0
            while True:
                if cnt == 50:  # 제스처 데이터 수집 완료 (0.04초씩 50프레임)
                    gesture_cnt += 1
                    cnt = 0
                    print(action_data.shape)
                    np.save(os.path.join(f'dataset/{self.action}', f'seq_{action_count}'), action_data)
                    action_data = []
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
                        joint = np.zeros((21, 4))
                        for j, lm in enumerate(res.landmark):
                            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                        # Compute angles between joints
                        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
                             :3]  # Parent joint
                        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                             :3]  # Child joint
                        v = v2 - v1  # [20, 3]
                        # Normalize v
                        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                        # Get angle using arcos of dot product
                        angle = np.arccos(np.einsum('nt,nt->n',
                                                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],
                                                    :]))  # [15,]

                        angle = np.degrees(angle)  # Convert radian to degree

                        angle_label = np.array([angle], dtype=np.float32)

                        if cnt > 0:
                            angle_label = np.append(angle_label, self.action_label)
                            d = np.concatenate([joint.flatten(), angle_label])
                            action_data = np.append(action_data, d.reshape(1, 100), axis=0)
                        else:
                            angle_label = np.append(angle_label, 0)
                            d = np.append(joint.flatten(), angle_label)
                            data.append(d)

                        dist = np.linalg.norm(joint[12] - joint[4])
                        if current_dist - dist > 0.03 and dist < 0.1 and cnt == 0:
                            action_count += 1
                            print(f'{self.action}_{action_count}')
                            data = np.array(data)

                            # 동작의 sequence를 -5프레임부터 시작
                            action_data = data[-5:, :-1]
                            labels = np.reshape([self.action_label] * 5, (5, 1))
                            action_data = np.concatenate([action_data, labels], axis=1)  # 라벨 붙이기
                            action_data = np.reshape(action_data, (5, 61))

                            # nothing data 분류
                            if data.size >= 50 * 61:  # data가 50프레임 이상 존재할 때만
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

    def split_video_cap(self):
        cv2.destroyAllWindows()
        cap = cv2.VideoCapture(self.video_path)  # opencv로 녹화 영상 틀기
        os.makedirs(f'dataset/{self.action}', exist_ok=True)

        action_count = 0
        if os.path.isfile(f'dataset/{self.action}/last_idx.txt'):
            f_r = open(f'dataset/{self.action}/last_idx.txt', 'r')
            action_count = int(f_r.readline())
            f_r.close()

        if cap.isOpened():
            cnt = 0  # 0보다 클 때 데이터 수집중
            current_dist = 0  # 최근 엄지와 중지 사이의 거리

            ret, img = cap.read()
            cv2.imshow('webcam', img)

            data = []  # 기본 데이터 배열
            action_data = []  # 제스처 데이터 배열

            gesture_cnt = 0
            while True:
                if cnt == 50:  # 제스처 데이터 수집 완료 (0.04초씩 50프레임)
                    gesture_cnt += 1
                    cnt = 0
                    print(action_data.shape)
                    np.save(os.path.join(f'dataset/{self.action}', f'seq_{action_count}'), action_data)
                    action_data = []
                    if gesture_cnt == 8:
                        self.record_index(action_count, 0)
                        break
                ret, img = cap.read()
                if not ret:
                    break

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if result.multi_hand_landmarks is not None:
                    for res in result.multi_hand_landmarks:
                        joint = np.zeros((21, 4))
                        for j, lm in enumerate(res.landmark):
                            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                        # Compute angles between joints
                        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
                             :3]  # Parent joint
                        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                             :3]  # Child joint
                        v = v2 - v1  # [20, 3]
                        # Normalize v
                        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                        # Get angle using arcos of dot product
                        angle = np.arccos(np.einsum('nt,nt->n',
                                                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],
                                                    :]))  # [15,]

                        angle = np.degrees(angle)  # Convert radian to degree

                        angle_label = np.array([angle], dtype=np.float32)

                        if cnt > 0:
                            angle_label = np.append(angle_label, self.action_label)
                            d = np.concatenate([joint.flatten(), angle_label])
                            action_data = np.append(action_data, d.reshape(1, 100), axis=0)
                            cnt += 1
                        else:
                            angle_label = np.append(angle_label, self.action_label)
                            d = np.append(joint.flatten(), angle_label)
                            data.append(d)

                        dist = np.linalg.norm(joint[12] - joint[0])
                        if current_dist - dist > 0.03 and dist < 1.25 and cnt == 0:
                            action_count += 1
                            print(f'{self.action}_{action_count}')
                            data = np.array(data)

                            # 동작의 sequence를 -5프레임부터 시작
                            action_data = data[-5:, :]
                            action_data = np.reshape(action_data, (5, 100))

                            data = []
                            cnt = 5

                        current_dist = dist
                        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                cv2.imshow('webcam', img)
                if cv2.waitKey(1) == ord('q'):
                    break

    def split_video_alt(self):
        cv2.destroyAllWindows()
        cap = cv2.VideoCapture(self.video_path)  # opencv로 녹화 영상 틀기
        os.makedirs(f'dataset/{self.action}', exist_ok=True)

        action_count = 0
        if os.path.isfile(f'dataset/{self.action}/last_idx.txt'):
            f_r = open(f'dataset/{self.action}/last_idx.txt', 'r')
            action_count = int(f_r.readline())
            f_r.close()

        if cap.isOpened():
            cnt = 0  # 0보다 클 때 데이터 수집중
            current_dist = 0  # 최근 엄지와 중지 사이의 거리

            ret, img = cap.read()
            cv2.imshow('webcam', img)

            data = []  # 기본 데이터 배열
            action_data = []  # 제스처 데이터 배열

            gesture_cnt = 0
            while True:
                if cnt == 50:  # 제스처 데이터 수집 완료 (0.04초씩 50프레임)
                    gesture_cnt += 1
                    cnt = 0
                    print(action_data.shape)
                    np.save(os.path.join(f'dataset/{self.action}', f'seq_{action_count}'), action_data)
                    action_data = []
                    if gesture_cnt == 8:
                        self.record_index(action_count, 0)
                        break
                ret, img = cap.read()
                if not ret:
                    break

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if result.multi_hand_landmarks is not None:
                    for res in result.multi_hand_landmarks:
                        joint = np.zeros((21, 4))
                        for j, lm in enumerate(res.landmark):
                            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                        # Compute angles between joints
                        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
                             :3]  # Parent joint
                        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                             :3]  # Child joint
                        v = v2 - v1  # [20, 3]
                        # Normalize v
                        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                        # Get angle using arcos of dot product
                        angle = np.arccos(np.einsum('nt,nt->n',
                                                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],
                                                    :]))  # [15,]

                        angle = np.degrees(angle)  # Convert radian to degree

                        angle_label = np.array([angle], dtype=np.float32)

                        if cnt > 0:
                            angle_label = np.append(angle_label, self.action_label)
                            d = np.concatenate([joint.flatten(), angle_label])
                            action_data = np.append(action_data, d.reshape(1, 100), axis=0)
                            cnt += 1
                        else:
                            angle_label = np.append(angle_label, self.action_label)
                            d = np.append(joint.flatten(), angle_label)
                            data.append(d)

                        v = np.outer(joint[4, :-1], joint[20, :-1])
                        dist = v[2, 0] + v[2, 1] + v[2, 2]
                        if dist > 0 and cnt == 0:
                            action_count += 1
                            print(f'{self.action}_{action_count}')
                            data = np.array(data)

                            # 동작의 sequence를 -5프레임부터 시작
                            action_data = data[-5:, :]
                            action_data = np.reshape(action_data, (5, 100))

                            data = []
                            cnt = 5

                        current_dist = dist
                        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                cv2.imshow('webcam', img)
                if cv2.waitKey(1) == ord('q'):
                    break

    def split_video_altf(self):
        cv2.destroyAllWindows()
        cap = cv2.VideoCapture(self.video_path)  # opencv로 녹화 영상 틀기
        os.makedirs(f'dataset/{self.action}', exist_ok=True)

        action_count = 0
        if os.path.isfile(f'dataset/{self.action}/last_idx.txt'):
            f_r = open(f'dataset/{self.action}/last_idx.txt', 'r')
            action_count = int(f_r.readline())
            f_r.close()

        if cap.isOpened():
            cnt = 0  # 0보다 클 때 데이터 수집중
            current_dist = 0  # 최근 엄지와 중지 사이의 거리

            ret, img = cap.read()
            cv2.imshow('webcam', img)

            data = []  # 기본 데이터 배열
            action_data = []  # 제스처 데이터 배열

            gesture_cnt = 0
            while True:
                if cnt == 50:  # 제스처 데이터 수집 완료 (0.04초씩 50프레임)
                    gesture_cnt += 1
                    cnt = 0
                    print(action_data.shape)
                    np.save(os.path.join(f'dataset/{self.action}', f'seq_{action_count}'), action_data)
                    action_data = []
                    if gesture_cnt == 8:
                        self.record_index(action_count, 0)
                        break

                ret, img = cap.read()
                if not ret:
                    break

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if result.multi_hand_landmarks is not None:
                    for res in result.multi_hand_landmarks:
                        joint = np.zeros((21, 4))
                        for j, lm in enumerate(res.landmark):
                            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                        # Compute angles between joints
                        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
                             :3]  # Parent joint
                        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                             :3]  # Child joint
                        v = v2 - v1  # [20, 3]
                        # Normalize v
                        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                        # Get angle using arcos of dot product
                        angle = np.arccos(np.einsum('nt,nt->n',
                                                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],
                                                    :]))  # [15,]

                        angle = np.degrees(angle)  # Convert radian to degree

                        angle_label = np.array([angle], dtype=np.float32)

                        if cnt > 0:
                            angle_label = np.append(angle_label, self.action_label)
                            d = np.concatenate([joint.flatten(), angle_label])
                            action_data = np.append(action_data, d.reshape(1, 100), axis=0)
                            cnt += 1
                        else:
                            angle_label = np.append(angle_label, self.action_label)
                            d = np.append(joint.flatten(), angle_label)
                            data.append(d)

                        v = np.outer(joint[0, :-1] - joint[12, :-1], joint[12, :-1])
                        # dist = v[0, 1] + v[1, 1] + v[2, 1]
                        dist = v[2, 0] + v[2, 1] - v[2, 2]
                        if dist > 0.08 > current_dist and cnt == 0:
                            action_count += 1
                            print(f'{self.action}_{action_count}')
                            data = np.array(data)

                            # 동작의 sequence를 -5프레임부터 시작
                            action_data = data[-5:, :]
                            action_data = np.reshape(action_data, (5, 100))

                            data = []
                            cnt = 5

                        current_dist = dist
                        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                cv2.imshow('webcam', img)
                if cv2.waitKey(1) == ord('q'):
                    break

    def split_video_nothing(self):
        cv2.destroyAllWindows()
        cap = cv2.VideoCapture(self.video_path)  # opencv로 녹화 영상 틀기
        os.makedirs(f'dataset/{self.action}', exist_ok=True)

        action_count = 0
        if os.path.isfile(f'dataset/{self.action}/last_idx.txt'):
            f_r = open(f'dataset/{self.action}/last_idx.txt', 'r')
            action_count = int(f_r.readline())
            f_r.close()

        if cap.isOpened():
            cnt = 0  # 0보다 클 때 데이터 수집중
            frame = 0

            ret, img = cap.read()
            cv2.imshow('webcam', img)

            data = []  # 기본 데이터 배열
            action_data = []  # 제스처 데이터 배열

            gesture_cnt = 0
            while True:
                if cnt == 50:  # 제스처 데이터 수집 완료 (0.04초씩 50프레임)
                    gesture_cnt += 1
                    cnt = 0
                    print(action_data.shape)
                    np.save(os.path.join(f'dataset/{self.action}', f'seq_{action_count}'), action_data)
                    action_data = []
                    if gesture_cnt == 8:
                        self.record_index(action_count, 0)
                        break
                ret, img = cap.read()
                if not ret:
                    break

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if result.multi_hand_landmarks is not None:
                    for res in result.multi_hand_landmarks:
                        joint = np.zeros((21, 4))
                        for j, lm in enumerate(res.landmark):
                            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                        # Compute angles between joints
                        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
                             :3]  # Parent joint
                        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                             :3]  # Child joint
                        v = v2 - v1  # [20, 3]
                        # Normalize v
                        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                        # Get angle using arcos of dot product
                        angle = np.arccos(np.einsum('nt,nt->n',
                                                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],
                                                    :]))  # [15,]

                        angle = np.degrees(angle)  # Convert radian to degree

                        angle_label = np.array([angle], dtype=np.float32)

                        if cnt > 0:
                            angle_label = np.append(angle_label, self.action_label)
                            d = np.concatenate([joint.flatten(), angle_label])
                            action_data = np.append(action_data, d.reshape(1, 100), axis=0)
                            cnt += 1
                        else:
                            angle_label = np.append(angle_label, 0)
                            d = np.append(joint.flatten(), angle_label)
                            data.append(d)
                        frame += 1

                        if cnt == 0 and frame % 120 == 30:
                            action_count += 1
                            print(f'{self.action}_{action_count}')
                            data = np.array(data)

                            # 동작의 sequence를 -5프레임부터 시작
                            action_data = data[-5:, :]
                            action_data = np.reshape(action_data, (5, 100))

                            data = []
                            cnt = 5

                        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                cv2.imshow('webcam', img)
                if cv2.waitKey(1) == ord('q'):
                    break

    def show_skeleton(self):  # MediaPipe 동작 확인용 코드
        cv2.destroyAllWindows()
        cap = cv2.VideoCapture(self.video_path)
        if cap.isOpened():
            cnt = 0
            current_dist = 0
            gesture_cnt = 0

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
