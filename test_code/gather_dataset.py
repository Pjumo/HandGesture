import cv2
import mediapipe as mp
import numpy as np
import os

action = 'doubleclick'
action_idx = 2

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
cv2.namedWindow('webcam', cv2.WINDOW_AUTOSIZE)

os.makedirs(f'dataset/{action}', exist_ok=True)
action_count = 0
if os.path.isfile(f'dataset/{action}/last_idx.txt'):
    f_r = open(f'dataset/{action}/last_idx.txt', 'r')
    action_count = int(f_r.readline())
    f_r.close()

while cap.isOpened():
    for i in range(10):
        action_count += 1
        data = []

        ret, img = cap.read()

        img = cv2.flip(img, 1)

        cv2.putText(img, f'Waiting for action {action_count}', org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('webcam', img)
        cv2.waitKey(2000)

        data_count = 0

        while data_count < 50:
            ret, img = cap.read()

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # Compute angles between joints
                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                         :3]  # Child joint
                    v = v2 - v1  # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                    angle = np.degrees(angle)  # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, action_idx)

                    d = np.concatenate([joint.flatten(), angle_label])

                    data.append(d)
                    data_count += 1

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('webcam', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)
        np.save(os.path.join(f'dataset/{action}', f'seq_{action_count}'), data)

    f_w = open(f'dataset/{action}/last_idx.txt', 'w')
    f_w.write(str(action_count))
    f_w.close()
    break
