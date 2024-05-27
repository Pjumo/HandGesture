import matplotlib.pyplot as plt

import gather_dataset_cam as cam
import train_gesture as train
import test_gesture as test
import numpy as np
import math
import os
import cv2
import mediapipe as mp
import pickle

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
# gathering data code
'''
for i in range(20):
    path = f'./video_webcam/altf{i+1}.mp4'
    webcam = cam.WebCamData(path)
    print(i+1)
    webcam.split_video_altf()
    # webcam.show_skeleton()
'''
# train.train_video()
train.train_radar()
# test.test_video()

# history = pickle.load(open('models/trainHistoryDict_webcam', 'rb'))
# fig = plt.figure()
# plt.plot(np.linspace(1, 200, 200), history['val_acc'])
# plt.xlabel('epoch')
# plt.ylabel('acc')
# plt.show()

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(
#     max_num_hands=1,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5)
# cap = cv2.VideoCapture('video_webcam/any5.mp4')
# cv2.namedWindow('webcam', cv2.WINDOW_AUTOSIZE)
# seq = []
# cnt = 0
# while cap.isOpened():
#     ret, img = cap.read()
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     result = hands.process(img)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#
#     if result.multi_hand_landmarks is not None:
#         for res in result.multi_hand_landmarks:
#             joint = np.zeros((21, 4))
#             for j, lm in enumerate(res.landmark):
#                 joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
#
#             # Compute angles between joints
#             v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
#                  :3]  # Parent joint
#             v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
#                  :3]  # Child joint
#             v = v2 - v1  # [20, 3]
#             # Normalize v
#             v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
#
#             # Get angle using arcos of dot product
#             angle = np.arccos(np.einsum('nt,nt->n',
#                                         v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
#                                         v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],
#                                         :]))  # [15,]
#
#             angle = np.degrees(angle)  # Convert radian to degree
#
#             angle_label = np.array([angle], dtype=np.float32)
#
#             joint_flat = np.append(joint.flatten(), angle_label)
#             seq.append(joint_flat)
#             mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
#     else:
#         joint = np.zeros(99)
#         seq.append(joint)
#     print(np.array(seq).shape)
#     cv2.imshow('webcam', img)
#     cnt += 1
#     if cv2.waitKey(1) == ord('q'):
#         break
#     if cnt == 1040:
#         np.save(os.path.join(f'dataset/any', f'seq_5'), seq[40:])
