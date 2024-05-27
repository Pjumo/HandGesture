import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from keras.models import load_model
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

actions = ['nothing', 'click', 'doubleclick', 'cap', 'alt', 'altf']
seq_length = 50
num = 1

model_c = load_model('models/model_webcam.h5')
model_r = load_model('models/model_radar_cnn.h5')

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


def test_video():
    cap = cv2.VideoCapture(f'video_webcam/any{num}.mp4')
    cap1 = cv2.VideoCapture(f'dataset/image/test_{num}.gif')
    cv2.namedWindow('webcam', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('gif', cv2.WINDOW_NORMAL)
    cv2.moveWindow('gif', 0, 0)
    cv2.resizeWindow('gif', 620, 500)
    cv2.moveWindow('webcam', 630, 0)

    seq_c = np.load(f'dataset/any/seq_{num}.npy')
    seq_r = np.load(f'dataset/any_r/any_{num}.npy').reshape((1000, 128))
    indices = np.where(seq_r == -math.inf)
    seq_r[indices] = 0

    # seq_front = np.zeros((128, 99))
    # seq_radar = np.append(seq_front, seq_radar, axis=1)
    #
    # plt.rcParams["figure.figsize"] = [7.5, 5]
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # div = make_axes_locatable(ax)
    # cax = div.append_axes('right', '5%', '5%')
    # im = ax.imshow(seq_radar, aspect='auto')
    # ax.set_xlabel('frame')
    # ax.set_ylabel('velocity(m/s)')
    # ax.set_xlim(0, 100)
    # xticks = np.array([-100, -80, -60, -40, -20, 0])
    # ax.set_xticklabels(xticks)
    # ax.set_yticks([0, 32, 64, 96, 128])
    # ax.set_yticklabels(np.array([64, 32, 0, -32, -64]) * 0.1201)
    # fig.colorbar(im, cax=cax)
    #
    # def animate(i):
    #     ax.set_xlim((0 + i), (100 + i))
    #     ax.set_xticklabels(xticks + i)
    #     return im
    #
    # anim = animation.FuncAnimation(fig, animate, frames=1000, interval=40, repeat=False)
    #
    # anim.save('dataset/image/test_1.gif', writer='imagemagick', fps=25)

    predict_delay = 0
    action_float_time = 0
    cnt = 0
    predict_c = 'nothing'
    predict_r = 'nothing'
    predict_ensemble = 'nothing'
    while cap.isOpened():
        ret, img = cap.read()
        ret1, img1 = cap1.read()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        hand_detected = 1
        cnt += 1
        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
        else:
            hand_detected = 0

        predict_delay += 1

        # if action_float_time == 20:
        #     action_float_time = 0

        if predict_delay >= 10 and action_float_time == 0 and cnt > seq_length:
            predict_delay = 0
            if hand_detected == 1:
                input_data_c = np.expand_dims(np.array(seq_c[(cnt - seq_length):cnt], dtype=np.float32), axis=0)
                y_predict_c = model_c.predict(input_data_c).squeeze()
                i_predict_c = int(np.argmax(y_predict_c))
                conf = y_predict_c[i_predict_c]
                predict_c = actions[i_predict_c]
            else:
                y_predict_c = [0, 0, 0, 0, 0, 0]
                predict_c = 'nothing'

            input_data_r = np.expand_dims(np.array(seq_r[(cnt - seq_length):cnt], dtype=np.float32), axis=0)
            y_predict_r = model_r.predict(input_data_r).squeeze()
            i_predict_r = int(np.argmax(y_predict_r))
            conf = y_predict_r[i_predict_r]
            predict_r = actions[i_predict_r]

            y_predict_ensemble = (y_predict_c + y_predict_r) / 2
            i_predict_ensemble = int(np.argmax(y_predict_ensemble))
            predict_ensemble = actions[i_predict_ensemble]

            # if conf > 0.5:
            #     this_action = actions[i_predict]
            # else:
            #     this_action = 'nothing'
        # if this_action != 'nothing':
        #     action_float_time += 1

        cv2.putText(img, f'camera : {predict_c}',
                    org=(10, 250),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 55, 55), thickness=2)
        cv2.putText(img, f'radar : {predict_r}',
                    org=(10, 300),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 55, 55), thickness=2)
        cv2.putText(img, f'ensemble : {predict_ensemble}',
                    org=(10, 350),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(55, 55, 255), thickness=2)

        cv2.imshow('webcam', img)
        cv2.imshow('gif', img1)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cap1.release()
    cv2.destroyAllWindows()
