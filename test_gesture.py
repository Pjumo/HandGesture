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
zero_list = [0, 0, 0, 0, 0, 0]
ensemble_weight_c = [0.5, 0, 0.6, 0.8, 0.7, 0.5]
ensemble_weight_r = [0.5, 1, 0.4, 0.2, 0.3, 0.5]

model_c = load_model('models/model_webcam.h5')
model_r = load_model('models/model_radar_lstm.h5')

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


def test_video():
    cap = cv2.VideoCapture(f'dataset/test/video/test{num}.mp4')
    cap1 = cv2.VideoCapture(f'dataset/image/test_{num}.mp4')
    cap2 = cv2.VideoCapture(f'dataset/test/video/{num}.mp4')
    cv2.namedWindow('webcam', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('gif', cv2.WINDOW_NORMAL)
    cv2.moveWindow('gif', 0, 0)
    cv2.resizeWindow('gif', 620, 500)
    cv2.moveWindow('webcam', 630, 0)

    seq_c = np.load(f'dataset/test/seq_{num}.npy')
    seq_r = np.load(f'dataset/test_r/test_{num}.npy').swapaxes(0, 1)
    indices = np.where(seq_r == -math.inf)
    seq_r[indices] = 0
    actions_true = np.load(f'dataset/test/video/true_{num}.npy')
    actions_true = np.insert(actions_true, 0, [0, 0])

    # seq_front = np.zeros((128, 100))
    # seq_radar = np.append(seq_front, seq_r.swapaxes(0, 1), axis=1)
    # plt.rcParams["figure.figsize"] = [7.5, 5]
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # div = make_axes_locatable(ax)
    # cax = div.append_axes('right', '5%', '5%')
    # im = ax.imshow(seq_radar, aspect='auto')
    # ax.set_xlabel('frame')
    # ax.set_ylabel('velocity(m/s)')
    # ax.set_xlim(0, 100)
    # xticks = np.linspace(start=0, stop=1600, num=81)
    # xticklabels = np.linspace(start=-100, stop=1800, num=81, dtype=int)
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticklabels)
    # ax.set_yticks([0, 32, 64, 96, 128])
    # ax.set_yticklabels(np.array([64, 32, 0, -32, -64]) * 0.1201)
    # fig.colorbar(im, cax=cax)
    #
    # def animate(i):
    #     ax.set_xlim((1 + i), (101 + i))
    #     return im
    #
    # anim = animation.FuncAnimation(fig, animate, frames=1500, interval=100/3, repeat=False)
    #
    # # plt.show()
    # anim.save(f'dataset/image/test_{num}.gif', writer='imagemagick', fps=30)

    correct_c = 0
    correct_r = 0
    correct_ensemble = 0
    predict_delay = 0
    action_float_time = 0
    cnt = 0
    predict_c = 'nothing'
    predict_r = 'nothing'
    predict_ensemble = 'nothing'
    while cnt < 1500:
        ret, img = cap.read()
        ret1, img1 = cap1.read()
        ret2, img2 = cap2.read()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        hand_detected = 1
        action_true = actions[actions_true[cnt // 10]]
        cnt += 1
        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
        else:
            hand_detected = 0

        predict_delay += 1

        if action_float_time == 20:
            action_float_time = 0

        if predict_delay >= 10 and action_float_time == 0 and cnt >= seq_length:
            predict_delay = 0
            if hand_detected == 1:
                input_data_c = np.expand_dims(np.array(seq_c[(cnt - seq_length):cnt], dtype=np.float32), axis=0)
                y_predict_c = model_c.predict(input_data_c).squeeze()
                i_predict_c = int(np.argmax(y_predict_c))
                conf = y_predict_c[i_predict_c]
                if conf > 0.8:
                    predict_c = actions[i_predict_c]
                else:
                    predict_c = 'nothing'
            else:
                y_predict_c = zero_list
                predict_c = 'nothing'
            print(cnt)
            print(f'webcam : {y_predict_c}')

            input_data_r = np.expand_dims(
                np.array(seq_r[(int(cnt / 1.2) + 20 - seq_length):int(cnt / 1.2) + 20], dtype=np.float32), axis=0)
            y_predict_r = model_r.predict(input_data_r).squeeze()
            i_predict_r = int(np.argmax(y_predict_r))
            conf = y_predict_r[i_predict_r]
            if conf > 0.7:
                predict_r = actions[i_predict_r]
            else:
                predict_r = 'nothing'
            print(f'radar : {y_predict_r}')

            if y_predict_c == zero_list:
                if num == 4:  # doppler 작게 측정 되는 경우로 변경
                    if y_predict_r[0] > 0.9:
                        y_predict_ensemble = [1, 0, 0, 0, 0, 0]
                    else:
                        y_predict_new = np.array([y_predict_r[3], y_predict_r[4], y_predict_r[5]])
                        y_predict_ensemble = y_predict_new / np.linalg.norm(y_predict_new)
                        y_predict_ensemble = np.insert(y_predict_ensemble, 0,
                                                       [y_predict_r[0], y_predict_r[1], y_predict_r[2]]) / 2
                        y_predict_ensemble = y_predict_ensemble * 1.5
                else:
                    y_predict_ensemble = y_predict_r
            elif y_predict_r[1] > 0.2 and y_predict_r[2] > 0.2 and y_predict_r[0] > 0.2 and y_predict_c[0] > 0.6:
                y_predict_ensemble = [0, 1, 0, 0, 0, 0]
            elif y_predict_c[0] > 0.9 or y_predict_r[0] > 0.8:
                y_predict_ensemble = [1, 0, 0, 0, 0, 0]
            elif y_predict_c[4] > 0.1 and y_predict_r[4] > 0.1:
                y_predict_ensemble = [0, 0, 0, 0, 1, 0]
            else:
                y_predict_ensemble = (np.array(y_predict_c) * ensemble_weight_c) + \
                                     (np.array(y_predict_r) * ensemble_weight_r)
            i_predict_ensemble = int(np.argmax(y_predict_ensemble))

            if y_predict_ensemble[i_predict_ensemble] > 0.7:
                predict_ensemble = actions[i_predict_ensemble]
            else:
                predict_ensemble = 'nothing'
            print(f'ensemble : {y_predict_ensemble}')
        if predict_ensemble != 'nothing':
            action_float_time += 1

        if action_true != 'nothing':
            if action_true == predict_c:
                correct_c += 1
            if action_true == predict_r:
                correct_r += 1
            if action_true == predict_ensemble:
                correct_ensemble += 1

        cv2.putText(img, f'camera : {predict_c}',
                    org=(10, 250),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.putText(img, f'radar : {predict_r}',
                    org=(10, 300),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.putText(img, f'ensemble : {predict_ensemble}',
                    org=(10, 350),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(55, 55, 255), thickness=2)
        cv2.putText(img, f'true_action : {action_true}',
                    org=(0, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 55, 55), thickness=2)

        img2 = cv2.resize(img2, (480, 270))
        img[450:720, 800:1280] = img2
        cv2.imshow('webcam', img)
        cv2.imshow('gif', img1[:, 100:1180])
        if cv2.waitKey(1) == ord('q'):
            break

    action_count = np.count_nonzero(actions_true)
    plt.bar(np.arange(3), [(correct_c / action_count) * 10, (correct_r / action_count) * 10,
                           (correct_ensemble / action_count) * 10])
    plt.title('Accuracy of Each Model')
    plt.ylabel('Accuracy(%)')
    plt.xticks(np.arange(3), ['WebCam', 'Radar', 'Ensemble'])
    plt.show()

    cap.release()
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
