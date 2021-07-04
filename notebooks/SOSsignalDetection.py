# import Dependendencies
import cv2
import numpy as np
import os

from matplotlib import pyplot as ptl
import time
import mediapipe as mp
#from DataCollections import DATA_PATH
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
#from LoadData import load_data
# Key points using MP Holistics

# Holistic model
mp_holistic = mp.solutions.holistic
# Drawing Utilities
mp_drawing = mp.solutions.drawing_utils
# path for exported data, numpy arrays
DATA_PATH = os.path.join('E:\\9th sem\\MS_Thesis\\Awiros\\Project\\data\\MP_DATA')


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # color conversion
    image.flags.writeable = False  # image is no longer writable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # image is no longer writable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # color conversion
    return image, results


def draw_landmark(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(1, 45, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(70, 26, 21), thickness=2, circle_radius=2)
                              )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(65, 150, 120), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(82, 96, 11), thickness=2, circle_radius=2)
                              )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(170, 54, 110), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(40, 56, 51), thickness=2, circle_radius=2)
                              )


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, face, lh, rh])


#preprocess Data and create Label and features
actions = np.array(['hello', 'thanks', 'iloveyou'])

no_sequences = 30
sequence_length = 30

label_map = {label: num for num, label in enumerate(actions)}


def load_data():
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    return np.array(sequences), labels
X, Y = load_data()

y = to_categorical(Y).astype(int)




cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        image, results = mediapipe_detection(frame, holistic)
        # dprint(results.pose_landmarks)
        # draw landmark
        draw_landmark(image, results)
        x = extract_keypoints(results)
        print(x)
        cv2.imshow('OpenCv Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
# print(x)
