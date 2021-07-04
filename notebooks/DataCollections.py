import os
import numpy as np
import cv2
import mediapipe as mp


# Key points using MP Holistics

# Holistic model
mp_holistic = mp.solutions.holistic
# Drawing Utilities
mp_drawing = mp.solutions.drawing_utils



# path for exported data, numpy arrays
DATA_PATH = os.path.join('E:\\9th sem\\MS_Thesis\\Awiros\\Project\\data\\MP_DATA')

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Thirty video of data
no_sequences = 30

# Videos are going to be 30 fps
sequence_length = 30

# for action in actions:
#     for sequence in range(no_sequences):
#         try:
#             os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
#         except:
#             pass
#
# def mediapipe_detection(image, model):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # color conversion
#     image.flags.writeable = False  # image is no longer writable
#     results = model.process(image)  # Make prediction
#     image.flags.writeable = True  # image is no longer writable
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # color conversion
#     return image, results
#
#
# def draw_landmarks(image, results):
#     mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
#                               mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
#                               mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
#     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
#                               mp_drawing.DrawingSpec(color=(1, 45, 10), thickness=2, circle_radius=4),
#                               mp_drawing.DrawingSpec(color=(70, 26, 21), thickness=2, circle_radius=2)
#                               )
#     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                               mp_drawing.DrawingSpec(color=(65, 150, 120), thickness=2, circle_radius=4),
#                               mp_drawing.DrawingSpec(color=(82, 96, 11), thickness=2, circle_radius=2)
#                               )
#     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                               mp_drawing.DrawingSpec(color=(170, 54, 110), thickness=2, circle_radius=4),
#                               mp_drawing.DrawingSpec(color=(40, 56, 51), thickness=2, circle_radius=2)
#                               )
#
#
# def extract_keypoints(results):
#     pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
#                      results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
#     face = np.array([[res.x, res.y, res.z] for res in
#                      results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
#     lh = np.array([[res.x, res.y, res.z] for res in
#                    results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
#     rh = np.array([[res.x, res.y, res.z] for res in
#                    results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
#         21 * 3)
#     return np.concatenate([pose, face, lh, rh])
#
#
# cap = cv2.VideoCapture(0)
# # Set mediapipe model
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     # NEW LOOP
#     # Loop through actions
#     for action in actions:
#         # Loop through sequences aka videos
#         for sequence in range(no_sequences):
#             # Loop through video length aka sequence length
#             for frame_num in range(sequence_length):
#
#                 # Read feed
#                 ret, frame = cap.read()
#
#                 # Make detections
#                 image, results = mediapipe_detection(frame, holistic)
#                 #                 print(results)
#
#                 # Draw landmarks
#                 draw_landmarks(image, results)
#
#                 # NEW Apply wait logic
#                 if frame_num == 0:
#                     cv2.putText(image, 'STARTING COLLECTION', (120, 200),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
#                     cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#                     # Show to screen
#                     cv2.imshow('OpenCV Feed', image)
#                     cv2.waitKey(2000)
#                 else:
#                     cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#                     # Show to screen
#                     cv2.imshow('OpenCV Feed', image)
#
#                 # NEW Export keypoints
#                 keypoints = extract_keypoints(results)
#                 npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
#                 np.save(npy_path, keypoints)
#
#                 # Break gracefully
#                 if cv2.waitKey(10) & 0xFF == ord('q'):
#                     break
#
#     cap.release()
#     cv2.destroyAllWindows()