from DataCollections import sequence_length,no_sequences,actions

import os
import numpy as np

label_map = {label: num for num, label in enumerate(actions)}


def load_data(DATA_PATH):
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
