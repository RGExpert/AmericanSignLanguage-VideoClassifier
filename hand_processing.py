#!/usr/bin/env python

import os
import numpy as np
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils         # type:ignore
mp_drawing_styles = mp.solutions.drawing_styles # type:ignore
mp_hands = mp.solutions.hands                   # type:ignore

def vectorize(hand_landmarks):
    vectors = np.empty((0, 3))

    # from palm to finger tips
    for iter in range(5):
        start_idx = 0
        next_idx = iter * 4 + 1
        for _ in range(4):
            vector = np.array([
                hand_landmarks.landmark[next_idx].x - hand_landmarks.landmark[start_idx].x,
                hand_landmarks.landmark[next_idx].y - hand_landmarks.landmark[start_idx].y,
                hand_landmarks.landmark[next_idx].z - hand_landmarks.landmark[start_idx].z
            ])

            normalized_vector = vector / np.linalg.norm(vector)
            vectors = np.vstack([vectors, normalized_vector])

            start_idx = next_idx
            next_idx = next_idx + 1

    # in between fingers
    for iter in range(3):
        start_idx = iter * 4 + 5
        next_idx = (iter + 1) * 4 + 5
        vector = np.array([
            hand_landmarks.landmark[next_idx].x - hand_landmarks.landmark[start_idx].x,
            hand_landmarks.landmark[next_idx].y - hand_landmarks.landmark[start_idx].y,
            hand_landmarks.landmark[next_idx].z - hand_landmarks.landmark[start_idx].z
        ])

        normalized_vector = vector / np.linalg.norm(vector)
        vectors = np.vstack([vectors, normalized_vector])
    return vectors.flatten()


def process_images(base_dir, max_images=100):
    features = []
    labels = []

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:

        for subdir, _, files in os.walk(base_dir):
            label = os.path.basename(subdir)
            for i, file in enumerate(files):
                file_path = os.path.join(subdir, file)

                img = cv2.imread(file_path)
                results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                if not results.multi_hand_landmarks:
                    continue

                for hand_landmarks in results.multi_hand_landmarks:
                    vectorized_data = vectorize(hand_landmarks)
                    features.append(vectorized_data)
                    labels.append(label)

                if i >= max_images:
                    break

        return np.array(features), np.array(labels)


if __name__ == "__main__":
    train_dir = "ASL_Alphabet_Dataset/asl_alphabet_train"  # Adjust the path to your dataset
    features, labels = process_images(train_dir)

    # Now features and labels are ready for machine learning
    print(f"Total samples: {len(features)}")
    print(f"Feature shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")


