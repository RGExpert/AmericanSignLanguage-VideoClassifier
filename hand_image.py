#!/usr/bin/env python

import numpy as np
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils         # type:ignore
mp_drawing_styles = mp.solutions.drawing_styles # type:ignore
mp_hands = mp.solutions.hands                   # type:ignore

def vectorize(
    hand_landmarks
):
    vectors = np.empty((0, 3))

    # from palm to finger
    for iter in range(5):
        start_idx = 0;
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
    for iter in range (3):
        start_idx = iter * 4 + 5
        next_idx = (iter + 1) * 4 + 5
        vector = np.array([
            hand_landmarks.landmark[next_idx].x - hand_landmarks.landmark[start_idx].x,
            hand_landmarks.landmark[next_idx].y - hand_landmarks.landmark[start_idx].y,
            hand_landmarks.landmark[next_idx].z - hand_landmarks.landmark[start_idx].z
        ])

        normalized_vector = vector / np.linalg.norm(vector)
        vectors = np.vstack([vectors, normalized_vector])

    print(f"Vectorized values are:\n{vectors}")


def process_images(image_files):
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
        for idx, file in enumerate(image_files):
            img = cv2.flip(cv2.imread(file), 1);
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if not results.multi_hand_landmarks:
                continue


            height = 512
            width = 512
            hand_wireframe = np.zeros((height, width, 3), np.uint8)
            for hand_landmarks in results.multi_hand_landmarks:
                print(f"Landmarks:\n {hand_landmarks}s")
                vectorize(hand_landmarks)
                mp_drawing.draw_landmarks(
                    hand_wireframe,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            cv2.imwrite(f"./tmp/hand_wireframe{str(idx)}.png", cv2.flip(hand_wireframe, 1))


if __name__ == "__main__":
    image_files = ["ASL_Alphabet_Dataset/asl_alphabet_train/A/1.jpg", "ASL_Alphabet_Dataset/asl_alphabet_train/A/2.jpg"]
    process_images(image_files)
