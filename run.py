#!/usr/bin/env python

import cv2
import keras
import joblib
import numpy as np
import mediapipe as mp
from hand_processing import vectorize

mp_drawing = mp.solutions.drawing_utils         # type:ignore
mp_drawing_styles = mp.solutions.drawing_styles # type:ignore
mp_hands = mp.solutions.hands                   # type:ignore

def load_model():
    model = keras.saving.load_model('ASL_model.keras')
    label_encoder = joblib.load('label_encoder.pkl')

    return model, label_encoder

def run_model_on_video():
    model, label_encoder = load_model()

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                    vectorized_data = vectorize(hand_landmarks)
                    X = np.array(vectorized_data).reshape(1, -1)
                    y_pred = model.predict(X, verbose=0)    # type: ignore

                    pred_class = np.argmax(y_pred, axis=1)
                    pred_label = label_encoder.inverse_transform(pred_class)[0]

                    print("\033c")
                    print(f"Predicted letter: {pred_label}")

            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

if __name__ == "__main__":
    run_model_on_video()
