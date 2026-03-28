#!/usr/bin/env python

import cv2
import keras
import joblib
import numpy as np
import mediapipe as mp
import os
from training import vectorize, HAND_MODEL_PATH
from mediapipe.tasks.python import vision

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

def load_model():
    model = keras.saving.load_model('models/ASL_model.keras')
    label_encoder = joblib.load('models/label_encoder.pkl')

    return model, label_encoder

def run_model_on_video():
    model, label_encoder = load_model()
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    RunningMode = vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=RunningMode.IMAGE
    )


    cap = cv2.VideoCapture(0)
    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            img.flags.writeable = False
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            )

            result = landmarker.detect(mp_image)

            if not result.hand_landmarks:
                continue

            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            for hand_landmarks in result.hand_landmarks:
                # mp_drawing.draw_landmarks(
                #     img,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                # mp_drawing_styles.get_default_hand_landmarks_style(),
                # mp_drawing_styles.get_default_hand_connections_style())

                vectorized_data = vectorize(hand_landmarks)
                X = np.array(vectorized_data).reshape(1, -1)
                y_pred = model.predict(X, verbose=0)    # type: ignore

                pred_class = np.argmax(y_pred, axis=1)
                pred_label = label_encoder.inverse_transform(pred_class)[0]

                print("\033c")
                print(f"Predicted letter: {pred_label}")

            cv2.imshow('MediaPipe Hands', cv2.flip(img, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

if __name__ == "__main__":
    run_model_on_video()
