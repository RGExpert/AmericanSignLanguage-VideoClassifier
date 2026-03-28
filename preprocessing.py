#!/usr/bin/env python3
import os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import os

# Stall on python3.13 maybe if you have an older python version/GPU support
# you can comment these lines out
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


HAND_MODEL_PATH = "models/hand_landmarker.task"

def vectorize(landmarks):
    """Convert hand landmarks to normalized vectors."""
    vectors = np.empty((0, 3))

    # from palm to finger tips
    for finger in range(5):
        start_idx = 0
        next_idx = finger * 4 + 1
        for _ in range(4):
            v = np.array([
                landmarks[next_idx].x - landmarks[start_idx].x,
                landmarks[next_idx].y - landmarks[start_idx].y,
                landmarks[next_idx].z - landmarks[start_idx].z
            ])
            norm = np.linalg.norm(v)
            if norm > 0:
                v /= norm
            vectors = np.vstack([vectors, v])
            start_idx = next_idx
            next_idx += 1

    # In between fingers
    for finger in range(3):
        start_idx = finger * 4 + 5
        next_idx = (finger + 1) * 4 + 5
        v = np.array([
            landmarks[next_idx].x - landmarks[start_idx].x,
            landmarks[next_idx].y - landmarks[start_idx].y,
            landmarks[next_idx].z - landmarks[start_idx].z,
        ])
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        vectors = np.vstack([vectors, v])

    return vectors.flatten()

def preprocess_images(base_dir, max_images=100):
    total_samples = 0
    for subdir, _, files in os.walk(base_dir):
        total_samples += min(len(files), max_images)

    VECTOR_SIZE = 69
    features = np.zeros((total_samples, VECTOR_SIZE), dtype=np.float32)
    labels = np.empty(total_samples, dtype=object)

    # Build the HandLandmarker task
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    RunningMode = vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=RunningMode.IMAGE
    )

    idx = 0
    with HandLandmarker.create_from_options(options) as landmarker:
        for subdir, _, files in os.walk(base_dir):
            label = os.path.basename(subdir)
            count = 0

            for file in files:
                if count >= max_images:
                    break

                img_path = os.path.join(subdir, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                )

                result = landmarker.detect(mp_image)
                if not result.hand_landmarks:
                    continue

                hand = result.hand_landmarks[0]
                features[idx] = vectorize(hand)
                labels[idx] = label
                idx += 1
                count += 1
    features = features[:idx]
    labels = labels[:idx]


    return features, labels

if __name__ == "__main__":
    train_dir = "data/ASL_Alphabet_Dataset/asl_alphabet_train"
    features, labels = preprocess_images(train_dir)

    print(f"Total samples: {len(features)}")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
