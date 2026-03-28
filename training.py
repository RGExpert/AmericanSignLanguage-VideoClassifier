#!/usr/bin/env python3
import os
import cv2
import joblib
import os
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks.python import vision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping

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

def extract_and_save_features(data_dir, max_images):
    X, y = preprocess_images(data_dir, max_images)
    np.save('models/features.npy', X)
    np.save('models/labels.npy', y)
    return X, y

def encode_and_save_labels(y):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(y)
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    return encoded_labels


def display_metrics(model, X_test, y_test, label_encoder):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

def train(
    data_dir: str,
    extract_features: bool=False,
    max_images: int=2000,
    test_size: float=0.2,
    epochs: int=200,
    batch_size: int=32,
    model_path: str="models/ASL_model.keras",
    features_path: str="models/features.npy",
    labels_path: str="models/labels.npy",
    encoder_path: str="models/label_encoder.pkl"
):
    if extract_features:
        X, y = extract_and_save_features(data_dir, max_images)
    else:
        X = np.load(features_path)
        y = np.load(labels_path, allow_pickle=True)

    if os.path.exists(encoder_path):
        label_encoder = joblib.load(encoder_path)
        encoded_labels = label_encoder.transform(y)
    else:
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(y)
        os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
        joblib.dump(label_encoder, encoder_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, encoded_labels, test_size=test_size, random_state=42  # remove stratify
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )

    num_classes = len(label_encoder.classes_)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    print(f"Model saved to {model_path}")
    print(f"LabelEncoder saved to {encoder_path}")


if __name__ == "__main__":
    train_dir = "data/ASL_Alphabet_Dataset/asl_alphabet_train"
    train(train_dir, False)

