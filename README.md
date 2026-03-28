# ASL Classifier

A classifier that can recognize hand symbols from **American Sign Language (ASL)** corresponding to the letters of the alphabet.

It uses **MediaPipe** to extract hand landmarks ([Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/index)), transforms the landmarks into vectors that better describe hand direction, and then uses a simple **MLP (Multi-Layer Perceptron)** with a single hidden layer to make predictions.  

Even a very simple MLP with a single layer can achieve **~98% accuracy** on the validation set.

---

## Features

- Real-time ASL letter recognition from webcam.
- Uses **MediaPipe Hand Landmarker** for robust hand tracking.
- Converts hand landmarks into vectors suitable for MLP input.
- Lightweight **MLP** model achieving high accuracy.
- Easy to retrain or use pre-trained weights.

---

## Setup

Create a virtual environment and install dependencies.

**Linux / macOS:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows**
```ps
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```
---

## Run
Weights are included, so you can run the model directly:
```bash
python run.py
```
Or you can retrain the model from scratch:
```bash
python train.py
```
