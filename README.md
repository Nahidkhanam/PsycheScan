# PsycheScan – AI Emotion Recognition System

PsycheScan is an AI-powered web application that detects human emotions from facial expressions using Deep Learning.  
The system uses a Convolutional Neural Network (CNN) trained on grayscale facial images (48x48) to classify emotions in real-time or from uploaded videos.

---

## Live Demo
🌐 Live App: https://psychescan.onrender.com

---

## Features

- 🔐 User Authentication (Sign Up / Login)
- 🎥 Live Camera Emotion Detection
- 📤 Video Upload Emotion Analysis
- 📊 Real-time Prediction Display
- 🧠 Deep Learning CNN Model
- 🌐 Deployed on Render

---

## Model Details

- Model Type: Convolutional Neural Network (CNN)
- Input Shape: 48x48 grayscale images
- Output Classes:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral
- Model Format: `.keras`
- Framework: TensorFlow / Keras

---

## 🏗️ Tech Stack

### Backend
- Flask
- Flask-SQLAlchemy
- Flask-Login
- Gunicorn

### AI / ML
- TensorFlow
- Keras
- OpenCV
- NumPy

### Database
- SQLite (default)

### Deployment
- Render

---

## Project Structure

```text
emotion_app/
│
├── app.py
├── cnn_first_model.keras
├── requirements.txt
├── runtime.txt
├── README.md
│
├── static/
│   ├── uploads/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│   ├── home.html
│   ├── login.html
│   ├── signup.html
│   ├── dashboard.html
│   ├── live.html
│   └── upload.html
│
└── instance/
    └── database.db
```


