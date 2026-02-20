import os
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, render_template, Response, request, send_file
from tensorflow.keras.models import load_model
from collections import Counter

app = Flask(__name__)

# ---------------- FOLDERS ----------------
UPLOAD_FOLDER = "uploads"
REPORT_FOLDER = "reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# ---------------- LOAD MODEL ----------------
model = load_model("cnn_first_model.keras", compile=False)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

classes = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

camera = None
emotion_list = []
is_recording = False


# ---------------- PREPROCESS ----------------
def preprocess(face):
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = np.expand_dims(face, -1)
    face = np.expand_dims(face, 0)
    return face


# ---------------- GENERATE FRAMES ----------------
def generate_frames():
    global camera, emotion_list, is_recording

    while is_recording:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]

            try:
                prediction = model.predict(preprocess(face), verbose=0)
                label = classes[np.argmax(prediction)]
                emotion_list.append(label)
            except:
                label = "Error"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ---------------- ANALYZE EMOTIONS ----------------
def analyze_emotions(emotions):

    report_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path = os.path.join(REPORT_FOLDER, report_name)

    if len(emotions) == 0:
        with open(report_path, "w") as f:
            f.write("No Face Detected\n")
        return report_name, "No Face Detected", {}

    counts = Counter(emotions)
    total = len(emotions)

    percentages = {}
    for emotion in classes:
        percentages[emotion] = (counts.get(emotion, 0) / total) * 100

    # -------- Valence Analysis --------
    positive = percentages["Happy"] + percentages["Surprise"]
    low_valence = (
        percentages["Sad"] +
        percentages["Angry"] +
        percentages["Fear"] +
        percentages["Disgust"]
    )

    if low_valence > positive:
        overall_mood = "LOW VALENCE"
    elif positive > low_valence:
        overall_mood = "HIGH VALENCE"
    else:
        overall_mood = "BALANCED"

    # -------- Final Emotion Decision --------
    if percentages["Neutral"] > 80:
        final_emotion = "Neutral"
    else:
        if overall_mood == "LOW VALENCE":
            low_emotions = ["Sad", "Angry", "Fear", "Disgust"]
            final_emotion = max(low_emotions, key=lambda e: percentages[e])

        elif overall_mood == "HIGH VALENCE":
            high_emotions = ["Happy", "Surprise"]
            final_emotion = max(high_emotions, key=lambda e: percentages[e])

        else:
            final_emotion = max(percentages, key=percentages.get)

    stability_score = percentages[final_emotion]

    # -------- CREATE REPORT --------
    with open(report_path, "w") as f:
        f.write("Emotion Analysis Report\n\n")

        for k, v in percentages.items():
            f.write(f"{k}: {v:.2f}%\n")

        f.write("\nOverall Mood: " + overall_mood + "\n")
        f.write("Final Emotion: " + final_emotion + "\n")
        f.write(f"Stability Score: {stability_score:.2f}%\n")

        if stability_score > 50:
            f.write("Emotion is STABLE\n")
        else:
            f.write("Emotion is UNSTABLE\n")

    return report_name, final_emotion, percentages


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("dashboard.html")


@app.route("/live")
def live():
    return render_template("live.html")


@app.route("/start")
def start():
    global camera, emotion_list, is_recording

    emotion_list = []
    camera = cv2.VideoCapture(0)
    is_recording = True

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/stop")
def stop():
    global camera, is_recording

    is_recording = False
    if camera:
        camera.release()

    return "Stopped"


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["video"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    cap = cv2.VideoCapture(filepath)
    emotions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]

            try:
                prediction = model.predict(preprocess(face), verbose=0)
                label = classes[np.argmax(prediction)]
                emotions.append(label)
            except:
                pass

    cap.release()

    report, final, data = analyze_emotions(emotions)

    return render_template("result.html",
                           emotion=final,
                           report=report,
                           data=data)


@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(REPORT_FOLDER, filename),
                     as_attachment=True)

@app.route("/get_result")
def get_result():
    global emotion_list

    report, final, data = analyze_emotions(emotion_list)

    return render_template("result.html",
                           emotion=final,
                           report=report,
                           data=data)
# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
