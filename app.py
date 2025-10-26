# Importing modules
import numpy as np
import streamlit as st
import cv2
import pandas as pd
from collections import Counter
import tempfile
import base64
import os

# Load CSV
try:
    df = pd.read_csv("muse_v3.csv")
except FileNotFoundError:
    st.error("muse_v3.csv not found in the project directory. Place it next to app.py.")
    st.stop()

df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']
df = df[['name', 'emotional', 'pleasant', 'link', 'artist']]

df = df.sort_values(by=["emotional", "pleasant"])
df.reset_index()
df_sad = df[:18000]
df_fear = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]

def fun(list):
    data = pd.DataFrame()
    if len(list) == 1:
        v = list[0]
        t = 30
        if v == 'Neutral':
            data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
        elif v == 'Angry':
            data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
        elif v == 'fear':
            data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
        elif v == 'happy':
            data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
        else:
            data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)

    elif len(list) == 2:
        times = [30, 20]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:
                data = pd.concat([df_sad.sample(n=t)])

    elif len(list) == 3:
        times = [55, 20, 15]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:
                data = pd.concat([df_sad.sample(n=t)])

    elif len(list) == 4:
        times = [30, 29, 18, 9]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:
                data = pd.concat([df_sad.sample(n=t)])
    else:
        times = [10, 7, 6, 5, 2]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:
                data = pd.concat([df_sad.sample(n=t)])
    return data

def pre(l):
    emotion_counts = Counter(l)
    result = []
    for emotion, count in emotion_counts.items():
        result.extend([emotion] * count)

    ul = []
    for x in result:
        if x not in ul:
            ul.append(x)
    return ul

# Model will be created and loaded lazily to avoid import-time TensorFlow errors
# so the Streamlit UI can start even if TensorFlow isn't compatible with the
# current Python interpreter. The model is built only when the user starts
# scanning.
model = None
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


def load_model():
    """Lazily import TensorFlow, build the model architecture, and load weights.
    Raises RuntimeError with a helpful message if TensorFlow import fails or
    if weights can't be loaded.
    """
    global model
    if model is not None:
        return model

    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
    except Exception as e:
        raise RuntimeError(
            "Failed to import TensorFlow. This often means the installed Python "
            "version and TensorFlow wheel are incompatible, or required redistributables "
            "are missing. See the README for recommended Python versions. Original error: " + str(e)
        ) from e

    m = Sequential()
    m.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    m.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.25))
    m.add(Flatten())
    m.add(Dense(1024, activation='relu'))
    m.add(Dropout(0.5))
    m.add(Dense(7, activation='softmax'))

    weights_path = 'model.h5'
    if not os.path.exists(weights_path):
        raise FileNotFoundError("model.h5 not found in project root. Place it next to app.py.")
    try:
        m.load_weights(weights_path)
    except Exception as e:
        raise RuntimeError("Failed to load model weights from model.h5: " + str(e)) from e

    model = m
    return model

cv2.ocl.setUseOpenCL(False)
cap = None

# Streamlit UI
page_bg_img = """
<style>
body {
        background-color: #0f1724; /* darker, neutral background for contrast */
}
.card {
    background: linear-gradient(180deg,#1f2937,#111827);
    padding: 12px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    color: #e5e7eb;
    margin-bottom: 12px;
}
.badge {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    background: #118217;
    color: #f8fafc;
    margin-right: 8px;
    font-weight: 600;
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black'>Emotion-Based Music Recommendation</h1>", unsafe_allow_html=True)
st.write('<p style="text-align: center; colour: black">Detect your facial emotion from webcam or a short video and get mood-matching song suggestions.</p>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.title("Controls")
st.sidebar.info("Choose input source and start scanning to detect emotions")
source = st.sidebar.radio("Input source:", options=["Webcam", "Upload video"], index=0)
uploaded_file = None
if source == "Upload video":
    uploaded_file = st.sidebar.file_uploader("Upload a video file (mp4, avi)", type=["mp4", "avi", "mov"])

scan = st.sidebar.button('Scan Emotion')

# Area placeholders
frame_slot = st.empty()
progress_slot = st.sidebar.empty()
result_slot = st.container()

detected_list = []

if scan:
    detected_list.clear()
    # Open source
    if source == "Webcam":
        if "streamlit_app" in os.environ:  # Detect Streamlit Cloud
            st.info("Using browser camera (Streamlit Cloud mode)")
            img_file_buffer = st.camera_input("Take a picture")
            if img_file_buffer is not None:
                bytes_data = img_file_buffer.getvalue()
                np_img = np.frombuffer(bytes_data, np.uint8)
                frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                cap = None  # No live video stream
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                detected_list.clear()
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y + h, x:x + w]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                    model = load_model()
                    prediction = model.predict(cropped_img)
                    max_index = int(np.argmax(prediction))
                    detected_list.append(emotion_dict[max_index])
                st.success(f"Detected emotion: {', '.join(detected_list) if detected_list else 'None'}")
        else:
            st.info("Using system webcam (local mode)")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.sidebar.warning("Could not open webcam. Try 'Upload video' instead.")
                cap = None

    # Load model now (lazy) so the UI can show even if TF import fails earlier.
    try:
        model = load_model()
    except Exception as e:
        st.error(str(e))
        st.stop()

    # If upload provided, save to temp file and open
    temp_video_path = None
    if uploaded_file is not None:
        temp_dir = tempfile.gettempdir()
        temp_video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        cap = cv2.VideoCapture(temp_video_path)

    if cap is None:
        st.error("No video source available. Aborting scan.")
    else:
        # Prefer the local haarcascade file if present in the repo root.
        cascade_path = "haarcascade_frontalface_default.xml" if os.path.exists("haarcascade_frontalface_default.xml") else (cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        face_cascade = cv2.CascadeClassifier(cascade_path)
        count = 0
        max_frames = 60  # process up to 60 frames or until video ends
        progress = progress_slot.progress(0)
        while count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                try:
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                    prediction = model.predict(cropped_img)
                    max_index = int(np.argmax(prediction))
                    detected_list.append(emotion_dict[max_index])
                    cv2.putText(frame, emotion_dict[max_index], (x + 20, y - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                except Exception:
                    # skip faces too small or processing errors
                    pass
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_slot.image(frame_rgb, channels='RGB')
            count += 1
            progress.progress(int(count / max_frames * 100))

        cap.release()
        # Remove temp file if used
        if temp_video_path is not None and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass

        detected_list = pre(detected_list)
        # with result_slot:
        if len(detected_list) == 0:
            st.warning("No faces/emotions detected.")
        else:
            st.success("Emotions successfully detected!")
            st.write("### Detected Emotion(s):")
            badges = "".join([f"<span class='badge'>{e}</span>" for e in detected_list])
            st.markdown(badges, unsafe_allow_html=True)
                # st.write(", ".join(detected_list))

new_df = fun(detected_list)

st.markdown("<h5 style='text-align: center; color: grey;'><b>Recommended song's with artist names</b></h5>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'><b>Click on the name of recommended song to reach website</b></h5>", unsafe_allow_html=True)
st.write("---------------------------------------------------------------------------------------------------------------------")

try:
    for i, row in new_df.head(30).iterrows():
        link = row.get('link', '#')
        artist = row.get('artist', '')
        name = row.get('name', 'Unknown')
        card_html = f"""
        <div class='card'>
          <div style='display:flex; justify-content:space-between; align-items:center'>
            <div>
              <a href='{link}' target='_blank' style='color:#e5e7eb; text-decoration:none; font-size:18px;'>{i+1}. {name}</a>
              <div style='color:#9ca3af; font-size:13px;'>{artist}</div>
            </div>
          </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
except Exception:
    pass
