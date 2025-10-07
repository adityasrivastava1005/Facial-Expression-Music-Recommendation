# Importing modules
import numpy as np
import streamlit as st
import cv2
import pandas as pd
import tensorflow
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
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

# Load model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

try:
    model.load_weights('model.h5')
except Exception as e:
    st.error("model.h5 not found or could not be loaded. Place it next to app.py.")
    st.stop()
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

cv2.ocl.setUseOpenCL(False)
cap = None

# Streamlit UI
page_bg_img = '''
<style>
body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white'><b>Emotion based music recommendation</b></h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'><b>Click on the name of recommended song to reach website</b></h5>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

list = []
with col2:
    st.info("Choose input source and then click 'SCAN EMOTION' to start detection")
    source = st.radio("Input source:", options=["Webcam", "Upload video"], index=0)
    uploaded_file = None
    if source == "Upload video":
        uploaded_file = st.file_uploader("Upload a video file (mp4, avi)", type=["mp4", "avi", "mov"])

    if st.button('SCAN EMOTION(Click here)'):
        list.clear()
        # Try opening webcam if chosen
        if source == "Webcam":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.warning("Could not open webcam. Try 'Upload video' instead.")
                cap = None

        # If upload provided, save to temp file and open
        temp_video_path = None
        if uploaded_file is not None:
            temp_video_path = os.path.join("/tmp", uploaded_file.name)
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            cap = cv2.VideoCapture(temp_video_path)

        if cap is None:
            st.error("No video source available. Aborting scan.")
        else:
            frame_slot = st.image([])
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            count = 0
            max_frames = 60  # process up to 60 frames or until video ends
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
                        list.append(emotion_dict[max_index])
                        cv2.putText(frame, emotion_dict[max_index], (x + 20, y - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    except Exception:
                        # skip faces too small or processing errors
                        pass
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_slot.image(frame_rgb, channels='RGB')
                count += 1

            cap.release()
            # Remove temp file if used
            if temp_video_path is not None and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except Exception:
                    pass

            list = pre(list)
            if len(list) == 0:
                st.warning("No faces/emotions detected.")
            else:
                st.success("Emotions successfully detected!")
                st.write("### Detected Emotion(s):")
                st.write(", ".join(list))

new_df = fun(list)
st.write("")

st.markdown("<h5 style='text-align: center; color: grey;'><b>Recommended song's with artist names</b></h5>", unsafe_allow_html=True)
st.write("---------------------------------------------------------------------------------------------------------------------")

try:
    for l, a, n, i in zip(new_df["link"], new_df['artist'], new_df['name'], range(30)):
        st.markdown(f"<h4 style='text-align: center;'><a href={l}>{i+1}. {n}</a></h4>", unsafe_allow_html=True)
        st.markdown(f"<h5 style='text-align: center; color: grey;'><i>{a}</i></h5>", unsafe_allow_html=True)
        st.write("---------------------------------------------------------------------------------------------------------------------")
except:
    pass
