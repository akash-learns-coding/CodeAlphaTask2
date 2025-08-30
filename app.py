import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from st_audiorec import st_audiorec  # <- Updated import

# Load trained model
model = tf.keras.models.load_model("saved_model.h5")

# Feature extraction function
def extract_features(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

# Streamlit UI
st.title("🎤 Real-Time Speech Emotion Recognition")
st.write("Click below to record your voice and detect emotion.")

# Record audio
audio_bytes = st_audiorec()  # returns audio as bytes

if audio_bytes is not None:
    # Save audio as WAV
    wav_file = "recorded.wav"
    with open(wav_file, "wb") as f:
        f.write(audio_bytes)

    # Load audio
    y, sr = librosa.load(wav_file, sr=None)
    features = extract_features(y, sr)
    features = features.reshape(1, -1)

    # Prediction
    prediction = model.predict(features)
    emotion = np.argmax(prediction, axis=1)

    emotion_labels = ["Angry", "Happy", "Sad", "Neutral"]
    st.success(f"Predicted Emotion: **{emotion_labels[emotion[0]]}**")

