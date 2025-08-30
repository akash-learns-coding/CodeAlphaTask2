import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from utils import extract_features
from model import create_model
import tensorflow as tf


EMOTIONS = ["angry", "happy", "sad", "neutral"]   # Adjust if needed
DATASET_PATH = "dataset/"   # Folder with audio data


X, y = [], []

print(" Loading dataset...")

for i, emotion in enumerate(EMOTIONS):
    folder = os.path.join(DATASET_PATH, emotion)
    if not os.path.exists(folder):
        print(f" Folder not found: {folder}")
        continue
    
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            file_path = os.path.join(folder, file)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(i)

print(f" Loaded {len(X)} samples.")

# -------------------------------
# 3. Preprocess
# -------------------------------
X = np.array(X)
X = np.expand_dims(X, -1)   # CNN expects (samples, height, width, channels)
y = to_categorical(np.array(y), num_classes=len(EMOTIONS))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. Build & Train Model
# -------------------------------
print(" Building model...")
model = create_model((X.shape[1], X.shape[2], 1), num_classes=len(EMOTIONS))

print(" Training started...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32
)

# -------------------------------
# 5. Save trained model
# -------------------------------
model.save("saved_model.h5")
print(" Model trained and saved as saved_model.h5")
