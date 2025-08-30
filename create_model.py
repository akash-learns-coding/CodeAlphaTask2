import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Create a simple dummy model
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(10,)),  # input with 10 features
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Save model
model.save("saved_model.h5")

print("✅ Dummy model saved as saved_model.h5")
