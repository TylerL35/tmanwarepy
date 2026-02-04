import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. LOAD DATA
df = pd.read_csv('test_case.csv')

# 2. PREPROCESS
# Drop non-feature columns
X = df.drop(['target_gesture', 'session_id', 'user_id'], axis=1).values
y_raw = df['target_gesture'].values

# Encode labels (e.g., 'neutral' -> 0, 'smile' -> 1)
encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)
num_classes = len(np.unique(y))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. BUILD THE MODEL
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2), # Prevents overfitting
    tf.keras.layers.Dense(num_classes, activation='softmax') # Outputs probabilities
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. TRAIN
print("Training model...")
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 5. SAVE
model.save('gesture_model.h5')
np.save('classes.npy', encoder.classes_)
print("✅ Model saved as gesture_model.h5")