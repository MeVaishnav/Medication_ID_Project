# src/train_model.py

import tensorflow as tf
from keras import layers, models
import os

# ---------------------------
# Paths
# ---------------------------
train_dir = "data/train"
test_dir = "data/test"

# ---------------------------
# Image Parameters
# ---------------------------
img_height, img_width = 180, 180
batch_size = 16

# ---------------------------
# Load Dataset
# ---------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Save class names BEFORE mapping
class_names = train_ds.class_names

# Normalize pixel values to [0,1]
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
test_ds = test_ds.map(lambda x, y: (x / 255.0, y))


# ---------------------------
# Model Architecture (CNN)
# ---------------------------
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])

# ---------------------------
# Compile Model
# ---------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------------------
# Train Model
# ---------------------------
print("🚀 Training model...")
history = model.fit(train_ds, validation_data=test_ds, epochs=10)

# ---------------------------
# Save Model
# ---------------------------
os.makedirs("models", exist_ok=True)
model.save("models/medicine_identifier.h5")
print("✅ Model saved to models/medicine_identifier.h5")

# ---------------------------
# Evaluate Model
# ---------------------------
test_loss, test_acc = model.evaluate(test_ds)
print(f"📊 Test accuracy: {test_acc:.2f}")

# ---------------------------
# Print Class Labels
# ---------------------------
print("Medicine Classes:",class_names)
