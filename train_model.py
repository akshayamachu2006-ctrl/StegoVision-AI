
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATASET_DIR = Path("dataset")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 4
EPOCHS = 10
SEED = 42

# Load normal and stego images from separate folders
train_data = keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["normal", "stego"],
    validation_split=0.25,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

val_data = keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["normal", "stego"],
    validation_split=0.25,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

# Normalize pixel values from 0–255 to 0–1
normalization = layers.Rescaling(1.0 / 255)

train_data = train_data.map(
    lambda images, labels: (normalization(images), labels)
)
val_data = val_data.map(
    lambda images, labels: (normalization(images), labels)
)

# Build the CNN model
model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# Train the model
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
)

# Save the trained model
model.save(MODEL_DIR / "stegovision_cnn.keras")

print("\nTraining complete!")
print("Model saved to model/stegovision_cnn.keras")
print("Classes: 0 = NORMAL, 1 = STEGO")
