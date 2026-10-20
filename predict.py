
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path("model/stegovision_cnn.keras")
IMAGE_SIZE = (128, 128)

def predict_image(image_path):
    if not MODEL_PATH.exists():
        print("Model not found. Train the model first.")
        return

    if not Path(image_path).is_file():
        print("Image not found:", image_path)
        return

    model = tf.keras.models.load_model(MODEL_PATH)

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    probability = float(model.predict(image_array, verbose=0)[0][0])

    if probability >= 0.5:
        label = "STEGO"
        confidence = probability * 100
    else:
        label = "NORMAL"
        confidence = (1 - probability) * 100

    print("\n===== STEGOVISION RESULT =====")
    print("Image:", image_path)
    print("Prediction:", label)
    print(f"Model score: {probability * 100:.2f}% STEGO")
    print(f"Predicted-class score: {confidence:.2f}%")
    print("\nExperimental result only; not proof of hidden data.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python predict.py "image_path"')
    else:
        predict_image(sys.argv[1])
