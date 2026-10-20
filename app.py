
from pathlib import Path
from flask import Flask, render_template, request, send_file
from PIL import Image
import io
import secrets

from steganography import encode_image, decode_image

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = BASE_DIR / "model" / "stegovision_cnn.keras"

# Limit uploads to 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/encode", methods=["GET", "POST"])
def encode():
    if request.method == "GET":
        return render_template("encode.html")

    image_file = request.files.get("image")
    message = request.form.get("message", "").strip()

    if not image_file or not image_file.filename:
        return render_template("encode.html", error="Please select an image.")

    if not allowed_file(image_file.filename):
        return render_template("encode.html", error="Unsupported image format.")

    if not message:
        return render_template("encode.html", error="Please enter a message.")

    input_path = UPLOAD_DIR / f"{secrets.token_hex(12)}_input.png"
    output_path = UPLOAD_DIR / f"{secrets.token_hex(12)}_stego.png"

    try:
        # Convert uploaded image to PNG for lossless storage
        image = Image.open(image_file.stream).convert("RGB")
        image.save(input_path, format="PNG")

        encode_image(str(input_path), str(output_path), message)

        return render_template(
            "encode.html",
            result="Message hidden successfully!",
            output_image="/" + output_path.relative_to(BASE_DIR).as_posix(),
            download_name=output_path.name,
        )

    except (ValueError, OSError) as exc:
        return render_template("encode.html", error=str(exc))

    finally:
        input_path.unlink(missing_ok=True)


@app.route("/download/<filename>")
def download(filename):
    # Only allow generated stego images from the uploads folder
    if Path(filename).name != filename or not filename.endswith("_stego.png"):
        return "Invalid filename", 400

    file_path = UPLOAD_DIR / filename

    if not file_path.is_file():
        return "File not found", 404

    return send_file(file_path, as_attachment=True, download_name=filename)


@app.route("/decode", methods=["GET", "POST"])
def decode():
    if request.method == "GET":
        return render_template("decode.html")

    image_file = request.files.get("image")

    if not image_file or not image_file.filename:
        return render_template("decode.html", error="Please select a stego image.")

    if not allowed_file(image_file.filename):
        return render_template("decode.html", error="Unsupported image format.")

    try:
        image = Image.open(image_file.stream).convert("RGB")

        # Decode directly from the uploaded image
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        temporary_path = UPLOAD_DIR / f"{secrets.token_hex(12)}_decode.png"
        try:
            image.save(temporary_path, format="PNG")
            message = decode_image(str(temporary_path))
        finally:
            temporary_path.unlink(missing_ok=True)

        return render_template("decode.html", message=message)

    except (ValueError, OSError) as exc:
        return render_template("decode.html", error=str(exc))


@app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method == "GET":
        return render_template("detect.html")

    image_file = request.files.get("image")

    if not image_file or not image_file.filename:
        return render_template("detect.html", error="Please select an image.")

    if not allowed_file(image_file.filename):
        return render_template("detect.html", error="Unsupported image format.")

    if not MODEL_PATH.is_file():
        return render_template("detect.html", error="Train the CNN model first.")

    try:
        import numpy as np
        import tensorflow as tf

        image = Image.open(image_file.stream).convert("RGB")
        image = image.resize((128, 128))

        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        model = tf.keras.models.load_model(MODEL_PATH)
        score = float(model.predict(image_array, verbose=0)[0][0])

        prediction = "STEGO" if score >= 0.5 else "NORMAL"
        confidence = score * 100 if score >= 0.5 else (1 - score) * 100

        return render_template(
            "detect.html",
            prediction=prediction,
            score=f"{score * 100:.2f}",
            confidence=f"{confidence:.2f}",
            warning="Experimental prediction only; not proof of hidden data.",
        )

    except (ValueError, OSError) as exc:
        return render_template("detect.html", error=str(exc))


@app.errorhandler(413)
def file_too_large(_error):
    return "Image is too large. Maximum size is 10 MB.", 413


if __name__ == "__main__":
    app.run(debug=True)
