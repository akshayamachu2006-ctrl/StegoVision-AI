
from pathlib import Path
from PIL import Image
import random

NORMAL_DIR = Path("dataset/normal")
STEGO_DIR = Path("dataset/stego")

STEGO_DIR.mkdir(parents=True, exist_ok=True)

def text_to_binary(text):
    data = text.encode("utf-8")
    return "".join(format(byte, "08b") for byte in data)

def encode_image(input_path, output_path, message):
    image = Image.open(input_path).convert("RGB")
    pixels = list(image.getdata())

    # Store message length first, so extraction is reliable
    message_bits = text_to_binary(message)
    header = format(len(message_bits), "032b")
    bits = header + message_bits

    if len(bits) > len(pixels) * 3:
        raise ValueError("Image is too small for this message")

    new_pixels = []
    bit_index = 0

    for pixel in pixels:
        channels = list(pixel)

        for channel in range(3):
            if bit_index < len(bits):
                channels[channel] = (
                    channels[channel] & 254
                ) | int(bits[bit_index])
                bit_index += 1

        new_pixels.append(tuple(channels))

    result = Image.new("RGB", image.size)
    result.putdata(new_pixels)
    result.save(output_path, format="PNG")

def main():
    images = [
        p for p in NORMAL_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]

    if not images:
        print("No images found in dataset/normal!")
        return

    created = 0

    for index, image_path in enumerate(images, start=1):
        try:
            message = f"StegoVision secret message {random.randint(1000, 9999)}"
            output_path = STEGO_DIR / f"stego_{image_path.stem}.png"

            encode_image(image_path, output_path, message)

            print(f"{index}. Created: {output_path}")
            created += 1

        except Exception as error:
            print(f"Skipped {image_path.name}: {error}")

    print(f"\nDataset generation complete: {created} stego images created.")
    print("Normal images:", len(images))
    print("Stego images:", created)

if __name__ == "__main__":
    main()
