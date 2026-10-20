from PIL import Image

DELIMITER = "1111111111111110"

def text_to_binary(text):
    return ''.join(format(ord(char), '08b') for char in text)

def binary_to_text(binary):
    chars = []
    for i in range(0, len(binary), 8):
        byte = binary[i:i+8]
        if len(byte) == 8:
            chars.append(chr(int(byte, 2)))
    return ''.join(chars)

def encode_image(input_path, output_path, message):
    image = Image.open(input_path).convert("RGB")
    pixels = list(image.getdata())

    binary_message = text_to_binary(message) + DELIMITER

    if len(binary_message) > len(pixels) * 3:
        raise ValueError("Message is too large for this image.")

    new_pixels = []
    bit_index = 0

    for pixel in pixels:
        new_pixel = list(pixel)

        for channel in range(3):
            if bit_index < len(binary_message):
                new_pixel[channel] = (
                    new_pixel[channel] & 254
                ) | int(binary_message[bit_index])
                bit_index += 1

        new_pixels.append(tuple(new_pixel))

    stego_image = Image.new("RGB", image.size)
    stego_image.putdata(new_pixels)
    stego_image.save(output_path)

def decode_image(image_path):
    image = Image.open(image_path).convert("RGB")
    pixels = list(image.getdata())

    binary_data = ""

    for pixel in pixels:
        for channel in pixel:
            binary_data += str(channel & 1)

    message_binary = binary_data.split(DELIMITER)[0]

    if not message_binary:
        return "No hidden message found."

    return binary_to_text(message_binary)

if __name__ == "__main__":
    input_image = "sample.jpg.png"
    output_image = "stego_image.png"
    secret_message = "Hello from StegoVision!"

    encode_image(input_image, output_image, secret_message)

    print("Message hidden successfully.")
    print("Extracted message:")
    print(decode_image(output_image))