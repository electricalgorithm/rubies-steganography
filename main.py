"""
Example usage of the encoder and decoder.
"""
from cv2 import imwrite

from core.encoder import RubiesEncoder
from core.decoder import RubiesDecoder


if __name__ == "__main__":
    # Create an encoder instance with carrier image.
    encoder = RubiesEncoder("carrier.jpg", secret_size=(500, 500))
    # Encode the secret images onto carrier.
    encoded_rgb = encoder.encode("flag_first_part.png", "flag_second_part.png")
    # Save the encoded image.
    imwrite("encoded_image.png", encoded_rgb)
    print("Encoded image saved successfully.")

    # Create a decoder instance with encoded image and carrier image.
    decoder = RubiesDecoder("encoded_image.png", original_image_path="carrier.jpg")
    # Decode the secret images from the encoded image.
    secret_image_a, secret_image_b = decoder.decode(secret_image_sizes=(500, 500))
    # Save the secret images.
    imwrite("secret_image_a.png", secret_image_a)
    imwrite("secret_image_b.png", secret_image_b)
    print("Secret images saved successfully.")
