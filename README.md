# Rubie's Steganography
This code implements the paper "Digital Image Steganography: An FFT Approach" by Tamer Rabie. It is a way to embedding two images to one carrier image.

### Installation

### Usage
Within the examples below, you will be completely understand all the functionality of `rubies` package.

#### Encode Two Images into an Image
```python
# Import cv2.imwrite to save the image.
from cv2 import imwrite

# Import Rubie's Encoder.
from rubies import Encoder

# Create an encoder instance with carrier image.
encoder = Encoder("carrier_image.png", secret_size=(500, 500))

# Encode the secret images onto carrier.
encoded_image = encoder.encode("secret_image_a.png", "secret_image_b.png")

# Save the encoded image.
imwrite("encoded_image.png", encoded_image)
```

#### Decode an Image with Unknown Secret Size

```python
# Import cv2.imwrite to save the image.
from cv2 import imwrite

# Import Rubie's Decoder.
from rubies import AutoDecoder

# Create a decoder instance with encoded image and carrier image.
decoder = AutoDecoder("encoded_image.png")

# Decode the secret images from the encoded image.
secret_image_a, secret_image_b = decoder.decode()

# Save the secret images.
imwrite("secret_image_a.png", secret_image_a)
imwrite("secret_image_b.png", secret_image_b)
```

#### Decode an Image with Known Secret Size

```python
# Import cv2.imwrite to save the image.
from cv2 import imwrite

# Import Rubie's Decoder.
from rubies import SimpleDecoder

# Create a decoder instance with encoded image and carrier image.
decoder = SimpleDecoder("encoded_image.png", secret_image_sizes=(500, 500))

# Decode the secret images from the encoded image.
secret_image_a, secret_image_b = decoder.decode()

# Save the secret images.
imwrite("secret_image_a.png", secret_image_a)
imwrite("secret_image_b.png", secret_image_b)
```
