"""
This module contains the Encoder class, which is responsible for encoding
images into one carrier image.
"""
import numpy as np
from .utils import Utilities


class Encoder:
    """It handles the encoding functionality."""

    def __init__(self, carrier_image: str | np.ndarray, secret_size: tuple) -> None:
        """It initializes the class."""
        self.secret_size = secret_size
        # Read the carrier image.
        self._image = Utilities.read_image(carrier_image)
        # Get the LAB components of the image.
        self._lum, _ch_a, _ch_b = Utilities.split_to_lab_components(self._image)
        self._ch_a_mag = Utilities.get_magnitude(_ch_a)
        self._ch_b_mag = Utilities.get_magnitude(_ch_b)
        self._ch_a_phase = Utilities.get_phase(_ch_a)
        self._ch_b_phase = Utilities.get_phase(_ch_b)
        # Store the encoded image.
        self._encoded_image: np.ndarray = None

    def encode(self, embed_img_a: str | np.ndarray, embed_img_b: str | np.ndarray) -> np.ndarray:
        """It encodes the images embedded into the carrier image."""
        # Recieve images.
        image_a = Utilities.read_image(embed_img_a, size=self.secret_size)
        image_b = Utilities.read_image(embed_img_b, size=self.secret_size)
        image_a = Utilities.rgb_to_gray(image_a)
        image_b = Utilities.rgb_to_gray(image_b)

        # Embed the images.
        _new_ch_a_mag = self._embed_onto("a", image_a)
        _new_ch_b_mag = self._embed_onto("b", image_b)

        # Create chroma components from the modified magnitude and phase.
        chrom_a = Utilities.ifft(_new_ch_a_mag, self._ch_a_phase)
        chrom_b = Utilities.ifft(_new_ch_b_mag, self._ch_b_phase)

        # Create the RGB image from LAB components.
        encoded_lab = Utilities.create_lab_from_complex(self._lum, chrom_a, chrom_b)
        self._encoded_image = Utilities.lab_to_rgb(encoded_lab)
        return self._encoded_image
    
    def save(self, path: str) -> None:
        """It saves the encoded image."""
        if self._encoded_image is None:
            raise ValueError("You should encode the images first.")
        Utilities.save_image(self._encoded_image, path)

    def _embed_onto(self, chroma_x: str, secret_image: np.ndarray) -> np.ndarray:
        """It embeds one image into carrier."""
        if chroma_x == "a":
            chroma_mag = self._ch_a_mag
        elif chroma_x == "b":
            chroma_mag = self._ch_b_mag
        else:
            raise ValueError("Invalid chroma component.")

        # Copy the magnitude spectum.
        new_chroma_mag = chroma_mag.copy()

        # Embed the secret image by adding on top of the freq magnitudes.
        return self._insert(new_chroma_mag, secret_image)

    @staticmethod
    def _insert(new_chroma_mags: np.ndarray, secret_image: np.ndarray) -> np.ndarray:
        """This is the logic where insertation has done."""
        # Calculate the padding.
        v_pad, h_pad = Utilities.calculate_padding(new_chroma_mags, secret_image)

        # Check if the secret image is too big to embed.
        if v_pad < 0 or h_pad < 0:
            raise ValueError("Secret image is too big to embed.")

        # Embed the secret image by adding on top of the freq magnitudes.
        secret_image = secret_image.astype(np.float64)
        secret_image *= 100
        # Add luminance to the secret image since
        # decoder will eliminate some.
        secret_image += 5000

        new_chroma_mags[
            v_pad : (v_pad + secret_image.shape[0]),
            h_pad - 1 : -1,
        ] += secret_image

        return new_chroma_mags
