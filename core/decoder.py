"""
This module contains the Decoder class, which is responsible for decoding
secret images from one image.
"""
import numpy as np
from core.utils import Utilities


class RubiesDecoder:
    """It handles the decoding functionality."""

    def __init__(self, encoded_image_path: str, original_image_path: str) -> None:
        self._original_image = Utilities.read_image(original_image_path)
        self._encoded_image = Utilities.read_image(encoded_image_path)

        # Get the LAB components of images.
        _, _orig_a, _orig_b = Utilities.split_to_lab_components(self._original_image)
        _, _enc_a, _enc_b = Utilities.split_to_lab_components(self._encoded_image)

        # Get the magnitude of chroma components.
        _orig_a_mag = Utilities.get_magnitude(_orig_a)
        _orig_b_mag = Utilities.get_magnitude(_orig_b)
        _enc_a_mag = Utilities.get_magnitude(_enc_a)
        _enc_b_mag = Utilities.get_magnitude(_enc_b)

        # Extract the difference.
        self._diff_a = self._extract_difference(_enc_a_mag, _orig_a_mag)
        self._diff_b = self._extract_difference(_enc_b_mag, _orig_b_mag)

    def decode(self, secret_image_sizes: tuple[int]) -> tuple[np.ndarray]:
        """It decodes the secret images from each chroma component."""
        # Crop the secret images.
        secret_image_a = self._crop(self._diff_a, secret_image_sizes)
        secret_image_b = self._crop(self._diff_b, secret_image_sizes)

        # Scale back the secret images.
        secret_image_a = Utilities.scale_back_from_float64_to_uint8(secret_image_a)
        secret_image_b = Utilities.scale_back_from_float64_to_uint8(secret_image_b)

        return (secret_image_a, secret_image_b)

    @staticmethod
    def _extract_difference(modified_mag, original_mag) -> np.ndarray:
        """It extracts the difference from the original and modified
        magnitudes."""
        return modified_mag - original_mag

    @staticmethod
    def _crop(whole_image, secret_image_sizes) -> np.ndarray:
        """It crops the secret images from the whole image."""
        v_pad = (whole_image.shape[0] - secret_image_sizes[0]) // 2
        h_pad = (whole_image.shape[1] - secret_image_sizes[1]) // 2
        return whole_image[
            v_pad : v_pad + secret_image_sizes[0], h_pad : h_pad + secret_image_sizes[1]
        ]
