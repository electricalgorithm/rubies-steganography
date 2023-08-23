"""
This module contains the BaseDecoder class, which is responsible for decoding
secret images from one image. Other decoders inherit from this class.
"""
import numpy as np
from .utils import Utilities


class RubiesDecoder:
    """It handles the decoding functionality without original image itself or secret image size."""

    def __init__(self, encoded_image_path: str) -> None:
        self._encoded_image = Utilities.read_image(encoded_image_path)

        # Get the LAB components of images.
        _, _enc_a, _enc_b = Utilities.split_to_lab_components(self._encoded_image)

        # Get the magnitude of chroma components.
        _enc_a_mag = Utilities.get_magnitude(_enc_a)
        _enc_b_mag = Utilities.get_magnitude(_enc_b)

        # Extract the difference.
        self._diff_a = self._extract_difference(_enc_a_mag)
        self._diff_b = self._extract_difference(_enc_b_mag)
    
    def decode(self, *args) -> tuple[np.ndarray]:
        """It decodes the secret images from each chroma component."""
        # Crop the secret images.
        secret_image_a = self._deinsert(self._diff_a, *args)
        secret_image_b = self._deinsert(self._diff_b, *args)

        # Scale back the secret images.
        secret_image_a = Utilities.scale_back_from_float64_to_uint8(secret_image_a)
        secret_image_b = Utilities.scale_back_from_float64_to_uint8(secret_image_b)

        return (secret_image_a, secret_image_b)

    @staticmethod
    def _extract_difference(modified_mag) -> np.ndarray:
        """It extracts the difference from the original and modified
        magnitudes."""
        average = np.mean(modified_mag)
        to_delete = 300
        modified_mag[0:to_delete, 0:to_delete] = average
        modified_mag[-to_delete:, -to_delete:] = average
        modified_mag[0:to_delete, -to_delete:] = average
        modified_mag[-to_delete:, 0:to_delete] = average
        modified_mag[modified_mag < average] = average
        return modified_mag - np.mean(modified_mag)
    
    @staticmethod
    def _deinsert() -> np.ndarray:
        """It crops the secret images from the whole image."""
        raise NotImplementedError("This method should be implemented in the child class.")
