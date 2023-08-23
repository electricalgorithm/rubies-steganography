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

        # Holds the decoded images.
        self._decoded_a: np.ndarray = None
        self._decoded_b: np.ndarray = None
    
    def decode(self, **kwargs) -> tuple[np.ndarray]:
        """It decodes the secret images from each chroma component."""
        # Crop the secret images.
        secret_image_a = self._deinsert(self._diff_a, **kwargs)
        secret_image_b = self._deinsert(self._diff_b, **kwargs)

        # Scale back the secret images.
        self._decoded_a = Utilities.scale_back_from_float64_to_uint8(secret_image_a)
        self._decoded_b = Utilities.scale_back_from_float64_to_uint8(secret_image_b)

        return (self._decoded_a, self._decoded_b)
    
    def save(self, path_a: str, path_b: str) -> None:
        """It saves the decoded images."""
        if self._decoded_a is None or self._decoded_b is None:
            raise ValueError("You should decode the images first.")
        Utilities.save_image(self._decoded_a, path_a)
        Utilities.save_image(self._decoded_b, path_b)

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
