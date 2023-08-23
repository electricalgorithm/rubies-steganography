"""
This module contains the Decoder class, which is responsible for decoding
secret images from one image.
"""
import numpy as np
from .utils import Utilities
from .decoder_base import RubiesDecoder


class SimpleDecoder(RubiesDecoder):
    """It handles the decoding functionality with a need of secret image sizes."""

    @staticmethod
    def _deinsert(whole_image, secret_image_sizes) -> np.ndarray:
        """It crops the secret images from the whole image."""
        v_pad = (whole_image.shape[0] - secret_image_sizes[0]) // 2
        h_pad = whole_image.shape[1] - secret_image_sizes[1]
        return whole_image[v_pad : v_pad + secret_image_sizes[0], h_pad - 1 : -1] // 100