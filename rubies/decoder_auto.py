"""
This module contains the Decoder class, which is responsible for decoding
secret images from one image.
"""
import numpy as np
from .extractor import ImageExtractor
from .decoder_base import RubiesDecoder


class AutoDecoder(RubiesDecoder):
    """It handles the decoding functionality without original image itself or secret image size."""

    @staticmethod
    def _deinsert(whole_image) -> np.ndarray:
        """It crops the secret images from the whole image."""
        return ImageExtractor(whole_image).extract()
