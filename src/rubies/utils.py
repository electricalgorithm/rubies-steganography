"""
This is a helper class that contains common functions for all the modules.
"""
import cv2
import numpy as np


class Utilities:
    """Common functions for all the modules."""

    @staticmethod
    def create_lab_from_complex(
        luminance: np.ndarray, chrom_a: np.ndarray, chrom_b: np.ndarray
    ) -> np.ndarray:
        """It creates the RGB image from LAB components."""
        real_chrom_a = np.abs(chrom_a).astype(np.uint8)
        real_chrom_b = np.abs(chrom_b).astype(np.uint8)
        return cv2.merge((luminance, real_chrom_a, real_chrom_b))

    @staticmethod
    def calculate_padding(max_elem: np.ndarray, min_elem: np.ndarray) -> list[int]:
        """It calculates the padding."""
        vertical_padding = (max_elem.shape[0] - min_elem.shape[0]) // 2
        horizontal_padding = max_elem.shape[1] - min_elem.shape[1]
        # vertical_padding = (max_elem.shape[0] - min_elem.shape[0]) // 2
        # horizontal_padding = (max_elem.shape[1] - min_elem.shape[1]) // 2
        return vertical_padding, horizontal_padding

    @staticmethod
    def split_to_lab_components(image: np.ndarray) -> list[np.ndarray]:
        """It splits the image to LAB components."""
        return cv2.split(Utilities.rgb_to_lab(image))

    @staticmethod
    def rgb_to_lab(rgb_image: np.ndarray) -> np.ndarray:
        """It converts the RGB image to LAB image."""
        return cv2.cvtColor(rgb_image, cv2.COLOR_BGR2LAB)

    @staticmethod
    def lab_to_rgb(lab_image: np.ndarray) -> np.ndarray:
        """It converts the LAB image to RGB image."""
        return cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    @staticmethod
    def rgb_to_gray(rgb_image: np.ndarray) -> np.ndarray:
        """It converts the RGB image to gray image."""
        return cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def read_image(image: str | np.ndarray, size: tuple | None = None) -> np.ndarray:
        """It reads the image."""
        if isinstance(image, str):
            # Read the carrier image.
            return_image = cv2.imread(image)
            if return_image is None:
                raise FileNotFoundError("Image not found.")
        elif isinstance(image, np.ndarray):
            return_image = image
        else:
            raise TypeError("Invalid image type.")
        if size is not None:
            return_image = cv2.resize(return_image, size)
        return return_image
    
    @staticmethod
    def save_image(image: np.ndarray, path: str) -> None:
        """It saves the image."""
        # If the image path is .jpg, raise an error.
        if path.endswith(".jpg") or path.endswith(".jpeg"):
            raise ValueError("Please use loseless image type to save.")
        cv2.imwrite(path, image)

    @staticmethod
    def ifft(magnitudes: np.ndarray, phases: np.ndarray) -> np.ndarray:
        """It returns the IFFT of the image."""
        component = np.multiply(magnitudes, np.exp(1j * phases))
        return np.fft.ifft2(component)

    @staticmethod
    def fft(component: np.ndarray) -> np.ndarray:
        """It returns the FFT of the image."""
        return np.fft.fft2(component)

    @staticmethod
    def get_phase(fft_component: np.ndarray) -> np.ndarray:
        """It returns the phase of the FFT of the image."""
        return np.angle(Utilities.fft(fft_component))

    @staticmethod
    def get_magnitude(fft_component: np.ndarray) -> np.ndarray:
        """It returns the magnitude of the FFT of the image."""
        return np.abs(Utilities.fft(fft_component))

    @staticmethod
    def scale_back_from_float64_to_uint8(image: np.ndarray) -> np.ndarray:
        """It scales back the image from float64 to uint8."""
        # Scale the image np.ndarray from float64 to uint8 with using
        # peak to peak normalization.
        double_version = (255 * (image - np.min(image))) / np.ptp(image)
        return double_version.astype(np.uint8)
