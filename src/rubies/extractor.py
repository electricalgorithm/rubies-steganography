import cv2
import numpy as np

from .utils import Utilities


class ImageExtractor:
    """This class is used to extract the image from the given black backgrounded image."""

    def __init__(self, image: np.ndarray) -> None:
        """Initializes the image extractor."""
        copied_image, half_width = self._read_image(image)
        cleared_bg = self._clear_background(copied_image)
        contour_mask = self._find_contour(cleared_bg, copied_image)
        min_x, max_x, min_y, max_y = self._calculate_edge_points(contour_mask)
        self._image = self._crop_image(
            image, min_x + half_width, max_x + half_width, min_y, max_y
        )

    def extract(self) -> np.ndarray:
        """Returns the extracted image."""
        return self._image

    @staticmethod
    def _read_image(image: np.ndarray) -> tuple[np.ndarray, int]:
        """Reads the image and returns it as a greyscale half right image."""
        # Convert to uin8.
        copied_image = Utilities.scale_back_from_float64_to_uint8(image.copy())
        # Convert the image to grayscale.
        if len(copied_image.shape) == 3:
            copied_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Crop the image only the right half of it and save its half width.
        half_width = copied_image.shape[1] // 2
        copied_image = copied_image[:, half_width:]
        return copied_image, half_width

    @staticmethod
    def _clear_background(image: np.ndarray) -> np.ndarray:
        """Clears the background of the image and returns it as a more cleaner backgrounded."""
        _, binary_mask = cv2.threshold(
            image, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV
        )
        # Erode the binary mask
        kernel = np.ones((15, 15), np.uint8)
        return cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    @staticmethod
    def _find_contour(mask: np.ndarray, gray_image: np.ndarray) -> np.ndarray:
        """Finds the contour of the image and returns it as a contour."""
        # Find contours in the eroded mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Sort the contours by area in descending order
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        # Find the biggest contour
        if len(contours_sorted) > 0:
            contour_to_show = contours_sorted[0]
            # Create a mask for the biggest contour
            contour_mask = np.bitwise_not(gray_image.copy())
            # Draw the biggest contour on the mask
            cv2.drawContours(contour_mask, [contour_to_show], 0, (0, 255, 0), -1)
            # Dilate the contour mask
            kernel = np.ones((100, 100), np.uint8)
            contour_mask = cv2.erode(contour_mask, kernel, iterations=1)
            contour_mask = cv2.dilate(contour_mask, kernel, iterations=1)
            # Apply binarization
            _, inv_contour_mask = cv2.threshold(
                contour_mask, 10, 255, cv2.THRESH_BINARY
            )

            return inv_contour_mask
        raise ValueError("No contour found.")

    @staticmethod
    def _calculate_edge_points(contour_mask: np.ndarray) -> tuple[int, int, int, int]:
        """Calculates the edge points of the image and returns it as a numpy array."""
        # Dilate one pixel, then substract the dilated image from the original mask to
        # obtain the inner boundary of the mask
        kernel = np.ones((1, 1), np.uint8)
        contour_mask_inner = cv2.dilate(contour_mask, kernel, iterations=1)
        contour_mask_inner = cv2.bitwise_not(contour_mask_inner)
        contour_mask_inner = cv2.subtract(contour_mask, contour_mask_inner)
        # Use Canny edge detection to find the edges of the inner boundary.
        edges = cv2.Canny(contour_mask_inner, 100, 200)
        # Find the coordinates of all the edge points
        edge_points = np.where(edges == 255)
        edge_points = np.array(edge_points).T

        # Find the minimum and maximum x and y coordinates
        min_x = np.min(edge_points[:, 1])
        max_x = np.max(edge_points[:, 1])
        min_y = np.min(edge_points[:, 0])
        max_y = np.max(edge_points[:, 0])
        return (min_x, max_x, min_y, max_y)

    @staticmethod
    def _crop_image(
        image: np.ndarray, min_x: int, max_x: int, min_y: int, max_y: int
    ) -> np.ndarray:
        """Crops the image and returns it as a cropped image."""
        return image[min_y:max_y, min_x:max_x]
