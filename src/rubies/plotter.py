"""
Plotter class for plotting data.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2


def show_3d_magnitude_spectrum(magnitude_spectrum, title="Magnitude Spectrum"):
    """3D magnitude spectrum plotter."""
    fig = plt.figure(figsize=(8, 6))
    mag_plot = fig.add_subplot(111, projection="3d")

    x_vals = np.arange(0, magnitude_spectrum.shape[1], 1)
    y_vals = np.arange(0, magnitude_spectrum.shape[0], 1)
    x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)

    db_spectrum = np.log10(1 + np.abs(magnitude_spectrum))

    mag_plot.plot_surface(x_mesh, y_mesh, db_spectrum, cmap="viridis")
    mag_plot.set_title(title)
    mag_plot.set_xlabel("X")
    mag_plot.set_ylabel("Y")
    mag_plot.set_zlabel("Magnitude (log(1+abs(mag)))")
    plt.show()


def show_difference_in_3d(
    original_version,
    modified_version,
    original_title="Original Magnitude Spectrum",
    modified_title="Magnitude Spectrum after IFFT",
):
    """Chrome-a and chrome-b magnitude spectrum plotter."""
    figure_window = plt.figure(figsize=(15, 6))

    original_plot = figure_window.add_subplot(121, projection="3d")
    x1_mesh, y1_mesh = np.meshgrid(
        np.arange(original_version.shape[1]), np.arange(original_version.shape[0])
    )
    original_plot.plot_surface(
        x1_mesh, y1_mesh, np.log(1 + np.abs(original_version)), cmap="viridis"
    )
    original_plot.set_title(original_title)
    original_plot.set_xlabel("X")
    original_plot.set_ylabel("Y")
    original_plot.set_zlabel("Magnitude (log(1+abs(mag)))")

    modified_plot = figure_window.add_subplot(122, projection="3d")
    x2_mesh, y2_mesh = np.meshgrid(
        np.arange(modified_version.shape[1]), np.arange(modified_version.shape[0])
    )
    modified_plot.plot_surface(
        x2_mesh, y2_mesh, np.log(1 + np.abs(modified_version)), cmap="viridis"
    )
    modified_plot.set_title(modified_title)
    modified_plot.set_xlabel("X")
    modified_plot.set_ylabel("Y")
    modified_plot.set_zlabel("Magnitude (log(1+abs(mag)))")

    plt.tight_layout()
    plt.show()


def show_image(title, image, ifft=False, phase=None):
    """Image shower."""
    plt.figure()
    plt.title(title)
    if ifft:
        image_to_show = np.abs(np.fft.ifft2(image * np.exp(1j * phase))).astype(
            np.uint8
        )
    else:
        image_to_show = image
    rgb_image = cv2.cvtColor(image_to_show, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image)
    plt.axis("off")
    plt.show()
