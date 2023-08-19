import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_magnitude_spectrum(magnitude_spectrum, title="Magnitude Spectrum"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    x = np.arange(0, magnitude_spectrum.shape[1], 1)
    y = np.arange(0, magnitude_spectrum.shape[0], 1)
    X, Y = np.meshgrid(x, y)

    ax.plot_surface(X, Y, np.log(1 + magnitude_spectrum), cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Magnitude (log)")
    plt.show()


def plot_chrominance_magnitude_spectrum_3d(chrom_a_fft_mag, chrom_b_fft_mag):
    fig = plt.figure(figsize=(15, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    x1, y1 = np.meshgrid(
        np.arange(chrom_a_fft_mag.shape[1]), np.arange(chrom_a_fft_mag.shape[0])
    )
    ax1.plot_surface(x1, y1, np.log(1 + chrom_a_fft_mag), cmap="viridis")
    ax1.set_title("Chrominance-a Magnitude Spectrum")

    ax2 = fig.add_subplot(122, projection="3d")
    x2, y2 = np.meshgrid(
        np.arange(chrom_b_fft_mag.shape[1]), np.arange(chrom_b_fft_mag.shape[0])
    )
    ax2.plot_surface(x2, y2, np.log(1 + chrom_b_fft_mag), cmap="viridis")
    ax2.set_title("Chrominance-b Magnitude Spectrum")

    plt.tight_layout()
    plt.show()


def plot_chrominance_magnitude_spectrum(chrom_a_fft_mag, chrom_b_fft_mag):
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(np.log(1 + chrom_a_fft_mag), cmap="gray")
    plt.title("Chrominance-a Magnitude Spectrum")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(np.log(1 + chrom_b_fft_mag), cmap="gray")
    plt.title("Chrominance-b Magnitude Spectrum")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_image(title, image):
    plt.figure()
    plt.title(title)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image)
    plt.axis("off")
    plt.show()


def encode_image(carrier_img, embed_img_a, embed_img_b):
    # Adım 1: JPEG imgesi alınır.
    carrier_img = cv2.imread(carrier_img)

    # Adım 2: RGB color'dan LAB color'a geçilir.
    carrier_lab = cv2.cvtColor(carrier_img, cv2.COLOR_BGR2LAB)

    # Adım 3: LAB color'un katmanları olan chrominance-a ve chrominance-b katmanlarının 2D FFT'si hesaplanır.
    luminance, chroma_a, chroma_b = cv2.split(carrier_lab)
    chrom_a_fft = np.fft.fft2(chroma_a)
    chrom_b_fft = np.fft.fft2(chroma_b)
    # show_image("chrom_a_fft", 20*np.log10(np.abs(chrom_a_fft)))
    # show_image("chrom_b_fft", 20*np.log10(np.abs(chrom_b_fft)))
    # show_image("luminance", luminance)

    # Adım 4: chrom_a_fft'i mangetude ve phase'ine ayrılır.
    chrom_a_fft_mag = np.abs(chrom_a_fft)
    chrom_a_fft_phase = np.angle(chrom_a_fft)
    # show_image("chrom_a_fft_mag", chrom_a_fft_mag)
    # show_image("chrom_a_fft_phase", chrom_a_fft_phase)

    # Adım 5: chrom_b_fft'i mangetude ve phase'ine ayrılır.
    chrom_b_fft_mag = np.abs(chrom_b_fft)
    chrom_b_fft_phase = np.angle(chrom_b_fft)

    # show_image("chrom_b_fft_mag", chrom_b_fft_mag)
    # show_image("chrom_b_fft_phase", chrom_b_fft_phase)
    # plot_chrominance_magnitude_spectrum_3d(chrom_a_fft_mag, chrom_b_fft_mag)

    # Adım 6: İki tane JPEG imgesi alınır (embed_img_a, embed_img_b)
    embed_img_a = cv2.imread(embed_img_a, cv2.IMREAD_GRAYSCALE)
    embed_img_b = cv2.imread(embed_img_b, cv2.IMREAD_GRAYSCALE)

    # Adım 7: embed_img_a, embed_img_b imgeleri carrier_img'ın yarısı olacak kadar scale edilir.
    fixed_shape = (500, 500)
    embed_img_a = cv2.resize(embed_img_a, fixed_shape)
    embed_img_b = cv2.resize(embed_img_b, fixed_shape)

    # Adım 8: embed_img_a'nın değerleri chrom_a_fft_mag'ın yüksek frekans değerlerine encode edilir. (encoded_a)
    encoding_factor = (1, 1)
    # place_tuple_a = ((chrom_a_fft_mag.shape[0] - embed_a_copy.shape[0], 0), (0, chrom_a_fft_mag.shape[1] - embed_a_copy.shape[1]))
    vertical_padding = (chrom_a_fft_mag.shape[0] - fixed_shape[0]) // 2
    horizontal_padding = chrom_a_fft_mag.shape[1] - fixed_shape[1]
    embed_a_copy = np.pad(
        embed_img_a,
        ((vertical_padding, vertical_padding), (horizontal_padding, 0)),
        mode="constant",
        constant_values=0,
    )
    # Add each element of emebed_a_copy +5.
    embed_a_copy = embed_a_copy + 15
    encoded_a = encoding_factor[0] * chrom_a_fft_mag + encoding_factor[1] * embed_a_copy

    # Adım 8: embed_img_b'nın değerleri chrom_b_fft_mag'ın yüksek frekans değerlerine encode edilir. (encoded_b)
    vertical_padding = (chrom_b_fft_mag.shape[0] - fixed_shape[0]) // 2
    horizontal_padding = chrom_b_fft_mag.shape[1] - fixed_shape[1]
    embed_b_copy = np.pad(
        embed_img_b,
        ((vertical_padding, vertical_padding), (horizontal_padding, 0)),
        mode="constant",
        constant_values=0
    )
    encoded_b = encoding_factor[0] * chrom_b_fft_mag + encoding_factor[1] * embed_b_copy
    plot_chrominance_magnitude_spectrum_3d(encoded_a, chrom_a_fft_mag)
    plot_chrominance_magnitude_spectrum_3d(encoded_b, chrom_b_fft_mag)

    # Adım 9: embed_img_a, chrom_a_fft_phase birleştirilir ve inverse FFT'si alınır. (encoded_chrome_a)
    encoded_chrome_a = np.fft.ifft2(encoded_a * np.exp(1j * chrom_a_fft_phase))
    # show_image("encoded_chrome_a", np.real(encoded_chrome_a))

    # Adım 10: embed_img_b, chrom_b_fft_phase birleştirilir ve inverse FFT'si alınır. (encoded_chrome_b)
    encoded_chrome_b = np.fft.ifft2(encoded_b * np.exp(1j * chrom_b_fft_phase))
    # show_image("encoded_chrome_b", np.real(encoded_chrome_b))

    # Adım 11: encoded_chrome_a, encoded_chrome_b ve luminance birleştirilerek LAB imgesi oluşturulur.
    encoded_lab = cv2.merge(
        (
            luminance,
            np.abs(encoded_chrome_a).astype(luminance.dtype),
            np.abs(encoded_chrome_b).astype(luminance.dtype),
        )
    )

    # Adım 12: LAB imgesi RGB'ye dönüştürülür.
    encoded_rgb = cv2.cvtColor(encoded_lab, cv2.COLOR_LAB2BGR)
    # show_image("encoded_rgb", encoded_rgb)

    # Adım 13: RGB imge JPEG olarak kaydedilir.
    cv2.imwrite(f"encoded_image_{encoding_factor[0]}_{encoding_factor[1]}.jpg", encoded_rgb)


# Uygulamayı çağırma
encode_image("carrier.jpg", "lenna.png", "parrots.png")
