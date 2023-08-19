import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_magnitude_spectrum(magnitude_spectrum, title="Magnitude Spectrum"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(0, magnitude_spectrum.shape[1], 1)
    y = np.arange(0, magnitude_spectrum.shape[0], 1)
    X, Y = np.meshgrid(x, y)

    ax.plot_surface(X, Y, np.log(1 + magnitude_spectrum), cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Magnitude (log)')
    plt.show()

def decode_image(encoded_image_path):
    # Adım 1: JPEG imgesi alınır.
    encoded_rgb = cv2.imread(encoded_image_path)

    # Adım 2: RGB color'dan LAB color'a geçilir.
    encoded_lab = cv2.cvtColor(encoded_rgb, cv2.COLOR_BGR2LAB)

    # Adım 3: LAB color'un katmanlarına ayrılır.
    luminance, encoded_chrome_a, encoded_chrome_b = cv2.split(encoded_lab)

    # Adım 4: chrom_a_fft_phase ve chrom_b_fft_phase elde edilir.
    chrom_a_fft_phase = np.angle(np.fft.fft2(encoded_chrome_a))
    chrom_b_fft_phase = np.angle(np.fft.fft2(encoded_chrome_b))

    # Adım 5: chrom_a_fft_mag ve chrom_b_fft_mag elde edilir.
    chrom_a_fft_mag = np.abs(np.fft.fft2(encoded_chrome_a))
    chrom_b_fft_mag = np.abs(np.fft.fft2(encoded_chrome_b))

    plot_3d_magnitude_spectrum(chrom_a_fft_mag)
    plot_3d_magnitude_spectrum(chrom_b_fft_mag)

    # Extract the region from chrom_a_fft_mag
    region_chrom_a = chrom_a_fft_mag[0:483, 0:1]

    # Extract the region from chrom_b_fft_mag
    region_chrom_b = chrom_b_fft_mag[594:595, 0:949]

    # Adım 6: Orjinal embed_img_a ve embed_img_b'nin boyutlarına uygun olarak geri ölçeklenir.
    embed_img_a = cv2.resize(encoded_chrome_a, (chrom_a_fft_mag.shape[1], chrom_a_fft_mag.shape[0]))
    embed_img_b = cv2.resize(encoded_chrome_b, (chrom_b_fft_mag.shape[1], chrom_b_fft_mag.shape[0]))

    # Adım 7: Original chrom_a_fft_mag ve chrom_b_fft_mag elde edilir.
    original_chrom_a_fft_mag = chrom_a_fft_mag - np.abs(embed_img_a)
    original_chrom_b_fft_mag = chrom_b_fft_mag - np.abs(embed_img_b)

    # Adım 8: Original chrom_a_fft ve chrom_b_fft elde edilir.
    original_chrom_a_fft = original_chrom_a_fft_mag * np.exp(1j * chrom_a_fft_phase)
    original_chrom_b_fft = original_chrom_b_fft_mag * np.exp(1j * chrom_b_fft_phase)

    # Adım 9: Original chrominance bileşenleri oluşturulur.
    original_chroma_a = np.fft.ifft2(original_chrom_a_fft)
    original_chroma_b = np.fft.ifft2(original_chrom_b_fft)

    # Adım 10: Original LAB bileşenleri birleştirilerek LAB imgesi oluşturulur.
    original_lab = cv2.merge((luminance, original_chroma_a.real.astype(luminance.dtype), original_chroma_b.real.astype(luminance.dtype)))

    # Adım 11: Original LAB imgesi RGB'ye dönüştürülür.
    original_rgb = cv2.cvtColor(original_lab, cv2.COLOR_LAB2BGR)

    # Adım 12: Orjinal resimleri PNG olarak kaydet
    cv2.imwrite("decoded_embed_img_a.png", (embed_img_a * 255).astype(np.uint8))
    cv2.imwrite("decoded_embed_img_b.png", (embed_img_b * 255).astype(np.uint8))
    cv2.imwrite("decoded_original_image.png", original_rgb)

# Decode işlemi için kaydedilmiş olan encoded_image.jpg dosyasını kullanıyoruz
decode_image("encoded_image.jpg")