# ============================================================
# SECTION 1: IMPORT LIBRARIES
# ============================================================
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math


# ============================================================
# SECTION 2: LOAD AND DISPLAY THE ORIGINAL IMAGE
# ============================================================
img_path = "../../images/lenna.png"
img = cv.imread(img_path, 0).astype(np.float64)

height, width = img.shape

MAX_INTENSITY = 255
MIN_INTENSITY = 0

plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")
plt.show()


# ============================================================
# SECTION 3: GENERATE DIFFERENT TYPES OF NOISE
# ============================================================

# 3A: Noise parameters
gaussian_mean, gaussian_std = 0, 20
uniform_min, uniform_max = -40, 40
impulse_probability = 0.05
speckle_mean, speckle_std = 0, 0.2
rayleigh_scale = 20
period, periodic_amplitude = 10, 20

# 3B: Gaussian noise
gaussian_noise = np.random.normal(gaussian_mean, gaussian_std, img.shape)
gaussian_img = np.clip(img + gaussian_noise, MIN_INTENSITY, MAX_INTENSITY)

# 3C: Uniform noise
uniform_noise = np.random.uniform(uniform_min, uniform_max, img.shape)
uniform_img = np.clip(img + uniform_noise, MIN_INTENSITY, MAX_INTENSITY)

# 3D: Impulse (salt-and-pepper) noise
random_values = np.random.rand(height, width)
impulse_img = img.copy()
impulse_img[random_values < impulse_probability / 2] = MIN_INTENSITY
impulse_img[random_values > 1 - impulse_probability / 2] = MAX_INTENSITY

# 3E: Speckle noise
speckle_noise = np.random.normal(speckle_mean, speckle_std, img.shape)
speckle_img = np.clip(img + img * speckle_noise, MIN_INTENSITY, MAX_INTENSITY)

# 3F: Poisson noise
poisson_img = np.clip(np.random.poisson(img), MIN_INTENSITY, MAX_INTENSITY)

# 3G: Rayleigh noise
rayleigh_noise = np.random.rayleigh(rayleigh_scale, img.shape)
rayleigh_img = np.clip(img + rayleigh_noise, MIN_INTENSITY, MAX_INTENSITY)

# 3H: Periodic noise
x = np.arange(width)
periodic_noise = periodic_amplitude * np.sin(2 * np.pi * x / period)
periodic_img = np.clip(img + periodic_noise, MIN_INTENSITY, MAX_INTENSITY)

# 3I: Display all noisy images
images = [
    ("Original", img), ("Gaussian", gaussian_img),
    ("Uniform", uniform_img), ("Impulse", impulse_img),
    ("Speckle", speckle_img), ("Poisson", poisson_img),
    ("Rayleigh", rayleigh_img), ("Periodic", periodic_img)
]

plt.figure(figsize=(12, 8))

for i, (title, image) in enumerate(images, 1):
    plt.subplot(2, 4, i)
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# SECTION 4: LINEAR SPATIAL-DOMAIN FILTERS
# ============================================================

# 4A: Select the Gaussian-noisy image
noisy = gaussian_img

# 4B: Mean filter and weighted-average filter kernels
mean_kernel = np.ones((3, 3), np.float32) / 9

weighted_kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], np.float32) / 16

# 4C: Apply the filters
mean3 = cv.filter2D(noisy, -1, mean_kernel)
weighted3 = cv.filter2D(noisy, -1, weighted_kernel)

# 4D: Display results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1), plt.imshow(noisy, cmap='gray'), plt.title("Noisy"), plt.axis("off")
plt.subplot(1, 3, 2), plt.imshow(mean3, cmap='gray'), plt.title("Mean Filter"), plt.axis("off")
plt.subplot(1, 3, 3), plt.imshow(weighted3, cmap='gray'), plt.title("Weighted Average"), plt.axis("off")

plt.show()


# ============================================================
# SECTION 5: NON-LINEAR SPATIAL-DOMAIN FILTERS
# ============================================================

# 5A: Parameters and input image
noisy = gaussian_img

kernel_size = 3
trim = 2
pad = kernel_size // 2

# 5B: Edge padding and neighbourhood windows
padded = np.pad(noisy, pad, mode="edge")

windows = np.lib.stride_tricks.sliding_window_view(
    padded, (kernel_size, kernel_size)
)

# 5C: Median filter
median = np.median(windows, axis=(-2, -1))

# 5D: Minimum filter
minimum = np.min(windows, axis=(-2, -1))

# 5E: Maximum filter
maximum = np.max(windows, axis=(-2, -1))

# 5F: Alpha-trimmed mean filter
windows = windows.reshape(windows.shape[0], windows.shape[1], -1)
sorted_windows = np.sort(windows, axis=-1)
trimmed = sorted_windows[..., trim//2 : -trim//2]
alpha_trimmed = np.mean(trimmed, axis=-1)

# 5G: Display results
plt.figure(figsize=(12, 4))

plt.subplot(1, 5, 1), plt.imshow(noisy, cmap="gray"), plt.title("Noisy"), plt.axis("off")
plt.subplot(1, 5, 2), plt.imshow(median, cmap="gray"), plt.title("Median"), plt.axis("off")
plt.subplot(1, 5, 3), plt.imshow(minimum, cmap="gray"), plt.title("Minimum"), plt.axis("off")
plt.subplot(1, 5, 4), plt.imshow(maximum, cmap="gray"), plt.title("Maximum"), plt.axis("off")
plt.subplot(1, 5, 5), plt.imshow(alpha_trimmed, cmap="gray"), plt.title("Alpha Trimmed"), plt.axis("off")

plt.show()


# ============================================================
# SECTION 6: ANISOTROPIC DIFFUSION FILTERING
# ============================================================

# 6A: Set parameters
img = gaussian_img.astype(float)

iterations = 10
K = 20
lam = 0.2

# 6B: Iteratively diffuse the image while preserving edges
for _ in range(iterations):
    n = np.roll(img, -1, 0) - img
    s = np.roll(img,  1, 0) - img
    e = np.roll(img, -1, 1) - img
    w = np.roll(img,  1, 1) - img

    img += lam * (
        n * np.exp(-(n/K) ** 2) +
        s * np.exp(-(s/K) ** 2) +
        e * np.exp(-(e/K) ** 2) +
        w * np.exp(-(w/K) ** 2)
    )

# 6C: Display the filtered image
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()


# ============================================================
# SECTION 7: FOURIER-TRANSFORM-BASED IMAGE DENOISING
# ============================================================

# 7A: Construct the frequency-distance grid
height, width = img.shape

u, v = np.meshgrid(
    np.arange(width) - width // 2,
    np.arange(height) - height // 2
)

distance = np.sqrt(u**2 + v**2)

# 7B: Construct the Gaussian low-pass mask
cutoff_frequency = 40
gaussian_mask = np.exp(
    -(distance**2) / (2 * cutoff_frequency**2)
)

# 7C: Transform the periodic-noise image into the frequency domain
frequency_image = np.fft.fftshift(
    np.fft.fft2(periodic_img)
)

# 7D: Apply the Gaussian mask
filtered_frequency = frequency_image * gaussian_mask

# 7E: Convert back to the spatial domain
denoised_image = np.abs(
    np.fft.ifft2(
        np.fft.ifftshift(filtered_frequency)
    )
)

denoised_image = np.clip(
    denoised_image, 0, 255
).astype(np.uint8)

# 7F: Display the noisy and denoised images
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(periodic_img, cmap="gray")
plt.title("Noisy")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(denoised_image, cmap="gray")
plt.title("FFT Denoised")
plt.axis("off")

plt.show()


# ============================================================
# SECTION 8: IMAGE RESTORATION USING POINT SPREAD FUNCTIONS
# ============================================================

# 8A: Parameters
out_of_focus_radius = 10
motion_length = 15
turbulence_strength = 0.001

# 8B: Out-of-focus blur — disk point spread function
out_of_focus_psf = np.zeros((height, width), np.float32)
cv.circle(
    out_of_focus_psf,
    (width // 2, height // 2),
    out_of_focus_radius,
    1,
    -1
)
out_of_focus_psf /= out_of_focus_psf.sum()

out_of_focus_blur = cv.filter2D(
    img, -1, out_of_focus_psf
)

# 8C: Motion blur — line point spread function
motion_psf = np.zeros(
    (motion_length, motion_length),
    np.float32
)
motion_psf[motion_length // 2, :] = 1
motion_psf /= motion_psf.sum()

motion_blur = cv.filter2D(
    img, -1, motion_psf
)

# 8D: Atmospheric turbulence — frequency response
u, v = np.meshgrid(
    np.arange(width) - width // 2,
    np.arange(height) - height // 2
)

distance = np.sqrt(u**2 + v**2)

turbulence_H = np.exp(
    -turbulence_strength * distance**(5 / 3)
)

frequency_image = np.fft.fftshift(
    np.fft.fft2(img)
)

turbulence_frequency = frequency_image * turbulence_H

turbulence_blur = np.abs(
    np.fft.ifft2(
        np.fft.ifftshift(turbulence_frequency)
    )
)

# 8E: Display the blurred images
images = [
    ("Original", img),
    ("Out-of-Focus", out_of_focus_blur),
    ("Motion Blur", motion_blur),
    ("Atmospheric Turbulence", turbulence_blur)
]

plt.figure(figsize=(12, 4))

for i, (title, image) in enumerate(images, 1):
    plt.subplot(1, 4, i)
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# SECTION 9: IMAGE RESTORATION USING INVERSE, PSEUDO-INVERSE,
#            AND WIENER FILTERS
# ============================================================

# 9A: Restoration parameters
blur_radius = 10
threshold = 0.01
noise_ratio = 0.01

# 9B: Create the out-of-focus point spread function
psf = np.zeros((height, width))
cv.circle(psf, (width // 2, height // 2), blur_radius, 1, -1)
psf /= psf.sum()

# 9C: Transform the image and point spread function into the frequency domain
F = np.fft.fftshift(np.fft.fft2(img))
H = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
G = F * H

# 9D: Apply inverse, pseudo-inverse, and Wiener restoration filters
inverse = G / H
pseudo_inverse = np.where(np.abs(H) > threshold, G / H, 0)
wiener = (np.conj(H) / (np.abs(H)**2 + noise_ratio)) * G

# 9E: Convert restored images back to the spatial domain
images = [
    ("Original", img),
    ("Blurred", np.abs(np.fft.ifft2(np.fft.ifftshift(G)))),
    ("Inverse", np.abs(np.fft.ifft2(np.fft.ifftshift(inverse)))),
    ("Pseudo-Inverse", np.abs(np.fft.ifft2(np.fft.ifftshift(pseudo_inverse)))),
    ("Wiener", np.abs(np.fft.ifft2(np.fft.ifftshift(wiener))))
]

# 9F: Display restoration results
plt.figure(figsize=(15, 4))

for i, (title, image) in enumerate(images, 1):
    plt.subplot(1, 5, i)
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()