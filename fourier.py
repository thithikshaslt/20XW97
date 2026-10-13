# Frequency Domain and Transforms - Answer Sheet
# Combined from all code cells in the original notebook.
# Markdown explanations and cell outputs are omitted.

# Imports

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Image Loading and Frequency-Domain Setup

img_path = "../../images/lenna.png"
img = cv.imread(img_path, 0).astype(np.float64)

M, N = img.shape

# Distance of every point from the centre
u, v = np.meshgrid(np.arange(N) - N / 2, np.arange(M) - M / 2)
D = np.sqrt(u**2 + v**2)

IMG = np.fft.fftshift(np.fft.fft2(img))


def out_of_focus_H(radius):
    psf = np.zeros((M, N), dtype=np.float32)
    cv.circle(psf, (N // 2, M // 2), radius, 1, -1)
    psf /= psf.sum()
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))


plt.imshow(img, cmap="gray")
plt.title("Input Image")
plt.xticks([]), plt.yticks([])
plt.show()

# 1. Gaussian Blur and Inverse Filtering

D0 = 10  # larger D0 -> weaker blur (H stays close to 1 over more frequencies)
H = np.exp(-(D**2) / (2 * D0**2))

# Step 1: Degrade with Gaussian blur
G = IMG * H
g = np.fft.ifft2(np.fft.ifftshift(G))

# Step 2: Apply inverse filter
F_hat = np.nan_to_num(G / H)
f_hat = np.fft.ifft2(np.fft.ifftshift(F_hat))

plt.figure(figsize=(12, 4))
plt.subplot(1, 4, 1), plt.imshow(img, cmap="gray"), plt.title("Input Image"), plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 2), plt.imshow(H, cmap="gray"), plt.title("PSF"), plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 3), plt.imshow(np.abs(g), cmap="gray"), plt.title("Blurred Image"), plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 4), plt.imshow(np.abs(f_hat), cmap="gray"), plt.title("Restored Image"), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()

# 2. Restoration Using an Incorrect Gaussian Blur Parameter

D0_true, D0_wrong = 10, 5  # D0_true sets the actual blur strength; using a smaller D0_wrong under-corrects it, leaving residual blur

H_true = np.exp(-(D**2) / (2 * D0_true**2))
H_wrong = np.exp(-(D**2) / (2 * D0_wrong**2))

G = IMG * H_true
g = np.fft.ifft2(np.fft.ifftshift(G))

F_hat = np.nan_to_num(G / H_wrong)
f_hat = np.fft.ifft2(np.fft.ifftshift(F_hat))

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1), plt.imshow(img, cmap="gray"), plt.title("Input Image"), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2), plt.imshow(np.abs(g), cmap="gray"), plt.title("Blurred (variance=10)"), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3), plt.imshow(np.abs(f_hat), cmap="gray"), plt.title("Restored (assumed variance=5)"), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()

# 3. Inverse Filtering for Out-of-Focus Blur

H = out_of_focus_H(radius=7)  # larger radius -> stronger, more spread-out blur

G = IMG * H
g = np.fft.ifft2(np.fft.ifftshift(G))

F_hat = np.nan_to_num(G / H)
f_hat = np.fft.ifft2(np.fft.ifftshift(F_hat))

plt.figure(figsize=(12, 4))
plt.subplot(1, 4, 1), plt.imshow(img, cmap="gray"), plt.title("Input Image"), plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 2), plt.imshow(np.abs(H), cmap="gray"), plt.title("Out-of-Focus PSF"), plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 3), plt.imshow(np.abs(g), cmap="gray"), plt.title("Blurred Image"), plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 4), plt.imshow(np.abs(f_hat), cmap="gray"), plt.title("Restored Image"), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()

# 4. Inverse Filtering for Gaussian-Blurred Image with Gaussian Noise

D0 = 10  # larger D0 -> weaker blur to restore from
H = np.exp(-(D**2) / (2 * D0**2))

g_clean = np.abs(np.fft.ifft2(np.fft.ifftshift(IMG * H)))

noise = np.random.normal(0, 5, img.shape)
g_noisy = g_clean + noise

G_noisy = np.fft.fftshift(np.fft.fft2(g_noisy))

F_hat = np.nan_to_num(G_noisy / H)
f_hat = np.fft.ifft2(np.fft.ifftshift(F_hat))

plt.figure(figsize=(12, 4))
plt.subplot(1, 4, 1), plt.imshow(img, cmap="gray"), plt.title("Input Image"), plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 2), plt.imshow(g_clean, cmap="gray"), plt.title("Blurred Image"), plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 3), plt.imshow(g_noisy, cmap="gray"), plt.title("Blurred + Noisy"), plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 4), plt.imshow(np.abs(f_hat), cmap="gray"), plt.title("Inverse Filter Restored"), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()

# 5. Wiener Filtering for Gaussian Noise and Out-of-Focus Blur

# Steps 1-3: noise-free image, out-of-focus PSF, and their spectra
H = out_of_focus_H(radius=7)  # larger radius -> stronger blur to restore from

# Step 4: blurred image
g = np.abs(np.fft.ifft2(np.fft.ifftshift(IMG * H)))

# Step 5: add Gaussian noise, mean 0, variance 10
noise = np.random.normal(0, np.sqrt(10), img.shape)
g_noisy = g + noise

# Step 6: spectrum of the noisy image
G_noisy = np.fft.fftshift(np.fft.fft2(g_noisy))

# Step 7: Wiener filter, K = 0.03
K = 0.03  # larger K suppresses noise amplification more, at the cost of leaving more residual blur
H_wiener = np.conj(H) / (np.abs(H) ** 2 + K)

# Step 8: restore
f_hat = np.fft.ifft2(np.fft.ifftshift(G_noisy * H_wiener))

plt.figure(figsize=(12, 4))
plt.subplot(1, 4, 1), plt.imshow(img, cmap="gray"), plt.title("Input Image"), plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 2), plt.imshow(g, cmap="gray"), plt.title("Blurred Image"), plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 3), plt.imshow(g_noisy, cmap="gray"), plt.title("Blurred + Noisy"), plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 4), plt.imshow(np.abs(f_hat), cmap="gray"), plt.title("Wiener Restored"), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()

# 6. Effect of Out-of-Focus PSF Radius

radii = [3, 5, 7, 10, 15]  # larger radius -> more blur and a more ill-conditioned (harder to invert) OTF

plt.figure(figsize=(15, 6))

for idx, r in enumerate(radii):

    H = out_of_focus_H(radius=r)
    g = np.abs(np.fft.ifft2(np.fft.ifftshift(IMG * H)))

    plt.subplot(2, len(radii), idx + 1)
    plt.imshow(g, cmap="gray")
    plt.title(f"Blurred R={r}")
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, len(radii), len(radii) + idx + 1)
    plt.imshow(np.abs(H), cmap="gray")
    plt.title(f"OTF R={r}")
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
