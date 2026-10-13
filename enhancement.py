# SECTION 1: Import libraries
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# SECTION 2: Display images
def show(images):
    plt.figure(figsize=(4 * len(images), 4))
    for i, (title, im) in enumerate(images.items()):
        plt.subplot(1, len(images), i + 1)
        if im.ndim == 2:
            plt.imshow(im, cmap="gray")
        else:
            plt.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()


# SECTION 3: Load the original grayscale image
# Assumes the script is saved in CA02/1_Image_Enhancement/.
img_path = Path(__file__).resolve().parents[2] / "images" / "lenna.png"
img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(
        f"Could not load image: {img_path}. Update img_path to the location of lenna.png."
    )

show({"Original Image": img})


# ============================================================
# SECTION 4: SPATIAL-DOMAIN FILTERING
# ============================================================

# 4A: Low-pass filtering using box and Gaussian filters
box_kernel = np.ones((5, 5), dtype=np.float32) / 25
gauss_1d = cv.getGaussianKernel(5, 1)
gauss_kernel = gauss_1d @ gauss_1d.T

box_lpf = cv.filter2D(img, -1, box_kernel)
gaussian_lpf = cv.filter2D(img, -1, gauss_kernel)

show({
    "Original": img,
    "Box LPF": box_lpf,
    "Gaussian LPF": gaussian_lpf,
})


# 4B: High-pass filtering, unsharp masking, and high-boost filtering
blurred = cv.filter2D(img, -1, gauss_kernel)
mask = cv.subtract(img, blurred)

# High-pass filter: original image minus blurred image
highpass = mask

# Unsharp masking: add the detail mask back to the original image
unsharp = cv.add(img, mask)

# High-boost filtering: amplify the detail mask
k = 2.5
highboost = np.clip(img.astype(np.float32) + k * mask, 0, 255).astype(np.uint8)

show({
    "Original": img,
    "Blurred": blurred,
    "High-pass": highpass,
    "Unsharp Masking": unsharp,
    "High-boost (k=2.5)": highboost,
})


# ============================================================
# SECTION 5: TRANSFORM-DOMAIN IMAGE ENHANCEMENT
# ============================================================

# 5A: Create ideal, Butterworth, and Gaussian low-pass filter masks
M, N = img.shape

# Calculate each frequency point's distance from the spectrum centre
u, v = np.meshgrid(np.arange(N) - N // 2, np.arange(M) - M // 2)
D = np.sqrt(u**2 + v**2)

d0, n = 30, 2

ideal_mask = (D <= d0).astype(np.float32)
butter_mask = 1 / (1 + (D / d0) ** (2 * n))
gauss_mask = np.exp(-(D**2) / (2 * d0**2))

# Compute the centred Fourier transform of the image
F = np.fft.fftshift(np.fft.fft2(img))


# 5B: Apply a frequency-domain mask and convert back to an image
def apply_mask(mask):
    filtered = np.fft.ifft2(np.fft.ifftshift(F * mask))
    return np.clip(np.abs(filtered), 0, 255).astype(np.uint8)


# Display low-pass masks and filtered images
show({
    "Ideal LPF Mask": (ideal_mask * 255).astype(np.uint8),
    "Butterworth LPF Mask": (butter_mask * 255).astype(np.uint8),
    "Gaussian LPF Mask": (gauss_mask * 255).astype(np.uint8),
    "Original": img,
    "Ideal LPF": apply_mask(ideal_mask),
    "Butterworth LPF": apply_mask(butter_mask),
    "Gaussian LPF": apply_mask(gauss_mask),
})


# 5C: Create and apply ideal, Butterworth, and Gaussian high-pass masks
# High-pass mask = 1 - corresponding low-pass mask
ideal_h = 1 - ideal_mask
butter_h = 1 - butter_mask
gauss_h = 1 - gauss_mask

show({
    "Ideal HPF Mask": (ideal_h * 255).astype(np.uint8),
    "Butterworth HPF Mask": (butter_h * 255).astype(np.uint8),
    "Gaussian HPF Mask": (gauss_h * 255).astype(np.uint8),
    "Original": img,
    "Ideal HPF": apply_mask(ideal_h),
    "Butterworth HPF": apply_mask(butter_h),
    "Gaussian HPF": apply_mask(gauss_h),
})


# 5D: High-frequency emphasis filtering
a_const, b_const = 0.5, 2.0
emphasis_mask = a_const + b_const * gauss_h

show({
    "Original": img,
    "High-Frequency Emphasis": apply_mask(emphasis_mask),
})


# 5E: Ideal band-pass and band-reject filtering
d0_band, w = 40, 20

bandreject_mask = np.ones((M, N), dtype=np.float32)
bandreject_mask[
    (D >= d0_band - w / 2) & (D <= d0_band + w / 2)
] = 0

# Band-pass mask keeps the frequencies removed by the band-reject mask
bandpass_mask = 1 - bandreject_mask

show({
    "Original": img,
    "Band Pass Mask": bandpass_mask,
    "Ideal Band-Pass": apply_mask(bandpass_mask),
    "Ideal Band-Reject": apply_mask(bandreject_mask),
})


# ============================================================
# SECTION 6: HOMOMORPHIC FILTERING
# ============================================================

# Reduce illumination variation and enhance reflectance details
gamma_l, gamma_h, c, d0_homo = 0.5, 2.0, 1, 30

# Log transform -> Fourier transform -> frequency filtering
img_log = np.log1p(img.astype(np.float32))
F_log = np.fft.fftshift(np.fft.fft2(img_log))

# Build the homomorphic filter
H = (
    (gamma_h - gamma_l)
    * (1 - np.exp(-c * (D**2) / (d0_homo**2)))
    + gamma_l
)

# Inverse Fourier transform and reverse the log transform
img_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(F_log * H)))
homomorphic = np.expm1(img_filtered)
homomorphic = cv.normalize(
    homomorphic, None, 0, 255, cv.NORM_MINMAX
).astype(np.uint8)

show({
    "Original": img,
    "Homomorphic Filter": homomorphic,
})