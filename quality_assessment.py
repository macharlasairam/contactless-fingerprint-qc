"""
quality_assessment.py

Core image-quality metrics for a contactless fingerprint capture pipeline.

Five independent checks (blur, brightness, glare, ROI completeness, ridge
clarity) are combined into a single composite score (0-100) by
quality_gate(). Each check is deliberately cheap (well under the stage
budgets in the assignment spec) so the whole gate runs in well under 300ms
on a laptop CPU.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Metric 1: Blur
# ---------------------------------------------------------------------------
def check_blur(image_bgr: np.ndarray, threshold: float = 10.0) -> dict:
    """
    Sharpness via Laplacian variance.

    The Laplacian is the 2nd derivative of the image. A sharp image has lots
    of high-frequency edge content, so the *variance* of the Laplacian
    response is high. A blurry image has smoothed-out transitions, so the
    variance collapses toward zero.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "blur_score": round(blur_score, 2),
        "is_blurry": blur_score < threshold,
    }


# ---------------------------------------------------------------------------
# Metric 2: Brightness
# ---------------------------------------------------------------------------
def check_brightness(image_bgr: np.ndarray, min_thresh: float = 50.0, max_thresh: float = 210.0) -> dict:
    """Mean grayscale intensity. Flags underexposed and overexposed frames."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    return {
        "brightness": round(brightness, 2),
        "too_dark": brightness < min_thresh,
        "too_bright": brightness > max_thresh,
    }


# ---------------------------------------------------------------------------
# Metric 3: Glare
# ---------------------------------------------------------------------------
def check_glare(image_bgr: np.ndarray, max_glare_ratio: float = 0.02, k: float = 1.5) -> dict:
    """
    Fraction of pixels that form a specular "hot spot" relative to the
    image's OWN brightness distribution.

    Originally this used a fixed I > 240 cutoff, but real phone captures
    almost never hit that: auto-exposure compensates for the bright patch
    before the sensor saturates. On our own test photos, glare shots maxed
    out between 177 and 220 -- never near 240 -- while their overall mean
    brightness varied a lot from shot to shot (90 to 152). A fixed cutoff
    can't handle that variation, so instead we flag pixels that sit well
    above THIS image's mean (mean + k*std), which catches a bright
    localized spot regardless of the scene's overall lighting level.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    spike_threshold = mean + k * std

    glare_pixels = int(np.sum(gray > spike_threshold))
    total_pixels = int(gray.size)
    glare_fraction = float(glare_pixels / total_pixels)

    return {
        "has_glare": glare_fraction > max_glare_ratio,
        "glare_fraction": round(glare_fraction, 4),
        "spike_threshold": round(spike_threshold, 1),
    }


# ---------------------------------------------------------------------------
# Metric 4: ROI (Region of Interest) completeness
# ---------------------------------------------------------------------------
def check_roi_completeness(image_bgr: np.ndarray, min_roi_ratio: float = 0.15) -> dict:
    """
    Estimates how much of the frame the finger occupies.

    Otsu's method automatically picks the grayscale threshold that best
    separates two populations of pixels (here: finger vs. background) by
    minimizing intra-class variance. It's a quick stand-in for a trained
    segmentation model -- good enough to catch "finger too far away" or
    "finger barely in frame" cases, not good enough for pixel-perfect
    segmentation (that's what the real pipeline's segmentation stage is for).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    foreground_pixels = int(np.sum(thresh > 0))
    total_pixels = int(gray.size)
    roi_fraction = float(foreground_pixels / total_pixels)

    return {
        "roi_fraction": round(roi_fraction, 4),
        "roi_complete": roi_fraction >= min_roi_ratio,
    }


# ---------------------------------------------------------------------------
# Metric 5: Ridge clarity
# ---------------------------------------------------------------------------
def check_ridge_clarity(image_bgr: np.ndarray, threshold: float = 15.0) -> dict:
    """
    Gabor-filter response variance as a proxy for ridge-valley definition.

    A Gabor kernel is a sinusoid windowed by a Gaussian -- it responds
    strongly to texture at a specific spatial frequency (lambd) and
    orientation (theta), which is exactly what fingerprint ridges are:
    a near-periodic stripe pattern. Convolving the image with the kernel
    and taking the variance of the response tells us how strongly that
    ridge-like frequency is present. Flat skin or smeared ridges give a
    low-variance (flat) response; crisp ridges give a high-variance response.

    Note: a single kernel orientation (theta=pi/4, i.e. 45 degrees) is a
    simplification -- ridges run in different directions across the
    fingertip. A production system would use a bank of kernels at several
    orientations and take the max response per pixel.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getGaborKernel(
        (21, 21), sigma=5.0, theta=np.pi / 4, lambd=10.0, gamma=0.5, psi=0
    )
    filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
    ridge_score = float(np.var(filtered) / 100.0)

    return {
        "ridge_score": round(ridge_score, 2),
        "ridges_clear": ridge_score >= threshold,
    }


# ---------------------------------------------------------------------------
# Master gate
# ---------------------------------------------------------------------------
def quality_gate(image_path_or_array, thresholds: dict = None) -> dict:
    """
    Runs all 5 checks, normalizes each into [0, 1], computes a weighted
    0-100 composite score, and returns a pass/fail decision plus a single
    human-readable guidance message.

    thresholds (optional) can override any of: blur_threshold,
    brightness_min, brightness_max, glare_max, roi_min, ridge_threshold.
    """
    t = {
        "blur_threshold": 10.0,
        "brightness_min": 50.0,
        "brightness_max": 210.0,
        "glare_max": 0.02,
        "roi_min": 0.15,
        "ridge_threshold": 15.0,
    }
    if thresholds:
        t.update(thresholds)

    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
    else:
        img = image_path_or_array

    if img is None:
        raise ValueError("Invalid image file or path provided.")

    blur_res = check_blur(img, t["blur_threshold"])
    bright_res = check_brightness(img, t["brightness_min"], t["brightness_max"])
    glare_res = check_glare(img, t["glare_max"])
    roi_res = check_roi_completeness(img, t["roi_min"])
    ridge_res = check_ridge_clarity(img, t["ridge_threshold"])

    # Normalize each metric to [0, 1] for the composite score.
    n_blur = min(1.0, blur_res["blur_score"] / 50.0)
    n_bright = max(0.0, 1.0 - abs(bright_res["brightness"] - 128.0) / 128.0)
    n_glare = max(0.0, 1.0 - (glare_res["glare_fraction"] / 0.02))
    n_roi = min(1.0, roi_res["roi_fraction"] / 0.35)
    n_ridge = min(1.0, ridge_res["ridge_score"] / 30.0)

    composite = (
        0.25 * n_blur
        + 0.15 * n_bright
        + 0.15 * n_glare
        + 0.20 * n_roi
        + 0.25 * n_ridge
    ) * 100.0
    composite_score = round(composite, 1)

    has_hard_failure = (
        blur_res["is_blurry"]
        or bright_res["too_dark"]
        or bright_res["too_bright"]
        or glare_res["has_glare"]
        or not roi_res["roi_complete"]
        or not ridge_res["ridges_clear"]
    )

    passed = (composite_score >= 60.0) and (not has_hard_failure)

    # NOTE: brightness is checked before blur on purpose. Low-light noise
    # flattens edge contrast, which drags Laplacian variance down too --
    # so a genuinely dark photo can trip the blur check as well. Diagnosing
    # brightness first avoids telling the user "hold steady" when the real
    # fix is "turn on the light."
    if bright_res["too_dark"]:
        guidance = "Too dark -- move to a brighter spot or turn on flash."
    elif bright_res["too_bright"]:
        guidance = "Too bright -- reduce direct light source exposure."
    elif blur_res["is_blurry"]:
        guidance = "Too blurry -- hold your hand steady and re-focus."
    elif glare_res["has_glare"]:
        guidance = "Glare detected -- tilt finger slightly to eliminate reflections."
    elif not roi_res["roi_complete"]:
        guidance = "Finger incomplete -- position your fingertip within the camera guide."
    elif not ridge_res["ridges_clear"]:
        guidance = "Low ridge contrast -- clean lens or re-position finger."
    else:
        guidance = "Good capture -- ready for processing."

    return {
        "passed": passed,
        "composite_score": composite_score,
        "blur": blur_res,
        "brightness": bright_res,
        "glare": glare_res,
        "roi": roi_res,
        "ridge": ridge_res,
        "guidance": guidance,
    }