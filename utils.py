"""
=========================================================
utils.py

Helper utilities for:
AI-Based Mechanical CAD Image Difference Detection

=========================================================
"""

import cv2
import numpy as np
from pathlib import Path


# -------------------------------------------------------
# SAFE IMAGE SAVING
# -------------------------------------------------------

def save_image(path: Path, image: np.ndarray):
    """
    Safely save an image to disk.

    Ensures directory exists and writes image using OpenCV.
    """

    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(path), image)

    except Exception as e:
        raise RuntimeError(f"Failed to save image: {str(e)}")


# -------------------------------------------------------
# FILE VALIDATION
# -------------------------------------------------------

def allowed_file(filename: str, allowed_ext=None):
    """
    Check if uploaded file has valid extension.
    """

    if allowed_ext is None:
        allowed_ext = {"png", "jpg", "jpeg"}

    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in allowed_ext
    )


# -------------------------------------------------------
# IMAGE NORMALIZATION
# -------------------------------------------------------

def normalize_image(image: np.ndarray):
    """
    Normalize image intensity for consistent processing.
    """

    if image is None:
        raise ValueError("Invalid image provided for normalization.")

    norm = cv2.normalize(
        image,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    )

    return norm


# -------------------------------------------------------
# RESIZE UTILITY (SAFE)
# -------------------------------------------------------

def resize_image(image: np.ndarray, width: int, height: int):
    """
    Resize image safely to given dimensions.
    """

    if image is None:
        raise ValueError("Cannot resize None image.")

    return cv2.resize(image, (width, height))


# -------------------------------------------------------
# GRAYSCALE CONVERSION
# -------------------------------------------------------

def to_gray(image: np.ndarray):
    """
    Convert BGR image to grayscale safely.
    """

    if image is None:
        raise ValueError("Cannot convert None image to grayscale.")

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# -------------------------------------------------------
# SUMMARY HELPER (OPTIONAL FALLBACK)
# -------------------------------------------------------

def generate_summary_fallback(stats: dict, ssim_score: float):
    """
    Simple fallback summary generator (used if needed).
    """

    regions = stats.get("changed_regions", 0)
    percent = stats.get("percentage_changed", 0)

    summary = f"SSIM: {round(ssim_score * 100, 2)}%. "

    summary += f"Detected {regions} changes. "

    if percent < 1:
        summary += "Minimal differences detected."
    elif percent < 5:
        summary += "Small engineering changes detected."
    else:
        summary += "Significant changes detected."

    return summary


# -------------------------------------------------------
# DIRECTORY SAFETY CHECK
# -------------------------------------------------------

def ensure_dir(path: Path):
    """
    Ensure a directory exists.
    """

    Path(path).mkdir(parents=True, exist_ok=True)