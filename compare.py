import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

# =======================================================
# CONFIG
# =======================================================

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "static" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

ORB_FEATURES = 4000  # slightly reduced for stability


# =======================================================
# LOAD IMAGE
# =======================================================

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    return img


# =======================================================
# PREPROCESS (STABLE FOR CAD)
# =======================================================

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


# =======================================================
# RESIZE
# =======================================================

def resize_pair(img1, img2):
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    return cv2.resize(img1, (w, h)), cv2.resize(img2, (w, h))


# =======================================================
# ALIGNMENT (STABLE + STRICT MATCHING)
# =======================================================

def align_images(img1, img2):

    g1 = preprocess(img1)
    g2 = preprocess(img2)

    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)

    kp1, des1 = orb.detectAndCompute(g1, None)
    kp2, des2 = orb.detectAndCompute(g2, None)

    if des1 is None or des2 is None:
        raise ValueError("Not enough features for alignment")

    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=60)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.70 * n.distance:   # stricter matching
            good.append(m)

    if len(good) < 25:  # STRICT (reduces false alignment)
        raise ValueError("Not enough good matches")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 4.0)

    if H is None:
        raise ValueError("Homography failed")

    h, w = img1.shape[:2]
    aligned = cv2.warpPerspective(img2, H, (w, h))

    return aligned


# =======================================================
# DIFFERENCE DETECTION (REDUCED NOISE)
# =======================================================

def compute_difference(img1, aligned):

    g1 = preprocess(img1)
    g2 = preprocess(aligned)

    diff = cv2.absdiff(g1, g2)

    # STRONG EDGE DETECTION (LESS SENSITIVE)
    edges = cv2.Canny(diff, 70, 180)

    # CLEAN NOISE
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    return diff, edges


# =======================================================
# ANALYSIS (STRICT FILTERING)
# =======================================================

def analyze(img1, aligned, mask):

    output = aligned.copy()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = 0
    pixels = 0
    largest = 0

    for c in contours:

        area = cv2.contourArea(c)

        # 🔥 STRICT FILTER (reduces noise detection)
        if area < 120:
            continue

        regions += 1
        pixels += int(area)
        largest = max(largest, area)

        x, y, w, h = cv2.boundingRect(c)

        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

    total = img1.shape[0] * img1.shape[1]
    percent = (pixels / total) * 100

    return output, {
        "changed_regions": regions,
        "changed_pixels": pixels,
        "percentage_changed": round(percent, 2),
        "largest_region": int(largest)
    }


# =======================================================
# MAIN PIPELINE
# =======================================================

def compare_images(path1, path2):

    img1 = load_image(path1)
    img2 = load_image(path2)

    img1, img2 = resize_pair(img1, img2)

    aligned = align_images(img1, img2)

    diff, mask = compute_difference(img1, aligned)

    highlighted, stats = analyze(img1, aligned, mask)

    # SSIM (stable scoring only)
    ssim_score = ssim(preprocess(img1), preprocess(aligned))

    summary = (
        f"SSIM: {round(ssim_score * 100, 2)}%. "
        f"Detected {stats['changed_regions']} meaningful changes. "
        f"{stats['percentage_changed']}% area modified."
    )

    # SAVE OUTPUTS
    cv2.imwrite(str(RESULT_DIR / "aligned.png"), aligned)
    cv2.imwrite(str(RESULT_DIR / "diff.png"), diff)
    cv2.imwrite(str(RESULT_DIR / "mask.png"), mask)
    cv2.imwrite(str(RESULT_DIR / "highlight.png"), highlighted)

    return {
        "aligned_image": str(RESULT_DIR / "aligned.png"),
        "difference_mask": str(RESULT_DIR / "mask.png"),
        "highlighted_image": str(RESULT_DIR / "highlight.png"),

        "ssim_score": round(ssim_score * 100, 2),

        "changed_regions": stats["changed_regions"],
        "changed_pixels": stats["changed_pixels"],
        "percentage_changed": stats["percentage_changed"],
        "largest_region": stats["largest_region"],

        "summary": summary
    }