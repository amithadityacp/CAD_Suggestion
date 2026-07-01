<!-- Each stage is implemented as its own well-commented function in `compare.py`, so the
code reads top-to-bottom in the same order as this diagram.

---

## Error Handling

The application gracefully handles and reports (via on-screen messages, never a raw
crash) the following situations:

- Missing or empty file uploads
- Unsupported file formats
- Corrupted or unreadable image files
- Insufficient ORB feature matches between the two drawings
- Homography/alignment failure
- Images with very different resolutions or aspect ratios
- Files exceeding the 16 MB upload limit
- Unexpected server errors (logged to console, friendly message shown to user)

---

## Tuning / Configuration

A few constants at the top of `compare.py` can be adjusted for your specific CAD
drawings without touching the pipeline logic:

| Constant             | Purpose                                                   | Default |
|-----------------------|-------------------------------------------------------------|---------|
| `TARGET_WIDTH`         | Width both images are resized to before processing         | 1000    |
| `ORB_MAX_FEATURES`     | Max ORB keypoints detected per image                        | 5000    |
| `GOOD_MATCH_PERCENT`   | Fraction of best feature matches kept for alignment          | 0.15    |
| `MIN_MATCH_COUNT`      | Minimum matches required to trust alignment                  | 10      |
| `MIN_CONTOUR_AREA`     | Minimum pixel area to count as a real modification (denoise) | 6       |
| `MERGE_DISTANCE`       | Pixel gap under which nearby regions are merged into one     | 15      |

Increase `MIN_CONTOUR_AREA` if you're seeing false positives from JPEG compression
noise; decrease it if very small real modifications are being missed.

---

## Limitations

- Designed for line-art style mechanical CAD drawings (exported/scanned images), not
  photographs of physical parts.
- Extremely rotated (>45°) or perspective-warped scans may exceed what homography
  alignment can correct.
- Very low-resolution or heavily compressed images reduce detection sensitivity for
  the smallest (2–5 pixel) modifications.

---

## License

This project is provided as-is for local/internal engineering use. -->





"""
compare.py
----------
Core computer vision pipeline for the CAD Difference Detector.

This module is completely independent of Flask. It exposes a single
public function, compare_images(), which takes two image file paths
and returns a dictionary of result image paths, statistics, and a
human-readable summary.

Pipeline (as required by the spec):

    Load -> Resize -> Grayscale -> Histogram Equalization ->
    ORB Feature Detection -> Feature Matching -> Homography Alignment ->
    Light Gaussian Blur -> SSIM Comparison -> Edge-XOR Difference ->
    Morphological Closing -> Morphological Opening -> Contour Detection ->
    Bounding Boxes -> Statistics -> Summary Generation

Every stage below is written as its own small function so the pipeline
reads top-to-bottom like the spec itself, and so each stage can be
tested or reused independently.

NOTE ON DIFFERENCE DETECTION (updated):
The small-change detection stage now uses an "edge-XOR" technique
(comparing Canny edges between the two images directly, with a small
tolerance dilation for alignment noise) combined with a thresholded
SSIM map. This is significantly more sensitive to thin, small (2-5px)
CAD features -- new holes, moved slot walls, chamfer edits -- than a
purely blurred/structural diff, while still catching solid/filled
area changes via SSIM. See build_difference_mask() for details.
"""

import os
import uuid

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from utils import ImageProcessingError

# ----------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# ----------------------------------------------------------------------

# All images are resized to this width before processing so that
# comparisons are consistent regardless of the original CAD export
# resolution. Height is scaled proportionally to preserve aspect ratio.
TARGET_WIDTH = 1000

# ORB feature detector settings. A high feature count improves alignment
# accuracy for detailed technical drawings that have many corners/lines.
ORB_MAX_FEATURES = 5000

# Fraction of best matches (by distance) kept for homography estimation.
# Keeping only the strongest matches makes alignment more robust to
# false matches caused by repetitive CAD patterns (e.g. hatching).
GOOD_MATCH_PERCENT = 0.15

# Minimum number of good matches required to trust a homography.
# Below this, alignment is considered unreliable.
MIN_MATCH_COUNT = 10

# Minimum contour area (in pixels) to count as a real modification.
# This filters out single-pixel noise from JPEG compression artifacts,
# while still being sensitive enough to catch small CAD edits (spec
# requires detecting changes as small as 2-5 pixels wide, so this is
# deliberately low).
MIN_CONTOUR_AREA = 6

# Distance (in pixels) used to merge nearby contours into a single
# bounding box. CAD edits often produce several small disconnected
# contour fragments (e.g. around a hole's circular edge) that really
# represent ONE modification and should be reported as one region.
MERGE_DISTANCE = 15

# Tolerance (in pixels) used to dilate edges before comparing them in
# the edge-XOR difference stage. This absorbs tiny sub-pixel
# misalignment left over from the homography step so we don't flag
# every existing line as "changed" just because it shifted by a
# fraction of a pixel. Keep this small -- too large and real small
# modifications get swallowed too.
EDGE_ALIGNMENT_TOLERANCE = 2

# Canny thresholds tuned for clean CAD line-art (mostly black lines on
# white background). Lower thresholds catch faint/thin lines; adjust
# upward if scanned drawings are noisy.
CANNY_LOW = 40
CANNY_HIGH = 120

# Threshold applied to the SSIM difference map to decide which pixels
# count as "different" for solid/filled area changes.
SSIM_DIFF_THRESHOLD = 40


# ----------------------------------------------------------------------
# STAGE 1: LOAD + RESIZE
# ----------------------------------------------------------------------

def load_and_resize(image_path, target_width=TARGET_WIDTH):
    """
    Load an image from disk and resize it to a standard width.

    Resizing to a consistent width is important because:
      - It keeps processing time predictable regardless of upload size.
      - It ensures both images being compared are on the same scale
        before we even attempt feature matching.

    Raises:
        ImageProcessingError if the file can't be read as an image
        (corrupted file, unsupported/undetected format, etc.)
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ImageProcessingError(
            f"Could not read image file: {os.path.basename(image_path)}. "
            "The file may be corrupted or in an unsupported format."
        )

    height, width = image.shape[:2]
    scale = target_width / float(width)
    new_height = int(height * scale)

    resized = cv2.resize(image, (target_width, new_height), interpolation=cv2.INTER_AREA)
    return resized


# ----------------------------------------------------------------------
# STAGE 2: GRAYSCALE + HISTOGRAM EQUALIZATION
# ----------------------------------------------------------------------

def to_grayscale_equalized(image):
    """
    Convert a BGR image to grayscale and apply histogram equalization.

    Grayscale simplifies CAD line-art (which is mostly black/white/gray)
    down to a single channel for faster, simpler processing.

    Histogram equalization redistributes pixel intensities so that
    faint lines (e.g. light pencil-style CAD strokes or slightly
    under-exposed scans) become more visible and consistent in contrast
    between Image A and Image B. This matters a lot for CAD diffing,
    since two exports of the "same" drawing can have slightly different
    brightness/contrast from different renderers or scanners.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized


# ----------------------------------------------------------------------
# STAGE 3: ORB FEATURE DETECTION + MATCHING
# ----------------------------------------------------------------------

def detect_and_match_features(gray_a, gray_b):
    """
    Detect ORB keypoints/descriptors in both images and match them.

    ORB (Oriented FAST and Rotated BRIEF) is used instead of SIFT/SURF
    because it's free to use, fast, and works well on line-art/CAD
    drawings which have lots of sharp corners and edges (ideal for
    FAST keypoint detection).

    BFMatcher (Brute-Force Matcher) with Hamming distance is used
    because ORB descriptors are binary strings, and Hamming distance
    is the correct metric for comparing binary descriptors.

    Returns:
        keypoints_a, keypoints_b, good_matches (sorted by distance)

    Raises:
        ImageProcessingError if too few features or matches are found
        to reliably align the images.
    """
    orb = cv2.ORB_create(ORB_MAX_FEATURES)

    keypoints_a, descriptors_a = orb.detectAndCompute(gray_a, None)
    keypoints_b, descriptors_b = orb.detectAndCompute(gray_b, None)

    if descriptors_a is None or descriptors_b is None:
        raise ImageProcessingError(
            "Could not detect enough distinguishing features in one of the "
            "drawings to align them. Try uploading clearer, higher-contrast images."
        )

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors_a, descriptors_b)

    # Sort matches by descriptor distance (lower = more similar) and
    # keep only the strongest fraction. This discards weak/ambiguous
    # matches that would otherwise distort the homography estimate.
    matches = sorted(matches, key=lambda m: m.distance)
    num_good_matches = max(int(len(matches) * GOOD_MATCH_PERCENT), MIN_MATCH_COUNT)
    good_matches = matches[:num_good_matches]

    if len(good_matches) < MIN_MATCH_COUNT:
        raise ImageProcessingError(
            "Not enough matching features were found between the two drawings "
            "to align them. Please make sure both images show the same CAD "
            "drawing (possibly from a different angle or version)."
        )

    return keypoints_a, keypoints_b, good_matches


# ----------------------------------------------------------------------
# STAGE 4: HOMOGRAPHY ALIGNMENT
# ----------------------------------------------------------------------

def align_images(image_b, keypoints_a, keypoints_b, good_matches, target_shape):
    """
    Compute a homography from Image B -> Image A's coordinate space and
    warp Image B to align with Image A.

    Homography (a perspective transform) is used rather than a simple
    affine transform because it can correct for small rotation,
    translation, AND slight scaling/perspective differences between two
    scans/exports of the same CAD drawing -- exactly the tolerances
    required by the spec.

    RANSAC is used inside findHomography() to robustly ignore outlier
    matches (mismatched keypoints) when computing the transform.

    Raises:
        ImageProcessingError if a valid homography cannot be computed.
    """
    points_a = np.zeros((len(good_matches), 2), dtype=np.float32)
    points_b = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points_a[i] = keypoints_a[match.queryIdx].pt
        points_b[i] = keypoints_b[match.trainIdx].pt

    homography_matrix, mask = cv2.findHomography(points_b, points_a, cv2.RANSAC, 5.0)

    if homography_matrix is None:
        raise ImageProcessingError(
            "Automatic alignment of the two drawings failed. This can happen "
            "if the drawings are too different or too rotated. Please try "
            "uploading images that are roughly the same orientation."
        )

    height, width = target_shape[:2]
    aligned_b = cv2.warpPerspective(image_b, homography_matrix, (width, height))

    return aligned_b


# ----------------------------------------------------------------------
# STAGE 5: DIFFERENCE DETECTION (SSIM + Edge-XOR)
# ----------------------------------------------------------------------
#
# Small CAD edits (a new hole, a moved slot edge, a chamfer change) are
# thin, sharp LINE features. Diffing a blurred/structural map alone
# tends to wash these out. So we compare EDGES DIRECTLY between the two
# images ("edge-XOR"), which is far more sensitive to small line-level
# changes, and combine it with a thresholded SSIM map to still catch
# filled/solid-area changes (e.g. a filled pocket or removed hatching).

def compute_ssim_diff(gray_a, gray_b_aligned):
    """
    Compute the Structural Similarity Index (SSIM) between the two
    aligned grayscale images.

    A smaller win_size (5 instead of the scikit-image default of 7) is
    used so SSIM is evaluated over smaller local neighborhoods -- this
    makes it noticeably more sensitive to small, localized CAD changes
    instead of averaging them out over a larger window.

    Returns:
        score  -- float in [-1, 1], where 1.0 means identical images
        diff   -- uint8 difference map, HIGH values = more difference
    """
    score, diff = ssim(gray_a, gray_b_aligned, full=True, win_size=5)
    # ssim's diff map is in range [0, 1]. Invert and scale to 0-255 so
    # that HIGH values represent HIGH difference (more intuitive for
    # thresholding).
    diff_scaled = (1.0 - diff) * 255
    diff_scaled = diff_scaled.astype("uint8")
    return score, diff_scaled


def compute_edge_difference(gray_a, gray_b_aligned):
    """
    Detect small CAD modifications by comparing EDGES between the two
    images directly, rather than diffing pixel intensities.

    Why this matters: a new hole, a moved line, or a chamfer edit shows
    up as an edge that exists in one image but not the other. This is
    much more sensitive to thin, small (2-5px) features than a blurred
    structural diff, which tends to smear/average small sharp changes
    away.

    To avoid flagging every line in the drawing as "changed" just
    because homography alignment isn't pixel-perfect, each edge map is
    slightly dilated before comparison -- an edge in B only counts as
    "new" if there's no edge NEAR the same location in A (within
    EDGE_ALIGNMENT_TOLERANCE pixels), and vice versa for edges removed
    from A.

    Returns:
        edge_diff -- uint8 binary map (0 or 255) of edges that differ
                     between the two images
    """
    # NOTE: edge detection runs on the UNBLURRED grayscale images so
    # thin CAD lines stay sharp for Canny (blurring here would soften
    # exactly the small features we need to detect).
    edges_a = cv2.Canny(gray_a, CANNY_LOW, CANNY_HIGH)
    edges_b = cv2.Canny(gray_b_aligned, CANNY_LOW, CANNY_HIGH)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (EDGE_ALIGNMENT_TOLERANCE * 2 + 1, EDGE_ALIGNMENT_TOLERANCE * 2 + 1),
    )
    edges_a_tolerant = cv2.dilate(edges_a, kernel, iterations=1)
    edges_b_tolerant = cv2.dilate(edges_b, kernel, iterations=1)

    # An edge pixel in B is "new" only if A has no edge nearby.
    new_in_b = cv2.bitwise_and(edges_b, cv2.bitwise_not(edges_a_tolerant))
    # An edge pixel in A is "removed" only if B has no edge nearby.
    removed_from_a = cv2.bitwise_and(edges_a, cv2.bitwise_not(edges_b_tolerant))

    edge_diff = cv2.bitwise_or(new_in_b, removed_from_a)
    return edge_diff


def build_difference_mask(gray_a, gray_b_aligned):
    """
    Combine edge-based difference and SSIM-based difference into a
    single binary difference mask, then clean it up with morphological
    operations.

    Steps:
        1. Light Gaussian blur (3x3) applied ONLY for the SSIM pass --
           edge detection runs on the un-blurred images so small/thin
           lines aren't softened away before Canny sees them.
        2. Edge-XOR difference (primary detector for thin line/small
           feature changes -- new holes, moved edges, chamfer changes).
        3. SSIM difference map, thresholded (catches solid/filled area
           changes that don't have a clean edge signature).
        4. Combine edge-diff + thresholded SSIM diff (logical OR).
        5. Morphological Closing -- bridges small gaps in a
           modification's outline (e.g. a new hole's circular edge is
           broken into arcs by Canny) into one solid, contour-able blob.
        6. Morphological Opening -- removes leftover single-pixel noise
           specks without erasing genuine small (2-5px) features.

    Returns:
        ssim_score, final_binary_mask (uint8, values 0 or 255)
    """
    # Step 1: blur only for the SSIM pass (structural/solid-area diff).
    blurred_a = cv2.GaussianBlur(gray_a, (3, 3), 0)
    blurred_b = cv2.GaussianBlur(gray_b_aligned, (3, 3), 0)

    # Step 2: edge-based difference (primary detector for small changes)
    edge_diff = compute_edge_difference(gray_a, gray_b_aligned)

    # Step 3: SSIM-based difference (catches solid/filled area changes)
    ssim_score, ssim_diff = compute_ssim_diff(blurred_a, blurred_b)
    _, ssim_thresh = cv2.threshold(ssim_diff, SSIM_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Step 4: combine both signals
    combined_mask = cv2.bitwise_or(edge_diff, ssim_thresh)

    # Step 5: Morphological Closing (dilate then erode) -- bridges
    # fragmented edge pieces of one modification into a solid blob.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)

    # Step 6: Morphological Opening (erode then dilate) -- removes tiny
    # isolated noise specks. Kept deliberately small (2x2) so it does
    # NOT erase genuine 2-5 pixel modifications, per spec.
    kernel_open = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    return ssim_score, opened


# ----------------------------------------------------------------------
# STAGE 6: CONTOUR DETECTION + BOUNDING BOX MERGING
# ----------------------------------------------------------------------

def find_change_regions(binary_mask):
    """
    Find contours in the binary difference mask and convert them into
    bounding boxes, filtering out anything too small to be a real
    CAD modification (noise).

    Returns:
        A list of bounding boxes: [(x, y, w, h), ...]
    """
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= MIN_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))

    return boxes


def _boxes_are_close(box1, box2, distance=MERGE_DISTANCE):
    """
    Determine whether two bounding boxes are close enough that they
    likely represent fragments of the SAME modification (e.g. the two
    arcs of a single new circular hole) and should be merged.

    Checks the gap between box edges (not just centers), so it works
    correctly regardless of box size.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Compute the horizontal and vertical gap between the two boxes.
    # A negative value means the boxes already overlap on that axis.
    gap_x = max(x2 - (x1 + w1), x1 - (x2 + w2), 0)
    gap_y = max(y2 - (y1 + h1), y1 - (y2 + h2), 0)

    return gap_x <= distance and gap_y <= distance


def merge_nearby_boxes(boxes):
    """
    Merge bounding boxes that are close together into a single larger
    bounding box, representing one coherent modification.

    This uses a simple iterative union-merge approach: repeatedly scan
    for any pair of "close" boxes, merge them into their combined
    bounding rectangle, and repeat until no more merges happen. This is
    a beginner-friendly alternative to a full clustering algorithm and
    works well for the relatively small number of change regions found
    in CAD diffs.
    """
    if not boxes:
        return []

    merged = list(boxes)
    changed = True

    while changed:
        changed = False
        result = []
        used = [False] * len(merged)

        for i in range(len(merged)):
            if used[i]:
                continue
            box_i = merged[i]
            x1, y1, w1, h1 = box_i

            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                box_j = merged[j]

                if _boxes_are_close(box_i, box_j):
                    # Merge box_j into box_i by taking the union rectangle.
                    x2, y2, w2, h2 = box_j
                    new_x = min(x1, x2)
                    new_y = min(y1, y2)
                    new_x2 = max(x1 + w1, x2 + w2)
                    new_y2 = max(y1 + h1, y2 + h2)
                    box_i = (new_x, new_y, new_x2 - new_x, new_y2 - new_y)
                    x1, y1, w1, h1 = box_i
                    used[j] = True
                    changed = True

            used[i] = True
            result.append(box_i)

        merged = result

    return merged


# ----------------------------------------------------------------------
# STAGE 7: DRAWING OUTPUT IMAGES
# ----------------------------------------------------------------------

def draw_highlighted_image(base_image, boxes):
    """
    Draw bounding boxes for each detected modification on a copy of
    the (aligned) base image, so the user can visually see exactly
    where changes occurred.
    """
    output = base_image.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return output


# ----------------------------------------------------------------------
# STAGE 8: STATISTICS
# ----------------------------------------------------------------------

def compute_statistics(binary_mask, boxes, ssim_score):
    """
    Compute the numeric statistics required by the spec:
      - changed pixel count
      - percentage of the drawing changed
      - number of modified regions
      - each region's bounding box
      - the single largest modified region
    """
    total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
    changed_pixels = int(np.count_nonzero(binary_mask))
    percentage_changed = round((changed_pixels / total_pixels) * 100, 3)

    region_stats = []
    for (x, y, w, h) in boxes:
        region_stats.append({
            "x": x, "y": y, "width": w, "height": h, "area": w * h
        })

    largest_region = None
    if region_stats:
        largest_region = max(region_stats, key=lambda r: r["area"])

    return {
        "ssim_score": round(float(ssim_score) * 100, 2),  # as a percentage
        "changed_pixels": changed_pixels,
        "percentage_changed": percentage_changed,
        "changed_regions": len(region_stats),
        "regions": region_stats,
        "largest_region": largest_region,
    }


# ----------------------------------------------------------------------
# STAGE 9: SUMMARY GENERATION (pure Python logic, no ML)
# ----------------------------------------------------------------------

def _describe_location(x, y, image_width, image_height):
    """
    Convert a bounding box's position into a plain-English location
    description (e.g. "near the bottom-right corner") by dividing the
    image into a 3x3 grid, similar to a tic-tac-toe layout.
    """
    horizontal_third = image_width / 3
    vertical_third = image_height / 3

    if x < horizontal_third:
        horizontal = "left"
    elif x < horizontal_third * 2:
        horizontal = "center"
    else:
        horizontal = "right"

    if y < vertical_third:
        vertical = "top"
    elif y < vertical_third * 2:
        vertical = "middle"
    else:
        vertical = "bottom"

    if horizontal == "center" and vertical == "middle":
        return "near the center"
    if vertical == "middle":
        return f"near the {horizontal} side"
    if horizontal == "center":
        return f"near the {vertical} center"
    return f"near the {vertical}-{horizontal} corner"


def _classify_region_size(area, largest_area):
    """
    Give a rough plain-English classification of a region's size
    relative to the largest detected change, to make the summary read
    more naturally (e.g. "a slight rib extension" vs "a significant
    modification").
    """
    if largest_area == 0:
        return "a minor change"
    ratio = area / largest_area
    if ratio >= 0.8:
        return "a significant modification"
    elif ratio >= 0.4:
        return "a moderate modification"
    else:
        return "a slight modification"


def generate_summary(stats, image_width, image_height):
    """
    Generate a human-readable summary of the comparison using plain
    Python string logic (no machine learning / no external API calls),
    matching the tone/style specified in the ROLE section.
    """
    ssim_score = stats["ssim_score"]
    num_regions = stats["changed_regions"]
    percentage_changed = stats["percentage_changed"]
    regions = stats["regions"]
    largest_region = stats["largest_region"]

    sentences = []

    # Opening sentence: overall similarity.
    if ssim_score >= 99:
        similarity_phrase = "nearly identical"
    elif ssim_score >= 95:
        similarity_phrase = "highly similar"
    elif ssim_score >= 85:
        similarity_phrase = "moderately similar"
    else:
        similarity_phrase = "substantially different"

    sentences.append(
        f"The CAD drawings are {similarity_phrase} with an SSIM score of {ssim_score}%."
    )

    # Region count sentence.
    if num_regions == 0:
        sentences.append("No significant engineering modifications were detected.")
    elif num_regions == 1:
        sentences.append("One engineering modification was detected.")
    else:
        sentences.append(f"{num_regions} engineering modifications were detected.")

    # Describe up to the 3 largest individual regions in plain English,
    # so the summary stays readable even when there are many changes.
    if regions:
        largest_area = largest_region["area"] if largest_region else 0
        sorted_regions = sorted(regions, key=lambda r: r["area"], reverse=True)

        for region in sorted_regions[:3]:
            location = _describe_location(
                region["x"], region["y"], image_width, image_height
            )
            size_desc = _classify_region_size(region["area"], largest_area)
            sentences.append(
                f"{size_desc.capitalize()} was detected {location}."
            )

        if len(sorted_regions) > 3:
            remaining = len(sorted_regions) - 3
            sentences.append(
                f"{remaining} additional smaller modification(s) were also found."
            )

    # Closing sentence: overall changed area.
    sentences.append(
        f"Approximately {percentage_changed}% of the drawing has changed."
    )

    return " ".join(sentences)


# ----------------------------------------------------------------------
# MAIN PUBLIC FUNCTION
# ----------------------------------------------------------------------

def compare_images(image1_path, image2_path, results_folder):
    """
    Run the full CAD comparison pipeline on two uploaded images and
    return a dictionary of result image paths, statistics, and summary.

    This is the ONLY function other modules (app.py) should call.

    Args:
        image1_path (str): filesystem path to Image A
        image2_path (str): filesystem path to Image B
        results_folder (str): folder where generated result images
                               (aligned, mask, highlighted) are saved

    Returns:
        dict matching the structure defined in the project spec.

    Raises:
        ImageProcessingError on any known/expected failure, with a
        human-readable message intended to be shown directly to the user.
    """
    # --- Load + resize both images to a consistent scale ---------------
    image_a = load_and_resize(image1_path)
    image_b = load_and_resize(image2_path)

    # --- Grayscale + histogram equalization -----------------------------
    gray_a = to_grayscale_equalized(image_a)
    gray_b = to_grayscale_equalized(image_b)

    # --- ORB feature detection + matching -------------------------------
    keypoints_a, keypoints_b, good_matches = detect_and_match_features(gray_a, gray_b)

    # --- Homography alignment (align B onto A's coordinate space) -------
    aligned_image_b = align_images(image_b, keypoints_a, keypoints_b, good_matches, image_a.shape)
    aligned_gray_b = to_grayscale_equalized(aligned_image_b)

    # --- SSIM + Edge-XOR + Morphology -> binary difference mask ---------
    ssim_score, binary_mask = build_difference_mask(gray_a, aligned_gray_b)

    # --- Contour detection + bounding box merging ------------------------
    raw_boxes = find_change_regions(binary_mask)
    merged_boxes = merge_nearby_boxes(raw_boxes)

    # --- Draw highlighted output image -----------------------------------
    highlighted_image = draw_highlighted_image(image_a, merged_boxes)

    # --- Statistics --------------------------------------------------------
    height, width = image_a.shape[:2]
    stats = compute_statistics(binary_mask, merged_boxes, ssim_score)

    # --- Summary -------------------------------------------------------------
    summary = generate_summary(stats, width, height)

    # --- Save all output images to the results folder ----------------------
    unique_id = uuid.uuid4().hex[:8]

    original_a_path = os.path.join(results_folder, f"{unique_id}_original_a.png")
    original_b_path = os.path.join(results_folder, f"{unique_id}_original_b.png")
    aligned_path = os.path.join(results_folder, f"{unique_id}_aligned.png")
    mask_path = os.path.join(results_folder, f"{unique_id}_mask.png")
    highlighted_path = os.path.join(results_folder, f"{unique_id}_highlighted.png")

    cv2.imwrite(original_a_path, image_a)
    cv2.imwrite(original_b_path, image_b)
    cv2.imwrite(aligned_path, aligned_image_b)
    cv2.imwrite(mask_path, binary_mask)
    cv2.imwrite(highlighted_path, highlighted_image)

    # --- Return paths as relative "static/..." URLs for Flask templates ---
    def to_static_url(path):
        return "static/" + os.path.relpath(path, start=os.path.join(results_folder, "..")).replace("\\", "/")

    return {
        "original_a": to_static_url(original_a_path),
        "original_b": to_static_url(original_b_path),
        "aligned_image": to_static_url(aligned_path),
        "difference_mask": to_static_url(mask_path),
        "highlighted_image": to_static_url(highlighted_path),
        "ssim_score": stats["ssim_score"],
        "changed_regions": stats["changed_regions"],
        "changed_pixels": stats["changed_pixels"],
        "percentage_changed": stats["percentage_changed"],
        "regions": stats["regions"],
        "largest_region": stats["largest_region"],
        "summary": summary,
    }