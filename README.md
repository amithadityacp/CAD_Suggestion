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

