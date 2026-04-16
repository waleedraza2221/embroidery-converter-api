"""
image_processor.py — AI-assisted image-to-embroidery digitizing pipeline.

Pipeline
--------
  1. Color quantization with KMeans (n_colors clusters).
  2. Per-color masks smoothed with Gaussian blur → clean, rounded contours.
  3. Shape analysis per contour via PCA:
       • long-axis angle  → stitch travel direction
       • aspect ratio     → classify stitch type
  4. Stitch-type assignment per shape:
       satin  → columns ⊥ to long axis  (elongated: aspect ≥ satin_aspect_ratio)
       tatami → scan-rows ‖ to long axis (broad fills, area ≥ min_fill_area_mm²)
       run    → spaced points on perimeter (tiny regions)
  5. Optional underlay run stitch (loose edge-walk) before fill / satin
     to anchor fabric and ensure needle register.
  6. Outline run stitch over fill / satin for crisp edges.
  7. Return {color_index, points:[{x,y,jump}]} segments in pyembroidery
     0.1-mm units, ready for /export-combined.

Stitch-direction notes
----------------------
  Tatami fill:
    • Rows are parallel to the shape's PCA long-axis angle so the
      scan direction follows the shape's natural flow.
    • Adjacent rows alternate direction (boustrophedon) — the needle
      turns at the contour edge rather than jumping back to the start,
      producing a smooth machine path with no wasted travel.
    • ⅓-brick tatami offset: row k shifts the stitch grid by
        (k % cycle) × (stitch_length / cycle)
      so no two adjacent rows share alignment lines → a locked, even grid.
    • Boundary stitches always placed at the exact run-start and run-end
      pixel for full coverage to the contour edge.

  Satin:
    • Long axis detected by PCA; stitch columns run perpendicular to it.
    • Mask is rotated so the long axis is horizontal; columns are scanned
      left→right; each column = one crossing stitch from edge to edge.
    • Boustrophedon columns (alternating top→bottom / bottom→top) keep
      the machine path travelling along the shape boundary between columns
      — no jump stitches needed within the satin block.

Dependencies:
    pip install opencv-python scikit-learn Pillow numpy
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

UNITS_PER_MM: float = 10.0   # 1 mm = 10 pyembroidery units (0.1 mm each)


# ── Rule set ──────────────────────────────────────────────────────────────────

@dataclass
class StitchRules:
    """All tunable parameters that drive auto-digitizing."""

    # ── Tatami fill ────────────────────────────────────────────────────────
    fill_density_mm: float = 0.4     # distance between parallel scan rows (mm)
    stitch_length_mm: float = 2.5    # max needle travel per stitch (mm)
    tatami_cycle: int = 3            # 2 = ½-brick offset, 3 = ⅓-brick offset

    # ── Satin ──────────────────────────────────────────────────────────────
    satin_col_spacing_mm: float = 0.4   # column spacing = effective density (mm)
    satin_aspect_ratio: float = 3.5     # PCA aspect-ratio threshold for satin

    # ── Run / outline ──────────────────────────────────────────────────────
    run_density_mm: float = 2.5      # spacing between run-stitch points (mm)

    # ── Classification ─────────────────────────────────────────────────────
    min_fill_area_mm2: float = 8.0   # below this area → run stitch only

    # ── Smoothing ──────────────────────────────────────────────────────────
    smooth_iters: int = 3            # Gaussian blur kernel half-size (0 = off)
                                     # 1→5 px, 2→7 px, 3→9 px

    # ── Angle override ─────────────────────────────────────────────────────
    stitch_angle: Optional[float] = None   # force fixed angle; None = auto PCA

    # ── Underlay ───────────────────────────────────────────────────────────
    underlay: bool = True            # loose edge-walk run before fill / satin


# ── Color quantization ────────────────────────────────────────────────────────

def quantize_colors(
    img_rgb: np.ndarray, n_colors: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce image to n_colors dominant colors via KMeans.

    Returns
    -------
    label_img : (H, W) int32 — cluster index per pixel
    palette   : (k, 3) uint8 — cluster-center RGB values
    """
    h, w = img_rgb.shape[:2]
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    n_clusters = min(n_colors, pixels.shape[0])
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(pixels)
    palette = km.cluster_centers_.astype(np.uint8)
    return labels.reshape(h, w).astype(np.int32), palette


# ── Shape analysis ────────────────────────────────────────────────────────────

def pca_aspect_and_angle(contour: np.ndarray) -> Tuple[float, float]:
    """
    Return (aspect_ratio, long_axis_angle_deg) via PCA on contour points.

    aspect_ratio = sqrt(λ₁ / λ₂) where λ₁ ≥ λ₂.
    A circle → ~1.0; a thin elongated strip → large number.
    """
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 3:
        return 1.0, 0.0
    _, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=np.array([]))
    ev0 = float(eigenvalues[0, 0])
    ev1 = float(eigenvalues[1, 0])
    aspect = math.sqrt(max(ev0, 1e-9) / max(ev1, 1e-9))
    angle = math.degrees(
        math.atan2(float(eigenvectors[0, 1]), float(eigenvectors[0, 0]))
    )
    return aspect, angle


# ── Mask smoothing ────────────────────────────────────────────────────────────

def smooth_mask(mask: np.ndarray, smooth_iters: int) -> np.ndarray:
    """
    Gaussian-blur a binary mask so findContours returns smooth, rounded paths.
    smooth_iters controls kernel size (0 = off).  Threshold at 127 preserves
    the approximate shape boundary without shrinkage.
    """
    if smooth_iters <= 0:
        return mask
    ksize = smooth_iters * 2 + 3     # always odd: 1→5, 2→7, 3→9
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), 0)
    _, out = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return out.astype(np.uint8)


# ── Rotation helpers ──────────────────────────────────────────────────────────

def _padded_rotation(
    mask: np.ndarray, angle_deg: float
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Pad a mask to prevent corner clipping, rotate by –angle_deg.
    Returns (rotated_mask, M_inv, pad_w, pad_h).
    """
    h, w = mask.shape
    diag = int(math.ceil(math.hypot(h, w)))
    pad_h = (diag - h) // 2 + 2
    pad_w = (diag - w) // 2 + 2
    padded = cv2.copyMakeBorder(
        mask, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0
    )
    ph, pw = padded.shape
    cx, cy = pw / 2.0, ph / 2.0
    M_fwd = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    M_inv = cv2.getRotationMatrix2D((cx, cy),  angle_deg, 1.0)
    rotated = cv2.warpAffine(padded, M_fwd, (pw, ph), flags=cv2.INTER_NEAREST)
    return rotated, M_inv, pad_w, pad_h


def _unrotate(
    rx: float, ry: float,
    M_inv: np.ndarray, pad_w: int, pad_h: int,
    px_to_units: float,
) -> Tuple[float, float]:
    """Unrotate a point from rotated-space back to original image coords
    and convert to pyembroidery 0.1-mm units."""
    pt = np.array([[[rx, ry]]], dtype=np.float32)
    pr = cv2.transform(pt, M_inv)
    return (float(pr[0, 0, 0]) - pad_w) * px_to_units, \
           (float(pr[0, 0, 1]) - pad_h) * px_to_units


# ── Stitch generators ─────────────────────────────────────────────────────────

def fill_stitch(
    mask: np.ndarray,
    angle_deg: float,
    density_px: float,
    length_px: float,
    px_to_units: float,
    cycle: int = 3,
) -> List[Dict]:
    """
    Tatami fill stitches — scan-rows parallel to angle_deg, boustrophedon.

    Row-k tatami offset = (k % cycle) × (length_px / cycle).
    With cycle=3 this produces a ⅓-brick grid: no two adjacent rows share
    common stitch-line positions, giving an even locked-grid appearance.

    Boundary stitches are placed at the exact run start and end pixel so
    the needle reaches all the way to the contour edge on every row.

    Boustrophedon: direction flips each row, so the last stitch of row k
    and the first stitch of row k+1 are at the same contour side — smooth
    machine travel with no cross-field jumps.
    """
    if density_px < 1:
        return []

    rotated, M_inv, pad_w, pad_h = _padded_rotation(mask, angle_deg)
    ph, pw = rotated.shape

    stitches: List[Dict] = []
    y = density_px / 2.0
    direction = 1
    row_idx = 0

    while y < ph:
        iy = int(round(y))
        if iy >= ph:
            break
        row = rotated[iy]
        nz = np.where(row > 127)[0]

        if nz.size > 0:
            # Split into contiguous runs (gap > 3 px = separate run)
            diffs = np.diff(nz.astype(np.int32))
            breaks = np.where(diffs > 3)[0]
            run_starts = np.concatenate([[nz[0]], nz[breaks + 1]]).astype(float)
            run_ends   = np.concatenate([nz[breaks], [nz[-1]]]).astype(float)

            offset = (row_idx % cycle) * (length_px / cycle)
            row_pts: List[Tuple[float, float]] = []

            for rs, re in zip(run_starts, run_ends):
                # Boundary anchor at run start
                row_pts.append((rs, float(iy)))
                # First tatami stitch (offset into the row)
                x = rs + offset
                # Skip if offset puts us too close to the boundary anchor
                if x - rs < length_px * 0.15:
                    x += length_px
                while x <= re:
                    row_pts.append((x, float(iy)))
                    x += length_px
                # Boundary anchor at run end
                if abs(row_pts[-1][0] - re) > length_px * 0.15:
                    row_pts.append((re, float(iy)))

            if direction == -1:
                row_pts = row_pts[::-1]

            for rx, ry in row_pts:
                ox, oy = _unrotate(rx, ry, M_inv, pad_w, pad_h, px_to_units)
                stitches.append({"x": ox, "y": oy, "jump": False})

        direction *= -1
        y += density_px
        row_idx += 1

    if stitches:
        stitches[0]["jump"] = True
    return stitches


def satin_stitch(
    mask: np.ndarray,
    long_axis_angle_deg: float,
    col_spacing_px: float,
    px_to_units: float,
) -> List[Dict]:
    """
    Satin stitch — columns perpendicular to the shape's long axis.

    Algorithm
    ---------
    1. Rotate mask so long axis → horizontal; satin columns become vertical.
    2. Walk x from left to right at col_spacing_px intervals.
    3. For each x: find topmost and bottommost mask pixel → one crossing stitch.
    4. Boustrophedon columns: alternate top→bottom / bottom→top per column.
       The machine travels along the shape boundary between columns — smooth
       path, no jump stitches needed.
    5. Unrotate each endpoint back to original image space.

    Geometry check: if long axis is at θ, stitches cross perpendicular = θ+90°.
    Rotating by –θ makes long axis horizontal → vertical in rotated space ✓.
    """
    if col_spacing_px < 1:
        return []

    rotated, M_inv, pad_w, pad_h = _padded_rotation(mask, long_axis_angle_deg)
    ph, pw = rotated.shape

    stitches: List[Dict] = []
    x = col_spacing_px / 2.0
    direction = 1   # 1 = top→bottom;  -1 = bottom→top

    while x < pw:
        ix = int(round(x))
        if ix >= pw:
            break
        col = rotated[:, ix]
        nz = np.where(col > 127)[0]

        if nz.size >= 2:
            y_top = float(nz[0])
            y_bot = float(nz[-1])
            if direction == 1:
                pairs = [(float(ix), y_top), (float(ix), y_bot)]
            else:
                pairs = [(float(ix), y_bot), (float(ix), y_top)]

            for rx, ry in pairs:
                ox, oy = _unrotate(rx, ry, M_inv, pad_w, pad_h, px_to_units)
                stitches.append({"x": ox, "y": oy, "jump": False})
            direction *= -1

        x += col_spacing_px

    if stitches:
        stitches[0]["jump"] = True
    return stitches


def run_stitch(
    contour: np.ndarray,
    spacing_px: float,
    px_to_units: float,
) -> List[Dict]:
    """
    Walk a closed contour placing stitch points every spacing_px pixels.
    Returns [{x, y, jump}]; first point is jump=True.
    """
    pts = contour.reshape(-1, 2).astype(float)
    n = len(pts)
    if n < 2:
        return []

    result = [{"x": pts[0, 0] * px_to_units, "y": pts[0, 1] * px_to_units, "jump": True}]
    carry = 0.0

    for i in range(1, n + 1):
        j = i % n
        ax, ay = pts[i - 1]
        bx, by = pts[j]
        dx, dy = bx - ax, by - ay
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-9:
            continue
        t = spacing_px - carry
        while t <= seg_len:
            f = t / seg_len
            result.append({
                "x": (ax + f * dx) * px_to_units,
                "y": (ay + f * dy) * px_to_units,
                "jump": False,
            })
            t += spacing_px
        carry = max(0.0, seg_len - (t - spacing_px))

    return result


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_image(
    img_rgb: np.ndarray,
    n_colors: int = 6,
    design_width_mm: float = 100.0,
    rules: Optional[StitchRules] = None,
    skip_background: bool = True,
) -> Tuple[List[Dict], List[str], Dict, List[Dict]]:
    """
    Full image-to-embroidery pipeline.

    Parameters
    ----------
    img_rgb          : H×W×3 uint8 RGB image
    n_colors         : number of thread colors to extract
    design_width_mm  : output design width in mm
    rules            : StitchRules (uses defaults if None)
    skip_background  : skip the most-dominant color (background)

    Returns
    -------
    segments     : [{color_index, points:[{x,y,jump}]}, …]
    thread_colors: [hex, …] indexed by color_index
    bbox         : {min_x, min_y, max_x, max_y} in pyembroidery units
    layers_info  : per-layer metadata for UI display
    """
    if rules is None:
        rules = StitchRules()

    h, w = img_rgb.shape[:2]
    px_to_units: float = (design_width_mm * UNITS_PER_MM) / w
    px_to_mm: float    = design_width_mm / w

    # Rule thresholds → pixel equivalents
    density_px    = rules.fill_density_mm      / px_to_mm
    length_px     = rules.stitch_length_mm     / px_to_mm
    run_px        = rules.run_density_mm       / px_to_mm
    satin_col_px  = rules.satin_col_spacing_mm / px_to_mm

    # ── Step 1: color quantization ──────────────────────────────────────────
    label_img, palette = quantize_colors(img_rgb, n_colors)

    # ── Step 2: build layer list, largest coverage first ───────────────────
    layer_list: List[Dict] = []
    for idx in range(len(palette)):
        layer_mask = (label_img == idx).astype(np.uint8) * 255
        area = int(np.sum(layer_mask > 0))
        r, g, b = int(palette[idx][0]), int(palette[idx][1]), int(palette[idx][2])
        layer_list.append({"idx": idx, "mask": layer_mask, "area": area,
                           "hex": f"#{r:02x}{g:02x}{b:02x}"})
    layer_list.sort(key=lambda lyr: -lyr["area"])

    if skip_background and len(layer_list) > 1:
        layer_list = layer_list[1:]

    all_segments:  List[Dict] = []
    thread_colors: List[str]  = []
    layers_info:   List[Dict] = []
    color_idx = 0

    for layer in layer_list:
        raw_mask = layer["mask"]

        # ── Step 3: smooth mask → clean, rounded contours ──────────────────
        s_mask = smooth_mask(raw_mask, rules.smooth_iters)

        contours, _ = cv2.findContours(
            s_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
        )
        contours = sorted(
            [c for c in contours if cv2.contourArea(c) > 9],
            key=lambda c: -cv2.contourArea(c),
        )
        if not contours:
            continue

        largest  = contours[0]
        area_px2 = cv2.contourArea(largest)
        area_mm2 = area_px2 * px_to_mm * px_to_mm

        # ── Step 4: shape classification via PCA ───────────────────────────
        aspect, long_angle = pca_aspect_and_angle(largest)

        if rules.stitch_angle is not None:
            fill_angle = rules.stitch_angle
            angle_auto = False
        else:
            fill_angle = long_angle     # tatami rows ‖ to long axis
            angle_auto = True

        if area_mm2 < rules.min_fill_area_mm2:
            stitch_type = "run"
        elif aspect >= rules.satin_aspect_ratio:
            stitch_type = "satin"
        else:
            stitch_type = "fill"

        segs_added   = 0
        stitch_count = 0

        # Smoothed filled mask (used for fill/satin body)
        fill_mask_img = np.zeros_like(raw_mask)
        cv2.drawContours(fill_mask_img, contours, -1, 255, cv2.FILLED)
        s_fill = smooth_mask(fill_mask_img, rules.smooth_iters)

        # ── Step 5: underlay — loose edge-walk before fill / satin ─────────
        #   Anchors the fabric, gives the needle a register path, and ensures
        #   the first fill row has a base to lock into.
        if rules.underlay and stitch_type in ("fill", "satin"):
            ul_pts = run_stitch(largest, run_px * 1.5, px_to_units)
            if len(ul_pts) >= 2:
                all_segments.append({"color_index": color_idx, "points": ul_pts})
                stitch_count += len([p for p in ul_pts if not p["jump"]])
                segs_added += 1

        # ── Step 6: main body ───────────────────────────────────────────────
        if stitch_type == "fill":
            body_pts = fill_stitch(
                s_fill, fill_angle, density_px, length_px,
                px_to_units, rules.tatami_cycle,
            )
            if body_pts:
                all_segments.append({"color_index": color_idx, "points": body_pts})
                stitch_count += len([p for p in body_pts if not p["jump"]])
                segs_added += 1

        elif stitch_type == "satin":
            body_pts = satin_stitch(s_fill, long_angle, satin_col_px, px_to_units)
            if body_pts:
                all_segments.append({"color_index": color_idx, "points": body_pts})
                stitch_count += len([p for p in body_pts if not p["jump"]])
                segs_added += 1

        # ── Step 7: run / outline ───────────────────────────────────────────
        if stitch_type == "run":
            # Small shapes: run stitch around the perimeter only
            for c in contours[:5]:
                if cv2.contourArea(c) * px_to_mm * px_to_mm < 0.5:
                    continue
                r_pts = run_stitch(c, run_px, px_to_units)
                if len(r_pts) >= 2:
                    all_segments.append({"color_index": color_idx, "points": r_pts})
                    stitch_count += len([p for p in r_pts if not p["jump"]])
                    segs_added += 1
        else:
            # Fill / satin: tight outline for sharp, clean edges
            outline_spacing = run_px * (0.6 if stitch_type == "fill" else 0.75)
            for c in contours[:5]:
                if cv2.contourArea(c) * px_to_mm * px_to_mm < 0.5:
                    continue
                o_pts = run_stitch(c, outline_spacing, px_to_units)
                if len(o_pts) >= 2:
                    all_segments.append({"color_index": color_idx, "points": o_pts})
                    stitch_count += len([p for p in o_pts if not p["jump"]])
                    segs_added += 1

        if segs_added > 0:
            thread_colors.append(layer["hex"])
            layers_info.append({
                "color_hex":     layer["hex"],
                "stitch_type":   stitch_type,
                "angle_deg":     round(fill_angle, 1),
                "angle_auto":    angle_auto,
                "aspect_ratio":  round(aspect, 2),
                "area_mm2":      round(area_mm2, 1),
                "contour_count": len(contours),
                "stitch_count":  stitch_count,
            })
            color_idx += 1

    if not all_segments:
        return [], [], {}, []

    all_x = [p["x"] for s in all_segments for p in s["points"]]
    all_y = [p["y"] for s in all_segments for p in s["points"]]
    bbox = {
        "min_x": min(all_x), "min_y": min(all_y),
        "max_x": max(all_x), "max_y": max(all_y),
    }
    return all_segments, thread_colors, bbox, layers_info
