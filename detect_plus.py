#!/usr/bin/env python3
"""
detect_plus.py  v1.12

Detects a "boxed plus" marker: a dark outer rectangle enclosing four white
corner squares arranged in a 2x2 grid, with the + shape formed by the dark
cross arms between them.

Detection pipeline:
  1. Preprocess: scale to 50%, light Gaussian blur, mild unsharp mask (USM)
     — used only for the coarse search step.
  2. Coarse: black-hat morphology on the 50% proc image finds dark-rectangle
     candidates in the road zone.
  3. Fine (_fine_detect_plus): for each candidate, crops the full-resolution
     gray image and applies adaptive threshold across a range of block sizes
     to isolate exactly four white corner squares as clean blobs.  The best-
     scoring 2x2 group is selected, subpixel corner refinement is applied to
     all 16 corners (4 squares x 4 corners), and the mean of the 4 inner
     corners gives an initial centre estimate.  The found group must be close
     to the candidate location and pass a size-uniformity check.
  4. Refine (refine_center_by_lines): re-detects the four white squares at
     full resolution using an adaptive block size scaled to the marker's
     apparent size.  Samples edge-gradient positions along each of the 8
     inner square edges (right/bottom of TL, left/bottom of TR, right/top
     of BL, left/top of BR), fits strictly parallel line pairs, and returns
     the mean of the 4 arm-line intersection points as the refined centre.
     Geometry sanity checks (square size uniformity, arm gap validity, and
     spatial ordering) reject false positives before the result is accepted.
  5. All returned coordinates are in original full-image pixels.

Can be run standalone:
    python detect_plus.py image.jpg [image2.jpg ...] [--verbose]

Or imported and called:
    from detect_plus import find_plus_markers
    results = find_plus_markers(gray, verbose=False)
"""

VERSION = "1.12"

import cv2
import numpy as np
import sys, os, argparse
from itertools import combinations

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(gray, scale=0.5, blur_r=1, usm_radius=1, usm_amount=0.3):
    """
    Prepare image for plus-marker detection:
      1. Scale to `scale` (default 0.5 = 50%)
      2. Light Gaussian blur (blur_r pixel radius)
      3. Unsharp mask (usm_radius, usm_amount)

    The USM step dramatically clarifies the four white corner squares as
    isolated bright blobs on the dark + rectangle background.
    Returns the processed image at reduced scale.
    """
    small = cv2.resize(gray, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    if blur_r > 0:
        k = blur_r * 2 + 1
        small = cv2.GaussianBlur(small, (k, k), 0)
    blur_usm  = cv2.GaussianBlur(small, (usm_radius * 2 + 1, usm_radius * 2 + 1), 0)
    sharpened = cv2.addWeighted(small, 1.0 + usm_amount, blur_usm, -usm_amount, 0)
    return sharpened

# ── Geometry helpers (mirrors detect_marker.py) ──────────────────────────────

def _order_rect(pts):
    """Order 4 points as TL, TR, BR, BL."""
    pts = np.array(pts, dtype=np.float32)
    pts = pts[np.argsort(pts[:, 1])]
    top = pts[:2][np.argsort(pts[:2, 0])]
    bot = pts[2:][np.argsort(pts[2:, 0])]
    return np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float32)


def _subpix_corners(gray, box, win=4):
    """Refine corner positions to sub-pixel accuracy."""
    h, w = gray.shape
    pts = np.clip(box, win + 1, min(h, w) - win - 2).astype(np.float32).reshape(-1, 1, 2)
    try:
        r = cv2.cornerSubPix(
            gray, pts, (win, win), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
        return r.reshape(-1, 2)
    except cv2.error:
        return box


def _rect_corners_from_contour(c):
    """Return 4 ordered corners from a contour via minAreaRect."""
    rect = cv2.minAreaRect(c)
    box  = cv2.boxPoints(rect).astype(np.float32)
    return _order_rect(box)


# ── Coarse search ─────────────────────────────────────────────────────────────

def _coarse_candidates(proc, scale=0.5, tophat_k=5, thresh=15,
                       min_area=8, max_area=600, max_aspect=1.8):
    """
    Find candidate dark-rectangle centers in the preprocessed (50%) image
    using black-hat morphology. Returns (cx, cy) in ORIGINAL full-image coords.
    """
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (tophat_k, tophat_k))
    blackhat = cv2.morphologyEx(proc, cv2.MORPH_BLACKHAT, kernel)
    _, th    = cv2.threshold(blackhat, thresh, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hits = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area < area < max_area):
            continue
        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        if rw < 1 or rh < 1:
            continue
        if max(rw, rh) / min(rw, rh) > max_aspect:
            continue
        cx, cy = rect[0]
        # Convert back to full-image coords
        hits.append((cx / scale, cy / scale))
    return hits


# ── Fine detection ────────────────────────────────────────────────────────────

def _fine_detect_plus(proc, cx_full, cy_full, scale=0.5, pad_full=120,
                      verbose=False):
    """
    Detect a boxed-plus marker near (cx_full, cy_full) in full-image coords.

    Despite the parameter name, `proc` is the 50%-scale preprocessed image
    used only to define the crop region; all detection runs on the full-
    resolution gray image passed to find_plus_markers (via gray captured in
    the closure — or via the `proc` image when called directly for testing).

    Crops a pad_fullxpad_full region around the candidate, applies adaptive
    threshold across block sizes (11,15) at 50% scale to find four white-
    square blobs in a 2x2 arrangement, scores all C(12,4)=495 quad
    combinations per threshold, subpixel-refines all 16 corners, and returns
    a result dict with centre, apparent size, and geometry.  The found group
    must be within 80% of pad_full of the candidate centre.

    All returned coordinates are in original full-image pixels.
    """
    h_proc, w_proc = proc.shape
    cx  = cx_full * scale
    cy  = cy_full * scale
    pad = int(pad_full * scale)

    x0 = max(0, int(cx) - pad)
    y0 = max(0, int(cy) - pad)
    x1 = min(w_proc, int(cx) + pad)
    y1 = min(h_proc, int(cy) + pad)
    crop = proc[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    off_proc = np.array([x0, y0], dtype=np.float32)

    # Quick pre-check on first threshold: if there aren't at least 4 small
    # similarly-sized blobs, skip this candidate entirely — it can't be a + box.
    adapt0 = cv2.adaptiveThreshold(crop, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, -5)
    ctrs0, _ = cv2.findContours(adapt0, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    small_blobs = [(cv2.contourArea(c), cv2.minAreaRect(c))
                   for c in ctrs0
                   if 5 < cv2.contourArea(c) < 800]
    small_blobs = [(a, r) for a, r in small_blobs
                   if r[1][0] >= 2 and r[1][1] >= 2
                   and max(r[1]) / min(r[1]) <= 3.0]
    if len(small_blobs) < 4:
        return None
    # Check uniformity: largest/smallest area ratio among top-4
    top4_areas = sorted([a for a,r in small_blobs], reverse=True)[:4]
    if top4_areas[-1] < 1 or top4_areas[0] / top4_areas[-1] > 8.0:
        return None

    # Adaptive threshold at 50% scale: sweep (block, C) pairs.
    # block=11 and block=15 are appropriate for white squares that are ~7-15px
    # at 50% scale (marker at 13-20m distance).  The best-scoring valid 2x2
    # group across all threshold settings is kept.
    best_group = None
    best_score = -1

    for block, C in [(11, 5), (15, 5), (11, 8), (15, 8)]:
        adapt = cv2.adaptiveThreshold(crop, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block, -C)
        ctrs, _ = cv2.findContours(adapt, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in ctrs:
            area = cv2.contourArea(c)
            if not (5 < area < 2000):
                continue
            rect = cv2.minAreaRect(c)
            rw, rh = rect[1]
            if rw < 2 or rh < 2:
                continue
            if max(rw, rh) / min(rw, rh) > 3.0:
                continue
            candidates.append((area, rw, rh, rect[0], c))

        if len(candidates) < 4:
            continue

        top = sorted(candidates, key=lambda x: -x[0])[:12]
        for quad in combinations(range(len(top)), 4):
            group   = [top[k] for k in quad]
            centers = np.array([r[3] for r in group])
            xs = np.sort(centers[:, 0])
            ys = np.sort(centers[:, 1])
            x_spread = xs[3] - xs[0]
            y_spread = ys[3] - ys[0]
            if x_spread < 3 or y_spread < 3:
                continue
            # 2x2: gap between inner pair > 15% of spread
            if (xs[2]-xs[1]) < 0.15*x_spread or (ys[2]-ys[1]) < 0.15*y_spread:
                continue
            # Roughly square arrangement
            if max(x_spread, y_spread) / max(min(x_spread, y_spread), 1) > 3.5:
                continue
            # Similar areas
            areas = [r[0] for r in group]
            if max(areas) / max(min(areas), 1) > 3.0:
                continue
            # Similar widths
            widths = [max(r[1], r[2]) for r in group]
            if max(widths) / max(min(widths), 1) > 2.0:
                continue
            # Gap between children should be similar to child size
            # (arm width ≈ white square size in a well-designed + box)
            child_w = np.mean([max(r[1],r[2]) for r in group])
            child_h = np.mean([min(r[1],r[2]) for r in group])
            x_gap = xs[2] - xs[1]
            y_gap = ys[2] - ys[1]
            if not (0.3 < x_gap/max(child_w,1) < 5.0):
                continue
            if not (0.3 < y_gap/max(child_h,1) < 5.0):
                continue

            mean_area   = np.mean(areas)
            uniformity  = min(areas) / max(max(areas), 1)
            squareness  = min(x_spread, y_spread) / max(max(x_spread, y_spread), 1)
            score = mean_area*10 + uniformity*500 + squareness*200

            if score > best_score:
                best_score = score
                best_group = group

    if best_group is None:
        return None

    # The group's centre must be near the candidate location
    group_cx = np.mean([r[3][0] for r in best_group]) + x0
    group_cy = np.mean([r[3][1] for r in best_group]) + y0
    if abs(group_cx - cx_full) > pad * 0.8 or abs(group_cy - cy_full) > pad * 0.8:
        return None

    # Order spatially: by (y-bucket, x)
    best_group = sorted(best_group, key=lambda r: (round(r[3][1]/3), r[3][0]))

    child_corners_list = []
    for _, rw, rh, (rcx, rcy), c in best_group:
        box      = _rect_corners_from_contour(c)
        ref_proc = _subpix_corners(crop, box) + off_proc
        ref_full = ref_proc / scale
        child_corners_list.append(ref_full)

    all_corners = np.vstack(child_corners_list)   # 16x2 full-image coords
    child_sizes = [np.linalg.norm(c[1]-c[0]) for c in child_corners_list]
    half_csize  = np.mean(child_sizes) * 0.5

    x_min = all_corners[:,0].min() - half_csize
    x_max = all_corners[:,0].max() + half_csize
    y_min = all_corners[:,1].min() - half_csize
    y_max = all_corners[:,1].max() + half_csize
    outer_corners = np.array([[x_min,y_min],[x_max,y_min],
                               [x_max,y_max],[x_min,y_max]], dtype=np.float32)

    # apparent_px and wh_ratio based on child-centre span, not the padded outer
    # box.  The centre span scales cleanly with camera distance and is
    # independent of the half_csize padding, giving a stable size measure.
    child_centers = np.array([c.mean(axis=0) for c in child_corners_list])
    span_w = child_centers[:,0].max() - child_centers[:,0].min()
    span_h = child_centers[:,1].max() - child_centers[:,1].min()
    apparent_px = (span_w * span_h) ** 0.5
    if (apparent_px < 10 or apparent_px > 40):
        # Reject candidates with implausible apparent size (too small = far away,
        # too large = close but not detected in coarse search).
        return None
    wh_ratio    = span_w / span_h if span_h > 0 else 1.0

    # Center: mean of the 4 inner corners (the corners of each white square
    # that face the arm gap). These subpixel-refine to the arm-crossing point,
    # giving a more accurate center than the mean of all 16 corners.
    # Child order: [TL-child, TR-child, BL-child, BR-child]
    # Inner corner of each: BR(idx 2), BL(idx 3), TR(idx 1), TL(idx 0)
    inner_corners = np.array([
        child_corners_list[0][2],   # TL-child's BR
        child_corners_list[1][3],   # TR-child's BL
        child_corners_list[2][1],   # BL-child's TR
        child_corners_list[3][0],   # BR-child's TL
    ])
    center = inner_corners.mean(axis=0)

    # inner_mean: average gray of the four white squares in preprocessed image
    ccs = np.array([c.mean(axis=0) for c in child_corners_list])
    inner_mean = float(np.mean([
        proc[int(np.clip(pt[1]*scale, 0, h_proc-1)),
             int(np.clip(pt[0]*scale, 0, w_proc-1))]
        for pt in ccs]))

    return {
        'center':        center,
        'outer_corners': outer_corners,
        'child_corners': child_corners_list,
        'all_corners':   all_corners,
        'apparent_px':   apparent_px,
        'wh_ratio':      wh_ratio,
        'inner_mean':    inner_mean,
        'n_children':    4,
    }


# ── Line-intersection centre refinement ──────────────────────────────────────

def _nearest_odd(n):
    """Return nearest odd integer >= n, minimum 3."""
    n = max(3, int(n))
    return n if n % 2 == 1 else n + 1


def refine_center_by_lines(gray, result, hw=4, n_samples=30, trim=0.15):
    """
    Refine the + centre using edge-line fitting from full-resolution square
    bounding boxes.

    Re-detects the four white squares directly on the full-resolution gray
    image using adaptive threshold with block sizes scaled to the marker's
    apparent size (trying 2x and 3x the estimated square width, plus a fixed
    fallback of 21).  One erosion pass ensures bounding boxes sit inside the
    white regions rather than straddling the edges.

    For each of the 8 inner square edges (TL/BL right edges → V_left arm;
    TR/BR left edges → V_right arm; TL/TR bottom edges → H_top arm;
    BL/BR top edges → H_bot arm) samples edge-gradient positions along only
    that square's own span.  Fits a line to each set of samples, then enforces
    strict parallelism within each arm pair by averaging directions.  The
    final centre is the mean of the 4 arm-line intersection points.

    Before accepting the result, applies geometry sanity checks:
      — Four square areas within 4x of each other
      — Four square linear sizes within 2.5x of each other
      — Both arm gaps positive (no overlap)
      — Both arm gaps ≤ 3x average square size
      — H and V arm gaps within 5x of each other
      — Spatial ordering valid: TL.x < TR.x, BL.x < BR.x,
        TL.y < BL.y, TR.y < BR.y

    Returns updated result dict with 'center' replaced by the refined value,
    '_squares_bbox', '_lines', and '_crossings' added.  Returns the original
    result unchanged (without '_squares_bbox') if any check fails, allowing
    the caller to detect and reject the candidate.
    """
    # ── Re-detect white squares at full resolution ────────────────────────────
    gray_full = gray
    cx_est = float(result['center'][0])
    cy_est = float(result['center'][1])

    # Scale the crop pad with the apparent marker size so we always have
    # enough margin even for larger (closer) markers.
    apparent = result.get('apparent_px', 25)
    # apparent_px is child-centre span ≈ arm_pitch.
    # Each white square ≈ apparent/1.5 px wide at full res.
    # Pad needs to comfortably contain 2 squares + arm gap on each side.
    pad = max(55, int(apparent * 3.5))
    x0 = max(0, int(cx_est) - pad)
    y0 = max(0, int(cy_est) - pad)
    x1 = min(gray_full.shape[1], int(cx_est) + pad)
    y1 = min(gray_full.shape[0], int(cy_est) + pad)
    crop = gray_full[y0:y1, x0:x1]
    if crop.size == 0:
        return result

    # Adaptive threshold block size must be larger than one white square
    # but smaller than two squares.  Estimated white square width ≈ apparent/1.5.
    # Try several block sizes from small to large; keep whichever finds 4 squares.
    sq_est = max(5, int(apparent / 1.5))
    block_sizes = sorted(set([
        _nearest_odd(sq_est * 2 + 1),   # ~2x square width
        _nearest_odd(sq_est * 3 + 1),   # ~3x square width
        21,                              # fixed fallback
    ]))

    squares = []
    for block in block_sizes:
        adapt = cv2.adaptiveThreshold(crop, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block, -8)
        adapt = cv2.erode(adapt, np.ones((3,3), np.uint8), iterations=1)
        ctrs, _ = cv2.findContours(adapt, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for c in ctrs:
            area = cv2.contourArea(c)
            if not (20 < area < 5000): continue
            rect = cv2.minAreaRect(c)
            rw, rh = rect[1]
            max_sq_px = sq_est * 3
            if not (4 < rw < max_sq_px and 4 < rh < max_sq_px): continue
            if max(rw, rh) / min(rw, rh) > 2.5: continue
            bx, by, bw, bh = cv2.boundingRect(c)
            cx_f, cy_f = bx + x0 + bw // 2, by + y0 + bh // 2
            cands.append((area, c, rect, bx+x0, by+y0, bw, bh, cx_f, cy_f))
        cands.sort(key=lambda s: (round(s[8] / 10), s[7]))
        if len(cands) >= 4:
            squares = cands
            break   # found enough with this block size

    squares.sort(key=lambda s: (round(s[8] / 10), s[7]))
    if len(squares) != 4:
        return result   # can't refine

    tl, tr, bl, br = squares   # each: (area, c, rect, bx, by, bw, bh, cx, cy)

    # Validate spatial ordering: after sorting, TL must be left of TR,
    # BL must be left of BR, and top row must be above bottom row.
    # If violated the four blobs are not in a genuine 2x2 arrangement.
    if tl[7] >= tr[7]:   # TL.cx >= TR.cx
        return result
    if bl[7] >= br[7]:   # BL.cx >= BR.cx
        return result
    if tl[8] >= bl[8]:   # TL.cy >= BL.cy
        return result
    if tr[8] >= br[8]:   # TR.cy >= BR.cy
        return result

    # ── Sample edge-gradient positions along each inner rectangle edge ────────
    def sample_edge(x_range, y_fixed, direction):
        """
        Vectorized edge-gradient sampler. Builds the full pixel-sample grid
        as numpy arrays instead of a Python loop, making it ~10x faster.
        """
        if direction == 'h':
            p1 = np.array([float(x_range[0]), float(y_fixed)])
            p2 = np.array([float(x_range[1]), float(y_fixed)])
            perp_idx = 1    # gradient measured in y
        else:
            p1 = np.array([float(y_fixed), float(x_range[0])])
            p2 = np.array([float(y_fixed), float(x_range[1])])
            perp_idx = 0    # gradient measured in x
        L = np.linalg.norm(p2 - p1)
        if L < 2: return None
        tang = (p2 - p1) / L

        # Sample positions along the edge
        ts = np.linspace(trim * L, (1 - trim) * L, n_samples)
        base_pts = p1 + ts[:, None] * tang    # (n_samples, 2)

        # Perpendicular offsets
        ds = np.linspace(-hw, hw, hw * 4 + 1)  # (n_d,)
        # All sample coords: (n_samples, n_d, 2)
        if direction == 'h':
            xs = np.round(base_pts[:, 0]).astype(int)             # (n,)
            ys = np.round(base_pts[:, 1:2] + ds[None, :]).astype(int)  # (n, n_d)
            # Clip to image bounds
            xs_c = np.clip(xs, 0, gray_full.shape[1]-1)
            ys_c = np.clip(ys, 0, gray_full.shape[0]-1)
            valid = (ys >= 0) & (ys < gray_full.shape[0])
            vals  = gray_full[ys_c, xs_c[:, None]].astype(float)  # (n, n_d)
        else:
            ys = np.round(base_pts[:, 1]).astype(int)
            xs = np.round(base_pts[:, 0:1] + ds[None, :]).astype(int)
            xs_c = np.clip(xs, 0, gray_full.shape[1]-1)
            ys_c = np.clip(ys, 0, gray_full.shape[0]-1)
            valid = (xs >= 0) & (xs < gray_full.shape[1])
            vals  = gray_full[ys_c[:, None], xs_c].astype(float)

        vals[~valid] = np.nan

        # For each sample position, find edge via gradient-weighted centroid
        pts = []
        for i in range(n_samples):
            v = vals[i]
            ok = ~np.isnan(v)
            if ok.sum() < 3: continue
            v_clean = np.where(ok, v, np.nanmean(v))
            g = np.abs(np.gradient(v_clean))
            if g.sum() < 1: continue
            d_e = float((g * ds).sum() / g.sum())
            pt = base_pts[i].copy()
            pt[perp_idx] += d_e
            pts.append(pt)
        return np.array(pts, np.float32) if len(pts) >= 3 else None

    def fit_line(pts):
        res = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        return (np.array([float(res[2]), float(res[3])]),
                np.array([float(res[0]), float(res[1])]))

    def avg_dir(d1, d2):
        if np.dot(d1, d2) < 0: d2 = -d2
        d = d1 + d2; return d / np.linalg.norm(d)

    def line_from_pts_fixed_dir(pts, direction):
        """Best-fit line through pts with a fixed direction (min perp residual)."""
        perp   = np.array([-direction[1], direction[0]])
        offset = float(np.mean(pts @ perp))
        return offset * perp, direction

    def intersect(p1, d1, p2, d2):
        A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
        if abs(np.linalg.det(A)) < 1e-8: return None
        return p1 + float(np.linalg.solve(A, p2 - p1)[0]) * d1

    # Collect edge samples — each over its own rectangle's span only
    bx = [s[3] for s in squares]; by_ = [s[4] for s in squares]
    bw = [s[5] for s in squares]; bh  = [s[6] for s in squares]

    samples = {
        'TL_right': sample_edge((by_[0], by_[0]+bh[0]), bx[0]+bw[0], 'v'),
        'BL_right': sample_edge((by_[2], by_[2]+bh[2]), bx[2]+bw[2], 'v'),
        'TR_left':  sample_edge((by_[1], by_[1]+bh[1]), bx[1],        'v'),
        'BR_left':  sample_edge((by_[3], by_[3]+bh[3]), bx[3],        'v'),
        'TL_bot':   sample_edge((bx[0], bx[0]+bw[0]), by_[0]+bh[0],  'h'),
        'TR_bot':   sample_edge((bx[1], bx[1]+bw[1]), by_[1]+bh[1],  'h'),
        'BL_top':   sample_edge((bx[2], bx[2]+bw[2]), by_[2],         'h'),
        'BR_top':   sample_edge((bx[3], bx[3]+bw[3]), by_[3],         'h'),
    }

    fitted = {name: fit_line(pts)
              for name, pts in samples.items() if pts is not None}
    required = ['TL_right','BL_right','TR_left','BR_left',
                'TL_bot','TR_bot','BL_top','BR_top']
    if not all(k in fitted for k in required):
        return result

    # Enforce strict parallelism within each arm, then average both arms
    d_v = avg_dir(
        avg_dir(fitted['TL_right'][1], fitted['BL_right'][1]),
        avg_dir(fitted['TR_left'][1],  fitted['BR_left'][1]))
    d_h = avg_dir(
        avg_dir(fitted['TL_bot'][1], fitted['TR_bot'][1]),
        avg_dir(fitted['BL_top'][1], fitted['BR_top'][1]))

    # Refit all 4 arm lines with the shared direction
    all_v_left  = np.vstack([samples['TL_right'], samples['BL_right']])
    all_v_right = np.vstack([samples['TR_left'],  samples['BR_left']])
    all_h_top   = np.vstack([samples['TL_bot'],   samples['TR_bot']])
    all_h_bot   = np.vstack([samples['BL_top'],   samples['BR_top']])

    L_vleft  = line_from_pts_fixed_dir(all_v_left,  d_v)
    L_vright = line_from_pts_fixed_dir(all_v_right, d_v)
    L_htop   = line_from_pts_fixed_dir(all_h_top,   d_h)
    L_hbot   = line_from_pts_fixed_dir(all_h_bot,   d_h)

    crossings = [intersect(*L_vleft,  *L_htop),
                 intersect(*L_vleft,  *L_hbot),
                 intersect(*L_vright, *L_htop),
                 intersect(*L_vright, *L_hbot)]
    crossings = [c for c in crossings if c is not None]
    if not crossings:
        return result

    refined = np.mean(crossings, axis=0)

    # ── Geometry sanity checks on the four detected squares ───────────────────
    # These catch false positives where 4 blobs happen to form a rough 2x2 but
    # are clearly not a + marker.
    sq_areas = [bw * bh for bx, by, bw, bh in [(s[3],s[4],s[5],s[6])
                                                 for s in squares]]
    sq_sizes  = [max(bw, bh) for bx, by, bw, bh in [(s[3],s[4],s[5],s[6])
                                                      for s in squares]]

    # 1. Square size uniformity: all four blobs should be similar in size.
    #    A genuine + has 4 nearly identical white squares.
    if min(sq_areas) < 1 or max(sq_areas) / min(sq_areas) > 4.0:
        return result   # blobs too dissimilar in area

    if max(sq_sizes) / max(min(sq_sizes), 1) > 2.5:
        return result   # blobs too dissimilar in linear size

    # 2. Arm gap validity: both arm gaps must be positive and similar to each
    #    other. A negative gap means blobs overlap; a huge gap asymmetry means
    #    the blobs aren't in a true 2x2 arrangement.
    order = sorted(range(4), key=lambda k: (round(squares[k][8]/10), squares[k][7]))
    tl_s, tr_s, bl_s, br_s = [squares[k] for k in order]
    v_gap = tr_s[3] - (tl_s[3] + tl_s[5])   # TR.left  - TL.right
    h_gap = bl_s[4] - (tl_s[4] + tl_s[6])   # BL.top   - TL.bottom
    if v_gap < 0 or h_gap < 0:
        return result   # blobs overlap — not a valid + layout

    avg_sq = np.mean(sq_sizes)
    if v_gap > avg_sq * 3 or h_gap > avg_sq * 3:
        return result   # arm gap far too large relative to square size

    if max(v_gap, h_gap) > max(min(v_gap, h_gap), 1) * 5:
        return result   # H and V gaps wildly different

    updated = dict(result)
    updated['center_corners'] = result['center']
    updated['center']         = refined
    updated['_lines'] = {'V_left': L_vleft, 'V_right': L_vright,
                         'H_top':  L_htop,  'H_bot':   L_hbot,
                         'dir_v': d_v, 'dir_h': d_h}
    updated['_squares_bbox'] = [(s[3],s[4],s[5],s[6]) for s in squares]
    updated['_crossings'] = crossings
    return updated


# ── Public interface ──────────────────────────────────────────────────────────

def find_plus_markers(gray,
                      y0_road=380, y1_road=550,
                      x0_road=300, x1_road=1550,
                      min_apparent=10, max_apparent=100,
                      min_inner_mean=140,
                      scale=0.5,
                      verbose=False):
    """
    Search for boxed-plus markers in `gray` (full-frame grayscale image).

    Preprocessing (scale + USM) is applied once to the whole image before
    both coarse and fine detection. All returned coordinates are in original
    full-image pixels.

    Returns a list of result dicts (see _fine_detect_plus for keys), each
    with additional keys:
      'rcx', 'rcy'  — the refined center in full-image coordinates
    """
    h_img, w_img = gray.shape

    # Preprocess full image once
    proc = preprocess(gray, scale=scale)

    # Road zone in proc coords
    y0p = max(0, int(y0_road * scale))
    y1p = min(proc.shape[0], int(y1_road * scale))
    x0p = max(0, int(x0_road * scale))
    x1p = min(proc.shape[1], int(x1_road * scale))
    proc_road = proc[y0p:y1p, x0p:x1p]

    # Coarse candidates on preprocessed road zone.
    # _coarse_candidates returns full-image coords (it divides by scale internally),
    # but proc_road is already cropped so we need to add the road zone offset.
    hits_road = _coarse_candidates(proc_road, scale=1.0)
    hits = [(hx + x0_road, hy + y0_road) for hx, hy in hits_road]

    if verbose:
        print(f"  [plus] Coarse candidates: {len(hits)}")
        for hx, hy in hits:
            print(f"    ({hx:.0f}, {hy:.0f})")

    results = []
    seen    = []

    def try_candidate(cx, cy):
        if any(abs(cx - s[0]) < 30 and abs(cy - s[1]) < 30 for s in seen):
            return
        r = _fine_detect_plus(proc, cx, cy, scale=scale, verbose=verbose)

        if r is None:
            # Proc-scale failed — try full-resolution detection directly.
            # Build a minimal stub result so refine_center_by_lines can run.
            stub = {'center': np.array([float(cx), float(cy)]),
                    'apparent_px': 25.0, 'wh_ratio': 1.0, 'inner_mean': 200}
            r_full = refine_center_by_lines(gray, stub)
            if '_squares_bbox' in r_full:
                # Measure apparent_px and inner_mean from the full-res squares
                sb = r_full['_squares_bbox']
                if len(sb) == 4:
                    scs = np.array([(bx+bw//2, by+bh//2)
                                    for bx,by,bw,bh in sb])
                    span_w = float(scs[:,0].max() - scs[:,0].min())
                    span_h = float(scs[:,1].max() - scs[:,1].min())
                    app = float(np.sqrt(span_w * span_h))
                    wh  = span_w / span_h if span_h > 0 else 1.0
                    # inner_mean: sample gray at square centres
                    im  = float(np.mean([
                        gray[int(np.clip(by+bh//2, 0, gray.shape[0]-1)),
                             int(np.clip(bx+bw//2, 0, gray.shape[1]-1))]
                        for bx,by,bw,bh in sb]))
                    r_full['apparent_px'] = app
                    r_full['wh_ratio']    = wh
                    r_full['inner_mean']  = im
                    r = r_full
            if r is None or '_squares_bbox' not in r:
                if verbose:
                    print(f"    ({cx:.0f},{cy:.0f}): fine detection failed")
                return

        rcx, rcy = float(r['center'][0]), float(r['center'][1])

        # Road zone check on refined center
        if not (x0_road <= rcx <= x1_road and y0_road <= rcy <= y1_road):
            if verbose:
                print(f"    ({cx:.0f},{cy:.0f}): rejected "
                      f"(center ({rcx:.0f},{rcy:.0f}) outside road zone)")
            return

        # Apparent size gate
        if not (min_apparent <= r['apparent_px'] <= max_apparent):
            if verbose:
                print(f"    ({cx:.0f},{cy:.0f}): rejected "
                      f"(apparent={r['apparent_px']:.1f} out of range "
                      f"{min_apparent}..{max_apparent})")
            return

        # Width/height ratio: a + box is roughly square; reject tall/wide distortions
        if r['wh_ratio'] < 0.65 or r['wh_ratio'] > 1.40:
            if verbose:
                print(f"    ({cx:.0f},{cy:.0f}): rejected "
                      f"(wh_ratio={r['wh_ratio']:.3f} outside 0.65..1.40)")
            return

        # Inner brightness (should be white inside)
        if r['inner_mean'] < min_inner_mean:
            if verbose:
                print(f"    ({cx:.0f},{cy:.0f}): rejected "
                      f"(inner_mean={r['inner_mean']:.1f} < {min_inner_mean})")
            return

        # Duplicate check: keep best apparent_px if overlapping
        dup_idx = next((i for i, s in enumerate(seen)
                        if abs(rcx - s[0]) < 25 and abs(rcy - s[1]) < 25), None)
        if dup_idx is not None:
            # Keep the one with more square outer rect (wh_ratio closer to 1)
            existing = results[dup_idx]
            if abs(r['wh_ratio'] - 1.0) < abs(existing['wh_ratio'] - 1.0):
                r['rcx'] = rcx;  r['rcy'] = rcy
                r = refine_center_by_lines(gray, r)
                if '_squares_bbox' not in r:
                    return
                r['rcx'] = float(r['center'][0])
                r['rcy'] = float(r['center'][1])
                results[dup_idx] = r
                seen[dup_idx]    = (r['rcx'], r['rcy'])
            return

        r['rcx'] = rcx;  r['rcy'] = rcy
        # Refine center via arm edge-line intersection
        r = refine_center_by_lines(gray, r)
        if '_squares_bbox' not in r:
            if verbose:
                print(f"    ({cx:.0f},{cy:.0f}): rejected "
                      f"(refine geometry check failed)")
            return
        r['rcx'] = float(r['center'][0])
        r['rcy'] = float(r['center'][1])
        seen.append((r['rcx'], r['rcy']))
        results.append(r)

    for cx, cy in hits:
        try_candidate(cx, cy)

    # Dense grid fallback if nothing found
    if not results:
        if verbose:
            print("  [plus] No coarse hit — trying dense grid fallback...")
        step = 40
        for gy in range(y0_road + 30, y1_road - 30, step):
            for gx in range(x0_road + 30, x1_road - 30, step):
                try_candidate(gx, gy)

    return results


# ── Annotation helper ─────────────────────────────────────────────────────────

def annotate_plus(img_bgr, results):
    """Draw detected boxed-plus results onto a copy of img_bgr."""
    vis = img_bgr.copy()
    colors = {'V_left': (255,120,0), 'V_right': (200,220,0),
              'H_top':  (0,80,255),  'H_bot':   (0,200,255)}
    for r in results:
        # Axis-aligned white-square bounding boxes (green)
        if '_squares_bbox' in r:
            for bx,by,bw,bh in r['_squares_bbox']:
                cv2.rectangle(vis, (bx,by), (bx+bw,by+bh), (0,220,0), 1)
        else:
            for child in r['child_corners']:
                cv2.polylines(vis, [child.astype(np.int32)], True, (0,220,0), 1)

        # Fitted arm lines (extended across the marker area only)
        if '_lines' in r and '_squares_bbox' in r:
            bboxes = r['_squares_bbox']
            # Clip lines to draw only within the + area (not across whole image)
            all_bx = [b[0] for b in bboxes]; all_by = [b[1] for b in bboxes]
            all_rx = [b[0]+b[2] for b in bboxes]; all_ry = [b[1]+b[3] for b in bboxes]
            margin = 8
            clip_x0, clip_x1 = min(all_bx)-margin, max(all_rx)+margin
            clip_y0, clip_y1 = min(all_by)-margin, max(all_ry)+margin
            for name in ('V_left','V_right','H_top','H_bot'):
                pt, dv = r['_lines'][name]
                p1 = pt - 80*dv;  p2 = pt + 80*dv
                # Clip to + area
                p1c = (int(np.clip(p1[0],clip_x0,clip_x1)),
                       int(np.clip(p1[1],clip_y0,clip_y1)))
                p2c = (int(np.clip(p2[0],clip_x0,clip_x1)),
                       int(np.clip(p2[1],clip_y0,clip_y1)))
                cv2.line(vis, p1c, p2c, colors[name], 1, cv2.LINE_AA)

        # Four crossing points (cyan dots)
        if '_crossings' in r:
            for pt in r['_crossings']:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0,255,255), -1)

        # Center crosshair (red)
        cx, cy = int(r['rcx']), int(r['rcy'])
        cv2.drawMarker(vis, (cx,cy), (0,0,255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(vis, f"PLUS ({r['rcx']:.1f},{r['rcy']:.1f})",
                    (cx-40, cy-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0,0,255), 1, cv2.LINE_AA)
    return vis


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Detect boxed-plus markers in images")
    ap.add_argument("inputs", nargs="+",
                    help="Image files (.jpg/.png) or directories")
    ap.add_argument("--version", action="version",
                    version=f"detect_plus.py v{VERSION}")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--road-y", type=int, nargs=2, metavar=("Y0", "Y1"),
                    default=None, help="Road zone rows, e.g. --road-y 380 550")
    ap.add_argument("--road-x", type=int, nargs=2, metavar=("X0", "X1"),
                    default=None, help="Road zone cols, e.g. --road-x 300 1550")
    args = ap.parse_args()

    road_y = args.road_y or [380, 550]
    road_x = args.road_x or [300, 1550]

    # Expand directories
    paths = []
    for inp in args.inputs:
        if os.path.isdir(inp):
            paths.extend(sorted(
                os.path.join(inp, f) for f in os.listdir(inp)
                if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png')))
        elif os.path.isfile(inp):
            paths.append(inp)
        else:
            print(f"Warning: {inp} not found, skipping")

    if not paths:
        sys.exit("No image files to process.")

    for path in paths:
        img  = cv2.imread(path)
        if img is None:
            print(f"Cannot read {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if args.verbose:
            print(f"\n{'='*60}")
            print(f"Image: {os.path.basename(path)}  ({img.shape[1]}x{img.shape[0]})")

        results = find_plus_markers(
            gray,
            y0_road=road_y[0], y1_road=road_y[1],
            x0_road=road_x[0], x1_road=road_x[1],
            verbose=args.verbose)

        if not results:
            print(f"{os.path.basename(path)}: no plus marker detected")
        else:
            for r in results:
                print(f"{os.path.basename(path)}: PLUS  "
                      f"center=({r['rcx']:.1f},{r['rcy']:.1f})  "
                      f"TL=({r['outer_corners'][0,0]:.1f},{r['outer_corners'][0,1]:.1f})  "
                      f"TR=({r['outer_corners'][1,0]:.1f},{r['outer_corners'][1,1]:.1f})  "
                      f"BR=({r['outer_corners'][2,0]:.1f},{r['outer_corners'][2,1]:.1f})  "
                      f"BL=({r['outer_corners'][3,0]:.1f},{r['outer_corners'][3,1]:.1f})  "
                      f"apparent={r['apparent_px']:.1f}px  "
                      f"inner_mean={r['inner_mean']:.0f}  "
                      f"wh={r['wh_ratio']:.3f}")
                if args.verbose:
                    print(f"  16 corners (all_corners mean = center):")
                    for k, pt in enumerate(r['all_corners']):
                        child_idx = k // 4
                        corner_idx = k % 4
                        names = ['TL','TR','BR','BL']
                        print(f"    child[{child_idx}].{names[corner_idx]}: "
                              f"({pt[0]:.2f},{pt[1]:.2f})")

            # Save annotated image
            base = os.path.splitext(path)[0]
            out  = base + '_plus_detected.jpg'
            cv2.imwrite(out, annotate_plus(img, results))
            if args.verbose:
                print(f"  Annotated image saved: {out}")


if __name__ == "__main__":
    main()
