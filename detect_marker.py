#!/usr/bin/env python3
"""
detect_marker.py  v1.0

Locates printed tracking markers in a full 1920x1080 image for camera calibration purposes.
Uses a coarse-to-fine pipeline:
  1. Coarse (1/4 scale): tophat morphology finds bright compact rectangles
  2. Verify: nested contour structure (CLAHE + multi-threshold) confirms identity
  3. Fine:   homography + subpixel refinement of all features
  4. Validate: white-field uniformity, apparent size, inlier count, road zone

Usage:
    python detect_marker.py image.jpg [image2.jpg ...] [options]
    python detect_marker.py /path/to/frame/directory/ [options]

Options:
    --verbose       Print all candidates and rejection reasons
    --max-dist N    Maximum detection distance in meters (default: 25.0)
    --debug         Save annotated debug crops

Changelog:
    v1.0  Duplicate detection resolved by quality comparison
          (n_inliers then wh_ratio); better of two overlapping
          detections kept; printing deferred until after all candidates
          processed; --road-x argument added.
    v0.9  Width/height ratio check (w/h >= 0.80) rejects tall/narrow spurious
          detections; adaptive ID fill threshold (white_level * 0.70, fallback 110);
          max apparent size raised to 135px; contour scoring uses squareness not area.
    v0.6  Adaptive ID fill threshold; road zone default 33%-48% of height;
          --road-y argument for per-camera tuning; median white_level.
    v0.5  Multi-threshold binary selection; max apparent size gate;
          road-zone center check; relaxed inlier count for small markers;
          directory input support; distance reporting.
    v0.4  White-field uniformity filter (wf_mean, wf_smooth).
    v0.3  CLAHE preprocessing; dense grid fallback; --verbose flag.
    v0.2  SuperPoint-style nested contour hierarchy; homography fit.
    v0.1  Initial release.
"""

VERSION = "1.0"

import cv2
import numpy as np
import sys, os, argparse, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ── Camera calibration ───────────────────────────────────────────────────────
# Inner border real-world size: 672/850 * 17 inches (2x2 tiled poster)
MARKER_REAL_M    = (672 / 850) * 17 * 0.0254   # 0.3414 m
# Focal length derived from known-distance calibration frame
# (58.930 px apparent size at 14.557 m)
FOCAL_LENGTH_PX  = 58.930 * 14.557 / MARKER_REAL_M   # 2512.89 px
# Set to None to suppress distance output if not yet calibrated
# FOCAL_LENGTH_PX = None

# Maximum valid detection distance in meters.
# At 25m the marker is ~34px apparent size — beyond this false positives
# from small scene features increase sharply.
MAX_DISTANCE_M  = 25.0
MIN_APPARENT_PX = MARKER_REAL_M * FOCAL_LENGTH_PX / MAX_DISTANCE_M  # ~34 px
SVG_TL  = np.array([89., 189.])
SVG_W   = SVG_H = 672.0

def _n(pt):  # normalise to [0,1]
    return [(pt[0]-SVG_TL[0])/SVG_W, (pt[1]-SVG_TL[1])/SVG_H]

CANON_CORNERS  = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32)
CANON_DOTS     = np.array([_n([x,y]) for y in [390,590]
                                      for x in [225,425,625]], dtype=np.float32)
CANON_ID_CELLS = [
    {'name':'left',
     'corners': np.array([_n([133,730]),_n([373,730]),
                           _n([373,820]),_n([133,820])], dtype=np.float32)},
    {'name':'right',
     'corners': np.array([_n([477,730]),_n([717,730]),
                           _n([717,820]),_n([477,820])], dtype=np.float32)},
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def order_corners(pts):
    pts = np.array(pts, dtype=np.float32)
    pts = pts[np.argsort(pts[:,1])]
    top = pts[:2][np.argsort(pts[:2,0])]
    bot = pts[2:][np.argsort(pts[2:,0])]
    return np.array([top[0],top[1],bot[1],bot[0]], dtype=np.float32)

def project(H, pts):
    ph = np.hstack([pts, np.ones((len(pts),1))]).T
    pr = H @ ph;  pr /= pr[2]
    return pr[:2].T

def refine_centroid(gray, cx, cy, r=6):
    x0,y0   = int(cx)-r, int(cy)-r
    x0c,y0c = max(0,x0),  max(0,y0)
    x1c,y1c = min(gray.shape[1],x0+2*r+1), min(gray.shape[0],y0+2*r+1)
    patch    = gray[y0c:y1c, x0c:x1c].astype(np.float64)
    med = np.median(patch)
    w   = np.where(patch < med, med-patch, 0.0)
    tot = w.sum()
    if tot < 1: return cx, cy
    rows,cols = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    return (w*cols).sum()/tot + x0c, (w*rows).sum()/tot + y0c

def subpix(gray, box):
    h,w = gray.shape
    cf  = np.clip(box, 2, min(h,w)-3).astype(np.float32).reshape(-1,1,2)
    try:
        r = cv2.cornerSubPix(gray, cf, (3,3), (-1,-1),
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
        return r.reshape(-1,2)
    except cv2.error:
        return box

# ── Stage 1: coarse detection at 1/4 scale ───────────────────────────────────

def coarse_candidates(gray, coarse_scale=0.25,
                      tophat_k=5, thresh=10,
                      min_area=10, max_area=800,
                      max_aspect=2.0):
    """
    Return list of (cx, cy) in FULL image coords for candidate marker centers.
    Uses white tophat to find compact bright rectangles.
    """
    small  = cv2.resize(gray, None, fx=coarse_scale, fy=coarse_scale,
                        interpolation=cv2.INTER_AREA)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tophat_k, tophat_k))
    tophat = cv2.morphologyEx(small, cv2.MORPH_TOPHAT, kernel)
    _, th  = cv2.threshold(tophat, thresh, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hits = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area < area < max_area): continue
        rect       = cv2.minAreaRect(c)
        rw, rh     = rect[1]
        if rw < 1 or rh < 1: continue
        if max(rw,rh)/min(rw,rh) > max_aspect: continue
        cx, cy = rect[0]
        hits.append((cx / coarse_scale, cy / coarse_scale))

    # Sort by proximity to horizontal center of image (vehicles pass through center)
    ih, iw = gray.shape
    img_cx = iw / 2
    hits.sort(key=lambda p: abs(p[0] - img_cx))
    return hits

# ── Stage 2: fine detection in a crop around a candidate ─────────────────────

def fine_detect(gray_full, cx, cy, search_r_factor=1.5):
    """
    Run the full nested-contour + homography pipeline in a crop
    around (cx,cy). search_r_factor scales the crop size relative
    to the expected marker size (estimated from coarse hit).

    Returns result dict or None.
    """
    # Crop with generous padding
    pad  = 80
    x0   = max(0, int(cx) - pad)
    y0   = max(0, int(cy) - pad)
    x1   = min(gray_full.shape[1], int(cx) + pad)
    y1   = min(gray_full.shape[0], int(cy) + pad)
    crop = gray_full[y0:y1, x0:x1]
    off  = np.array([x0, y0], dtype=np.float32)

    if crop.size == 0:
        return None

    # CLAHE with large tiles to enhance contrast between marker and background
    tile    = max(8, min(32, crop.shape[0]//4))
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile, tile))
    enhanced = clahe.apply(crop)

    # Try a range of thresholds and pick the one yielding the best nested
    # square contour. Otsu alone often picks the dark/mid split; higher values
    # better isolate the white posterboard field.
    otsu_val = int(cv2.threshold(enhanced, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    # Test Otsu and several steps above it
    candidates = sorted(set([otsu_val,
                              otsu_val + 10, otsu_val + 20,
                              otsu_val + 30, otsu_val + 40]))

    best_binary   = None
    best_hier     = None
    best_contours = None
    best_score    = -1

    for t in candidates:
        if t > 200: continue
        _, bin_t = cv2.threshold(enhanced, t, 255, cv2.THRESH_BINARY)
        ctrs, hier = cv2.findContours(bin_t, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if hier is None: continue
        # Score = number of nested square contours with 5+ children
        score = 0
        for i,(c,h) in enumerate(zip(ctrs, hier[0])):
            if cv2.contourArea(c) < 100: continue
            children = [j for j,hh in enumerate(hier[0]) if hh[3]==i]
            if len(children) < 5: continue
            rect = cv2.minAreaRect(c); rw,rh = rect[1]
            if rw > 1 and rh > 1 and max(rw,rh)/min(rw,rh) < 1.5:
                score += len(children) * 10 + int(cv2.contourArea(c))
        if score > best_score:
            best_score    = score
            best_binary   = bin_t
            best_contours = ctrs
            best_hier     = hier

    if best_binary is None:
        return None
    binary    = best_binary
    contours  = best_contours
    hierarchy = best_hier

    # Find best inner-border contour: nested, square-ish, has 5+ children.
    # Also accept a contour with 1 child where that child has 5+ children
    # (handles one extra nesting level from the posterboard border).
    # For distant/small markers, also consider top-level contours (no parent)
    # if they have enough children and are square-ish.
    def count_effective_children(idx):
        """Children count, stepping through single-child wrappers."""
        children = [j for j,hh in enumerate(hierarchy[0]) if hh[3]==idx]
        if len(children) == 1:
            grandchildren = [j for j,hh in enumerate(hierarchy[0])
                             if hh[3]==children[0]]
            if len(grandchildren) >= 5:
                return grandchildren, contours[children[0]]
        if len(children) >= 5:
            return children, contours[idx]
        return None, None

    best, best_score = None, -1
    for i,(c,h) in enumerate(zip(contours, hierarchy[0])):
        area = cv2.contourArea(c)
        if area < 100: continue
        children, inner_c = count_effective_children(i)
        if children is None: continue
        rect     = cv2.minAreaRect(inner_c)
        rw,rh    = rect[1]
        if rw < 1 or rh < 1: continue
        aspect   = max(rw,rh)/min(rw,rh)
        if aspect > 1.5: continue
        # Score favours: more children, squarer shape, size in plausible range.
        # Normalise area so large spurious contours don't dominate.
        if rw < 10 or rh < 10: continue   # too small to be real
        if rw > 140 or rh > 140: continue  # too large (marker max ~120px at 8m)
        squareness = 1.0 / aspect          # 1.0 = perfect square
        parent_bonus = 0 if h[3] < 0 else 50
        score    = len(children)*100 + int(squareness*50) + parent_bonus
        if score > best_score:
            best_score = score
            best       = (i, inner_c, children)

    if best is None:
        return None
    _, inner_c, children = best

    # Subpixel refine border corners
    box          = cv2.boxPoints(cv2.minAreaRect(inner_c))
    rect_corners = order_corners(subpix(crop, box))   # crop coords

    # Homography: canonical design -> crop image
    H, _ = cv2.findHomography(CANON_CORNERS, rect_corners)
    if H is None:
        return None

    # Predict and refine dot centers
    pred_dots    = project(H, CANON_DOTS)
    refined_dots = [refine_centroid(crop, px, py) for px,py in pred_dots]

    # Predict ID cell locations and sample intensity to determine filled/empty.
    # Use an adaptive threshold: midpoint between expected black (~30) and the
    # white field mean. This handles varying exposure better than a fixed value.
    wf_sample_mask = np.zeros(crop.shape, dtype=np.uint8)
    cv2.fillConvexPoly(wf_sample_mask, rect_corners.astype(np.int32), 255)
    for dx,dy in refined_dots:
        cv2.circle(wf_sample_mask, (int(round(dx)), int(round(dy))), 8, 0, -1)
    wf_vals_for_id = crop[wf_sample_mask > 0].astype(np.float32)
    white_level = float(np.median(wf_vals_for_id)) if len(wf_vals_for_id) > 0 else 0.0
    if white_level > 120:
        id_fill_thresh = white_level * 0.70
    else:
        id_fill_thresh = 110.0

    id_results = []
    for cell in CANON_ID_CELLS:
        pred_corners = project(H, cell['corners'])
        mask         = np.zeros(crop.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, pred_corners.astype(np.int32), 255)
        mean_val     = cv2.mean(crop, mask=mask)[0]
        filled       = mean_val < id_fill_thresh
        id_results.append({
            'name':           cell['name'],
            'corners_crop':   pred_corners,
            'mean_intensity': mean_val,
            'filled':         filled,
        })

    # Also find detected ID cell contours among children (aspect > 1.5)
    id_detected = []
    for ci in children:
        c2   = contours[ci]
        area2 = cv2.contourArea(c2)
        if area2 < 10: continue
        x,y,w,h2 = cv2.boundingRect(c2)
        if w / max(h2,1) > 1.5:
            box2 = cv2.boxPoints(cv2.minAreaRect(c2))
            id_detected.append(order_corners(subpix(crop, box2)))

    # Decode marker ID
    lf = id_results[0]['filled']
    rf = id_results[1]['filled']
    marker_id = {(True,False):1, (False,True):2,
                 (True,True):3,  (False,False):None}[(lf,rf)]

    # Re-fit H with all 10 points for accuracy
    all_canon = np.vstack([CANON_CORNERS, CANON_DOTS])
    all_image = np.vstack([rect_corners,
                           np.array(refined_dots, dtype=np.float32)])
    H2, inliers = cv2.findHomography(all_canon, all_image, cv2.RANSAC, 1.0)
    n_inliers   = int(inliers.sum()) if inliers is not None else 0

    def to_full(p): return np.array(p, dtype=np.float32) + off

    # ── Apparent size, aspect ratio and distance ─────────────────────────────
    corners_full = to_full(rect_corners)
    sides = [np.linalg.norm(corners_full[1]-corners_full[0]),
             np.linalg.norm(corners_full[2]-corners_full[1]),
             np.linalg.norm(corners_full[3]-corners_full[2]),
             np.linalg.norm(corners_full[0]-corners_full[3])]
    apparent_px = (sides[0]*sides[1]*sides[2]*sides[3])**0.25
    distance_m  = (MARKER_REAL_M * FOCAL_LENGTH_PX / apparent_px
                   if FOCAL_LENGTH_PX else None)
    # Width-to-height ratio of detected rectangle.
    # A valid marker held roughly upright should always be wider than ~0.65×height.
    # (Horizontal foreshortening from perspective is possible; vertical is not.)
    w_mean = (sides[0] + sides[2]) / 2   # average of top and bottom
    h_mean = (sides[1] + sides[3]) / 2   # average of left and right
    wh_ratio = w_mean / h_mean if h_mean > 0 else 1.0

    # ── White field uniformity check ─────────────────────────────────────
    # Build mask of inner rectangle, excluding dots and ID cells
    wf_mask = np.zeros(crop.shape, dtype=np.uint8)
    cv2.fillConvexPoly(wf_mask, rect_corners.astype(np.int32), 255)
    for dx,dy in refined_dots:
        cv2.circle(wf_mask, (int(round(dx)), int(round(dy))), 8, 0, -1)
    for cell in id_results:
        cv2.fillConvexPoly(wf_mask,
                           cell['corners_crop'].astype(np.int32), 0)
    wf_vals = crop[wf_mask > 0].astype(np.float32)
    if len(wf_vals) == 0:
        wf_mean, wf_smooth = 0.0, 0.0
    else:
        lap     = cv2.Laplacian(crop, cv2.CV_32F)
        lap_vals = np.abs(lap[wf_mask > 0])
        wf_mean      = float(wf_vals.mean())
        wf_smooth    = float((lap_vals < 15).sum() / len(lap_vals))

    return {
        'marker_id':       marker_id,
        'rect_corners':    to_full(rect_corners),
        'refined_dots':    [to_full([d])[0] for d in refined_dots],
        'id_cells':        [{**r, 'corners': to_full(r['corners_crop'])}
                            for r in id_results],
        'id_detected':     [to_full(d) for d in id_detected],
        'H':               H2,
        'n_inliers':       n_inliers,
        'apparent_px':     apparent_px,
        'distance_m':      distance_m,
        'wh_ratio':        wh_ratio,
        'wf_mean':         wf_mean,
        'wf_smooth':       wf_smooth,
        # keep crop-space data for debug drawing
        '_crop':           crop,
        '_off':            off,
        '_rect_crop':      rect_corners,
        '_dots_crop':      refined_dots,
        '_id_crop':        id_results,
        '_box_inner':      box,
    }

# ── Annotate full image ───────────────────────────────────────────────────────

def annotate(img_bgr, results):
    vis = img_bgr.copy()
    for r in results:
        mid = r['marker_id']
        # Border: yellow
        box = r['rect_corners'].astype(np.int32)
        cv2.polylines(vis, [box], True, (0,220,220), 2)
        # Dots: red
        for dx,dy in r['refined_dots']:
            cv2.drawMarker(vis,(int(round(dx)),int(round(dy))),
                           (0,0,255), cv2.MARKER_CROSS, 14, 2)
        # ID cells
        for cell in r['id_cells']:
            color = (255,0,255) if cell['filled'] else (0,140,255)
            cv2.polylines(vis,[cell['corners'].astype(np.int32)],True,color,2)
        # Label
        cx = int(r['rect_corners'][:,0].mean())
        cy = int(r['rect_corners'][:,1].mean())
        label = f"M{mid}" if mid else "M?"
        cv2.putText(vis, label, (cx-12, cy-35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,220,220), 2)
    return vis

# ── Main ─────────────────────────────────────────────────────────────────────

def process(image_path, debug=False, verbose=False, road_y=None, road_x=None):
    img  = cv2.imread(image_path)
    if img is None:
        sys.exit(f"Cannot read {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    x0_road = 300   # left edge of road zone (vehicles won't appear left of this)
    x1_road = 1550  # right edge of road zone (vehicles won't appear right of this)

    y0_road = 380 # h * 33 // 100   # ~360px for 1080p  (top of road zone)
    y1_road = 518 #h * 48 // 100   # ~518px for 1080p  (bottom of road zone)
    if road_y is not None:
        y0_road, y1_road = road_y
    if road_x is not None:
        x0_road, x1_road = road_x
    gray_road = gray[y0_road:y1_road, x0_road:x1_road]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Image: {os.path.basename(image_path)}  ({w}x{h})")

    hits_road = coarse_candidates(gray_road)
    hits = [(cx + x0_road, cy + y0_road) for cx, cy in hits_road]

    if verbose:
        print(f"Coarse candidates (road zone y={y0_road}..{y1_road} x={x0_road}..{x1_road}): {len(hits)}")
        for cx, cy in hits:
            print(f"  ({cx:.0f}, {cy:.0f})")

    results      = []
    seen_centers = []

    def try_candidate(cx, cy):
        # Early skip only if crop center is very close to an already-accepted
        # result's detected center (avoids re-running fine_detect redundantly)
        if any(abs(cx - r['_rcx']) < 40 and abs(cy - r['_rcy']) < 40
               for r in results):
            return
        r = fine_detect(gray, cx, cy)
        if r is None:
            if verbose: print(f"  ({cx:.0f},{cy:.0f}): fine detection failed")
            return
        if r['marker_id'] is None:
            if verbose:
                print(f"  ({cx:.0f},{cy:.0f}): rejected (id=None)")
            return
        # Apparent size sanity: marker can't be closer than ~8m (>100px) or
        # further than MAX_DISTANCE_M (<MIN_APPARENT_PX)
        if r['apparent_px'] < MIN_APPARENT_PX or r['apparent_px'] > 135:
            if verbose:
                print(f"  ({cx:.0f},{cy:.0f}): rejected "
                      f"(apparent={r['apparent_px']:.1f}px out of range "
                      f"{MIN_APPARENT_PX:.1f}..135)")
            return
        # Width/height ratio: marker should never appear more than ~1.5x taller
        # than wide. Minimum w/h = 0.65 rejects spurious tall-narrow detections.
        if r['wh_ratio'] < 0.80:
            if verbose:
                print(f"  ({cx:.0f},{cy:.0f}): rejected "
                      f"(wh_ratio={r['wh_ratio']:.3f} < 0.80, too tall/narrow)")
            return
        # Inlier requirement: relax to 6 only for small/distant markers
        min_inliers = 6 if r['apparent_px'] < 50 else 8
        if r['n_inliers'] < min_inliers:
            if verbose:
                print(f"  ({cx:.0f},{cy:.0f}): rejected "
                      f"(inliers={r['n_inliers']}<{min_inliers})")
            return
        if r['wf_mean'] < 80 or r['wf_smooth'] < 0.35:
            if verbose:
                print(f"  ({cx:.0f},{cy:.0f}): rejected "
                      f"(wf_mean={r['wf_mean']:.1f}, wf_smooth={r['wf_smooth']:.3f})")
            return
        rcx = r['rect_corners'][:,0].mean()
        rcy = r['rect_corners'][:,1].mean()
        # Best center estimate: project canonical center (0.5, 0.5) through
        # the 10-point refined homography (uses all 4 corners + 6 dots).
        # H is in crop-local coords so add the crop offset back.
        if r['H'] is not None:
            canon_center = np.array([[0.5, 0.5]], dtype=np.float32)
            off = r['_off']
            proj = project(r['H'], canon_center)[0]
            rcx = float(proj[0]) + float(off[0])
            rcy = float(proj[1]) + float(off[1])
        # Reject if the detected center is outside the road zone
        if rcy < y0_road or rcy > y1_road or rcx < x0_road or rcx > x1_road:
            if verbose:
                print(f"  ({cx:.0f},{cy:.0f}): rejected "
                      f"(center ({rcx:.0f},{rcy:.0f}) outside road zone "
                      f"x={x0_road}..{x1_road} y={y0_road}..{y1_road})")
            return

        # Check for near-duplicate: same marker already detected nearby.
        # Use detected center distance (not crop center) with a tight radius.
        # If duplicate found, keep the better one (more inliers, then wh_ratio).
        def quality(res):
            return (res['n_inliers'], res['wh_ratio'])

        dup_idx = None
        for i, existing in enumerate(results):
            ex_cx = existing['_rcx']
            ex_cy = existing['_rcy']
            if abs(rcx - ex_cx) < 25 and abs(rcy - ex_cy) < 25:
                dup_idx = i
                break

        if dup_idx is not None:
            existing = results[dup_idx]
            if quality(r) > quality(existing):
                if verbose:
                    print(f"  ({cx:.0f},{cy:.0f}): replacing existing result "
                          f"(inliers {existing['n_inliers']}->{r['n_inliers']}, "
                          f"wh {existing['wh_ratio']:.3f}->{r['wh_ratio']:.3f})")
                r['_rcx'] = rcx;  r['_rcy'] = rcy
                results[dup_idx] = r
            else:
                if verbose:
                    print(f"  ({cx:.0f},{cy:.0f}): rejected as duplicate "
                          f"(worse than existing: inliers={r['n_inliers']} "
                          f"wh={r['wh_ratio']:.3f})")
            return

        r['_rcx'] = rcx;  r['_rcy'] = rcy
        seen_centers.append((rcx, rcy))
        results.append(r)

    for cx, cy in hits:
        try_candidate(cx, cy)

    if not results:
        if verbose:
            print("  No marker found from tophat — trying dense grid fallback...")
        step = 60
        for gy in range(y0_road + 40, y1_road - 40, step):
            for gx in range(x0_road + 40, x1_road - 40, step):
                try_candidate(gx, gy)

    # Print all accepted results (after duplicate resolution)
    for r in results:
        mid    = r['marker_id']
        rcx    = r['_rcx']
        rcy    = r['_rcy']
        dist_str = (f"  dist={r['distance_m']:.3f}m"
                    if r['distance_m'] is not None else "")
        print(f"{os.path.basename(image_path)}: "
              f"MARKER {mid}  center=({rcx:.1f},{rcy:.1f})  "
              f"TL=({r['rect_corners'][0,0]:.1f},{r['rect_corners'][0,1]:.1f})  "
              f"TR=({r['rect_corners'][1,0]:.1f},{r['rect_corners'][1,1]:.1f})  "
              f"BR=({r['rect_corners'][2,0]:.1f},{r['rect_corners'][2,1]:.1f})  "
              f"BL=({r['rect_corners'][3,0]:.1f},{r['rect_corners'][3,1]:.1f})"
              f"{dist_str}")
        if verbose:
            print(f"    inliers={r['n_inliers']}/10  "
                  f"wf_mean={r['wf_mean']:.0f}  wf_smooth={r['wf_smooth']:.3f}")
            print(f"    Dots (full image):")
            for i,(px,py) in enumerate(r['refined_dots']):
                print(f"      {i}: ({px:.2f},{py:.2f})")
            print(f"    ID cells:")
            for cell in r['id_cells']:
                print(f"      {cell['name']:5s}: "
                      f"{'FILLED' if cell['filled'] else 'empty '}  "
                      f"mean={cell['mean_intensity']:.1f}")

    if not results:
        print(f"{os.path.basename(image_path)}: no marker detected")

    if results:
        base = os.path.splitext(image_path)[0]
        out  = base + '_detected.jpg'
        cv2.imwrite(out, annotate(img, results))
        if verbose:
            print(f"Annotated image saved: {out}")

    return results

def main():
    ap = argparse.ArgumentParser(
        description="Detect printed tracking markers in images or a directory of images")
    ap.add_argument("inputs", nargs="+",
                    help="Image files (.jpg/.png) or directories containing them")
    ap.add_argument("--version", action="version", version=f"detect_marker.py v{VERSION}")
    ap.add_argument("--debug",   action="store_true")
    ap.add_argument("--verbose", action="store_true",
                    help="Print all candidates and rejection reasons")
    ap.add_argument("--max-dist", type=float, default=None,
                    help="Maximum detection distance in meters (default: 25.0). "
                         "Rejects markers that appear too small.")
    ap.add_argument("--road-y", type=int, nargs=2, metavar=("Y0", "Y1"),
                    default=None,
                    help="Road zone pixel rows, e.g. --road-y 380 518")
    ap.add_argument("--road-x", type=int, nargs=2, metavar=("X0", "X1"),
                    default=None,
                    help="Road zone pixel columns, e.g. --road-x 300 1550")
    args = ap.parse_args()

    # Expand any directories into sorted lists of image files
    paths = []
    for inp in args.inputs:
        if os.path.isdir(inp):
            exts = ('.jpg', '.jpeg', '.png')
            found = sorted(
                p for p in (os.path.join(inp, f) for f in os.listdir(inp))
                if os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts
            )
            if not found:
                print(f"Warning: no .jpg/.png files found in {inp}")
            paths.extend(found)
        elif os.path.isfile(inp):
            paths.append(inp)
        else:
            print(f"Warning: {inp} is not a file or directory, skipping")

    if not paths:
        sys.exit("No image files to process.")

    if args.max_dist is not None:
        global MAX_DISTANCE_M
        global MIN_APPARENT_PX
        MAX_DISTANCE_M  = args.max_dist
        MIN_APPARENT_PX = MARKER_REAL_M * FOCAL_LENGTH_PX / MAX_DISTANCE_M

    for p in paths:
        process(p, args.debug, args.verbose,
                road_y=args.road_y,
                road_x=args.road_x)

if __name__ == "__main__":
    main()
