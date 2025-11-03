import os, glob, math
import cv2
import numpy as np
from tifffile import imwrite
from ome_types.model import OME, Image, Pixels, TiffData

# ---------------- CONFIG ----------------
IMAGE_FOLDER = r"C:\Users\Priyanshu Srivastava\Downloads\Images"
COLS, ROWS   = 7, 10                   # 7 columns (X), 10 rows (Y)
OVERLAP_FRAC = 0.08                    # 5–10% expected
LABEL_TILES  = True                    # draw index on tiles (debug)

# Alignment fallback options
USE_ECC_FALLBACK = True
ECC_ITERS = 100
ECC_EPS   = 1e-5

# ---------------- LOAD ----------------
paths = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")),
               key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
if not paths:
    raise RuntimeError("No JPGs found in IMAGE_FOLDER.")
images_bgr = [cv2.imread(p, cv2.IMREAD_COLOR) for p in paths]
H, W = images_bgr[0].shape[:2]
N = len(images_bgr)
print(f"Loaded {N} images; tile = {W}x{H}")
ROWS = math.ceil(N / COLS)

# ---------------- GRID SETUP ----------------
x_step = int(W * (1 - OVERLAP_FRAC))
y_step = int(H * (1 - OVERLAP_FRAC))
canvas_w = x_step * (COLS - 1) + W
canvas_h = y_step * (ROWS - 1) + H
print(f"Initial canvas: {canvas_w} x {canvas_h}")

def serpentine_xy(idx):
    row_from_bottom = idx // COLS
    in_row_index    = idx % COLS
    col = in_row_index if row_from_bottom % 2 == 0 else COLS - 1 - in_row_index
    return col, row_from_bottom

def serpentine_index(col, row):
    return row * COLS + (col if row % 2 == 0 else (COLS - 1 - col))

pos_init = []
for i in range(N):
    c, r = serpentine_xy(i)
    x = c * x_step
    y = canvas_h - H - r * y_step
    pos_init.append([x, y])

# ---------------- LABEL UTILITY ----------------
def label_tile(img, idx, fname):
    if not LABEL_TILES: return img
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    fg, bg = (255,255,255), (0,0,0)
    big = str(idx)
    (tw, th), _ = cv2.getTextSize(big, font, 2.5, 5)
    cx, cy = out.shape[1]//2, out.shape[0]//2
    x1, y1 = cx - tw//2 - 8, cy - th//2 - 8
    x2, y2 = x1 + tw + 16, y1 + th + 16
    cv2.rectangle(out, (x1,y1), (x2,y2), bg, -1)
    cv2.putText(out, big, (x1+8, y2-8), font, 2.5, fg, 5, cv2.LINE_AA)
    small = f"#{idx}  {os.path.basename(fname)}"
    (tw2, th2), _ = cv2.getTextSize(small, font, 1.2, 2)
    cv2.rectangle(out, (20,20), (20+tw2+12, 20+th2+12), bg, -1)
    cv2.putText(out, small, (26, 20+th2+8), font, 1.2, fg, 2, cv2.LINE_AA)
    return out

# ---------------- ALIGNMENT HELPERS ----------------
def phase_shift(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
    a, b = a[:h, :w], b[:h, :w]
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a*win, b*win)
    return dx, dy, resp

def ecc_affine(a, b):
    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
    a, b = a[:h, :w], b[:h, :w]
    a = cv2.normalize(a, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    b = cv2.normalize(b, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_AFFINE,
                                        (ECC_ITERS, ECC_EPS), None, 5)
        return warp, cc
    except cv2.error:
        return None, -1

# ---------------- FEATURE-BASED HOMOGRAPHY ----------------
try:
    _ = cv2.SIFT_create
    FEAT = 'SIFT'
except AttributeError:
    FEAT = 'ORB'
print("Using feature detector:", FEAT)

def _create_detector():
    return cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.01, edgeThreshold=5) \
           if FEAT == 'SIFT' else \
           cv2.ORB_create(nfeatures=8000, scaleFactor=1.2, edgeThreshold=8, fastThreshold=7)

def homography_shift(a_gray, b_gray, max_ratio=0.8, ransac_thresh=5.0, min_inliers=8):
    det = _create_detector()
    kpa, desa = det.detectAndCompute(a_gray, None)
    kpb, desb = det.detectAndCompute(b_gray, None)
    if desa is None or desb is None or len(kpa) < 4 or len(kpb) < 4:
        return 0.0, 0.0, 0.0, None
    norm = cv2.NORM_L2 if FEAT == 'SIFT' else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm, crossCheck=False)
    matches = bf.knnMatch(desb, desa, k=2)  # query=b, train=a
    good = [m for m, n in matches if m.distance < max_ratio * n.distance] if matches else []
    if len(good) < 4:
        return 0.0, 0.0, 0.0, None
    ptsB = np.float32([kpb[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ptsA = np.float32([kpa[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, ransac_thresh)
    if H is None or mask is None:
        return 0.0, 0.0, 0.0, None
    inliers = int(mask.ravel().sum())
    if inliers < min_inliers:
        return 0.0, 0.0, 0.0, None
    tx, ty = float(H[0, 2]), float(H[1, 2])
    conf = (inliers / max(1, len(good))) * (1.0 + np.log1p(inliers) / 5.0)
    return tx, ty, conf, H

# ---------------- TILE ALIGNMENT ----------------
gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images_bgr]
LEFT_W   = max(12, int(W * max(OVERLAP_FRAC, 0.1)))   # widen a bit for robustness
BOTTOM_H = max(12, int(H * max(OVERLAP_FRAC, 0.1)))
pos_refined = np.array(pos_init, dtype=np.float32)

for idx in range(N):
    x, y = pos_refined[idx]
    c, r = serpentine_xy(idx)
    dxs, dys, wts = [], [], []

    # --- LEFT neighbor (expected overlap) ---
    left_col = c - 1 if r % 2 == 0 else c + 1
    if 0 <= left_col < COLS:
        left_idx = serpentine_index(left_col, r)
        if 0 <= left_idx < N:
            A = gray[left_idx][:, W-LEFT_W:W]   # ref: left tile's right strip
            B = gray[idx][:, 0:LEFT_W]          # mov: current tile's left strip
            tx, ty, conf, _ = homography_shift(A, B)
            if conf > 0.05:
                dx_canvas = x_step - tx
                dy_canvas = ty
            else:
                dx, dy, resp = phase_shift(A, B)
                dx_canvas = x_step - dx
                dy_canvas = dy
                if resp < 0.05 and USE_ECC_FALLBACK:
                    warp, cc = ecc_affine(A, B)
                    if warp is not None and cc > 0:
                        dx_canvas = x_step - warp[0,2]
                        dy_canvas = warp[1,2]
                        resp = max(resp, cc)
                conf = max(0.05, float(conf), float(resp))
            dxs.append(dx_canvas); dys.append(dy_canvas); wts.append(conf)

    # --- BOTTOM neighbor (expected overlap) ---
    if r - 1 >= 0:
        bottom_idx = serpentine_index(c, r - 1)
        if 0 <= bottom_idx < N:
            A = gray[bottom_idx][H-BOTTOM_H:H, :]   # ref: bottom tile's top strip
            B = gray[idx][0:BOTTOM_H, :]            # mov: current tile's bottom strip
            tx, ty, conf, _ = homography_shift(A, B)
            if conf > 0.05:
                dx_canvas = tx
                dy_canvas = y_step - ty
            else:
                dx, dy, resp = phase_shift(A, B)
                dx_canvas = dx
                dy_canvas = y_step - dy
                if resp < 0.05 and USE_ECC_FALLBACK:
                    warp, cc = ecc_affine(A, B)
                    if warp is not None and cc > 0:
                        dx_canvas = warp[0,2]
                        dy_canvas = y_step - warp[1,2]
                        resp = max(resp, cc)
                conf = max(0.05, float(conf), float(resp))
            dxs.append(dx_canvas); dys.append(dy_canvas); wts.append(conf)

    # weighted update
    if wts:
        w = np.array(wts, dtype=np.float32)
        dx_mean = float(np.dot(np.array(dxs), w) / w.sum())
        dy_mean = float(np.dot(np.array(dys), w) / w.sum())
        pos_refined[idx, 0] = pos_init[idx][0] + (dx_mean - (x_step if (0 <= left_col < COLS) else 0))
        pos_refined[idx, 1] = pos_init[idx][1] + (dy_mean - (y_step if (r - 1 >= 0) else 0))
    pos_refined[idx, 0] = float(np.clip(pos_refined[idx, 0], 0, canvas_w - W))
    pos_refined[idx, 1] = float(np.clip(pos_refined[idx, 1], 0, canvas_h - H))

# ---------------- MULTI-BAND BLENDING ----------------

 # ---------------- LIGHTWEIGHT FEATHER BLENDING (memory-safe) ----------------
LABEL_TILES = False  # disable labels for final

# Feather is much lighter than multiband
blender = cv2.detail_FeatherBlender()
blender.setSharpness(0.02)  # wider, smoother seam

corners, sizes, masks, tiles = [], [], [], []
for idx, img in enumerate(images_bgr):
    x, y = pos_refined[idx]
    x, y = int(round(x)), int(round(y))
    corners.append((x, y))
    sizes.append((img.shape[1], img.shape[0]))
    tiles.append(img)  # no labels for final
    masks.append(255 * np.ones((img.shape[0], img.shape[1]), np.uint8))

# No exposure compensation and no seam finding (they triggered OOM)
minx = min(c[0] for c in corners); miny = min(c[1] for c in corners)
maxx = max(c[0]+s[0] for c,s in zip(corners,sizes))
maxy = max(c[1]+s[1] for c,s in zip(corners,sizes))
blender.prepare((minx, miny, maxx, maxy))

# FeatherBlender REQUIRES CV_16SC3 tiles
for i, (img, mask, (x, y)) in enumerate(zip(tiles, masks, corners)):
    try:
        blender.feed(img.astype(np.int16), mask.astype(np.uint8), (x, y))
    except cv2.error as e:
        print(f"Skipped tile {i} at ({x},{y}): {e}")

result, result_mask = blender.blend(None, None)

# Convert blended result back to uint8 before any further processing
result = np.clip(result, 0, 255).astype(np.uint8)

# --- Fill tiny holes and inpaint small gaps (cheap + safe) ---
gray_r = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
holes = (gray_r == 0).astype(np.uint8) * 255
if np.count_nonzero(holes) > 0:
    mean_val = cv2.mean(result, mask=(gray_r > 0).astype(np.uint8))
    result[gray_r == 0] = np.uint8(mean_val[:3])
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    small_gaps = cv2.morphologyEx(holes, cv2.MORPH_OPEN, k, iterations=1)
    if np.count_nonzero(small_gaps) > 0:
        result = cv2.inpaint(result, small_gaps, 3, cv2.INPAINT_TELEA)



# ---------------- REMOVE BLACK BORDERS ----------------
gray_mask = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
coords = cv2.findNonZero(binary)
if coords is not None:
    x, y, w, h = cv2.boundingRect(coords)
    pad = 5
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(result.shape[1], x + w + pad)
    y2 = min(result.shape[0], y + h + pad)
    result = result[y1:y2, x1:x2].copy()
    print(f" Cropped borders → {result.shape[1]}×{result.shape[0]}")
else:
    print(" No non-black pixels found — skipping crop.")

# ---------------- OME-TIFF + JPG ----------------
ome = OME(images=[Image(
    id="Image:0", name="Stitched-Serpentine-Registered",
    pixels=Pixels(
        dimension_order="XYZCT",
        size_x=int(result.shape[1]),
        size_y=int(result.shape[0]),
        size_z=1, size_c=3, size_t=1,
        type="uint8",
        tiff_data=[TiffData(first_z=0, first_t=0, ifd=0)]
))])
ome_xml = ome.to_xml()

ome_path = os.path.join(IMAGE_FOLDER, "stitched_registered.ome.tif")
imwrite(ome_path, result, description=ome_xml)
jpg_path = os.path.join(IMAGE_FOLDER, "stitched_registered.jpg")
cv2.imwrite(jpg_path, cv2.cvtColor(result, cv2.COLOR_BGR2RGB), [int(cv2.IMWRITE_JPEG_QUALITY), 95])

print(" Saved:")
print("  •", ome_path)
print("  •", jpg_path)
