"""
@file: denoiser.py
@author: Mert-chan
@date: 2026-11-24
@brief: Density-based clustering denoiser for SEALION FMCW ionograms that keeps spread-F echoes. Steps: crop, column-median background filter, DBSCAN on tilted pixel coordinates with per-frame eps and m, cluster filters (size, neighbor ratio NR, stripe test), regrowth of small pieces next to the kept echo. Input: raw 738 x 500 ionogram PNG (load_crop) or its 315 x 396 crop (denoise). Output: echo mask, True = echo.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, convolve, median_filter
from sklearn.cluster import DBSCAN


# Crop of the raw PNG (738 x 500): 0-1000 km on rows 474 to 24, frequency from column 38 at 25 px/MHz
#==================================
CROP_ROWS = (114, 429)                  # 800 km to 100 km, 2.22 km/px
CROP_COLS = (38, 434)                   # 2 to 17.8 MHz, 0.04 MHz/px
KM_TOP    = 800.0
KM_BOT    = 100.0
#==================================


# Parameters (tuned by trial and error on CPN and BCL frames, tied to the pixel scale above)
#==================================
K_SIGMA     = 1.5                       # foreground: pixel darker than its column median by K_SIGMA sigma_c
MEDIAN_SIZE = 2
MEDIAN_MIN  = 1500                      # fewer foreground pixels than this: no median, it would erase a sparse dotted trace
KILL_KM     = 700.0                     # everything above is erased
BAND_KM     = (200.0, 600.0)            # n = foreground pixels in this band, drives the per-frame parameters
ROW_FACTOR  = 2.0                       # DBSCAN runs on (ROW_FACTOR * row + col, col): flat and rising steps short, vertical steps long
EPS_MIN     = 3.0                       # eps falls linearly from EPS_MAX (EPS_MAX_SP if n < SPARSE_N) to EPS_MIN at n = EPS_NREF
EPS_MAX     = 12.0
EPS_MAX_SP  = 18.0
EPS_NREF    = 5500
SPARSE_N    = 2250
DENSE_N     = 4000
M_DIV       = 450                       # m = n // M_DIV, clipped to M_RANGE
M_RANGE     = (2, 10)
MIN_PX      = 100                       # smallest cluster kept, and the empty-frame rule
SPARSE_A0   = 500                       # fewer foreground pixels than this: small clusters are kept, a sparse trace breaks into pieces under MIN_PX
KERNEL      = np.array([[0.25, 0.25, 1.0],
                        [1.0,  0.0,  1.0],
                        [1.0,  0.25, 0.25]], dtype=np.float32)   # neighbor ratio NR: full weight for horizontal and rising neighbors
NR_TAU      = (0.40, 0.49, 0.60)        # NR threshold for clusters centred in BAND_KM: sparse, mid, dense frames
NR_SPARSE   = 0.20                      # NR threshold for clusters centred in BAND_KM on frames under SPARSE_A0, a dotted trace has few neighbors
NR_OUT      = 0.65                      # NR threshold for clusters centred outside BAND_KM
LINE_E      = 0.85                      # elongation and angle (deg from the frequency axis) for the line relaxation
LINE_THETA  = 70.0
LINE_RELAX  = 0.75
STRIPE_THETA = 75.0                     # narrow vertical stripe: steeper than this and narrower than STRIPE_COLS
STRIPE_COLS  = 35
GROW_PX     = 6                         # an erased cluster within GROW_PX of the kept echo comes back (no size test) if its NR
GROW_RELAX  = 0.95                      # is at least GROW_RELAX times its threshold, repeated until nothing joins
#==================================


def load_crop(path):
    """Raw ionogram PNG to the crop, float32 in [0, 1]."""
    img = np.array(Image.open(path).convert("L"), dtype=np.float32)
    return img[CROP_ROWS[0]:CROP_ROWS[1], CROP_COLS[0]:CROP_COLS[1]] / 255.0


def row_of(km, H):
    return int(round((KM_TOP - km) / (KM_TOP - KM_BOT) * H))


def foreground(crop):
    """Pixels more than K_SIGMA sigma_c darker than their column median, 2 x 2 median if at least MEDIAN_MIN pixels pass,
    nothing above KILL_KM."""
    col_sub = (crop - np.median(crop, axis=0)).astype(np.float32)
    sigma   = np.where((col_sub > 0).sum(axis=0) > 5, col_sub.clip(min=0).std(axis=0), col_sub.std(axis=0))
    binary  = np.where(col_sub < -K_SIGMA * sigma, 0.0, 1.0).astype(np.float32)
    fg      = binary == 0 if (binary == 0).sum() < MEDIAN_MIN else median_filter(binary, size=MEDIAN_SIZE) == 0
    fg[:row_of(KILL_KM, len(fg))] = False
    return fg


def n_band(fg):
    return int(fg[row_of(BAND_KM[1], len(fg)):row_of(BAND_KM[0], len(fg))].sum())


def frame_params(n):
    """Per-frame DBSCAN eps and m: a noisy frame (large n) gets a small eps and a large m."""
    eps_max = EPS_MAX_SP if n < SPARSE_N else EPS_MAX
    return eps_max - (eps_max - EPS_MIN) * min(n / EPS_NREF, 1.0), int(np.clip(n // M_DIV, *M_RANGE))


def clusters(fg, eps, m):
    """DBSCAN on the foreground pixels. Returns the pixel coordinates (row, col) and their labels, -1 = unclustered."""
    coords = np.column_stack(np.where(fg))
    if len(coords) == 0:
        return coords, np.zeros(0, dtype=int)
    pts = np.column_stack([ROW_FACTOR * coords[:, 0] + coords[:, 1], coords[:, 1]])
    return coords, DBSCAN(eps=eps, min_samples=m).fit_predict(pts)


def cluster_filters(fg, coords, labels, n):
    """Keep flag per foreground pixel: the cluster has more than MIN_PX pixels, a mean NR at or above its threshold and is
    not a narrow vertical stripe. A frame with fewer than SPARSE_A0 foreground pixels skips the size test and uses NR_SPARSE
    in the band. Elongation e and angle theta come from the PCA of the cluster pixels. Erased clusters within GROW_PX of the
    kept echo with NR at least GROW_RELAX times their threshold are taken back."""
    H, W  = fg.shape
    keep  = np.zeros(len(labels), dtype=bool)
    valid = labels >= 0
    if not valid.any():
        return keep
    ul, inv = np.unique(labels[valid], return_inverse=True)
    sizes   = np.bincount(inv)
    vc      = coords[valid]

    nr  = (convolve(fg.astype(np.float32), KERNEL, mode="constant", cval=0.0) / KERNEL.sum())[vc[:, 0], vc[:, 1]]
    cnr = np.bincount(inv, weights=nr) / sizes

    km      = KM_TOP - (np.bincount(inv, weights=vc[:, 0]) / sizes / H) * (KM_TOP - KM_BOT)
    outside = (km < BAND_KM[0]) | (km > BAND_KM[1])
    sparse  = fg.sum() < SPARSE_A0
    tau     = NR_SPARSE if sparse else NR_TAU[0] if n < SPARSE_N else NR_TAU[1] if n < DENSE_N else NR_TAU[2]

    dr     = vc[:, 0] - (np.bincount(inv, weights=vc[:, 0].astype(np.float64)) / sizes)[inv]
    dc     = vc[:, 1] - (np.bincount(inv, weights=vc[:, 1].astype(np.float64)) / sizes)[inv]
    denom  = np.maximum(sizes - 1, 1).astype(np.float64)
    var_r  = np.bincount(inv, weights=dr * dr) / denom
    var_c  = np.bincount(inv, weights=dc * dc) / denom
    cov_rc = np.bincount(inv, weights=dr * dc) / denom
    trace  = var_r + var_c
    lam1   = trace / 2 + np.sqrt(np.maximum((trace / 2) ** 2 - (var_r * var_c - cov_rc ** 2), 0.0))
    e      = np.where(trace > 0, lam1 / np.where(trace > 0, trace, 1.0), 0.5).astype(np.float32)
    v_row  = np.abs(cov_rc)
    v_col  = np.abs(lam1 - var_r)
    theta  = np.where((v_row + v_col) > 1e-6, np.degrees(np.arctan2(v_row, v_col)), np.where(var_r > var_c, 90.0, 0.0)).astype(np.float32)

    col_min = np.full(len(ul), W, dtype=np.float64)
    col_max = np.zeros(len(ul), dtype=np.float64)
    np.minimum.at(col_min, inv, vc[:, 1].astype(np.float64))
    np.maximum.at(col_max, inv, vc[:, 1].astype(np.float64))

    thr    = np.where(outside, NR_OUT, tau) * np.where((e > LINE_E) & (theta < LINE_THETA), LINE_RELAX, 1.0)
    small  = (sizes <= MIN_PX) & ~sparse
    stripe = (theta > STRIPE_THETA) & (col_max - col_min < STRIPE_COLS)
    erase  = small | (cnr < thr) | stripe
    keep[valid] = ~erase[inv]

    # small eps on a noisy frame breaks an echo into pieces under MIN_PX: take back pieces that touch the kept echo
    cand = erase & ~stripe & (cnr >= GROW_RELAX * thr)
    while cand.any():
        kept = np.zeros((H, W), dtype=bool)
        kept[coords[keep, 0], coords[keep, 1]] = True
        near = binary_dilation(kept, structure=np.ones((3, 3), dtype=bool), iterations=GROW_PX)[vc[:, 0], vc[:, 1]]
        hit  = cand & (np.bincount(inv, weights=near, minlength=len(ul)) > 0)
        if not hit.any():
            break
        cand &= ~hit
        keep[valid] |= hit[inv]
    return keep


def denoise(crop):
    """Echo mask of one crop."""
    fg = foreground(crop)
    n  = n_band(fg)
    coords, labels = clusters(fg, *frame_params(n))
    keep = cluster_filters(fg, coords, labels, n)
    mask = np.zeros(fg.shape, dtype=bool)
    mask[coords[keep, 0], coords[keep, 1]] = True
    return mask
