"""
Lists the images in your dataset with the highest sigma, so you can
test the Moderate / Severe bands in the UI.

Run from the backend folder:   python find_severe.py
(needs opencv-python-headless, numpy, scipy - already installed for the server)
"""
import os, sys
import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad

BASE = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\param\OneDrive\Desktop\ScoliosisDataSet\dataset"
MASK_DIR = os.path.join(BASE, "masks")
CUTS = (0.0019, 0.0186, 0.0393)


def centroids(path, min_area=50):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, b = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    n, _, stats, c = cv2.connectedComponentsWithStats(b, connectivity=8)
    pts = [c[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return np.array(sorted(pts, key=lambda p: p[1]))


def sigma(pts, s_factor=100):
    y = pts[:, 1] - pts[0, 1]
    x = pts[:, 0] - pts[0, 0]
    keep = np.concatenate(([True], np.diff(y) > 1e-6))
    y, x = y[keep], x[keep]
    spl = UnivariateSpline(y, x, k=3, s=s_factor * len(y))
    d = spl.derivative()
    s, _ = quad(lambda t: np.sqrt(1 + d(t) ** 2), y[0], y[-1], limit=200)
    dist = np.hypot(spl(y[-1]) - spl(y[0]), y[-1] - y[0])
    return (1 - dist / s) ** 2


def band(v):
    return "Minimal" if v < CUTS[0] else "Mild" if v < CUTS[1] else "Moderate" if v < CUTS[2] else "Severe"


rows = []
for f in sorted(os.listdir(MASK_DIR)):
    pts = centroids(os.path.join(MASK_DIR, f))
    if 12 <= len(pts) <= 18:          # skip masks with obvious junk
        rows.append((sigma(pts), len(pts), f))

rows.sort(reverse=True)
print(f"scored {len(rows)} masks.  Top 12 by sigma (raw image has the same name, .jpg instead of .png):\n")
for s, n, f in rows[:12]:
    print(f"{s:.4f}  {band(s):9s} {n:2d} vertebrae  {f}")
