"""
SpineMetric backend.
Run:  uvicorn app:app --port 8000
Needs unet_best.pt in the same folder as this file.
"""
import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "unet_best.pt")
THR, ERODE = 0.7, 5
SIGMA_CUTS = (0.0019, 0.0186, 0.0393)   # provisional bins
DEV = "cuda" if torch.cuda.is_available() else "cpu"

model = smp.Unet("resnet34", encoder_weights=None, in_channels=1, classes=1)
model.load_state_dict(torch.load(CKPT, map_location=DEV))
model.to(DEV).eval()

app = FastAPI(title="SpineMetric")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def centroids_from_binary(b, min_area=50):
    n, labels, stats, c = cv2.connectedComponentsWithStats(b, connectivity=8)
    pts = [c[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return np.array(sorted(pts, key=lambda p: p[1]))


def _spline(pts, s_factor=100):
    y = pts[:, 1] - pts[0, 1]
    x = pts[:, 0] - pts[0, 0]
    keep = np.concatenate(([True], np.diff(y) > 1e-6))
    y, x = y[keep], x[keep]
    return UnivariateSpline(y, x, k=3, s=s_factor * len(y)), y


def sigma_of(pts):
    spl, y = _spline(pts)
    dspl = spl.derivative()
    s, _ = quad(lambda t: np.sqrt(1 + dspl(t) ** 2), y[0], y[-1], limit=200)
    d = np.hypot(spl(y[-1]) - spl(y[0]), y[-1] - y[0])
    return float((1 - d / s) ** 2), float(d), float(s)


def cobb_of(pts):
    """Rough Cobb proxy: max change in curve tangent angle (degrees). Unvalidated."""
    spl, y = _spline(pts)
    span = y[-1] - y[0]
    yy = np.linspace(y[0] + 0.1 * span, y[-1] - 0.1 * span, 200)   # ignore noisy 10% at each end
    ang = np.degrees(np.arctan(spl.derivative()(yy)))
    return float(ang.max() - ang.min())


def curve_of(pts):
    spl, y = _spline(pts)
    yy = np.linspace(y[0], y[-1], 200)
    return np.column_stack([spl(yy) + pts[0, 0], yy + pts[0, 1]])


def severity(sig):
    if sig < SIGMA_CUTS[0]:
        return "Minimal"
    if sig < SIGMA_CUTS[1]:
        return "Mild"
    if sig < SIGMA_CUTS[2]:
        return "Moderate"
    return "Severe"


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "spinemetric.html"))


@app.get("/health")
def health():
    return {"status": "ok", "device": DEV}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = np.frombuffer(await file.read(), np.uint8)
    orig = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if orig is None:
        raise HTTPException(400, "Could not read image")
    H, W = orig.shape

    img = cv2.resize(orig, (512, 512)).astype(np.float32) / 255.0
    with torch.no_grad():
        x = torch.tensor(img)[None, None].to(DEV)
        p = torch.sigmoid(model(x))[0, 0].cpu().numpy()

    b = cv2.erode((p > THR).astype(np.uint8) * 255, np.ones((ERODE, 1), np.uint8))
    pts = centroids_from_binary(b, 30)
    if len(pts) < 5:
        raise HTTPException(422, f"Only {len(pts)} vertebrae detected; need at least 5")

    sig, d, s = sigma_of(pts)

    n = int(len(pts))
    coverage = float((pts[-1, 1] - pts[0, 1]) / 512)
    warnings = []
    if n < 10:
        warnings.append(f"Only {n} vertebrae detected (expected about 12-17).")
    if n > 20:
        warnings.append(f"{n} vertebrae detected, more than expected; bands may be split.")
    if coverage < 0.25:
        warnings.append("The detected spine covers only a small part of the image.")
    sx, sy = W / 512, H / 512
    scale = lambda a: [[round(float(px * sx), 1), round(float(py * sy), 1)] for px, py in a]

    return {
        "sigma": round(sig, 6),
        "severity_band": severity(sig),
        "arc_length_px": round(s, 2),
        "chord_px": round(d, 2),
        "cobb_estimate_deg": round(cobb_of(pts), 1),
        "n_vertebrae_detected": n,
        "confidence": "low" if warnings else "ok",
        "warnings": warnings,
        "sigma_cuts": list(SIGMA_CUTS),
        "image_size": {"width": W, "height": H},
        "centroids": scale(pts),
        "curve": scale(curve_of(pts)),
        "model_version": "unet-resnet34-v1",
    }
